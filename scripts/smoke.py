"""Exercise real pretrained VQA, adapter training and checkpoint recovery on GPU."""
import argparse
import copy
import json
import time
from pathlib import Path

import torch

from sema_vqa.model import ContinualVQA


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--assets", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--image-root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.manual_seed(17)
    started = time.time()
    manifest = json.loads(Path(args.manifest).read_text())
    row = manifest["tasks"][0]["train"][0]
    model = ContinualVQA(args.assets)
    before = model.answer(row, args.image_root, 16)
    params = [p for layer in model.adapted for p in layer.mixture.functional.parameters()]
    optimizer = torch.optim.AdamW(params, lr=2e-4)
    initial_weight = model.adapted[-1].mixture.functional[0].up_proj.weight.detach().clone()
    model.set_mode("functional")
    loss = model.loss(row, args.image_root)
    assert torch.isfinite(loss)
    loss.backward()
    optimizer.step()
    assert not torch.equal(initial_weight, model.adapted[-1].mixture.functional[0].up_proj.weight)
    model.set_mode("functional")
    with torch.no_grad():
        after_loss = float(model.loss(row, args.image_root))
        features = model.describe(row, args.image_root, "features")
    assert all(x.shape == (1, 4096) and torch.isfinite(x).all() for x in features)
    rd_losses = []
    for layer, feature in zip(model.adapted, features):
        mixture = layer.mixture
        assert next(mixture.parameters()).dtype == torch.float32
        x = feature.unsqueeze(1)
        start_rd = float(mixture.rd_loss(x).detach())
        rd_optimizer = torch.optim.SGD(mixture.rd.parameters(), lr=0.01)
        for _ in range(4):
            rd_optimizer.zero_grad(set_to_none=True)
            rd_loss = mixture.rd_loss(x)
            mixture.observe(x)
            rd_loss.backward()
            rd_optimizer.step()
        end_rd = float(mixture.rd_loss(x).detach())
        assert end_rd < start_rd, (start_rd, end_rd)
        rd_losses.append({"before": start_rd, "after": end_rd})
    for layer in model.adapted:
        layer.mixture.freeze()
    state = copy.deepcopy(model.adapter_state())
    after = model.answer(row, args.image_root, 16)
    for layer in model.adapted:
        layer.mixture.expand()
    model.load_adapter_state(state)
    restored = model.answer(row, args.image_root, 16)
    assert restored == after
    assert all(layer.mixture.num_experts == 1 for layer in model.adapted)
    # Verify cached context decoding agrees with the uncached full-prefix path.
    inputs, _ = model.prepare(row, args.image_root, with_answer=False)
    with torch.inference_mode():
        uncached_ids = model.language.generate(inputs_embeds=inputs,
            attention_mask=torch.ones(inputs.shape[:2], device=model.device, dtype=torch.long),
            max_new_tokens=16, do_sample=False, use_cache=False,
            pad_token_id=model.tokenizer.eos_token_id)
    uncached = model.tokenizer.decode(uncached_ids[0], skip_special_tokens=True).strip()
    assert after == uncached, (after, uncached)
    result = {"real_id": row["id"], "image": row["image"], "question": row["question"],
              "target": row["answer"], "initial_answer": before, "trained_answer": after,
              "loss_before": float(loss.detach()), "loss_after": after_loss,
              "descriptor_losses": rd_losses, "adapter_dtype": "float32", "backbone_dtype": "bfloat16",
              "checkpoint_generation_equal": restored == after,
              "cached_generation_equal": after == uncached,
              "gpu": torch.cuda.get_device_name(), "peak_memory_gb": torch.cuda.max_memory_allocated() / 1e9,
              "seconds": time.time() - started}
    Path(args.output).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
