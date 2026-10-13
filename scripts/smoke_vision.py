"""Actual-image proof of vision gradients, nonexpansion learning and restoration."""
import argparse
import json
from pathlib import Path
import time

import torch

from sema_vqa.vision_model import VisionVQA
from sema_vqa.vision_run import optimizer_step


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--assets", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--image-root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.manual_seed(17)
    began = time.time()
    rows = json.loads(Path(args.manifest).read_text())["tasks"][0]["train"][:4]
    model = VisionVQA(args.assets)
    model.prepare_task("sema", 0)
    model.eval()
    with torch.no_grad():
        initial_loss = float(model.loss(rows, args.image_root))
    vision = model.visual_encoder.blocks[0].adapter.up_proj.weight
    before = vision.detach().clone()
    opt_head = torch.optim.AdamW(model.head_parameters(), lr=2e-5)
    opt_vision = torch.optim.SGD(model.functional_parameters(), lr=0.005, momentum=0.9)
    model.train()
    for _ in range(4):
        optimizer_step(lambda b: model.loss(b, args.image_root), rows, 2, [opt_head, opt_vision])
    assert not torch.equal(before, vision), "Vision adapter did not learn"
    model.eval()
    with torch.no_grad():
        trained_loss = float(model.loss(rows, args.image_root))
    assert trained_loss < initial_loss, "Actual-example VQA loss did not decrease"
    features = model.features(rows, args.image_root)
    changed_questions = [dict(row, question="Different question", answer="Different answer") for row in rows]
    alternate = model.features(changed_questions, args.image_root)
    assert all(torch.equal(a, b) for a, b in zip(features, alternate)), "Language leaked into vision descriptor"
    rd_losses = []
    for mixture, data in zip(model.mixtures, features):
        optimizer = torch.optim.SGD(mixture.rd.parameters(), lr=0.01)
        initial = float(mixture.rd_loss(data).detach())
        for _ in range(20):
            optimizer.zero_grad(set_to_none=True)
            loss = mixture.rd_loss(data)
            mixture.observe(data)
            loss.backward()
            optimizer.step()
        final = float(mixture.rd_loss(data).detach())
        assert final < initial
        rd_losses.append([initial, final])
    model.prepare_task("sema", 1)
    assert not list(model.functional_parameters())
    readout = model.text_decoder.cls.predictions.transform.dense.weight
    old_readout = readout.detach().clone()
    optimizer_step(lambda b: model.loss(b, args.image_root), rows, 2, [opt_head])
    assert not torch.equal(readout, old_readout), "No-expansion path froze VQA readout"
    model.mixtures[0].expand()
    old_expert = model.mixtures[0].functional[0].up_proj.weight.detach().clone()
    new_expert = model.mixtures[0].functional[1].up_proj.weight
    optimizer = torch.optim.SGD(model.functional_parameters(), lr=0.005)
    optimizer_step(lambda b: model.loss(b, args.image_root), rows, 2, [optimizer, opt_head])
    assert not torch.equal(new_expert, torch.zeros_like(new_expert)), "Expanded expert did not learn"
    assert torch.equal(old_expert, model.mixtures[0].functional[0].up_proj.weight), "Old expert changed"
    model.freeze_vision()
    model.eval()
    answers = model.answer(rows, args.image_root)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    temporary = output / "restore-smoke.pt"
    torch.save(model.checkpoint_state(), temporary)
    del opt_head, opt_vision, optimizer, model
    torch.cuda.empty_cache()
    restored = VisionVQA(args.assets)
    restored.load_checkpoint_state(torch.load(temporary, map_location="cpu", weights_only=False))
    restored.eval()
    after = restored.answer(rows, args.image_root)
    assert answers == after, "Dynamic checkpoint generation changed"
    assert restored.expert_counts == [2, 1, 1]
    temporary.unlink()
    report = {"initial_loss": initial_loss, "trained_loss": trained_loss, "rd_losses": rd_losses,
              "image_only_descriptor": True, "no_expansion_readout_update": True,
              "expanded_expert_update": True, "old_expert_unchanged": True,
              "checkpoint_generation_equal": True, "experts": restored.expert_counts,
              "examples": [{"id": row["id"], "target": row["answer"], "prediction": answer} for row, answer in zip(rows, answers)],
              "parameters": restored.parameter_counts(), "seconds": time.time() - began,
              "peak_allocated_GB": torch.cuda.max_memory_allocated() / 2**30}
    (output / "result.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
