"""Exercise actual pretrained BLIP, expert training/isolation, routing and restore."""
import argparse
import json
from pathlib import Path

import torch

from sema_vqa.readout_model import ReadoutMoEVQA
from sema_vqa.readout_router import FeatureRouter


def main():
    parser = argparse.ArgumentParser()
    for name in ("assets", "manifest", "image-root", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.manual_seed(17)
    torch.cuda.manual_seed_all(17)
    rows = json.loads(args.manifest.read_text())["tasks"][0]["train"][:4]
    model = ReadoutMoEVQA(args.assets)
    initial_features = model.routing_features(rows, args.image_root)
    changed_labels = [dict(row, answer="not the actual answer", task="irrelevant") for row in rows]
    assert torch.equal(initial_features, model.routing_features(changed_labels, args.image_root))
    changed_questions = [dict(row, question="What color is the object?") for row in rows]
    assert not torch.equal(initial_features, model.routing_features(changed_questions, args.image_root))
    first = model.add_expert()
    model.train_expert(first)
    optimizer = torch.optim.AdamW(list(model.expert_parameters(first)), lr=3e-4)
    model.eval()
    with torch.no_grad():
        before = float(model.loss(rows, args.image_root))
    model.train()
    for _ in range(60):
        optimizer.zero_grad(set_to_none=True)
        loss = model.loss(rows, args.image_root)
        loss.backward()
        optimizer.step()
    model.eval()
    with torch.no_grad():
        after = float(model.loss(rows, args.image_root))
    assert after < before, (before, after)
    assert torch.equal(initial_features, model.routing_features(rows, args.image_root))
    old = [p.detach().clone() for p in model.expert_parameters(first)]
    old_answers = model.answer(rows, args.image_root)
    second = model.add_expert()
    model.set_expert(first)
    assert model.answer(rows, args.image_root) == old_answers
    model.train_expert(second)
    new = [p.detach().clone() for p in model.expert_parameters(second)]
    optimizer = torch.optim.AdamW(list(model.expert_parameters(second)), lr=3e-4)
    model.train()
    for _ in range(20):
        optimizer.zero_grad(set_to_none=True)
        model.loss(rows, args.image_root).backward()
        optimizer.step()
    assert all(torch.equal(before_p, p) for before_p, p in zip(old, model.expert_parameters(first)))
    assert any(not torch.equal(before_p, p) for before_p, p in zip(new, model.expert_parameters(second)))
    model.eval()
    model.set_expert(first)
    assert model.answer(rows, args.image_root) == old_answers
    assert torch.equal(initial_features, model.routing_features(rows, args.image_root))
    router = FeatureRouter(memory_per_task=4)
    router.observe(initial_features, first, model.num_experts)
    assert router.predict(initial_features).tolist() == [first] * len(rows)
    state = model.checkpoint_state()
    counts = model.parameter_counts()
    del optimizer, model
    torch.cuda.empty_cache()
    restored = ReadoutMoEVQA(args.assets)
    restored.load_checkpoint_state(state)
    restored.set_expert(first)
    restored.eval()
    assert restored.answer(rows, args.image_root) == old_answers
    assert torch.equal(initial_features, restored.routing_features(rows, args.image_root))
    result = {"examples": [row["id"] for row in rows], "train_loss_before": before, "train_loss_after": after,
              "old_expert_weights_and_answers_preserved": True, "new_expert_updated": True,
              "routing_features_answer_independent_and_expert_invariant": True,
              "checkpoint_generation_identical": True, "expert_count": restored.num_experts,
              "parameter_counts": counts, "answers": old_answers,
              "scope": "Four training examples only; not evidence of held-out VQA performance"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
