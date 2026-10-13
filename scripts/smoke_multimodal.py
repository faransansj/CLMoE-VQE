"""G0 GPU checks on supplied real VQA/caption manifest and BLIP assets.

This is technical acceptance only, not held-out learnability evidence. No data
is synthesized and no archived VQA output may be overwritten.
"""
import argparse
import gc
import hashlib
import json
from pathlib import Path


def frozen_digest(model):
    digest = hashlib.sha256()
    for name, parameter in model.named_parameters():
        if ".experts." not in name:
            digest.update(name.encode())
            digest.update(parameter.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def assert_same_bank(first, second):
    import torch
    for key in ("version", "rank", "alpha", "num_experts", "active_expert", "trainable_expert", "training"):
        assert first[key] == second[key], key
    assert first["projections"].keys() == second["projections"].keys()
    for name, bank in first["projections"].items():
        assert bank.keys() == second["projections"][name].keys()
        for key, value in bank.items():
            assert torch.equal(value, second["projections"][name][key]), (name, key)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("manifest", "assets", "image-root", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--updates", type=int, default=4, help="Real optimizer updates per family before runner resume check")
    args = parser.parse_args(argv)
    if args.updates < 1:
        parser.error("At least one actual update per family is required")
    from sema_multimodal.run import safe_output, main as run, optimizer_step, atomic_json
    output = safe_output(args.output, smoke=True)
    if output.exists() and any(output.iterdir()):
        raise ValueError("Smoke requires a fresh output directory, never overwrite evidence")
    output.mkdir(parents=True, exist_ok=True)
    from sema_multimodal.data import validate_manifest, public_rows, training_rows
    manifest = validate_manifest(json.loads(args.manifest.read_text()), args.image_root)
    vqa = next(task for task in manifest["tasks"] if task["family"] == "vqa")
    caption = next(task for task in manifest["tasks"] if task["family"] == "caption")
    for task in (vqa, caption):
        if len(task["train"]) < 5 or len(task["validation"]) < 2:
            raise ValueError("Smoke needs five real train images and two validation images per family")
    import os
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import torch
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(17)
    if not torch.cuda.is_available():
        raise RuntimeError("G0 acceptance requires an actual GPU")
    torch.cuda.manual_seed_all(17)
    from sema_multimodal.model import MultimodalReadout
    from sema_vqa.readout_router import FeatureRouter
    model = MultimodalReadout(args.assets)
    original_base = frozen_digest(model)
    vqa_rows, caption_rows = vqa["train"][:2], caption["train"][:2]
    combined = vqa_rows + caption_rows
    public = public_rows(combined)
    frozen_features = model.routing_features(public, args.image_root)
    changed = [dict(row, target="hidden wrong target", references=["hidden"], task_id="hidden", family="hidden", source="hidden")
               for row in combined]
    assert torch.equal(frozen_features, model.routing_features(public_rows(changed), args.image_root))
    first = model.add_expert()
    model.train_expert(first)
    optimizer = torch.optim.AdamW(list(model.expert_parameters(first)), lr=3e-4)
    first_before = [parameter.detach().cpu().clone() for parameter in model.expert_parameters(first)]
    losses = {"vqa": [], "caption": []}
    for update in range(args.updates):
        batch = training_rows(vqa_rows, seed=17, epoch=update)
        value, _ = optimizer_step(model, batch, args.image_root, 1, optimizer)
        losses["vqa"].append(value)
    assert any(not torch.equal(before, parameter.detach().cpu())
               for before, parameter in zip(first_before, model.expert_parameters(first)))
    model.eval()
    old_predictions = model.answer(public_rows(vqa_rows), args.image_root)
    first_frozen = [parameter.detach().cpu().clone() for parameter in model.expert_parameters(first)]
    second = model.add_expert()
    model.train_expert(second)
    second_before = [parameter.detach().cpu().clone() for parameter in model.expert_parameters(second)]
    optimizer = torch.optim.AdamW(list(model.expert_parameters(second)), lr=3e-4)
    for update in range(args.updates):
        batch = training_rows(caption_rows, seed=17, epoch=update)
        value, _ = optimizer_step(model, batch, args.image_root, 1, optimizer)
        losses["caption"].append(value)
    assert all(torch.equal(before, parameter.detach().cpu())
               for before, parameter in zip(first_frozen, model.expert_parameters(first)))
    assert all(parameter.grad is None and not parameter.requires_grad for parameter in model.expert_parameters(first))
    assert any(not torch.equal(before, parameter.detach().cpu())
               for before, parameter in zip(second_before, model.expert_parameters(second)))
    assert frozen_digest(model) == original_base
    assert torch.equal(frozen_features, model.routing_features(public, args.image_root))
    model.set_expert(first)
    model.eval()
    assert model.answer(public_rows(vqa_rows), args.image_root) == old_predictions
    model.set_expert(second)
    captions, caption_generation = model.generate(public_rows(caption_rows), args.image_root)
    assert len(captions) == 2 and any(text.strip() for text in captions), captions
    assert all(info["max_new_tokens"] == 64 for info in caption_generation)
    state = model.checkpoint_state()
    state_path = output / "expert_roundtrip.pt"
    torch.save(state, state_path)
    router = FeatureRouter(memory_per_task=2)
    router.observe(frozen_features[:2].cpu(), first, model.num_experts)
    router.observe(frozen_features[2:].cpu(), second, model.num_experts)
    native_routes = router.predict(frozen_features.cpu()).tolist()
    model.load_checkpoint_state(torch.load(state_path, map_location="cpu", weights_only=False))
    assert_same_bank(state, model.checkpoint_state())
    restored_captions, restored_information = model.generate(public_rows(caption_rows), args.image_root)
    assert restored_captions == captions and restored_information == caption_generation
    counts = model.parameter_counts()
    del model, optimizer, state
    gc.collect()
    torch.cuda.empty_cache()
    common = ["--manifest", str(args.manifest), "--assets", str(args.assets), "--image-root", str(args.image_root),
              "--method", "replay_lora", "--task-ids", vqa["task_id"], caption["task_id"],
              "--epochs", "1", "--batch-size", "3", "--micro-batch-size", "2",
              "--train-limit", "5", "--eval-limit", "2", "--replay-per-task", "3", "--smoke-output"]
    full_path, resume_path = output / "uninterrupted", output / "resumed"
    full = run(common + ["--output", str(full_path)])
    gc.collect()
    torch.cuda.empty_cache()
    interrupted = run(common + ["--output", str(resume_path), "--stop-after-steps", "3"])
    saved_partial = torch.load(resume_path / "latest.pt", map_location="cpu", weights_only=False)
    assert interrupted["status"] == "interrupted"
    assert saved_partial["state"]["task_index"] == 1
    assert saved_partial["state"]["stage"] == "train"
    assert saved_partial["state"]["epoch"] == 0 and saved_partial["state"]["batch_cursor"] == 1
    assert saved_partial["optimizer"] is not None and saved_partial["optimizer"]["state"]
    del saved_partial
    gc.collect()
    torch.cuda.empty_cache()
    resumed = run(common + ["--output", str(resume_path)])
    assert resumed["status"] == full["status"] == "complete"
    assert resumed["matrix"] == full["matrix"]
    assert resumed["pre_task_scores"] == full["pre_task_scores"]
    for key in ("steps", "image_exposures", "target_tokens", "current_exposures", "replay_exposures"):
        assert resumed["resources"][key] == full["resources"][key], key
    assert full["resources"]["steps"] == 4
    assert full["resources"]["image_exposures"] == 10
    assert full["resources"]["replay_exposures"] == 2
    full_checkpoint = torch.load(full_path / "latest.pt", map_location="cpu", weights_only=False)
    resumed_checkpoint = torch.load(resume_path / "latest.pt", map_location="cpu", weights_only=False)
    assert_same_bank(full_checkpoint["model"], resumed_checkpoint["model"])
    assert torch.equal(full_checkpoint["rng"]["torch"], resumed_checkpoint["rng"]["torch"])
    assert all(torch.equal(first_rng, second_rng) for first_rng, second_rng in
               zip(full_checkpoint["rng"]["cuda"], resumed_checkpoint["rng"]["cuda"]))
    assert full_checkpoint["rng"]["python"] == resumed_checkpoint["rng"]["python"]
    for filename in sorted(full_path.glob("predictions_*.jsonl")):
        assert filename.read_bytes() == (resume_path / filename.name).read_bytes(), filename.name
    events = [json.loads(line) for line in (resume_path / "events.jsonl").read_text().splitlines()]
    partial = [event for event in events if event["event"] == "train_batch" and event["task"] == 1 and event["batch"] == 1]
    assert len(partial) == 1 and partial[0]["images"] == 2 and partial[0]["replay_images"] == 1
    evidence = {"schema_version": 1, "manifest_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
                "actual_gpu": torch.cuda.get_device_name(), "train_losses": losses,
                "both_family_experts_updated": True, "old_expert_and_frozen_backbone_unchanged": True,
                "frozen_router_public_view_invariant": True, "native_routes": native_routes,
                "oracle_routes": [first, first, second, second], "oracle_is_diagnostic_only": True,
                "caption_predictions": captions, "caption_generation": caption_generation,
                "checkpoint_caption_roundtrip_identical": True,
                "interrupted_mid_epoch_optimizer_resume_weights_predictions_rng_identical": True,
                "partial_replay_batch_preserved": partial[0], "parameter_counts": counts,
                "runner_resources": full["resources"],
                "scope": "Real tiny training/validation technical smoke only; not G1 held-out learnability or confirmation"}
    atomic_json(output / "smoke.json", evidence)
    print(json.dumps(evidence, indent=2))
    return evidence


if __name__ == "__main__":
    main()
