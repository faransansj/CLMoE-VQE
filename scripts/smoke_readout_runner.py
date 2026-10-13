"""Real-image end-to-end runner smoke; isolated outputs, including report recovery."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    for name in ("manifest", "assets", "image-root", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--compare-to", type=Path,
                        help="Optional previous smoke with identical examples/settings; require prediction-byte parity")
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Runner smoke requires an unused output path")
    args.output.mkdir(parents=True)
    manifest = copy.deepcopy(json.loads(args.manifest.read_text()))
    manifest["tasks"] = manifest["tasks"][:2]
    for task in manifest["tasks"]:
        task["train"] = task["train"][:3]
        task["validation"] = task["validation"][:2]
        task["eval"] = []
    manifest["smoke_only"] = True
    source = args.output / "manifest.json"
    source.write_text(json.dumps(manifest))
    observations = {}
    for method in ("expand", "replay_lora"):
        output = args.output / method
        command = [sys.executable, "-m", "sema_vqa.readout_run", "--manifest", str(source),
                   "--assets", str(args.assets), "--image-root", str(args.image_root),
                   "--output", str(output), "--method", method, "--eval-split", "validation",
                   "--epochs", "1", "--batch-size", "2", "--micro-batch-size", "1",
                   "--router-text", "stem", "--replay-per-task", "2", "--oracle-diagnostic"]
        subprocess.run(command, check=True)
        result_path = output / "results.json"
        original = result_path.read_bytes()
        result = json.loads(original)
        assert result["completed"] == 2
        assert result["expert_count"] == (2 if method == "expand" else 1)
        events = [json.loads(line) for line in (output / "events.jsonl").read_text().splitlines()]
        training = [event for event in events if event["event"] == "train"]
        assert [event["examples"] for event in training] == [3, 3]
        assert [event["steps"] for event in training] == [2, 2]
        if method == "replay_lora":
            assert training[1]["replay_examples"] == 1
            assert result["router_memory_bytes"] == 0
        checkpoint_hash, events_hash = digest(output / "latest.pt"), digest(output / "events.jsonl")
        # Only this script's newly created derived report is removed. This
        # reproduces the state after checkpoint commit but before report commit.
        result_path.unlink()
        subprocess.run(command, check=True)
        assert result_path.read_bytes() == original
        assert digest(output / "latest.pt") == checkpoint_hash
        assert digest(output / "events.jsonl") == events_hash
        observations[method] = {"completed_tasks": 2, "training_examples": [3, 3],
                                "partial_batch_retained": True, "missing_report_restored": True,
                                "no_retraining_or_checkpoint_mutation_on_resume": True}
        if args.compare_to is not None:
            paths = sorted(output.glob("predictions_*.jsonl"))
            old = args.compare_to / method
            assert len(paths) == 5
            assert [path.name for path in paths] == [path.name for path in sorted(old.glob("predictions_*.jsonl"))]
            for path in paths:
                assert path.read_bytes() == (old / path.name).read_bytes()
            observations[method]["previous_prediction_files_byte_identical"] = len(paths)
        observations[method]["router_memory_bytes"] = result["router_memory_bytes"]
    report = {"scope": "Six training and four validation examples, not a performance result",
              "stem_routing_exercised": True, "methods": observations}
    (args.output / "smoke.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
