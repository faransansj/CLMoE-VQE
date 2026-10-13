"""Launch fixed train-only calibration or independently seeded confirmation."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


def calibration_jobs(plan):
    jobs = []
    for epochs in plan["calibration"].get("epoch_candidates", [plan["epochs_per_functional_phase"]]):
        for protocol in plan["protocols"]:
            for lr in plan["calibration"]["shared_head_lr_candidates"]:
                jobs.append({"protocol": protocol, "method": "sequential", "seed": 17, "lr": lr, "threshold": 1.0, "epochs": epochs})
                for threshold in plan["calibration"]["sema_threshold_candidates"]:
                    jobs.append({"protocol": protocol, "method": "sema", "seed": 17, "lr": lr, "threshold": threshold, "epochs": epochs})
    return jobs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--stage", choices=("calibration", "confirmation", "exposure"), required=True)
    parser.add_argument("--selection")
    parser.add_argument("--run-group", help="Separate artifact cohort; defaults to stage")
    parser.add_argument("--manifest-root", help="Explicit corrected-reference manifest directory")
    parser.add_argument("--exposure-source-group", help="Completed parent SEMA cohort, for exposure controls only")
    parser.add_argument("--plan", default="configs/vision_poc.json")
    parser.add_argument("--index", type=int, default=int(os.environ.get("SLURM_ARRAY_TASK_ID", "0")))
    args = parser.parse_args()
    if (args.stage == "exposure") != bool(args.exposure_source_group):
        raise ValueError("Exposure stage requires its parent source group, and other stages forbid it")
    plan = json.loads(Path(args.plan).read_text())
    if args.stage == "calibration":
        jobs = calibration_jobs(plan)
    else:
        if not args.selection:
            raise ValueError("Confirmation requires recorded train-only selection")
        selection = json.loads(Path(args.selection).read_text())
        jobs = []
        for protocol in plan["protocols"]:
            chosen = selection["selected"][protocol]
            for seed in plan["confirmation"]["seeds"]:
                for method in plan["confirmation"]["primary_methods"]:
                    jobs.append({"protocol": protocol, "method": method, "seed": seed, **chosen})
            for method in plan["confirmation"]["ablation_methods_seed17"]:
                jobs.append({"protocol": protocol, "method": method, "seed": 17, **chosen})
    job = jobs[args.index]
    root = Path(args.root)
    tag = f"{job['method']}_lr{job['lr']:g}_t{job['threshold']:g}"
    output = root / "vision_runs" / (args.run_group or args.stage) / job["protocol"] / tag / f"seed{job['seed']}"
    output.mkdir(parents=True, exist_ok=True)
    (output / "plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    if args.selection:
        (output / "selection.json").write_bytes(Path(args.selection).read_bytes())
    manifest_root = Path(args.manifest_root) if args.manifest_root else root / "vision_manifests"
    command = [sys.executable, "-m", "sema_vqa.vision_run", "--manifest", str(manifest_root / f"{job['protocol']}.json"),
               "--assets", str(root / "models/blip"), "--image-root", str(root / "images"), "--output", str(output),
               "--method", job["method"], "--seed", str(job["seed"]), "--lr", str(job["lr"]), "--threshold", str(job["threshold"]),
               "--epochs", str(job.get("epochs", plan["epochs_per_functional_phase"])), "--image-size", str(plan["image_size"]),
               "--batch-size", str(plan["effective_batch_size"]), "--micro-batch-size", str(plan["micro_batch_size"]),
               "--adapter-lr", str(plan["adapter_lr"]), "--rd-epochs", str(plan["rd_epochs"]), "--rd-lr", str(plan["rd_lr"]),
               "--detect-batch-size", str(plan["detect_batch_size"]), "--eval-split", "validation" if args.stage == "calibration" else "eval"]
    if args.stage == "exposure":
        source_tag = f"sema_lr{job['lr']:g}_t{job['threshold']:g}"
        source = root / "vision_runs" / args.exposure_source_group / job["protocol"] / source_tag / f"seed{job['seed']}"
        command.extend(["--match-exposure-from", str(source)])
    if (output / "latest.pt").exists():
        command.append("--resume")
    print(json.dumps({"index": args.index, "job": job, "command": command}), flush=True)
    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
