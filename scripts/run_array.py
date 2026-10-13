"""Map one SLURM array index to a frozen POC configuration."""
import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True)
    p.add_argument("--plan", default="configs/poc.json")
    p.add_argument("--index", type=int, default=int(os.environ.get("SLURM_ARRAY_TASK_ID", "0")))
    args = p.parse_args()
    plan = json.loads(Path(args.plan).read_text())
    cases = []
    for protocol in plan["protocols"]:
        cases.extend((protocol, method, seed) for seed in plan["seeds"] for method in plan["primary_methods"])
        cases.extend((protocol, method, 17) for method in plan["ablation_methods_seed17"])
    protocol, method, seed = cases[args.index]
    root = Path(args.root)
    output = root / "runs" / protocol / method / f"seed{seed}"
    output.mkdir(parents=True, exist_ok=True)
    provenance = {str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                  for pattern in ("sema_vqa/*.py", "scripts/*.py", "configs/*.json")
                  for path in Path(".").glob(pattern)}
    if (output / "code_sha256.json").exists():
        if json.loads((output / "code_sha256.json").read_text()) != provenance:
            raise ValueError("Experiment source code changed; use a new run root")
    (output / "code_sha256.json").write_text(json.dumps(provenance, indent=2) + "\n")
    command = [sys.executable, "-m", "sema_vqa.run", "--manifest", str(root / "manifests" / f"{protocol}.json"),
               "--assets", str(root / "models"), "--image-root", str(root / "images"),
               "--output", str(output), "--method", method, "--seed", str(seed),
               "--layers", "3", "--bottleneck", str(plan["bottleneck"]),
               "--rd-dim", str(plan["rd_dim"]), "--threshold", str(plan["expansion_threshold"]),
               "--epochs", str(plan["functional_epochs"]), "--lr", str(plan["functional_lr"]),
               "--batch-size", str(plan["effective_batch_size"]), "--rd-lr", str(plan["rd_lr"]),
               "--rd-epochs", str(plan["rd_epochs"]), "--detect-batch-size", str(plan["detection_batch_size"]),
               "--max-new-tokens", str(plan["max_new_tokens"])]
    if (output / "latest.pt").exists():
        command.append("--resume")
    print(json.dumps({"array_index": args.index, "command": command}), flush=True)
    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
