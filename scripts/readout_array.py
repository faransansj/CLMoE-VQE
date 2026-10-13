"""Execute one fully specified readout experiment; never infer evaluation settings."""
import argparse
import json
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--jobs", type=Path, required=True)
    parser.add_argument("--index", type=int, required=True)
    parser.add_argument("--module", choices=("sema_vqa.readout_run", "scripts.ablate_readout_routing"),
                        default="sema_vqa.readout_run")
    args = parser.parse_args()
    jobs = json.loads(args.jobs.read_text())
    if not 0 <= args.index < len(jobs):
        parser.error("Array index outside declared jobs")
    job = jobs[args.index]
    command = [sys.executable, "-m", args.module]
    for key, value in job.items():
        flag = "--" + key.replace("_", "-")
        if isinstance(value, bool):
            if value:
                command.append(flag)
        else:
            command.extend((flag, str(value)))
    print(json.dumps({"index": args.index, "command": command}), flush=True)
    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
