#!/bin/bash
#SBATCH -J multimodal-routing
#SBATCH --partition=suma_rtx4090
#SBATCH --exclude=cs-gpu-01,node40
#SBATCH --qos=base_qos
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=0-23:00:00
#SBATCH --chdir=/home/faransansj/CLMoE-Multimodal-20261013
#SBATCH --output=/home/faransansj/CLMoE-Multimodal-20261013/multimodal-routing-%A_%a.log
set -euo pipefail
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONPATH="$PWD"
"${MULTIMODAL_PYTHON:-.venv/bin/python}" -c '
import json, os, subprocess, sys
from pathlib import Path
payload = json.loads(Path(os.environ["MULTIMODAL_ROUTING_JOBS"]).read_text())
job = payload["jobs"][int(os.environ["SLURM_ARRAY_TASK_ID"])]
argv = job["argv"]
if not isinstance(argv, list) or not all(isinstance(value, str) for value in argv):
    raise ValueError("Every routing job requires literal CLI arguments")
if "--output" not in argv or "--parent" not in argv:
    raise ValueError("Every routing job requires an explicit parent and new output path")
if "--allow-eval" in argv:
    # Inject the external lock binding so the locked grid never hashes itself.
    argv = [*argv, "--confirmation-lock", os.environ["MULTIMODAL_CONFIRMATION_LOCK"],
            "--confirmation-lock-sha256", os.environ["MULTIMODAL_CONFIRMATION_LOCK_SHA256"]]
subprocess.run([sys.executable, "-m", "scripts.ablate_multimodal_routing", *argv], check=True)
'
