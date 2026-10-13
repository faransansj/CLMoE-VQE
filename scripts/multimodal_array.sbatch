#!/bin/bash
#SBATCH -J multimodal-cl
#SBATCH --partition=suma_rtx4090
#SBATCH --exclude=cs-gpu-01,node40
#SBATCH --qos=base_qos
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=0-23:00:00
#SBATCH --chdir=/home/faransansj/CLMoE-Multimodal-20261013
#SBATCH --output=/home/faransansj/CLMoE-Multimodal-20261013/multimodal-%A_%a.log
set -euo pipefail
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONPATH="$PWD"
# Jobs are declared argument-vector arrays (not shell snippets), so task order,
# method, scientific settings and unique multimodal output remain explicit.
"${MULTIMODAL_PYTHON:-.venv/bin/python}" -c '
import json, os, subprocess, sys
from pathlib import Path
path = Path(os.environ["MULTIMODAL_JOBS"])
payload = json.loads(path.read_text())
jobs = payload["jobs"] if isinstance(payload, dict) else payload
index = int(os.environ["SLURM_ARRAY_TASK_ID"])
job = jobs[index]
argv = job["argv"] if isinstance(job, dict) else job
if not isinstance(argv, list) or not all(isinstance(value, str) for value in argv):
    raise ValueError("Every declared job must be a list of literal CLI strings (or {argv:[...]})")
if "--output" not in argv:
    raise ValueError("Every array job needs its own explicit multimodal output path")
subprocess.run([sys.executable, "-m", "sema_multimodal.run", *argv], check=True)
'
