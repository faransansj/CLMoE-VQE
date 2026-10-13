#!/bin/bash
#SBATCH -J vqa-readout-runner-smoke
#SBATCH --partition=suma_rtx4090
#SBATCH --exclude=cs-gpu-01,node40
#SBATCH --qos=base_qos
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=0-01:00:00
#SBATCH --chdir=/home/faransansj/CLMoE-VQE
#SBATCH --output=/home/faransansj/CLMoE-VQE/readout-runner-smoke-%j.log
set -euo pipefail
export OMP_NUM_THREADS=4
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
export PYTHONPATH="$PWD"
ROOT=/scratch2/faransansj/CLMoE-VQE
ARGS=(--assets "$ROOT/models/blip" --image-root "$ROOT/images"
      --output "$ROOT/readout_runner_smoke_$SLURM_JOB_ID")
if [[ -n "${READOUT_SMOKE_REFERENCE:-}" ]]; then
    ARGS+=(--manifest "$READOUT_SMOKE_REFERENCE/manifest.json" --compare-to "$READOUT_SMOKE_REFERENCE")
else
    ARGS+=(--manifest "$ROOT/vision_manifests_referenced/cl4vqa.json")
fi
.venv/bin/python -m scripts.smoke_readout_runner "${ARGS[@]}"
