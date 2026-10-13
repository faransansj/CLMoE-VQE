#!/bin/bash
#SBATCH -J sema-vision-refine
#SBATCH --partition=suma_rtx4090
#SBATCH --exclude=cs-gpu-01,node40
#SBATCH --qos=base_qos
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=0-23:00:00
#SBATCH --array=0-23%4
#SBATCH --chdir=/home/faransansj/CLMoE-VQE
#SBATCH --output=/home/faransansj/CLMoE-VQE/vision-refine-%A_%a.log
set -euo pipefail
export OMP_NUM_THREADS=4
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
export PYTHONPATH="$PWD"
ROOT=/scratch2/faransansj/CLMoE-VQE
hostname
.venv/bin/python -c "import torch; print('CUDA allocation:', torch.ones(1, device='cuda'))"
if (( SLURM_ARRAY_TASK_ID < 18 )); then
  PLAN=configs/vision_short_poc.json
  GROUP=calibration_short
  INDEX=$SLURM_ARRAY_TASK_ID
else
  PLAN=configs/vision_midrate_poc.json
  GROUP=calibration_midrate
  INDEX=$((SLURM_ARRAY_TASK_ID - 18))
fi
.venv/bin/python scripts/vision_array.py --root "$ROOT" --stage calibration --run-group "$GROUP" --manifest-root "$ROOT/vision_manifests_referenced" --plan "$PLAN" --index "$INDEX"
