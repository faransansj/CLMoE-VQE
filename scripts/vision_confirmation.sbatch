#!/bin/bash
#SBATCH -J sema-vision-confirm
#SBATCH --partition=suma_rtx4090
#SBATCH --exclude=cs-gpu-01,node40
#SBATCH --qos=base_qos
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=0-23:00:00
#SBATCH --array=0-25%4
#SBATCH --chdir=/home/faransansj/CLMoE-VQE
#SBATCH --output=/home/faransansj/CLMoE-VQE/vision-confirm-%A_%a.log
set -euo pipefail
export OMP_NUM_THREADS=4
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
export PYTHONPATH="$PWD"
hostname
.venv/bin/python -c "import torch; print('CUDA allocation:', torch.ones(1, device='cuda'))"
ROOT=/scratch2/faransansj/CLMoE-VQE
if (( SLURM_ARRAY_TASK_ID < 16 )); then
  PLAN=configs/vision_search_poc.json
  REPORT=calibration
  GROUP=confirmation
  INDEX=$SLURM_ARRAY_TASK_ID
else
  PLAN=configs/vision_independent_poc.json
  REPORT=calibration_independent
  GROUP=confirmation_independent
  INDEX=$((SLURM_ARRAY_TASK_ID - 16))
fi
.venv/bin/python scripts/vision_array.py --root "$ROOT" --stage confirmation --index "$INDEX" --run-group "$GROUP" --manifest-root "$ROOT/vision_manifests_referenced" --plan "$PLAN" --selection "$ROOT/vision_report/$REPORT/selection.json"
