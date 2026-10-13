#!/bin/bash
#SBATCH -J sema-vision-exposure
#SBATCH --partition=suma_rtx4090
#SBATCH --exclude=cs-gpu-01,node40
#SBATCH --qos=base_qos
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=0-23:00:00
#SBATCH --array=0-11%4
#SBATCH --chdir=/home/faransansj/CLMoE-VQE
#SBATCH --output=/home/faransansj/CLMoE-VQE/vision-exposure-%A_%a.log
set -euo pipefail
export OMP_NUM_THREADS=4
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
export PYTHONPATH="$PWD"
hostname
.venv/bin/python -c "import torch; print('CUDA allocation:', torch.ones(1, device='cuda'))"
ROOT=/scratch2/faransansj/CLMoE-VQE
if (( SLURM_ARRAY_TASK_ID < 6 )); then
  SOURCE=confirmation
  REPORT=calibration
  INDEX=$SLURM_ARRAY_TASK_ID
else
  SOURCE=confirmation_independent
  REPORT=calibration_independent
  INDEX=$((SLURM_ARRAY_TASK_ID - 6))
fi
.venv/bin/python scripts/vision_array.py --root "$ROOT" --stage exposure --index "$INDEX" --run-group "exposure_$SOURCE" --exposure-source-group "$SOURCE" --manifest-root "$ROOT/vision_manifests_referenced" --plan configs/vision_exposure_poc.json --selection "$ROOT/vision_report/$REPORT/selection.json"
