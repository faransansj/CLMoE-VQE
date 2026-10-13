#!/bin/bash
#SBATCH -J sema-vision-cal
#SBATCH --partition=suma_rtx4090
#SBATCH --exclude=cs-gpu-01,node40
#SBATCH --qos=base_qos
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=0-23:00:00
#SBATCH --array=0-11%4
#SBATCH --chdir=/home/faransansj/CLMoE-VQE
#SBATCH --output=/home/faransansj/CLMoE-VQE/vision-cal-%A_%a.log
set -euo pipefail
export OMP_NUM_THREADS=4
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
export PYTHONPATH="$PWD"
hostname
.venv/bin/python -c "import torch; print('CUDA allocation:', torch.ones(1, device='cuda'))"
.venv/bin/python scripts/vision_array.py --root /scratch2/faransansj/CLMoE-VQE --stage calibration --run-group "${VISION_RUN_GROUP:-calibration}" --manifest-root /scratch2/faransansj/CLMoE-VQE/vision_manifests_referenced
