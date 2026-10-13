#!/bin/bash
#SBATCH -J sema-vqa-smoke
#SBATCH --partition=suma_rtx4090
#SBATCH --exclude=cs-gpu-01
#SBATCH --qos=base_qos
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=00:20:00
#SBATCH --chdir=/home/faransansj/CLMoE-VQE
#SBATCH --output=/home/faransansj/CLMoE-VQE/smoke-%j.log
set -euo pipefail
export OMP_NUM_THREADS=4
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
export PYTHONPATH="$PWD"
hostname
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
.venv/bin/python -c "import torch; print('CUDA allocation:', torch.ones(1, device='cuda'))"
.venv/bin/python scripts/smoke.py \
  --assets /scratch2/faransansj/CLMoE-VQE/models \
  --manifest artifacts/smoke_manifest.json \
  --image-root /scratch2/faransansj/CLMoE-VQE/smoke_images \
  --output smoke_result.json
