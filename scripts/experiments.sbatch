#!/bin/bash
#SBATCH -J sema-vqa-poc
#SBATCH --partition=suma_rtx4090
#SBATCH --exclude=cs-gpu-01,node40
#SBATCH --qos=base_qos
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=0-23:00:00
#SBATCH --array=0-15%4
#SBATCH --chdir=/home/faransansj/CLMoE-VQE
#SBATCH --output=/home/faransansj/CLMoE-VQE/experiment-%A_%a.log
set -euo pipefail
export OMP_NUM_THREADS=4
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
export PYTHONPATH="$PWD"
hostname
.venv/bin/python -c "import torch; print('CUDA allocation:', torch.ones(1, device='cuda'))"
.venv/bin/python scripts/run_array.py --root /scratch2/faransansj/CLMoE-VQE
