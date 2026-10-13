#!/bin/bash
#SBATCH -J sema-vision-smoke
#SBATCH --partition=suma_rtx4090
#SBATCH --exclude=cs-gpu-01,node40
#SBATCH --qos=base_qos
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=0-01:00:00
#SBATCH --chdir=/home/faransansj/CLMoE-VQE
#SBATCH --output=/home/faransansj/CLMoE-VQE/vision-smoke-%j.log
set -euo pipefail
export OMP_NUM_THREADS=4
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
export PYTHONPATH="$PWD"
hostname
.venv/bin/python scripts/smoke_vision.py --assets /scratch2/faransansj/CLMoE-VQE/models/blip --manifest /scratch2/faransansj/CLMoE-VQE/vision_manifests_referenced/cl4vqa.json --image-root /scratch2/faransansj/CLMoE-VQE/images --output /scratch2/faransansj/CLMoE-VQE/vision_smoke
