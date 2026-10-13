#!/bin/bash
#SBATCH -J multimodal-g0
#SBATCH --partition=suma_rtx4090
#SBATCH --exclude=cs-gpu-01,node40
#SBATCH --qos=base_qos
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=0-02:00:00
#SBATCH --chdir=/home/faransansj/CLMoE-Multimodal-20261013
#SBATCH --output=/home/faransansj/CLMoE-Multimodal-20261013/g0-%j.log
set -euo pipefail
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
export PYTHONPATH="$PWD"
OLD=/scratch2/faransansj/CLMoE-VQE
NEW=/scratch2/faransansj/multimodal_g0_${SLURM_JOB_ID}
.venv/bin/python -m scripts.smoke_readout_runner \
    --manifest "$OLD/readout_runner_smoke_2448777/manifest.json" \
    --compare-to "$OLD/readout_runner_smoke_2448777" \
    --assets "$OLD/models/blip" --image-root "$OLD/images" \
    --output "$NEW/vqa_parity"
.venv/bin/python -m scripts.smoke_multimodal \
    --manifest /scratch2/faransansj/multimodal_data_20261013/manifest.json \
    --assets "$OLD/models/blip" \
    --image-root /scratch2/faransansj/multimodal_images_20261013 \
    --output "$NEW/multimodal"
