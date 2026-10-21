#!/bin/bash
#SBATCH -J week1batch1
#SBATCH --partition=suma_rtx4090
#SBATCH --exclude=cs-gpu-01,node40
#SBATCH --qos=base_qos
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=0-4:00:00
#SBATCH --chdir=/home/faransansj/CLMoE-Multimodal-20261013
#SBATCH --output=/home/faransansj/CLMoE-Multimodal-20261013/artifacts/week1-batch1/__RUNID__/week1-batch1-%j.log
#SBATCH --error=/home/faransansj/CLMoE-Multimodal-20261013/artifacts/week1-batch1/__RUNID__/week1-batch1-%j.err.log
set -euo pipefail
# No manual CUDA_VISIBLE_DEVICES declaration: the scheduler allocates gres=gpu:1.
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONPATH="$PWD"

printf 'week1-batch1 job: job_id=%s node=%s partition=%s gres_alloc=%s\n' \
    "$SLURM_JOB_ID" "${SLURM_NODELIST:-unknown}" "${SLURM_JOB_PARTITION:-unknown}" "${CUDA_VISIBLE_DEVICES:-unset}"
"${MULTIMODAL_PYTHON:-.venv/bin/python}" "$MULTIMODAL_WEEK1_SCRIPT" regenerate \
    --cases "$MULTIMODAL_WEEK1_CASES" \
    --output "$MULTIMODAL_WEEK1_OUTPUT"
