#!/bin/bash
#SBATCH -J vqa-readout
#SBATCH --partition=suma_rtx4090
#SBATCH --exclude=cs-gpu-01,node40
#SBATCH --qos=base_qos
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=0-23:00:00
#SBATCH --chdir=/home/faransansj/CLMoE-VQE
#SBATCH --output=/home/faransansj/CLMoE-VQE/readout-%A_%a.log
set -euo pipefail
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
export PYTHONPATH="$PWD"
.venv/bin/python scripts/readout_array.py --jobs "${READOUT_JOBS:?Pass a declared job JSON through sbatch --export}" --index "$SLURM_ARRAY_TASK_ID" --module "${READOUT_MODULE:-sema_vqa.readout_run}"
