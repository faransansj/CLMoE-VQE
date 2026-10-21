#!/bin/bash
#SBATCH -J stage05trunc
#SBATCH -p dell_rtx3090
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time 0-03:00:00
#SBATCH --output=/home/faransansj/CLMoE-Multimodal-20261013/stage05-truncation-%j.log
cd /home/faransansj/CLMoE-Multimodal-20261013
export PATH=/opt/ohpc/pub/apps/java/22.0.2/bin:$PATH
export PYTHONDONTWRITEBYTECODE=1
export STAGE05_ATTEMPT=a1
echo "host=$(hostname) job=$(squeue --job $SLURM_JOB_ID -h -o %b 2>/dev/null || echo n/a) gres=${SLURM_JOB_GPUS:-none} cpus=$SLURM_CPUS_PER_TASK mem=$SLURM_MEM_PER_NODE"
./.venv/bin/python scripts/stage05_truncation_rescore.py
status=$?
echo "stage05_truncation_rescore exit=$status"
exit $status
