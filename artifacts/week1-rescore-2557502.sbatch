#!/bin/bash
#SBATCH -J w1rescore
#SBATCH -p dell_rtx3090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time 0-02:00:00
#SBATCH --output=/home/faransansj/CLMoE-Multimodal-20261013/week1-rescore-%j.log
cd /home/faransansj/CLMoE-Multimodal-20261013
export PATH=/opt/ohpc/pub/apps/java/22.0.2/bin:$PATH
export PYTHONDONTWRITEBYTECODE=1
./.venv/bin/python scripts/week1_per_example_rescore.py
status=$?
echo "week1_per_example_rescore exit=$status"
exit $status
