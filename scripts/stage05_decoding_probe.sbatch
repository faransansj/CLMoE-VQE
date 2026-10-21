#!/bin/bash
#SBATCH -J stage05probe
#SBATCH --partition=suma_rtx4090
#SBATCH --exclude=cs-gpu-01,node40
#SBATCH --qos=base_qos
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=0-4:00:00
#SBATCH --chdir=/home/faransansj/CLMoE-Multimodal-20261013
#SBATCH --output=/home/faransansj/CLMoE-Multimodal-20261013/artifacts/stage05-logs/stage05-decoding-probe-%j.log
#SBATCH --error=/home/faransansj/CLMoE-Multimodal-20261013/artifacts/stage05-logs/stage05-decoding-probe-%j.err.log

set -euo pipefail
# No manual CUDA_VISIBLE_DEVICES declaration: the scheduler allocates gres=gpu:1.
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONPATH="$PWD"
export PYTHONDONTWRITEBYTECODE=1
# Java for the COCO PTB tokenizer used by the untouched CIDEr-D scorer.
export PATH=/opt/ohpc/pub/apps/java/22.0.2/bin:$PATH

printf 'stage05 decoding probe job: job_id=%s node=%s partition=%s gres_alloc=%s\n' \
    "$SLURM_JOB_ID" "${SLURM_NODELIST:-unknown}" "${SLURM_JOB_PARTITION:-unknown}" "${CUDA_VISIBLE_DEVICES:-unset}"

MODE="${STAGE05_MODE:-probe}"
if [ "$MODE" = "smoke" ]; then
    "${MULTIMODAL_PYTHON:-.venv/bin/python}" scripts/stage05_decoding_probe.py smoke \
        --output-dir "artifacts/stage05-decoding-probe-smoke/$SLURM_JOB_ID" \
        --index artifacts/multimodal-confirmation-checkpoints.json
else
    "${MULTIMODAL_PYTHON:-.venv/bin/python}" scripts/stage05_decoding_probe.py probe \
        --output-dir "artifacts/stage05-decoding-probe-run/$SLURM_JOB_ID" \
        --index artifacts/multimodal-confirmation-checkpoints.json \
        --report artifacts/stage05-decoding-probe.json \
        --predictions artifacts/stage05-decoding-predictions.jsonl.gz
fi
echo "stage05_decoding_probe mode=$MODE finished"
