#!/bin/bash
#SBATCH -J stage1strip
#SBATCH --partition=suma_rtx4090
#SBATCH --exclude=cs-gpu-01,node40
#SBATCH --qos=base_qos
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=0-3:00:00
#SBATCH --chdir=/home/faransansj/CLMoE-Multimodal-20261013
#SBATCH --output=/home/faransansj/CLMoE-Multimodal-20261013/artifacts/stage1-logs/stage1-%x-%j.log
#SBATCH --error=/home/faransansj/CLMoE-Multimodal-20261013/artifacts/stage1-logs/stage1-%x-%j.err.log

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

printf 'stage1 job: job_id=%s node=%s partition=%s gres_alloc=%s mode=%s\n' \
    "$SLURM_JOB_ID" "${SLURM_NODELIST:-unknown}" "${SLURM_JOB_PARTITION:-unknown}" \
    "${CUDA_VISIBLE_DEVICES:-unset}" "${STAGE1_MODE:-unset}"

MODE="${STAGE1_MODE:?STAGE1_MODE must be fidelity|strip|evaluate}"
PY="${MULTIMODAL_PYTHON:-.venv/bin/python}"

if [ "$MODE" = "fidelity" ]; then
    "$PY" scripts/stage1_period_strip_train.py fidelity \
        --output-dir "artifacts/multimodal_runs/stage1_period_strip/fidelity_seed137" \
        --report artifacts/stage1-fidelity.json \
        --log "artifacts/stage1-logs/stage1-fidelity-$SLURM_JOB_ID.jsonl" \
        --probe-gz artifacts/stage05-decoding-predictions.jsonl.gz
elif [ "$MODE" = "strip" ]; then
    "$PY" scripts/stage1_period_strip_train.py strip \
        --output-root "artifacts/multimodal_runs/stage1_period_strip" \
        --report artifacts/stage1-period-strip.json \
        --predictions artifacts/stage1-period-strip-predictions.jsonl.gz \
        --log "artifacts/stage1-logs/stage1-strip-$SLURM_JOB_ID.jsonl" \
        --probe-gz artifacts/stage05-decoding-predictions.jsonl.gz \
        --fidelity-report artifacts/stage1-fidelity.json
else
    "$PY" scripts/stage1_period_strip_train.py evaluate \
        --output-root "artifacts/multimodal_runs/stage1_period_strip" \
        --report artifacts/stage1-period-strip.json \
        --predictions artifacts/stage1-period-strip-predictions.jsonl.gz \
        --log "artifacts/stage1-logs/stage1-evaluate-$SLURM_JOB_ID.jsonl" \
        --probe-gz artifacts/stage05-decoding-predictions.jsonl.gz \
        --fidelity-report artifacts/stage1-fidelity.json
fi
echo "stage1 mode=$MODE finished"
