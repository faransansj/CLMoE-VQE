#!/bin/bash
#SBATCH -J p0strip
#SBATCH --partition=suma_rtx4090
#SBATCH --exclude=cs-gpu-01,node40
#SBATCH --qos=base_qos
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=0-4:00:00
#SBATCH --chdir=/home/faransansj/CLMoE-Multimodal-20261013
#SBATCH --output=/home/faransansj/CLMoE-Multimodal-20261013/artifacts/p0_strip/logs/p0strip-%x-%j.log
#SBATCH --error=/home/faransansj/CLMoE-Multimodal-20261013/artifacts/p0_strip/logs/p0strip-%x-%j.err.log

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

MODE="${P0_MODE:?P0_MODE must be gate|cell}"
JOB="${P0_G4_JOB:?P0_G4_JOB must name one G4 job}"
RUN_ID="${P0_RUN_ID:?P0_RUN_ID must be a unique safe card/run identifier}"
PREREG="${P0_PREREG_COMMIT:-}"
PY="${MULTIMODAL_PYTHON:-.venv/bin/python}"
printf 'p0 job: job_id=%s node=%s partition=%s mode=%s g4_job=%s run_id=%s strip=%s prereg=%s\n' \
    "${SLURM_JOB_ID:-none}" "${SLURMD_NODENAME:-${HOSTNAME:-unknown}}" "${SLURM_JOB_PARTITION:-unknown}" \
    "$MODE" "$JOB" "$RUN_ID" "${P0_STRIP:-0}" "${PREREG:-none}"

EXTRA=()
if [ "${P0_STRIP:-0}" = "1" ]; then EXTRA+=(--strip); fi
if [ -n "${P0_JOBS:-}" ]; then EXTRA+=(--jobs "$P0_JOBS"); fi
if [ -n "${P0_OUTPUT_ROOT:-}" ]; then EXTRA+=(--output-root "$P0_OUTPUT_ROOT"); fi
if [ -n "${P0_REPORT_ROOT:-}" ]; then EXTRA+=(--report-root "$P0_REPORT_ROOT"); fi
if [ "${P0_VALIDATE_ONLY:-0}" = "1" ]; then EXTRA+=(--validate-only); fi
if [ -n "$PREREG" ]; then EXTRA+=(--prereg-commit "$PREREG"); fi
# P0_REQUIRE_PREREG_LOCK=1 rejects missing Git history, draft/TBD locks and dirty/untracked
# execution code before importing models. Otherwise provenance is recorded, not enforced.
if [ "${P0_REQUIRE_PREREG_LOCK:-0}" = "1" ]; then EXTRA+=(--require-prereg-lock); fi
"$PY" scripts/p0_strip_runner.py "$MODE" --g4-job "$JOB" --run-id "$RUN_ID" ${EXTRA[@]+"${EXTRA[@]}"}
echo "p0 mode=$MODE job=$JOB finished"
