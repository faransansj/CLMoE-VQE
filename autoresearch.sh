#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
export PYTHONHASHSEED=0
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export LC_ALL=C TZ=UTC
exec "${AUTORESEARCH_PYTHON:-.venv/bin/python}" -m scripts.benchmark_vqa_scoring
