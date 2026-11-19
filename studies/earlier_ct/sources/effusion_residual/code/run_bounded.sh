#!/usr/bin/env bash
set -euo pipefail
umask 077
export CUDA_VISIBLE_DEVICES="$1"
shift
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 PYTHONPATH=.
for _ in {1..30}; do
  if [[ -s phase2/prior_train/shard0.json && -s phase2/prior_train/shard1.json ]]; then break; fi
  sleep 2
done
test -s phase2/prior_train/shard0.json
test -s phase2/prior_train/shard1.json
for seed in "$@"; do
  .hf-venv/bin/python -u bounded_residual.py "$seed"
  .hf-venv/bin/python -u eval_bounded.py "$seed"
done
