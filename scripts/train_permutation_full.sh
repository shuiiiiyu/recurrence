#!/usr/bin/env bash
set -euo pipefail
PYTHON=${PYTHON:-/data/shencanyu/envs/state-rec/bin/python}
$PYTHON scripts/train.py --dataset permutation --permutation_subset S3_len100_100k "$@"
