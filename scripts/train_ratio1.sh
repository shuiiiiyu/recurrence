#!/usr/bin/env bash
set -euo pipefail
PYTHON=${PYTHON:-/data/shencanyu/envs/state-rec/bin/python}
$PYTHON scripts/train.py --model ratio1 "$@"
