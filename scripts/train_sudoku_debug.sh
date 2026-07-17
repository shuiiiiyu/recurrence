#!/usr/bin/env bash
set -euo pipefail
PYTHON=${PYTHON:-/data/shencanyu/envs/state-rec/bin/python}
$PYTHON scripts/train.py --dataset sudoku --model baseline --num_layers 4 --d_model 64 --num_heads 4 --batch_size 8 --max_train_samples 16 --max_val_samples 16 --max_steps 1 --eval_every 1 --device cpu "$@"
