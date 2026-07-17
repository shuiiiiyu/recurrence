#!/usr/bin/env bash
set -euo pipefail
PYTHON=${PYTHON:-/data/shencanyu/envs/state-rec/bin/python}

$PYTHON scripts/train.py --model baseline --num_layers 4 --d_model 64 --num_heads 4 --seq_len 16 --train_samples 128 --val_samples 64 --batch_size 16 --max_steps 1 --eval_every 1 --device cpu
$PYTHON scripts/train.py --model ratio1 --num_layers 4 --d_model 64 --num_heads 4 --ratio1_feedback_source_layer 4 --ratio1_feedback_target_layer 2 --seq_len 16 --train_samples 128 --val_samples 64 --batch_size 16 --max_steps 1 --eval_every 1 --device cpu
$PYTHON scripts/train.py --model ratiolt1 --num_layers 4 --d_model 64 --num_heads 4 --ratiolt1_entry_layers 1 --ratiolt1_loop_start_layer 2 --ratiolt1_loop_end_layer 4 --ratiolt1_num_loops 2 --seq_len 16 --train_samples 128 --val_samples 64 --batch_size 16 --max_steps 1 --eval_every 1 --device cpu

# Dataset loader checks. Keep these tiny; they only prove formatting/training runs.
scripts/train_sudoku_debug.sh
scripts/train_permutation_debug.sh
