from __future__ import annotations

from dataclasses import dataclass
import torch

num_layers = 6
d_model = 512
num_heads = 8
ffn_mult = 4
dropout = 0.0

ratio1_feedback_source_layer = 6
ratio1_feedback_target_layer = 3
ratio1_feedback_gate_init = -4.0

ratiolt1_entry_layers = 2
ratiolt1_loop_start_layer = 3
ratiolt1_loop_end_layer = 6
ratiolt1_num_loops = 2
ratiolt1_feedback_gate_init = -4.0
append_internal_steps_to_cache = True

ratiogt1_entry_layers = 2
ratiogt1_loop_start_layer = 3
ratiogt1_loop_end_layer = 6
ratiogt1_num_loops = 2

num_states = 16
seq_len = 64
train_samples = 20000
val_samples = 2000

batch_size = 64
epochs = 1
max_steps = None
learning_rate = 3e-4
weight_decay = 0.01
grad_clip = 1.0
eval_every = 200
seed = 1337
device = "cuda" if torch.cuda.is_available() else "cpu"

# Dataset paths and debug sampling. Set max_*_samples to None for full datasets.
sudoku_root = "/data/shencanyu/data/raw/sudoku-extreme"
permutation_root = "/data/shencanyu/data/raw/permutation"
permutation_subset = "S5_len50_100k"
babi_root = "/data/shencanyu/data/raw/babi/tasks_1-20_v1-2"
babi_version = "en-10k"
babi_task = "qa2"
max_train_samples = None
max_val_samples = 10000
max_test_samples = None
early_stopping = False
early_stopping_patience = 5
early_stopping_min_delta = 1e-4
best_metric = "val_loss"


@dataclass
class ModelConfig:
    num_layers: int = num_layers
    d_model: int = d_model
    num_heads: int = num_heads
    ffn_mult: int = ffn_mult
    dropout: float = dropout
    vocab_size: int = 0
    num_classes: int = num_states
    max_seq_len: int = seq_len
    ratio1_feedback_source_layer: int = ratio1_feedback_source_layer
    ratio1_feedback_target_layer: int = ratio1_feedback_target_layer
    ratio1_feedback_gate_init: float = ratio1_feedback_gate_init
    ratiolt1_entry_layers: int = ratiolt1_entry_layers
    ratiolt1_loop_start_layer: int = ratiolt1_loop_start_layer
    ratiolt1_loop_end_layer: int = ratiolt1_loop_end_layer
    ratiolt1_num_loops: int = ratiolt1_num_loops
    ratiolt1_feedback_gate_init: float = ratiolt1_feedback_gate_init
    append_internal_steps_to_cache: bool = append_internal_steps_to_cache
    ratiogt1_entry_layers: int = ratiogt1_entry_layers
    ratiogt1_loop_start_layer: int = ratiogt1_loop_start_layer
    ratiogt1_loop_end_layer: int = ratiogt1_loop_end_layer
    ratiogt1_num_loops: int = ratiogt1_num_loops
