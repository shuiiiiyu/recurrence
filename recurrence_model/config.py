from __future__ import annotations

from dataclasses import dataclass
import torch

num_layers = 24
d_model = 384
num_heads = 6
ffn_mult = 4
dropout = 0.0

ratio1_feedback_source_layer = 24
ratio1_feedback_target_layer = 8
ratio1_feedback_gate_init = -4.0

ratiolt1_entry_layers = 7
ratiolt1_loop_start_layer = 8
ratiolt1_loop_end_layer = 24
ratiolt1_num_loops = 2
ratiolt1_feedback_gate_init = -4.0
append_internal_steps_to_cache = True

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
permutation_subset = "S3_len100_100k"
max_train_samples = None
max_val_samples = None


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
