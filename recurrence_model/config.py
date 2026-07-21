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

ratiogt1_entry_layers = 0
ratiogt1_loop_start_layer = 1
ratiogt1_loop_end_layer = 6
ratiogt1_num_loops = 2

depthstepgt1_chunk_size = 10
depthstepgt1_memory_tokens = 1
depthstepgt1_feedback_source_layer = 6
depthstepgt1_feedback_target_layer = 3
depthstepgt1_feedback_gate_init = -4.0

depthratio1_feedback_source_layer = 6
depthratio1_feedback_target_layer = 3
depthratio1_feedback_gate_init = -2.0

depthratiolt1_num_steps = 2
depthratiolt1_feedback_source_layer = 6
depthratiolt1_feedback_target_layer = 3
depthratiolt1_feedback_gate_init = -2.0

deltaproduct_num_householder = 4
deltaproduct_use_output_gate = True
deltaproduct_use_forget_gate = True
deltaproduct_allow_neg_eigval = True
deltaproduct_use_short_conv = True
deltaproduct_conv_size = 4

mamba_d_state = 16
mamba_d_conv = 4
mamba_expand = 2
mamba_use_fast_path = True

brt_block_size = 16
brt_num_states = 64
brt_gate_type = "lstm"
brt_single_gate = False
brt_skip_ffn = False
brt_layer_indices = "-3"

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
    depthstepgt1_chunk_size: int = depthstepgt1_chunk_size
    depthstepgt1_memory_tokens: int = depthstepgt1_memory_tokens
    depthstepgt1_feedback_source_layer: int = depthstepgt1_feedback_source_layer
    depthstepgt1_feedback_target_layer: int = depthstepgt1_feedback_target_layer
    depthstepgt1_feedback_gate_init: float = depthstepgt1_feedback_gate_init
    depthratio1_feedback_source_layer: int = depthratio1_feedback_source_layer
    depthratio1_feedback_target_layer: int = depthratio1_feedback_target_layer
    depthratio1_feedback_gate_init: float = depthratio1_feedback_gate_init
    depthratiolt1_num_steps: int = depthratiolt1_num_steps
    depthratiolt1_feedback_source_layer: int = depthratiolt1_feedback_source_layer
    depthratiolt1_feedback_target_layer: int = depthratiolt1_feedback_target_layer
    depthratiolt1_feedback_gate_init: float = depthratiolt1_feedback_gate_init
    deltaproduct_num_householder: int = deltaproduct_num_householder
    deltaproduct_use_output_gate: bool = deltaproduct_use_output_gate
    deltaproduct_use_forget_gate: bool = deltaproduct_use_forget_gate
    deltaproduct_allow_neg_eigval: bool = deltaproduct_allow_neg_eigval
    deltaproduct_use_short_conv: bool = deltaproduct_use_short_conv
    deltaproduct_conv_size: int = deltaproduct_conv_size
    mamba_d_state: int = mamba_d_state
    mamba_d_conv: int = mamba_d_conv
    mamba_expand: int = mamba_expand
    mamba_use_fast_path: bool = mamba_use_fast_path
    brt_block_size: int = brt_block_size
    brt_num_states: int = brt_num_states
    brt_gate_type: str = brt_gate_type
    brt_single_gate: bool = brt_single_gate
    brt_skip_ffn: bool = brt_skip_ffn
    brt_layer_indices: str = brt_layer_indices
