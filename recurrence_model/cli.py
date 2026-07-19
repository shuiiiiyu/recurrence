from __future__ import annotations

import argparse

from .config import *
from .training import train


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train recurrence models on token-label tasks")
    p.add_argument("--model", choices=["baseline", "ratio1", "ratiolt1", "ratiogt1"], default="ratio1")
    p.add_argument("--dataset", choices=["synthetic", "sudoku", "permutation", "babi"], default="synthetic")

    p.add_argument("--num_layers", type=int, default=num_layers)
    p.add_argument("--d_model", type=int, default=d_model)
    p.add_argument("--num_heads", type=int, default=num_heads)
    p.add_argument("--ffn_mult", type=int, default=ffn_mult)
    p.add_argument("--dropout", type=float, default=dropout)
    p.add_argument("--ratio1_feedback_source_layer", "--ratio1_source_layer", dest="ratio1_feedback_source_layer", type=int, default=ratio1_feedback_source_layer)
    p.add_argument("--ratio1_feedback_target_layer", "--ratio1_target_layer", dest="ratio1_feedback_target_layer", type=int, default=ratio1_feedback_target_layer)
    p.add_argument("--ratio1_feedback_gate_init", type=float, default=ratio1_feedback_gate_init)
    p.add_argument("--ratiolt1_entry_layers", type=int, default=ratiolt1_entry_layers)
    p.add_argument("--ratiolt1_loop_start_layer", type=int, default=ratiolt1_loop_start_layer)
    p.add_argument("--ratiolt1_loop_end_layer", type=int, default=ratiolt1_loop_end_layer)
    p.add_argument("--ratiolt1_num_loops", type=int, default=ratiolt1_num_loops)
    p.add_argument("--ratiolt1_feedback_gate_init", type=float, default=ratiolt1_feedback_gate_init)
    p.add_argument("--ratiogt1_entry_layers", type=int, default=ratiogt1_entry_layers)
    p.add_argument("--ratiogt1_loop_start_layer", type=int, default=ratiogt1_loop_start_layer)
    p.add_argument("--ratiogt1_loop_end_layer", type=int, default=ratiogt1_loop_end_layer)
    p.add_argument("--ratiogt1_num_loops", type=int, default=ratiogt1_num_loops)
    p.add_argument("--no_internal_cache", action="store_true")

    p.add_argument("--num_states", type=int, default=num_states)
    p.add_argument("--seq_len", type=int, default=seq_len)
    p.add_argument("--train_samples", type=int, default=train_samples)
    p.add_argument("--val_samples", type=int, default=val_samples)

    p.add_argument("--sudoku_root", type=str, default=sudoku_root)
    p.add_argument("--permutation_root", type=str, default=permutation_root)
    p.add_argument("--permutation_subset", type=str, default=permutation_subset)
    p.add_argument("--babi_root", type=str, default=babi_root)
    p.add_argument("--babi_version", type=str, default=babi_version)
    p.add_argument("--babi_task", type=str, default=babi_task, help="Task id such as qa1, qa2, ..., qa20")
    p.add_argument("--max_train_samples", type=int, default=max_train_samples)
    p.add_argument("--max_val_samples", type=int, default=max_val_samples, help="Validation samples held out from the training split")
    p.add_argument("--max_test_samples", type=int, default=max_test_samples, help="Final test samples from the test split")

    p.add_argument("--batch_size", type=int, default=batch_size)
    p.add_argument("--epochs", type=float, default=epochs, help="Number of passes over the training set")
    p.add_argument("--max_steps", type=int, default=max_steps, help="Manual override for total optimizer steps")
    p.add_argument("--lr", type=float, default=learning_rate)
    p.add_argument("--weight_decay", type=float, default=weight_decay)
    p.add_argument("--grad_clip", type=float, default=grad_clip)
    p.add_argument("--eval_every", type=int, default=eval_every)
    p.add_argument("--early_stopping", action="store_true", default=early_stopping)
    p.add_argument("--early_stopping_patience", type=int, default=early_stopping_patience)
    p.add_argument("--early_stopping_min_delta", type=float, default=early_stopping_min_delta)
    p.add_argument("--best_metric", choices=["val_loss", "val_token_acc", "val_exact_acc", "val_final_acc"], default=best_metric, help="Validation metric used to select the checkpoint for final test")
    p.add_argument("--seed", type=int, default=seed)
    p.add_argument("--device", type=str, default=device)
    p.add_argument("--wandb", action="store_true", help="Log metrics to Weights & Biases")
    p.add_argument("--wandb_project", type=str, default="state-reccurence")
    p.add_argument("--wandb_entity", type=str, default=None)
    p.add_argument("--wandb_run_name", type=str, default=None)
    p.add_argument("--wandb_dir", type=str, default="wandb")
    return p


def main() -> None:
    train(build_argparser().parse_args())


if __name__ == "__main__":
    main()
