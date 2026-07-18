from __future__ import annotations

import copy
import itertools
import math
import random
from argparse import Namespace
from typing import Iterable, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .config import ModelConfig
from .data import collate_token_labels, ignore_label_id, make_datasets
from .models import make_model

try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None


def maybe_init_wandb(args: Namespace, cfg: ModelConfig, n_params: int):
    if not args.wandb:
        return None
    try:
        import wandb
    except ImportError as exc:
        raise ImportError("wandb logging was requested, but wandb is not installed in this environment") from exc

    run = wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=args.wandb_run_name,
        dir=args.wandb_dir,
        config={
            **vars(args),
            "params": n_params,
            "resolved_vocab_size": cfg.vocab_size,
            "resolved_max_seq_len": cfg.max_seq_len,
        },
    )
    return run


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def cycle(loader: DataLoader) -> Iterable[dict]:
    while True:
        for batch in loader:
            yield batch


def token_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(logits.reshape(-1, logits.size(-1)), labels.reshape(-1), ignore_index=ignore_label_id)


@torch.no_grad()
def model_debug_metrics(model: nn.Module) -> dict:
    metrics = {}
    gate_logit = getattr(model, "feedback_gate_logit", None)
    if gate_logit is not None:
        metrics["debug/feedback_gate"] = torch.sigmoid(gate_logit.detach()).item()
        metrics["debug/feedback_gate_logit"] = gate_logit.detach().item()
    return metrics


@torch.no_grad()
def batch_metrics(logits: torch.Tensor, labels: torch.Tensor) -> Tuple[int, int, int, int, int, int]:
    preds = logits.argmax(dim=-1)
    mask = labels.ne(ignore_label_id)
    token_correct = preds.eq(labels).logical_and(mask).sum().item()
    token_total = mask.sum().item()

    has_label = mask.any(dim=1)
    exact = preds.eq(labels).logical_or(~mask).all(dim=1).logical_and(has_label)
    exact_correct = exact.sum().item()
    exact_total = has_label.sum().item()
    final_correct = 0
    final_total = 0
    for row_idx in has_label.nonzero(as_tuple=False).flatten():
        label_positions = mask[row_idx].nonzero(as_tuple=False).flatten()
        final_pos = label_positions[-1]
        final_correct += int(preds[row_idx, final_pos].item() == labels[row_idx, final_pos].item())
        final_total += 1
    return token_correct, token_total, exact_correct, exact_total, final_correct, final_total


def make_progress(iterable, **kwargs):
    if tqdm is None:
        return iterable
    return tqdm(iterable, dynamic_ncols=True, **kwargs)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: str,
    max_batches: int | None = None,
    show_progress: bool = True,
) -> Tuple[float, float, float, float]:
    model.eval()
    total_loss = 0.0
    total_items = 0
    token_correct = 0
    token_total = 0
    exact_correct = 0
    exact_total = 0
    final_correct = 0
    final_total = 0
    eval_batches = len(loader) if max_batches is None else min(max_batches, len(loader))
    iterator = enumerate(itertools.islice(loader, eval_batches))
    if show_progress:
        iterator = make_progress(iterator, total=eval_batches, desc="eval", leave=False)
    for idx, batch in iterator:
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        logits = model(input_ids)
        loss = token_loss(logits, labels)
        bsz = input_ids.size(0)
        total_loss += loss.item() * bsz
        total_items += bsz
        tc, tt, ec, et, fc, ft = batch_metrics(logits, labels)
        token_correct += tc
        token_total += tt
        exact_correct += ec
        exact_total += et
        final_correct += fc
        final_total += ft
        if show_progress and tqdm is not None:
            iterator.set_postfix(
                loss=f"{total_loss / max(1, total_items):.4f}",
                token_acc=f"{token_correct / max(1, token_total):.4f}",
                exact_acc=f"{exact_correct / max(1, exact_total):.4f}",
                final_acc=f"{final_correct / max(1, final_total):.4f}",
            )
    model.train()
    return (
        total_loss / total_items,
        token_correct / max(1, token_total),
        exact_correct / max(1, exact_total),
        final_correct / max(1, final_total),
    )


def build_config(args: Namespace, vocab_size: int, max_seq_len: int) -> ModelConfig:
    return ModelConfig(
        num_layers=args.num_layers,
        d_model=args.d_model,
        num_heads=args.num_heads,
        ffn_mult=args.ffn_mult,
        dropout=args.dropout,
        vocab_size=vocab_size,
        num_classes=vocab_size,
        max_seq_len=max_seq_len,
        ratio1_feedback_source_layer=args.ratio1_feedback_source_layer,
        ratio1_feedback_target_layer=args.ratio1_feedback_target_layer,
        ratiolt1_entry_layers=args.ratiolt1_entry_layers,
        ratiolt1_loop_start_layer=args.ratiolt1_loop_start_layer,
        ratiolt1_loop_end_layer=args.ratiolt1_loop_end_layer,
        ratiolt1_num_loops=args.ratiolt1_num_loops,
        append_internal_steps_to_cache=not args.no_internal_cache,
        ratiogt1_entry_layers=args.ratiogt1_entry_layers,
        ratiogt1_loop_start_layer=args.ratiogt1_loop_start_layer,
        ratiogt1_loop_end_layer=args.ratiogt1_loop_end_layer,
        ratiogt1_num_loops=args.ratiogt1_num_loops,
    )


def train(args: Namespace) -> None:
    set_seed(args.seed)
    train_ds, val_ds, test_ds = make_datasets(args)
    cfg = build_config(args, train_ds.vocab_size, train_ds.max_seq_len)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0, collate_fn=collate_token_labels)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=collate_token_labels)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=collate_token_labels)
    model = make_model(args.model, cfg).to(args.device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"dataset={args.dataset} model={args.model} params={n_params/1e6:.2f}M device={args.device}")
    print(f"train_samples={len(train_ds)} val_samples={len(val_ds)} test_samples={len(test_ds)}")
    print(cfg)
    wandb_run = maybe_init_wandb(args, cfg, n_params)

    steps_per_epoch = math.ceil(len(train_ds) / args.batch_size)
    if args.max_steps is not None:
        total_steps = args.max_steps
        effective_epochs = total_steps / steps_per_epoch
        print(f"steps_per_epoch={steps_per_epoch} total_steps={total_steps} effective_epochs={effective_epochs:.3f} (max_steps override)")
    else:
        total_steps = math.ceil(args.epochs * steps_per_epoch)
        effective_epochs = total_steps / steps_per_epoch
        print(f"steps_per_epoch={steps_per_epoch} epochs={args.epochs} total_steps={total_steps} effective_epochs={effective_epochs:.3f}")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    train_iter = cycle(train_loader)
    model.train()
    best_val_loss = float("inf")
    best_step = 0
    best_state = copy.deepcopy(model.state_dict())
    bad_evals = 0

    progress = make_progress(range(1, total_steps + 1), total=total_steps, desc="train", initial=0)
    last_step = 0
    for step in progress:
        last_step = step
        batch = next(train_iter)
        input_ids = batch["input_ids"].to(args.device)
        labels = batch["labels"].to(args.device)
        logits = model(input_ids)
        loss = token_loss(logits, labels)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        opt.step()

        current_epoch = step / steps_per_epoch
        if tqdm is not None:
            progress.set_postfix(
                loss=f"{loss.item():.4f}",
                epoch=f"{current_epoch:.2f}/{effective_epochs:.2f}",
            )

        if step == 1 or step % args.eval_every == 0:
            val_loss, token_acc, exact_acc, final_acc = evaluate(model, val_loader, args.device)
            improved = val_loss < best_val_loss - args.early_stopping_min_delta
            if improved:
                best_val_loss = val_loss
                best_step = step
                best_state = copy.deepcopy(model.state_dict())
                bad_evals = 0
            else:
                bad_evals += 1
            metrics = {
                "train/loss": loss.item(),
                "val/loss": val_loss,
                "val/token_acc": token_acc,
                "val/exact_acc": exact_acc,
                "val/final_acc": final_acc,
                "early_stop/best_val_loss": best_val_loss,
                "early_stop/best_step": best_step,
                "early_stop/bad_evals": bad_evals,
                "step": step,
            }
            metrics.update(model_debug_metrics(model))
            if wandb_run is not None:
                wandb_run.log(metrics, step=step)
            if tqdm is not None:
                progress.set_postfix(
                    loss=f"{loss.item():.4f}",
                    val_loss=f"{val_loss:.4f}",
                    token_acc=f"{token_acc:.4f}",
                    exact_acc=f"{exact_acc:.4f}",
                    final_acc=f"{final_acc:.4f}",
                    epoch=f"{current_epoch:.2f}/{effective_epochs:.2f}",
                )
            print(
                f"step={step:05d} train_loss={loss.item():.4f} "
                f"val_loss={val_loss:.4f} token_acc={token_acc:.4f} "
                f"exact_acc={exact_acc:.4f} final_acc={final_acc:.4f} "
                f"best_step={best_step} bad_evals={bad_evals}"
            )
            if args.early_stopping and bad_evals >= args.early_stopping_patience:
                print(
                    f"early stopping at step={step}; "
                    f"best_step={best_step} best_val_loss={best_val_loss:.4f}"
                )
                break

    model.load_state_dict(best_state)
    test_loss, test_token_acc, test_exact_acc, test_final_acc = evaluate(model, test_loader, args.device)
    final_metrics = {
        "test/loss": test_loss,
        "test/token_acc": test_token_acc,
        "test/exact_acc": test_exact_acc,
        "test/final_acc": test_final_acc,
        "test/best_step": best_step,
    }
    final_metrics.update(model_debug_metrics(model))
    if wandb_run is not None:
        wandb_run.log(final_metrics, step=last_step)
    print(
        f"test best_step={best_step} test_loss={test_loss:.4f} "
        f"token_acc={test_token_acc:.4f} exact_acc={test_exact_acc:.4f} "
        f"final_acc={test_final_acc:.4f}"
    )
    if wandb_run is not None:
        wandb_run.finish()
