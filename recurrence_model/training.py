from __future__ import annotations

import csv
import copy
from contextlib import nullcontext
from datetime import datetime
import itertools
import math
import os
import random
import time
from argparse import Namespace
from pathlib import Path
from typing import Iterable, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

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


def cycle(loader: DataLoader, sampler: DistributedSampler | None = None) -> Iterable[dict]:
    epoch = 0
    while True:
        if sampler is not None:
            sampler.set_epoch(epoch)
        for batch in loader:
            yield batch
        epoch += 1


def token_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(logits.float().reshape(-1, logits.size(-1)), labels.reshape(-1), ignore_index=ignore_label_id)


def autocast_context(args_or_device, amp: str | None = None):
    if isinstance(args_or_device, Namespace):
        device = args_or_device.device
        amp_mode = args_or_device.amp
    else:
        device = str(args_or_device)
        amp_mode = "none" if amp is None else amp
    if amp_mode == "bf16" and str(device).startswith("cuda"):
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def best_metric_score(metrics: dict, metric_name: str) -> tuple[tuple[float, ...], tuple[bool, ...]]:
    metric_specs = {
        "val_final_acc_then_loss": (("val_final_acc", True), ("val_loss", False)),
        "val_exact_acc_then_loss": (("val_exact_acc", True), ("val_loss", False)),
        "val_token_acc_then_loss": (("val_token_acc", True), ("val_loss", False)),
    }
    if metric_name in metric_specs:
        specs = metric_specs[metric_name]
    elif metric_name in metrics:
        specs = ((metric_name, metric_name != "val_loss"),)
    else:
        raise ValueError(
            f"Unknown best_metric={metric_name}; available metrics: "
            f"{sorted(metrics)} plus {sorted(metric_specs)}"
        )
    return tuple(float(metrics[name]) for name, _ in specs), tuple(higher for _, higher in specs)


def initial_best_score(higher_is_better: tuple[bool, ...]) -> tuple[float, ...]:
    return tuple(-float("inf") if higher else float("inf") for higher in higher_is_better)


def is_better_score(
    score: tuple[float, ...],
    best_score: tuple[float, ...],
    higher_is_better: tuple[bool, ...],
    min_delta: float,
) -> bool:
    for cur, best, higher in zip(score, best_score, higher_is_better):
        if higher:
            if cur > best + min_delta:
                return True
            if cur < best - min_delta:
                return False
        else:
            if cur < best - min_delta:
                return True
            if cur > best + min_delta:
                return False
    return False


def format_best_score(score: tuple[float, ...]) -> str:
    if len(score) == 1:
        return f"{score[0]:.4f}"
    return "(" + ", ".join(f"{value:.4f}" for value in score) + ")"


def format_duration(seconds: float) -> str:
    seconds = max(0, int(seconds))
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours:d}h{minutes:02d}m{secs:02d}s"
    return f"{minutes:d}m{secs:02d}s"


def text_progress(step: int, total_steps: int, start_time: float, width: int = 24) -> str:
    frac = min(1.0, max(0.0, step / max(1, total_steps)))
    filled = int(round(width * frac))
    bar = "#" * filled + "-" * (width - filled)
    elapsed = time.time() - start_time
    eta = elapsed * (1.0 / frac - 1.0) if frac > 0 else 0.0
    return (
        f"progress=[{bar}] {step}/{total_steps} {frac * 100:.1f}% "
        f"elapsed={format_duration(elapsed)} eta={format_duration(eta)}"
    )


def setup_distributed(args: Namespace) -> tuple[bool, int, int, int, str]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size <= 1:
        return False, 0, 0, 1, args.device
    if not torch.cuda.is_available():
        raise RuntimeError("DDP training requires CUDA")
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    return True, rank, local_rank, world_size, f"cuda:{local_rank}"


def cleanup_distributed(is_ddp: bool) -> None:
    if is_ddp and dist.is_initialized():
        dist.destroy_process_group()


def is_main_process(rank: int) -> bool:
    return rank == 0


def main_print(rank: int, *args, **kwargs) -> None:
    if is_main_process(rank):
        print(*args, **kwargs)


def reduce_mean_scalar(value: float, device: str, is_ddp: bool) -> float:
    if not is_ddp:
        return float(value)
    tensor = torch.tensor(float(value), device=device)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    tensor /= dist.get_world_size()
    return tensor.item()


def unwrap_model(model: nn.Module) -> nn.Module:
    return model.module if isinstance(model, (nn.DataParallel, DistributedDataParallel)) else model


def model_state_dict(model: nn.Module) -> dict:
    return copy.deepcopy(unwrap_model(model).state_dict())


def load_model_state_dict(model: nn.Module, state_dict: dict) -> None:
    normalized = {}
    for key, value in state_dict.items():
        normalized[key.removeprefix("module.")] = value
    unwrap_model(model).load_state_dict(normalized)


def safe_name(text: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in text).strip("_")


def default_checkpoint_path(args: Namespace) -> Path:
    if args.wandb_run_name:
        name = args.wandb_run_name
    elif args.dataset == "permutation":
        name = f"{args.dataset}-{args.permutation_subset}-{args.model}-seed{args.seed}"
    elif args.dataset == "babi":
        name = f"{args.dataset}-{args.babi_version}-{args.babi_task}-{args.model}-seed{args.seed}"
    else:
        name = f"{args.dataset}-{args.model}-seed{args.seed}"
    return Path(args.checkpoint_dir) / f"{safe_name(name)}.pt"


def save_checkpoint(
    path: Path,
    model: nn.Module,
    args: Namespace,
    cfg: ModelConfig,
    step: int,
    best_metric: str,
    best_score: float,
    best_metric_values: dict,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state": model_state_dict(model),
        "args": vars(args),
        "model_config": cfg.__dict__,
        "step": step,
        "best_metric": best_metric,
        "best_score": best_score,
        "best_metric_values": best_metric_values,
    }
    torch.save(payload, path)


def load_checkpoint(model: nn.Module, checkpoint_path: str, device: str) -> dict:
    payload = torch.load(checkpoint_path, map_location=device)
    state_dict = payload["model_state"] if isinstance(payload, dict) and "model_state" in payload else payload
    load_model_state_dict(model, state_dict)
    return payload if isinstance(payload, dict) else {"model_state": state_dict}


def append_result_csv(
    args: Namespace,
    cfg: ModelConfig,
    n_params: int,
    status: str,
    selected_step: int | str,
    metrics: dict,
    checkpoint_path: Path,
    world_size: int,
) -> None:
    if args.no_save_results:
        return
    path = Path(args.results_csv)
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "dataset": args.dataset,
        "subset": args.permutation_subset if args.dataset == "permutation" else args.babi_version if args.dataset == "babi" else "",
        "task": args.babi_task if args.dataset == "babi" else "",
        "model": args.model,
        "status": status,
        "source": "test",
        "selected_step": selected_step,
        "loss": metrics.get("test/loss", ""),
        "token_acc": metrics.get("test/token_acc", ""),
        "exact_acc": metrics.get("test/exact_acc", ""),
        "final_acc": metrics.get("test/final_acc", ""),
        "train_loss": metrics.get("train/loss", ""),
        "best_metric": args.best_metric,
        "best_score": metrics.get("best/metric_score", ""),
        "num_layers": cfg.num_layers,
        "d_model": cfg.d_model,
        "num_heads": cfg.num_heads,
        "ffn_mult": cfg.ffn_mult,
        "batch_size_per_gpu": args.batch_size,
        "world_size": world_size,
        "global_batch_size": args.batch_size * world_size,
        "max_steps": args.max_steps,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "amp": args.amp,
        "seed": args.seed,
        "params": n_params,
        "wandb_project": args.wandb_project,
        "wandb_run_name": args.wandb_run_name or "",
        "checkpoint_path": str(checkpoint_path),
    }
    write_header = not path.exists()
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


@torch.no_grad()
def model_debug_metrics(model: nn.Module) -> dict:
    metrics = {}
    inner_model = unwrap_model(model)
    gate_logit = getattr(inner_model, "feedback_gate_logit", None)
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
    return iterable


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: str,
    amp: str = "none",
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
        attention_mask = batch["attention_mask"].to(device)
        with autocast_context(device, amp):
            logits = model(input_ids, attention_mask=attention_mask)
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
        if show_progress and hasattr(iterator, "set_postfix"):
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
        ratio1_feedback_gate_init=args.ratio1_feedback_gate_init,
        ratiolt1_entry_layers=args.ratiolt1_entry_layers,
        ratiolt1_loop_start_layer=args.ratiolt1_loop_start_layer,
        ratiolt1_loop_end_layer=args.ratiolt1_loop_end_layer,
        ratiolt1_num_loops=args.ratiolt1_num_loops,
        ratiolt1_feedback_gate_init=args.ratiolt1_feedback_gate_init,
        append_internal_steps_to_cache=not args.no_internal_cache,
        ratiogt1_entry_layers=args.ratiogt1_entry_layers,
        ratiogt1_loop_start_layer=args.ratiogt1_loop_start_layer,
        ratiogt1_loop_end_layer=args.ratiogt1_loop_end_layer,
        ratiogt1_num_loops=args.ratiogt1_num_loops,
        depthstepgt1_chunk_size=args.depthstepgt1_chunk_size,
        depthstepgt1_memory_tokens=args.depthstepgt1_memory_tokens,
        depthstepgt1_feedback_source_layer=args.depthstepgt1_feedback_source_layer,
        depthstepgt1_feedback_target_layer=args.depthstepgt1_feedback_target_layer,
        depthstepgt1_feedback_gate_init=args.depthstepgt1_feedback_gate_init,
        depthratio1_feedback_source_layer=args.depthratio1_feedback_source_layer,
        depthratio1_feedback_target_layer=args.depthratio1_feedback_target_layer,
        depthratio1_feedback_gate_init=args.depthratio1_feedback_gate_init,
        depthratiolt1_num_steps=args.depthratiolt1_num_steps,
        depthratiolt1_feedback_source_layer=args.depthratiolt1_feedback_source_layer,
        depthratiolt1_feedback_target_layer=args.depthratiolt1_feedback_target_layer,
        depthratiolt1_feedback_gate_init=args.depthratiolt1_feedback_gate_init,
        deltaproduct_num_householder=args.deltaproduct_num_householder,
        deltaproduct_use_output_gate=args.deltaproduct_use_output_gate,
        deltaproduct_use_forget_gate=args.deltaproduct_use_forget_gate,
        deltaproduct_allow_neg_eigval=args.deltaproduct_allow_neg_eigval,
        deltaproduct_use_short_conv=args.deltaproduct_use_short_conv,
        deltaproduct_conv_size=args.deltaproduct_conv_size,
        mamba_d_state=args.mamba_d_state,
        mamba_d_conv=args.mamba_d_conv,
        mamba_expand=args.mamba_expand,
        mamba_use_fast_path=args.mamba_use_fast_path,
        brt_block_size=args.brt_block_size,
        brt_num_states=args.brt_num_states,
        brt_gate_type=args.brt_gate_type,
        brt_single_gate=args.brt_single_gate,
        brt_skip_ffn=args.brt_skip_ffn,
        brt_layer_indices=args.brt_layer_indices,
    )


def train(args: Namespace) -> None:
    is_ddp, rank, local_rank, world_size, device = setup_distributed(args)
    args.device = device
    try:
        set_seed(args.seed + rank)
        train_ds, val_ds, test_ds = make_datasets(args)
        cfg = build_config(args, train_ds.vocab_size, train_ds.max_seq_len)

        train_sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True, seed=args.seed) if is_ddp else None
        train_loader = DataLoader(
            train_ds,
            batch_size=args.batch_size,
            shuffle=train_sampler is None,
            sampler=train_sampler,
            num_workers=0,
            collate_fn=collate_token_labels,
        )
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=collate_token_labels)
        test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=collate_token_labels)
        model = make_model(args.model, cfg).to(args.device)
        if is_ddp:
            model = DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank)
        n_params = sum(p.numel() for p in unwrap_model(model).parameters())
        main_print(rank, f"dataset={args.dataset} model={args.model} params={n_params/1e6:.2f}M device={args.device}")
        main_print(rank, f"ddp={is_ddp} world_size={world_size} batch_size_per_gpu={args.batch_size} global_batch_size={args.batch_size * world_size}")
        main_print(rank, f"amp={args.amp}")
        main_print(rank, f"train_samples={len(train_ds)} val_samples={len(val_ds)} test_samples={len(test_ds)}")
        main_print(rank, cfg)
        wandb_run = maybe_init_wandb(args, cfg, n_params) if is_main_process(rank) else None
        checkpoint_path = default_checkpoint_path(args)

        if args.eval_checkpoint is not None:
            payload = load_checkpoint(model, args.eval_checkpoint, args.device)
            checkpoint_step = payload.get("step", "")
            if is_ddp:
                dist.barrier()
            if is_main_process(rank):
                print(f"loaded checkpoint={args.eval_checkpoint} checkpoint_step={checkpoint_step}")
                test_loss, test_token_acc, test_exact_acc, test_final_acc = evaluate(unwrap_model(model), test_loader, args.device, amp=args.amp)
                final_metrics = {
                    "test/loss": test_loss,
                    "test/token_acc": test_token_acc,
                    "test/exact_acc": test_exact_acc,
                    "test/final_acc": test_final_acc,
                    "test/best_step": checkpoint_step,
                }
                final_metrics.update(model_debug_metrics(model))
                append_result_csv(args, cfg, n_params, "eval_checkpoint", checkpoint_step, final_metrics, checkpoint_path, world_size)
                if wandb_run is not None:
                    wandb_run.log(final_metrics)
                    wandb_run.finish()
                print(
                    f"test checkpoint_step={checkpoint_step} test_loss={test_loss:.4f} "
                    f"token_acc={test_token_acc:.4f} exact_acc={test_exact_acc:.4f} "
                    f"final_acc={test_final_acc:.4f}"
                )
            if is_ddp:
                dist.barrier()
            return

        if args.init_checkpoint is not None:
            payload = load_checkpoint(model, args.init_checkpoint, args.device)
            checkpoint_step = payload.get("step", "")
            main_print(rank, f"initialized training from checkpoint={args.init_checkpoint} checkpoint_step={checkpoint_step}")

        if not args.no_save_checkpoint:
            main_print(rank, f"best checkpoints will be saved to {checkpoint_path}")

        steps_per_epoch = len(train_loader)
        if args.max_steps is not None:
            total_steps = args.max_steps
            effective_epochs = total_steps / steps_per_epoch
            main_print(rank, f"steps_per_epoch={steps_per_epoch} total_steps={total_steps} effective_epochs={effective_epochs:.3f} (max_steps override)")
        else:
            total_steps = math.ceil(args.epochs * steps_per_epoch)
            effective_epochs = total_steps / steps_per_epoch
            main_print(rank, f"steps_per_epoch={steps_per_epoch} epochs={args.epochs} total_steps={total_steps} effective_epochs={effective_epochs:.3f}")

        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        train_iter = cycle(train_loader, train_sampler)
        model.train()
        best_score, higher_is_better = best_metric_score(
            {"val_loss": 0.0, "val_token_acc": 0.0, "val_exact_acc": 0.0, "val_final_acc": 0.0},
            args.best_metric,
        )
        best_score = initial_best_score(higher_is_better)
        best_metric_values = {}
        best_step = 0
        best_state = model_state_dict(model) if is_main_process(rank) else None
        bad_evals = 0

        progress_iter = range(1, total_steps + 1)
        progress = make_progress(progress_iter, total=total_steps, desc="train", initial=0) if is_main_process(rank) else progress_iter
        last_step = 0
        last_train_loss = None
        start_time = time.time()
        for step in progress:
            last_step = step
            batch = next(train_iter)
            input_ids = batch["input_ids"].to(args.device)
            labels = batch["labels"].to(args.device)
            attention_mask = batch["attention_mask"].to(args.device)
            with autocast_context(args):
                logits = model(input_ids, attention_mask=attention_mask)
                loss = token_loss(logits, labels)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step()
            train_loss = reduce_mean_scalar(loss.item(), args.device, is_ddp)
            last_train_loss = train_loss

            current_epoch = step / steps_per_epoch
            if is_main_process(rank) and hasattr(progress, "set_postfix"):
                progress.set_postfix(
                    loss=f"{train_loss:.4f}",
                    epoch=f"{current_epoch:.2f}/{effective_epochs:.2f}",
                )

            stop_training = False
            if step == 1 or step % args.eval_every == 0:
                if is_main_process(rank):
                    val_loss, token_acc, exact_acc, final_acc = evaluate(unwrap_model(model), val_loader, args.device, amp=args.amp)
                    val_metrics = {
                        "val_loss": val_loss,
                        "val_token_acc": token_acc,
                        "val_exact_acc": exact_acc,
                        "val_final_acc": final_acc,
                    }
                    score, higher_is_better = best_metric_score(val_metrics, args.best_metric)
                    improved = is_better_score(score, best_score, higher_is_better, args.early_stopping_min_delta)
                    if improved:
                        best_score = score
                        best_metric_values = dict(val_metrics)
                        best_step = step
                        best_state = model_state_dict(model)
                        if not args.no_save_checkpoint:
                            save_checkpoint(
                                checkpoint_path,
                                model,
                                args,
                                cfg,
                                step,
                                args.best_metric,
                                best_score,
                                best_metric_values,
                            )
                        bad_evals = 0
                    else:
                        bad_evals += 1
                    metrics = {
                        "train/loss": train_loss,
                        "val/loss": val_loss,
                        "val/token_acc": token_acc,
                        "val/exact_acc": exact_acc,
                        "val/final_acc": final_acc,
                        "best/metric_score": best_score[0],
                        "best/step": best_step,
                        "best/bad_evals": bad_evals,
                        "step": step,
                    }
                    if len(best_score) > 1:
                        metrics["best/tiebreak_score"] = best_score[1]
                    for name, value in best_metric_values.items():
                        metrics[f"best/{name}"] = value
                    metrics.update(model_debug_metrics(model))
                    if wandb_run is not None:
                        wandb_run.log(metrics, step=step)
                    if hasattr(progress, "set_postfix"):
                        progress.set_postfix(
                            loss=f"{train_loss:.4f}",
                            val_loss=f"{val_loss:.4f}",
                            token_acc=f"{token_acc:.4f}",
                            exact_acc=f"{exact_acc:.4f}",
                            final_acc=f"{final_acc:.4f}",
                            epoch=f"{current_epoch:.2f}/{effective_epochs:.2f}",
                        )
                    print(
                        f"{text_progress(step, total_steps, start_time)} "
                        f"step={step:05d} train_loss={train_loss:.4f} "
                        f"val_loss={val_loss:.4f} token_acc={token_acc:.4f} "
                        f"exact_acc={exact_acc:.4f} final_acc={final_acc:.4f} "
                        f"best_metric={args.best_metric} best_score={format_best_score(best_score)} "
                        f"best_step={best_step} bad_evals={bad_evals}"
                    )
                    if args.early_stopping and bad_evals >= args.early_stopping_patience:
                        print(
                            f"early stopping at step={step}; "
                            f"best_step={best_step} best_metric={args.best_metric} "
                            f"best_score={format_best_score(best_score)}"
                        )
                        stop_training = True
                if is_ddp:
                    stop_tensor = torch.tensor(int(stop_training), device=args.device)
                    dist.broadcast(stop_tensor, src=0)
                    stop_training = bool(stop_tensor.item())
                if stop_training:
                    break

        if is_ddp:
            dist.barrier()
        if is_main_process(rank):
            load_model_state_dict(model, best_state)
            test_loss, test_token_acc, test_exact_acc, test_final_acc = evaluate(unwrap_model(model), test_loader, args.device, amp=args.amp)
            final_metrics = {
                "test/loss": test_loss,
                "test/token_acc": test_token_acc,
                "test/exact_acc": test_exact_acc,
                "test/final_acc": test_final_acc,
                "test/best_step": best_step,
                "train/loss": last_train_loss if last_train_loss is not None else "",
                "best/metric_score": best_score[0],
            }
            if len(best_score) > 1:
                final_metrics["best/tiebreak_score"] = best_score[1]
            final_metrics.update(model_debug_metrics(model))
            append_result_csv(args, cfg, n_params, "completed", best_step, final_metrics, checkpoint_path, world_size)
            if wandb_run is not None:
                wandb_run.log(final_metrics, step=last_step)
            print(
                f"test best_step={best_step} test_loss={test_loss:.4f} "
                f"token_acc={test_token_acc:.4f} exact_acc={test_exact_acc:.4f} "
                f"final_acc={test_final_acc:.4f}"
            )
            if wandb_run is not None:
                wandb_run.finish()
        if is_ddp:
            dist.barrier()
    finally:
        cleanup_distributed(is_ddp)
