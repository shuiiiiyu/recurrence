from __future__ import annotations

import math
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from recurrence_model.cli import build_argparser
from recurrence_model.data import PermutationDataset, collate_token_labels
from recurrence_model.models import make_model
from recurrence_model.training import build_config, cycle, evaluate, make_progress, maybe_init_wandb, set_seed, token_loss


def main() -> None:
    parser = build_argparser()
    parser.set_defaults(
        dataset="permutation",
        permutation_subset="S3_len100_100k",
        model="baseline",
        num_layers=4,
        d_model=128,
        num_heads=4,
        batch_size=8,
        max_train_samples=64,
        epochs=100,
        eval_every=20,
        device="cuda",
        lr=1e-4,
    )
    args = parser.parse_args()
    if args.dataset != "permutation":
        raise ValueError("overfit_permutation_train_eval.py is only intended for --dataset permutation")

    set_seed(args.seed)
    train_ds = PermutationDataset(
        args.permutation_root,
        args.permutation_subset,
        "train",
        args.max_train_samples,
        args.seed,
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_token_labels,
    )
    eval_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_token_labels,
    )

    cfg = build_config(args, train_ds.vocab_size, train_ds.max_seq_len)
    model = make_model(args.model, cfg).to(args.device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"overfit_train_eval dataset=permutation model={args.model} params={n_params/1e6:.2f}M device={args.device}")
    print(f"train_eval_samples={len(train_ds)} permutation_subset={args.permutation_subset}")
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

    progress = make_progress(range(1, total_steps + 1), total=total_steps, desc="overfit", initial=0)
    for step in progress:
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
        if step == 1 or step % args.eval_every == 0:
            train_eval_loss, token_acc, exact_acc, final_acc = evaluate(
                model,
                eval_loader,
                args.device,
                max_batches=len(eval_loader),
            )
            metrics = {
                "train/loss": loss.item(),
                "train_eval/loss": train_eval_loss,
                "train_eval/token_acc": token_acc,
                "train_eval/exact_acc": exact_acc,
                "train_eval/final_acc": final_acc,
                "step": step,
            }
            if wandb_run is not None:
                wandb_run.log(metrics, step=step)
            if hasattr(progress, "set_postfix"):
                progress.set_postfix(
                    loss=f"{loss.item():.4f}",
                    train_eval_loss=f"{train_eval_loss:.4f}",
                    token_acc=f"{token_acc:.4f}",
                    exact_acc=f"{exact_acc:.4f}",
                    final_acc=f"{final_acc:.4f}",
                    epoch=f"{current_epoch:.2f}/{effective_epochs:.2f}",
                )
            print(
                f"step={step:05d} train_loss={loss.item():.4f} "
                f"train_eval_loss={train_eval_loss:.4f} token_acc={token_acc:.4f} "
                f"exact_acc={exact_acc:.4f} final_acc={final_acc:.4f}"
            )
        elif hasattr(progress, "set_postfix"):
            progress.set_postfix(loss=f"{loss.item():.4f}", epoch=f"{current_epoch:.2f}/{effective_epochs:.2f}")

    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()
