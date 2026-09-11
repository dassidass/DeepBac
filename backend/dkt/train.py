"""
Training entry point.

    python -m dkt.train
    python -m dkt.train --csv data/interactions.csv --epochs 40 --hidden-dim 256

Writes a checkpoint containing the weights *and* the vocabulary. The two are
saved together on purpose: skill indices are derived from whatever data the run
saw, so weights without their vocabulary are not a model, they are noise
addressed by the wrong keys.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from dkt import config
from dkt.data import Vocabulary, build_loader, build_loaders, load_interactions, preprocess
from dkt.evaluate import evaluate_model, report_baselines
from dkt.model import ContinuousDKTLSTM, masked_mae, masked_mse


def run_epoch(
    model: ContinuousDKTLSTM,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
) -> tuple[float, float]:
    """One pass. Passing ``optimizer=None`` makes it an evaluation pass."""
    training = optimizer is not None
    model.train(training)

    total_mse = 0.0
    total_mae = 0.0
    n_batches = 0

    # torch.enable_grad is the default; disabling it for evaluation saves the
    # activation memory that would otherwise be retained for a backward pass
    # that never happens.
    with torch.set_grad_enabled(training):
        for batch in loader:
            skill_ids = batch["skill_ids"].to(device)
            score_norm = batch["score_norm"].to(device)
            time_norm = batch["time_norm"].to(device)
            lengths = batch["lengths"].to(device)

            pred, mask = model(skill_ids, score_norm, time_norm, lengths)
            if pred.numel() == 0:
                continue

            target = score_norm[:, 1:]
            loss = masked_mse(pred, target, mask)

            if training:
                optimizer.zero_grad()
                loss.backward()
                # LSTMs on long sequences produce occasional very large
                # gradients; clipping keeps one pathological batch from
                # destroying an otherwise converged model.
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.TRAIN.grad_clip)
                optimizer.step()

            total_mse += float(loss.detach().cpu())
            total_mae += float(masked_mae(pred, target, mask).detach().cpu())
            n_batches += 1

    n = max(n_batches, 1)
    return total_mse / n, total_mae / n


def save_checkpoint(
    path: Path,
    model: ContinuousDKTLSTM,
    vocab: Vocabulary,
    metrics: dict,
    args: argparse.Namespace,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "vocab": vocab.to_dict(),
            "model_config": {
                "num_skills": model.num_skills,
                "embed_dim": model.embed_dim,
                "hidden_dim": model.hidden_dim,
                "num_layers": args.layers,
                "dropout": args.dropout,
            },
            "metrics": metrics,
        },
        path,
    )
    print(f"[train] checkpoint written to {path}")


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train the continuous DKT-LSTM")
    p.add_argument("--csv", type=Path, default=config.DATA.csv_path)
    p.add_argument("--epochs", type=int, default=config.TRAIN.epochs)
    p.add_argument("--batch-size", type=int, default=config.TRAIN.batch_size)
    p.add_argument("--lr", type=float, default=config.TRAIN.lr)
    p.add_argument("--embed-dim", type=int, default=config.MODEL.embed_dim)
    p.add_argument("--hidden-dim", type=int, default=config.MODEL.hidden_dim)
    p.add_argument("--layers", type=int, default=config.MODEL.num_layers)
    p.add_argument("--dropout", type=float, default=config.MODEL.dropout)
    p.add_argument("--max-seq-len", type=int, default=config.DATA.max_seq_len)
    p.add_argument("--score-scale", type=float, default=config.DATA.score_scale)
    p.add_argument("--patience", type=int, default=config.TRAIN.patience)
    p.add_argument("--seed", type=int, default=config.DATA.seed)
    p.add_argument("--checkpoint", type=Path, default=config.TRAIN.checkpoint_path)
    p.add_argument(
        "--no-save",
        action="store_true",
        help="Run without writing a checkpoint (hyper-parameter sweeps)",
    )
    return p


def main() -> None:
    args = build_arg_parser().parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train] device: {device}")

    raw = load_interactions(args.csv)
    df, vocab = preprocess(
        raw,
        score_scale=args.score_scale,
        train_user_frac=config.DATA.train_user_frac,
        val_user_frac=config.DATA.val_user_frac,
        min_seq_len=config.DATA.min_seq_len,
        seed=args.seed,
    )
    split_users = df.groupby("split")["user_id"].nunique().to_dict()
    print(
        f"[train] users: {df['user_id'].nunique()} · interactions: {len(df)} · "
        f"skills: {vocab.num_skills - 1} (+1 unknown slot)"
    )
    print(
        "[train] learners per split — "
        + " · ".join(f"{name}: {split_users.get(name, 0)}" for name in ("train", "val", "test"))
    )

    train_loader, val_loader = build_loaders(
        df, batch_size=args.batch_size, max_seq_len=args.max_seq_len
    )
    print(
        f"[train] sequences — train: {len(train_loader.dataset)} · val: {len(val_loader.dataset)}"
    )
    if len(val_loader.dataset) == 0:
        raise SystemExit("Validation split is empty. Lower DKT_TRAIN_USER_FRAC or add learners.")

    # Predicting the training mean is the score any useful model must beat. It
    # is printed before training so the epoch numbers below have a reference.
    report_baselines(df)

    model = ContinuousDKTLSTM(
        num_skills=vocab.num_skills,
        embed_dim=args.embed_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.layers,
        dropout=args.dropout,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2
    )

    best_val = float("inf")
    best_state: dict | None = None
    epochs_without_improvement = 0

    for epoch in range(1, args.epochs + 1):
        tr_mse, tr_mae = run_epoch(model, train_loader, optimizer, device)
        val_mse, val_mae = run_epoch(model, val_loader, None, device)
        scheduler.step(val_mse)

        print(
            f"[train] epoch {epoch:03d}  "
            f"train MSE {tr_mse:.6f} MAE {tr_mae:.6f}  |  "
            f"val MSE {val_mse:.6f} MAE {val_mae:.6f}"
        )

        if val_mse < best_val - 1e-6:
            best_val = val_mse
            # Kept on CPU so the checkpoint loads on a machine without a GPU.
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.patience:
                print(f"[train] early stop at epoch {epoch} (no improvement in {args.patience})")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    print(f"[train] best validation MSE: {best_val:.6f}")
    metrics = {"val": evaluate_model(model, val_loader, device)}
    print("[train] validation metrics:")
    for name, value in metrics["val"].items():
        print(f"          {name}: {value:.6f}")

    # The test split is scored exactly here: once, after the epoch has been
    # chosen on validation. Any number quoted outside this project should come
    # from this block, not from the line above it.
    if (df["split"] == "test").any():
        test_loader = build_loader(
            df, "test", batch_size=args.batch_size, max_seq_len=args.max_seq_len
        )
        report_baselines(df, split="test")
        metrics["test"] = evaluate_model(model, test_loader, device)
        print("[train] test metrics:")
        for name, value in metrics["test"].items():
            print(f"          {name}: {value:.6f}")
    else:
        print(
            "[train] no test split — DKT_TRAIN_USER_FRAC and DKT_VAL_USER_FRAC leave "
            "no learners for one"
        )

    if not args.no_save:
        save_checkpoint(args.checkpoint, model, vocab, metrics, args)


if __name__ == "__main__":
    main()
