"""
Metrics and baselines.

An MSE figure on its own says nothing. If most answers score 100, predicting
"100 every time" already achieves a low error, and a model that merely learned
the class balance looks successful. Every metric here is therefore reported
next to two baselines a real model has to beat.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

from dkt.model import ContinuousDKTLSTM


def report_baselines(df: pd.DataFrame, split: str = "val") -> dict[str, float]:
    """Error of two trivial predictors, measured on one held-out split.

    · **global mean** — always predict the training cohort's mean score.
    · **per-skill mean** — predict each skill's training mean, falling back to
      the global mean for skills unseen in training. This one is strong: much of
      the variance in exercise scores is item difficulty, not learner state, and
      a knowledge tracer earns its keep only by beating it.
    """
    train = df[df["split"] == "train"]
    held_out = df[df["split"] == split]

    if train.empty or held_out.empty:
        return {}

    global_mean = float(train["score_norm"].mean())
    skill_means = train.groupby("skill_idx")["score_norm"].mean()

    target = held_out["score_norm"].astype("float32")
    per_skill_pred = held_out["skill_idx"].map(skill_means).fillna(global_mean).astype("float32")

    results = {
        "baseline_global_mean_mse": float(((target - global_mean) ** 2).mean()),
        "baseline_global_mean_mae": float((target - global_mean).abs().mean()),
        "baseline_skill_mean_mse": float(((target - per_skill_pred) ** 2).mean()),
        "baseline_skill_mean_mae": float((target - per_skill_pred).abs().mean()),
    }

    print(f"[eval] baselines on the {split} split:")
    print(
        f"          global mean ({global_mean:.3f}): "
        f"MSE {results['baseline_global_mean_mse']:.6f} · "
        f"MAE {results['baseline_global_mean_mae']:.6f}"
    )
    print(
        f"          per-skill mean:        "
        f"MSE {results['baseline_skill_mean_mse']:.6f} · "
        f"MAE {results['baseline_skill_mean_mae']:.6f}"
    )
    return results


def _binary_auc(scores: torch.Tensor, labels: torch.Tensor) -> float:
    """ROC AUC via the rank identity, so SciPy is not a dependency.

    AUC equals the probability that a randomly chosen positive outranks a
    randomly chosen negative; the Mann-Whitney form below computes exactly that
    and handles ties by averaging ranks.
    """
    if labels.numel() == 0:
        return float("nan")
    n_pos = int(labels.sum())
    n_neg = int(labels.numel() - n_pos)
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    order = torch.argsort(scores)
    ranks = torch.empty_like(order, dtype=torch.float64)
    ranks[order] = torch.arange(1, scores.numel() + 1, dtype=torch.float64)

    # Average ranks within groups of equal score.
    sorted_scores = scores[order]
    i = 0
    while i < sorted_scores.numel():
        j = i
        while j + 1 < sorted_scores.numel() and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        if j > i:
            idx = order[i : j + 1]
            ranks[idx] = ranks[idx].mean()
        i = j + 1

    sum_pos_ranks = float(ranks[labels.bool()].sum())
    return (sum_pos_ranks - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


@torch.no_grad()
def evaluate_model(
    model: ContinuousDKTLSTM,
    loader: DataLoader,
    device: torch.device,
    pass_threshold: float = 0.5,
) -> dict[str, float]:
    """Regression and ranking metrics over every valid next-step position.

    ``pass_threshold`` binarises both prediction and target so the result can be
    compared with the AUC figures the knowledge-tracing literature reports for
    binary DKT. The binarisation is for comparability only — the model is
    trained and served as a regressor.
    """
    model.eval()

    preds: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []

    for batch in loader:
        skill_ids = batch["skill_ids"].to(device)
        score_norm = batch["score_norm"].to(device)
        time_norm = batch["time_norm"].to(device)
        lengths = batch["lengths"].to(device)

        pred, mask = model(skill_ids, score_norm, time_norm, lengths)
        if pred.numel() == 0:
            continue

        target = score_norm[:, 1:]
        preds.append(pred[mask].detach().cpu())
        targets.append(target[mask].detach().cpu())

    if not preds:
        return {}

    p = torch.cat(preds).double()
    t = torch.cat(targets).double()

    mse = float(((p - t) ** 2).mean())
    mae = float((p - t).abs().mean())

    # R²: the share of target variance the model explains. Negative means the
    # model is worse than predicting the mean, which MSE alone would not make
    # obvious.
    ss_res = float(((t - p) ** 2).sum())
    ss_tot = float(((t - t.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    labels = (t >= pass_threshold).double()
    auc = _binary_auc(p, labels)
    accuracy = float(((p >= pass_threshold).double() == labels).double().mean())

    return {
        "mse": mse,
        "rmse": mse**0.5,
        "mae": mae,
        "r2": r2,
        "auc_binarized": auc,
        "accuracy_binarized": accuracy,
        "n_predictions": float(p.numel()),
    }


def main() -> None:
    """Evaluate a saved checkpoint against a CSV, without retraining."""
    import argparse

    from dkt import config
    from dkt.data import build_loader, load_interactions, preprocess
    from dkt.inference import load_checkpoint

    parser = argparse.ArgumentParser(description="Evaluate a trained DKT checkpoint")
    parser.add_argument("--csv", type=Path, default=config.DATA.csv_path)
    parser.add_argument("--checkpoint", type=Path, default=config.TRAIN.checkpoint_path)
    parser.add_argument("--batch-size", type=int, default=config.TRAIN.batch_size)
    parser.add_argument(
        "--split",
        choices=("val", "test"),
        default="test",
        help="Which held-out split to score. Reported results use test",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, vocab = load_checkpoint(args.checkpoint, device)

    raw = load_interactions(args.csv)
    # Re-deriving the split from the same CSV and seed reproduces the exact
    # train/validation partition the checkpoint was fitted under. If the CSV has
    # changed since training, the mapping below will not match the checkpoint —
    # which the guard reports rather than silently scoring the wrong skills.
    df, fresh_vocab = preprocess(
        raw,
        score_scale=vocab.score_scale,
        train_user_frac=config.DATA.train_user_frac,
        val_user_frac=config.DATA.val_user_frac,
        min_seq_len=config.DATA.min_seq_len,
        seed=config.DATA.seed,
    )
    if fresh_vocab.skill_to_idx != vocab.skill_to_idx:
        print(
            "[eval] warning: this CSV yields a different skill index map than the checkpoint. "
            "Results below are not comparable to the training run; retrain or evaluate on the "
            "original export."
        )

    if not (df["split"] == args.split).any():
        raise SystemExit(
            f"The {args.split} split is empty. DKT_TRAIN_USER_FRAC and DKT_VAL_USER_FRAC "
            "leave no learners for it."
        )

    loader = build_loader(
        df, args.split, batch_size=args.batch_size, max_seq_len=config.DATA.max_seq_len
    )

    print(f"[eval] split: {args.split} · learners: {df[df['split'] == args.split]['user_id'].nunique()}")
    report_baselines(df, split=args.split)
    metrics = evaluate_model(model, loader, device)
    print("[eval] model:")
    for name, value in metrics.items():
        print(f"          {name}: {value:.6f}")


if __name__ == "__main__":
    main()
