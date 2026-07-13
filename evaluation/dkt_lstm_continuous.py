#!/usr/bin/env python3
"""
Deep Knowledge Tracing (DKT) — LSTM variant for continuous score prediction.

Tailored to CSV schema:
  user_id, subject, course_id, course_part_id, score, question_started_at, respones_time_ms

Uses course_part_id as micro-skill id, predicts normalized next-step score (0..1) via MSE.
Updated for Thesis: 80/10/10 split and rigorous academic metrics (MSE, RMSE, MAE, R2).
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


# ---------------------------------------------------------------------------
# 1. Data loading & preprocessing
# ---------------------------------------------------------------------------


def load_and_preprocess(
    csv_path: Path,
    train_frac: float = 0.8,
    val_frac: float = 0.1,
    seed: int = 42,
) -> tuple[pd.DataFrame, dict, dict]:
    """
    Load CSV, sort by user and time, build skill index map and train/val/test indices for splitting.

    Returns:
        df: processed frame with columns skill_idx, score_norm, time_norm, split
        meta: dict with num_skills, skill_to_idx, idx_to_skill, time_min, time_max (train)
    """
    df = pd.read_csv(csv_path)
    required = [
        "user_id",
        "course_part_id",
        "score",
        "respones_time_ms",
        "question_started_at",
    ]
    for c in required:
        if c not in df.columns:
            raise ValueError(f"Missing column: {c}. Found: {list(df.columns)}")

    df = df.sort_values(["user_id", "question_started_at"], kind="mergesort").reset_index(
        drop=True
    )

    # Contiguous skill indices for Embedding (0 .. num_skills - 1)
    unique_skills = sorted(df["course_part_id"].unique())
    skill_to_idx = {s: i for i, s in enumerate(unique_skills)}
    df["skill_idx"] = df["course_part_id"].map(skill_to_idx).astype("int64")

    # Normalized score in [0, 1]
    df["score_norm"] = df["score"].astype("float32") / 100.0
    df["score_norm"] = df["score_norm"].clip(0.0, 1.0)

    # -----------------------------------------------------------
    # Split Users into Train (80%), Val (10%), Test (10%)
    # -----------------------------------------------------------
    user_ids = list(df["user_id"].unique())
    random.Random(seed).shuffle(user_ids)
    
    n_users = len(user_ids)
    n_train = max(1, int(n_users * train_frac))
    n_val = max(1, int(n_users * val_frac))
    
    train_users = set(user_ids[:n_train])
    val_users = set(user_ids[n_train : n_train + n_val])
    test_users = set(user_ids[n_train + n_val :])

    def assign_split(uid):
        if uid in train_users: return "train"
        if uid in val_users: return "val"
        return "test"

    df["split"] = df["user_id"].map(assign_split)

    # Time: ms -> seconds (fit Min-Max on TRAIN users only, then apply to all)
    time_sec = df["respones_time_ms"].astype("float32") / 1000.0
    train_mask = df["split"] == "train"
    t_min = float(time_sec[train_mask].min())
    t_max = float(time_sec[train_mask].max())
    if t_max <= t_min:
        t_max = t_min + 1e-6

    df["time_norm"] = ((time_sec - t_min) / (t_max - t_min)).clip(0.0, 1.0).astype("float32")

    meta = {
        "num_skills": len(unique_skills),
        "skill_to_idx": skill_to_idx,
        "idx_to_skill": {i: s for s, i in skill_to_idx.items()},
        "time_min": t_min,
        "time_max": t_max,
    }
    return df, meta, {"train": train_users, "val": val_users, "test": test_users}


# ---------------------------------------------------------------------------
# 2. Per-user sequences → Dataset
# ---------------------------------------------------------------------------


class UserSequenceDataset(Dataset):
    def __init__(self, df: pd.DataFrame, split: str):
        sub = df[df["split"] == split]
        self.sequences: list[torch.Tensor] = []
        for _, g in sub.groupby("user_id", sort=False):
            g = g.sort_values("question_started_at", kind="mergesort")
            seq = torch.stack(
                [
                    torch.tensor(g["skill_idx"].values, dtype=torch.long),
                    torch.tensor(g["score_norm"].values, dtype=torch.float32),
                    torch.tensor(g["time_norm"].values, dtype=torch.float32),
                ],
                dim=1,
            )  # (T, 3)
            if seq.size(0) > 0:
                self.sequences.append(seq)

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.sequences[idx]


def collate_pad(batch: list[torch.Tensor]) -> dict[str, torch.Tensor]:
    lengths = torch.tensor([b.size(0) for b in batch], dtype=torch.long)
    T_max = int(lengths.max())
    B = len(batch)
    skill_ids = torch.zeros(B, T_max, dtype=torch.long)
    score_norm = torch.zeros(B, T_max, dtype=torch.float32)
    time_norm = torch.zeros(B, T_max, dtype=torch.float32)

    for i, seq in enumerate(batch):
        L = seq.size(0)
        skill_ids[i, :L] = seq[:, 0]
        score_norm[i, :L] = seq[:, 1]
        time_norm[i, :L] = seq[:, 2]

    return {
        "skill_ids": skill_ids,
        "score_norm": score_norm,
        "time_norm": time_norm,
        "lengths": lengths,
    }


# ---------------------------------------------------------------------------
# 3. Model: LSTM + next-step continuous DKT head
# ---------------------------------------------------------------------------


class ContinuousDKTLSTM(nn.Module):
    def __init__(
        self,
        num_skills: int,
        embed_dim: int = 64,
        hidden_dim: int = 128,
        num_layers: int = 1,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.skill_emb = nn.Embedding(num_skills, embed_dim)
        input_dim = embed_dim + 2  # + score + time
        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_dim + embed_dim, 1)

    def forward(
        self,
        skill_ids: torch.Tensor,
        score_norm: torch.Tensor,
        time_norm: torch.Tensor,
        lengths: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        B, T = skill_ids.shape
        emb = self.skill_emb(skill_ids)
        x = torch.cat(
            [emb, score_norm.unsqueeze(-1), time_norm.unsqueeze(-1)],
            dim=-1,
        )

        packed = pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        out_packed, _ = self.lstm(packed)
        lstm_out, _ = pad_packed_sequence(out_packed, batch_first=True, total_length=T)

        if T < 2:
            return torch.zeros(B, 0, device=skill_ids.device), torch.zeros(
                B, 0, dtype=torch.bool, device=skill_ids.device
            )

        h_t = lstm_out[:, :-1, :]  # after step t, t = 0..T-2
        next_skill = skill_ids[:, 1:]
        emb_next = self.skill_emb(next_skill)
        cat = torch.cat([h_t, emb_next], dim=-1)
        pred = torch.sigmoid(self.fc(cat)).squeeze(-1)  # (B, T-1)

        ar = torch.arange(T - 1, device=lengths.device).unsqueeze(0).expand(B, -1)
        mask = ar < (lengths.unsqueeze(1) - 1)

        return pred, mask


def masked_mse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    diff = (pred - target) ** 2
    diff = diff * mask.float()
    denom = mask.float().sum().clamp_min(1.0)
    return diff.sum() / denom


# ---------------------------------------------------------------------------
# 4. Training & Academic Evaluation functions
# ---------------------------------------------------------------------------


def run_epoch(
    model: ContinuousDKTLSTM,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
) -> float:
    train = optimizer is not None
    model.train(train)
    total_loss = 0.0
    n_batches = 0

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

        if train:
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

        total_loss += float(loss.detach().cpu())
        n_batches += 1

    return total_loss / max(n_batches, 1)


def evaluate_thesis_metrics(model: ContinuousDKTLSTM, loader: DataLoader, device: torch.device):
    """
    Computes rigorous academic metrics (MSE, RMSE, MAE, R-squared) 
    using sklearn for accurate reporting in the thesis.
    """
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in loader:
            skill_ids = batch["skill_ids"].to(device)
            score_norm = batch["score_norm"].to(device)
            time_norm = batch["time_norm"].to(device)
            lengths = batch["lengths"].to(device)

            pred, mask = model(skill_ids, score_norm, time_norm, lengths)
            if pred.numel() == 0:
                continue

            target = score_norm[:, 1:]
            
            # Extract only valid (masked) predictions and targets
            valid_preds = pred[mask].cpu().numpy()
            valid_targets = target[mask].cpu().numpy()

            all_preds.extend(valid_preds)
            all_targets.extend(valid_targets)

    if not all_preds:
        return 0.0, 0.0, 0.0, 0.0

    mse = mean_squared_error(all_targets, all_preds)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(all_targets, all_preds)
    r2 = r2_score(all_targets, all_preds)

    return mse, rmse, mae, r2


def main():
    parser = argparse.ArgumentParser(description="Train continuous DKT (LSTM) on interaction CSV")
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path(__file__).resolve().parent / "dkt_synthetic_interactions.csv",
        help="Path to interactions CSV",
    )
    parser.add_argument("--epochs", type=int, default=50) 
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--embed-dim", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # 1. Load Data with 80-10-10 Split
    df, meta, splits = load_and_preprocess(args.csv, train_frac=0.8, val_frac=0.1, seed=args.seed)
    
    n_train = len(splits['train'])
    n_val = len(splits['val'])
    n_test = len(splits['test'])
    print(f"\n[Data Split] Train: {n_train} | Val: {n_val} | Test: {n_test} users")
    print(f"Total Rows: {len(df)} | Unique Skills: {meta['num_skills']}\n")

    # 2. Build Datasets
    train_ds = UserSequenceDataset(df, "train")
    val_ds = UserSequenceDataset(df, "val")
    test_ds = UserSequenceDataset(df, "test")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_pad)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_pad)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_pad)

    # 3. Init Model
    model = ContinuousDKTLSTM(
        num_skills=meta["num_skills"],
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
    patience = 5  
    patience_counter = 0
    model_save_path = "deepbac_dkt_model.pth"

    # 4. Training Loop (Train / Val)
    for epoch in range(1, args.epochs + 1):
        tr_loss = run_epoch(model, train_loader, optimizer, device)
        val_loss = run_epoch(model, val_loader, None, device)
        scheduler.step(val_loss)
        
        print(f"Epoch {epoch:03d}  train MSE: {tr_loss:.5f}  val MSE: {val_loss:.5f}", end="")

        if val_loss < best_val:
            best_val = val_loss
            patience_counter = 0 
            torch.save(model.state_dict(), model_save_path)
            print("  --> Model improved and saved!")
        else:
            patience_counter += 1
            print(f"  --> No improvement. Patience: {patience_counter}/{patience}")
            
        if patience_counter >= patience:
            print(f"\n[!] Early Stopping triggered at Epoch {epoch}! The model stopped improving.")
            break

    print(f"\nTraining finished. Best validation MSE: {best_val:.5f}")
    print(f"The best weights are securely saved in: {model_save_path}")

    # ==========================================
    # --- (final Evaluation) ---
    # ==========================================
    print("\n" + "="*60)
    print("FINAL MODEL EVALUATION REPORT (TEST SET)")
    print("="*60)
    
    # تحميل أفضل أوزان للنموذج
    model.load_state_dict(torch.load(model_save_path))
    
    # حساب المقاييس العلمية الدقيقة على مجموعة الاختبار (Test Set)
    test_mse, test_rmse, test_mae, test_r2 = evaluate_thesis_metrics(model, test_loader, device)
    
    print(f"Dataset Split (Users) : Train ({n_train}) | Val ({n_val}) | Test ({n_test})")
    print(f"Mean Squared Error (MSE)       : {test_mse:.5f}")
    print(f"Root Mean Squared Error (RMSE) : {test_rmse:.5f}")
    print(f"Mean Absolute Error (MAE)      : {test_mae:.5f}")
    print(f"R-squared (R²) Score           : {test_r2:.5f}")
    print("\nThesis Interpretations:")
    print(f"- The model's predictions deviate by an average of ±{(test_mae * 100):.2f}% from the actual student scores.")
    print(f"- An R² of {test_r2:.4f} indicates how well the model explains the variance in student performance.")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()