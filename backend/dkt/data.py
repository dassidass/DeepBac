"""
Interaction loading, preprocessing and batching.

One *interaction* is one graded answer: who answered, which micro-skill it
belonged to, what score they earned, and how long they took. The micro-skill is
``course_part_id`` — the same authored lesson section the RAG backend retrieves
over, which is what lets a mastery estimate point at a specific section a
student should reread.

The pipeline is: load → index skills → normalise → split by user → group into
per-learner chronological sequences → pad into batches.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

# The production CSV ships a misspelled column. It is accepted verbatim rather
# than corrected, because renaming it would break every previously exported
# file; the canonical spelling is accepted too.
RESPONSE_TIME_COLUMNS = ("response_time_ms", "respones_time_ms")
TIME_COLUMNS = ("question_started_at", "created_at", "answered_at")

# Index reserved for a skill never seen during training. Inference meets these
# constantly: a student opens a lesson authored after the last training run.
UNK_SKILL = "<unk>"


@dataclass
class Vocabulary:
    """Everything inference needs that is not a model weight.

    Saved inside the checkpoint. A checkpoint without it is unusable: the skill
    ids would be re-derived in a different order and every embedding lookup
    would silently address the wrong skill.
    """

    skill_to_idx: dict[int, int]
    time_min: float
    time_max: float
    score_scale: float

    @property
    def num_skills(self) -> int:
        """Includes the trailing unknown-skill slot."""
        return len(self.skill_to_idx) + 1

    @property
    def unk_idx(self) -> int:
        return len(self.skill_to_idx)

    def encode_skill(self, skill_id: int) -> int:
        return self.skill_to_idx.get(int(skill_id), self.unk_idx)

    def normalize_time(self, response_time_ms: float) -> float:
        """Min-max using the *training* range, clipped — the same transform the
        model was fitted under. Re-deriving it from live data would shift the
        input distribution under a model that cannot know it moved."""
        seconds = float(response_time_ms) / 1000.0
        span = max(self.time_max - self.time_min, 1e-6)
        return min(max((seconds - self.time_min) / span, 0.0), 1.0)

    def normalize_score(self, score: float) -> float:
        return min(max(float(score) / self.score_scale, 0.0), 1.0)

    def to_dict(self) -> dict:
        # JSON/torch round-trips turn integer keys into strings; converting back
        # is done in from_dict so callers never see the difference.
        return {
            "skill_to_idx": {str(k): v for k, v in self.skill_to_idx.items()},
            "time_min": self.time_min,
            "time_max": self.time_max,
            "score_scale": self.score_scale,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Vocabulary":
        return cls(
            skill_to_idx={int(k): int(v) for k, v in d["skill_to_idx"].items()},
            time_min=float(d["time_min"]),
            time_max=float(d["time_max"]),
            score_scale=float(d["score_scale"]),
        )


def _resolve_column(df: pd.DataFrame, candidates: tuple[str, ...], label: str) -> str:
    for name in candidates:
        if name in df.columns:
            return name
    raise ValueError(f"No {label} column found. Tried {candidates}; got {list(df.columns)}")


def load_interactions(csv_path: Path) -> pd.DataFrame:
    """Read a CSV export and normalise its column names."""
    df = pd.read_csv(csv_path)

    for required in ("user_id", "course_part_id", "score"):
        if required not in df.columns:
            raise ValueError(f"Missing column: {required}. Found: {list(df.columns)}")

    rt_col = _resolve_column(df, RESPONSE_TIME_COLUMNS, "response-time")
    ts_col = _resolve_column(df, TIME_COLUMNS, "timestamp")

    df = df.rename(columns={rt_col: "response_time_ms", ts_col: "question_started_at"})
    df = df.dropna(subset=["user_id", "course_part_id", "score"])
    df["response_time_ms"] = df["response_time_ms"].fillna(0)
    return df


def preprocess(
    df: pd.DataFrame,
    *,
    score_scale: float = 100.0,
    train_user_frac: float = 0.8,
    val_user_frac: float = 0.1,
    min_seq_len: int = 2,
    seed: int = 42,
) -> tuple[pd.DataFrame, Vocabulary]:
    """Index skills, normalise targets and inputs, and split by user.

    Returns the frame with ``skill_idx``, ``score_norm``, ``time_norm`` and
    ``split`` columns, plus the vocabulary needed at inference time.

    The split is three-way and by learner. Validation chooses the epoch and the
    hyper-parameters; test is read once, when a number is reported. Keeping them
    separate is what stops early stopping from quietly selecting the checkpoint
    that happens to suit the set the result will be quoted from. Setting
    ``val_user_frac`` so that the two fractions sum to 1 leaves no test split,
    and the evaluation script then says so rather than scoring an empty set.
    """
    # mergesort is stable, so interactions recorded within the same second keep
    # their insertion order instead of being shuffled by the sort.
    df = df.sort_values(["user_id", "question_started_at"], kind="mergesort").reset_index(drop=True)

    # Drop learners too short to produce a next-step target.
    counts = df.groupby("user_id")["score"].transform("size")
    df = df[counts >= min_seq_len].reset_index(drop=True)
    if df.empty:
        raise ValueError(f"No learner has at least {min_seq_len} interactions")

    observed_max = float(df["score"].max())
    if observed_max > score_scale:
        raise ValueError(
            f"score {observed_max} exceeds the configured scale {score_scale}; "
            "set DKT_SCORE_SCALE to the real maximum"
        )
    if score_scale >= 100 and observed_max <= 10:
        # Not fatal — a cohort can genuinely score badly — but training on
        # targets squeezed into [0, 0.1] produces a model that looks converged
        # and predicts nothing, so it is worth saying out loud.
        print(
            f"[data] warning: max score is {observed_max} but DKT_SCORE_SCALE={score_scale}. "
            "If the grader emits 0-10, set DKT_SCORE_SCALE=10."
        )

    unique_skills = sorted(int(s) for s in df["course_part_id"].unique())
    skill_to_idx = {s: i for i, s in enumerate(unique_skills)}
    df["skill_idx"] = df["course_part_id"].astype(int).map(skill_to_idx).astype("int64")

    df["score_norm"] = (df["score"].astype("float32") / score_scale).clip(0.0, 1.0)

    # Split by user, not by row: a learner appears entirely in one split.
    user_ids = list(df["user_id"].unique())
    random.Random(seed).shuffle(user_ids)
    n_users = len(user_ids)
    n_train = max(1, int(n_users * train_user_frac))
    n_val = max(1, int(n_users * val_user_frac)) if n_users > n_train else 0
    n_val = min(n_val, max(0, n_users - n_train))

    train_users = set(user_ids[:n_train])
    val_users = set(user_ids[n_train : n_train + n_val])

    def _assign(user_id) -> str:
        if user_id in train_users:
            return "train"
        if user_id in val_users:
            return "val"
        return "test"

    df["split"] = df["user_id"].map(_assign)

    # Fit the time scaler on training users only. Fitting on everything leaks
    # the validation distribution into the model's input normalisation.
    time_sec = df["response_time_ms"].astype("float32") / 1000.0
    train_mask = df["split"] == "train"
    t_min = float(time_sec[train_mask].min())
    t_max = float(time_sec[train_mask].max())
    if t_max <= t_min:
        t_max = t_min + 1e-6
    df["time_norm"] = ((time_sec - t_min) / (t_max - t_min)).clip(0.0, 1.0).astype("float32")

    vocab = Vocabulary(
        skill_to_idx=skill_to_idx,
        time_min=t_min,
        time_max=t_max,
        score_scale=float(score_scale),
    )
    return df, vocab


class UserSequenceDataset(Dataset):
    """One sample is one learner's chronological interaction sequence.

    Learners with more than ``max_seq_len`` interactions are cut into
    consecutive windows rather than truncated. Truncating would throw away the
    most recent interactions, which are the ones a mastery estimate depends on.
    """

    def __init__(self, df: pd.DataFrame, split: str, max_seq_len: int = 200):
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

            for start in range(0, seq.size(0), max_seq_len):
                window = seq[start : start + max_seq_len]
                # A window of length 1 yields no (t, t+1) pair.
                if window.size(0) >= 2:
                    self.sequences.append(window)

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.sequences[idx]


def collate_pad(batch: list[torch.Tensor]) -> dict[str, torch.Tensor]:
    """Right-pad to the longest sequence in the batch.

    Padded positions hold zeros, which is a *valid* skill index. They are never
    scored: the length vector drives both ``pack_padded_sequence`` and the loss
    mask, so padding influences neither the hidden state nor the gradient.
    """
    lengths = torch.tensor([b.size(0) for b in batch], dtype=torch.long)
    t_max = int(lengths.max())
    b = len(batch)

    skill_ids = torch.zeros(b, t_max, dtype=torch.long)
    score_norm = torch.zeros(b, t_max, dtype=torch.float32)
    time_norm = torch.zeros(b, t_max, dtype=torch.float32)

    for i, seq in enumerate(batch):
        length = seq.size(0)
        skill_ids[i, :length] = seq[:, 0]
        score_norm[i, :length] = seq[:, 1]
        time_norm[i, :length] = seq[:, 2]

    return {
        "skill_ids": skill_ids,
        "score_norm": score_norm,
        "time_norm": time_norm,
        "lengths": lengths,
    }


def build_loader(
    df: pd.DataFrame,
    split: str,
    *,
    batch_size: int = 32,
    max_seq_len: int = 200,
    shuffle: bool = False,
) -> DataLoader:
    """One loader for one split. Shuffling belongs to training only."""
    dataset = UserSequenceDataset(df, split, max_seq_len=max_seq_len)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_pad)


def build_loaders(
    df: pd.DataFrame,
    *,
    batch_size: int = 32,
    max_seq_len: int = 200,
) -> tuple[DataLoader, DataLoader]:
    """Training and validation loaders. The test split is built on demand, by
    :func:`build_loader`, so it cannot be consumed by accident during training."""
    train_loader = build_loader(
        df, "train", batch_size=batch_size, max_seq_len=max_seq_len, shuffle=True
    )
    val_loader = build_loader(df, "val", batch_size=batch_size, max_seq_len=max_seq_len)
    return train_loader, val_loader
