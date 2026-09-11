"""
Serving-time logic: load a checkpoint, turn a learner's history into a state,
and answer questions about it.

Three questions the platform asks, in increasing order of usefulness:

1. *How will this learner do on section X next?* — a single prediction.
2. *Which sections are weakest right now?* — the same prediction over every
   section the learner has touched, sorted ascending.
3. *What should they revise next?* — the weakest sections, filtered to those
   with enough evidence to be worth acting on.

All three read from the same final hidden state, so they are one forward pass.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from dkt.data import Vocabulary
from dkt.model import ContinuousDKTLSTM


@dataclass
class Interaction:
    """One graded answer in a learner's history."""

    course_part_id: int
    score: float
    response_time_ms: float = 0.0


def load_checkpoint(
    path: Path, device: torch.device | None = None
) -> tuple[ContinuousDKTLSTM, Vocabulary]:
    """Rebuild a model and its vocabulary from disk.

    ``weights_only=False`` is required because the checkpoint carries the
    vocabulary dict alongside the tensors. The file is produced by this project
    and read from local disk; never point this at an untrusted checkpoint.
    """
    device = device or torch.device("cpu")
    blob = torch.load(path, map_location=device, weights_only=False)

    cfg = blob["model_config"]
    model = ContinuousDKTLSTM(
        num_skills=cfg["num_skills"],
        embed_dim=cfg["embed_dim"],
        hidden_dim=cfg["hidden_dim"],
        num_layers=cfg.get("num_layers", 1),
        dropout=cfg.get("dropout", 0.0),
    )
    model.load_state_dict(blob["state_dict"])
    model.to(device)
    model.eval()

    return model, Vocabulary.from_dict(blob["vocab"])


class KnowledgeTracer:
    """Stateless wrapper around a checkpoint.

    Holds no per-learner state: a request carries its own history. That keeps
    the service horizontally scalable and means a learner's newest answer is
    reflected immediately, with no cache to invalidate.
    """

    def __init__(self, checkpoint_path: Path, device: torch.device | None = None):
        self.device = device or torch.device("cpu")
        self.model, self.vocab = load_checkpoint(checkpoint_path, self.device)
        self.checkpoint_path = checkpoint_path

    def _encode_history(
        self, history: list[Interaction]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """History (chronological, oldest first) → padded batch of size 1."""
        if not history:
            raise ValueError("History is empty; the model has nothing to condition on")

        skills = torch.tensor(
            [[self.vocab.encode_skill(i.course_part_id) for i in history]], dtype=torch.long
        )
        scores = torch.tensor(
            [[self.vocab.normalize_score(i.score) for i in history]], dtype=torch.float32
        )
        times = torch.tensor(
            [[self.vocab.normalize_time(i.response_time_ms) for i in history]],
            dtype=torch.float32,
        )
        lengths = torch.tensor([len(history)], dtype=torch.long)

        return (
            skills.to(self.device),
            scores.to(self.device),
            times.to(self.device),
            lengths.to(self.device),
        )

    def predict(
        self, history: list[Interaction], candidate_part_ids: list[int]
    ) -> dict[int, float]:
        """Predicted score in [0, 1] for each candidate section.

        Sections absent from the training vocabulary map to the unknown slot, so
        they all receive the same prediction — the model's prior for a learner
        in this state, with no item-specific information. That is the honest
        answer for a lesson written after the last training run, and it is
        flagged by ``known`` in :meth:`weakest_parts`.
        """
        if not candidate_part_ids:
            return {}

        skills, scores, times, lengths = self._encode_history(history)
        candidates = torch.tensor(
            [[self.vocab.encode_skill(p) for p in candidate_part_ids]], dtype=torch.long
        ).to(self.device)

        preds = self.model.predict_next(skills, scores, times, lengths, candidates)
        return {pid: float(preds[0, i]) for i, pid in enumerate(candidate_part_ids)}

    def weakest_parts(
        self,
        history: list[Interaction],
        candidate_part_ids: list[int] | None = None,
        top_n: int = 5,
        min_attempts: int = 1,
    ) -> list[dict]:
        """Sections ranked from weakest predicted performance upward.

        ``min_attempts`` filters out sections the learner has barely touched.
        Recommending revision of a section attempted once is noise: a single
        answer is as likely to reflect a misread question as a knowledge gap.
        """
        if candidate_part_ids is None:
            attempts: dict[int, int] = {}
            for interaction in history:
                attempts[interaction.course_part_id] = attempts.get(interaction.course_part_id, 0) + 1
            candidate_part_ids = [p for p, n in attempts.items() if n >= min_attempts]
        else:
            attempts = {}
            for interaction in history:
                attempts[interaction.course_part_id] = attempts.get(interaction.course_part_id, 0) + 1

        if not candidate_part_ids:
            return []

        predictions = self.predict(history, candidate_part_ids)
        rows = [
            {
                "course_part_id": pid,
                "predicted_score": round(score * self.vocab.score_scale, 2),
                "predicted_normalized": round(score, 4),
                "attempts": attempts.get(pid, 0),
                "known": pid in self.vocab.skill_to_idx,
            }
            for pid, score in predictions.items()
        ]
        rows.sort(key=lambda r: r["predicted_normalized"])
        return rows[:top_n]

    def mastery_profile(self, history: list[Interaction]) -> dict:
        """A compact summary for a dashboard.

        ``observed`` is what the learner actually scored; ``predicted`` is what
        the model expects next. The gap between them is the interesting part: a
        section with a high observed average and a low prediction is one the
        learner is losing, which a simple average over past answers can never
        show.
        """
        if not history:
            return {"interactions": 0, "parts": []}

        observed: dict[int, list[float]] = {}
        for interaction in history:
            observed.setdefault(interaction.course_part_id, []).append(interaction.score)

        part_ids = list(observed.keys())
        predictions = self.predict(history, part_ids)

        parts = [
            {
                "course_part_id": pid,
                "attempts": len(values),
                "observed_mean": round(sum(values) / len(values), 2),
                "predicted_score": round(predictions[pid] * self.vocab.score_scale, 2),
                "known": pid in self.vocab.skill_to_idx,
            }
            for pid, values in observed.items()
        ]
        parts.sort(key=lambda p: p["predicted_score"])

        return {
            "interactions": len(history),
            "distinct_parts": len(parts),
            "overall_predicted_mean": round(
                sum(p["predicted_score"] for p in parts) / len(parts), 2
            ),
            "parts": parts,
        }
