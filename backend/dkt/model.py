"""
The model: an LSTM knowledge tracer with a continuous next-step head.

Classical Deep Knowledge Tracing (Piech et al., 2015) predicts the probability
that the next answer is *correct* — a Bernoulli target trained with binary
cross-entropy. This variant predicts the next *score* in [0, 1] and trains with
mean squared error, because the platform's grader returns a graded score rather
than a binary verdict. Collapsing a 0-100 score to right/wrong before training
would discard exactly the signal that distinguishes a near-miss from a blank
answer, which is the distinction a revision recommendation rests on.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class ContinuousDKTLSTM(nn.Module):
    """LSTM over a learner's history, with a next-step regression head.

    At step ``t`` the input is ``[embed(skill_t), score_t, time_t]``. The LSTM
    carries the learner's state forward as ``h_t``. The prediction for step
    ``t+1`` is conditioned on ``h_t`` *and* on the embedding of the skill about
    to be attempted:

        score_hat(t+1) = sigmoid(W · [h_t ; embed(skill_{t+1})])

    Conditioning on the next skill is what makes the model useful as a backend
    rather than only as a benchmark number. It answers "how will this learner do
    on *this* section", which is a question the platform can act on, instead of
    "how will they do next", which it cannot.

    Response time is an input feature, not a target. A fast wrong answer and a
    slow wrong answer are different evidence about mastery, and the LSTM is free
    to use that difference.
    """

    def __init__(
        self,
        num_skills: int,
        embed_dim: int = 64,
        hidden_dim: int = 128,
        num_layers: int = 1,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.num_skills = num_skills
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim

        self.skill_emb = nn.Embedding(num_skills, embed_dim)
        input_dim = embed_dim + 2  # skill embedding + score + response time

        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            # PyTorch ignores dropout on a single layer and warns; guard it.
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_dim + embed_dim, 1)

    def encode(
        self,
        skill_ids: torch.Tensor,
        score_norm: torch.Tensor,
        time_norm: torch.Tensor,
        lengths: torch.Tensor,
    ) -> torch.Tensor:
        """Run the LSTM and return per-step hidden states, shape (B, T, H).

        Packing is what keeps padding out of the recurrence: without it the
        LSTM would consume the zero-padded tail and the final hidden state would
        depend on how long the *longest* sequence in the batch happened to be.
        """
        emb = self.skill_emb(skill_ids)
        x = torch.cat([emb, score_norm.unsqueeze(-1), time_norm.unsqueeze(-1)], dim=-1)

        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        out_packed, _ = self.lstm(packed)
        lstm_out, _ = pad_packed_sequence(
            out_packed, batch_first=True, total_length=skill_ids.size(1)
        )
        return lstm_out

    def forward(
        self,
        skill_ids: torch.Tensor,
        score_norm: torch.Tensor,
        time_norm: torch.Tensor,
        lengths: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Next-step predictions and the mask marking which of them are real.

        Returns ``pred`` and ``mask``, both (B, T-1). Position ``t`` predicts the
        score at ``t+1``.
        """
        b, t = skill_ids.shape

        if t < 2:
            empty = torch.zeros(b, 0, device=skill_ids.device)
            return empty, empty.bool()

        lstm_out = self.encode(skill_ids, score_norm, time_norm, lengths)

        h_t = lstm_out[:, :-1, :]  # state after steps 0 .. T-2
        emb_next = self.skill_emb(skill_ids[:, 1:])  # skills at steps 1 .. T-1
        pred = torch.sigmoid(self.fc(torch.cat([h_t, emb_next], dim=-1))).squeeze(-1)

        # Position t is scored only when both t and t+1 are real interactions,
        # i.e. t < length - 1.
        ar = torch.arange(t - 1, device=lengths.device).unsqueeze(0).expand(b, -1)
        mask = ar < (lengths.unsqueeze(1) - 1)

        return pred, mask

    @torch.no_grad()
    def predict_next(
        self,
        skill_ids: torch.Tensor,
        score_norm: torch.Tensor,
        time_norm: torch.Tensor,
        lengths: torch.Tensor,
        candidate_skills: torch.Tensor,
    ) -> torch.Tensor:
        """Score a set of candidate skills against the learner's current state.

        ``candidate_skills`` is (B, K). Returns (B, K) predicted scores in
        [0, 1]. This is the inference path: take the final hidden state and ask
        it about every section the platform might recommend next.
        """
        self.eval()
        lstm_out = self.encode(skill_ids, score_norm, time_norm, lengths)

        # Gather the last *real* state per sequence, not lstm_out[:, -1].
        idx = (lengths - 1).clamp(min=0).view(-1, 1, 1).expand(-1, 1, lstm_out.size(-1))
        h_last = lstm_out.gather(1, idx).squeeze(1)  # (B, H)

        k = candidate_skills.size(1)
        h_rep = h_last.unsqueeze(1).expand(-1, k, -1)  # (B, K, H)
        emb_cand = self.skill_emb(candidate_skills)  # (B, K, E)
        return torch.sigmoid(self.fc(torch.cat([h_rep, emb_cand], dim=-1))).squeeze(-1)


def masked_mse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean squared error over masked positions only.

    Normalising by the number of *real* positions rather than by the tensor size
    keeps the loss comparable across batches with different padding, which
    otherwise makes a batch of short sequences look artificially good.
    """
    diff = (pred - target) ** 2 * mask.float()
    denom = mask.float().sum().clamp_min(1.0)
    return diff.sum() / denom


def masked_mae(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean absolute error over masked positions; reported, never optimised.

    MAE is on the same scale as the score, so it reads directly: 0.15 means the
    typical prediction is 15 score-points off.
    """
    diff = (pred - target).abs() * mask.float()
    denom = mask.float().sum().clamp_min(1.0)
    return diff.sum() / denom
