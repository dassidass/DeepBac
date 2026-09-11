"""
Configuration for the DKT service.

Everything is read from the environment so the training job, the export script
and the inference API agree on one source of truth. Defaults are chosen so the
whole pipeline runs against the bundled synthetic CSV with no configuration.

MySQL settings deliberately use the same variable names as the RAG backend
(``DB_HOST``, ``DB_USER``, ``DB_PASSWORD``, ``DB_NAME``). When the two services
are merged into one repository they read the same ``.env`` and therefore the
same database.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

try:  # Optional: the services run without it, reading the real environment.
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # pragma: no cover
    pass


ROOT = Path(__file__).resolve().parent.parent


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return float(raw)
    except ValueError:
        return default


@dataclass
class DataConfig:
    """Where interactions come from, and how their columns are named."""

    csv_path: Path = field(
        default_factory=lambda: Path(
            os.getenv("DKT_CSV", str(ROOT / "data" / "dkt_interactions.csv"))
        )
    )

    # The platform's AI grader returns 0-100 even though the column is declared
    # as "0-10" in an old schema comment. Getting this wrong silently trains the
    # model on targets clipped to 0.1, so it is configurable and validated at
    # load time rather than assumed.
    score_scale: float = field(default_factory=lambda: _env_float("DKT_SCORE_SCALE", 100.0))

    # Fraction of *users* (never of rows) held out for validation. Splitting by
    # row would leak a learner's future into their own training history, which
    # is the classic way to report a knowledge-tracing result that cannot be
    # reproduced in production.
    train_user_frac: float = field(default_factory=lambda: _env_float("DKT_TRAIN_USER_FRAC", 0.8))

    # The next fraction of users, held out for model selection. Whatever is left
    # after train and validation becomes the test split, which is read once, when
    # a result is reported — never to choose an epoch or a hyper-parameter.
    val_user_frac: float = field(default_factory=lambda: _env_float("DKT_VAL_USER_FRAC", 0.1))

    # Learners with fewer interactions than this carry no usable signal: a
    # sequence of length 1 produces zero next-step targets.
    min_seq_len: int = field(default_factory=lambda: _env_int("DKT_MIN_SEQ_LEN", 2))

    # Long tails dominate padded batches. Sequences longer than this are split
    # into consecutive windows rather than truncated, so no interaction is lost.
    max_seq_len: int = field(default_factory=lambda: _env_int("DKT_MAX_SEQ_LEN", 200))

    seed: int = field(default_factory=lambda: _env_int("DKT_SEED", 42))


@dataclass
class ModelConfig:
    embed_dim: int = field(default_factory=lambda: _env_int("DKT_EMBED_DIM", 64))
    hidden_dim: int = field(default_factory=lambda: _env_int("DKT_HIDDEN_DIM", 128))
    num_layers: int = field(default_factory=lambda: _env_int("DKT_LAYERS", 1))
    dropout: float = field(default_factory=lambda: _env_float("DKT_DROPOUT", 0.0))


@dataclass
class TrainConfig:
    epochs: int = field(default_factory=lambda: _env_int("DKT_EPOCHS", 20))
    batch_size: int = field(default_factory=lambda: _env_int("DKT_BATCH_SIZE", 32))
    lr: float = field(default_factory=lambda: _env_float("DKT_LR", 1e-3))
    grad_clip: float = field(default_factory=lambda: _env_float("DKT_GRAD_CLIP", 5.0))
    # Stop when validation stops improving. Knowledge-tracing models on small
    # cohorts overfit within a handful of epochs.
    patience: int = field(default_factory=lambda: _env_int("DKT_PATIENCE", 5))
    checkpoint_path: Path = field(
        default_factory=lambda: Path(
            os.getenv("DKT_CHECKPOINT", str(ROOT / "checkpoints" / "dkt_lstm_paper.pt"))
        )
    )


@dataclass
class ServiceConfig:
    host: str = field(default_factory=lambda: os.getenv("DKT_API_HOST", "0.0.0.0"))
    port: int = field(default_factory=lambda: _env_int("DKT_API_PORT", 5002))
    # Shared with the RAG backend when the two are merged, so one token works
    # against both services.
    jwt_secret: str = field(default_factory=lambda: os.getenv("JWT_SECRET", ""))
    auth_disabled: bool = field(
        default_factory=lambda: os.getenv("AUTH_DISABLED", "").strip().lower()
        in {"1", "true", "yes", "on"}
    )


@dataclass
class DatabaseConfig:
    host: str = field(default_factory=lambda: os.getenv("DB_HOST", "127.0.0.1"))
    port: int = field(default_factory=lambda: _env_int("DB_PORT", 3306))
    user: str = field(default_factory=lambda: os.getenv("DB_USER", ""))
    password: str = field(default_factory=lambda: os.getenv("DB_PASSWORD", ""))
    database: str = field(default_factory=lambda: os.getenv("DB_NAME", ""))

    def is_configured(self) -> bool:
        return bool(self.user and self.database)


DATA = DataConfig()
MODEL = ModelConfig()
TRAIN = TrainConfig()
SERVICE = ServiceConfig()
DATABASE = DatabaseConfig()
