"""
HTTP service for knowledge tracing.

    uvicorn dkt.api:app --port 5002

Endpoints mirror the RAG backend's conventions — bearer authentication with the
same ``JWT_SECRET``, ``/api/health`` reporting each dependency separately — so
that after the merge one token and one health check cover both services.

    POST /api/dkt/predict    scores for named sections, given a history
    POST /api/dkt/weakest    the sections to revise first
    POST /api/dkt/mastery    full profile: observed versus predicted
    GET  /api/dkt/history/{user_id}   history read from MySQL, if configured
    GET  /api/health

A history may be sent inline or, when MySQL is configured, read from the
platform database by user id. Inline is the default because it keeps the
service deployable next to a database it has no credentials for.
"""

from __future__ import annotations

from pathlib import Path

import torch
from fastapi import Depends, FastAPI, HTTPException, Header
from pydantic import BaseModel, Field

from dkt import config
from dkt.inference import Interaction, KnowledgeTracer

app = FastAPI(
    title="DeepBac DKT service",
    description="LSTM knowledge tracing over course-part interactions",
    version="1.0.0",
)

# Loaded lazily so the process starts even with no checkpoint on disk. The
# health endpoint then reports the model as unavailable instead of the container
# crash-looping before anyone can read the error.
_tracer: KnowledgeTracer | None = None
_load_error: str | None = None


def get_tracer() -> KnowledgeTracer:
    global _tracer, _load_error
    if _tracer is None:
        path = Path(config.TRAIN.checkpoint_path)
        if not path.exists():
            _load_error = f"checkpoint not found: {path}"
            raise HTTPException(status_code=503, detail=f"Model unavailable — {_load_error}")
        try:
            _tracer = KnowledgeTracer(path, torch.device("cpu"))
            _load_error = None
        except Exception as exc:  # pragma: no cover
            _load_error = str(exc)
            raise HTTPException(status_code=503, detail=f"Model unavailable — {exc}") from exc
    return _tracer


def require_auth(authorization: str | None = Header(default=None)) -> dict:
    """Verify the platform's bearer token.

    Predictions describe an individual learner's performance, so they are not
    public. ``AUTH_DISABLED=1`` bypasses this for local work only.
    """
    if config.SERVICE.auth_disabled:
        return {"userId": 0, "role": "anonymous"}

    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Access denied. No token provided.")

    if not config.SERVICE.jwt_secret:
        raise HTTPException(status_code=503, detail="JWT_SECRET is not configured")

    try:
        import jwt  # PyJWT

        return jwt.decode(
            authorization.removeprefix("Bearer "),
            config.SERVICE.jwt_secret,
            algorithms=["HS256"],
        )
    except ImportError as exc:  # pragma: no cover
        raise HTTPException(status_code=503, detail="PyJWT is not installed") from exc
    except Exception as exc:
        raise HTTPException(status_code=401, detail="Invalid token.") from exc


# ---------------------------------------------------------------------------
# Request and response models
# ---------------------------------------------------------------------------


class InteractionIn(BaseModel):
    course_part_id: int = Field(..., description="Lesson section that was practised")
    score: float = Field(..., description="Awarded score, on the scale the model was trained with")
    response_time_ms: float = Field(0.0, description="Time to answer, in milliseconds")


class PredictRequest(BaseModel):
    # Chronological, oldest first. The order is the signal: reversing it makes
    # the model read the learner's progress backwards.
    history: list[InteractionIn] = Field(..., min_length=1)
    candidate_part_ids: list[int] = Field(..., min_length=1)


class WeakestRequest(BaseModel):
    history: list[InteractionIn] = Field(..., min_length=1)
    candidate_part_ids: list[int] | None = None
    top_n: int = 5
    min_attempts: int = 1


class MasteryRequest(BaseModel):
    history: list[InteractionIn] = Field(..., min_length=1)


def _to_interactions(rows: list[InteractionIn]) -> list[Interaction]:
    return [
        Interaction(
            course_part_id=r.course_part_id,
            score=r.score,
            response_time_ms=r.response_time_ms,
        )
        for r in rows
    ]


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.post("/api/dkt/predict")
def predict(req: PredictRequest, _: dict = Depends(require_auth)) -> dict:
    """Predicted score for each candidate section, given the learner's history."""
    tracer = get_tracer()
    predictions = tracer.predict(_to_interactions(req.history), req.candidate_part_ids)
    scale = tracer.vocab.score_scale

    return {
        "predictions": [
            {
                "course_part_id": pid,
                "predicted_score": round(value * scale, 2),
                "predicted_normalized": round(value, 4),
                "known": pid in tracer.vocab.skill_to_idx,
            }
            for pid, value in predictions.items()
        ],
        "score_scale": scale,
        "history_length": len(req.history),
    }


@app.post("/api/dkt/weakest")
def weakest(req: WeakestRequest, _: dict = Depends(require_auth)) -> dict:
    """Sections to revise first, weakest predicted performance first."""
    tracer = get_tracer()
    rows = tracer.weakest_parts(
        _to_interactions(req.history),
        candidate_part_ids=req.candidate_part_ids,
        top_n=req.top_n,
        min_attempts=req.min_attempts,
    )
    return {"weakest": rows, "score_scale": tracer.vocab.score_scale}


@app.post("/api/dkt/mastery")
def mastery(req: MasteryRequest, _: dict = Depends(require_auth)) -> dict:
    """Observed versus predicted performance for every section touched."""
    tracer = get_tracer()
    profile = tracer.mastery_profile(_to_interactions(req.history))
    return {**profile, "score_scale": tracer.vocab.score_scale}


@app.get("/api/dkt/history/{user_id}")
def history(user_id: int, limit: int = 500, _: dict = Depends(require_auth)) -> dict:
    """Read one learner's interactions from the platform database.

    Convenience for clients that would otherwise assemble the history
    themselves. Requires MySQL credentials; without them the inline endpoints
    above still work.
    """
    if not config.DATABASE.is_configured():
        raise HTTPException(status_code=503, detail="Database not configured (set DB_* variables)")

    from dkt.scripts.export_interactions import fetch_interactions

    rows = fetch_interactions(user_id=user_id, limit=limit)
    return {"user_id": user_id, "interactions": rows, "count": len(rows)}


@app.get("/api/health")
def health() -> dict:
    """Each dependency reported separately, so a failure names the component."""
    checkpoint = Path(config.TRAIN.checkpoint_path)
    model_state: dict = {"checkpoint": str(checkpoint), "loaded": _tracer is not None}

    if _tracer is not None:
        model_state.update(
            {
                "skills": _tracer.vocab.num_skills - 1,
                "score_scale": _tracer.vocab.score_scale,
                "hidden_dim": _tracer.model.hidden_dim,
            }
        )
    elif not checkpoint.exists():
        model_state["error"] = "checkpoint not found — run python -m dkt.train"
    elif _load_error:
        model_state["error"] = _load_error

    return {
        "status": "OK" if checkpoint.exists() else "DEGRADED",
        "model": model_state,
        "database": "configured" if config.DATABASE.is_configured() else "not configured",
        "auth": "disabled" if config.SERVICE.auth_disabled else "bearer",
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=config.SERVICE.host, port=config.SERVICE.port)
