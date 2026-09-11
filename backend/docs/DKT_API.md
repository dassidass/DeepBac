# DKT API reference

Base path `/api/dkt`. Every endpoint requires `Authorization: Bearer <jwt>`,
signed with the same `JWT_SECRET` the RAG service uses, so one token works
against both. `AUTH_DISABLED=1` bypasses this for local work only.

A **history** is a list of interactions in chronological order, oldest first.
The order is the signal: reversing it makes the model read the learner's
progress backwards.

```json
{ "course_part_id": 149, "score": 70, "response_time_ms": 8400 }
```

`score` is on the scale the model was trained with, reported as `score_scale` in
every response.

---

## POST /api/dkt/predict

Predicted score for each named section, given the history.

```bash
curl -X POST http://localhost:5002/api/dkt/predict \
  -H 'Content-Type: application/json' -H "Authorization: Bearer $TOKEN" \
  -d '{
        "history": [
          {"course_part_id": 34, "score": 40, "response_time_ms": 9000},
          {"course_part_id": 21, "score": 100, "response_time_ms": 4000}
        ],
        "candidate_part_ids": [34, 21, 999999]
      }'
```

```json
{
  "predictions": [
    { "course_part_id": 34, "predicted_score": 54.36, "predicted_normalized": 0.5436, "known": true },
    { "course_part_id": 21, "predicted_score": 55.05, "predicted_normalized": 0.5505, "known": true },
    { "course_part_id": 999999, "predicted_score": 65.56, "predicted_normalized": 0.6556, "known": false }
  ],
  "score_scale": 100.0,
  "history_length": 2
}
```

`known: false` marks a section absent from the training vocabulary. All such
sections share one embedding and therefore one prediction: the model's prior for
a learner in this state, carrying no item-specific information. Label or
suppress those in the interface rather than presenting them as measurements.

---

## POST /api/dkt/weakest

Sections to revise first, weakest predicted performance first.

| Field | Default | Description |
|---|---|---|
| `history` | required | The learner's interactions |
| `candidate_part_ids` | null | Restrict to these sections. When omitted, uses every section in the history |
| `top_n` | 5 | How many to return |
| `min_attempts` | 1 | Skip sections attempted fewer times than this |

```json
{
  "weakest": [
    { "course_part_id": 34, "predicted_score": 62.38, "attempts": 2, "known": true },
    { "course_part_id": 21, "predicted_score": 63.03, "attempts": 2, "known": true }
  ],
  "score_scale": 100.0
}
```

Raise `min_attempts` for anything the student will act on. Recommending a
section attempted once is noise: a single answer is as likely to reflect a
misread question as a knowledge gap.

---

## POST /api/dkt/mastery

Observed versus predicted performance for every section in the history.

```json
{
  "interactions": 200,
  "distinct_parts": 20,
  "overall_predicted_mean": 80.05,
  "score_scale": 100.0,
  "parts": [
    { "course_part_id": 34, "attempts": 13, "observed_mean": 84.62, "predicted_score": 73.64, "known": true },
    { "course_part_id": 21, "attempts": 12, "observed_mean": 69.58, "predicted_score": 74.18, "known": true }
  ]
}
```

The gap between the two columns is the point. A section with a high
`observed_mean` and a low `predicted_score` is one the learner is losing — an
average over past answers can never show that, because it weights an answer from
September like one from this morning.

---

## GET /api/dkt/history/{user_id}

Reads one learner's interactions from the platform database, for clients that
would otherwise assemble the history themselves. Query parameter `limit`
defaults to 500.

Requires `DB_*` credentials. Returns `503` without them; the inline endpoints
above are unaffected.

---

## GET /api/health

Unauthenticated. Each dependency reported separately.

```json
{
  "status": "OK",
  "model": {
    "checkpoint": "checkpoints/dkt_lstm_paper.pt",
    "loaded": true,
    "skills": 178,
    "score_scale": 100.0,
    "hidden_dim": 128
  },
  "database": "not configured",
  "auth": "bearer"
}
```

`loaded` reports whether the checkpoint is in memory, and reads `false` until
the first prediction request: the model loads lazily so that a missing file
produces a readable health response instead of a process that exits at startup.
`status` is `DEGRADED` when no checkpoint file exists; the `model.error` field
then says so.

---

## Status codes

| Code | Condition |
|---|---|
| 200 | Success |
| 401 | Token missing or invalid |
| 422 | Malformed body — most often an empty `history` |
| 503 | No checkpoint, a checkpoint that will not load, `JWT_SECRET` unset, or the database unconfigured for the history endpoint |

---

## Feeding a prediction back into an explanation

`predicted_normalized` is exactly what `POST /api/rag/ask` accepts as `mastery`.
Sending it steers the explanation to one of three difficulty tiers — scaffolded
below 0.50, examination-level to 0.80, stretch above it — at no extra model
call, because it modifies a prompt the RAG service was assembling anyway.

```bash
# weakest → { "course_part_id": 149, "predicted_normalized": 0.34 }
curl -X POST http://localhost:5001/api/rag/ask \
  -H 'Content-Type: application/json' -H "Authorization: Bearer $TOKEN" \
  -d '{"question":"اشرح لي مظاهر الصراع بين المعسكرين","mastery":0.34}'
```

The `predicted_score` on the 0-100 scale works equally well; the RAG service
accepts either and rejects neither silently. Omitting the field is always valid
and yields an unadapted answer. See [API.md](API.md) §Composing with the
knowledge tracer and [RAG_PIPELINE.md](RAG_PIPELINE.md) §6a.
