"""
Export real learner interactions from the platform database to a training CSV.

    python -m dkt.scripts.export_interactions --out data/interactions.csv

Two tables record graded answers, and both are used:

* ``exercise_qa``     — answers to exercises attached to a lesson part.
* ``smart_review_qa`` — answers produced during an AI-generated review session.

They are unioned because a knowledge tracer needs the learner's whole timeline.
Training on exercises alone would show a student who practises mostly through
review sessions as barely active, and predict accordingly.

Rows without a ``course_part_id``, a score, or a timestamp are dropped: the
model cannot place an interaction with no skill, has no target without a score,
and cannot order it without a time.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from dkt import config

# ``question_started_at`` is when the learner *saw* the question, which is the
# correct ordering key; ``created_at`` is when the row was written. Only
# smart_review_qa records the former, so exercise_qa falls back to the latter.
EXPORT_QUERY = """
SELECT
    t.user_id,
    t.subject_id        AS subject,
    t.course_id,
    t.course_part_id,
    t.score,
    t.question_started_at,
    t.response_time_ms
FROM (
    SELECT
        eq.user_id,
        c.subject_id,
        eq.course_id,
        eq.part_id                       AS course_part_id,
        eq.score,
        eq.created_at                    AS question_started_at,
        COALESCE(eq.response_time_ms, 0) AS response_time_ms
    FROM exercise_qa eq
    LEFT JOIN courses c ON c.id = eq.course_id
    WHERE eq.part_id IS NOT NULL
      AND eq.score IS NOT NULL

    UNION ALL

    SELECT
        sq.user_id,
        c.subject_id,
        sq.course_id,
        sq.course_part_id,
        sq.score,
        COALESCE(sq.question_started_at, sq.created_at) AS question_started_at,
        COALESCE(sq.response_time_ms, 0)                AS response_time_ms
    FROM smart_review_qa sq
    LEFT JOIN courses c ON c.id = sq.course_id
    WHERE sq.course_part_id IS NOT NULL
      AND sq.score IS NOT NULL
      AND sq.completion_status = 'answered'
) AS t
WHERE t.question_started_at IS NOT NULL
{user_filter}
ORDER BY t.user_id, t.question_started_at
{limit_clause}
"""

FIELDNAMES = [
    "user_id",
    "subject",
    "course_id",
    "course_part_id",
    "score",
    "question_started_at",
    "response_time_ms",
]


def _connect():
    """Open a MySQL connection, preferring PyMySQL and falling back to the
    official connector — whichever the deployment happens to have installed."""
    db = config.DATABASE
    if not db.is_configured():
        raise SystemExit("Set DB_USER, DB_PASSWORD and DB_NAME before exporting")

    try:
        import pymysql

        return pymysql.connect(
            host=db.host,
            port=db.port,
            user=db.user,
            password=db.password,
            database=db.database,
            charset="utf8mb4",
            cursorclass=pymysql.cursors.DictCursor,
        )
    except ImportError:
        pass

    try:
        import mysql.connector

        return mysql.connector.connect(
            host=db.host,
            port=db.port,
            user=db.user,
            password=db.password,
            database=db.database,
            charset="utf8mb4",
        )
    except ImportError as exc:
        raise SystemExit("Install pymysql (pip install pymysql) to export from MySQL") from exc


def fetch_interactions(user_id: int | None = None, limit: int | None = None) -> list[dict]:
    """Run the export query and return rows as dictionaries."""
    sql = EXPORT_QUERY.format(
        user_filter="AND t.user_id = %s" if user_id is not None else "",
        limit_clause=f"LIMIT {int(limit)}" if limit else "",
    )
    params = (user_id,) if user_id is not None else ()

    conn = _connect()
    try:
        try:  # PyMySQL returns dicts through the cursor class set above.
            cursor = conn.cursor()
        except TypeError:  # pragma: no cover
            cursor = conn.cursor(dictionary=True)

        cursor.execute(sql, params)
        rows = cursor.fetchall()
        cursor.close()
    finally:
        conn.close()

    normalised: list[dict] = []
    for row in rows:
        if not isinstance(row, dict):  # mysql-connector without dictionary=True
            row = dict(zip(FIELDNAMES, row))
        row["question_started_at"] = str(row["question_started_at"])
        normalised.append({k: row.get(k) for k in FIELDNAMES})
    return normalised


def main() -> None:
    parser = argparse.ArgumentParser(description="Export interactions from MySQL to CSV")
    parser.add_argument("--out", type=Path, default=Path("data/interactions.csv"))
    parser.add_argument("--user", type=int, default=None, help="Export one learner only")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--min-interactions",
        type=int,
        default=2,
        help="Drop learners with fewer interactions than this",
    )
    args = parser.parse_args()

    rows = fetch_interactions(user_id=args.user, limit=args.limit)
    if not rows:
        raise SystemExit("No interactions found. Has anyone answered a graded question yet?")

    counts: dict[int, int] = {}
    for row in rows:
        counts[row["user_id"]] = counts.get(row["user_id"], 0) + 1
    kept = [r for r in rows if counts[r["user_id"]] >= args.min_interactions]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(kept)

    dropped = len(rows) - len(kept)
    print(f"[export] wrote {len(kept)} interactions to {args.out}")
    print(f"[export] learners: {len({r['user_id'] for r in kept})}")
    print(f"[export] distinct parts: {len({r['course_part_id'] for r in kept})}")
    if dropped:
        print(f"[export] dropped {dropped} row(s) from learners below the minimum")


if __name__ == "__main__":
    main()
