"""
Generate the synthetic interaction corpus the knowledge tracer is trained on.

    python -m dkt.scripts.generate_synthetic --out data/dkt_interactions.csv
    python -m dkt.scripts.generate_synthetic --sql bacheroes.sql   # real ids from a dump

Why synthetic data exists here: a knowledge tracer needs hundreds of thousands
of interactions, and a platform before launch has none. This is the classic
cold-start problem, and the generator below answers it by simulating a cohort
whose latent structure the model can be built and debugged against, months
before the first real learner arrives.

The process is IRT-inspired rather than a fitted IRT response model. Item
response theory motivates the latent-ability variable; practice and noise
generate the longitudinal trajectory:

    Ability(u) ~ N(0, 0.4)
    y_hat(t)   = sigmoid(Ability(u) + 1.2 * t_frac + eps),  eps ~ N(0, 0.1)
    score      = 100 * y_hat(t) + N(0, 2.5)

``t_frac`` is normalised interaction progress, which is how the Power Law of
Practice enters: every learner improves along the same monotone curve, offset by
a static ability. The additive score noise reproduces the natural irregularity
of real assessment marks.

**What it can and cannot show.** A model that recovers this process validates
the code path — gradients flow, learner state is carried, predictions track the
target. It does not validate that the model works on human learners, because the
data was produced by exactly the kind of latent-trait process the model assumes.
Any number measured on this file is a test of the implementation. Real
interaction logs are the only thing that settles the method, and
docs/DKT_EVALUATION.md says so in its own words.

Passing ``--sql`` parses a live database dump so the generated rows carry real
``course_id`` and ``course_part_id`` values and preserve foreign-key integrity
against the production schema. A recommendation produced from this file then
points at a lesson section that actually exists. Without it, a curriculum of the
same shape is invented, so the script runs with no dump present.
"""

from __future__ import annotations

import argparse
import csv
import math
import random
import re
from datetime import datetime, timedelta
from pathlib import Path

FIELDNAMES = [
    "user_id",
    "subject",
    "course_id",
    "course_part_id",
    "score",
    "question_started_at",
    "response_time_ms",
]

# Synthetic learner ids start high so they can never collide with real users if
# the file is ever loaded into a database by accident.
USER_ID_START = 90001


def parse_sql_dump(path: Path) -> tuple[dict[int, int], list[tuple[int, int]]]:
    """Extract (course → subject) and (course, part) pairs from a MySQL dump.

    Using real identifiers means the synthetic file can be joined against the
    real curriculum, so a recommendation produced from it points at a lesson
    that actually exists.
    """
    text = path.read_text(encoding="utf-8", errors="replace")

    course_subject: dict[int, int] = {}
    courses_insert = re.search(r"INSERT INTO `courses`.*?\n(.*?);\n", text, re.DOTALL)
    if not courses_insert:
        raise SystemExit("No INSERT INTO `courses` found in the dump")
    for m in re.finditer(r"\((\d+),\s*(\d+),", courses_insert.group(1)):
        course_subject[int(m.group(1))] = int(m.group(2))

    parts: list[tuple[int, int]] = []
    # A dump can split one table across several INSERT statements, so every
    # chunk is scanned rather than only the first.
    for chunk in text.split("INSERT INTO `course_parts`")[1:]:
        values_at = chunk.find("VALUES")
        if values_at == -1:
            continue
        body = chunk[values_at:]

        # Stop at the next table, or the final chunk would run to end of file.
        cut = len(body)
        for stop in ("\n\nCREATE TABLE ", "\n\nINSERT INTO ", "\n\n--\n-- Table structure"):
            found = body.find(stop)
            if found != -1:
                cut = min(cut, found)

        for line in body[:cut].splitlines():
            line = line.strip()
            if not line.startswith("("):
                continue
            m = re.match(r"\((\d+),\s*(\d+),\s*'", line)
            if not m:
                continue
            part_id, course_id = int(m.group(1)), int(m.group(2))
            if course_id and course_id in course_subject:
                parts.append((course_id, part_id))

    return course_subject, parts


def synthetic_curriculum(
    n_courses: int, parts_per_course: int
) -> tuple[dict[int, int], list[tuple[int, int]]]:
    """Invent a curriculum of the same shape when no dump is available."""
    course_subject: dict[int, int] = {}
    parts: list[tuple[int, int]] = []
    part_id = 1
    for course_id in range(1, n_courses + 1):
        course_subject[course_id] = (course_id % 9) + 1
        for _ in range(parts_per_course):
            parts.append((course_id, part_id))
            part_id += 1
    return course_subject, parts


def generate(
    course_subject: dict[int, int],
    part_pairs: list[tuple[int, int]],
    *,
    n_rows: int,
    n_students: int,
    seed: int,
    ability_sd: float = 0.4,
    practice_gain: float = 1.2,
    mastery_noise_sd: float = 0.1,
    score_noise_sd: float = 2.5,
) -> list[dict]:
    """Simulate a cohort working through the curriculum.

    One row is one graded answer: who answered, which lesson section, what score
    they earned, and how long they took. Response time is not part of the
    mastery equation; it is generated as a correlate of mastery so the file has
    the same columns as a real export, and the model is free to use it.
    """
    rng = random.Random(seed)

    all_parts = list(part_pairs)
    if not all_parts:
        raise SystemExit("Curriculum contains no parts")

    user_ids = [USER_ID_START + i for i in range(n_students)]

    # Heterogeneous learners. Without this the cohort is one average student
    # repeated N times, and a model conditioned on learner state has nothing to
    # condition on.
    ability = {u: rng.gauss(0, ability_sd) for u in user_ids}
    speed_factor = {u: rng.lognormvariate(0, 0.25) for u in user_ids}

    per_student, remainder = divmod(n_rows, n_students)
    if remainder:
        raise SystemExit(f"--rows ({n_rows}) must divide evenly by --students ({n_students})")
    if per_student < 2:
        raise SystemExit("Each learner needs at least 2 interactions to produce a target")

    rows: list[dict] = []
    base_time = datetime(2025, 9, 1, 8, 0, 0)
    denom = per_student - 1

    for user_id in user_ids:
        for step in range(per_student):
            course_id, part_id = rng.choice(all_parts)
            subject_id = course_subject.get(course_id, 1)

            t_frac = step / denom
            logit = ability[user_id] + practice_gain * t_frac + rng.gauss(0, mastery_noise_sd)
            mastery = 1 / (1 + math.exp(-logit))

            score = mastery * 100.0 + rng.gauss(0, score_noise_sd)
            score = round(min(100.0, max(0.0, score)), 2)

            # Lognormal response time: heavily right-skewed, like real answer
            # times, and faster as the learner improves.
            base_ms = math.exp(rng.gauss(math.log(6500), 0.45)) * speed_factor[user_id]
            base_ms *= math.exp(-0.35 * mastery)
            response_ms = int(max(2500, min(480_000, base_ms)))

            timestamp = base_time + timedelta(
                seconds=step * 84 + rng.randint(-25, 55),
                milliseconds=rng.randint(0, 999),
            )

            rows.append(
                {
                    "user_id": user_id,
                    "subject": (subject_id % 9) + 1,
                    "course_id": course_id,
                    "course_part_id": part_id,
                    "score": score,
                    "question_started_at": timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                    "response_time_ms": response_ms,
                }
            )

    # Contiguous per-learner sequences, matching what the export query produces.
    rows.sort(key=lambda r: (r["user_id"], r["question_started_at"]))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the synthetic interaction CSV")
    parser.add_argument("--out", type=Path, default=Path("data/dkt_interactions.csv"))
    parser.add_argument("--sql", type=Path, default=None, help="MySQL dump for real course ids")
    parser.add_argument("--rows", type=int, default=1_000_000)
    parser.add_argument("--students", type=int, default=10_000)
    parser.add_argument("--courses", type=int, default=40, help="Used only without --sql")
    parser.add_argument("--parts-per-course", type=int, default=4, help="Used only without --sql")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ability-sd", type=float, default=0.4)
    parser.add_argument(
        "--practice-gain", type=float, default=1.2, help="Coefficient on normalised progress"
    )
    parser.add_argument(
        "--mastery-noise-sd", type=float, default=0.1, help="Epsilon inside the sigmoid"
    )
    parser.add_argument(
        "--score-noise-sd", type=float, default=2.5, help="Additive noise on the 0-100 score"
    )
    args = parser.parse_args()

    if args.sql:
        course_subject, part_pairs = parse_sql_dump(args.sql)
        if not part_pairs:
            raise SystemExit("No course parts parsed from the dump")
    else:
        course_subject, part_pairs = synthetic_curriculum(args.courses, args.parts_per_course)

    rows = generate(
        course_subject,
        part_pairs,
        n_rows=args.rows,
        n_students=args.students,
        seed=args.seed,
        ability_sd=args.ability_sd,
        practice_gain=args.practice_gain,
        mastery_noise_sd=args.mastery_noise_sd,
        score_noise_sd=args.score_noise_sd,
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

    print(f"[synthetic] wrote {len(rows)} rows to {args.out}")
    print(f"[synthetic] learners: {len({r['user_id'] for r in rows})}")
    print(f"[synthetic] courses: {len({r['course_id'] for r in rows})}")
    print(f"[synthetic] parts: {len({r['course_part_id'] for r in rows})}")


if __name__ == "__main__":
    main()
