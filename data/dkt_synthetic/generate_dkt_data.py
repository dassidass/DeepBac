"""Generate synthetic DKT training CSV aligned with bacheroes schema."""
import csv
import math
import random
import re
from datetime import datetime, timedelta
from pathlib import Path

SQL_PATH = Path(__file__).resolve().parent / "bacheroes(2).sql"
OUT_PATH = Path(__file__).resolve().parent / "dkt_synthetic_interactions.csv"

N_ROWS = 1000000
N_STUDENTS = 10000
USER_ID_START = 90001  # synthetic IDs, distinct from dump users

SUBJECT_ENCODE: dict[int, int] = {
    1: 1, 2: 2, 4: 3, 5: 4, 6: 5, 7: 6, 8: 7, 9: 8, 10: 9, 0: 1,
}

def parse_sql(text: str):
    course_subject: dict[int, int] = {}
    cm = re.search(r"INSERT INTO `courses`.*?\n(.*?);\n", text, re.DOTALL)
    assert cm, "courses INSERT not found"
    for m in re.finditer(r"\((\d+),\s*(\d+),", cm.group(1)):
        course_subject[int(m.group(1))] = int(m.group(2))

    parts: list[tuple[int, int]] = []
    for chunk in text.split("INSERT INTO `course_parts`")[1:]:
        vi = chunk.find("VALUES")
        if vi == -1:
            continue
        body = chunk[vi:]
        cut = len(body)
        for stop in ("\n\nCREATE TABLE ", "\n\nINSERT INTO ", "\n\n--\n-- Table structure"):
            p = body.find(stop)
            if p != -1:
                cut = min(cut, p)
        body = body[:cut]
        snippet = body
        for line in snippet.splitlines():
            line = line.strip()
            if not line.startswith("("):
                continue
            m = re.match(r"\((\d+),\s*(\d+),\s*'", line)
            if not m:
                continue
            part_id, course_id = int(m.group(1)), int(m.group(2))
            if course_id == 0 or course_id not in course_subject:
                continue
            parts.append((course_id, part_id))
    return course_subject, parts

def main():
    random.seed(42)
    text = SQL_PATH.read_text(encoding="utf-8", errors="replace")
    course_subject, part_pairs = parse_sql(text)
    if not part_pairs:
        raise SystemExit("No valid course parts parsed")

    by_course: dict[int, list[int]] = {}
    for cid, pid in part_pairs:
        by_course.setdefault(cid, []).append(pid)

    course_ids = sorted(by_course.keys())
    random.shuffle(course_ids)

    def pick_part_for_course(cid: int) -> int:
        return random.choice(by_course[cid])

    user_ids = [USER_ID_START + i for i in range(N_STUDENTS)]

    # قدرة الطالب وسرعته
    ability = {u: random.gauss(0, 0.4) for u in user_ids}
    speed_factor = {u: random.lognormvariate(0, 0.25) for u in user_ids}

    per = N_ROWS // N_STUDENTS
    if per * N_STUDENTS != N_ROWS:
        raise SystemExit("N_ROWS must be divisible by N_STUDENTS")
    assignment = user_ids * per
    random.shuffle(assignment)

    rows: list[dict] = []
    base = datetime(2025, 9, 1, 8, 0, 0)

    for i in range(N_ROWS):
        u = assignment[i]
        cid = course_ids[i % len(course_ids)]
        sid = course_subject[cid]
        subj = SUBJECT_ENCODE.get(sid, 1)
        pid = pick_part_for_course(cid)

        t_frac = i / max(N_ROWS - 1, 1)
        
        logit = ability[u] + 1.2 * t_frac + random.gauss(0, 0.1)
        
        p_mastery = 1 / (1 + math.exp(-logit))
        
        raw_score = p_mastery * 100 + random.gauss(0, 2.5)
        score = int(max(0, min(100, raw_score))) # حصر العلامة بين 0 و 100

        correct = p_mastery >= 0.5

        base_ms = math.exp(random.gauss(math.log(6500), 0.45)) * speed_factor[u]
        if not correct:
            base_ms *= random.uniform(1.1, 1.65)
        response_ms = int(max(2500, min(480_000, base_ms)))

        ts = base + timedelta(seconds=i * 84 + random.randint(-25, 55), milliseconds=random.randint(0, 999))

        rows.append(
            {
                "user_id": u,
                "subject": subj,
                "course_id": cid,
                "course_part_id": pid,
                "score": score,
                "question_started_at": ts.strftime("%Y-%m-%d %H:%M:%S"),
                "respones_time_ms": response_ms,
            }
        )

    rows.sort(key=lambda r: (r["user_id"], r["question_started_at"]))

    with OUT_PATH.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "user_id",
                "subject",
                "course_id",
                "course_part_id",
                "score",
                "question_started_at",
                "respones_time_ms",
            ],
        )
        w.writeheader()
        w.writerows(rows)

    print(f"Wrote {len(rows)} rows to {OUT_PATH} with SMOOTHED Learning Curves.")

if __name__ == "__main__":
    main()