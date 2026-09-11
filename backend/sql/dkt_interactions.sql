-- Interaction export for knowledge tracing.
--
-- The DKT service adds no tables of its own. It reads graded answers the
-- platform already records, and writes nothing back to the database — model
-- state lives entirely in the checkpoint file. That keeps the platform schema
-- unchanged whether or not the service is deployed.
--
-- Two tables record graded answers and both are needed. Training on exercises
-- alone would make a student who practises mainly through AI review sessions
-- look inactive, and the model would predict accordingly.

SET NAMES utf8mb4;

-- ---------------------------------------------------------------------------
-- The export view
-- ---------------------------------------------------------------------------
--
-- `course_part_id` is the micro-skill: the authored lesson section. It is the
-- same unit the RAG backend retrieves over, which is what lets a weakness
-- prediction resolve to a specific section the student can be sent back to.

CREATE OR REPLACE VIEW dkt_interactions AS
SELECT
    eq.user_id,
    c.subject_id                     AS subject,
    eq.course_id,
    eq.part_id                       AS course_part_id,
    eq.score,
    -- exercise_qa does not record when the question was shown, only when the
    -- row was written. It is the best ordering key available for this source.
    eq.created_at                    AS question_started_at,
    COALESCE(eq.response_time_ms, 0) AS response_time_ms,
    'exercise'                       AS source
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
    -- smart_review_qa records the moment the question appeared, which is the
    -- correct ordering key; created_at is the fallback.
    COALESCE(sq.question_started_at, sq.created_at),
    COALESCE(sq.response_time_ms, 0),
    'smart_review'
FROM smart_review_qa sq
LEFT JOIN courses c ON c.id = sq.course_id
WHERE sq.course_part_id IS NOT NULL
  AND sq.score IS NOT NULL
  -- Generated-but-unanswered questions carry no evidence about the learner.
  AND sq.completion_status = 'answered';

-- ---------------------------------------------------------------------------
-- Indexes on the underlying tables
-- ---------------------------------------------------------------------------
-- The export is read per learner in chronological order. Without these it is a
-- full scan of both tables, which is tolerable nightly and painful when the
-- history endpoint serves one learner on request.
--
-- Run these once; MySQL has no CREATE INDEX IF NOT EXISTS, so a duplicate-name
-- error on a second run is expected and harmless.

-- CREATE INDEX idx_exqa_user_created  ON exercise_qa (user_id, created_at);
-- CREATE INDEX idx_srqa_user_started  ON smart_review_qa (user_id, question_started_at);

-- ---------------------------------------------------------------------------
-- Sanity checks before training
-- ---------------------------------------------------------------------------
-- Knowledge tracing needs repeated attempts on the same section by the same
-- learner. Without them there is no trajectory to trace, and the model cannot
-- beat a per-section average no matter how it is tuned. Check that first.

-- SELECT
--     COUNT(*)                                   AS interactions,
--     COUNT(DISTINCT user_id)                    AS learners,
--     COUNT(DISTINCT course_part_id)             AS sections,
--     COUNT(*) / NULLIF(COUNT(DISTINCT user_id), 0)  AS per_learner,
--     COUNT(*) / NULLIF(COUNT(DISTINCT CONCAT(user_id, ':', course_part_id)), 0)
--                                                AS attempts_per_learner_section
-- FROM dkt_interactions;

-- Learners with enough history to be worth modelling.
-- SELECT user_id, COUNT(*) AS n
-- FROM dkt_interactions
-- GROUP BY user_id
-- HAVING n >= 20
-- ORDER BY n DESC;

-- Score range. The platform's AI grader emits 0-100 even though an old schema
-- comment says 0-10; DKT_SCORE_SCALE must match whatever this returns.
-- SELECT MIN(score) AS min_score, MAX(score) AS max_score, AVG(score) AS avg_score
-- FROM dkt_interactions;
