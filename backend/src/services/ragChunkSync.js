const qdrantSemantic = require('./qdrantSemantic');

/**
 * Build `rag_chunks` from `course_parts` — one chunk per part.
 *
 * Strategy:
 *   1. Each non-empty `course_parts` row → exactly ONE rag_chunks row.
 *      section_title = part.title, content = part.content.
 *   2. If a course has NO parts but has legacy `courses.content`, that single
 *      field becomes one chunk (backward-compat for old data).
 *   3. No sliding-window splitting: every chunk maps 1-to-1 with a part.
 *
 * Supports:
 *  - Full sync at startup:      syncRagChunksFromCourses(pool)
 *  - Incremental after save:    reindexCourseAsync(pool, courseId)
 *  - Subject-level CLI rebuild: embedSubjectCourses(name, pool)
 */

/**
 * Legacy dumps can leave rag_chunks.id = 0 → duplicate primary key on next INSERT.
 */
async function pruneInvalidRagChunkIds(pool) {
  try {
    await pool.execute('DELETE FROM rag_chunks WHERE id <= 0');
  } catch (e) {
    console.warn('[RAG sync] could not delete id<=0 rows:', e.message);
  }
}

async function nextRagChunkId(pool) {
  const [[row]] = await pool.query('SELECT IFNULL(MAX(id), 0) AS m FROM rag_chunks');
  const m = parseInt(String(row.m), 10);
  return Math.max(1, (Number.isFinite(m) ? m : 0) + 1);
}

/**
 * Build the list of chunks for one course.
 * Returns: Array<{ partId, sectionTitle, content, chunkIndex }>
 *
 * Rule: one entry per non-empty course_part.
 * Fallback: if no parts exist but courses.content is non-empty, use that as a single chunk.
 *
 * `partId` is carried through to the stored row, and it is the join that makes
 * the platform's two model-backed services one product: the knowledge tracer's
 * micro-skill *is* `course_parts.id`, so a prediction that a learner is weak on
 * section 149 resolves to retrievable text with no mapping table in between.
 * The legacy single-blob fallback has no part and stores null.
 */
async function buildChunksForCourse(pool, course) {
  const [parts] = await pool.execute(
    `SELECT id, title, content, position
     FROM course_parts
     WHERE course_id = ?
       AND content IS NOT NULL
       AND TRIM(COALESCE(content, '')) <> ''
     ORDER BY position ASC, id ASC`,
    [course.id]
  );

  if (parts.length > 0) {
    return parts.map((p, i) => ({
      partId: p.id,
      sectionTitle: (p.title || `جزء ${i + 1}`).trim().substring(0, 200),
      content: (p.content || '').trim(),
      chunkIndex: i
    }));
  }

  // Fallback: legacy courses.content (no parts yet)
  const legacy = (course.content || '').trim();
  if (legacy) {
    const title = (course.title || 'الدرس').trim().substring(0, 200);
    return [{ partId: null, sectionTitle: title, content: legacy, chunkIndex: 0 }];
  }

  return [];
}

/**
 * Insert the precomputed chunks for one course into rag_chunks.
 * Deletes old chunks for that course first.
 * Returns number of chunks inserted.
 * @param {{ skipQdrant?: boolean }} [opts] — set skipQdrant for offline seeding when embeddings API is unavailable
 */
async function insertChunksForCourse(pool, courseId, chunks, opts = {}) {
  const skipQdrant = Boolean(opts && opts.skipQdrant);
  if (!skipQdrant) {
    await qdrantSemantic.deleteQdrantByCourseId(courseId);
  }
  await pool.execute('DELETE FROM rag_chunks WHERE course_id = ?', [courseId]);
  if (chunks.length === 0) return 0;

  let nextId = await nextRagChunkId(pool);
  for (const chunk of chunks) {
    await pool.execute(
      'INSERT INTO rag_chunks (id, course_id, part_id, chunk_index, section_title, content) VALUES (?, ?, ?, ?, ?, ?)',
      [
        nextId++,
        courseId,
        chunk.partId ?? null,
        chunk.chunkIndex,
        chunk.sectionTitle,
        chunk.content
      ]
    );
  }

  // Keep AUTO_INCREMENT aligned
  try {
    await pool.query(`ALTER TABLE rag_chunks AUTO_INCREMENT = ${nextId}`);
  } catch (_) { /* non-critical */ }

  if (!skipQdrant) {
    await qdrantSemantic.upsertQdrantForCourse(pool, courseId);
  }
  return chunks.length;
}

async function processCourse(pool, course) {
  try {
    const chunks = await buildChunksForCourse(pool, course);
    return await insertChunksForCourse(pool, course.id, chunks);
  } catch (error) {
    console.error(`[RAG sync] course ${course.id} (${course.title}):`, error.message);
    return 0;
  }
}

/** Courses that have text in course_parts or fallback in courses.content */
async function fetchCoursesEligibleForRag(pool) {
  const [courses] = await pool.execute(
    `SELECT DISTINCT c.id, c.title, c.content, c.category
     FROM courses c
     WHERE (c.content IS NOT NULL AND TRIM(COALESCE(c.content, '')) <> '')
        OR EXISTS (
          SELECT 1 FROM course_parts p
          WHERE p.course_id = c.id
            AND p.content IS NOT NULL
            AND TRIM(COALESCE(p.content, '')) <> ''
        )
     ORDER BY c.id ASC`
  );
  return courses;
}

/**
 * Full rebuild — one chunk per course_part.
 * Called at server startup in the background.
 * @param {import('mysql2/promise').Pool} pool
 * @param {{ quiet?: boolean }} [opts]
 */
async function syncRagChunksFromCourses(pool, opts = {}) {
  const quiet = Boolean(opts.quiet);
  if (!pool) {
    if (!quiet) console.warn('[RAG sync] skipped: no DB pool');
    return { ok: false, reason: 'no_pool', courses: 0, chunks: 0 };
  }

  const t0 = Date.now();
  await pruneInvalidRagChunkIds(pool);

  const courses = await fetchCoursesEligibleForRag(pool);
  if (!quiet) {
    console.log(`[RAG sync] indexing ${courses.length} course(s) → one chunk per part…`);
  }

  let totalChunks = 0;
  for (const course of courses) {
    const n = await processCourse(pool, course);
    totalChunks += n;
    if (!quiet && n > 0) {
      console.log(`[RAG sync]   course ${course.id} "${course.title}" → ${n} part-chunk(s)`);
    }
  }

  // Remove chunks whose parent course no longer exists
  const [orphanCourseRows] = await pool.execute(
    `SELECT DISTINCT rc.course_id AS course_id FROM rag_chunks rc
     WHERE NOT EXISTS (SELECT 1 FROM courses c WHERE c.id = rc.course_id)`
  );
  const orphanCourseIds = [
    ...new Set(orphanCourseRows.map((r) => r.course_id).filter((x) => x != null))
  ];
  if (orphanCourseIds.length) {
    await qdrantSemantic.deleteQdrantByCourseIds(orphanCourseIds);
  }
  const [delOrphan] = await pool.execute(
    `DELETE rc FROM rag_chunks rc
     WHERE NOT EXISTS (SELECT 1 FROM courses c WHERE c.id = rc.course_id)`
  );

  // Remove chunks for courses that now have no content anywhere
  const [emptyCourseRows] = await pool.execute(
    `SELECT DISTINCT rc.course_id AS course_id FROM rag_chunks rc
     INNER JOIN courses c ON c.id = rc.course_id
     WHERE (c.content IS NULL OR TRIM(COALESCE(c.content, '')) = '')
       AND NOT EXISTS (
         SELECT 1 FROM course_parts p
         WHERE p.course_id = c.id
           AND p.content IS NOT NULL
           AND TRIM(COALESCE(p.content, '')) <> ''
       )`
  );
  const emptyCourseIds = [...new Set(emptyCourseRows.map((r) => r.course_id))];
  if (emptyCourseIds.length) {
    await qdrantSemantic.deleteQdrantByCourseIds(emptyCourseIds);
  }
  const [delEmpty] = await pool.execute(
    `DELETE rc FROM rag_chunks rc
     INNER JOIN courses c ON c.id = rc.course_id
     WHERE (c.content IS NULL OR TRIM(COALESCE(c.content, '')) = '')
       AND NOT EXISTS (
         SELECT 1 FROM course_parts p
         WHERE p.course_id = c.id
           AND p.content IS NOT NULL
           AND TRIM(COALESCE(p.content, '')) <> ''
       )`
  );

  const removedStale = (delOrphan.affectedRows || 0) + (delEmpty.affectedRows || 0);
  const ms = Date.now() - t0;

  if (!quiet) {
    console.log(
      `[RAG sync] ✅ finished in ${ms}ms — ${totalChunks} chunk(s) across ${courses.length} course(s),` +
      ` removed stale: ${removedStale}`
    );
  }

  return { ok: true, courses: courses.length, chunks: totalChunks, removedStale, ms };
}

/**
 * Reindex one subject by exact subjects.name (CLI / manual trigger).
 */
async function embedSubjectCourses(subjectName, pool) {
  if (!pool) throw new Error('No DB pool');
  await pruneInvalidRagChunkIds(pool);
  const [subjects] = await pool.execute(
    'SELECT id, name FROM subjects WHERE name = ? ORDER BY id ASC LIMIT 1',
    [subjectName]
  );
  if (subjects.length === 0) {
    console.log(`[RAG] Subject "${subjectName}" not found`);
    return 0;
  }
  const subjectId = subjects[0].id;
  const [courses] = await pool.execute(
    `SELECT DISTINCT c.id, c.title, c.content, c.category
     FROM courses c
     WHERE c.subject_id = ?
       AND (
         (c.content IS NOT NULL AND TRIM(COALESCE(c.content, '')) <> '')
         OR EXISTS (
           SELECT 1 FROM course_parts p
           WHERE p.course_id = c.id
             AND p.content IS NOT NULL
             AND TRIM(COALESCE(p.content, '')) <> ''
         )
       )
     ORDER BY c.id`,
    [subjectId]
  );
  console.log(`[RAG] Subject "${subjects[0].name}" (id ${subjectId}): ${courses.length} course(s)`);
  let total = 0;
  for (const course of courses) {
    total += await processCourse(pool, course);
  }
  return total;
}

/**
 * Reindex a single course after any admin save/update.
 * Fire-and-forget: does NOT block the HTTP response.
 * @param {import('mysql2/promise').Pool} pool
 * @param {number} courseId
 * @param {{ skipQdrant?: boolean }} [opts]
 */
async function reindexCourseAsync(pool, courseId, opts = {}) {
  if (!pool || !courseId) return;
  try {
    const [[courseRow]] = await pool.execute(
      `SELECT c.id, c.title, c.content, c.category FROM courses c WHERE c.id = ?`,
      [courseId]
    );
    if (!courseRow) {
      await qdrantSemantic.deleteQdrantByCourseId(courseId);
      await pool.execute('DELETE FROM rag_chunks WHERE course_id = ?', [courseId]);
      console.log(`[RAG] course ${courseId} not found → rag_chunks cleared`);
      return;
    }

    await pruneInvalidRagChunkIds(pool);
    const chunks = await buildChunksForCourse(pool, courseRow);
    const n = await insertChunksForCourse(pool, courseId, chunks, opts);

    console.log(
      n > 0
        ? `[RAG] course ${courseId} "${courseRow.title}" → ${n} part-chunk(s) indexed`
        : `[RAG] course ${courseId} "${courseRow.title}" has no content → rag_chunks cleared`
    );
  } catch (err) {
    console.error(`[RAG] reindexCourseAsync failed for course ${courseId}:`, err.message);
  }
}

module.exports = {
  syncRagChunksFromCourses,
  reindexCourseAsync,
  embedSubjectCourses,
  fetchCoursesEligibleForRag,
  // kept for external scripts that may import them
  buildChunksForCourse,
  insertChunksForCourse,
  pruneInvalidRagChunkIds,
  nextRagChunkId
};
