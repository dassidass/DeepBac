/**
 * Push all existing `rag_chunks` rows into Qdrant (Cohere embeddings).
 * Use after enabling QDRANT_URL + COHERE_API_KEY or to repair drift.
 *
 *   node ingest/qdrant_backfill.js
 */

const { pool } = require('../config/database');
const qdrantSemantic = require('../services/qdrantSemantic');

if (require.main === module) {
  (async () => {
    if (!qdrantSemantic.isSemanticRagEnabled()) {
      console.error(
        'Set QDRANT_URL and COHERE_API_KEY (and leave RAG_USE_QDRANT unset or true). Optionally QDRANT_API_KEY for Qdrant Cloud.'
      );
      process.exitCode = 1;
      return;
    }

    try {
      console.log('[Qdrant backfill] ensuring Qdrant collection…');
      await qdrantSemantic.ensureCollection();
      const [courses] = await pool.execute(
        'SELECT DISTINCT course_id FROM rag_chunks ORDER BY course_id ASC'
      );
      console.log(`[Qdrant backfill] ${courses.length} course(s) with chunks…`);
      let i = 0;
      for (const row of courses) {
        const id = row.course_id;
        await qdrantSemantic.upsertQdrantForCourse(pool, id, { awaitUpsert: true });
        i += 1;
        if (i % 20 === 0) console.log(`[Qdrant backfill] … ${i}/${courses.length}`);
      }
      console.log('[Qdrant backfill] done.');
    } catch (e) {
      const detail =
        typeof qdrantSemantic.formatSemanticAccessError === 'function'
          ? qdrantSemantic.formatSemanticAccessError(e)
          : e instanceof Error
            ? e.message
            : String(e);
      console.error('[Qdrant backfill] failed:\n', detail);
      process.exitCode = 1;
    } finally {
      await pool.end().catch(() => {});
    }
  })();
}

module.exports = {};
