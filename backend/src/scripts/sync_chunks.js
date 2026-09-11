/**
 * Rebuild the retrieval corpus from the curriculum tables.
 *
 *   node src/scripts/sync_chunks.js                     rebuild everything
 *   node src/scripts/sync_chunks.js --subject "الفلسفة"  rebuild one subject
 *   node src/scripts/sync_chunks.js --course 42          rebuild one course
 *   node src/scripts/sync_chunks.js --skip-qdrant        MySQL only, no embedding calls
 *
 * `--skip-qdrant` matters when seeding a fresh database offline or when the
 * embedding provider is unavailable: `rag_chunks` is rebuilt, the keyword
 * retriever starts working immediately, and the vectors can be filled in later
 * with backfill_qdrant.js.
 */

const { pool } = require('../config/database');
const {
  syncRagChunksFromCourses,
  embedSubjectCourses,
  reindexCourseAsync
} = require('../services/ragChunkSync');
const qdrantSemantic = require('../services/qdrantSemantic');

const argv = process.argv.slice(2);
const argValue = (name, fallback = null) => {
  const i = argv.indexOf(name);
  if (i === -1) return fallback;
  const v = argv[i + 1];
  return v == null || v.startsWith('--') ? fallback : v;
};

async function main() {
  if (!pool) throw new Error('Database pool unavailable — check DB_* variables in .env');

  const skipQdrant = argv.includes('--skip-qdrant');
  const subject = argValue('--subject');
  const courseId = argValue('--course');

  console.log(
    `[sync] semantic indexing ${
      skipQdrant ? 'disabled (--skip-qdrant)' : qdrantSemantic.isSemanticRagEnabled() ? 'on' : 'off (not configured)'
    }`
  );

  if (courseId) {
    await reindexCourseAsync(pool, Number(courseId), { skipQdrant });
    return;
  }

  if (subject) {
    const total = await embedSubjectCourses(subject, pool);
    console.log(`[sync] subject "${subject}" → ${total} chunk(s)`);
    return;
  }

  const result = await syncRagChunksFromCourses(pool);
  console.log(
    `[sync] ${result.chunks} chunk(s) across ${result.courses} course(s), ` +
      `stale removed: ${result.removedStale}, ${result.ms}ms`
  );
}

if (require.main === module) {
  main()
    .catch((e) => {
      console.error('[sync] failed:', e.message);
      process.exitCode = 1;
    })
    .finally(() => pool && pool.end().catch(() => {}));
}

module.exports = {};
