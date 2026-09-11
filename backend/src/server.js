/**
 * Standalone RAG service.
 *
 * This is the retrieval-and-generation slice of the DeepBac platform, extracted
 * so it can be run, measured and cited on its own. It exposes the same
 * `/api/rag/*` contract the production platform mounts, talks to the same MySQL
 * schema, and indexes into the same Qdrant collection.
 *
 * Startup sequence:
 *   1. Verify the MySQL pool answers a ping.
 *   2. Optionally rebuild `rag_chunks` from `course_parts` in the background,
 *      and mirror the result into Qdrant. Set RAG_AUTO_SYNC=0 to skip, which is
 *      what you want on a read-only replica or during an evaluation run that
 *      must index a frozen corpus.
 *   3. Serve.
 */

const express = require('express');
const cors = require('cors');
const helmet = require('helmet');
const rateLimit = require('express-rate-limit');
const morgan = require('morgan');
const dotenv = require('dotenv');

dotenv.config();

const { pool, pingDatabase } = require('./config/database');
const { isHostedGeneratorConfigured, LLM_CHAT_MODEL } = require('./config/llm');
const qdrantSemantic = require('./services/qdrantSemantic');

const app = express();

// Behind a reverse proxy, rate limiting must read the forwarded address rather
// than the proxy's own. Express rejects `true` here in combination with
// express-rate-limit v7, so the hop count is explicit.
app.set('trust proxy', Number(process.env.TRUST_PROXY_HOPS || 0));

app.use(helmet());
app.use(cors({ origin: process.env.CLIENT_URL || '*', credentials: true }));
app.use(express.json({ limit: '1mb' }));
app.use(morgan('short'));

// Each answer costs one embedding call and one generation call, so the limit is
// deliberately tighter than a normal CRUD API would need.
app.use(
  rateLimit({
    windowMs: 15 * 60 * 1000,
    max: Number(process.env.RATE_LIMIT_MAX || 200),
    standardHeaders: true,
    legacyHeaders: false
  })
);

app.use('/api/rag', require('./routes/rag').router);

/** Reports every dependency separately, so a failure names the component at fault. */
app.get('/api/health', async (req, res) => {
  const db = await pingDatabase();
  res.json({
    status: db.ok ? 'OK' : 'DEGRADED',
    database: db,
    retrieval: qdrantSemantic.isSemanticRagEnabled() ? 'qdrant+embeddings' : 'keyword-only',
    qdrantCollection: qdrantSemantic.COLLECTION,
    vectorSize: qdrantSemantic.VECTOR_SIZE,
    generator: isHostedGeneratorConfigured() ? LLM_CHAT_MODEL : 'not configured'
  });
});

app.use('*', (req, res) => res.status(404).json({ error: 'Route not found' }));

// eslint-disable-next-line no-unused-vars
app.use((err, req, res, next) => {
  console.error(err.stack);
  res.status(500).json({ error: 'Internal server error' });
});

const PORT = Number(process.env.PORT || 5001);
const autoSyncOff = /^(0|false|no)$/i.test(String(process.env.RAG_AUTO_SYNC ?? '').trim());

if (require.main === module) {
  app.listen(PORT, async () => {
    console.log(`DeepBac RAG service listening on port ${PORT}`);

    const db = await pingDatabase();
    if (!db.ok) {
      console.error('MySQL unreachable:', db.reason);
      return;
    }
    console.log(
      `Retrieval mode: ${qdrantSemantic.isSemanticRagEnabled() ? 'semantic (Qdrant)' : 'keyword fallback'}`
    );

    if (autoSyncOff) {
      console.log('[RAG sync] skipped (RAG_AUTO_SYNC=0)');
      return;
    }

    // Deferred so the port is already accepting connections: a full re-index
    // embeds every chunk and can take minutes on a cold collection.
    setImmediate(async () => {
      try {
        const { syncRagChunksFromCourses } = require('./services/ragChunkSync');
        const result = await syncRagChunksFromCourses(pool);
        console.log(
          `[RAG sync] ${result.chunks} chunk(s) across ${result.courses} course(s) in ${result.ms}ms`
        );
      } catch (e) {
        console.error('[RAG sync] failed:', e.message);
      }
    });
  });
}

module.exports = { app };
