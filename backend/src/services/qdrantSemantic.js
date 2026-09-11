/**
 * Qdrant + Cohere embeddings for RAG retrieval.
 *
 * Enabled when QDRANT_URL and COHERE_API_KEY are set, unless RAG_USE_QDRANT=0|false|no.
 * Chunk indexing uses inputType=search_document; queries use search_query (Cohere best practice).
 */

const { QdrantClient } = require('@qdrant/js-client-rest');
const { CohereClient } = require('cohere-ai');

const COLLECTION =
  (process.env.RAG_QDRANT_COLLECTION && String(process.env.RAG_QDRANT_COLLECTION).trim()) ||
  'bac_rag_chunks';
const EMBEDDING_MODEL =
  (process.env.RAG_EMBEDDING_MODEL && String(process.env.RAG_EMBEDDING_MODEL).trim()) ||
  'embed-multilingual-v3.0';

const defaultVectorDims = EMBEDDING_MODEL.includes('embed-v4') ? 1536 : 1024;
const VECTOR_SIZE = Math.max(
  1,
  parseInt(String(process.env.RAG_EMBEDDING_DIMENSIONS || String(defaultVectorDims)), 10) ||
    defaultVectorDims
);
const EMBED_BATCH = Math.min(
  96,
  Math.min(
    100,
    Math.max(1, parseInt(String(process.env.RAG_EMBED_BATCH_SIZE || '32'), 10) || 32)
  )
);

/**
 * Normalizes Qdrant URL (trim, strip trailing slashes, strip UTF-8 BOM on copy-paste).
 */
function qdrantUrl() {
  const u = process.env.QDRANT_URL;
  if (!u || String(u).trim() === '') return '';
  let s = String(u).trim().replace(/^\uFEFF/, '');
  s = s.replace(/\/+$/, '');
  return s;
}

/** Trimmed Qdrant Cloud / local API key. */
function qdrantApiKey() {
  const k = process.env.QDRANT_API_KEY;
  if (!k || String(k).trim() === '') return undefined;
  return String(k).trim().replace(/^\uFEFF/, '');
}

/**
 * Qdrant Cloud “Database API keys” with granular access are JWTs (often start with eyJ).
 * REST docs allow `api-key` or `Authorization: Bearer`; some setups work reliably with Bearer only.
 */
function buildQdrantClientOptions() {
  const url = qdrantUrl();
  const key = qdrantApiKey();
  const forceApiKeyHeader = /^(1|true|yes)$/i.test(
    String(process.env.QDRANT_USE_API_KEY_HEADER ?? '').trim()
  );
  const forceBearer = /^(1|true|yes)$/i.test(String(process.env.QDRANT_USE_BEARER_AUTH ?? '').trim());

  const opts = {
    url,
    checkCompatibility: false
  };
  if (key) {
    if (forceApiKeyHeader) {
      opts.apiKey = key;
    } else if (forceBearer || /^eyJ/i.test(key)) {
      opts.headers = { Authorization: `Bearer ${key}` };
    } else {
      opts.apiKey = key;
    }
  }
  return opts;
}

function cohereToken() {
  const t = process.env.COHERE_API_KEY || process.env.COHERE_TOKEN;
  return t && String(t).trim() !== '' ? String(t).trim() : '';
}

function isSemanticRagEnabled() {
  const explicitOff = /^(0|false|no)$/i.test(String(process.env.RAG_USE_QDRANT ?? '').trim());
  if (explicitOff) return false;
  return Boolean(qdrantUrl() && cohereToken());
}

let _qdrant;
function getQdrant() {
  if (!isSemanticRagEnabled()) return null;
  if (!_qdrant) {
    _qdrant = new QdrantClient(buildQdrantClientOptions());
  }
  return _qdrant;
}

let _cohere;
function getCohere() {
  if (!cohereToken()) return null;
  if (!_cohere) {
    _cohere = new CohereClient({ token: cohereToken() });
  }
  return _cohere;
}

/**
 * Qdrant point IDs must be unsigned int64 or UUID — arbitrary strings (e.g. `c1_i0`) return 400 Bad Request.
 * Encode as course_id * CHUNK_INDEX_SPACE + chunk_index (per course, chunk_index must be below CHUNK_INDEX_SPACE).
 */
const CHUNK_INDEX_SPACE = 1_000_000;

function qdrantPointId(courseId, chunkIndex) {
  const c = Number(courseId);
  const i = Number(chunkIndex);
  if (!Number.isFinite(c) || !Number.isFinite(i) || i < 0 || i >= CHUNK_INDEX_SPACE) {
    throw new Error(
      `[Qdrant] chunk_index out of range for composite id: course ${courseId} chunk ${chunkIndex} (must be 0..${CHUNK_INDEX_SPACE - 1})`
    );
  }
  const id = c * CHUNK_INDEX_SPACE + i;
  if (!Number.isSafeInteger(id)) {
    throw new Error(`[Qdrant] point id not a safe integer for course ${courseId}`);
  }
  return id;
}

function embeddingInputForChunk(courseTitle, sectionTitle, content) {
  const t = `${courseTitle || ''}\n${sectionTitle || ''}\n${content || ''}`.trim();
  return t.slice(0, 30000);
}

function extractFloatEmbeddings(resp) {
  const emb = resp && resp.embeddings;
  if (!emb) return [];
  if (Array.isArray(emb.float)) return emb.float;
  return [];
}

/**
 * Safe one-line summary for debugging 403 (no secrets).
 */
function qdrantAuthModeLabel() {
  const key = qdrantApiKey();
  if (!key) return 'no key';
  if (/^(1|true|yes)$/i.test(String(process.env.QDRANT_USE_API_KEY_HEADER ?? '').trim())) {
    return 'api-key header (QDRANT_USE_API_KEY_HEADER=1)';
  }
  if (/^(1|true|yes)$/i.test(String(process.env.QDRANT_USE_BEARER_AUTH ?? '').trim())) {
    return 'Authorization: Bearer (QDRANT_USE_BEARER_AUTH=1)';
  }
  if (/^eyJ/i.test(key)) {
    return 'Authorization: Bearer (JWT-style Database API key)';
  }
  return 'api-key header';
}

function qdrantEnvDebugSummary() {
  const raw = qdrantUrl();
  if (!raw) return 'QDRANT_URL is empty.';
  try {
    const u = new URL(raw);
    const hasKey = Boolean(qdrantApiKey());
    return (
      `QDRANT_URL → ${u.protocol}//${u.host} (port ${u.port || 'default'})\n` +
      `QDRANT_API_KEY → ${hasKey ? 'set' : 'not set'}\n` +
      `Auth mode → ${qdrantAuthModeLabel()}`
    );
  } catch {
    return `QDRANT_URL is not a valid URL: ${raw.slice(0, 80)}${raw.length > 80 ? '…' : ''}`;
  }
}

/**
 * Human-readable error for Cohere / Qdrant failures (403, etc.).
 * @param {unknown} err
 */
function formatSemanticAccessError(err) {
  const e = err && typeof err === 'object' ? err : {};
  const name = /** @type {{ name?: string }} */ (e).name || 'Error';
  const message = err instanceof Error ? err.message : String(err);
  const statusCode = /** @type {{ statusCode?: number }} */ (e).statusCode;
  const body = /** @type {{ body?: unknown }} */ (e).body;
  const cause = /** @type {{ cause?: unknown }} */ (e).cause;
  const lines = [`${name}: ${message}`];
  if (statusCode != null) lines.push(`HTTP status: ${statusCode}`);
  if (body != null) {
    try {
      lines.push(`Body: ${typeof body === 'string' ? body : JSON.stringify(body)}`);
    } catch {
      lines.push(`Body: (unprintable)`);
    }
  }
  if (cause != null) {
    lines.push(`Caused by: ${cause instanceof Error ? cause.message : String(cause)}`);
  }
  const text = lines.join('\n');
  const isDenied = /403|401|Forbidden|Unauthorized/i.test(message + String(statusCode));
  if (!isDenied) {
    return text;
  }
  const qdrantFail = /^\[Qdrant\]|Qdrant|Unexpected Response/i.test(message);
  const hintQdrant =
    'This failure happened while talking to Qdrant (not Cohere yet).\n' +
    '· Copy QDRANT_URL exactly from Qdrant Cloud → cluster → “Endpoint” (include https and port :6333).\n' +
    '· Create a new Database API key on that same cluster (keys expire by default; use Manage/Write, not collection-only unless you know what you are doing).\n' +
    '· JWT-style keys (`eyJ…`) use Authorization: Bearer in this app. If you still get 403, try QDRANT_USE_API_KEY_HEADER=1 to force the classic `api-key` header instead.\n' +
    '· Local Docker: QDRANT_URL=http://127.0.0.1:6333 and unset QDRANT_API_KEY.\n' +
    '· Current env (no secrets):\n' +
    qdrantEnvDebugSummary();
  const hintCohere =
    '· Cohere: set COHERE_API_KEY from https://dashboard.cohere.com — the key must be allowed to call Embed v2.';
  return (
    `${text}\n\n` +
    (qdrantFail
      ? `403 / access denied (Qdrant).\n\n${hintQdrant}`
      : `403 / access denied.\n\n${hintQdrant}\n\n${hintCohere}`)
  );
}

async function collectionExists(client, name) {
  const data = await client.getCollections();
  const list = data && data.collections ? data.collections : [];
  return list.some((c) => (c && c.name) === name);
}

let _ensured;
async function ensureCollection() {
  const client = getQdrant();
  if (!client) return false;
  if (_ensured) return true;

  try {
    const exists = await collectionExists(client, COLLECTION);
    if (!exists) {
      await client.createCollection(COLLECTION, {
        vectors: { size: VECTOR_SIZE, distance: 'Cosine' }
      });
      console.log(`[Qdrant] collection "${COLLECTION}" created (${VECTOR_SIZE} dims, Cosine)`);
    }
  } catch (err) {
    const inner = err instanceof Error ? err.message : String(err);
    throw new Error(`[Qdrant] ${inner}`, { cause: err });
  }
  _ensured = true;
  return true;
}

/**
 * @param {string[]} texts
 * @param {'search_document' | 'search_query'} [inputType]
 * @returns {Promise<number[][]>}
 */
async function embedTexts(texts, inputType = 'search_document') {
  const cohere = getCohere();
  if (!cohere || !texts.length) return [];

  /** @type {number[][]} */
  const out = [];
  for (let i = 0; i < texts.length; i += EMBED_BATCH) {
    const batch = texts.slice(i, i + EMBED_BATCH);
    const req = {
      texts: batch,
      model: EMBEDDING_MODEL,
      inputType,
      embeddingTypes: ['float']
    };
    if (EMBEDDING_MODEL.includes('embed-v4')) {
      if ([256, 512, 1024, 1536].includes(VECTOR_SIZE)) {
        req.outputDimension = VECTOR_SIZE;
      }
    }

    const res = await cohere.v2.embed(req);
    const floats = extractFloatEmbeddings(res);
    if (floats.length !== batch.length) {
      console.error('[Qdrant] Cohere returned wrong number of vectors');
      throw new Error('Cohere embedding count mismatch');
    }
    for (const v of floats) {
      const len = v ? v.length : 0;
      if (len !== VECTOR_SIZE) {
        console.error(
          `[Qdrant] Cohere dim ${len}, expected ${VECTOR_SIZE} — set RAG_EMBEDDING_DIMENSIONS to match your model (v3 multilingual ≈ 1024, embed-v4 uses outputDimension)`
        );
        throw new Error('Embedding dimension mismatch');
      }
      out.push(v);
    }
  }
  return out;
}

async function deleteQdrantByCourseId(courseId) {
  const client = getQdrant();
  if (!client || courseId == null) return;
  try {
    if (!(await collectionExists(client, COLLECTION))) return;
    await client.delete(COLLECTION, {
      wait: true,
      filter: {
        must: [{ key: 'course_id', match: { value: Number(courseId) } }]
      }
    });
  } catch (e) {
    const msg = e && e.message ? e.message : String(e);
    if (/404|not found/i.test(msg)) return;
    console.warn(`[Qdrant] delete course ${courseId}:`, msg);
  }
}

async function deleteQdrantByCourseIds(courseIds) {
  if (!courseIds || !courseIds.length) return;
  for (const id of courseIds) {
    await deleteQdrantByCourseId(id);
  }
}

/**
 * Upsert all MySQL rag_chunks for a course into Qdrant (after DB insert).
 * Set RAG_QDRANT_ASYNC=1 to not block (e.g. large startup full-sync).
 * Pass { awaitUpsert: true } from scripts that must serialize work (backfill CLI).
 * @param {import('mysql2/promise').Pool} pool
 * @param {number} courseId
 * @param {{ awaitUpsert?: boolean }} [opts]
 */
async function upsertQdrantForCourse(pool, courseId, opts = {}) {
  if (!isSemanticRagEnabled() || !pool || !courseId) return;

  const client = getQdrant();
  if (!client) return;

  const run = async () => {
    await ensureCollection();

    const [[meta]] = await pool.execute(
      `SELECT c.id, c.title, c.subject_id FROM courses c WHERE c.id = ?`,
      [courseId]
    );
    if (!meta) return;

    const [rows] = await pool.execute(
      `SELECT id, course_id, part_id, chunk_index, section_title, content
       FROM rag_chunks
       WHERE course_id = ?
       ORDER BY chunk_index ASC`,
      [courseId]
    );
    if (!rows.length) return;

    const subjectId = meta.subject_id != null ? Number(meta.subject_id) : 0;

    const texts = rows.map((r) => embeddingInputForChunk(meta.title, r.section_title, r.content));
    const vectors = await embedTexts(texts, 'search_document');
    if (vectors.length !== rows.length) {
      console.error('[Qdrant] upsert: embedding count mismatch');
      return;
    }

    const points = rows.map((r, i) => ({
      id: qdrantPointId(courseId, r.chunk_index),
      vector: vectors[i],
      payload: {
        subject_id: subjectId,
        course_id: Number(courseId),
        chunk_db_id: Number(r.id),
        // The knowledge tracer's micro-skill. Carried in the payload so a
        // "revise section 149" recommendation can be filtered for inside the
        // vector search, rather than retrieved broadly and filtered after.
        course_part_id: r.part_id != null ? Number(r.part_id) : null,
        chunk_index: Number(r.chunk_index),
        section_title: String(r.section_title || '').slice(0, 500),
        course_title: String(meta.title || '').slice(0, 500)
      }
    }));

    const UPSERT_BATCH = 64;
    for (let j = 0; j < points.length; j += UPSERT_BATCH) {
      await client.upsert(COLLECTION, {
        wait: true,
        points: points.slice(j, j + UPSERT_BATCH)
      });
    }
  };

  const asyncUpsert =
    !opts.awaitUpsert &&
    /^(1|true|yes)$/i.test(String(process.env.RAG_QDRANT_ASYNC || '').trim());
  if (asyncUpsert) {
    run().catch((e) =>
      console.warn(`[Qdrant] upsert course ${courseId} failed:`, e instanceof Error ? e.message : e)
    );
    return;
  }

  try {
    await run();
  } catch (e) {
    const msg = e instanceof Error ? e.message : String(e);
    console.warn(`[Qdrant] upsert course ${courseId} failed:`, msg);
  }
}

/**
 * Semantic search: returns same row shape as keyword RAG (plus numeric score from Qdrant).
 * @param {import('mysql2/promise').Pool} pool
 * @param {string} query
 * @param {number} subjectId
 * @param {number} limit
 */
async function searchChunksByEmbedding(pool, query, subjectId, limit = 5) {
  if (!isSemanticRagEnabled() || !pool) return [];

  const q = (query || '').trim();
  if (!q) return [];

  const client = getQdrant();
  if (!client) return [];

  try {
    if (!(await collectionExists(client, COLLECTION))) return [];

    const qEmb = await embedTexts([q], 'search_query');
    const qv = qEmb[0];
    if (!qv) return [];

    const mult = Math.max(2, parseInt(String(process.env.RAG_QDRANT_OVERFETCH || '4'), 10) || 4);
    const fetchLimit = Math.min(80, limit * mult);

    const scoreThresholdEnv = process.env.RAG_QDRANT_SCORE_THRESHOLD;
    const score_threshold =
      scoreThresholdEnv !== undefined && String(scoreThresholdEnv).trim() !== ''
        ? Number(scoreThresholdEnv)
        : undefined;

    const hits = await client.search(COLLECTION, {
      vector: qv,
      limit: fetchLimit,
      filter: {
        must: [{ key: 'subject_id', match: { value: Number(subjectId) } }]
      },
      with_payload: true,
      score_threshold: Number.isFinite(score_threshold) ? score_threshold : undefined
    });

    /** @type {number[]} */
    const orderedChunkIds = [];
    const scoreByChunkId = new Map();
    for (const h of hits) {
      const pl = h.payload || {};
      const cid = Number(pl.chunk_db_id);
      if (!Number.isFinite(cid)) continue;
      if (!scoreByChunkId.has(cid)) {
        orderedChunkIds.push(cid);
        scoreByChunkId.set(cid, typeof h.score === 'number' ? h.score : 0);
      }
    }

    if (!orderedChunkIds.length) return [];

    const placeholders = orderedChunkIds.map(() => '?').join(',');
    const [rows] = await pool.execute(
      `SELECT 
        rc.id,
        rc.course_id,
        rc.part_id,
        rc.section_title,
        rc.content,
        c.title AS course_title,
        c.category
      FROM rag_chunks rc
      JOIN courses c ON c.id = rc.course_id
      WHERE rc.id IN (${placeholders})`,
      orderedChunkIds
    );

    const orderMap = new Map();
    orderedChunkIds.forEach((id, idx) => {
      if (!orderMap.has(id)) orderMap.set(id, idx);
    });

    const scored = rows.map((r) => ({
      ...r,
      score: scoreByChunkId.get(r.id) ?? 0
    }));
    scored.sort((a, b) => (orderMap.get(a.id) ?? 999) - (orderMap.get(b.id) ?? 999));

    const byKey = new Map();
    for (const c of scored) {
      const key = `${c.course_id}::${(c.section_title || '').slice(0, 120)}`;
      const prev = byKey.get(key);
      if (!prev || c.score > prev.score) byKey.set(key, c);
    }
    const deduped = [...byKey.values()].sort((a, b) => b.score - a.score);
    return deduped.slice(0, limit);
  } catch (e) {
    console.warn('[Qdrant] search failed:', e.message || e);
    return [];
  }
}

module.exports = {
  isSemanticRagEnabled,
  ensureCollection,
  embedTexts,
  deleteQdrantByCourseId,
  deleteQdrantByCourseIds,
  upsertQdrantForCourse,
  searchChunksByEmbedding,
  formatSemanticAccessError,
  COLLECTION,
  VECTOR_SIZE
};
