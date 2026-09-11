/**
 * Shared RAG chunk retrieval (MySQL + optional Qdrant semantic).
 * Used by routes/rag.js and offline export scripts.
 */

const { pool } = require('../config/database');
const qdrantSemantic = require('./qdrantSemantic');

/**
 * Arabic and English function words. Matching on these inflates the keyword
 * score of unrelated lessons: almost every chunk in the corpus contains them,
 * so a query built out of them retrieves whatever happens to be longest.
 */
const QUERY_STOPWORDS = new Set(
  [
    'من',
    'في',
    'على',
    'إلى',
    'عن',
    'مع',
    'هو',
    'هي',
    'هذا',
    'هذه',
    'ذلك',
    'تلك',
    'ما',
    'ماذا',
    'لماذا',
    'كيف',
    'متى',
    'أين',
    'هل',
    'لما',
    'لان',
    'لأن',
    'كان',
    'كانت',
    'يكون',
    'تكون',
    'التي',
    'الذي',
    'الذين',
    'قد',
    'لم',
    'لن',
    'أن',
    'إن',
    'أو',
    'كل',
    'بعض',
    'غير',
    'سوى',
    'بين',
    'أيضا',
    'أيضاً',
    'جدا',
    'فقط',
    'اشرح',
    'اشرحي',
    'وضح',
    'علمني',
    'درس',
    'الدرس',
    'شرح',
    'سؤال',
    'السؤال',
    'the',
    'and',
    'for',
    'with',
    'that',
    'this',
    'what',
    'when',
    'where',
    'how',
    'why'
  ].map((w) => w.toLowerCase())
);

function queryMatchTokens(query) {
  const q = (query || '').trim();
  if (!q) return [];
  const arabic = (q.match(/[\u0600-\u06FF]{3,}/g) || []).filter((w) => !QUERY_STOPWORDS.has(w));
  const latin = (q.toLowerCase().match(/[a-z]{4,}/g) || []).filter((w) => !QUERY_STOPWORDS.has(w));
  const spaced = q
    .toLowerCase()
    .split(/\s+/)
    .map((w) => w.replace(/^[^\u0600-\u06FFa-z0-9]+|[^\u0600-\u06FFa-z0-9]+$/gi, ''))
    .filter((w) => w.length > 3 && !QUERY_STOPWORDS.has(w));
  return [...new Set([...arabic, ...latin, ...spaced])];
}

function calculateTextSimilarity(query, text, titleHint = '') {
  const queryWords = queryMatchTokens(query);
  if (queryWords.length === 0) return 0;

  const textLower = text.toLowerCase();
  const titleLower = (titleHint || '').toLowerCase();

  let score = 0;
  for (const word of queryWords) {
    const w = typeof word === 'string' && /[\u0600-\u06FF]/.test(word) ? word : word.toLowerCase();
    const slice = /[\u0600-\u06FF]/.test(w) ? text : textLower;
    if (slice.includes(w)) {
      score += 1;
      if (textLower.match(new RegExp(`#+.*${escapeRegex(w)}`, 'i'))) {
        score += 2;
      }
    }
    if (titleLower && titleLower.includes(w)) {
      score += 3;
    }
  }

  return score;
}

function escapeRegex(s) {
  return s.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
}

/**
 * Keyword overlap retrieval (fallback when Qdrant/embeddings are off or return nothing).
 */
async function retrieveRelevantChunksKeyword(query, subjectId, limit = 5) {
  try {
    const [chunks] = await pool.execute(
      `
      SELECT 
        rc.id,
        rc.course_id,
        rc.part_id,
        rc.section_title,
        rc.content,
        c.title as course_title,
        c.category
      FROM rag_chunks rc
      JOIN courses c ON c.id = rc.course_id
      WHERE c.subject_id = ?
      ORDER BY rc.course_id, rc.chunk_index
    `,
      [subjectId]
    );

    const titleHintFor = (chunk) => `${chunk.section_title || ''} ${chunk.course_title || ''}`;

    const scoredChunks = chunks.map((chunk) => ({
      ...chunk,
      score: calculateTextSimilarity(
        query,
        `${chunk.content} ${chunk.section_title} ${chunk.course_title}`,
        titleHintFor(chunk)
      )
    }));

    scoredChunks.sort((a, b) => b.score - a.score);

    const best = scoredChunks[0];
    if (!best || best.score < 2) {
      return [];
    }

    const minScore = Math.max(2, best.score * 0.42);
    const strong = scoredChunks.filter((c) => c.score >= minScore);
    if (strong.length === 0) {
      return [];
    }

    const byKey = new Map();
    for (const c of strong) {
      const key = `${c.course_id}::${(c.section_title || '').slice(0, 120)}`;
      const prev = byKey.get(key);
      if (!prev || c.score > prev.score) {
        byKey.set(key, c);
      }
    }
    const deduped = [...byKey.values()].sort((a, b) => b.score - a.score);

    return deduped.slice(0, limit);
  } catch (error) {
    console.error('Error retrieving chunks (keyword):', error);
    return [];
  }
}

/**
 * Retrieve relevant chunks — Qdrant semantic search when configured, else keyword overlap.
 */
async function retrieveRelevantChunks(query, subject = 'التاريخ والجغرافيا', limit = 5) {
  try {
    if (!pool) {
      return [];
    }

    const [subjects] = await pool.execute('SELECT id FROM subjects WHERE name = ?', [subject]);

    if (subjects.length === 0) {
      return [];
    }

    const subjectId = subjects[0].id;

    if (qdrantSemantic.isSemanticRagEnabled()) {
      const semantic = await qdrantSemantic.searchChunksByEmbedding(pool, query, subjectId, limit);
      if (semantic.length > 0) {
        return semantic;
      }
    }

    return retrieveRelevantChunksKeyword(query, subjectId, limit);
  } catch (error) {
    console.error('Error retrieving chunks:', error);
    return [];
  }
}

module.exports = {
  retrieveRelevantChunks
};
