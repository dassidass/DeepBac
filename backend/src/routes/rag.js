/**
 * RAG endpoints.
 *
 *   POST /api/rag/ask        retrieve → ground → generate, using the hosted LLM
 *   POST /api/rag/ask-local  the same, against a locally served model
 *   GET  /api/rag/subjects   subjects that currently have an indexed corpus
 *   GET  /api/rag/stats      corpus size statistics for one subject
 *
 * The two `ask` routes are deliberately near-identical. They are the two arms
 * of the generator comparison, and keeping the retrieval step byte-for-byte
 * identical is what makes the generator the only variable between them.
 */

const express = require('express');
const axios = require('axios');
const OpenAI = require('openai');

const { authMiddleware } = require('../middleware/auth');
const { pool } = require('../config/database');
const {
  LLM_API_KEY,
  LLM_API_URL,
  LLM_CHAT_MODEL,
  LLM_TIMEOUT_MS,
  LOCAL_OPENAI_BASE_URL,
  LOCAL_OPENAI_MODEL,
  LOCAL_OPENAI_API_KEY,
  LOCAL_OPENAI_TIMEOUT_MS
} = require('../config/llm');

const { retrieveRelevantChunks } = require('../services/ragRetrieval');
const {
  isChunkEvidenceWeak,
  isLikelyAlgerianBacQuestion,
  buildRagGroundedPrompts,
  buildRagKnowledgeFallbackPrompts,
  staticNoLessonMatchMessage
} = require('../services/ragAnswerModes');
const { resolveDifficultyTier } = require('../services/masterySteering');
const { detectRegister, registerInstruction } = require('../services/languageRegister');
const {
  screenQuestion,
  injectionRefusalMessage,
  verifyCitations
} = require('../services/promptGuard');

const router = express.Router();

const localOpenAI = new OpenAI({
  baseURL: LOCAL_OPENAI_BASE_URL,
  apiKey: LOCAL_OPENAI_API_KEY
});

const DEFAULT_SUBJECT = process.env.RAG_DEFAULT_SUBJECT || 'التاريخ والجغرافيا';

/** How many chunks the hosted path puts in the prompt. */
const TOP_K = Number(process.env.RAG_TOP_K || 5);
/** The local model has a smaller usable context, so it gets a shorter prompt. */
const TOP_K_LOCAL = Number(process.env.RAG_TOP_K_LOCAL || 3);

/**
 * Caps one chunk's contribution to the prompt.
 *
 * A single course part can run to several thousand characters. Without a cap,
 * one long part crowds out the four other retrieved chunks, and recall that the
 * retriever earned is thrown away at prompt-assembly time.
 */
const truncateForPrompt = (text, maxChars) => {
  const s = String(text || '');
  const n = Number(maxChars);
  if (!Number.isFinite(n) || n <= 0) return s;
  if (s.length <= n) return s;
  return `${s.slice(0, n)}\n\n[…تم اقتصار النص للسرعة…]`;
};

const coerceBoolean = (v) => {
  if (v === true || v === false) return v;
  const s = String(v ?? '').trim().toLowerCase();
  if (['1', 'true', 'yes', 'y', 'on'].includes(s)) return true;
  return false;
};

/** Renders retrieved rows as numbered, attributed source blocks. */
const buildContext = (chunks, perChunkChars, withScores = false) =>
  chunks
    .map((chunk, i) => {
      const score = withScores ? ` (درجة الارتباط: ${chunk.score})` : '';
      return (
        `[مصدر ${i + 1}: ${chunk.course_title} - ${chunk.section_title}]${score}\n` +
        truncateForPrompt(chunk.content, perChunkChars)
      );
    })
    .join('\n\n---\n\n');

/**
 * The citation list returned to the client next to every answer.
 *
 * `coursePartId` is the knowledge tracer's micro-skill key, so a client holding
 * an answer already knows which section to send back to the DKT service — and
 * which mastery score to send here on the next question.
 */
const toSources = (chunks) =>
  chunks.map((chunk) => ({
    courseId: chunk.course_id,
    coursePartId: chunk.part_id ?? null,
    courseTitle: chunk.course_title,
    sectionTitle: chunk.section_title,
    category: chunk.category,
    relevanceScore: chunk.score
  }));

/**
 * POST /api/rag/ask
 * Body: { question, subject?, mastery?, debugIncludeContext? }
 *
 * Three answer modes, decided before any token is generated:
 *
 *   grounded            strong retrieval — answer strictly from the chunks
 *   knowledge_fallback  weak retrieval, but the question is curricular — answer
 *                       from model knowledge, saying so in the first sentence
 *   no_match            weak retrieval and off-syllabus — refuse without calling
 *                       the model at all
 *
 * The third mode is what keeps an un-answerable question cheap, and it is also
 * the honest response: the platform sells grounded answers, so silently
 * switching to free-form generation would misrepresent the product.
 *
 * Two further layers shape the prompt without touching those modes. `mastery` —
 * a predicted score for this lesson section, supplied by the DKT service —
 * selects a difficulty tier; the detected language register selects how the
 * answer is worded. Both are optional, both degrade to nothing when absent, and
 * neither can loosen a grounding rule. Neither costs a model call: they modify
 * a prompt that was being assembled anyway.
 */
router.post('/ask', authMiddleware, async (req, res) => {
  try {
    const { question, subject = DEFAULT_SUBJECT, mastery, debugIncludeContext } = req.body;

    if (!question || question.trim().length === 0) {
      return res.status(400).json({ error: 'Question is required' });
    }

    if (!LLM_API_KEY) {
      return res.status(503).json({ error: 'Generator not configured (set LLM_API_KEY)' });
    }

    // Screened before retrieval: a question written to override the system
    // instruction should cost neither an embedding call nor a generation call.
    const screening = screenQuestion(question);
    if (!screening.safe) {
      return res.json({
        answer: injectionRefusalMessage(),
        sources: [],
        confidence: 'low',
        answerMode: 'rejected',
        rejectionReason: screening.reason,
        chunksUsed: 0
      });
    }

    const difficulty = resolveDifficultyTier(mastery);
    const register = detectRegister(question);
    const promptOptions = {
      difficulty,
      languageInstruction: registerInstruction(register)
    };

    const relevantChunks = await retrieveRelevantChunks(question, subject, TOP_K);

    const weakEvidence = isChunkEvidenceWeak(relevantChunks);
    const bacLikely = isLikelyAlgerianBacQuestion(question, subject);

    let systemPrompt;
    let userPrompt;
    let answerMode = 'grounded';
    // Grounded answers copy from the context, so sampling stays low. The
    // fallback has to compose prose, and 0.3 there reads mechanically.
    let temperature = 0.3;
    let contextUsed = null;

    if (!weakEvidence && relevantChunks.length > 0) {
      contextUsed = buildContext(relevantChunks, 1400);
      ({ systemPrompt, userPrompt } = buildRagGroundedPrompts(
        subject,
        contextUsed,
        question,
        promptOptions
      ));
    } else if (bacLikely) {
      answerMode = 'knowledge_fallback';
      temperature = 0.45;
      // Weak chunks are still passed, shortened and labelled with their score,
      // as optional hints. The prompt tells the model it may ignore them.
      contextUsed = relevantChunks.length > 0 ? buildContext(relevantChunks, 900, true) : '';
      ({ systemPrompt, userPrompt } = buildRagKnowledgeFallbackPrompts(
        subject,
        question,
        contextUsed,
        promptOptions
      ));
    } else {
      return res.json({
        answer: staticNoLessonMatchMessage(),
        sources: toSources(relevantChunks),
        confidence: 'low',
        answerMode: 'no_match',
        languageRegister: register,
        chunksUsed: relevantChunks.length
      });
    }

    const aiResponse = await axios.post(
      LLM_API_URL,
      {
        model: LLM_CHAT_MODEL,
        messages: [
          { role: 'system', content: systemPrompt },
          { role: 'user', content: userPrompt }
        ],
        max_tokens: Number(process.env.LLM_MAX_TOKENS || 1500),
        temperature
      },
      {
        headers: {
          'Content-Type': 'application/json',
          Authorization: `Bearer ${LLM_API_KEY}`
        },
        timeout: LLM_TIMEOUT_MS
      }
    );

    const answer = aiResponse.data.choices[0].message.content;

    // Checked, reported, and never silently rewritten. A fabricated citation is
    // worth surfacing to the client, but editing the model's text after the
    // fact would hand the student an answer nobody wrote, with no record of
    // what was removed.
    const citations = answerMode === 'grounded' ? verifyCitations(answer, relevantChunks) : null;
    if (citations && !citations.ok) {
      console.warn('[RAG] unverified citations in a grounded answer:', citations.unverified);
    }

    res.json({
      answer,
      sources: toSources(relevantChunks),
      answerMode,
      languageRegister: register,
      ...(difficulty ? { difficulty: { tier: difficulty.tier, mastery: difficulty.mastery } } : {}),
      ...(citations ? { citationCheck: { ok: citations.ok, unverified: citations.unverified } } : {}),
      // Reported to the student as a badge, and logged for the evaluation runs.
      confidence:
        answerMode === 'knowledge_fallback'
          ? relevantChunks.length >= 1
            ? 'medium'
            : 'low'
          : relevantChunks.length >= 3
            ? 'high'
            : relevantChunks.length >= 1
              ? 'medium'
              : 'low',
      chunksUsed: relevantChunks.length,
      // Opt-in only: the prompt context is course material the platform sells,
      // so it is not returned to ordinary clients.
      ...(coerceBoolean(debugIncludeContext) && contextUsed ? { context: contextUsed } : {})
    });
  } catch (error) {
    const statusCode = error.response?.status;
    const apiError = error.response?.data?.error?.message || '';
    console.error('RAG generation error:', { status: statusCode, message: error.message, apiError });

    // A spent quota is an operational condition, not a client error. Answering
    // 200 with an empty answer lets the caller degrade to plain chat instead of
    // showing the student a failure.
    if (statusCode === 402 || /Insufficient|quota/i.test(apiError)) {
      return res.json({
        answer: '',
        sources: [],
        confidence: 'low',
        answerMode: 'fallback',
        chunksUsed: 0,
        _error: apiError || 'AI service temporarily unavailable'
      });
    }

    res.status(500).json({ error: 'Failed to generate answer', details: error.message });
  }
});

/**
 * POST /api/rag/ask-local
 *
 * Same retrieval, generated by a locally served OpenAI-compatible model. This
 * is scenario B in the evaluation: it isolates the cost of running a small
 * self-hosted model instead of a hosted frontier one, with retrieval held
 * constant. It has no knowledge-fallback mode on purpose — a small fine-tuned
 * model outside its retrieved context is exactly the failure this system exists
 * to avoid.
 */
router.post('/ask-local', authMiddleware, async (req, res) => {
  try {
    const { question, subject = DEFAULT_SUBJECT, debugIncludeContext } = req.body;

    if (!question || question.trim().length === 0) {
      return res.status(400).json({ error: 'Question is required' });
    }

    const relevantChunks = await retrieveRelevantChunks(question, subject, TOP_K_LOCAL);

    if (relevantChunks.length === 0) {
      return res.json({
        answer: staticNoLessonMatchMessage(),
        sources: [],
        confidence: 'low',
        answerMode: 'no_match',
        chunksUsed: 0
      });
    }

    const contextForModel = buildContext(relevantChunks, 1200);
    const { systemPrompt, userPrompt } = buildRagGroundedPrompts(
      subject,
      contextForModel,
      question
    );

    const completion = await localOpenAI.chat.completions.create(
      {
        model: LOCAL_OPENAI_MODEL,
        messages: [
          { role: 'system', content: systemPrompt },
          { role: 'user', content: userPrompt }
        ],
        temperature: 0.3,
        // Kept short: latency on a single self-hosted GPU scales with output
        // length, and the grounded answer does not need to be long.
        max_tokens: Number(process.env.LOCAL_MAX_TOKENS || 350)
      },
      { timeout: Math.max(1000, LOCAL_OPENAI_TIMEOUT_MS) }
    );

    const answer = completion?.choices?.[0]?.message?.content || '';

    res.json({
      answer,
      sources: toSources(relevantChunks),
      answerMode: 'grounded',
      confidence:
        relevantChunks.length >= 3 ? 'high' : relevantChunks.length >= 1 ? 'medium' : 'low',
      chunksUsed: relevantChunks.length,
      ...(coerceBoolean(debugIncludeContext) ? { context: contextForModel } : {})
    });
  } catch (error) {
    console.error('RAG (local) error:', error.message);
    res.status(500).json({ error: 'Failed to generate answer (local)', details: error.message });
  }
});

/**
 * GET /api/rag/subjects
 * Only subjects with at least one indexed chunk, so the client never offers a
 * subject that would answer every question with `no_match`.
 */
router.get('/subjects', authMiddleware, async (req, res) => {
  try {
    const [subjects] = await pool.execute(`
      SELECT DISTINCT s.id, s.name, s.icon, s.color
      FROM subjects s
      JOIN courses c ON c.subject_id = s.id
      JOIN rag_chunks rc ON rc.course_id = c.id
      ORDER BY s.name
    `);
    res.json(subjects);
  } catch (error) {
    console.error('Error fetching RAG subjects:', error);
    res.status(500).json({ error: 'Failed to fetch subjects' });
  }
});

/**
 * GET /api/rag/stats?subject=...
 * Corpus statistics for one subject. Chunk-length distribution is the figure
 * that matters operationally: it predicts how much of the prompt budget a
 * single retrieved chunk will consume.
 */
router.get('/stats', authMiddleware, async (req, res) => {
  try {
    const { subject = DEFAULT_SUBJECT } = req.query;

    const [subjects] = await pool.execute('SELECT id FROM subjects WHERE name = ?', [subject]);
    if (subjects.length === 0) {
      return res.status(404).json({ error: 'Subject not found' });
    }

    const [stats] = await pool.execute(
      `
      SELECT
        COUNT(DISTINCT c.id) AS total_courses,
        COUNT(rc.id) AS total_chunks,
        AVG(LENGTH(rc.content)) AS avg_chunk_size,
        MIN(LENGTH(rc.content)) AS min_chunk_size,
        MAX(LENGTH(rc.content)) AS max_chunk_size
      FROM courses c
      LEFT JOIN rag_chunks rc ON rc.course_id = c.id
      WHERE c.subject_id = ?
    `,
      [subjects[0].id]
    );

    res.json({
      subject,
      ...stats[0],
      avg_chunk_size: Math.round(stats[0].avg_chunk_size || 0),
      min_chunk_size: Math.round(stats[0].min_chunk_size || 0),
      max_chunk_size: Math.round(stats[0].max_chunk_size || 0)
    });
  } catch (error) {
    console.error('Error fetching RAG stats:', error);
    res.status(500).json({ error: 'Failed to fetch statistics' });
  }
});

module.exports = { router };
