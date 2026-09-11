/**
 * One grounded chat completion, sharing the exact branching of POST /api/rag/ask
 * (grounded vs knowledge_fallback vs static no_match).
 *
 * It exists as a separate module so the offline evaluation exporter produces
 * answers under literally the same prompts as the live endpoint. If the two ever
 * drifted, every measured number would describe a system nobody is serving.
 */

const axios = require('axios');
const { LLM_API_KEY, LLM_API_URL, LLM_CHAT_MODEL, LLM_TIMEOUT_MS } = require('../config/llm');
const {
  isChunkEvidenceWeak,
  isLikelyAlgerianBacQuestion,
  buildRagGroundedPrompts,
  buildRagKnowledgeFallbackPrompts,
  staticNoLessonMatchMessage
} = require('./ragAnswerModes');

function truncateForPrompt(text, maxChars) {
  const s = String(text || '');
  const n = Number(maxChars);
  if (!Number.isFinite(n) || n <= 0) return s;
  if (s.length <= n) return s;
  return `${s.slice(0, n)}\n\n[…تم اقتصار النص للسرعة…]`;
}

function buildGroundedContext(relevantChunks) {
  return relevantChunks
    .map(
      (chunk, i) =>
        `[مصدر ${i + 1}: ${chunk.course_title} - ${chunk.section_title}]\n${truncateForPrompt(chunk.content, 1400)}`
    )
    .join('\n\n---\n\n');
}

/**
 * @param {string} question
 * @param {string} subject
 * @param {Array<object>} relevantChunks rows from retrieveRelevantChunks
 * @returns {Promise<{ answer: string, model: string, context_used_for_prompt: string|null, answerMode: string }>}
 */
async function generateRagAnswer(question, subject, relevantChunks) {
  const weakEvidence = isChunkEvidenceWeak(relevantChunks);
  const bacLikely = isLikelyAlgerianBacQuestion(question, subject);

  let systemPrompt;
  let userPrompt;
  let answerMode = 'grounded';
  let temperature = 0.3;
  let contextUsed = null;

  if (!weakEvidence && relevantChunks.length > 0) {
    contextUsed = buildGroundedContext(relevantChunks);
    ({ systemPrompt, userPrompt } = buildRagGroundedPrompts(subject, contextUsed, question));
  } else if (bacLikely) {
    answerMode = 'knowledge_fallback';
    temperature = 0.45;
    contextUsed =
      relevantChunks.length > 0
        ? relevantChunks
            .map(
              (chunk, i) =>
                `[مصدر ${i + 1}: ${chunk.course_title} - ${chunk.section_title}] (درجة الارتباط: ${chunk.score})\n${truncateForPrompt(chunk.content, 900)}`
            )
            .join('\n\n---\n\n')
        : '';
    ({ systemPrompt, userPrompt } = buildRagKnowledgeFallbackPrompts(subject, question, contextUsed));
  } else {
    return {
      answer: staticNoLessonMatchMessage(),
      model: LLM_CHAT_MODEL,
      context_used_for_prompt: null,
      answerMode: 'no_match'
    };
  }

  const aiResponse = await axios.post(
    LLM_API_URL,
    {
      model: LLM_CHAT_MODEL,
      messages: [
        { role: 'system', content: systemPrompt },
        { role: 'user', content: userPrompt }
      ],
      max_tokens: 1500,
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
  return {
    answer,
    model: LLM_CHAT_MODEL,
    context_used_for_prompt: answerMode === 'grounded' ? contextUsed : contextUsed || null,
    answerMode
  };
}

module.exports = {
  generateRagAnswer,
  buildGroundedContext,
  truncateForPrompt
};
