/**
 * Generator (LLM) settings.
 *
 * The generator is reached over the OpenAI-compatible chat-completions
 * protocol, so any provider exposing that shape can be swapped in by changing
 * `LLM_API_URL` and `LLM_CHAT_MODEL` alone. The reference deployment uses
 * DeepSeek for the hosted path and a locally served fine-tuned ALLaM model for
 * the on-premise path.
 *
 * No key is ever hardcoded: an absent key is reported at startup rather than
 * silently replaced by a default.
 */

const dotenv = require('dotenv');
const path = require('path');

dotenv.config();
dotenv.config({ path: path.join(__dirname, '..', '..', '.env'), override: false });

const LLM_API_KEY = process.env.LLM_API_KEY || process.env.DEEPSEEK_API_KEY || '';
const LLM_API_URL =
  process.env.LLM_API_URL ||
  process.env.DEEPSEEK_API_URL ||
  'https://api.deepseek.com/v1/chat/completions';

/** Model used to write the final grounded answer. */
const LLM_CHAT_MODEL = process.env.LLM_CHAT_MODEL || process.env.DEEPSEEK_CHAT_MODEL || 'deepseek-chat';

/** Stronger model used only offline, as the LLM-as-judge in the evaluation harness. */
const LLM_JUDGE_MODEL = process.env.LLM_JUDGE_MODEL || 'deepseek-v4-pro';

/** Wall-clock budget for one generation call, in milliseconds. */
const LLM_TIMEOUT_MS = Number(process.env.LLM_TIMEOUT_MS || 120000);

/**
 * Local OpenAI-compatible server (vLLM, llama.cpp, TGI, …) used by the
 * `/api/rag/ask-local` variant. This is the self-hosted arm of the generator
 * comparison: same retrieval, a small local model instead of a hosted one.
 */
const LOCAL_OPENAI_BASE_URL = process.env.LOCAL_OPENAI_BASE_URL || 'http://localhost:8000/v1';
const LOCAL_OPENAI_MODEL = process.env.LOCAL_OPENAI_MODEL || 'local-model';
const LOCAL_OPENAI_API_KEY = process.env.LOCAL_OPENAI_API_KEY || 'local-api-key';
const LOCAL_OPENAI_TIMEOUT_MS = Number(process.env.LOCAL_OPENAI_TIMEOUT_MS || 120000);

function isHostedGeneratorConfigured() {
  return Boolean(LLM_API_KEY && LLM_API_URL);
}

module.exports = {
  LLM_API_KEY,
  LLM_API_URL,
  LLM_CHAT_MODEL,
  LLM_JUDGE_MODEL,
  LLM_TIMEOUT_MS,
  LOCAL_OPENAI_BASE_URL,
  LOCAL_OPENAI_MODEL,
  LOCAL_OPENAI_API_KEY,
  LOCAL_OPENAI_TIMEOUT_MS,
  isHostedGeneratorConfigured
};
