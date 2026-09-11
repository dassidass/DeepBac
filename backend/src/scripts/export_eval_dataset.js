/**
 * Build the evaluation export: for every question in a JSONL test set, record
 * what the retriever returned and what the grounded generator answered.
 *
 *   node src/scripts/export_eval_dataset.js --dataset ./eval/test_dataset.jsonl \
 *                                           --out ./eval/rag_export.jsonl
 *   node src/scripts/export_eval_dataset.js --skip-generation   retrieval only
 *
 * Input lines: { "question": "...", "answer": "..." }  (`answer` is the gold answer)
 *
 * Output lines carry the question, the gold answer, every retrieved chunk with
 * its score, and the generated answer with the mode that produced it. That is
 * the file a judge model scores: keeping retrieval and generation in one record
 * is what allows faithfulness to be judged against the exact context the model
 * actually saw, rather than against a reconstruction of it.
 */

const fs = require('fs');
const path = require('path');
const { pool } = require('../config/database');
const { retrieveRelevantChunks } = require('../services/ragRetrieval');
const { generateRagAnswer } = require('../services/ragAnswer');

const argv = process.argv.slice(2);
const argValue = (name, fallback = null) => {
  const i = argv.indexOf(name);
  if (i === -1) return fallback;
  const v = argv[i + 1];
  return v == null || v.startsWith('--') ? fallback : v;
};

const DATASET_PATH = path.resolve(process.cwd(), argValue('--dataset', './eval/test_dataset.jsonl'));
const OUT_PATH = path.resolve(process.cwd(), argValue('--out', './eval/rag_export.jsonl'));
const SUBJECT = process.env.SUBJECT_NAME || argValue('--subject') || 'التاريخ والجغرافيا';
const CHUNK_LIMIT = Number(process.env.CHUNK_LIMIT || argValue('--chunk-limit') || 5);
const LIMIT = Number(argValue('--limit', '0')) || 0;
const SKIP_GENERATION = argv.includes('--skip-generation');

/** Chunk content is kept in full: faithfulness cannot be judged against an excerpt. */
const serializeChunks = (chunks) =>
  chunks.map((c) => ({
    id: c.id,
    course_id: c.course_id,
    course_title: c.course_title,
    section_title: c.section_title,
    category: c.category,
    relevanceScore: c.score,
    content: c.content
  }));

function readDataset(file) {
  return fs
    .readFileSync(file, 'utf8')
    .split(/\r?\n/)
    .filter((l) => l.trim() !== '')
    .map((l, i) => {
      try {
        return JSON.parse(l);
      } catch {
        console.warn(`[export] line ${i + 1} is not valid JSON — skipped`);
        return null;
      }
    })
    .filter(Boolean);
}

async function main() {
  if (!pool) throw new Error('Database pool unavailable — check DB_* variables in .env');
  if (!fs.existsSync(DATASET_PATH)) throw new Error(`Dataset not found: ${DATASET_PATH}`);

  const rows = readDataset(DATASET_PATH);
  const total = LIMIT > 0 ? Math.min(LIMIT, rows.length) : rows.length;
  fs.mkdirSync(path.dirname(OUT_PATH), { recursive: true });
  const out = fs.createWriteStream(OUT_PATH, { encoding: 'utf8' });

  console.log(`[export] ${total} question(s) · subject "${SUBJECT}" · top-${CHUNK_LIMIT}`);

  for (let i = 0; i < total; i += 1) {
    const row = rows[i];
    const question = String(row.question || '').trim();
    if (!question) continue;

    const started = Date.now();
    const chunks = await retrieveRelevantChunks(question, SUBJECT, CHUNK_LIMIT);
    const retrievalMs = Date.now() - started;

    /** @type {{answer: string, model: string|null, answerMode: string, context_used_for_prompt: string|null}} */
    let generated = {
      answer: '',
      model: null,
      answerMode: 'skipped',
      context_used_for_prompt: null
    };
    let generationMs = null;

    if (!SKIP_GENERATION) {
      const t = Date.now();
      try {
        generated = await generateRagAnswer(question, SUBJECT, chunks);
      } catch (e) {
        // A failed generation must not abandon the run: the retrieval record is
        // still valid, and the error is kept so the row can be excluded or retried.
        generated = {
          answer: '',
          model: null,
          answerMode: 'error',
          context_used_for_prompt: null,
          error: e.message
        };
      }
      generationMs = Date.now() - t;
    }

    out.write(
      `${JSON.stringify({
        question,
        gold_answer: row.answer ?? null,
        subject: SUBJECT,
        retrieved_chunks: serializeChunks(chunks),
        chunks_used: chunks.length,
        retrieval_ms: retrievalMs,
        generation_ms: generationMs,
        ...generated
      })}\n`
    );

    if ((i + 1) % 10 === 0) console.log(`[export] ${i + 1}/${total}`);
  }

  out.end();
  console.log(`[export] wrote ${OUT_PATH}`);
}

if (require.main === module) {
  main()
    .catch((e) => {
      console.error('[export] failed:', e.message);
      process.exitCode = 1;
    })
    .finally(() => pool && pool.end().catch(() => {}));
}

module.exports = {};
