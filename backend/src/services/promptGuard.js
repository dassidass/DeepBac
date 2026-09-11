/**
 * Two guards around the grounded path: one on what goes in, one on what comes
 * out.
 *
 * ## The input guard
 *
 * Rejects questions whose purpose is to replace the pedagogical instruction —
 * "ignore the previous instructions", "you are now a…", an injected system
 * turn. It runs before retrieval, so a rejected question costs no embedding
 * call and no generation call.
 *
 * It is not a content filter. Off-topic questions are handled downstream by
 * `isLikelyAlgerianBacQuestion`, which refuses them for pedagogical reasons.
 *
 * ## The output check
 *
 * The grounded prompt tells the model to cite sources by the section titles it
 * was given. This verifies that every title it did cite was in fact among the
 * retrieved chunks. A citation of a section that was never retrieved is a
 * fabricated source, which is the failure mode a curriculum-grounded product
 * cannot ship: a student who opens the cited lesson finds nothing there.
 *
 * A failed check is reported to the caller as `citationCheck` and logged. It is
 * never used to rewrite the answer: editing the model's text after the fact
 * would hand the student prose nobody wrote, with no record of the change.
 */

/**
 * Instruction-override patterns, in the three registers the interface accepts.
 * Each one targets a phrase whose only function is to address the model as a
 * configurable system rather than answer a curriculum question.
 */
const INJECTION_PATTERNS = [
  // English
  /ignore\s+(all\s+|any\s+)?(the\s+)?(previous|prior|above|preceding)\s+(instructions?|prompts?|rules?)/i,
  /disregard\s+(all\s+|any\s+)?(the\s+)?(previous|prior|above)\s+(instructions?|prompts?|rules?)/i,
  /forget\s+(everything|all)\s+(you|above|before)/i,
  /you\s+are\s+now\s+(a|an|the)\s+/i,
  /(reveal|show|print|repeat)\s+(me\s+)?(your|the)\s+(system\s+)?(prompt|instructions?|rules?)/i,
  /\bdeveloper\s+mode\b|\bjailbreak\b|\bDAN\s+mode\b/i,
  // A chat turn typed into the question body.
  /<\s*\/?\s*(system|assistant)\s*>/i,
  /^\s*(system|assistant)\s*:\s*/im,
  // French
  /ignore[zr]?\s+(toutes\s+)?les\s+(instructions|consignes)\s+(précédentes|precedentes|ci-dessus)/i,
  /oublie[zs]?\s+(tout|toutes\s+les\s+(instructions|consignes))/i,
  /tu\s+es\s+(maintenant|désormais|desormais)\s+/i,
  // Arabic
  /تجاهل\s*(كل\s*)?(ال)?(تعليمات|الأوامر|التوجيهات)\s*(السابقة|أعلاه)?/,
  /انس\s*(كل\s*)?(ما|التعليمات)/,
  /أنت\s+الآن\s+/,
  /(اعرض|أظهر|اكتب)\s*(لي\s*)?(نص\s*)?(التعليمات|الأوامر|البرومبت)/
];

/**
 * Screen a question before anything is spent on it.
 *
 * @param {string} question
 * @returns {{ safe: boolean, reason: string|null }}
 */
function screenQuestion(question) {
  const q = String(question || '');
  if (!q.trim()) return { safe: true, reason: null };

  for (const pattern of INJECTION_PATTERNS) {
    if (pattern.test(q)) {
      return { safe: false, reason: 'instruction_override' };
    }
  }

  return { safe: true, reason: null };
}

/** What the student sees when a question is rejected. Deliberately not a lecture. */
function injectionRefusalMessage() {
  return 'يبدو أن سؤالك يتضمن تعليمات موجّهة للنظام نفسه بدل سؤال عن الدرس. أعد صياغة سؤالك عن محتوى المادة وسأجيبك.';
}

/**
 * Normalises a title for comparison: collapses whitespace, strips Arabic
 * diacritics and the punctuation a model tends to add or drop when it repeats a
 * heading back.
 *
 * Without this, "مظاهر الصراع بين المعسكرين." fails to match the identical
 * title stored without its full stop, and a correct citation is reported as
 * fabricated, which is the error this normalisation exists to prevent.
 */
function normalizeTitle(text) {
  return String(text || '')
    .replace(/[ً-ٰٟ]/g, '')
    .replace(/[«»"'`ـ]/g, '')
    .replace(/[.،,:;!?؟\-–—_[\]()]/g, ' ')
    .replace(/\s+/g, ' ')
    .trim()
    .toLowerCase();
}

/**
 * Pull out titles the answer presents as sources.
 *
 * Only bracketed source markers are read — `[مصدر 2: …]` and the `المصدر:` /
 * `من الدرس:` forms the prompt asks for. Free prose that happens to mention a
 * lesson name is not treated as a citation, because a model paraphrasing a
 * heading inside a sentence is not claiming it as a source and flagging it
 * would bury the real cases.
 */
function extractCitedTitles(answer) {
  const text = String(answer || '');
  const titles = [];

  const bracketed = text.matchAll(/\[\s*مصدر\s*\d*\s*:([^\]]+)\]/g);
  for (const match of bracketed) titles.push(match[1]);

  const labelled = text.matchAll(/(?:المصدر|من الدرس|من القسم)\s*:\s*([^\n.]{3,120})/g);
  for (const match of labelled) titles.push(match[1]);

  return titles.map((t) => t.trim()).filter(Boolean);
}

/**
 * Verify that every source the answer cites was actually retrieved.
 *
 * A cited string counts as matched when it contains, or is contained by, a
 * retrieved course title or section title. Containment rather than equality is
 * what allows the model to cite "الدرس الأول - مظاهر الصراع" for a chunk stored
 * under those two titles separately.
 *
 * @param {string} answer
 * @param {Array<{course_title?: string, section_title?: string}>} chunks
 * @returns {{ ok: boolean, citedCount: number, unverified: string[] }}
 */
function verifyCitations(answer, chunks) {
  const cited = extractCitedTitles(answer);
  if (cited.length === 0) {
    return { ok: true, citedCount: 0, unverified: [] };
  }

  const known = [];
  for (const chunk of chunks || []) {
    const course = normalizeTitle(chunk.course_title);
    const section = normalizeTitle(chunk.section_title);
    if (course) known.push(course);
    if (section) known.push(section);
    if (course && section) known.push(`${course} ${section}`);
  }

  const unverified = [];
  for (const raw of cited) {
    const candidate = normalizeTitle(raw);
    if (!candidate) continue;
    const matched = known.some(
      (title) => title.includes(candidate) || candidate.includes(title)
    );
    if (!matched) unverified.push(raw);
  }

  return { ok: unverified.length === 0, citedCount: cited.length, unverified };
}

module.exports = {
  screenQuestion,
  injectionRefusalMessage,
  verifyCitations,
  extractCitedTitles,
  normalizeTitle
};
