/**
 * Which register the student wrote in, and what to tell the model about it.
 *
 * Retrieval is already multilingual and needs none of this: the embedding model
 * maps Modern Standard Arabic, French and Algerian Darja into one vector space,
 * so a Darja question retrieves formal Arabic lessons with no translation step.
 * Generation is where the registers stop being interchangeable. A student who
 * writes "chnou hiya la crise de 1929" is not asking for an answer in Darja —
 * Darja is not a written examination language, and an answer in it would be
 * unusable for revision — but answering them in unmarked formal Arabic, when
 * they typed half their question in French, loses the technical vocabulary they
 * actually study in.
 *
 * So the rule this module encodes is not "reply in the language you were given".
 * It is: the body of the answer stays in the examination's own language, and the
 * register only decides how much of the student's vocabulary is carried along.
 *
 * Detection is a script-and-lexicon heuristic, not a classifier. It is
 * deliberately biased toward `msa`, the safe default: a missed Darja question
 * still gets a correct answer in the language the examination is written in,
 * whereas a false Darja positive on a formal question makes the answer read as
 * unserious. Any question this gets wrong is wrong only in tone.
 */

/**
 * Darja markers that cannot be Modern Standard Arabic. Function words and
 * question words only: content words are shared across registers, and matching
 * on them would classify most formal questions as dialect.
 */
const DARJA_MARKERS =
  /(^|\s)(كيفاش|واش|وشنو|شنو|علاش|وقتاش|قداه|كيما|بزاف|شوية|راني|راهو|راها|راك|نحب|تحب|يحب|باش|بالاك|خويا|درك|دروك|هاكا|هكذاك|ماشي|مكانش|كاين|كاينة|نقدر|تقدرش|عندي|عندك|ندير|تدير|يدير|حاب|صح|برك|ثمة)(\s|$)/;

/** The same, typed in Latin script — how Darja is usually written on a phone. */
const DARJA_LATIN_MARKERS =
  /(^|\s)(kifech|kifache|wach|wech|chnou|chno|3lach|3lah|wa9tach|9adah|kima|bezaf|chwiya|rani|rahou|raki|nheb|theb|bach|khoya|derk|daba|makanch|kayen|ndir|tdir|sah|barka)(\s|$)/i;

/** French function words. Content words like "crise" appear in Arabic text too. */
const FRENCH_MARKERS =
  /(^|\s)(le|la|les|un|une|des|du|de|c'est|qu'est|quoi|pourquoi|comment|quelle?|quels?|est-ce|dans|pour|avec|sur|entre|expliquer?|explique|donne|donner|résume|resume|exercice|cours|leçon|lecon)(\s|$)/i;

const ARABIC_SCRIPT = /[؀-ۿ]/g;
const LATIN_SCRIPT = /[A-Za-z]/g;

/**
 * @typedef {'msa'|'french'|'darja'|'mixed'} Register
 */

/**
 * Classify one question.
 *
 * `mixed` is a first-class outcome, not an error state: code-switching between
 * Darja and French inside a single sentence is the ordinary way Algerian
 * students type, and collapsing it into one of the pure registers would discard
 * the fact that they used both.
 *
 * @param {string} question
 * @returns {Register}
 */
function detectRegister(question) {
  const q = String(question || '').trim();
  if (!q) return 'msa';

  const arabicChars = (q.match(ARABIC_SCRIPT) || []).length;
  const latinChars = (q.match(LATIN_SCRIPT) || []).length;
  const total = arabicChars + latinChars;
  if (total === 0) return 'msa';

  const arabicShare = arabicChars / total;

  const hasDarja = DARJA_MARKERS.test(q) || DARJA_LATIN_MARKERS.test(q);
  const hasFrench = FRENCH_MARKERS.test(q);

  // Both scripts in meaningful proportion is code-switching by definition,
  // whatever the lexicons find.
  if (arabicShare > 0.15 && arabicShare < 0.85) return 'mixed';

  if (arabicShare >= 0.85) {
    return hasDarja ? 'darja' : 'msa';
  }

  // Latin script: French unless the words are transliterated Darja.
  if (hasDarja) return hasFrench ? 'mixed' : 'darja';
  return hasFrench ? 'french' : 'msa';
}

const INSTRUCTIONS = {
  msa: 'اكتب الإجابة بالعربية الفصحى المستعملة في مواضيع البكالوريا.',
  darja:
    'صاغ الطالب سؤاله بالدارجة الجزائرية. افهم السؤال كما هو، ثم أجب بالعربية الفصحى ' +
    'لأنها لغة الامتحان، بأسلوب مباشر وبسيط. لا تكتب الإجابة بالدارجة، ولا تعلّق على لغة السؤال.',
  french:
    "Réponds en français, la langue dans laquelle l'élève a posé sa question et dans " +
    'laquelle cette matière est évaluée. Garde la terminologie officielle du programme ' +
    'algérien, et conserve les termes arabes des sources cités tels quels.',
  mixed:
    'خلط الطالب بين العربية والفرنسية في سؤاله، وهو أمر معتاد. أجب بلغة المادة كما تُمتحَن: ' +
    'العربية الفصحى للمواد المُدرَّسة بالعربية، والفرنسية للمواد المُدرَّسة بالفرنسية. ' +
    'أبقِ المصطلحات التقنية بلغتها الأصلية كما وردت في السياق، وأضف مقابلها بين قوسين عند أول ورود.'
};

/**
 * The prompt line for a register, or an empty string for the default.
 *
 * `msa` returns nothing on purpose. Adding "answer in formal Arabic" to a prompt
 * that is already written entirely in formal Arabic spends context on an
 * instruction the model was going to follow anyway.
 *
 * @param {Register} register
 * @returns {string}
 */
function registerInstruction(register) {
  if (register === 'msa') return '';
  return INSTRUCTIONS[register] || '';
}

module.exports = {
  detectRegister,
  registerInstruction
};
