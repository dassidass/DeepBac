/**
 * Prompt construction and answer-mode routing.
 *
 * Retrieval quality is decided here, before generation, because the three
 * outcomes need three different prompts:
 *
 *   grounded            the model may use only the retrieved text
 *   knowledge_fallback  retrieval failed but the question is curricular, so the
 *                       model answers from its own knowledge and must say so
 *   no_match            neither applies, and nothing is generated
 *
 * All prompt text is Arabic because the corpus, the students and the expected
 * answers are Arabic. Instructing an Arabic answer in English measurably
 * increases code-switching in the output.
 */

/** Non–Bac-Algeria chat / tech noise — conservative list */
const OFF_TOPIC_HINT =
  /(^|\s)(docker|kubernetes|react\.?js|\bnpm\b|\byarn\b|\bgithub\b|\bgitlab\b|^hello\b|^hi\b|^hey\b)(\s|$)/i;

/**
 * Decides whether retrieval produced usable evidence.
 *
 * The two retrievers return scores on incompatible scales, and this function is
 * the single place that reconciles them:
 *   · keyword overlap returns integers, and its own floor already rejects
 *     anything below 2 — so a score >= 2 means the keyword path accepted it;
 *   · Qdrant cosine similarity lands in (0, 1), where a top hit below roughly
 *     0.42 has, in this corpus, consistently been a different lesson that
 *     merely shares vocabulary.
 *
 * Only the top hit is examined. If the best chunk is off-topic, the tail is
 * necessarily worse.
 */
function isChunkEvidenceWeak(chunks) {
  if (!Array.isArray(chunks) || chunks.length === 0) return true;
  const best = chunks[0].score;
  if (typeof best !== 'number' || !Number.isFinite(best)) return false;
  if (best >= 2) return false;
  const t = Number(process.env.RAG_SEMANTIC_WEAK_TOP_SCORE || '0.42');
  return best < t;
}

/**
 * Decides whether a question is plausibly curricular, which gates the
 * knowledge-fallback mode.
 *
 * The test is deliberately permissive. A false negative refuses a legitimate
 * student question and is visible immediately; a false positive only produces
 * an answer that is explicitly labelled as coming from outside the lessons.
 * The caller has already supplied a subject drawn from the database, which is
 * itself strong evidence, so the remaining checks exist mainly to reject small
 * talk and developer-tooling chatter.
 */
function isLikelyAlgerianBacQuestion(question, subject = '') {
  const q = String(question || '').trim();
  if (q.length < 6) return false;
  if (OFF_TOPIC_HINT.test(q)) return false;

  const bacHints =
    /بكالوريا|الباك|bac\b|البكالوريا|شهادة البكالوريا|البكالوريا الجزائرية|امتحان.*وطني|المنهج|الجزائر|شعبة|الجذع المشترك|علوم تجريبية|تقني رياضي|آداب وفلسفة|تسيير واقتصاد|لغات أجنبية|رياضيات|السنة الأولى|السنة الثانية|ثانوي|التعليم الثانوي|فرض|اختبار|موضوع الامتحان|نصوص الأدبية|الاستيعاب المكتوب|الاستيعاب الشفوي/i;
  if (bacHints.test(q)) return true;

  if (subject && String(subject).trim().length > 1) return true;

  if (/[\u0600-\u06FF]{12,}/.test(q)) return true;

  return q.length >= 28;
}

const LATEX_RULES = `للمعادلات الرياضية والفيزيائية استخدم LaTeX داخل Markdown: داخل السطر بـ $...$، ومعادلات منفصلة بـ $$...$$. لا تستخدم الأقواس المربعة [ ] وحدها لعرض المعادلات. اكتب الفهارس السفلية بشكل صريح مثل $z_{1}$ و $z_{2}$، ومجموعة الأعداد المركبة كـ $\\mathbb{C}$، والمميز كـ $\\Delta$، والكسور بـ \\frac. للصيغ والتفاعلات الكيميائية استخدم \\ce{...} داخل التعبير الرياضي، مثال: $\\ce{H2SO4}$ أو $$\\ce{2Na + Cl2 -> 2NaCl}$$`;

const BAC_METHODOLOGY = `منهجية الإجابة (مستوى البكالوريا الجزائرية) — طبّق ما يناسب نوع السؤال فقط:
- تعريف / مفهوم: تعريف دقيق مختصر ثم بسطه بجمل واضحة دون حشو.
- سؤال «لماذا / كيف / ما الأثر»: أسباب ثم نتائج أو مراحل بترقيم منطقي.
- تاريخ وجغرافيا: التسلسل الزمني أو المكاني عند الحاجة، مع ربط الأحداث بالسياق.
- نص أدبي / تعبير: مقدمة موجزة، عرض منظم (أفكار مترابطة)، خاتمة تقييمية قصيرة عند المطلوب.
- علوم: استخدم القوانين والتعريفات الرسمية، خطوات الحل إن كان سؤالاً حسابياً، والوحدات عند اللزوم.
- لا تتجاوز عمقاً يفوق مستوى البكالوريا؛ إن غابت معلومة دقيقة (تاريخ إصدار، رقم درس في الكتاب)، صرّح بالتحفظ بدل الاختلاق.`;

/**
 * Optional prompt layers, appended only when the caller supplies them.
 *
 * Each returns an empty string when its input is absent, so a request with no
 * mastery score and no detected register produces exactly the unadapted prompt.
 * Adaptation is strictly additive; nothing here can remove or weaken a grounding
 * rule above it.
 *
 * @param {{ difficulty?: {label: string, instruction: string}|null, languageInstruction?: string }} options
 */
function buildAdaptiveLayers(options = {}) {
  const difficulty = options.difficulty
    ? `\n\nتكييف مستوى الإجابة (${options.difficulty.label}):\n${options.difficulty.instruction}`
    : '';
  const language = options.languageInstruction
    ? `\n\nلغة الإجابة:\n${options.languageInstruction}`
    : '';
  return { difficulty, language };
}

/**
 * The grounded prompt: the model may use the retrieved text and nothing else.
 *
 * Layer order is deliberate and load-bearing. The grounding rules come first
 * and the adaptive layers last, so that a difficulty instruction can change how
 * an answer is *presented* but never what it is allowed to be drawn from. A
 * foundational learner gets more scaffolding around the same retrieved facts,
 * never invented ones to fill a gap the corpus left.
 *
 * @param {string} subject
 * @param {string} context retrieved chunks, already rendered and truncated
 * @param {string} question
 * @param {{ difficulty?: object|null, languageInstruction?: string }} [options]
 */
function buildRagGroundedPrompts(subject, context, question, options = {}) {
  const { difficulty, language } = buildAdaptiveLayers(options);

  const systemPrompt = `أنت مساعد تعليمي متخصص في مساعدة طلاب البكالوريا الجزائرية في مادة ${subject}.

مهمتك هي الإجابة على أسئلة الطلاب بناءً على محتوى الدروس المتاحة فقط.

قواعد الإجابة:
1. استخدم فقط المعلومات الموجودة في السياق المعطى؛ لا تستعين بمعرفة عامة خارج النصوص المعروضة.
2. إذا كان السياق لا يتعلق مباشرة بالسؤال (مثلاً السؤال عن موضوع والمقاطع عن موضوع آخر)، قل بوضوح: «لا يوجد في الدروس المعروضة جواباً كافياً عن هذا السؤال» ولا تخمّن.
3. لا تخلط بين موضوعين مختلفين في إجابة واحدة إلا إذا كان السياق يدعم ذلك صراحة.
4. قدم إجابات واضحة ومنظمة وسهلة الفهم.
5. استخدم اللغة العربية بشكل أساسي.
6. إذا لم تكن المعلومات كافية، اذكر ذلك بوضوح.
7. رتب الإجابة بشكل منطقي مع استخدام النقاط والترقيم عند الحاجة.
8. أشر إلى المصادر بذكر عنوان الدرس أو القسم كما ورد في السياق حرفياً، ولا تنسب أي معلومة إلى عنوان غير موجود في السياق.
9. ${LATEX_RULES}

${BAC_METHODOLOGY}${difficulty}${language}

السياق من الدروس:
${context}`;

  const userPrompt = `السؤال: ${question}

أجب عن هذا السؤال فقط. إذا لم يكن السياق أعلاه يجيب عن السؤال مباشرة، قل إن الدروس المعروضة لا تتضمن جواباً كافياً ولا تنتقل إلى موضوع آخر.`;

  return { systemPrompt, userPrompt };
}

/**
 * When retrieval is empty or weak: answer from model knowledge + Bac methodology.
 * @param {string} optionalWeakContext - truncated chunk text for light hints only (may be empty)
 * @param {{ difficulty?: object|null, languageInstruction?: string }} [options]
 */
function buildRagKnowledgeFallbackPrompts(subject, question, optionalWeakContext, options = {}) {
  const { difficulty, language } = buildAdaptiveLayers(options);
  const weakBlock =
    optionalWeakContext && String(optionalWeakContext).trim().length > 0
      ? `\n\nمقاطع مسترجعة ذات ارتباط ضعيف (لا تُلزمك؛ لا تنسخها إن لم تكن مفيدة):\n${optionalWeakContext}`
      : '';

  const systemPrompt = `أنت مساعد تعليمي لطلاب البكالوريا في الجزائر في مادة ${subject}.

وضع الاسترجاع (RAG): لم تُسترجع من دروس المنصة مقاطع كافية أو ذات صلة قوية بالسؤال. السؤال يبدو مرتبطاً بالبكالوريا الجزائرية.

التزم بما يلي:
1. افتتح بجملة واحدة واضحة: أن الدروس المتاحة على المنصة لم توفّر نصاً كافياً، وأنك تكمّل من معرفتك بالمنهاج الجزائري للبكالوريا مع التحفظ من اختلاف التفاصيل بين الشعب أو السنوات.
2. أجب بمعرفتك العامة بالمادة على مستوى البكالوريا في الجزائر؛ لا تُدخل معلومات تخص مناهج دول أخرى إلا للمقارنة المختصرة عند الحاجة.
3. ${BAC_METHODOLOGY}
4. ${LATEX_RULES}
5. إن كان السؤال خارج المنهاج أو غير أكاديمي، ارفض الإجابة بلطف.${difficulty}${language}${weakBlock}`;

  const userPrompt = `السؤال: ${question}

قدّم إجابة كاملة مناسبة لامتحان البكالوريا الجزائرية في هذه المادة.`;

  return { systemPrompt, userPrompt };
}

function staticNoLessonMatchMessage() {
  return 'عذراً، لم أجد معلومات كافية في الدروس المتاحة للإجابة على سؤالك، والسؤال لا يبدو مرتبطاً بوضوح بمقرر البكالوريا الجزائرية في هذه المادة. جرّب صياغة أخرى أو اختر المادة المناسبة من القائمة.';
}

module.exports = {
  isChunkEvidenceWeak,
  isLikelyAlgerianBacQuestion,
  buildRagGroundedPrompts,
  buildRagKnowledgeFallbackPrompts,
  buildAdaptiveLayers,
  staticNoLessonMatchMessage
};
