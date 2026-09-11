/**
 * Mastery-conditioned difficulty steering.
 *
 * The knowledge tracer (`dkt/`) predicts a continuous mastery score in [0, 1]
 * for a lesson section. This module turns that one number into a constraint
 * inside the prompt the RAG pipeline was going to send anyway.
 *
 * The mechanism is deliberately cheap: no second model call, no re-retrieval,
 * no re-ranking. One paragraph is added to a system prompt that is already
 * being assembled, so an adapted answer costs exactly what an unadapted one
 * costs. That is the whole reason the coupling is a prompt constraint rather
 * than a model of its own.
 *
 * Three tiers, with boundaries at 0.50 and 0.80:
 *
 *   < 0.50   foundational  scaffolded, hint-rich, partially completed work
 *   ≤ 0.80   standard      an exercise at the examination's own difficulty
 *   > 0.80   challenge     multi-step synthesis above routine questions
 *
 * The boundaries are pedagogical, not fitted: they mark "cannot yet do this
 * alone", "can do this", and "is not being stretched". They are configurable
 * because a future cohort measurement is the only thing that could justify
 * moving them, and hard-coding would make that measurement unactionable.
 *
 * **Degrading to nothing is the designed behaviour.** With no mastery score —
 * an anonymous student, a cold-start learner, an unreachable DKT service — this
 * module returns null and the prompt is byte-for-byte the unadapted one that
 * the generation study measured. Adaptation is additive, and its absence is
 * never an error.
 */

/** Below this, the learner cannot yet work unaided. */
const FOUNDATIONAL_MAX = Number(process.env.RAG_MASTERY_FOUNDATIONAL_MAX || 0.5);
/** Above this, examination-level work no longer stretches the learner. */
const CHALLENGE_MIN = Number(process.env.RAG_MASTERY_CHALLENGE_MIN || 0.8);

/**
 * Accepts a mastery score on either scale and returns it in [0, 1].
 *
 * The DKT API reports both `predicted_normalized` (0-1) and `predicted_score`
 * (0-100, the platform's grading scale). Clients send whichever they have, and
 * confusing the two would put every learner in the challenge tier — 55 read as
 * a normalised score is not 0.55, it is off the scale entirely. Anything above
 * 1 is therefore interpreted as a percentage rather than clipped, and anything
 * unparseable returns null rather than a default, because a wrong tier is worse
 * than no tier.
 *
 * @param {unknown} value
 * @returns {number|null}
 */
function normalizeMastery(value) {
  if (value === null || value === undefined || value === '') return null;
  const n = Number(value);
  if (!Number.isFinite(n) || n < 0) return null;
  if (n > 100) return null;
  const scaled = n > 1 ? n / 100 : n;
  return Math.min(1, Math.max(0, scaled));
}

const TIERS = {
  foundational: {
    tier: 'foundational',
    label: 'تأسيسي',
    instruction: `مستوى الطالب في هذا الجزء ضعيف (إتقان مُقدَّر: {PCT}%). لذلك:
- ابدأ بتذكير مختصر بالمفهوم أو القاعدة قبل الإجابة.
- فصّل الخطوات واحدة واحدة، ولا تدمج خطوتين في سطر واحد.
- اشرح المصطلحات التقنية عند أول ورود لها.
- إن طُلب تمرين، اجعله تطبيقاً مباشراً مع تلميحات وبداية حل مكتملة جزئياً.
- تجنّب التوسع في الحالات الخاصة والاستثناءات في هذه المرحلة.`
  },
  standard: {
    tier: 'standard',
    label: 'معياري',
    instruction: `مستوى الطالب في هذا الجزء متوسط (إتقان مُقدَّر: {PCT}%). لذلك:
- اكتب الإجابة بصيغة موضوع البكالوريا المعتادة ومستوى صعوبتها.
- اذكر الخطوات الأساسية دون إفراط في التبسيط ودون حذف مبرّرات الانتقال بينها.
- إن طُلب تمرين، اجعله في مستوى سؤال امتحان عادي.`
  },
  challenge: {
    tier: 'challenge',
    label: 'تحدٍّ',
    instruction: `مستوى الطالب في هذا الجزء متقدّم (إتقان مُقدَّر: {PCT}%). لذلك:
- اختصر التعريفات الأساسية وانتقل سريعاً إلى جوهر السؤال.
- اربط بين أكثر من عنصر من المقرر، وأبرز الحالات الحدّية ومواضع فقدان النقاط في التصحيح.
- إن طُلب تمرين، اجعله سؤالاً تركيبياً متعدد الخطوات في مستوى أصعب مواضيع البكالوريا.
- لا تُبسّط ما لا يحتاج تبسيطاً؛ الإسهاب هنا يُضيّع وقت الطالب.`
  }
};

/**
 * Map a mastery score to a tier and the prompt paragraph that expresses it.
 *
 * @param {unknown} mastery mastery in [0, 1] or a 0-100 percentage
 * @returns {{ tier: string, label: string, mastery: number, instruction: string }|null}
 */
function resolveDifficultyTier(mastery) {
  const value = normalizeMastery(mastery);
  if (value === null) return null;

  const chosen =
    value < FOUNDATIONAL_MAX
      ? TIERS.foundational
      : value > CHALLENGE_MIN
        ? TIERS.challenge
        : TIERS.standard;

  return {
    tier: chosen.tier,
    label: chosen.label,
    mastery: Number(value.toFixed(4)),
    // The percentage is stated to the model rather than only the tier name.
    // A tier label alone flattens 0.05 and 0.49 into the same instruction; the
    // number lets the model modulate within the band it was given.
    instruction: chosen.instruction.replace('{PCT}', String(Math.round(value * 100)))
  };
}

module.exports = {
  resolveDifficultyTier,
  normalizeMastery,
  FOUNDATIONAL_MAX,
  CHALLENGE_MIN
};
