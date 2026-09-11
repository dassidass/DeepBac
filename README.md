# DeepBac

**A curriculum-grounded Intelligent Tutoring System for the Algerian Baccalaureate.**

This repository accompanies the paper on DeepBac. It holds the datasets, model weights, evaluation
scripts and backend services behind the results reported there.

General-purpose large language models fail in high-stakes localized examination contexts: they
hallucinate facts, are unaware of local curriculum constraints, and handle naturalistic trilingual
student input poorly. DeepBac addresses all three for the Algerian Baccalaureate (BAC) — a
centralized national examination taken by over 800,000 candidates across 69 wilayas, graded against
the *Corrigés Types*, the official rubrics published by the Ministry of National Education that
define not only the correct answer but the exact stepwise methodology expected of students.

The system is deployed at **[deepbac.com](https://deepbac.com)**.

## Three mechanisms

**1. Curriculum-grounded retrieval over a dual-database backend.** MySQL is the system of record: it
holds the four-level curriculum hierarchy (Subject → Chapter → Lesson → Lesson-Element) together
with the full pedagogical text. Qdrant Cloud holds only 1024-dimensional dense semantic pointer
vectors — never the text itself — with the corresponding MySQL chunk identifiers as metadata. This
pointer-based decoupling keeps approximate nearest-neighbour search fast while leaving relational
integrity with MySQL. Content is embedded with Cohere `embed-multilingual-v3.0`, which maps Modern
Standard Arabic, French and Algerian Darja into a unified semantic space, so a Darja query retrieves
formal Arabic or French curriculum content without any translation middleware.

**2. An LSTM Deep Knowledge Tracing module that steers generation.** The tracer treats a student's
interaction history as a continuous time series over micro-skills aligned with MySQL
`course_part_id` entries. Each step is encoded as the target micro-skill identifier, the normalized
evaluation score, and the response time as a proxy for cognitive effort. The model outputs a
continuous mastery prediction in [0, 1], which is injected into the LLM prompt as a difficulty
constraint at three tiers — foundational below 0.50, standard from 0.50 to 0.80, and challenge above
0.80. Because the mastery score modifies a prompt that is already being assembled, adaptation
carries **zero additional inference overhead**: no second model call is required.

**3. A QLoRA fine-tuning benchmark across four open-source LLMs.** ALLaM-7B-Instruct, LLaMA-3-8B,
Qwen2.5-7B and DeepSeek-R1-Distill-Qwen-7B were each fine-tuned under identical hyper-parameters on
491 question–answer pairs drawn from official History and Geography *Corrigés Types*, then evaluated
across training convergence, lexical overlap, semantic alignment and LLM-as-a-Judge assessment.

## Repository structure

```
data/
  sft_dataset/
    bac_history_geo_491qa.jsonl      491 verified instruction-output pairs, Alpaca JSON
  dkt_synthetic/
    generate_dkt_data.py             IRT-inspired cold-start simulator
    dkt_interactions_sample.csv      1,000,000 interactions, 10,000 synthetic learners

evaluation/
  dkt_lstm_continuous.py             LSTM-DKT training and evaluation
  eval_bleu_rouge_bertscore_.llm_judge.py
                                     BLEU, ROUGE-L, BERTScore and LLM-as-a-Judge scoring
  models_evaluation_results.csv      Per-question scores behind the fine-tuning benchmark

models/
  lstm_dkt/lstm_dkt_model.pth        Trained knowledge-tracing weights
  finetuned_llm/README.md            Link to the QLoRA adapter on the Hugging Face Hub

backend/                             The deployed services — see backend/README.md
  src/                               Node.js orchestrator: retrieval, grounding, generation
  dkt/                               The knowledge tracer, packaged for training and serving
  sql/                               Curriculum schema and the interaction export view
  docs/                              Architecture, pipeline, data model, API, evaluation
```

The fine-tuned ALLaM adapter is hosted separately, at
[Belkacemdz/allam-deepbac-adapter](https://huggingface.co/Belkacemdz/allam-deepbac-adapter).

## Results

### Fine-tuning benchmark

Four models fine-tuned with QLoRA under identical settings: rank 16, α 32, 4-bit NF4 with bfloat16
compute, 3 epochs, learning rate 2×10⁻⁴, on a single NVIDIA RTX 3060 (12 GB) via Unsloth. Factual
accuracy and format compliance are LLM-as-a-Judge scores on a 1–10 scale. Lower is better for the
two losses, higher for the rest.

| Model | Train loss | Eval loss | BERTScore F1 | Factual | Format |
|---|---|---|---|---|---|
| ALLaM-7B | **0.599** | 1.312 | **0.689** | **4.86** | 5.12 |
| LLaMA-3-8B | 0.688 | 1.173 | 0.681 | 4.56 | **5.17** |
| Qwen2.5-7B | 0.700 | **1.120** | 0.677 | 3.95 | 4.47 |
| DeepSeek-R1-Distill-Qwen-7B | 1.557 | 1.841 | 0.658 | 1.79 | 3.54 |

Semantic alignment exceeds 0.65 BERTScore for every model, but factual scores stay modest — 4.86 out
of 10 at best. Fine-tuning adapts pedagogical register; on this dataset it does not by itself
deliver the factual reliability high-stakes tutoring requires. That is what motivates retrieval
grounding.

### Retrieval grounding

LLM-as-a-Judge faithfulness, 1–10 scale, DeepSeek-V4-Pro as judge, over a balanced set of 30 BAC
History–Geography questions covering the highest-risk hallucination topics. A 10-question subset was
independently validated by a certified Algerian BAC teacher.

| Architecture | Faithfulness |
|---|---|
| Fine-tuned ALLaM-7B, no retrieval | 2.00 |
| ALLaM-7B + retrieval (Qdrant + MySQL) | 3.98 |
| DeepSeek API + retrieval — production | **9.89** |

### Knowledge tracing

Trained on 80% of the synthetic interaction corpus, validated on 10%, and tested on a held-out 10%
split of 1,000 synthetic learner trajectories. Regression metrics are appropriate because the model
predicts a continuous mastery score rather than a binary pass/fail outcome.

| Metric | Score |
|---|---|
| Mean squared error | 0.0012 |
| Root mean squared error | 0.0352 |
| Mean absolute error | 0.0280 (2.80%) |
| R² | 0.9116 |

The model accounts for over 91% of the variance in learner performance trajectories, and predictions
deviate from subsequent scores by under 3 percentage points on average — a margin comfortably finer
than the 0.50 and 0.80 boundaries the three-tier difficulty steering depends on.

### Beta deployment pilot

A structured 12-item survey of 25 Algerian BAC students (12 male, 13 female; Science, Mathematics
and Literature streams; 18 from Biskra wilaya and 7 from other regions), drawn from over 130
registered beta users.

| Indicator | Result |
|---|---|
| Overall satisfaction (1–5 Likert) | 4.32 / 5 |
| Students rating 5/5 | 60% |
| Rated better or much better than ChatGPT | 76% |
| Rated inferior to ChatGPT | 4% |
| Comprehension aided — "absolutely" | 72% |
| Comprehension aided — "somewhat" | 28% |
| Total comprehension assistance rate | 100% |
| Would recommend | 100% |

Open-text responses consistently identified strict adherence to the official BAC grading methodology
as the single most valued differentiator. The main reported limitation was geometric proof exercises
requiring diagram interpretation, which makes Vision-Language Model integration the most critical
enhancement for the next version.

## Datasets

**BAC curriculum knowledge base.** Built from three authoritative sources: lesson summaries authored
and verified by certified Algerian secondary educators across multiple BAC streams; official
historical examination papers (*Annales*) covering more than ten years; and the official annotated
*Corrigés Types*. Algerian academic PDFs interleave right-to-left Arabic with LaTeX formulas, which
defeats standard OCR on both reading order and formula parsing, so extraction was VLM-assisted and
followed by manual expert review and LaTeX transcription. The corpus itself is licensed course
material and is not redistributed here.

**SFT instruction dataset — 491 pairs.** Manually constructed from official History and Geography
*Corrigés Types*, in Alpaca JSON with `instruction` and `output` fields. Three annotation criteria
governed every pair: the output follows the official stepwise BAC methodology rather than only
giving the correct final answer; all dates, proper nouns, geographic data, treaty names and legal
references are cross-validated against at least two independent official sources; and all historical
legislation, formulas and named events are transcribed verbatim. History and Geography were chosen
deliberately as the highest-density test bed for hallucination risk, giving worst-case rather than
average-case estimates.

**Synthetic DKT corpus.** No live student interaction data existed at design time, a classical
cold-start problem. A simulation engine generates 1,000,000 interaction rows across 10,000 synthetic
student profiles. It is not a fitted IRT response model; item response theory motivates the latent
ability variable, while practice and noise generate the longitudinal trajectory:

```
Ability(u) ~ N(0, 0.4)
y(t)       = sigmoid( Ability(u) + 1.2 · t_frac + eps ),   eps ~ N(0, 0.1)
```

where `t_frac` is normalized interaction progress following the Power Law of Practice. An additive
score noise term `N(0, 2.5)` reproduces the natural irregularity of real assessment marks. The
generator parses the live MySQL schema and maintains full foreign-key integrity across `user_id`,
`subject_id`, `course_id` and `course_part_id`, so synthetic records are directly compatible with
the production database.

## Running the system

The deployed services live in [`backend/`](backend), which documents its own setup. In short:

```bash
cd backend
npm install && npm start                 # retrieval and generation, port 5001

pip install -r requirements.txt
uvicorn dkt.api:app --port 5002          # knowledge tracing, port 5002
```

[`backend/README.md`](backend/README.md) covers configuration and the request flow, and
[`backend/docs/`](backend/docs) covers the architecture, the retrieval pipeline, the data model, the
API reference and the knowledge-tracing model in detail.

## Scope and limitations

The results above are promising but bounded, and the paper states these limits explicitly. The
faithfulness evaluation covers 30 History–Geography questions. The knowledge-tracing benchmark uses
synthetic rather than longitudinal student data, so it validates the implementation rather than the
method. Multilingual performance is not stratified by Modern Standard Arabic, French, Darja or
code-switching, so trilingual input is a system capability rather than an experimentally validated
performance claim. Retrieval is dense-only. The 25-student pilot is geographically concentrated.

Planned next steps are a controlled ablation with the base model held fixed, a larger and
language-stratified test set, validation of knowledge tracing on real learner trajectories, hybrid
BM25 and dense retrieval, and diagram input through Vision-Language Models.

## Citing

If you use DeepBac, its datasets or its model weights, please cite the accompanying paper.

## License

MIT.
