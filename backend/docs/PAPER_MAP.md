# Paper map

Where each mechanism the paper describes is implemented in this folder.

`backend/` is the platform's two model-backed services: the curriculum-grounded retrieval and
generation service, and the LSTM knowledge tracer that steers it. The client, the authoring tools
and the grader belong to the wider platform and are not part of it; those rows are marked
**platform** so a reader can tell at a glance what is here and what is not.

The fine-tuning artefacts the paper releases — the 491-pair instruction set, the per-question
benchmark scores, and the ALLaM adapter — live at the root of this repository, alongside the
original training and evaluation scripts. This folder does not duplicate them.

## 1. Architecture

| Paper | Where |
|---|---|
| Dual-database design: MySQL system of record, Qdrant index | [ARCHITECTURE.md](ARCHITECTURE.md) §2, [DATA_MODEL.md](DATA_MODEL.md) §1 |
| Qdrant stores pointer vectors, never chunk text | `services/qdrantSemantic.js` — the payload has no text field; [DATA_MODEL.md](DATA_MODEL.md) §3 |
| `chunk_db_id` as the pointer back to MySQL | `searchChunksByEmbedding`, which resolves ids to text in one `SELECT` |
| Four-level curriculum hierarchy | `subjects → units → courses → course_parts` in [`sql/schema.sql`](../sql/schema.sql). The paper's *Chapter / Lesson / Lesson-Element* are this repository's *unit / course / course_part* |
| Cohere `embed-multilingual-v3.0`, 1024 dimensions, cosine | `RAG_EMBEDDING_MODEL` / `RAG_EMBEDDING_DIMENSIONS`; [RAG_PIPELINE.md](RAG_PIPELINE.md) §2 |
| Trilingual retrieval without a translation layer | One embedding space for MSA, French and Darja; [RAG_PIPELINE.md](RAG_PIPELINE.md) §2 |
| Node.js orchestrator, REST, JWT | `src/server.js`, `src/middleware/auth.js` |
| Role-based access control | **platform** — this service verifies tokens issued elsewhere and does not own roles |
| React PWA, KaTeX rendering, Chart.js dashboards | **platform** |
| Hosted generator in production (`deepseek-chat`) | `src/config/llm.js`, `LLM_API_URL` |
| Self-hosted generator for the research arm | `POST /api/rag/ask-local` — same retrieval, a small local model in place of the hosted one |

## 2. Prompting and generation

| Paper | Where |
|---|---|
| Pedagogical persona, answer only from retrieved context | `buildRagGroundedPrompts`, rule 1 |
| Never introduce a fact absent from the context | Rules 1–3 and 8 of the same prompt |
| *Corrigés types* methodology, step by step | `BAC_METHODOLOGY`, carried by both the grounded and the fallback prompt |
| Difficulty instruction injected from the mastery score | `services/masterySteering.js`; [RAG_PIPELINE.md](RAG_PIPELINE.md) §6a |
| Three tiers at 0.50 and 0.80 | `RAG_MASTERY_FOUNDATIONAL_MAX` / `RAG_MASTERY_CHALLENGE_MIN`, those exact defaults |
| Zero additional inference overhead for adaptation | The paragraph joins a prompt already being assembled; no second model call exists in `routes/rag.js` |
| Language instruction from the detected register | `services/languageRegister.js` |
| Format constraints: numbered steps, LaTeX, structured layout | `LATEX_RULES` and `BAC_METHODOLOGY` |
| Adversarial hardening against prompt injection | `services/promptGuard.js`, whose module comment states the scope of each of its two checks |

## 3. Knowledge tracing

| Paper | Where |
|---|---|
| LSTM over interaction history, standard gate equations | `dkt/model.py`, `ContinuousDKTLSTM` (PyTorch `nn.LSTM`) |
| Input = skill embedding, normalised score, response time | `ContinuousDKTLSTM.encode`, input dimension `embed_dim + 2` |
| Continuous mastery output in [0, 1] | Sigmoid regression head; trained with masked MSE, not cross-entropy |
| Micro-skills aligned with `course_part_id` | `dkt/data.py`, and `rag_chunks.part_id` on the retrieval side — the same key |
| IRT-inspired cold-start simulator, Eq. (3) | `dkt/scripts/generate_synthetic.py` |
| 10,000 learners, 1,000,000 interactions | `--rows 1000000 --students 10000`; regenerate with the command in [DKT_EVALUATION.md](DKT_EVALUATION.md) §6 |
| Generator preserves foreign-key integrity against a live dump | `parse_sql_dump`, used via `--sql` |
| 80 / 10 / 10 split | `DKT_TRAIN_USER_FRAC`, `DKT_VAL_USER_FRAC`; split by learner, never by row |
| MSE 0.0012, RMSE 0.0352, MAE 0.0280, R² 0.9116 | Measured on the held-out test split; [DKT_EVALUATION.md](DKT_EVALUATION.md) §4 gives the configuration and the command that reproduces it |
| Difficulty steering drives the generator | `POST /api/rag/ask` with `mastery`; [API.md](API.md) §Composing with the knowledge tracer |

## 4. What lives elsewhere

Not everything the paper describes is in this folder. Each row says where it is instead, so a reader
following a claim knows which direction to look:

| Paper | Where |
|---|---|
| QLoRA fine-tuning benchmark across four models | Repository root: `evaluation/models_evaluation_results.csv` carries the per-question BLEU, ROUGE-L, BERTScore and judge scores, and `evaluation/eval_bleu_rouge_bertscore_.llm_judge.py` is the script that produced them |
| The 491 BAC question–answer pairs | Repository root: `data/sft_dataset/bac_history_geo_491qa.jsonl` |
| The fine-tuned ALLaM adapter | Hugging Face Hub; see `models/finetuned_llm/README.md` |
| The original knowledge-tracing script and its trained weights | Repository root: `evaluation/dkt_lstm_continuous.py` and `models/lstm_dkt/lstm_dkt_model.pth`. The `dkt/` package here is the same model and the same split policy, packaged for reuse and serving |
| The BAC curriculum corpus | Not released: licensed course material authored by teachers |
| The N = 25 pilot study | Not released: survey responses from identifiable minors |
| React PWA, gamification, dashboards, role-based access control | **platform** — closed client and platform services |
| Transport and storage security, Law 18-07 compliance | Deployment configuration rather than application code |
