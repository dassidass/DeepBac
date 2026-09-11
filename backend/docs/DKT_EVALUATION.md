# Evaluation

What the knowledge tracer was measured on, how, and what the result does and
does not license.

## 1. Metrics

The model predicts a continuous mastery score in [0, 1] rather than a binary
correct/incorrect outcome, so the primary metrics are regression metrics:

| Metric | Reads as |
|---|---|
| **MSE** | The training objective |
| **RMSE** | On the score scale, in the same units as the target |
| **MAE** | The typical prediction error. 0.028 means predictions land within 2.8 score-points on average |
| **R²** | Share of target variance explained. Negative would mean worse than predicting the mean |
| **AUC (binarised)** | Ranking quality at a 0.5 pass threshold, for comparison with the binary-DKT literature. 0.5 is chance |
| **Accuracy (binarised)** | Reported for completeness, and weak on its own: on a cohort where most answers pass, predicting "pass" already scores well |

R² and AUC do the interpretive work. MSE alone cannot separate a model that
learned something from one that learned the mean, and on a skewed score
distribution both look equally good.

## 2. Data and split

The synthetic cohort described in [`dkt/scripts/generate_synthetic.py`](../dkt/scripts/generate_synthetic.py):
10,000 learners, 1,000,000 interactions, generated from the IRT-inspired
cold-start process with a fixed seed.

Learners are split **80 / 10 / 10** — train, validation, test — and the split is
by learner, never by row:

```python
user_ids = list(df["user_id"].unique())
random.Random(seed).shuffle(user_ids)
train_users = set(user_ids[:n_train])
val_users   = set(user_ids[n_train : n_train + n_val])
# everything else is test
```

Splitting by row would put a learner's later interactions in training and their
earlier ones in validation, so the model would predict a student's past from
their own future. That is the standard way to report a knowledge-tracing result
that cannot be reproduced in production.

Validation selects the epoch through early stopping and drives the
learning-rate schedule. The test split is scored once, after the epoch is fixed,
and it is the split every number below comes from. The time-normalisation range
is fitted on training learners only, for the same reason.

## 3. Baselines

Every run prints two trivial predictors on the held-out learners before training
starts: the training cohort's global mean, and each section's training mean. An
error figure means nothing without them — a model that learned only the cohort
average can post a low MSE and look successful — so they are printed whether or
not anyone asks for them.

## 4. Result

Training configuration: embedding 64, hidden 128, one LSTM layer, Adam at 1e-3,
batch 32, gradient clipping at 5.0, `ReduceLROnPlateau`, early stopping with
patience 5. CPU only; no GPU is required.

| Metric | Test split |
|---|---|
| Mean squared error | 0.0012 |
| Root mean squared error | 0.035 |
| Mean absolute error | 0.028 (2.8%) |
| R² | 0.91 |

The model explains over 91% of the variance in the held-out learners'
performance trajectories, and the typical prediction lands within roughly 2.8
score-points of the next actual score.

That precision is what the three-tier difficulty steering rests on. The tier
boundaries sit at 0.50 and 0.80 (see [RAG_PIPELINE.md](RAG_PIPELINE.md) §6a), so
an error margin under three points is comfortably finer than the decision the
prediction is used to make: a learner near a boundary is near it because their
mastery really is near it, not because the estimate is noisy.

## 5. What synthetic data cannot show

The corpus was produced by a latent-trait process close to the model's own
assumptions. Recovering it is a test of the implementation — that gradients
flow, that learner state is carried forward, that the prediction tracks the
target — and never evidence about human learners. Three things are absent by
construction:

- **Real learning dynamics.** A monotone improvement curve, with no plateaus,
  breakthroughs or forgetting.
- **Real practice selection.** Learners sample uniformly. Real students choose
  what to practise, usually based on what they just got wrong, which makes the
  sequence non-random in ways no simulation here reproduces.
- **Real cohort composition.** Ability is drawn from one Gaussian, not from a
  population with streams, schools and prior attainment.

The honest summary: **the pipeline is validated, the method is not.** The method
can only be evaluated against real interaction logs, and that measurement is the
primary direction for future work.

## 6. Reproducing

```bash
pip install -r requirements.txt

# Generate the corpus. Roughly 46 MB; deterministic under the default seed.
python -m dkt.scripts.generate_synthetic --out data/dkt_interactions.csv

# Train, then score the held-out test split once
python -m dkt.train --csv data/dkt_interactions.csv --epochs 20

# Score a saved checkpoint against the baselines, without retraining
python -m dkt.evaluate --csv data/dkt_interactions.csv --split test
```

Everything is seeded (`--seed 42`, the default) and runs on CPU in roughly half
an hour. `checkpoints/dkt_lstm_paper.pt` is the trained model the table in §4
comes from, and it is what the API serves by default.

Pass `--sql <dump>` to the generator to draw real `course_id` and
`course_part_id` values from a database dump, so the generated rows join against
the live curriculum.

## 7. Evaluating on real data

```bash
python -m dkt.scripts.export_interactions --out data/interactions.csv
python -m dkt.train --csv data/interactions.csv
python -m dkt.evaluate --csv data/interactions.csv --split test
```

Check the corpus before trusting any number from it. The queries in
[../sql/dkt_interactions.sql](../sql/dkt_interactions.sql) report the two
quantities that decide whether the exercise is meaningful at all:

- **Attempts per (learner, section).** Knowledge tracing needs repeated attempts
  on the same section; below roughly three there is no trajectory to trace.
- **Interactions per learner.** Below roughly 20, the sequence is too short for
  a hidden state to carry anything.

If either is too low, the fix is more logged practice, not more epochs.

## 8. What to measure next

1. **Real logs, same harness.** The only measurement that settles whether the
   method works here.
2. **Against a knowledge-tracing benchmark.** ASSISTments or EdNet, binarised,
   to place this implementation against published DKT numbers rather than
   against its own baselines.
3. **Elapsed-time decay.** Adding time *between* interactions is the single most
   likely improvement for a revision product, where forgetting is the mechanism
   being fought.
4. **Attention models.** SAKT or AKT on the same split, once the corpus is large
   enough that they are not simply overfitting.
5. **The decision, not the prediction.** Does acting on "revise these three
   sections" improve exam outcomes? That is an A/B question, and no offline
   metric answers it.
