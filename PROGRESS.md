# Progress / Handoff

This file exists so a fresh Claude Code session (e.g. on a different machine) can pick up
exactly where things left off. The roadmap HTML's checkboxes live in browser `localStorage`,
which does **not** travel with the repo — this file is the portable source of truth for
"what's done and what's next." Update it whenever a task gets marked done, or ask Claude to.

## How to resume on a new machine

1. Clone/pull the repo.
2. `uv sync --system-certs` to install dependencies.
3. Copy `.env.example` to `.env` and fill in real `DATABASE_URL`/`REDIS_URL` (Neon/Upstash — see README).
4. `uv run alembic upgrade head` to apply migrations.
5. Open `ai_ml_engineer_build_first_roadmap.html` in a browser — it will *not* show prior
   progress (fresh localStorage on a new machine/browser). Trust this file instead, or ask
   Claude to re-seed the HTML's checkboxes from the state described below.
6. `data/` and `uploads/` are gitignored — datasets and uploaded files do **not** transfer
   via git. Re-download `telco_churn.csv` (see Stage 2 section below for source) into `data/`
   if continuing Stage 2 work.
7. Tell Claude: "read PROGRESS.md and tell me what's next."

## Current position: Stage 2 — Classical Machine Learning (Telco Customer Churn)

### Stages 0 and 1: fully complete
- **Stage 0** (Production Python Foundation): FastAPI app, Pydantic schemas, config, async
  SQLAlchemy + Alembic, dataset CRUD via service/repository layers, structured logging,
  global exception handling, unit + integration tests, Dockerfile, docker-compose (API +
  local Postgres + local Redis), health/readiness endpoints, type hints. All 16/16 tasks.
- **Stage 1** (Data Ingestion & Data Engineering): CSV upload endpoint, Pandas parsing,
  column-type detection (numeric/categorical/text heuristic), row/column counts, missing
  values, unique counts, min/max/mean/median/std, profile endpoint, background job
  processing (FastAPI `BackgroundTasks`, not a real queue — Redis is provisioned but unused
  so far), job states (queued/running/completed/failed), error persistence, idempotency,
  tests. All 16/16 tasks. Full build log in `README.md` under "Stage 1."

### Stage 2: complete (21/21 tasks)

**Done:**
1. Choose a real tabular dataset — [Telco Customer Churn](https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv)
   (IBM's official sample data, 7,043 rows, 21 columns). Saved to `data/telco_churn.csv`
   (gitignored — re-download if missing).
2. Define the prediction target — `Churn` (binary: `Yes`/`No`, ~26.54% churn rate — **class imbalance**, matters for every metric downstream).
3. Separate features from labels.
4. Create train/validation/test splits (70/15/15, stratified on `Churn`).
5. Build preprocessing pipeline.
6. Handle missing values.
7. Encode categorical features.
8. Create baseline model (`DummyClassifier`, majority-class — accuracy 0.7348, the floor every real model must beat).
9. Train logistic regression — accuracy 0.8059, but **churn-class recall only 0.59** (misses 41% of actual churners). This gap between "looks fine on accuracy" and "mediocre at the thing that matters" is deliberate setup for Stage 3 (ML Failure Lab).
10. Train decision tree (`max_depth=5`) — accuracy 0.7926 (lower than logistic regression), but churn recall 0.62 (higher). First concrete example of "no single winner" between models — precision/recall tradeoff, not a strict improvement. Both models still miss ~40% of actual churners.
11. Train random forest (`n_estimators=100, max_depth=5`) — accuracy 0.7898, but churn recall dropped to **0.42**, the worst of the three real models. Majority voting across trees biases toward the majority class ("no churn"), hurting minority-class (churn) recall — ensembling reduces variance but doesn't help class imbalance, can even worsen it.
12. Train XGBoost (`n_estimators=100, max_depth=5`, untuned) — accuracy **0.7718** and churn recall **0.50**, the lowest accuracy of the four real models and worse recall than both logistic regression and the plain decision tree. Confirms "fancier algorithm" isn't automatically better: dataset is small (~4,900 train rows) and mostly linear, XGBoost defaults weren't tuned, and — the real underlying issue — **none of the four models handle class imbalance yet**, so all of them are biased toward predicting the majority class. That fix (class weights / resampling / threshold tuning) is intentionally still not applied; it's Stage 3 material.
13. Compare model metrics — results collected into a `results` list (dicts of accuracy/precision/recall/f1 per model) and printed as one `pd.DataFrame` table at the end of the script instead of scattered per-model prints. Final validation-set comparison:

   | model | accuracy | precision | recall | f1 |
   |---|---|---|---|---|
   | Logistic Regression | 0.8059 | 0.6459 | 0.5929 | 0.6183 |
   | Decision Tree | 0.7926 | 0.6063 | 0.6214 | 0.6138 |
   | Random Forest | 0.7898 | 0.6611 | 0.4250 | 0.5174 |
   | XGBoost | 0.7718 | 0.5809 | 0.5000 | 0.5374 |

   No single model wins every metric: best accuracy/F1 is Logistic Regression, best recall (catches the most actual churners) is Decision Tree, best precision (fewest false alarms) is Random Forest. This is the concrete evidence base for Stage 3's precision/recall-tradeoff and class-imbalance lessons.
14. Implement cross-validation — `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)` + `cross_val_score(..., scoring="recall")` per model, run on `X_train`/`y_train`. 5-fold CV recall (mean ± std):

   | model | single-split recall | CV recall (mean ± std) |
   |---|---|---|
   | Logistic Regression | 0.593 | 0.547 ± 0.030 |
   | Decision Tree | 0.621 | 0.521 ± 0.070 |
   | Random Forest | 0.425 | 0.436 ± 0.022 |
   | XGBoost | 0.500 | 0.531 ± 0.029 |

   Important finding: the single train/val split ranking was misleading. Decision tree looked
   best on recall (0.621) but has the widest CV spread (± 0.070) — unstable, that number was
   partly luck. XGBoost looked worst on the single split (0.500) but has a better and more
   stable true average (0.531 ± 0.029) than the decision tree. Cross-validation is what
   revealed this — a single split isn't trustworthy enough to rank models on its own.

   **Gotcha hit while adding this:** `from xgboost import XGBClassifier, cv` accidentally
   imported XGBoost's own `cv` function under the same name intended for
   `StratifiedKFold(...)`, causing `InvalidParameterError` (cv was a function, not a
   splitter) and then `NameError` once the bad import was removed but the `cv = StratifiedKFold(...)`
   assignment line was never actually added. Fixed by importing only `XGBClassifier` from
   xgboost and explicitly adding `cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`
   right after the preprocessor fit, before any model pipeline uses it.
15. Persist trained model artifacts — `joblib.dump(logreg_pipeline, "models/churn_logreg_pipeline.joblib")`
   at the end of the script. Chose logistic regression (0.547 CV recall, most stable of the
   two closest contenders) over XGBoost to persist first. `joblib.dump` auto-creates missing
   parent directories, so `models/` didn't need to be manually created. `models/` added to
   `.gitignore` (binary artifacts, same treatment as `data/`/`uploads/`). Verified by loading
   the file back with `joblib.load(...)` and confirming `.named_steps` shows both the
   `preprocessor` (numeric + categorical sub-pipelines) and fitted `classifier` intact.

All of this lives in **`scripts/train_churn_model.py`** — a standalone script (not yet wired
into the FastAPI app), run with `uv run python scripts/train_churn_model.py`. Deliberately
kept as a script for now since this stage's early tasks are data-science exploration, not
API work — the API tasks (`POST /models/train`, etc.) come later in the stage, after the
scikit-learn exploration is settled.

**Known gotcha already handled:** `TotalCharges` reads as text from the raw CSV — 11 rows
have a blank/whitespace string (not a real `NaN`) for brand-new customers with `tenure=0`.
Fixed with `pd.to_numeric(df["TotalCharges"], errors="coerce")` before anything else.

16. Create model metadata record — `models/churn_logreg_pipeline.metadata.json` written
   alongside the `.joblib` artifact: model type, UTC training timestamp, source dataset path,
   train/val row counts, single-split metrics (accuracy/precision/recall/f1), and CV recall
   mean/std. Sklearn's metric functions returned native Python floats in this environment (not
   `numpy.float64`), so `json.dump` serialized directly without needing manual `float(...)`
   casts on the `results` entries — only `cv_recall_mean`/`cv_recall_std` were explicitly cast
   (defensive, in case that ever changes with a different sklearn version/build).

17. `POST /api/v1/models/train` — `app/services/model_service.py`'s `train_churn_model(db, dataset_id)`.
   Re-reads the dataset's CSV via `dataset.storage_path`, re-implements the same
   preprocessing pipeline as the script (numeric impute+scale, categorical impute+one-hot),
   trains a single hardcoded `LogisticRegression` (not all 4 models — a deliberate
   simplification for the API path vs. the script's exploration), saves the artifact to
   `models/<uuid>_churn_logreg.joblib`, and creates a `TrainedModel` DB row (new table,
   migration `74975db85037`) with `accuracy`/`precision`/`recall`/`f1` stored as real columns
   (not JSON) — a genuine model registry now, not just a metadata JSON file. Verified:
   `POST {"dataset_id": 14}` against the real uploaded telco-churn dataset →
   `201`, model `id=1`, accuracy 0.8056 (close to the script's 0.8059; small difference is an
   80/20 split here vs. 70/15/15 in the script).
18. `GET /api/v1/models/{id}` — returns `ModelResponse` (id/name/model_type/dataset_id/created_at). Verified working.
19. `GET /api/v1/models/{id}/metrics` — returns `ModelMetrics` (accuracy/precision/recall/f1) by
   reusing `model_service.get_model` and letting FastAPI's `response_model` pick the matching
   fields off the `TrainedModel` ORM object. `ModelNotFoundError` → `404` verified on both
   endpoints (registered in `main.py`, same pattern as `DatasetNotFoundError`/`JobNotFoundError`).

**Known duplication worth being aware of, not yet reconciled:** the preprocessing pipeline
now exists in two places — `scripts/train_churn_model.py` (exploration, 4 models, CV) and
`app/services/model_service.py` (API path, LogisticRegression only). They're currently kept
in sync by hand. Not fixed yet; worth a "share this via a common module" pass eventually but
not blocking the remaining tasks.

20. `POST /api/v1/models/{id}/predict` — `model_service.predict_churn(model, features)` loads
   the model's `.joblib` pipeline, derives `expected_columns` from `pipeline.feature_names_in_`
   (not hardcoded — works for any model's actual feature set), rejects both missing *and*
   unexpected feature keys via `InvalidPredictionInputError` → `400`, builds a single-row
   DataFrame, and returns `(prediction_label, churn_probability)` from `predict`/`predict_proba`.
21. Inference validation and error handling — `InvalidPredictionInputError` registered in
   `main.py` (400), reusing `ModelNotFoundError` (404) for unknown model ids. Verified all 3
   paths against a freshly trained model: happy path (realistic at-risk customer — new,
   month-to-month, fiber, no add-ons, high charges — correctly predicted `"Yes"` at 77%
   probability), missing-field rejection (`400`), unknown model id (`404`).

**Gotcha hit and fixed:** `models/` wasn't a mounted Docker volume (unlike `uploads_data`/
`postgres_data` in `docker-compose.yml`), so a container rebuild wiped trained model artifacts
while their DB rows survived (Postgres persists) — predicting against an orphaned model id
fell through to the global handler as a clean `500` (confirmed live). Fixed by adding
`models_data:/app/models` to the `api` service, same pattern as `uploads_data`. Verified: `pwd`
inside the container and `Path("models").resolve()` both confirmed `/app/models` is the
correct mount target (matches `MODEL_DIR = Path("models")` in `model_service.py` resolved
against the `Dockerfile`'s `WORKDIR /app`) — then trained a model, rebuilt, and confirmed both
the `.joblib` file and a live prediction against it survived.

## Stage 2 is complete (21/21). Stage 3 is complete (17/17).

Deliberately break models and diagnose why — overfitting, underfitting, data leakage, class
imbalance, precision/recall tradeoffs, threshold tuning, confusion matrices, ROC-AUC, feature
importance, distribution shift. Stage 2 already surfaced most of the raw material this stage
digs into: class imbalance (~26.5% churn rate) that every model struggled with, the random
forest's majority-vote bias hurting minority-class recall, and the cross-validation finding
that a single train/val split ranking (decision tree "winning" on recall) was partly
luck/unstable compared to the CV mean.

All Stage 3 work lives in a new **`scripts/failure_lab.py`** (separate from
`scripts/train_churn_model.py` — deliberately: Stage 2's script is a training/comparison
pipeline, Stage 3 is a series of diagnostic experiments, different purpose).

**Done:**
1. Create a deliberately overfit model — `DecisionTreeClassifier(max_depth=None, min_samples_leaf=1)`,
   no complexity limits, on the same train/val split pattern as Stage 2.
2. Compare training and validation performance — train accuracy **0.9980** (near-perfect,
   the tree memorized individual training rows) vs. val accuracy **0.7293**, a **26.87-point
   gap**. Notably, val accuracy here is *worse* than Stage 2's majority-class baseline
   (0.7348) — despite near-perfect training performance, the overfit model is actually worse
   than doing nothing on new data. Stage 2's properly-capped tree (`max_depth=5`) scored
   0.7926 val accuracy for comparison — less "training performance," better real performance.

3. Reduce model complexity and compare results — swept `max_depth` across
   `[1, 2, 3, 5, 10, 20, None]`. Clean bias-variance curve: val accuracy climbs with train
   accuracy through depth 5 (**peak val accuracy 0.7946 at depth 5**, gap stays under 1%),
   then train keeps climbing while val *drops* past depth 5 — depth 10 gap jumps to 11%,
   depth 20/unbounded gap ~27% (matches the standalone overfit experiment). `max_depth=1`
   (0.7347 val accuracy) is nearly identical to the Stage 2 baseline (0.7348) — a 1-question
   tree barely beats guessing the majority class. Confirms Stage 2's `max_depth=5` choice
   (made somewhat arbitrarily at the time) was actually close to optimal for this dataset.

4. Create an underfit model — standalone `DecisionTreeClassifier(max_depth=1)`, isolated from
   the sweep table as its own labeled result. Train 0.7347, val 0.7345, gap ~0.0002 — near-zero
   gap but both scores mediocre (barely above the 0.7348 baseline). Clean contrast established
   with the overfit case: small train/val gap alone doesn't mean "good," it means
   "consistent" — need both a small gap *and* genuinely good scores (which `max_depth=5`,
   gap 0.0078 and val 0.7946, actually achieves).

5. Add useful features and measure improvement — engineered `NumServices` (count of 6
   service add-on columns with `"Yes"`) and `AvgMonthlyCharge` (`TotalCharges / tenure`,
   zero-safe). Tested at both `max_depth=1` and `max_depth=3`, against the original (non-
   engineered) columns at the same depths. Result: **identical accuracy in both cases** (down
   to 4 decimal places) — the engineered features added zero measurable signal. Real,
   evidence-based explanation, not a bug: both features are just repackaged versions of
   columns the tree already had direct access to (`NumServices` summarizes 6 columns the tree
   can already split on individually; `AvgMonthlyCharge` is a ratio of two already-present
   columns) — decision trees can implicitly combine existing features across splits, so
   pre-combining them for it added nothing. Noted as a case where feature engineering likely
   matters more for models that *can't* naturally combine features (e.g. logistic regression)
   than for trees — not verified yet, a good candidate for a future experiment.

6. Introduce target leakage intentionally + 7. Observe unrealistically high validation
   performance — simulated a realistic leak: `CancellationRequestFiled`, a synthetic column
   ~95% equal to the true `Churn` label (5% randomly flipped noise), representing a real
   scenario (a field that only gets populated *because* a customer is already churning, so it
   wouldn't genuinely be available before the outcome). Added at `max_depth=5` (same capacity
   as the honest baseline, isolating the leak as the only variable). Result: val accuracy
   jumped from **0.7946 (honest) to 0.9560 (leaked)** — a 16-point jump, landing almost
   exactly at the 95% "correctness" the leaked feature was built with, strong evidence the
   model largely just learned to repeat that one column back. Key insight documented: leakage
   is invisible to train/val-gap checks and cross-validation (the leak is present equally in
   both splits) — the only real defense is asking "would this feature genuinely be available
   before the prediction is needed, in production?", not a metric-based check.

8. Remove the leaked feature and retest — retrained the same `max_depth=5` tree without
   `CancellationRequestFiled`. Val accuracy: **0.7946**, matching the original honest baseline
   exactly, confirming the entire 16-point jump (0.7946 → 0.9560) was caused solely by that
   one column. Closes the leakage experiment loop: introduce → observe suspicious jump →
   remove → confirm it was the cause — the actual workflow for debugging real leakage bugs.

9. Create an imbalanced classification dataset — subsampled churned customers in the
   *training* set down to a 5% churn rate (190 of 1308 kept), while leaving `X_val`/`y_val`
   at the natural ~26.5% distribution — deliberately modeling a realistic scenario (few
   labeled positive examples to train on, but the real world still has its natural mix).
   Trained the same `max_depth=5` tree on this skewed data. Result: val accuracy 0.7397
   (barely dropped from 0.7946) but **val recall collapsed to 0.0250** (from 0.6214) — the
   model essentially stopped predicting churn at all, defaulting close to the majority-class
   baseline while still *looking* reasonable on accuracy alone.
10. Compare accuracy with precision, recall and F1 — built a reusable `evaluate()` helper and
    a 3-row comparison table (baseline / honest / 5%-imbalance-trained). The imbalance-trained
    model's **precision was actually the highest of the three (0.8235)** — on the rare
    occasions it predicts churn, it's usually right — but recall (0.0250) and F1 (0.0484,
    the harmonic mean, correctly punished by the near-zero recall) exposed how genuinely
    broken it is despite the deceptively normal-looking accuracy (0.7397).

**Gotcha hit and fixed:** `clean_pipeline` and `imb_pipeline` originally shared the *same*
`preprocessor` `ColumnTransformer` object (not a copy) across multiple `Pipeline`s built at
different points in the script. Since `Pipeline.fit()` refits whatever transformer object it
holds *in place*, fitting `imb_pipeline` (on the imbalanced subset) silently overwrote the
shared `preprocessor`'s learned state (median values, one-hot categories), corrupting
`clean_pipeline`'s later predictions even though `clean_pipeline` itself was never re-fit —
its accuracy shifted from 0.7946 to a wrong 0.787979 with no code touching it directly. Fixed
by giving each pipeline its own `sklearn.base.clone(preprocessor)` instead of sharing the
object. Earlier pipelines in this script (`overfit_pipeline`, the complexity sweep loop,
`underfit_pipeline`) have the same latent hazard but never showed a visible bug, purely
because they always happened to be refit on identical data each time — not something to rely
on going forward. General rule adopted: always `clone()` a shared transformer before handing
it to a new `Pipeline`.

11. Generate a confusion matrix — `confusion_matrix(...).ravel()` (order: `tn, fp, fn, tp`)
    for both the honest and 5%-imbalance-trained models. Honest: TN 1340, FP 212, FN 222, TP
    339 — reasonably balanced. 5%-imbalance: TN 1549, FP 3, FN **547**, TP 14 — of 561 actual
    churners, only 14 were caught; 547 real customers who churned would have received zero
    retention outreach. Makes the abstract "2.5% recall" number concrete as an actual count
    of missed customers rather than a percentage.

12. Experiment with classification thresholds — swept threshold `[0.1..0.9]` on
    `clean_pipeline.predict_proba(X_val)[:, 1]` instead of using `.predict()`'s baked-in 0.5
    cutoff. Clear precision/recall tradeoff: threshold 0.1 → recall 0.964/precision 0.386;
    threshold 0.9 → precision 1.0/recall 0.002. **F1 peaks near threshold 0.3 (0.6076)**,
    marginally above the default 0.5 (0.6070) — a flat plateau from ~0.2-0.5, sharp drop-off
    after. Noted: thresholds 0.6 and 0.7 gave identical results — a decision tree only outputs
    a handful of distinct leaf probabilities, so threshold changes within a gap between two
    leaf values have zero effect (unlike logistic regression's continuous probability output).
    Also a small (~0.005) precision discrepancy at threshold=0.5 vs. `clean_pipeline.predict()`'s
    own reported precision — attributed to tie-breaking on a leaf with an exact 50/50 split,
    not a bug. Practical link back to the imbalance experiment: threshold lowering is a
    retrain-free way to partially rescue a low-recall model, distinct from fixing the
    training data itself.

13. Compare ROC-AUC across models — baseline 0.5000 (confirms zero ranking ability, as
    expected), honest 0.8290, 5%-imbalance-trained **0.8003** (surprisingly close to honest
    despite 0.025 recall!), overfit 0.6526. Key finding: the imbalance-trained model's real
    problem is *calibration*, not ranking ability — its AUC is nearly as good as the honest
    model, meaning it still correctly ranks real churners as higher-risk on average, it just
    learned to output uniformly lower probabilities (too few positive examples during
    training), so almost nothing crosses the default 0.5 threshold. Directly validates the
    earlier threshold-tuning task: this specific model is a strong candidate for threshold
    lowering. The overfit model, by contrast, has genuinely weaker ranking ability (0.65,
    well below both real models) — a different, deeper failure that threshold tuning can't
    fix. This distinction (calibration failure vs. genuine ranking failure) is invisible to
    accuracy/precision/recall/F1 alone, all of which depend on one committed threshold.

14. Measure feature importance — via `.named_steps["classifier"].feature_importances_` paired
    with `.named_steps["preprocessor"].get_feature_names_out()` (needed to map the 45 expanded
    one-hot columns back to real names). Honest model: `Contract_Month-to-month` dominates at
    51%, `tenure` 17.8%, `InternetService_Fiber optic` 15.3% — top 3 account for ~84%, matches
    known real-world churn analysis (no-commitment contracts churn most). Leaky model:
    `CancellationRequestFiled` alone is **92.1%** of total importance — a single feature
    dominating that heavily is itself an independent red flag, catchable even without
    checking the train/val gap or suspicious accuracy. Confirms feature importance as a
    genuinely separate diagnostic tool from the metrics-based checks used earlier.

15. Create a distribution-shift experiment — simulated a 20% price increase (`MonthlyCharges`/
    `TotalCharges` scaled up on `X_val`, `y_val` left unchanged — deliberately isolating "does
    the model's behavior degrade under input drift" from "how would customers actually react
    to a real price hike"). Result: accuracy 0.7946→0.7743, **precision 0.6152→0.5709** (the
    biggest drop), recall nearly flat (0.6043→0.6025), F1 0.6097→0.5863. Explanation: inflated
    charge values pushed more customers across the tree's learned risk thresholds, so it
    flagged churn more often overall — catching about the same real churners (flat recall)
    but with more false alarms (precision drop). Identified as the most dangerous failure mode
    covered in this stage: unlike overfitting (visible train/val gap) or leakage (suspiciously
    perfect numbers), distribution shift produces quiet, gradual decay with no obvious single-
    snapshot red flag — only catchable by monitoring performance over time against a stable
    baseline, which is why production ML systems need drift detection, not just train-once.

16. Document each failure and its root cause — **`docs/failure_lab_findings.md`**, one
    section per failure mode (symptom/root cause/detection/fix, grounded in the actual
    numbers produced), plus a cross-cutting lessons section. Covers all 7: overfitting,
    underfitting, leakage, imbalance, threshold sensitivity, ranking-vs-calibration
    (ROC-AUC), distribution shift.

17. Add automated tests for critical preprocessing behavior — `tests/unit/test_preprocessing.py`,
    two regression tests targeting the two real bugs found earlier in this stage:
    `test_total_charges_blank_string_becomes_nan` (locks in the `pd.to_numeric(errors="coerce")`
    fix) and `test_cloned_preprocessor_does_not_leak_state_between_pipelines` (locks in the
    `clone()` fix for the shared-preprocessor mutation bug). Both pass; full suite (17 tests
    total across unit + integration) confirmed green.

## Stage 3 is complete (17/17). In progress: Stage 4 — Deep Learning with PyTorch (4/20 tasks done)

Replace one classical model with a neural network built from a custom PyTorch training loop
(not a high-level abstraction) — tensors, Datasets/DataLoaders, a feed-forward network, loss
functions, backprop, optimizers, checkpointing, GPU/CPU handling, and exposing training as a
background job in the platform (tying back to Stage 1's job-queue system). Same Telco churn
dataset as Stage 2, for a direct comparison against the classical models. All work lives in a
new **`scripts/train_churn_nn.py`** (same standalone-script pattern as Stages 2/3).

**Done:**
1. Install and configure PyTorch — `uv add --system-certs torch`. Installed version
   `2.13.0+cu130`. `torch.cuda.is_available()` → `False` (no NVIDIA GPU on this machine,
   expected) — training will run on CPU throughout this stage.

2. Convert a dataset into tensors — reused the Stage 2 `ColumnTransformer` preprocessing
   pattern (impute + scale numeric, impute + one-hot categorical), fit on `X_train` only,
   then converted the resulting numpy arrays to `torch.float32` tensors. Labels reshaped via
   `.unsqueeze(1)` from `(N,)` to `(N, 1)` to match the shape PyTorch loss functions expect
   against the network's output layer.
3. Implement a PyTorch Dataset — `ChurnDataset(Dataset)` with `__len__`/`__getitem__`.
4. Implement a DataLoader — `train_loader` (`shuffle=True`, batches the model actually
   trains on) and `val_loader` (`shuffle=False`, evaluation only), `batch_size=32` as a
   starting point (to be tuned explicitly later in this stage).

**Gotcha hit and fixed:** the `ChurnDataset` class was defined but never instantiated —
`train_dataset`/`val_dataset` (and `BATCH_SIZE`) were referenced by the `DataLoader` calls
without ever being created, `NameError`. Fixed by adding the missing instantiation lines
between the class definition and the `DataLoader` calls. Verified: 177 train batches / 45 val
batches, consistent with ~5,634 train rows and ~1,409 val rows at batch size 32.

**Next task:** "Build a small feed-forward neural network" + "Implement forward pass" — the
actual network architecture, defined as a PyTorch `nn.Module`.

**Security note (joblib):** `joblib.load`/pickle-based formats can execute arbitrary code if
loading an untrusted file. Fine here since we only ever load artifacts this same project
trained and saved locally — never load a `.joblib` file from an external/untrusted source
without treating it like arbitrary code execution risk.

## Working conventions established this build (read before continuing)

- **README.md** is a step-by-step build log, in the order things were actually built — check
  it before re-deriving how something was set up. Every new shell command introduced should
  be added to it.
- **User is learning this stack.** Default mode: explain what to do and where, let the user
  make the edit themselves, verify after. Only apply edits directly when explicitly asked
  ("you do it").
- **Docker Compose requires a rebuild after any code/dependency/migration change**:
  `docker compose up --build -d`. If a migration was added, also re-run it inside the
  container: `docker compose exec api uv run alembic upgrade head` — the Docker Postgres
  and the Neon (hosted) Postgres are separate databases, each needs migrations applied
  independently.
- **Roadmap progress tracking**: `ai_ml_engineer_build_first_roadmap.html` has a one-time
  JS seed block near the bottom (`SEED_KEY` / `stageNDone` arrays) that pre-checks completed
  tasks in a fresh browser's `localStorage`. Bump the `SEED_KEY` version string and update
  the relevant `stageNDone` array whenever a task is confirmed done — the seed only re-fires
  when the key changes.
