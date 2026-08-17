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

### Stage 2: in progress (19/21 tasks done)

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

**Next task:** `POST /models/{id}/predict` and "Add inference validation and error handling" —
the last 2 of Stage 2's 21 tasks. Predict needs to: load the persisted `.joblib` pipeline for
the given model id, accept a single customer's feature values as a request body, validate
them (right fields present, right types, handle unknown categorical values gracefully —
`OneHotEncoder(handle_unknown="ignore")` already helps here), run `.predict()`/`.predict_proba()`,
and return a clean prediction + probability, with sensible error responses (404 if model
doesn't exist, 400/422 if the input is malformed) rather than a raw 500 on bad input.

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
