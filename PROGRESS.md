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

### Stage 2: in progress (10/21 tasks done)

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

All of this lives in **`scripts/train_churn_model.py`** — a standalone script (not yet wired
into the FastAPI app), run with `uv run python scripts/train_churn_model.py`. Deliberately
kept as a script for now since this stage's early tasks are data-science exploration, not
API work — the API tasks (`POST /models/train`, etc.) come later in the stage, after the
scikit-learn exploration is settled.

**Known gotcha already handled:** `TotalCharges` reads as text from the raw CSV — 11 rows
have a blank/whitespace string (not a real `NaN`) for brand-new customers with `tenure=0`.
Fixed with `pd.to_numeric(df["TotalCharges"], errors="coerce")` before anything else.

**Next task:** "Train random forest" — then XGBoost, compare metrics, implement
cross-validation, persist model artifacts, model metadata record, and finally the API tasks
(`POST /models/train`, `GET /models/{id}`, `GET /models/{id}/metrics`,
`POST /models/{id}/predict`, inference validation/error handling).

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
