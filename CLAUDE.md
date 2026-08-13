# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

A FastAPI backend built incrementally by following the build-first AI/ML Engineer roadmap in `ai_ml_engineer_build_first_roadmap.html` (an interactive, localStorage-backed checklist — open it in a browser to see/tick progress). README.md is maintained as a step-by-step build log in the same order stages/tasks were implemented — check it before re-deriving how something was set up. **`PROGRESS.md` is the authoritative source of truth for current position in the roadmap** (stage, task, what's done/next) — read it at the start of a session before asking the user what's next; it's kept up to date automatically (see "Progress tracking" below) and survives across machines/sessions in a way the HTML's localStorage checkboxes and this chat history do not.

## Learning mode (always in effect)

The user is doing this project specifically to learn AI/ML and Python — not to have a working app produced for them. Because of this:

- **Do not write or edit code unless explicitly told to** ("you do it," "fix it," "apply that"). Default response to any task, bug, or next roadmap step is *guidance*: explain what to do, which file, and why — in enough detail that the user can type it themselves — then wait for them to report back.
- Reading, running commands, and verifying (tests, curl, logs, `docker compose exec`, etc.) to check the user's own work or diagnose an error is fine and expected — the restriction is on writing/editing files, not on investigation.
- When the user reports a result (test output, an error, a curl response), check it against what was expected and explain what it means — don't just move on.
- This applies project-wide, not just to app code — scripts, tests, config, migrations, everything.

## Progress tracking (do this automatically, without being asked)

Whenever a roadmap task is verified complete in conversation (tested/confirmed working, not just written):

1. **Update the roadmap HTML's seed block** near the bottom of `ai_ml_engineer_build_first_roadmap.html` (`SEED_KEY` constant + `stageNDone` arrays): add the task's index to the relevant stage's array, and bump the `SEED_KEY` version string (e.g. `v15` → `v16`) so the seed re-fires on next page load — it only runs once per key. This is how completed tasks show up checked in a fresh browser/machine with empty `localStorage`.
2. **Update `PROGRESS.md`** — move the task from "next" to "done," update the task-count fraction, and update the "Next task" pointer. Keep the "Known gotchas" / "Working conventions" sections current if something new and non-obvious was learned.

Do both without waiting for the user to ask "update the html" / "update progress" — they've asked for this to happen automatically every time, on every machine this project is worked on from (enforced by this file being committed to git, not by any session-specific memory).

## Commands

This project uses `uv` for all dependency and environment management.

```bash
# Add a dependency (always use --system-certs on this machine — see note below)
uv add --system-certs <package>
uv add --system-certs --dev <package>

# Install/sync dependencies after pulling changes
uv sync

# Run the API with auto-reload
uv run uvicorn app.main:app --reload

# Run tests
uv run pytest

# Alembic migrations
uv run alembic revision -m "<message>"
uv run alembic upgrade head
```

App runs at `http://127.0.0.1:8000`; routes are mounted under `/api/v1` (e.g. `/api/v1/ping`), interactive docs at `/docs`.

**TLS note:** `uv add`/`uv sync` fail with `invalid peer certificate: UnknownIssuer` on this machine without `--system-certs` — always include that flag.

## Architecture

- `app/main.py` — FastAPI app instance; includes versioned routers under `/api/v1`.
- `app/api/v1/routes.py` — route definitions for API v1.
- `app/schemas/` — Pydantic request/response models.
- `app/config.py` — `Settings` (pydantic-settings) loaded from `.env`, with in-code defaults as fallback. Adding a new env var requires adding a matching field to `Settings` *and* the var in `.env` — fields not declared are ignored (`extra="ignore"`).
- `app/db.py` — async SQLAlchemy engine/session (`asyncpg` driver) and `Base` declarative class. `get_db()` is the FastAPI dependency for a request-scoped `AsyncSession`.
- `alembic/` — migrations, configured in `alembic/env.py` to read the DB URL from `app.config.settings` rather than `alembic.ini`. Alembic's own migration runner is sync (`psycopg2`-style), so it uses the plain `postgresql://` URL, while the app itself uses `postgresql+asyncpg://` — see `_to_async_url()` in `app/db.py` for the scheme/query-string handling this requires (Neon's connection string includes `sslmode`/`channel_binding` query params that `asyncpg` doesn't accept, so they're stripped and SSL is passed via `connect_args` instead).
- Every package directory under `app/` needs an `__init__.py` (even empty) — omitting one breaks `from app.x.y import ...` with `ModuleNotFoundError`.
- `src/ai_ml/` is a leftover stub from `uv init` and is not part of the actual application (the app lives under `app/`).

## Infrastructure

No local Postgres/Redis — both are hosted:
- Postgres: Neon (`DATABASE_URL` in `.env`)
- Redis: Upstash, TLS required (`REDIS_URL` must use `rediss://`, not `redis://`)

`.env` is gitignored; copy `.env.example` and fill in real values.
