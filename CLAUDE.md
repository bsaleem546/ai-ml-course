# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

A FastAPI backend built incrementally by following the build-first AI/ML Engineer roadmap in `ai_ml_engineer_build_first_roadmap.html` (an interactive, localStorage-backed checklist — open it in a browser to see/tick progress). Currently mid Stage 0 ("Production Python Foundation"). README.md is maintained as a step-by-step build log in the same order stages/tasks were implemented — check it before re-deriving how something was set up.

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
