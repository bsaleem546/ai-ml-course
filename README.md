# ai-ml

AI/ML Engineer build-first roadmap project.

This README is written as a step-by-step build log, in the order things were actually set up. If you need to rebuild this project from scratch, follow it top to bottom.

## Stage 0 — Production Python Foundation

### 1. Install uv

Windows PowerShell:

```bash
irm https://astral.sh/uv/install.ps1 | iex
```

Restart the terminal, then verify:

```bash
uv --version
```

### 2. Initialize the project

```bash
uv init
```

### 3. Pin the Python version

```bash
uv python install 3.13
uv python pin 3.13
```

> If `requires-python` in `pyproject.toml` says something else (e.g. `>=3.13`), pin to a version that satisfies it.

### 4. Add runtime dependencies

```bash
uv add --system-certs fastapi pydantic "uvicorn[standard]"
uv add --system-certs pydantic-settings
```

> **Note:** if you hit `invalid peer certificate: UnknownIssuer`, it's a local TLS/proxy issue. Always add `--system-certs` to `uv add`/`uv sync` on this machine.

### 5. Add dev dependencies

```bash
uv add --system-certs --dev pytest
```

### 6. Create the app structure

```
app/
  __init__.py
  main.py
  config.py
  db.py
  api/
    __init__.py
    v1/
      __init__.py
      routes.py
  schemas/
    __init__.py
    dataset.py
```

Every folder under `app/` needs an `__init__.py` (even if empty) or Python won't treat it as an importable package — this caused a `ModuleNotFoundError: No module named 'app.api'` early on.

`app/api/v1/routes.py`:

```python
from fastapi import APIRouter

from app.schemas.dataset import DatasetCreate, DatasetResponse

router = APIRouter()


@router.get("/ping")
def ping():
    return {"status": "ok"}


@router.post("/datasets/test", response_model=DatasetResponse)
def create_dataset_test(payload: DatasetCreate):
    return DatasetResponse(id=1, name=payload.name, description=payload.description)
```

`app/main.py`:

```python
from fastapi import FastAPI

from app.api.v1.routes import router as v1_router

app = FastAPI(title="AI/ML Engineer Platform")

app.include_router(v1_router, prefix="/api/v1")
```

`app/schemas/dataset.py`:

```python
from pydantic import BaseModel

class DatasetCreate(BaseModel):
    name: str
    description: str | None = None

class DatasetResponse(BaseModel):
    id: int
    name: str
    description: str | None = None
```

### 7. Run the app

```bash
uv run uvicorn app.main:app --reload
```

- Ping route: http://127.0.0.1:8000/api/v1/ping (note the `/api/v1` prefix — `/ping` alone 404s)
- Interactive docs: http://127.0.0.1:8000/docs

### 8. Configuration via environment variables

```bash
uv add --system-certs pydantic-settings
```

`app/config.py`:

```python
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    database_url: str = "postgresql://postgres:postgres@localhost:5432/ai_ml"
    redis_url: str = "redis://localhost:6379/0"
    environment: str = "development"


settings = Settings()
```

These are just fallback defaults. Any matching var in `.env` (case-insensitive: `DATABASE_URL` → `database_url`) overrides it. `.env` is gitignored — copy `.env.example` to `.env` and fill in real values:

```bash
cp .env.example .env
```

> Adding a new setting later? Add the field to the `Settings` class **and** the var to `.env` — vars not declared as fields are ignored (`extra="ignore"`).

#### Where to get DATABASE_URL and REDIS_URL (free hosted, no local install needed)

- **Postgres:** [Neon](https://neon.tech) — create a project, open the database, click **Connect**, copy the `postgresql://...` string into `DATABASE_URL`.
- **Redis:** [Upstash](https://upstash.com) — create a Redis database, on the **Details** tab reveal the token, then build the URL as:
  ```
  REDIS_URL=rediss://default:<token>@<endpoint>:6379
  ```
  (note `rediss://` with double "s" — Upstash requires TLS)

### 9. Database connection (async SQLAlchemy)

```bash
uv add --system-certs sqlalchemy asyncpg alembic
```

`app/db.py`:

```python
from collections.abc import AsyncGenerator

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase

from app.config import settings


def _to_async_url(url: str) -> str:
    if url.startswith("postgresql://"):
        url = url.replace("postgresql://", "postgresql+asyncpg://", 1)
    url, _, _ = url.partition("?")
    return url


engine = create_async_engine(_to_async_url(settings.database_url), connect_args={"ssl": "require"})
async_session_factory = async_sessionmaker(engine, expire_on_commit=False)


class Base(DeclarativeBase):
    pass


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    async with async_session_factory() as session:
        yield session
```

Why the URL munging: Neon's connection string starts with `postgresql://` and has query params (`?sslmode=require&channel_binding=require`) meant for `psycopg2`. The async `asyncpg` driver needs the `postgresql+asyncpg://` scheme and doesn't understand those query params, so they're stripped and SSL is passed explicitly via `connect_args`.

Sanity check the connection:

```bash
uv run python -c "
import asyncio
from app.db import engine
from sqlalchemy import text

async def main():
    async with engine.connect() as conn:
        result = await conn.execute(text('SELECT 1'))
        print('Connected:', result.scalar())

asyncio.run(main())
"
```

Should print `Connected: 1`.

### 10. Migrations with Alembic

```bash
uv run alembic init alembic
```

In `alembic/env.py`, point Alembic at `settings.database_url` instead of the placeholder in `alembic.ini` — add this right after `config = context.config`:

```python
from app.config import settings

config = context.config
config.set_main_option("sqlalchemy.url", settings.database_url)
```

(Use the plain `postgresql://` URL here, not `+asyncpg` — Alembic's default migration runner is sync and uses `psycopg2`.)

Create and apply migrations:

```bash
uv run alembic revision -m "init"
uv run alembic upgrade head
```

### Sync everything

Whenever dependencies are added, make sure they're installed:

```bash
uv sync
```

## Run

```bash
uv run uvicorn app.main:app --reload
```

- App: http://127.0.0.1:8000/api/v1/ping
- Docs: http://127.0.0.1:8000/docs

## Test

```bash
uv run pytest
```
