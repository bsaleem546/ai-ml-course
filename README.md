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

Alembic's sync runner needs the `psycopg2` driver installed too (separate from `asyncpg`, which the app uses):

```bash
uv add --system-certs psycopg2-binary
```

Create and apply migrations:

```bash
uv run alembic revision -m "init"
uv run alembic upgrade head
```

> `ERROR: Target database is not up to date`: means a migration was created but never applied. Run `uv run alembic upgrade head` before creating the next revision.

### 11. Dataset model + autogenerated migrations

For `alembic revision --autogenerate` to detect model changes, `alembic/env.py` needs `target_metadata` pointed at your models' `Base.metadata` (it defaults to `None`). Setting `target_metadata = Base.metadata` alone isn't enough — each model also has to actually be **imported** somewhere so it registers itself on `Base.metadata` (otherwise `Base.metadata` is empty and you get `Can't proceed with --autogenerate option; environment script ... does not provide a MetaData object`):

```python
from app.db import Base
from app.models.dataset import Dataset  # noqa: F401 — registers Dataset on Base.metadata

target_metadata = Base.metadata
```

`app/models/dataset.py`:

```python
from sqlalchemy import String
from sqlalchemy.orm import Mapped, mapped_column

from app.db import Base


class Dataset(Base):
    __tablename__ = "datasets"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String, nullable=False)
    description: Mapped[str | None] = mapped_column(String, nullable=True)
```

Generate and apply the migration:

```bash
uv run alembic revision --autogenerate -m "create datasets table"
uv run alembic upgrade head
```

### Cleaning up a messy migration chain (squashing no-op revisions)

Re-running `alembic revision`/`--autogenerate` with no actual model changes creates an empty no-op migration (`upgrade()`/`downgrade()` both just `pass`). This happened twice here, leaving a 4-revision chain (`init` → `create datasets table` → `init` → `create datasets table`) where only one revision actually did anything.

**Constraint:** real databases already had `alembic_version` stamped at the current head — renaming/renumbering revisions would make Alembic think those databases need re-migrating (or worse, not recognize the stored version at all). The fix is to **squash down to a single file that keeps the existing head's revision id**, so already-migrated databases stay "at head" with zero extra commands, while any fresh database gets one clean migration instead of four:

1. Delete every migration file except the one matching the current head id.
2. In that surviving file, set `down_revision` to `None` (it's now the root) and copy the real `upgrade()`/`downgrade()` bodies over from whichever old revision actually did the work.
3. Verify:
   ```bash
   uv run alembic heads      # should show exactly one head
   uv run alembic history    # should show <base> -> <head-id>
   ```
4. Confirm existing databases don't try to re-run anything:
   ```bash
   uv run alembic upgrade head
   ```
   No `Running upgrade` log line printed = already recognized as up to date.
5. Confirm a genuinely fresh database still builds correctly from the single squashed migration (e.g. via `docker compose down -v` to wipe the local Postgres volume, then `docker compose up --build -d` + `docker compose exec api uv run alembic upgrade head`).

> **Gotcha:** if testing this through Docker, remember `docker compose exec` runs against whatever image was last built — deleting/editing migration files on the host doesn't reach a running container until you `docker compose up --build -d` again (same as any other code change).

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

### Unit test dependencies

```bash
uv add --system-certs --dev pytest-asyncio pytest-mock
```

`asyncio_mode = "auto"` is set under `[tool.pytest.ini_options]` in `pyproject.toml` so async test functions run without needing an `@pytest.mark.asyncio` decorator on each one.

### Integration test dependencies

```bash
uv add --system-certs --dev httpx
```

Integration tests (`tests/integration/`) run real requests through the FastAPI app into the actual Neon database, using `httpx.AsyncClient` + `ASGITransport`. `tests/integration/conftest.py`'s `client` fixture deletes all rows from `datasets` after each test so nothing persists.

> **Gotcha:** `app/db.py`'s `engine` is created once at import time, and asyncpg connections are bound to whichever event loop created them. `pytest-asyncio`'s default is a **new event loop per test function**, which breaks any test after the first with `asyncpg.exceptions._base.InterfaceError: cannot perform operation: another operation is in progress`. Fixed by forcing one shared event loop for the whole test session in `pyproject.toml`:
> ```toml
> [tool.pytest.ini_options]
> asyncio_mode = "auto"
> asyncio_default_fixture_loop_scope = "session"
> asyncio_default_test_loop_scope = "session"
> ```

```bash
uv run pytest
```

## Docker

`Dockerfile` builds the API image using `uv` (same tool as local dev). `.dockerignore` excludes `.venv`, `.git`, `.env`, and caches from the build context — secrets are never baked into the image, they're passed in at run time.

Build the image:

```bash
docker build -t ai-ml-course .
```

Run it, passing local `.env` values in (points at the same hosted Neon/Upstash used locally):

```bash
docker run --env-file .env -p 8000:8000 ai-ml-course
```

- App: http://127.0.0.1:8000/api/v1/ping
- Docs: http://127.0.0.1:8000/docs

## Docker Compose (API + local Postgres + local Redis)

`docker-compose.yml` runs the full stack locally: the API (built from `Dockerfile`), a local Postgres 16 container, and a local Redis 7 container — no dependency on hosted Neon/Upstash. The `api` service's `environment:` block overrides `.env` with container-network URLs (service names like `postgres`/`redis` resolve via Docker's internal DNS) and sets `DB_SSL_REQUIRED: "false"`, since the local Postgres container doesn't have TLS configured (unlike Neon, which requires it — see `db_ssl_required` in `app/config.py` and `app/db.py`).

Build and start everything (detached):

```bash
docker compose up --build -d
```

First run only (or after wiping the `postgres_data` volume) — the local Postgres starts empty, so apply migrations inside the running `api` container:

```bash
docker compose exec api uv run alembic upgrade head
```

Check container status:

```bash
docker compose ps
```

Tail logs (all services, or a specific one):

```bash
docker compose logs -f
docker compose logs -f api
```

Stop everything (keeps the `postgres_data` volume, so data persists):

```bash
docker compose down
```

Stop everything **and** wipe the Postgres volume (fresh empty DB next time):

```bash
docker compose down -v
```

Verify it's working:

```bash
curl http://127.0.0.1:8000/api/v1/ping
curl -X POST http://127.0.0.1:8000/api/v1/datasets -H "Content-Type: application/json" -d '{"name":"docker-test"}'
curl http://127.0.0.1:8000/api/v1/datasets
```

> **Port conflicts:** if `docker compose up` fails with `port is already allocated` (5432, 6379, or 8000), something on your host is already using that port — check with `sudo lsof -i :<port>`, then either stop that process or change the host-side port in `docker-compose.yml`'s `ports:` mapping (e.g. `"5433:5432"` — only the first number, the host port, needs to change).
