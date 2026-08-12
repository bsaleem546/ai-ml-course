import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy import delete

from app.db import async_session_factory
from app.main import app
from app.models.dataset import Dataset
from app.models.ingestion_job import IngestionJob


@pytest.fixture
async def client():
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as ac:
        yield ac

    async with async_session_factory() as session:
        await session.execute(delete(IngestionJob))
        await session.execute(delete(Dataset))
        await session.commit()

