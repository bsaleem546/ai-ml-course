import logging

from sqlalchemy.ext.asyncio import AsyncSession

from app.models.dataset import Dataset
from app.repositories import dataset_repository

import uuid
from pathlib import Path

UPLOAD_DIR = Path("uploads")
MAX_UPLOAD_SIZE_BYTES = 10 * 1024 * 1024  # 10 MB

class InvalidFileError(Exception):
    pass

logger = logging.getLogger(__name__)

class DatasetNotFoundError(Exception):
    def __init__(self, dataset_id: int) -> None:
        self.dataset_id = dataset_id
        super().__init__(f"Dataset {dataset_id} not found")


async def create_dataset(db: AsyncSession, name: str, description: str | None) -> Dataset:
    dataset = await dataset_repository.create(db, name=name, description=description)
    logger.info("dataset created id=%s name=%s", dataset.id, dataset.name)
    return dataset


async def list_datasets(db: AsyncSession) -> list[Dataset]:
    return await dataset_repository.get_all(db)


async def get_dataset(db: AsyncSession, dataset_id: int) -> Dataset:
    dataset = await dataset_repository.get_by_id(db, dataset_id)
    if dataset is None:
        logger.warning("dataset not found id=%s", dataset_id)
        raise DatasetNotFoundError(dataset_id)
    return dataset


async def delete_dataset(db: AsyncSession, dataset_id: int) -> None:
    dataset = await get_dataset(db, dataset_id)
    logger.info("dataset deleted id=%s name=%s", dataset.id, dataset.name)
    await dataset_repository.delete(db, dataset)
    
    
async def create_dataset_from_csv(
    db: AsyncSession,
    name: str,
    filename: str,
    content_type: str,
    content: bytes,
) -> Dataset:
    if not filename.lower().endswith(".csv"):
        raise InvalidFileError(f"Expected a .csv file, got: {filename}")

    if len(content) > MAX_UPLOAD_SIZE_BYTES:
        raise InvalidFileError(
            f"File too large: {len(content)} bytes (max {MAX_UPLOAD_SIZE_BYTES})"
        )

    UPLOAD_DIR.mkdir(exist_ok=True)
    storage_path = UPLOAD_DIR / f"{uuid.uuid4()}_{filename}"
    storage_path.write_bytes(content)

    dataset = await dataset_repository.create_with_file(
        db,
        name=name,
        filename=filename,
        content_type=content_type,
        size_bytes=len(content),
        storage_path=str(storage_path),
    )
    logger.info("dataset uploaded id=%s filename=%s size=%s", dataset.id, filename, len(content))
    return dataset