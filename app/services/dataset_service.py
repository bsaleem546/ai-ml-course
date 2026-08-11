import logging

from sqlalchemy.ext.asyncio import AsyncSession

from app.models.dataset import Dataset
from app.repositories import dataset_repository

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