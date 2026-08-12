from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.dataset import Dataset


async def create(db: AsyncSession, name: str, description: str | None) -> Dataset:
    dataset = Dataset(name=name, description=description)
    db.add(dataset)
    await db.commit()
    await db.refresh(dataset)
    return dataset


async def get_all(db: AsyncSession) -> list[Dataset]:
    result = await db.execute(select(Dataset))
    return list(result.scalars().all())


async def get_by_id(db: AsyncSession, dataset_id: int) -> Dataset | None:
    result = await db.execute(select(Dataset).where(Dataset.id == dataset_id))
    return result.scalar_one_or_none()


async def delete(db: AsyncSession, dataset: Dataset) -> None:
    await db.delete(dataset)
    await db.commit()
    
    
async def create_with_file(
    db: AsyncSession,
    name: str,
    filename: str,
    content_type: str,
    size_bytes: int,
    storage_path: str,
) -> Dataset:
    dataset = Dataset(
        name=name,
        filename=filename,
        content_type=content_type,
        size_bytes=size_bytes,
        storage_path=storage_path,
    )
    db.add(dataset)
    await db.commit()
    await db.refresh(dataset)
    return dataset