from fastapi import APIRouter, Depends, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.db import get_db
from app.schemas.dataset import DatasetCreate, DatasetResponse
from app.services import dataset_service

from sqlalchemy import text

router = APIRouter()


@router.get("/ping")
def ping():
    return {"status": "ok"}

@router.get("/health")
def health():
    return {"status": "ok"}

@router.get("/ready")
async def ready(db: AsyncSession = Depends(get_db)):
    await db.execute(text("SELECT 1"))
    return {"status": "ready"}

@router.post("/datasets", response_model=DatasetResponse, status_code=status.HTTP_201_CREATED)
async def create_dataset(payload: DatasetCreate, db: AsyncSession = Depends(get_db)):
    dataset = await dataset_service.create_dataset(db, name=payload.name, description=payload.description)
    return dataset


@router.get("/datasets", response_model=list[DatasetResponse])
async def list_datasets(db: AsyncSession = Depends(get_db)):
    return await dataset_service.list_datasets(db)


@router.get("/datasets/{dataset_id}", response_model=DatasetResponse)
async def get_dataset(dataset_id: int, db: AsyncSession = Depends(get_db)):
    return await dataset_service.get_dataset(db, dataset_id)


@router.delete("/datasets/{dataset_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_dataset(dataset_id: int, db: AsyncSession = Depends(get_db)):
    await dataset_service.delete_dataset(db, dataset_id)