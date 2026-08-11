from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.db import get_db
from app.schemas.dataset import DatasetCreate, DatasetResponse
from app.services import dataset_service
from app.services.dataset_service import DatasetNotFoundError

router = APIRouter()


@router.get("/ping")
def ping():
    return {"status": "ok"}


@router.post("/datasets/test", response_model=DatasetResponse)
def create_dataset_test(payload: DatasetCreate):
    return DatasetResponse(id=1, name=payload.name, description=payload.description)


@router.post("/datasets", response_model=DatasetResponse, status_code=status.HTTP_201_CREATED)
async def create_dataset(payload: DatasetCreate, db: AsyncSession = Depends(get_db)):
    dataset = await dataset_service.create_dataset(db, name=payload.name, description=payload.description)
    return dataset


@router.get("/datasets", response_model=list[DatasetResponse])
async def list_datasets(db: AsyncSession = Depends(get_db)):
    return await dataset_service.list_datasets(db)


@router.get("/datasets/{dataset_id}", response_model=DatasetResponse)
async def get_dataset(dataset_id: int, db: AsyncSession = Depends(get_db)):
    try:
        return await dataset_service.get_dataset(db, dataset_id)
    except DatasetNotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))


@router.delete("/datasets/{dataset_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_dataset(dataset_id: int, db: AsyncSession = Depends(get_db)):
    try:
        await dataset_service.delete_dataset(db, dataset_id)
    except DatasetNotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
