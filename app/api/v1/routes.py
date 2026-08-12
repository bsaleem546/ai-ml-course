from fastapi import APIRouter, Depends, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.db import get_db
from app.models.dataset import Dataset
from app.schemas.dataset import DatasetCreate, DatasetProfile, DatasetResponse
from app.services import dataset_service

from fastapi import File, Form, UploadFile

from sqlalchemy import text

router = APIRouter()


@router.get("/ping")
def ping() -> dict[str, str]: 
    return {"status": "ok"}

@router.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}

@router.get("/ready")
async def ready(db: AsyncSession = Depends(get_db)) -> dict[str, str]:
    await db.execute(text("SELECT 1"))
    return {"status": "ready"}

@router.post("/datasets", response_model=DatasetResponse, status_code=status.HTTP_201_CREATED)
async def create_dataset(payload: DatasetCreate, db: AsyncSession = Depends(get_db)) -> Dataset:
    dataset = await dataset_service.create_dataset(db, name=payload.name, description=payload.description)
    return dataset


@router.get("/datasets", response_model=list[DatasetResponse])
async def list_datasets(db: AsyncSession = Depends(get_db)) -> list[Dataset]:
    return await dataset_service.list_datasets(db)


@router.get("/datasets/{dataset_id}", response_model=DatasetResponse)
async def get_dataset(dataset_id: int, db: AsyncSession = Depends(get_db)) -> Dataset:
    return await dataset_service.get_dataset(db, dataset_id)


@router.delete("/datasets/{dataset_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_dataset(dataset_id: int, db: AsyncSession = Depends(get_db)) -> None:
    await dataset_service.delete_dataset(db, dataset_id)
    
    
@router.post("/datasets/upload", response_model=DatasetResponse, status_code=status.HTTP_201_CREATED)
async def upload_dataset(
    name: str = Form(...),
    file: UploadFile = File(...),
    db: AsyncSession = Depends(get_db),
) -> Dataset:
    content = await file.read()
    return await dataset_service.create_dataset_from_csv(
        db,
        name=name,
        filename=file.filename,
        content_type=file.content_type,
        content=content,
    )
    
    
@router.get("/datasets/{dataset_id}/profile", response_model=DatasetProfile)
async def get_dataset_profile(dataset_id: int, db: AsyncSession = Depends(get_db)) -> DatasetProfile:
    dataset = await dataset_service.get_dataset(db, dataset_id)
    return dataset_service.profile_dataset(dataset)