from fastapi import APIRouter, Depends, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.db import get_db
from app.models.dataset import Dataset
from app.models.ingestion_job import IngestionJob
from app.schemas.dataset import DatasetCreate, DatasetProfile, DatasetResponse
from app.services import dataset_service

from fastapi import File, Form, UploadFile

from sqlalchemy import text

from fastapi import BackgroundTasks

from app.schemas.dataset import IngestionJobResponse
from app.services import job_service

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
    
    
@router.get("/datasets/{dataset_id}/profile", response_model=DatasetProfile)
async def get_dataset_profile(dataset_id: int, db: AsyncSession = Depends(get_db)) -> DatasetProfile:
    dataset = await dataset_service.get_dataset(db, dataset_id)
    return dataset_service.profile_dataset(dataset)


@router.post("/datasets/upload", response_model=IngestionJobResponse, status_code=status.HTTP_202_ACCEPTED)
async def upload_dataset(
    background_tasks: BackgroundTasks,
    name: str = Form(...),
    file: UploadFile = File(...),
    db: AsyncSession = Depends(get_db),
) -> IngestionJob:
    content = await file.read()
    dataset = await dataset_service.create_dataset_from_csv(
        db,
        name=name,
        filename=file.filename,
        content_type=file.content_type,
        content=content,
    )
    job = await job_service.create_job(db, dataset_id=dataset.id)
    background_tasks.add_task(job_service.run_ingestion_job, job.id)
    return job


@router.get("/jobs/{job_id}", response_model=IngestionJobResponse)
async def get_job(job_id: int, db: AsyncSession = Depends(get_db)) -> IngestionJob:
    return await job_service.get_job(db, job_id)