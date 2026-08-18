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

from app.models.trained_model import TrainedModel
from app.schemas.model import ModelMetrics, ModelResponse, ModelTrainRequest, PredictRequest, PredictResponse
from app.services import model_service

from app.services import nn_training_job_service
from app.schemas.dataset import NnTrainingJobResponse

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


@router.post("/models/train", response_model=ModelResponse, status_code=status.HTTP_201_CREATED)
async def train_model(payload: ModelTrainRequest, db: AsyncSession = Depends(get_db)) -> TrainedModel:
    return await model_service.train_churn_model(db, dataset_id=payload.dataset_id)


@router.get("/models/{model_id}", response_model=ModelResponse)
async def get_model(model_id: int, db: AsyncSession = Depends(get_db)) -> TrainedModel:
    return await model_service.get_model(db, model_id)


@router.get("/models/{model_id}/metrics", response_model=ModelMetrics)
async def get_model_metrics(model_id: int, db: AsyncSession = Depends(get_db)) -> TrainedModel:
    return await model_service.get_model(db, model_id)


@router.post("/models/{model_id}/predict", response_model=PredictResponse)
async def predict(model_id: int, payload: PredictRequest, db: AsyncSession = Depends(get_db)) -> PredictResponse:
    model = await model_service.get_model(db, model_id)
    prediction, probability = model_service.predict_churn(model, payload.features)
    return PredictResponse(churn_prediction=prediction, churn_probability=probability)

@router.post("/models/train-nn", response_model=NnTrainingJobResponse, status_code=status.HTTP_202_ACCEPTED)
async def train_model_nn(
    payload: ModelTrainRequest,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
) -> NnTrainingJobResponse:
    job = await nn_training_job_service.create_job(db, dataset_id=payload.dataset_id)
    background_tasks.add_task(nn_training_job_service.run_training_job, job.id)
    return job


@router.get("/nn-jobs/{job_id}", response_model=NnTrainingJobResponse)
async def get_nn_job(job_id: int, db: AsyncSession = Depends(get_db)) -> NnTrainingJobResponse:
    job = await nn_training_job_service.get_job(db, job_id)
    return job