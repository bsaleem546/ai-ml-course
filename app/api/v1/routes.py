from fastapi import APIRouter

from app.schemas.dataset import DatasetCreate, DatasetResponse

router = APIRouter()


@router.get("/ping")
def ping():
    return {"status": "ok"}


@router.post("/datasets/test", response_model=DatasetResponse)
def create_dataset_test(payload: DatasetCreate):
    return DatasetResponse(id=1, name=payload.name, description=payload.description)