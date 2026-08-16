from datetime import datetime

from pydantic import BaseModel


class ModelTrainRequest(BaseModel):
    dataset_id: int


class ModelResponse(BaseModel):
    id: int
    name: str
    model_type: str
    dataset_id: int
    created_at: datetime


class ModelMetrics(BaseModel):
    accuracy: float
    precision: float
    recall: float
    f1: float
