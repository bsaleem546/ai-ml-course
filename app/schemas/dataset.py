from pydantic import BaseModel

class DatasetCreate(BaseModel):
    name: str
    description: str | None = None

class DatasetResponse(BaseModel):
    id: int
    name: str
    description: str | None = None