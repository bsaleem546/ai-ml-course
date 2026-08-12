from pydantic import BaseModel


class DatasetCreate(BaseModel):
    name: str
    description: str | None = None


class DatasetResponse(BaseModel):
    id: int
    name: str
    description: str | None = None
    filename: str | None = None
    content_type: str | None = None
    size_bytes: int | None = None
