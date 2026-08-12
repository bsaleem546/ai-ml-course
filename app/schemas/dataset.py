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

class ColumnProfile(BaseModel):
    name: str
    dtype: str  # "numeric" | "categorical" | "text"
    missing_count: int
    missing_percentage: float
    unique_count: int
    min: float | None = None
    max: float | None = None
    mean: float | None = None
    median: float | None = None
    std: float | None = None


class DatasetProfile(BaseModel):
    row_count: int
    column_count: int
    columns: list[ColumnProfile]