import logging

from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse

from app.api.v1.routes import router as v1_router
from app.logging_config import setup_logging
from app.services.dataset_service import DatasetNotFoundError

logger = logging.getLogger(__name__)

setup_logging()

app = FastAPI(title="AI/ML Engineer Platform")

app.include_router(v1_router, prefix="/api/v1")

@app.exception_handler(DatasetNotFoundError)
async def dataset_not_found_handler(request: Request, exc: DatasetNotFoundError) -> JSONResponse:
    return JSONResponse(
        status_code=status.HTTP_404_NOT_FOUND,
        content={"detail": str(exc)},
    )

@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.exception("unhandled exception on %s %s", request.method, request.url.path)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "Internal server error"},
    )