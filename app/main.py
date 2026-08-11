from fastapi import FastAPI

from app.api.v1.routes import router as v1_router

from app.logging_config import setup_logging

setup_logging()

app = FastAPI(title="AI/ML Engineer Platform")

app.include_router(v1_router, prefix="/api/v1")