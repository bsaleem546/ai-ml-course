import logging

from sqlalchemy.ext.asyncio import AsyncSession

from app.db import async_session_factory
from app.models.ingestion_job import IngestionJob
from app.repositories import job_repository
from app.schemas.dataset import JobStatus
from app.services import dataset_service, profile_service
import pandas as pd

logger = logging.getLogger(__name__)


async def create_job(db: AsyncSession, dataset_id: int) -> IngestionJob:
    return await job_repository.create(db, dataset_id=dataset_id)


async def run_ingestion_job(job_id: int) -> None:
    async with async_session_factory() as db:
        job = await job_repository.get_by_id(db, job_id)
        if job is None or job.status != JobStatus.QUEUED:
            return

        await job_repository.update_status(db, job, status=JobStatus.RUNNING)

        try:
            dataset = await dataset_service.get_dataset(db, job.dataset_id)
            df = pd.read_csv(dataset.storage_path)
            profile = profile_service.build_profile(df)
            await job_repository.update_status(
                db, job, status=JobStatus.COMPLETED, profile_json=profile.model_dump_json()
            )
            logger.info("job completed id=%s dataset_id=%s", job.id, job.dataset_id)
        except Exception as e:
            await job_repository.update_status(db, job, status=JobStatus.FAILED, error_message=str(e))
            logger.exception("job failed id=%s dataset_id=%s", job.id, job.dataset_id)
            

class JobNotFoundError(Exception):
    def __init__(self, job_id: int) -> None:
        self.job_id = job_id
        super().__init__(f"Job {job_id} not found")


async def get_job(db: AsyncSession, job_id: int) -> IngestionJob:
    job = await job_repository.get_by_id(db, job_id)
    if job is None:
        raise JobNotFoundError(job_id)
    return job