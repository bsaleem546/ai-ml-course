from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.ingestion_job import IngestionJob


async def create(db: AsyncSession, dataset_id: int) -> IngestionJob:
    job = IngestionJob(dataset_id=dataset_id, status="queued")
    db.add(job)
    await db.commit()
    await db.refresh(job)
    return job


async def get_by_id(db: AsyncSession, job_id: int) -> IngestionJob | None:
    result = await db.execute(select(IngestionJob).where(IngestionJob.id == job_id))
    return result.scalar_one_or_none()


async def update_status(
    db: AsyncSession,
    job: IngestionJob,
    status: str,
    error_message: str | None = None,
    profile_json: str | None = None,
) -> IngestionJob:
    job.status = status
    job.error_message = error_message
    job.profile_json = profile_json
    db.add(job)
    await db.commit()
    await db.refresh(job)
    return job