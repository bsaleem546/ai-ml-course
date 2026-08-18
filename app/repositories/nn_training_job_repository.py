from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.nn_training_job import NnTrainingJob
async def create(db: AsyncSession, dataset_id: int) -> NnTrainingJob:
    job = NnTrainingJob(dataset_id=dataset_id, status="queued")
    db.add(job)
    await db.commit()
    await db.refresh(job)
    return job


async def get_by_id(db: AsyncSession, job_id: int) -> NnTrainingJob | None:
    result = await db.execute(select(NnTrainingJob).where(NnTrainingJob.id == job_id))
    return result.scalar_one_or_none()


async def update_status(
    db: AsyncSession,
    job: NnTrainingJob,
    status: str,
    error_message: str | None = None,
    artifact_path: str | None = None,
    accuracy: float | None = None,
    precision: float | None = None,
    recall: float | None = None,
    f1: float | None = None,
) -> NnTrainingJob:
    job.status = status
    job.error_message = error_message
    job.artifact_path = artifact_path
    job.accuracy = accuracy
    job.precision = precision
    job.recall = recall
    job.f1 = f1
    db.add(job)
    await db.commit()
    await db.refresh(job)
    return job