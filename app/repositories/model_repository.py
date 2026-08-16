from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.trained_model import TrainedModel


async def create(
    db: AsyncSession,
    name: str,
    model_type: str,
    dataset_id: int,
    artifact_path: str,
    accuracy: float,
    precision: float,
    recall: float,
    f1: float,
    created_at,
) -> TrainedModel:
    model = TrainedModel(
        name=name,
        model_type=model_type,
        dataset_id=dataset_id,
        artifact_path=artifact_path,
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        f1=f1,
        created_at=created_at,
    )
    db.add(model)
    await db.commit()
    await db.refresh(model)
    return model


async def get_by_id(db: AsyncSession, model_id: int) -> TrainedModel | None:
    result = await db.execute(select(TrainedModel).where(TrainedModel.id == model_id))
    return result.scalar_one_or_none()