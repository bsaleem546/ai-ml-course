import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.trained_model import TrainedModel
from app.repositories import model_repository
from app.services import dataset_service

MODEL_DIR = Path("models")
NUMERIC_FEATURES = ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"]

logger = logging.getLogger(__name__)


class ModelNotFoundError(Exception):
    def __init__(self, model_id: int) -> None:
        self.model_id = model_id
        super().__init__(f"Model {model_id} not found")


async def get_model(db: AsyncSession, model_id: int) -> TrainedModel:
    model = await model_repository.get_by_id(db, model_id)
    if model is None:
        logger.warning("model not found id=%s", model_id)
        raise ModelNotFoundError(model_id)
    return model


async def train_churn_model(db: AsyncSession, dataset_id: int) -> TrainedModel:
    dataset = await dataset_service.get_dataset(db, dataset_id)
    df = pd.read_csv(dataset.storage_path)

    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df = df.drop(columns=["customerID"])
    y = df["Churn"].map({"Yes": 1, "No": 0})
    X = df.drop(columns=["Churn"])

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    categorical_features = [c for c in X.columns if c not in NUMERIC_FEATURES]
    preprocessor = ColumnTransformer(transformers=[
        ("numeric", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]), NUMERIC_FEATURES),
        ("categorical", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]), categorical_features),
    ])

    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(max_iter=1000)),
    ])
    pipeline.fit(X_train, y_train)
    preds = pipeline.predict(X_val)

    MODEL_DIR.mkdir(exist_ok=True)
    artifact_path = MODEL_DIR / f"{uuid.uuid4()}_churn_logreg.joblib"
    joblib.dump(pipeline, artifact_path)

    model = await model_repository.create(
        db,
        name="churn_logreg",
        model_type="LogisticRegression",
        dataset_id=dataset.id,
        artifact_path=str(artifact_path),
        accuracy=accuracy_score(y_val, preds),
        precision=precision_score(y_val, preds),
        recall=recall_score(y_val, preds),
        f1=f1_score(y_val, preds),
        created_at=datetime.now(timezone.utc),
    )
    logger.info("model trained id=%s dataset_id=%s accuracy=%.4f", model.id, dataset.id, model.accuracy)
    return model


class InvalidPredictionInputError(Exception):
    pass


def predict_churn(model: TrainedModel, features: dict) -> tuple[str, float]:
    pipeline = joblib.load(model.artifact_path)
    expected_columns = list(pipeline.feature_names_in_)

    missing = set(expected_columns) - set(features.keys())
    if missing:
        raise InvalidPredictionInputError(f"Missing required fields: {sorted(missing)}")

    unexpected = set(features.keys()) - set(expected_columns)
    if unexpected:
        raise InvalidPredictionInputError(f"Unknown fields: {sorted(unexpected)}")

    row = pd.DataFrame([{col: features[col] for col in expected_columns}])

    try:
        prediction = pipeline.predict(row)[0]
        probability = pipeline.predict_proba(row)[0][1]
    except (ValueError, TypeError) as e:
        raise InvalidPredictionInputError(f"Could not run prediction on given input: {e}")

    return ("Yes" if prediction == 1 else "No"), float(probability)