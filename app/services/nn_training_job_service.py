import logging

import pandas as pd
import torch
import torch.nn as nn
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sqlalchemy.ext.asyncio import AsyncSession
from torch.utils.data import DataLoader, Dataset

from app.db import async_session_factory
from app.models.nn_training_job import NnTrainingJob
from app.repositories import nn_training_job_repository
from app.services import dataset_service

logger = logging.getLogger(__name__)
 
NUMERIC_FEATURES = ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"]
MODEL_DIR = "models"


class ChurnDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class ChurnNet(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.layer1 = nn.Linear(input_dim, 32)
        self.relu1 = nn.ReLU()
        self.layer2 = nn.Linear(32, 16)
        self.relu2 = nn.ReLU()
        self.output = nn.Linear(16, 1)

    def forward(self, x):
        x = self.relu1(self.layer1(x))
        x = self.relu2(self.layer2(x))
        x = self.output(x)
        return x


async def create_job(db: AsyncSession, dataset_id: int) -> NnTrainingJob:
    return await nn_training_job_repository.create(db, dataset_id=dataset_id)

class NnTrainingJobNotFoundError(Exception):
    def __init__(self, job_id: int) -> None:
        self.job_id = job_id
        super().__init__(f"NN training job {job_id} not found")


async def get_job(db: AsyncSession, job_id: int) -> NnTrainingJob:
    job = await nn_training_job_repository.get_by_id(db, job_id)
    if job is None:
        raise NnTrainingJobNotFoundError(job_id)
    return job


async def run_training_job(job_id: int) -> None:
    async with async_session_factory() as db:
        job = await nn_training_job_repository.get_by_id(db, job_id)
        if job is None or job.status != "queued":
            return

        await nn_training_job_repository.update_status(db, job, status="running")

        try:
            dataset = await dataset_service.get_dataset(db, job.dataset_id)
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

            X_train_processed = preprocessor.fit_transform(X_train)
            X_val_processed = preprocessor.transform(X_val)

            X_train_tensor = torch.tensor(X_train_processed, dtype=torch.float32)
            y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1)
            X_val_tensor = torch.tensor(X_val_processed, dtype=torch.float32)
            y_val_tensor = torch.tensor(y_val.values, dtype=torch.float32).unsqueeze(1)

            train_loader = DataLoader(ChurnDataset(X_train_tensor, y_train_tensor), batch_size=32, shuffle=True)
            val_loader = DataLoader(ChurnDataset(X_val_tensor, y_val_tensor), batch_size=32, shuffle=False)

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = ChurnNet(input_dim=X_train_tensor.shape[1])
            model.to(device)
            criterion = nn.BCEWithLogitsLoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

            artifact_path = f"{MODEL_DIR}/nn_job_{job_id}_best.pt"
            best_val_loss = float("inf")
            patience = 3
            epochs_without_improvement = 0

            for epoch in range(1, 21):
                model.train()
                for X_batch, y_batch in train_loader:
                    X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                    optimizer.zero_grad()
                    outputs = model(X_batch)
                    loss = criterion(outputs, y_batch)
                    loss.backward()
                    optimizer.step()

                model.eval()
                total_loss = 0.0
                with torch.no_grad():
                    for X_batch, y_batch in val_loader:
                        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                        outputs = model(X_batch)
                        loss = criterion(outputs, y_batch)
                        total_loss += loss.item() * X_batch.size(0)
                val_loss = total_loss / len(val_loader.dataset)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    epochs_without_improvement = 0
                    torch.save(model.state_dict(), artifact_path)
                else:
                    epochs_without_improvement += 1

                if epochs_without_improvement >= patience:
                    break

            model.load_state_dict(torch.load(artifact_path))
            model.eval()
            all_preds, all_labels = [], []
            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                    probs = torch.sigmoid(model(X_batch))
                    all_preds.append((probs > 0.5).float())
                    all_labels.append(y_batch)
            all_preds = torch.cat(all_preds).numpy()
            all_labels = torch.cat(all_labels).numpy()

            await nn_training_job_repository.update_status(
                db, job, status="completed",
                artifact_path=artifact_path,
                accuracy=accuracy_score(all_labels, all_preds),
                precision=precision_score(all_labels, all_preds),
                recall=recall_score(all_labels, all_preds),
                f1=f1_score(all_labels, all_preds),
            )
            logger.info("nn training job completed id=%s dataset_id=%s", job.id, job.dataset_id)
        except Exception as e:
            await nn_training_job_repository.update_status(db, job, status="failed", error_message=str(e))
            logger.exception("nn training job failed id=%s dataset_id=%s", job.id, job.dataset_id)
            
class NnTrainingJobNotFoundError(Exception):
    def __init__(self, job_id: int) -> None:
        self.job_id = job_id
        super().__init__(f"NN training job {job_id} not found")


async def get_job(db: AsyncSession, job_id: int) -> NnTrainingJob:
    job = await nn_training_job_repository.get_by_id(db, job_id)
    if job is None:
        raise NnTrainingJobNotFoundError(job_id)
    return job