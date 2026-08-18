import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import torch.nn as nn

DATA_PATH = "data/telco_churn.csv"

df = pd.read_csv(DATA_PATH)
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df = df.drop(columns=["customerID"])

y = df["Churn"].map({"Yes": 1, "No": 0})
X = df.drop(columns=["Churn"])

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

numeric_features = ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"]
categorical_features = [c for c in X.columns if c not in numeric_features]

preprocessor = ColumnTransformer(transformers=[
    ("numeric", Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ]), numeric_features),
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


class ChurnDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


train_dataset = ChurnDataset(X_train_tensor, y_train_tensor)
val_dataset = ChurnDataset(X_val_tensor, y_val_tensor)

BATCH_SIZE = 32

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

print(f"train batches: {len(train_loader)}, val batches: {len(val_loader)}")

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
    
model = ChurnNet(input_dim=X_train_tensor.shape[1])
print(model)

criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    total_loss = 0.0
    for X_batch, y_batch in loader:
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * X_batch.size(0)
    return total_loss / len(loader.dataset)


def evaluate(model, loader, criterion):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for X_batch, y_batch in loader:
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            total_loss += loss.item() * X_batch.size(0)
    return total_loss / len(loader.dataset)


EPOCHS = 20
for epoch in range(1, EPOCHS + 1):
    train_loss = train_one_epoch(model, train_loader, criterion, optimizer)
    val_loss = evaluate(model, val_loader, criterion)
    print(f"Epoch {epoch}/{EPOCHS} — train_loss: {train_loss:.4f}, val_loss: {val_loss:.4f}")
    
    
def run_experiment(batch_size, lr, epochs=20):
    trial_model = ChurnNet(input_dim=X_train_tensor.shape[1])
    trial_criterion = nn.BCEWithLogitsLoss()
    trial_optimizer = torch.optim.Adam(trial_model.parameters(), lr=lr)
    trial_train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    trial_val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    best_val_loss = float("inf")
    for epoch in range(1, epochs + 1):
        train_one_epoch(trial_model, trial_train_loader, trial_criterion, trial_optimizer)
        val_loss = evaluate(trial_model, trial_val_loader, trial_criterion)
        best_val_loss = min(best_val_loss, val_loss)
    return best_val_loss


print("\n=== Batch size sweep (lr=0.001) ===")
for bs in [8, 32, 128, 512]:
    best_val = run_experiment(batch_size=bs, lr=0.001)
    print(f"batch_size={bs}: best val_loss={best_val:.4f}")

print("\n=== Learning rate sweep (batch_size=32) ===")
for lr in [0.0001, 0.001, 0.01, 0.1]:
    best_val = run_experiment(batch_size=32, lr=lr)
    print(f"lr={lr}: best val_loss={best_val:.4f}")