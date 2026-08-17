import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

import numpy as np

DATA_PATH = "data/telco_churn.csv"

df = pd.read_csv(DATA_PATH)
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df = df.drop(columns=["customerID"])

y = df["Churn"].map({"Yes": 1, "No": 0})
X = df.drop(columns=["Churn"])

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.30, stratify=y, random_state=42
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

# Deliberately overfit: no depth limit, no minimum samples per leaf —
# the tree can keep splitting until it perfectly separates the training data.
overfit_pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", DecisionTreeClassifier(max_depth=None, min_samples_leaf=1, random_state=42)),
])
overfit_pipeline.fit(X_train, y_train)

train_acc = accuracy_score(y_train, overfit_pipeline.predict(X_train))
val_acc = accuracy_score(y_val, overfit_pipeline.predict(X_val))

print("=== Overfit decision tree (unbounded depth) ===")
print(f"Train accuracy: {train_acc:.4f}")
print(f"Val accuracy:   {val_acc:.4f}")
print(f"Gap:            {train_acc - val_acc:.4f}")


print("\n=== Complexity sweep (max_depth) ===")
depths = [1, 2, 3, 5, 10, 20, None]
complexity_results = []

for depth in depths:
    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", DecisionTreeClassifier(max_depth=depth, min_samples_leaf=1, random_state=42)),
    ])
    pipeline.fit(X_train, y_train)

    train_acc = accuracy_score(y_train, pipeline.predict(X_train))
    val_acc = accuracy_score(y_val, pipeline.predict(X_val))

    complexity_results.append({
        "max_depth": depth if depth is not None else "unbounded",
        "train_acc": train_acc,
        "val_acc": val_acc,
        "gap": train_acc - val_acc,
    })

complexity_df = pd.DataFrame(complexity_results)
print(complexity_df.to_string(index=False))

print("\n=== Underfit decision tree (max_depth=1) ===")
underfit_pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", DecisionTreeClassifier(max_depth=1, random_state=42)),
])
underfit_pipeline.fit(X_train, y_train)

underfit_train_acc = accuracy_score(y_train, underfit_pipeline.predict(X_train))
underfit_val_acc = accuracy_score(y_val, underfit_pipeline.predict(X_val))

print(f"Train accuracy: {underfit_train_acc:.4f}")
print(f"Val accuracy:   {underfit_val_acc:.4f}")
print(f"Gap:            {underfit_train_acc - underfit_val_acc:.4f}")
print("Baseline (majority-class) accuracy from Stage 2: 0.7348")

print("\n=== Feature engineering: does it fix underfitting? ===")

service_cols = ["OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"]

X_train_fe = X_train.copy()
X_val_fe = X_val.copy()

for data in (X_train_fe, X_val_fe):
    data["NumServices"] = (data[service_cols] == "Yes").sum(axis=1)
    data["AvgMonthlyCharge"] = data["TotalCharges"] / data["tenure"].replace(0, 1)

numeric_features_fe = numeric_features + ["NumServices", "AvgMonthlyCharge"]
preprocessor_fe = ColumnTransformer(transformers=[
    ("numeric", Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ]), numeric_features_fe),
    ("categorical", Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ]), categorical_features),
])

fe_pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor_fe),
    ("classifier", DecisionTreeClassifier(max_depth=1, random_state=42)),
])
fe_pipeline.fit(X_train_fe, y_train)

fe_train_acc = accuracy_score(y_train, fe_pipeline.predict(X_train_fe))
fe_val_acc = accuracy_score(y_val, fe_pipeline.predict(X_val_fe))

print(f"Train accuracy: {fe_train_acc:.4f}")
print(f"Val accuracy:   {fe_val_acc:.4f}")
print(f"(vs. underfit baseline: train 0.7347, val 0.7345)")

print("\n=== Feature engineering at max_depth=3 (enough capacity to use it) ===")
fe_pipeline_d3 = Pipeline(steps=[
    ("preprocessor", preprocessor_fe),
    ("classifier", DecisionTreeClassifier(max_depth=3, random_state=42)),
])
fe_pipeline_d3.fit(X_train_fe, y_train)

fe_d3_train_acc = accuracy_score(y_train, fe_pipeline_d3.predict(X_train_fe))
fe_d3_val_acc = accuracy_score(y_val, fe_pipeline_d3.predict(X_val_fe))

print(f"Train accuracy: {fe_d3_train_acc:.4f}")
print(f"Val accuracy:   {fe_d3_val_acc:.4f}")
print(f"(vs. non-engineered max_depth=3 from sweep: train 0.7915, val 0.7875)")


print("\n=== Target leakage experiment ===")

rng = np.random.default_rng(42)

def add_leaked_feature(X_split, y_split):
    X_split = X_split.copy()
    noisy_label = y_split.copy()
    flip_mask = rng.random(len(y_split)) < 0.05
    noisy_label = noisy_label.mask(pd.Series(flip_mask, index=y_split.index), 1 - noisy_label)
    X_split["CancellationRequestFiled"] = noisy_label
    return X_split

X_train_leak = add_leaked_feature(X_train, y_train)
X_val_leak = add_leaked_feature(X_val, y_val)

numeric_features_leak = numeric_features + ["CancellationRequestFiled"]
preprocessor_leak = ColumnTransformer(transformers=[
    ("numeric", Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ]), numeric_features_leak),
    ("categorical", Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ]), categorical_features),
])

leaky_pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor_leak),
    ("classifier", DecisionTreeClassifier(max_depth=5, random_state=42)),
])
leaky_pipeline.fit(X_train_leak, y_train)

leaky_train_acc = accuracy_score(y_train, leaky_pipeline.predict(X_train_leak))
leaky_val_acc = accuracy_score(y_val, leaky_pipeline.predict(X_val_leak))

print(f"WITH leaked feature — train: {leaky_train_acc:.4f}, val: {leaky_val_acc:.4f}")
print(f"(honest max_depth=5 from sweep, no leak — train: 0.8024, val: 0.7946)")