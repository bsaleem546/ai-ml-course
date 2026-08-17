import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

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
