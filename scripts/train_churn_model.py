import pandas as pd
from sklearn.model_selection import train_test_split

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from xgboost import XGBClassifier

from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.model_selection import cross_val_score, StratifiedKFold

DATA_PATH = "data/telco_churn.csv"  # update to wherever you saved it

results = []

df = pd.read_csv(DATA_PATH)

# TotalCharges is read as text — some rows have blank/whitespace strings
# instead of a number (new customers with tenure=0, no charges yet).
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
# print(f"TotalCharges missing after coercion: {df['TotalCharges'].isna().sum()}")

# customerID is a unique identifier, not a predictive feature — drop it.
df = df.drop(columns=["customerID"])

# Separate features from the label.
y = df["Churn"].map({"Yes": 1, "No": 0})
X = df.drop(columns=["Churn"])

# print(f"X shape: {X.shape}")
# print(f"y shape: {y.shape}, churn rate: {y.mean():.2%}")

X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, stratify=y, random_state=42
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=42
)

# print(f"train: {X_train.shape[0]} rows, churn rate {y_train.mean():.2%}")
# print(f"val:   {X_val.shape[0]} rows, churn rate {y_val.mean():.2%}")
# print(f"test:  {X_test.shape[0]} rows, churn rate {y_test.mean():.2%}")


numeric_features = ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"]
categorical_features = [c for c in X.columns if c not in numeric_features]

numeric_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])

categorical_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore")),
])

preprocessor = ColumnTransformer(transformers=[
    ("numeric", numeric_pipeline, numeric_features),
    ("categorical", categorical_pipeline, categorical_features),
])

X_train_transformed = preprocessor.fit_transform(X_train)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
# print(f"X_train_transformed shape: {X_train_transformed.shape}")

# Baseline: always predict the majority class.
baseline = DummyClassifier(strategy="most_frequent")
baseline.fit(X_train, y_train)
baseline_preds = baseline.predict(X_val)
# print(f"\nBaseline accuracy: {accuracy_score(y_val, baseline_preds):.4f}")

# Logistic regression, using the same preprocessing pipeline built earlier.
logreg_pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(max_iter=1000)),
])
logreg_pipeline.fit(X_train, y_train)
logreg_preds = logreg_pipeline.predict(X_val)

results.append({
    "model": "Logistic Regression",
    "accuracy": accuracy_score(y_val, logreg_preds),
    "precision": precision_score(y_val, logreg_preds),
    "recall": recall_score(y_val, logreg_preds),
    "f1": f1_score(y_val, logreg_preds),
})

logreg_cv_scores = cross_val_score(logreg_pipeline, X_train, y_train, cv=cv, scoring="recall")
print(f"Logistic regression CV recall: {logreg_cv_scores.mean():.4f} (+/- {logreg_cv_scores.std():.4f})")
# print(f"\nLogistic regression accuracy: {accuracy_score(y_val, logreg_preds):.4f}")
# print(classification_report(y_val, logreg_preds, target_names=["no churn", "churn"]))

tree_pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", DecisionTreeClassifier(max_depth=5, random_state=42)),
])
tree_pipeline.fit(X_train, y_train)
tree_preds = tree_pipeline.predict(X_val)

results.append({
    "model": "Decision Tree",
    "accuracy": accuracy_score(y_val, tree_preds),
    "precision": precision_score(y_val, tree_preds),
    "recall": recall_score(y_val, tree_preds),
    "f1": f1_score(y_val, tree_preds),
})

tree_cv_scores = cross_val_score(tree_pipeline, X_train, y_train, cv=cv, scoring="recall")
print(f"Decision tree CV recall: {tree_cv_scores.mean():.4f} (+/- {tree_cv_scores.std():.4f})")
# print(f"\nDecision tree accuracy: {accuracy_score(y_val, tree_preds):.4f}")
# print(classification_report(y_val, tree_preds, target_names=["no churn", "churn"]))


forest_pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)),
])
forest_pipeline.fit(X_train, y_train)
forest_preds = forest_pipeline.predict(X_val)

results.append({
    "model": "Random Forest",
    "accuracy": accuracy_score(y_val, forest_preds),
    "precision": precision_score(y_val, forest_preds),
    "recall": recall_score(y_val, forest_preds),
    "f1": f1_score(y_val, forest_preds),
})

forest_cv_scores = cross_val_score(forest_pipeline, X_train, y_train, cv=cv, scoring="recall")
print(f"Random forest CV recall: {forest_cv_scores.mean():.4f} (+/- {forest_cv_scores.std():.4f})")
# print(f"\nRandom forest accuracy: {accuracy_score(y_val, forest_preds):.4f}")
# print(classification_report(y_val, forest_preds, target_names=["no churn", "churn"]))

xgb_pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", XGBClassifier(n_estimators=100, max_depth=5, random_state=42, eval_metric="logloss")),
])
xgb_pipeline.fit(X_train, y_train)
xgb_preds = xgb_pipeline.predict(X_val)

results.append({
    "model": "XGBoost",
    "accuracy": accuracy_score(y_val, xgb_preds),
    "precision": precision_score(y_val, xgb_preds),
    "recall": recall_score(y_val, xgb_preds),
    "f1": f1_score(y_val, xgb_preds),
})

xgb_cv_scores = cross_val_score(xgb_pipeline, X_train, y_train, cv=cv, scoring="recall")
print(f"XGBoost CV recall: {xgb_cv_scores.mean():.4f} (+/- {xgb_cv_scores.std():.4f})")
# print(f"\nXGBoost accuracy: {accuracy_score(y_val, xgb_preds):.4f}")
# print(classification_report(y_val, xgb_preds, target_names=["no churn", "churn"]))

results_df = pd.DataFrame(results)
print("\n=== Model comparison ===")
print(results_df.to_string(index=False))