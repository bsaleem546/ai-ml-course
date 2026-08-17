import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline


def test_total_charges_blank_string_becomes_nan():
    df = pd.DataFrame({"TotalCharges": ["29.85", " ", "1889.5", ""]})

    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    assert df["TotalCharges"].isna().sum() == 2
    assert df["TotalCharges"].iloc[0] == 29.85
    assert df["TotalCharges"].iloc[2] == 1889.5


def test_cloned_preprocessor_does_not_leak_state_between_pipelines():
    base_preprocessor = ColumnTransformer([
        ("num", SimpleImputer(strategy="median"), ["x"]),
    ])

    df_a = pd.DataFrame({"x": [1.0, 2.0, 3.0]})   # median = 2.0
    df_b = pd.DataFrame({"x": [100.0, 200.0, 300.0]})  # median = 200.0

    pipeline_a = Pipeline([("preprocessor", clone(base_preprocessor))])
    pipeline_a.fit(df_a)

    pipeline_b = Pipeline([("preprocessor", clone(base_preprocessor))])
    pipeline_b.fit(df_b)

    result_a = pipeline_a.transform(pd.DataFrame({"x": [None]}))
    assert result_a[0][0] == 2.0