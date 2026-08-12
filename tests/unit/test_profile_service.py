import pandas as pd

from app.services.profile_service import build_profile


def test_build_profile_detects_numeric_and_missing_values():
    df = pd.DataFrame({
        "age": [25, 30, None, 40],
        "notes": ["a", "b", "c", "d"],
    })

    profile = build_profile(df)

    assert profile.row_count == 4
    assert profile.column_count == 2

    age_col = next(c for c in profile.columns if c.name == "age")
    assert age_col.dtype == "numeric"
    assert age_col.missing_count == 1
    assert age_col.missing_percentage == 25.0
    assert age_col.min == 25.0
    assert age_col.max == 40.0


def test_build_profile_classifies_low_cardinality_as_categorical():
    df = pd.DataFrame({"city": ["NYC"] * 80 + ["LA"] * 20})

    profile = build_profile(df)

    city_col = profile.columns[0]
    assert city_col.dtype == "categorical"
    assert city_col.unique_count == 2


def test_build_profile_classifies_high_cardinality_as_text():
    df = pd.DataFrame({"email": [f"user{i}@example.com" for i in range(100)]})

    profile = build_profile(df)

    email_col = profile.columns[0]
    assert email_col.dtype == "text"
