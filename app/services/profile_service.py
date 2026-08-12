import pandas as pd

from app.schemas.dataset import ColumnProfile, DatasetProfile

CATEGORICAL_UNIQUE_RATIO_THRESHOLD = 0.5


def _safe_float(value: float) -> float | None:
    return None if pd.isna(value) else float(value)


def _classify_column(series: pd.Series, row_count: int) -> str:
    if pd.api.types.is_numeric_dtype(series):
        return "numeric"
    unique_ratio = series.nunique() / row_count if row_count else 0
    return "categorical" if unique_ratio < CATEGORICAL_UNIQUE_RATIO_THRESHOLD else "text"


def build_profile(df: pd.DataFrame) -> DatasetProfile:
    row_count = len(df)
    columns = []

    for column_name in df.columns:
        series = df[column_name]
        dtype = _classify_column(series, row_count)
        missing_count = int(series.isna().sum())

        column_profile = ColumnProfile(
            name=str(column_name),
            dtype=dtype,
            missing_count=missing_count,
            missing_percentage=round(missing_count / row_count * 100, 2) if row_count else 0.0,
            unique_count=int(series.nunique()),
        )

        if dtype == "numeric":
            column_profile.min = _safe_float(series.min())
            column_profile.max = _safe_float(series.max())
            column_profile.mean = _safe_float(series.mean())
            column_profile.median = _safe_float(series.median())
            column_profile.std = _safe_float(series.std())

        columns.append(column_profile)

    return DatasetProfile(row_count=row_count, column_count=len(df.columns), columns=columns)