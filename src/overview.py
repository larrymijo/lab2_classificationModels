"""Quick overview helpers for comparing the candidate Lab 2 datasets (Step 1)."""
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
READERS = {".csv": pd.read_csv, ".xlsx": pd.read_excel, ".xls": pd.read_excel}


def find_datasets(data_dir=DATA_DIR):
    """Return all readable dataset files in data/ (searches subfolders too)."""
    return sorted(p for p in Path(data_dir).rglob("*") if p.suffix.lower() in READERS)


def load_dataset(path):
    """Load a CSV or Excel file into a DataFrame."""
    path = Path(path)
    reader = READERS.get(path.suffix.lower())
    if reader is None:
        raise ValueError(f"Unsupported file type: {path.suffix}")
    return reader(path)


def id_like_columns(df):
    """Integer/text columns where every row has a different value (IDs, names, codes)."""
    return [
        col for col in df.columns
        if df[col].nunique() == len(df) and not pd.api.types.is_float_dtype(df[col])
    ]


def summarize(df):
    """One-row summary of a dataset: size, column types, missing data, duplicates."""
    rows, cols = df.shape
    n_numeric = df.select_dtypes("number").shape[1]
    return {
        "rows": rows,
        "columns": cols,
        "numeric_cols": n_numeric,
        "categorical_cols": cols - n_numeric,
        "missing_cells_%": round(df.isna().mean().mean() * 100, 2),
        "cols_with_missing": int(df.isna().any().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
        "id_like_cols": id_like_columns(df),
        "constant_cols": [c for c in df.columns if df[c].nunique(dropna=False) <= 1],
    }


def compare_datasets(data_dir=DATA_DIR):
    """Summary table with one row per dataset file."""
    records = [{"dataset": f"{p.parent.name}/{p.stem}", **summarize(load_dataset(p))} for p in find_datasets(data_dir)]
    return pd.DataFrame(records).set_index("dataset")


def column_profile(df):
    """Per-column overview: dtype, number of unique values, % missing, an example value."""
    return pd.DataFrame({
        "dtype": df.dtypes.astype(str),
        "unique": df.nunique(),
        "missing_%": (df.isna().mean() * 100).round(2),
        "example": df.iloc[0] if len(df) else None,
    })


def candidate_targets(df, min_unique=20):
    """Numeric columns with enough distinct values to be a sensible regression target."""
    numeric = df.select_dtypes("number")
    stats = pd.DataFrame({
        "unique": numeric.nunique(),
        "missing_%": (numeric.isna().mean() * 100).round(2),
        "min": numeric.min(),
        "median": numeric.median(),
        "max": numeric.max(),
        "skew": numeric.skew().round(2),
    })
    return stats[stats["unique"] >= min_unique].sort_values("unique", ascending=False)