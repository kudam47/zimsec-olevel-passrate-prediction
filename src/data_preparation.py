"""
Data preparation module — replaces the original data_preparation.py.

Used by main.py's /analysis route. Loads the dataset (via multi-glob
ingestion, so uploads are picked up automatically), engineers features,
and saves a cleaned CSV to data/cleaned_data.csv for downstream use.
"""
from pathlib import Path
import pandas as pd

import sys as _sys
from pathlib import Path as _Path
_WEBAPP_DIR = _Path(__file__).resolve().parent.parent / "webapp"
if str(_WEBAPP_DIR) not in _sys.path:
    _sys.path.insert(0, str(_WEBAPP_DIR))

from ingestion import (
    ingest_multi_glob, build_default_patterns,
    REQUIRED_COLS, DEDUP_KEYS,
)

# Project root — this file lives in webapp/, .parent.parent gets the project root
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"


def run_data_preparation():
    """
    Load → clean → feature-engineer → save.

    Returns the cleaned DataFrame. Raises FileNotFoundError if no data is found.
    """
    # 1. Multi-glob load: master + raw/ + uploads/ + nested zimsec_*.csv
    patterns = build_default_patterns(DATA_DIR)
    df = ingest_multi_glob(
        patterns=patterns,
        required_cols=REQUIRED_COLS,
        dedup_keys=DEDUP_KEYS,
        exclude_files=["cleaned_data.csv"],
        verbose=False,
    )

    # Drop provenance column — downstream code expects the original schema
    df = df.drop(columns=["_source_file"], errors="ignore")

    # 2. Feature engineering (idempotent — safe to re-run)
    df["Setting_Nov"] = (df["Setting"] == "November").astype(int)
    df["Is_Urban"]    = (df["Rural_Urban"] == "Urban").astype(int)

    # 3. Light cleaning — clip implausible values, fill any nulls with column median
    if "Pass_Rate_Pct" in df.columns:
        df["Pass_Rate_Pct"] = df["Pass_Rate_Pct"].clip(lower=0, upper=100)

    numeric_cols = df.select_dtypes(include=["number"]).columns
    for col in numeric_cols:
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].median())

    # 4. Persist
    out_path = DATA_DIR / "cleaned_data.csv"
    df.to_csv(out_path, index=False)

    return df


if __name__ == "__main__":
    df = run_data_preparation()
    print(f"Cleaned dataset: {df.shape}")
    print(f"Saved to: {DATA_DIR / 'cleaned_data.csv'}")
