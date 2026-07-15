"""
Multi-pattern CSV ingestion for the ZIMSEC analytics project.

Combines a master dataset with any newly uploaded files, validates schema,
deduplicates by district-year-sitting, and tracks provenance.

Used by:
- webapp/main.py (the FastAPI dashboard) — see load_raw_data()
- notebooks/zimsec_olevel_predictive_model.ipynb — for fresh ingestion
"""
import glob
from pathlib import Path
import pandas as pd


# Canonical schema and dedup keys for ZIMSEC O-Level data.
# Keep these in one place so the notebook and webapp agree.
REQUIRED_COLS = ["District", "Province", "Year", "Setting", "Pass_Rate_Pct"]
DEDUP_KEYS    = ["District", "Year", "Setting"]


def ingest_multi_glob(patterns: list,
                      required_cols: list = None,
                      dedup_keys: list = None,
                      exclude_files: list = None,
                      verbose: bool = False) -> pd.DataFrame:
    """
    Load and concatenate every CSV that matches ANY of the given glob patterns.

    Parameters
    ----------
    patterns      : list of str  — glob patterns, e.g. ["data/raw/*.csv", "data/uploads/*.csv"]
    required_cols : list         — columns every file MUST have; files missing any are skipped
    dedup_keys    : list         — columns to dedupe on (e.g. ["District","Year","Setting"])
    exclude_files : list         — filenames to skip (e.g. ["cleaned_data.csv"])
    verbose       : bool         — print per-file load summary (off by default for webapp use)

    Returns
    -------
    pd.DataFrame with an extra `_source_file` column tracking provenance.

    Raises
    ------
    FileNotFoundError if no files match any pattern.
    ValueError       if all matched files fail schema validation.
    """
    exclude_files = set(exclude_files or [])

    # 1. Collect matching files from ALL patterns, deduplicated by absolute path
    all_files = set()
    matches_by_pattern = {}
    for pat in patterns:
        hits = glob.glob(pat, recursive=True)
        # Resolve to absolute paths so the same file matched by two patterns is counted once
        resolved = {str(Path(h).resolve()) for h in hits if Path(h).is_file()}
        matches_by_pattern[pat] = len(resolved)
        all_files.update(resolved)

    # Filter excluded names and keep deterministic order
    files = sorted(f for f in all_files if Path(f).name not in exclude_files)

    if verbose:
        print(f"Patterns: {len(patterns)}")
        for pat, n in matches_by_pattern.items():
            print(f"  {pat:50s} → {n} match(es)")
        print(f"Unique files (post-exclude): {len(files)}\n")

    if not files:
        raise FileNotFoundError(f"No files matched any of: {patterns}")

    # 2. Load each file with schema validation
    frames, skipped = [], []
    for fp in files:
        path = Path(fp)
        try:
            df = pd.read_csv(path)
        except Exception as e:
            skipped.append((path.name, f"read error: {e}"))
            continue

        if required_cols:
            missing = [c for c in required_cols if c not in df.columns]
            if missing:
                skipped.append((path.name, f"missing columns: {missing}"))
                continue

        df["_source_file"] = path.name
        frames.append(df)

        if verbose:
            print(f"  ✓ {path.name:50s} {len(df):>5,} rows × {df.shape[1]:>2} cols")

    if not frames:
        raise ValueError(
            f"No files loaded successfully. {len(skipped)} skipped: {skipped}"
        )

    # 3. Concatenate and dedupe
    combined = pd.concat(frames, ignore_index=True)

    if dedup_keys:
        before = len(combined)
        combined = combined.drop_duplicates(subset=dedup_keys, keep="last")
        if verbose:
            print(f"\nDeduped on {dedup_keys}: removed {before - len(combined):,} duplicate row(s)")

    if verbose:
        print(f"\nFinal: {len(combined):,} rows from {len(frames)} file(s)")
        if skipped:
            print(f"Skipped {len(skipped)} file(s):")
            for name, reason in skipped:
                print(f"  ✗ {name}: {reason}")

    return combined


def build_default_patterns(data_dir: Path) -> list:
    """
    Standard pattern list for the ZIMSEC project.

    Picks up:
      - the master file in data/
      - anything in data/raw/
      - anything uploaded via the webapp (data/uploads/)
      - any nested zimsec_*.csv across the data tree

    Returns a list of glob pattern strings.
    """
    return [
        str(data_dir / "zimsec_olevel_district_data.csv"),
        str(data_dir / "raw" / "*.csv"),
        str(data_dir / "uploads" / "*.csv"),
        str(data_dir / "**" / "zimsec_*.csv"),
    ]
