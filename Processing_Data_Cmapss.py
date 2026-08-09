"""
Processing_Data_Cmapss.py
=========================

CMAPSS (NASA Turbofan) data-processing pipeline for RUL prediction.
This script processes the datasets FD001-FD004 and performs train / validation /
test splits used throughout the thesis

    1. Load the raw NASA text files (train, test, and test-RUL).
    2. Clean:   drop trailing blank columns, name the 26 columns.
    3. Label:   compute per-cycle RUL.
                - train: RUL = final_cycle - current_cycle (run-to-failure).
                - test : RUL = provided_final_RUL + (final_cycle - current_cycle).
    4. Cap:     clip RUL at RUL_CAP (=125).
    5. Scale:   StandardScaler fit on TRAIN sensor columns only, then applied
                to validation and test.
    6. Split:   80/20 train/validation split at the ENGINE level
                (whole engine sequences stay together), seed = RANDOM_SEED.
                NB! note taht - The test set is the official NASA hold-out and is left untouched.
    7. Window:  sliding windows of length WINDOW_SIZE (=30) for the sequential
                deep-learning models; the RUL of a window is the RUL at its last
                time step.
    8. Save:    - Non_Windowed/ -> tabular CSVs (features + ids) for classical /
                                    non-sequential models.
                - Windowed/  ->  3-D X and 1-D y as .npy plus window-level id CSVs
                                    for sequential models.

Reproducibility constants include RUL_CAP, WINDOW_SIZE, RANDOM_SEED


Author: Tiaan Mare
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ========================================
# CONFIGURATION  
# ========================

# Maximum RUL value 
RUL_CAP = 125

# Window length /time steps
WINDOW_SIZE = 30

# Seed 
RANDOM_SEED = 42

# 20% of training engines are taken out for validation.
VAL_SIZE = 0.2

# Sub-datasets to process.
DATASETS = ["FD001", "FD002", "FD003", "FD004"]

# Identifier (non-feature) columns.
ID_COLS = ["engine", "cycle", "RUL"]


# ========================
# PATH config
# ======================

def _find_data_root() -> Path:
    """Walk up from this file to find Data_CMAPSS in the repo, else use a local copy."""
    here = Path(__file__).resolve()
    for parent in [here.parent, *here.parents]:
        candidate = parent / "CodeBase_Experiments" / "0_Data_Processing" / "Data_CMAPSS"
        if (candidate / "1_Raw_CMAPSS").exists():
            return candidate
    # its expected to finsd local Data_CMAPSS folder alongside this script.
    return here.parent / "Data_CMAPSS"


DATA_ROOT = _find_data_root()
RAW_DATA_DIR = DATA_ROOT / "1_Raw_CMAPSS"
CLEANED_DATA_DIR = DATA_ROOT / "2_Cleaned_Data"
NON_WINDOWED_DIR = CLEANED_DATA_DIR / "Non_Windowed"
WINDOWED_DIR = CLEANED_DATA_DIR / "Windowed"


def _column_names() -> list[str]:
    """26 columns: engine, cycle, 3 operating settings, 21 sensors."""
    cols = ["engine", "cycle"]
    cols += [f"os{i}" for i in range(1, 4)]
    cols += [f"s{i}" for i in range(1, 22)]
    return cols


COLUMN_NAMES = _column_names()


# ==========================================================
# LOAD, CLEAN, LABEL, CAP
# ==========================================================

def load_and_prepare(dataset: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load one sub-dataset and return cleaned, RUL-labelled, RUL-capped train/test frames."""
    raw_train = pd.read_csv(RAW_DATA_DIR / f"train_{dataset}.txt", header=None, sep=" ")
    raw_test = pd.read_csv(RAW_DATA_DIR / f"test_{dataset}.txt", header=None, sep=" ")
    raw_rul = pd.read_csv(RAW_DATA_DIR / f"RUL_{dataset}.txt", header=None, sep=" ")

    # Drop the two trailing all-NaN columns produced by the trailing separator.
    raw_train = raw_train.drop(columns=[c for c in (26, 27) if c in raw_train.columns])
    raw_test = raw_test.drop(columns=[c for c in (26, 27) if c in raw_test.columns])
    raw_rul = raw_rul.drop(columns=[c for c in (1,) if c in raw_rul.columns])

    raw_train.columns = COLUMN_NAMES
    raw_test.columns = COLUMN_NAMES
    raw_rul.columns = ["final_RUL"]

    # --- Label the TRAINING set (engines run to failure) -----------------
    final_cycle = raw_train.groupby("engine")["cycle"].transform("max")
    raw_train["RUL"] = final_cycle - raw_train["cycle"]

    # --- Label the TEST set (engines stop early; NASA gives final RUL) ----
    raw_rul["engine"] = range(1, len(raw_rul) + 1)
    test_final_cycle = raw_test.groupby("engine")["cycle"].transform("max")
    raw_test = raw_test.merge(raw_rul, on="engine")
    raw_test["RUL"] = raw_test["final_RUL"] + (test_final_cycle - raw_test["cycle"])
    raw_test = raw_test.drop(columns=["final_RUL"])

    # --- Cap RUL at the piecewise-linear ceiling -------------------------
    raw_train["RUL"] = raw_train["RUL"].clip(upper=RUL_CAP)
    raw_test["RUL"] = raw_test["RUL"].clip(upper=RUL_CAP)

    return raw_train, raw_test


# =============================================================================
# 5. STANDARDISATION  (fit on train only, apply to test)
# =============================================================================

def standardize(train_df: pd.DataFrame, test_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Standardise sensor/setting columns; scaler is fit on TRAIN only (no leakage)."""
    feature_cols = [c for c in train_df.columns if c not in ID_COLS]

    scaler = StandardScaler()
    train_out = train_df.copy()
    test_out = test_df.copy()

    train_out[feature_cols] = scaler.fit_transform(train_df[feature_cols])
    test_out[feature_cols] = scaler.transform(test_df[feature_cols])

    return train_out, test_out


# =============================================================================
# 6. ENGINE-LEVEL TRAIN / VALIDATION SPLIT
# =============================================================================

def engine_level_split(train_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split into train/val by whole engines (sequences stay intact), seed = RANDOM_SEED."""
    engines = train_df["engine"].unique()
    eng_train, eng_val = train_test_split(
        engines, test_size=VAL_SIZE, random_state=RANDOM_SEED
    )
    train_split = train_df[train_df["engine"].isin(eng_train)].reset_index(drop=True)
    val_split = train_df[train_df["engine"].isin(eng_val)].reset_index(drop=True)
    return train_split, val_split


def split_features_ids(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Separate a frame into (feature columns, identifier columns)."""
    features = df.drop(columns=ID_COLS, errors="ignore").reset_index(drop=True)
    ids = df[ID_COLS].reset_index(drop=True)
    return features, ids


# =============================================================================
# 7. SLIDING WINDOWS  (for sequential models)
# =============================================================================

def create_sliding_windows(
    df: pd.DataFrame,
    window_size: int,
    id_col: str = "engine",
    time_col: str = "cycle",
    target_col: str = "RUL",
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """
    Convert a sequence frame into sliding windows.

    For each engine's time-ordered sequence, produce windows of `window_size`.
    Each window yields:
        X   : sensor features over the window       -> (window_size, n_features)
        y   : RUL at the LAST time step of the window
        ids : engine + start/end cycle + RUL for the window
    """
    feature_cols = [c for c in df.columns if c not in (id_col, time_col, target_col)]

    X, y, id_rows = [], [], []

    for _, seq in df.groupby(id_col):
        seq = seq.sort_values(time_col).reset_index(drop=True)
        if len(seq) < window_size:
            continue  # engine sequence shorter than the window is skipped

        for i in range(len(seq) - window_size + 1):
            window = seq.iloc[i : i + window_size]
            X.append(window[feature_cols].values)
            y.append(window[target_col].iloc[-1])
            id_rows.append(
                {
                    id_col: window[id_col].iloc[-1],
                    f"start_{time_col}": window[time_col].iloc[0],
                    f"end_{time_col}": window[time_col].iloc[-1],
                    target_col: window[target_col].iloc[-1],
                }
            )

    return np.asarray(X), np.asarray(y), pd.DataFrame(id_rows)


# =============================================================================
# 8. SAVE
# =============================================================================

def save_non_windowed(name: str, splits: dict[str, tuple[pd.DataFrame, pd.DataFrame]]) -> None:
    """Save tabular features + ids for train/val/test to Non_Windowed/."""
    NON_WINDOWED_DIR.mkdir(parents=True, exist_ok=True)
    for split_name, (features, ids) in splits.items():
        features.to_csv(NON_WINDOWED_DIR / f"{name}_{split_name}_features.csv", index=False)
        ids.to_csv(NON_WINDOWED_DIR / f"{name}_{split_name}_ids.csv", index=False)


def save_windowed(name: str, windows: dict[str, tuple[np.ndarray, np.ndarray, pd.DataFrame]]) -> None:
    """Save 3-D X, 1-D y (.npy) and window-level ids (.csv) for train/val/test to Windowed/."""
    WINDOWED_DIR.mkdir(parents=True, exist_ok=True)
    for split_name, (X, y, ids) in windows.items():
        np.save(WINDOWED_DIR / f"{name}_X_{split_name}_windowed.npy", X)
        np.save(WINDOWED_DIR / f"{name}_y_{split_name}_windowed.npy", y)
        ids.to_csv(WINDOWED_DIR / f"{name}_{split_name}_ids_windowed.csv", index=False)


# ====================
# Execution of defs
# ======================

def process_dataset(dataset: str) -> None:
    """Run the full pipeline for a single sub-dataset and write all outputs."""
    print(f"[{dataset}] loading and labelling ...")
    train_df, test_df = load_and_prepare(dataset)

    print(f"[{dataset}] standardising (scaler fit on train only) ...")
    train_std, test_std = standardize(train_df, test_df)

    print(f"[{dataset}] engine-level {int((1 - VAL_SIZE) * 100)}/"
          f"{int(VAL_SIZE * 100)} train/val split (seed={RANDOM_SEED}) ...")
    train_split, val_split = engine_level_split(train_std)

    # ---- Non-windowed (tabular) outputs --------------------------------
    non_windowed = {
        "train": split_features_ids(train_split),
        "val": split_features_ids(val_split),
        "test": split_features_ids(test_std),
    }
    save_non_windowed(dataset, non_windowed)

    # ---- Windowed (sequential) outputs ---------------------------------
    print(f"[{dataset}] building sliding windows (size={WINDOW_SIZE}) ...")
    windowed = {
        "train": create_sliding_windows(train_split, WINDOW_SIZE),
        "val": create_sliding_windows(val_split, WINDOW_SIZE),
        "test": create_sliding_windows(test_std, WINDOW_SIZE),
    }
    save_windowed(dataset, windowed)

    n_train_eng = train_split["engine"].nunique()
    n_val_eng = val_split["engine"].nunique()
    print(
        f"[{dataset}] done. "
        f"train engines={n_train_eng}, val engines={n_val_eng}, "
        f"train windows={len(windowed['train'][1])}, "
        f"val windows={len(windowed['val'][1])}, "
        f"test windows={len(windowed['test'][1])}"
    )


def main() -> None:
    print("=" * 70)
    print("CMAPSS data processing")
    print(f"  raw data     : {RAW_DATA_DIR}")
    print(f"  cleaned data : {CLEANED_DATA_DIR}")
    print("=" * 70)

    if not RAW_DATA_DIR.exists():
        raise FileNotFoundError(
            f"Raw CMAPSS data not found at: {RAW_DATA_DIR}\n"
            "Place the NASA text files (train_FD00X.txt, test_FD00X.txt, "
            "RUL_FD00X.txt) there, or edit RAW_DATA_DIR at the top of this script."
        )

    for dataset in DATASETS:
        process_dataset(dataset)

    print("=" * 70)
    print("All datasets processed successfully.")
    print("=" * 70)


if __name__ == "__main__":
    main()
