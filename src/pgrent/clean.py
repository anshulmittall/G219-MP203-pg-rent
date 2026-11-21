"""Fix the data problems that data_checks.py found in train.csv.

The raw file is never changed. We clean a copy in memory and also save it to
data/processed/train_clean.csv so it can be inspected.

Run:  PYTHONPATH=src python -m pgrent.clean
"""

import os

import numpy as np
import pandas as pd

from pgrent.config import load_params
from pgrent.data_checks import normalise_label, run_all_checks, typo_threshold


def tidy_categories(df, clean_levels):
    """Map every text column to its clean spelling. Used for train, for
    later.csv and for the holdout file, so all three speak the same language.
    clean_levels is {column: [allowed values]} taken from later.csv."""
    df = df.copy()
    for col, levels in clean_levels.items():
        df[col] = df[col].map(lambda v: normalise_label(v, levels))
    return df


def clean_levels_from(later, params):
    """later.csv has the clean spellings, so it defines the allowed levels."""
    return {col: sorted(later[col].unique()) for col in params["columns"]["categorical"]}


def clean_train(train, later, holdout, params):
    """Apply one fix for every problem the checks report as present."""
    cols = params["columns"]
    findings = run_all_checks(train, later, holdout, params)
    df = train.copy()

    # 1. Leak: drop the column. We would not know it when we predict.
    leak = findings["leaking_column"]
    if leak["present"]:
        df = df.drop(columns=[leak["column"]])

    # 2. Duplicates: keep the first copy of each row.
    if findings["duplicate_rows"]["present"]:
        df = df.drop_duplicates()
        df = df.drop_duplicates(subset=[c for c in df.columns if c != cols["id"]])

    # 3. Spelling: 'SEMI', 'sem.', ' semi' -> 'semi'.
    df = tidy_categories(df, clean_levels_from(later, params))

    # 4. Unit mix-up: wrong = right x factor, so right = wrong / factor.
    unit = findings["unit_mixup"]
    if unit["present"]:
        rows = df[cols["group"]] == unit["group_value"]
        df.loc[rows, unit["column"]] = df.loc[rows, unit["column"]] / unit["factor"]

    # 5. Sentinel: the special number really means "no reading", so blank it.
    #    The model's imputer fills blanks with the median later.
    sent = findings["sentinel_value"]
    if sent["present"]:
        df.loc[df[sent["column"]] == sent["value"], sent["column"]] = np.nan

    # 6. Stuck sensor: readings in the stuck period are not real, so blank them.
    stuck = findings["stuck_sensor"]
    if stuck["present"]:
        in_period = df[cols["date"]].between(stuck["start_date"], stuck["end_date"])
        stuck_value = df.loc[in_period, stuck["column"]].mode()[0]
        df.loc[in_period & (df[stuck["column"]] == stuck_value), stuck["column"]] = np.nan

    # 7. Target typos: an extra zero made the rent 10x too big. Divide by 10
    #    (we keep the row: the features are fine, only the rent was mistyped).
    if findings["target_typos"]["present"]:
        threshold = typo_threshold(df[cols["target"]], params["checks"]["typo_gap_ratio"])
        too_big = df[cols["target"]] >= threshold
        df[cols["target"]] = df[cols["target"]].astype(float)
        df.loc[too_big, cols["target"]] = df.loc[too_big, cols["target"]] / 10

    return df.sort_values([cols["date"], cols["id"]]).reset_index(drop=True)


def load_all(params):
    """Read the three CSVs and return (cleaned train, later, holdout).
    later and holdout get the same spelling clean-up as train."""
    train = pd.read_csv(params["data"]["train"])
    later = pd.read_csv(params["data"]["later"])
    holdout = pd.read_csv(params["data"]["holdout"])
    levels = clean_levels_from(later, params)
    clean = clean_train(train, later, holdout, params)
    return clean, tidy_categories(later, levels), tidy_categories(holdout, levels)


def main():
    params = load_params()
    clean, _, _ = load_all(params)
    out = params["data"]["clean_train"]
    os.makedirs(os.path.dirname(out), exist_ok=True)
    clean.to_csv(out, index=False)
    print(f"\nSaved {len(clean)} cleaned rows to {out}")
    print(clean.describe().T[["count", "min", "mean", "max"]])


if __name__ == "__main__":
    main()
