"""Find the seven kinds of data problem in train.csv.

Each check is one small function that looks at the data and returns a dict
with the same keys as the matching block in reports/findings.yaml.
clean.py calls these same functions, so every fix we make follows from
something a check actually found.

Run:  PYTHONPATH=src python -m pgrent.data_checks
"""

import numpy as np
import pandas as pd

from pgrent.config import load_params


# ---------------------------------------------------------------------------
# Helper used by the spelling check (and by clean.py)
# ---------------------------------------------------------------------------
def normalise_label(value, clean_levels):
    """Turn one messy category spelling into its clean level.

    'Semi ', 'SEMI', 'sem.' -> 'semi'.  Steps: strip spaces, lowercase,
    drop a trailing '.', then match the clean level that starts with it
    ('unf' is the start of 'unfurnished'). If nothing matches, return the
    tidied text unchanged so we can see it in value_counts().
    """
    text = str(value).strip().lower().rstrip(".")
    for level in clean_levels:
        if level.lower() == text:
            return level
    for level in clean_levels:
        if level.lower().startswith(text):
            return level
    return text


# ---------------------------------------------------------------------------
# 1. Leaking column
# ---------------------------------------------------------------------------
def find_leaking_column(train, holdout, target, min_spearman):
    """A leak is filled in only after the event, so the holdout file (which
    is what we get at prediction time) does not have it. It also moves
    almost exactly with the target.

    We use Spearman (rank) correlation, because the 10x target typos would
    drag a normal Pearson correlation down.
    """
    not_in_holdout = [c for c in train.columns if c not in holdout.columns and c != target]
    for col in not_in_holdout:
        if pd.api.types.is_numeric_dtype(train[col]):
            rho = train[col].corr(train[target], method="spearman")
            print(f"  leak? {col}: missing from holdout, spearman with target = {rho:.4f}")
            if rho >= min_spearman:
                return {"present": True, "column": col}
    return {"present": False, "column": None}


# ---------------------------------------------------------------------------
# 2. Unit mix-up in one group
# ---------------------------------------------------------------------------
def find_unit_mixup(train, group_col, numeric_cols, factors, trigger):
    """Compare each group's median with the median of all the other groups.
    If one group is about 10/12/60/100/1000 times bigger (or smaller), that
    office used a different unit. factor = wrong / right, snapped to the
    nearest exact conversion.
    """
    for col in numeric_cols:
        for value in train[group_col].unique():
            in_group = train.loc[train[group_col] == value, col]
            others = train.loc[train[group_col] != value, col]
            if others.median() == 0 or in_group.median() == 0:
                continue
            ratio = in_group.median() / others.median()
            if ratio >= trigger or ratio <= 1 / trigger:
                # snap to the closest exact unit conversion (on a log scale)
                big = ratio if ratio > 1 else 1 / ratio
                snapped = min(factors, key=lambda f: abs(np.log(big / f)))
                factor = snapped if ratio > 1 else 1 / snapped
                print(f"  unit mix-up: {col} in {group_col}={value!r}, ratio {ratio:.2f} -> factor {factor}")
                return {"present": True, "column": col, "group_value": value, "factor": factor}
    print("  unit mix-up: every group's median is within", trigger, "x of the others")
    return {"present": False, "column": None, "group_value": None, "factor": None}


# ---------------------------------------------------------------------------
# 3. Sentinel value
# ---------------------------------------------------------------------------
def find_sentinel(train, later, numeric_cols, candidates, min_count):
    """A sentinel is a special number (-1, 999, ...) typed instead of a blank.
    To count, it must repeat at least min_count times AND lie outside the
    range the clean later.csv shows for that column (so a real wifi speed of
    99 Mbps is not called a sentinel).
    """
    best = None
    for col in numeric_cols:
        low, high = later[col].min(), later[col].max()
        for value in candidates:
            count = int((train[col] == value).sum())
            impossible = value < low or value > high
            if count >= min_count and impossible:
                print(f"  sentinel: {col} has {value} x{count} (clean range {low}..{high})")
                if best is None or count > best["count"]:
                    best = {"present": True, "column": col, "value": value, "count": count}
    if best is None:
        print("  sentinel: no candidate value repeats outside the clean range")
        return {"present": False, "column": None, "value": None, "count": None}
    return best


# ---------------------------------------------------------------------------
# 4. Duplicate rows
# ---------------------------------------------------------------------------
def find_duplicates(train, id_col):
    """duplicated() marks every copy after the first, so its sum is the
    number of EXTRA rows we would delete. We also check with the id column
    left out, in case a copy was given a new id.
    """
    exact = int(train.duplicated().sum())
    without_id = int(train.drop(columns=[id_col]).duplicated().sum())
    print(f"  duplicates: exact copies = {exact}, copies ignoring {id_col} = {without_id}")
    count = max(exact, without_id)
    if count == 0:
        return {"present": False, "count": None}
    return {"present": True, "count": count}


# ---------------------------------------------------------------------------
# 5. Stuck sensor
# ---------------------------------------------------------------------------
def longest_run(values):
    """Return (start_index, end_index) of the longest block of equal
    consecutive values in a list."""
    best_start, best_end = 0, 0
    start = 0
    for i in range(1, len(values) + 1):
        # a run ends when the value changes or the list ends
        if i == len(values) or values[i] != values[start]:
            if i - 1 - start > best_end - best_start:
                best_start, best_end = start, i - 1
            start = i
    return best_start, best_end


def find_stuck_sensor(train, date_col, id_col, numeric_cols, min_days):
    """Sort by date. A stuck device repeats one reading on every row for many
    days in a row, so look for the longest run of identical values in each
    column and measure how many different days it covers.
    """
    ordered = train.sort_values([date_col, id_col]).reset_index(drop=True)
    best = None
    for col in numeric_cols:
        start, end = longest_run(ordered[col].tolist())
        run = ordered.loc[start:end]
        days = run[date_col].nunique()
        if days >= min_days and (best is None or days > best["days"]):
            best = {
                "column": col,
                "value": ordered.loc[start, col],
                "start_date": run[date_col].min(),
                "end_date": run[date_col].max(),
                "rows": len(run),
                "days": days,
            }
    if best is None:
        print("  stuck sensor: no column repeats one value for", min_days, "days")
        return {"present": False, "column": None, "start_date": None, "end_date": None}
    print(f"  stuck sensor: {best['column']} = {best['value']} on {best['rows']} rows, "
          f"{best['start_date']} to {best['end_date']} ({best['days']} days)")
    return {"present": True, "column": best["column"],
            "start_date": best["start_date"], "end_date": best["end_date"]}


# ---------------------------------------------------------------------------
# 6. Target typos (an extra zero)
# ---------------------------------------------------------------------------
def typo_threshold(target, gap_ratio):
    """Sort the target and find the biggest jump between neighbours (as a
    ratio). If that jump is at least gap_ratio, everything above it is a
    typo. Returns the first value above the gap, or None if there is no gap.
    """
    values = np.sort(target.dropna().to_numpy(dtype=float))
    jumps = values[1:] / values[:-1]
    i = int(np.argmax(jumps))
    if jumps[i] < gap_ratio:
        return None
    print(f"  target gap: {values[i]:.0f} -> {values[i + 1]:.0f} (x{jumps[i]:.1f})")
    return values[i + 1]


def find_target_typos(train, target, gap_ratio):
    threshold = typo_threshold(train[target], gap_ratio)
    if threshold is None:
        print("  target typos: no big gap in the sorted target")
        return {"present": False, "count": None}
    count = int((train[target] >= threshold).sum())
    print(f"  target typos: {count} rows at or above {threshold:.0f}")
    return {"present": True, "count": count}


# ---------------------------------------------------------------------------
# 7. Spelling variants
# ---------------------------------------------------------------------------
def find_spelling_variants(train, later, text_cols):
    """later.csv has the clean spellings. If train has more distinct
    spellings than later has levels, the extra ones are variants.
    true_levels = how many real categories remain after normalising.
    """
    for col in text_cols:
        clean_levels = sorted(later[col].unique())
        raw_count = train[col].nunique()
        fixed = train[col].map(lambda v: normalise_label(v, clean_levels))
        print(f"  spelling: {col}: {raw_count} raw spellings -> {fixed.nunique()} after cleaning "
              f"{sorted(fixed.unique())}")
        if raw_count > len(clean_levels):
            return {"present": True, "column": col, "true_levels": int(fixed.nunique())}
    return {"present": False, "column": None, "true_levels": None}


# ---------------------------------------------------------------------------
# Run all seven
# ---------------------------------------------------------------------------
def run_all_checks(train, later, holdout, params):
    """Run the seven checks in a sensible order and return one dict whose
    keys match data_issues in findings.yaml."""
    cols, chk = params["columns"], params["checks"]
    numeric = cols["numeric"]

    # The unit check compares groups, so the group column must be spelled
    # consistently first (in our file it already is).
    tidy = train.copy()
    tidy[cols["group"]] = tidy[cols["group"]].map(
        lambda v: normalise_label(v, sorted(later[cols["group"]].unique())))

    return {
        "leaking_column": find_leaking_column(train, holdout, cols["target"], chk["leak_min_spearman"]),
        "unit_mixup": find_unit_mixup(tidy, cols["group"], numeric,
                                      chk["unit_factors"], chk["unit_ratio_trigger"]),
        "sentinel_value": find_sentinel(train, later, numeric,
                                        chk["sentinel_candidates"], chk["sentinel_min_count"]),
        "duplicate_rows": find_duplicates(train, cols["id"]),
        "stuck_sensor": find_stuck_sensor(train, cols["date"], cols["id"], numeric, chk["stuck_min_days"]),
        "target_typos": find_target_typos(train, cols["target"], chk["typo_gap_ratio"]),
        "spelling_variants": find_spelling_variants(train, later, cols["categorical"]),
    }


def main():
    params = load_params()
    train = pd.read_csv(params["data"]["train"])
    later = pd.read_csv(params["data"]["later"])
    holdout = pd.read_csv(params["data"]["holdout"])
    print(f"train.csv: {len(train)} rows, {train.shape[1]} columns\n")

    findings = run_all_checks(train, later, holdout, params)

    # Print in the same shape as findings.yaml, ready to copy over.
    print("\ndata_issues:")
    for kind, details in findings.items():
        print(f"  {kind}:")
        for key, value in details.items():
            print(f"    {key}: {yaml_text(value)}")


def yaml_text(value):
    """How one value is written in YAML: null, true/false, "text" or a number."""
    if value is None:
        return "null"
    if isinstance(value, (bool, np.bool_)):
        return "true" if value else "false"
    if isinstance(value, str):
        return f'"{value}"'
    if isinstance(value, (np.integer, np.floating)):
        value = value.item()
    return str(value)


if __name__ == "__main__":
    main()
