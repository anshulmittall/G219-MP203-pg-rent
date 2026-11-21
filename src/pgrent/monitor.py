"""Two months later: drift, per-group error, retrain and the promotion rule.

PROBLEM.md section 7, step by step:
  1. PSI (10 bins) for every numeric column, cleaned train vs later.csv.
  2. v1's MAE on later.csv, separately for each locality.
  3. Sort later.csv by date. v2 = same model settings, trained on cleaned
     train + the first 70% of later (the later rows weighted more, see
     choose_later_weight). Compare v1 and v2 on the last 30%.
  4. Promote v2 only if its MAE there is at least 15% lower than v1's.
     v2 is always registered (version 2); it only gets @champion (and
     replaces models/model.joblib) if promoted.

Run after train.py:  PYTHONPATH=src python -m pgrent.monitor
"""

import json

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from mlflow.tracking import MlflowClient
from sklearn.metrics import mean_absolute_error

from pgrent.clean import load_all
from pgrent.config import load_params
from pgrent.train import build_pipeline, features, log_and_register, setup_mlflow


def psi(expected, actual, bins=10):
    """Population Stability Index between a reference sample (expected =
    training data) and a new sample (actual = later.csv).

    Bin edges are the deciles of the training data, so each bin holds about
    10% of the training rows. Columns with few distinct values (sharing,
    meals_per_day) give repeated deciles; np.unique merges them into fewer
    bins. The outer edges are opened to -inf/+inf so later values outside
    the training range still land in the first or last bin.

        PSI = sum over bins of (actual% - expected%) * ln(actual% / expected%)

    Rule of thumb: < 0.1 stable, 0.1 - 0.25 some change, > 0.25 big shift.
    """
    expected = pd.Series(expected).dropna().to_numpy(dtype=float)
    actual = pd.Series(actual).dropna().to_numpy(dtype=float)

    edges = np.unique(np.quantile(expected, np.linspace(0, 1, bins + 1)))
    edges[0], edges[-1] = -np.inf, np.inf

    exp_pct = np.histogram(expected, edges)[0] / len(expected)
    act_pct = np.histogram(actual, edges)[0] / len(actual)

    # an empty bin would give ln(0); replace 0% with a tiny number
    exp_pct = np.clip(exp_pct, 1e-6, None)
    act_pct = np.clip(act_pct, 1e-6, None)
    return float(np.sum((act_pct - exp_pct) * np.log(act_pct / exp_pct)))


def fit_weighted(candidate, params, old_rows, new_rows, weight):
    """Train one pipeline on old + new rows, where every new row counts
    `weight` times as much as an old one (sample_weight)."""
    cols = params["columns"]
    combined = pd.concat([old_rows, new_rows], ignore_index=True)
    weights = np.r_[np.ones(len(old_rows)), np.full(len(new_rows), weight)]
    pipe = build_pipeline(candidate, params)
    pipe.fit(features(combined, params), combined[cols["target"]], model__sample_weight=weights)
    return pipe


def choose_later_weight(candidate, params, clean, later_fit):
    """Sector West rents jumped in July, so the old January-June rows describe
    old prices. Giving the new rows more weight lets v2 learn the new level.

    The weight is chosen using ONLY the first 70% of later.csv: we split it
    again (first 70% / last 30% by date), try each weight, and keep the one
    with the lowest MAE. The final check rows are never looked at here.
    """
    cols = params["columns"]
    n_inner = int(len(later_fit) * params["monitoring"]["later_fit_fraction"])
    inner_fit, inner_val = later_fit.iloc[:n_inner], later_fit.iloc[n_inner:]
    scores = {}
    for w in params["monitoring"]["later_weight_options"]:
        pipe = fit_weighted(candidate, params, clean, inner_fit, w)
        scores[w] = mean_absolute_error(inner_val[cols["target"]],
                                        pipe.predict(features(inner_val, params)))
        print(f"  later weight {w:>4}: inner MAE {scores[w]:8.1f}")
    return min(scores, key=scores.get)


def mae_by_group(model, df, params):
    """v1's MAE on df, one number per value of the group column."""
    cols = params["columns"]
    out = {}
    for value, part in df.groupby(cols["group"]):
        out[value] = float(mean_absolute_error(part[cols["target"]], model.predict(features(part, params))))
    return out


def main():
    params = load_params()
    cols, mon = params["columns"], params["monitoring"]
    setup_mlflow(params)
    client = MlflowClient()
    name = params["mlflow"]["registered_model_name"]

    versions = client.search_model_versions(f"name='{name}'")
    if len(versions) != 1:
        raise SystemExit(f"Expected only v1 in the registry, found {len(versions)} versions. "
                         "Run train.py first (it resets the registry).")

    clean, later, _ = load_all(params)
    v1 = mlflow.sklearn.load_model(f"models:/{name}/1")

    # --- 1. drift --------------------------------------------------------
    print("\nPSI, cleaned train vs later.csv (10 bins):")
    psi_values = {c: round(psi(clean[c], later[c], mon["psi_bins"]), 4) for c in cols["numeric"]}
    for c, v in sorted(psi_values.items(), key=lambda kv: -kv[1]):
        print(f"  {c:<24} {v:.4f}")
    most_drifted = max(psi_values, key=psi_values.get)

    # --- 2. MAE per group ----------------------------------------------
    group_mae = {k: round(v, 2) for k, v in mae_by_group(v1, later, params).items()}
    worst_group = max(group_mae, key=group_mae.get)
    print(f"\nv1 MAE on later.csv by {cols['group']}:")
    for k, v in group_mae.items():
        print(f"  {k:<12} {v:8.1f}")

    # --- 3. retrain v2 ---------------------------------------------------
    later = later.sort_values([cols["date"], cols["id"]]).reset_index(drop=True)
    n_fit = int(len(later) * mon["later_fit_fraction"])
    later_fit, check = later.iloc[:n_fit], later.iloc[n_fit:]
    print(f"\nlater.csv: first {len(later_fit)} rows for retraining "
          f"(to {later_fit[cols['date']].max()}), last {len(check)} rows to compare "
          f"(from {check[cols['date']].min()})")

    # v2 uses exactly the same model settings as v1, read back from v1's run
    v1_run = client.get_run(client.get_model_version(name, "1").run_id)
    candidate = next(c for c in params["model"]["candidates"]
                     if c["name"] == v1_run.data.params["candidate"])
    weight = choose_later_weight(candidate, params, clean, later_fit)
    print(f"  -> v2 counts each later.csv row {weight}x")
    v2 = fit_weighted(candidate, params, clean, later_fit, weight)

    mae_v1 = float(mean_absolute_error(check[cols["target"]], v1.predict(features(check, params))))
    mae_v2 = float(mean_absolute_error(check[cols["target"]], v2.predict(features(check, params))))

    # --- 4. promotion rule ----------------------------------------------
    promote = mae_v2 <= (1 - mon["promote_if_mae_drop_at_least"]) * mae_v1
    decision = "promote" if promote else "keep"
    print(f"\nlast 30%: v1 MAE {mae_v1:.1f}, v2 MAE {mae_v2:.1f} "
          f"({(1 - mae_v2 / mae_v1) * 100:.1f}% lower) -> {decision}")

    run_id, version = log_and_register(
        v2, params, "v2",
        {"model_type": candidate["type"], "candidate": candidate["name"],
         "later_rows_used": len(later_fit), "later_weight": weight, **candidate["params"]},
        {"mae_v1_on_check": mae_v1, "mae_v2_on_check": mae_v2, "mae_later_v1": float(
            mean_absolute_error(later[cols["target"]], v1.predict(features(later, params))))},
        len(clean) + len(later_fit))
    client.set_model_version_tag(name, str(version), "decision", decision)
    if promote:
        client.set_registered_model_alias(name, params["mlflow"]["champion_alias"], version)
        joblib.dump(v2, params["paths"]["model"])
        print(f"v2 = version {version} is now @champion; models/model.joblib replaced")
    else:
        print(f"v2 = version {version} registered, v1 stays @champion")

    report = {
        "psi": psi_values,
        "most_drifted_feature": most_drifted,
        "mae_by_group": group_mae,
        "worst_served_group": worst_group,
        "mae_v1_on_check": round(mae_v1, 2),
        "mae_v2_on_check": round(mae_v2, 2),
        "decision": decision,
        "v2_run_id": run_id,
        "champion_version": version if promote else 1,
    }
    with open(params["paths"]["monitoring_report"], "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nwrote {params['paths']['monitoring_report']}")


if __name__ == "__main__":
    main()
