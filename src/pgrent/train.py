"""Train model v1 on the cleaned train.csv, with MLflow tracking.

What happens:
  1. Clean train.csv (clean.py) and sort it by date.
  2. Hold out the last 20% of dates as a validation set.
  3. One MLflow run per candidate in params.yaml; each logs data_sha256,
     its settings and mae_val.
  4. The candidate with the lowest mae_val is refit on ALL cleaned train
     in one more run ("v1"), registered in the model registry (version 1),
     given the alias @champion, and saved to models/model.joblib for the API.

Run:  PYTHONPATH=src python -m pgrent.train
"""

import os

import joblib
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from pgrent.clean import load_all
from pgrent.config import data_sha256, load_params

MODEL_TYPES = {
    "ridge": Ridge,
    "random_forest": RandomForestRegressor,
    "gradient_boosting": GradientBoostingRegressor,
}


def build_pipeline(candidate, params):
    """Preprocessing + regressor in one sklearn Pipeline.

    Numbers: fill blanks (stuck-sensor rows) with the median, then scale.
    Text:    one-hot encode; an unseen category becomes all zeros instead of
             crashing the API.
    Only standard sklearn pieces are used, so the saved model can be loaded
    by the API without our own code.
    """
    cols = params["columns"]
    preprocess = ColumnTransformer([
        ("num", Pipeline([("impute", SimpleImputer(strategy="median")),
                          ("scale", StandardScaler())]), cols["numeric"]),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cols["categorical"]),
    ])
    settings = dict(candidate["params"])
    if candidate["type"] != "ridge":            # Ridge has no randomness
        settings["random_state"] = params["model"]["random_state"]
    model = MODEL_TYPES[candidate["type"]](**settings)
    return Pipeline([("prep", preprocess), ("model", model)])


def features(df, params):
    """The model's input columns, in a fixed order."""
    cols = params["columns"]
    return df[cols["numeric"] + cols["categorical"]]


def setup_mlflow(params):
    mlflow.set_tracking_uri(params["mlflow"]["tracking_uri"])
    mlflow.set_experiment(params["mlflow"]["experiment_name"])


def log_and_register(pipe, params, run_name, extra_params, metrics, train_rows):
    """One MLflow run that logs a fitted model and registers it.
    Returns (run_id, registered version number)."""
    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_param("data_sha256", data_sha256(params["data"]["train"]))
        mlflow.log_param("train_rows", train_rows)
        mlflow.log_params(extra_params)
        mlflow.log_metrics(metrics)
        info = mlflow.sklearn.log_model(
            sk_model=pipe,
            name="model",
            registered_model_name=params["mlflow"]["registered_model_name"],
            serialization_format="cloudpickle",
        )
    return run.info.run_id, int(info.registered_model_version)


def main():
    params = load_params()
    cols = params["columns"]
    setup_mlflow(params)
    client = MlflowClient()
    name = params["mlflow"]["registered_model_name"]

    # Start the registry fresh so v1 is always version 1 (re-running this
    # script would otherwise keep adding versions).
    if client.search_registered_models(filter_string=f"name='{name}'"):
        client.delete_registered_model(name)

    clean, _, _ = load_all(params)
    n_val = int(len(clean) * params["model"]["validation_fraction"])
    fit_part, val_part = clean.iloc[:-n_val], clean.iloc[-n_val:]   # clean is sorted by date
    print(f"\nfit on {len(fit_part)} rows ({fit_part[cols['date']].min()} .. {fit_part[cols['date']].max()}), "
          f"validate on {len(val_part)} rows ({val_part[cols['date']].min()} .. {val_part[cols['date']].max()})")

    # --- one run per candidate setting --------------------------------------
    results = []
    for cand in params["model"]["candidates"]:
        pipe = build_pipeline(cand, params)
        pipe.fit(features(fit_part, params), fit_part[cols["target"]])
        mae = mean_absolute_error(val_part[cols["target"]], pipe.predict(features(val_part, params)))
        with mlflow.start_run(run_name=cand["name"]):
            mlflow.log_param("data_sha256", data_sha256(params["data"]["train"]))
            mlflow.log_param("model_type", cand["type"])
            mlflow.log_params(cand["params"])
            mlflow.log_metric("mae_val", mae)
        results.append((mae, cand))
        print(f"  {cand['name']:<18} mae_val = {mae:8.1f} Rs/month")

    best_mae, best = min(results, key=lambda r: r[0])
    print(f"\nbest: {best['name']} (mae_val {best_mae:.1f})")

    # --- v1: refit the best setting on ALL cleaned train ---------------------
    v1 = build_pipeline(best, params)
    v1.fit(features(clean, params), clean[cols["target"]])
    run_id, version = log_and_register(
        v1, params, "v1",
        {"model_type": best["type"], "candidate": best["name"], **best["params"]},
        {"mae_val": best_mae}, len(clean))
    client.set_registered_model_alias(name, params["mlflow"]["champion_alias"], version)

    os.makedirs(os.path.dirname(params["paths"]["model"]), exist_ok=True)
    joblib.dump(v1, params["paths"]["model"])

    print(f"\nv1 run id: {run_id}")
    print(f"registered '{name}' version {version} -> @{params['mlflow']['champion_alias']}")
    print(f"saved {params['paths']['model']}")


if __name__ == "__main__":
    main()
