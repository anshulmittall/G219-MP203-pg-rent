"""Predict rent_rs for every row of holdout_features.csv with the @champion
model from the MLflow registry.

Run last:  PYTHONPATH=src python -m pgrent.predict
"""

import os

import mlflow.sklearn
import pandas as pd

from pgrent.clean import load_all
from pgrent.config import load_params
from pgrent.train import features, setup_mlflow


def main():
    params = load_params()
    setup_mlflow(params)
    uri = f"models:/{params['mlflow']['registered_model_name']}@{params['mlflow']['champion_alias']}"
    model = mlflow.sklearn.load_model(uri)

    _, _, holdout = load_all(params)          # holdout with tidy spellings
    out = pd.DataFrame({
        "record_id": holdout[params["columns"]["id"]],
        "prediction": model.predict(features(holdout, params)).round(2),
    })

    path = params["paths"]["predictions"]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    out.to_csv(path, index=False)
    print(f"\n{uri}: wrote {len(out)} predictions to {path}")
    print(out["prediction"].describe())


if __name__ == "__main__":
    main()
