"""pgrent: fair-rent model for the Kochi Student Housing Board (MP203).

Modules:
    config       read params.yaml, hash of train.csv
    data_checks  find the seven kinds of data problem in train.csv
    clean        fix those problems
    train        train v1 with MLflow tracking and the model registry
    monitor      drift (PSI), MAE per locality, retrain v2, promotion rule
    predict      predict holdout_features.csv with the @champion model
"""
