"""Small helpers every other module needs: settings and the data hash."""

import hashlib

import yaml

PARAMS_FILE = "params.yaml"


def load_params(path=PARAMS_FILE):
    """Read params.yaml into a plain Python dict.

    safe_load only builds dicts, lists, strings and numbers, so a YAML file
    can never run code on our machine.
    """
    with open(path) as f:
        return yaml.safe_load(f)


def data_sha256(path):
    """SHA-256 fingerprint of a file, read as raw bytes.

    We log this for train.csv in every MLflow run so anyone can prove which
    exact file a model was trained on.
    """
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()
