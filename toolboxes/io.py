import json
import joblib
import pandas as pd


def save_artifact(path, artifact):
    joblib.dump(artifact, path)


def load_artifact(path):
    return joblib.load(path)


def save_config(path, config):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_predictions(path, predictions):
    if isinstance(predictions, pd.DataFrame):
        predictions.to_csv(path, index=False)
    else:
        pd.DataFrame(predictions).to_csv(path, index=False)


def load_predictions(path):
    return pd.read_csv(path)