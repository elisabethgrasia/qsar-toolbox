from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression


MODEL_REGISTRY = {
    "rf": RandomForestClassifier,
    "svm": SVC,
    "logreg": LogisticRegression,
    "xgb": XGBClassifier
}

def list_models():
    return sorted(MODEL_REGISTRY.keys())


def make_model(name, **params):
     if name not in MODEL_REGISTRY:
        available = ", ".join(MODEL_REGISTRY.keys())
        raise ValueError(f"Unknown model '{name}'. Available: {available}")
        return MODEL_REGISTRY[name](**params)


