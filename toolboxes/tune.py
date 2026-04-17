# tune.py

import numpy as np
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold

from .registry import make_model


def get_default_search_space(model_name):
    if model_name == "rf":
        return {
            "n_estimators": [100, 200, 300, 500],
            "max_depth": [None, 5, 10, 20],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
            "class_weight": ["balanced", "balanced_subsample", None],
        }

    if model_name == "svm":
        return {
            "C": [0.1, 1.0, 10.0, 100.0],
            "kernel": ["linear", "rbf"],
            "probability": [True],
            "class_weight": ["balanced", None],
        }

    if model_name == "logreg":
        return {
            "C": [0.01, 0.1, 1.0, 10.0, 100.0],
            "class_weight": ["balanced", None],
            "max_iter": [1000],
        }

    raise ValueError(f"No default search space defined for model: {model_name}")


def tune_classifier(
    X,
    y,
    model_name,
    model_params=None,
    param_distributions=None,
    n_iter=20,
    cv=5,
    scoring="roc_auc",
    random_state=42,
    n_jobs=-1,
):
    model_params = model_params or {}
    X = np.asarray(X)
    y = np.asarray(y)

    base_model = make_model(model_name, **model_params)

    if param_distributions is None:
        param_distributions = get_default_search_space(model_name)

    cv_splitter = StratifiedKFold(
        n_splits=cv,
        shuffle=True,
        random_state=random_state,
    )

    search = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=param_distributions,
        n_iter=n_iter,
        scoring=scoring,
        cv=cv_splitter,
        random_state=random_state,
        n_jobs=n_jobs,
        refit=True,
        verbose=1,
    )

    search.fit(X, y)

    return {
        "best_model": search.best_estimator_,
        "best_params": search.best_params_,
        "best_score": search.best_score_,
        "cv_results": search.cv_results_,
        "search_object": search,
    }

def tune_rf_classifier(
    X,
    y,
    n_iter=20,
    cv=5,
    scoring="roc_auc",
    random_state=42,
    n_jobs=-1,
):
    X = np.asarray(X)
    y = np.asarray(y)

    model = make_model("rf", random_state=random_state)

    param_distributions = {
        "n_estimators": [100, 200, 300, 500],
        "max_depth": [None, 5, 10, 20],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "class_weight": ["balanced", "balanced_subsample", None],
    }

    cv_splitter = StratifiedKFold(
        n_splits=cv,
        shuffle=True,
        random_state=random_state,
    )

    search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_distributions,
        n_iter=n_iter,
        scoring=scoring,
        cv=cv_splitter,
        random_state=random_state,
        n_jobs=n_jobs,
        refit=True,
    )

    search.fit(X, y)

    return {
        "best_model": search.best_estimator_,
        "best_params": search.best_params_,
        "best_score": search.best_score_,
        "cv_results": search.cv_results_,
    }