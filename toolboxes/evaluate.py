import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    matthews_corrcoef,
)


def evaluate_binary_classifier(y_true, y_pred=None, y_score=None):
    y_true = np.asarray(y_true)

    metrics = {}

    if y_pred is not None:
        y_pred = np.asarray(y_pred)
        metrics["accuracy"] = float(accuracy_score(y_true, y_pred))
        metrics["precision"] = float(precision_score(y_true, y_pred, zero_division=0))
        metrics["recall"] = float(recall_score(y_true, y_pred, zero_division=0))
        metrics["f1"] = float(f1_score(y_true, y_pred, zero_division=0))
        metrics["mcc"] = float(matthews_corrcoef(y_true, y_pred))

    if y_score is not None:
        y_score = np.asarray(y_score)
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_score))
        metrics["pr_auc"] = float(average_precision_score(y_true, y_score))

    return metrics

def evaluate_multitask_classifier(Y_true, Y_pred=None, Y_score=None):
    Y_true = np.asarray(Y_true)

    n_tasks = Y_true.shape[1]
    results = {}

    for i in range(n_tasks):
        y_true = Y_true[:, i]

        y_pred = None
        if Y_pred is not None:
            y_pred = Y_pred[:, i]

        y_score = None
        if Y_score is not None:
            y_score = Y_score[:, i]

        metrics = evaluate_binary_classifier(
            y_true=y_true,
            y_pred=y_pred,
            y_score=y_score,
        )

        results[f"task_{i}"] = metrics

    return results

def evaluate_model(model, X, y_true):
    X = np.asarray(X)
    y_true = np.asarray(y_true)

    y_pred = model.predict(X)

    y_score = None
    if hasattr(model, "predict_proba"):
        y_score = model.predict_proba(X)[:, 1]
    elif hasattr(model, "decision_function"):
        y_score = model.decision_function(X)

    metrics = evaluate_binary_classifier(
        y_true=y_true,
        y_pred=y_pred,
        y_score=y_score,
    )

    return {
        "metrics": metrics,
        "y_true": y_true,
        "y_pred": y_pred,
        "y_score": y_score,
    }