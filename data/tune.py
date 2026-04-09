def tune_classifier(...)

def get_default_search_space(model_name):
    if model_name == "rf":
        return {
            "n_estimators": [100, 200, 300],
            "max_depth": [None, 10, 20],
            "min_samples_split": [2, 5, 10]
        }
    elif model_name == "xgb":
        return {
            "n_estimators": [100, 200, 300],
            "learning_rate": [0.01, 0.1, 0.2],
            "max_depth": [3, 6, 9]
        }
    else:
        raise ValueError(f"Unknown model '{model_name}' for tuning.")
    
    