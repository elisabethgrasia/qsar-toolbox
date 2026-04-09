from registry import make_model

config = {
    "model_name": "rf",
    "model_params": {"n_estimators": 200}
}

model = make_model("rf", **config)

def train_model(X_train, y_train):
    model.fit(X_train, y_train)


