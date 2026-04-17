import numpy as np
from sklearn.model_selection import train_test_split

from .featurizers import featurize_smiles
from .registry import make_model


def train_from_smiles(
    smiles,
    y,
    featurizer_name="morgan",
    featurizer_params=None,
    model_name="rf",
    model_params=None,
    test_size=0.2,
    random_state=42,
):
    featurizer_params = featurizer_params or {}
    model_params = model_params or {}

    X, valid_mask = featurize_smiles(
        smiles,
        method=featurizer_name,
        **featurizer_params,
    )

    smiles = np.asarray(smiles)
    y = np.asarray(y)

    X = X[valid_mask]
    y = y[valid_mask]
    smiles = smiles[valid_mask]

    X_train, X_valid, y_train, y_valid, smiles_train, smiles_valid = train_test_split(
        X,
        y,
        smiles,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    model = make_model(model_name, **model_params)
    model.fit(X_train, y_train)

    return {
        "model": model,
        "X_train": X_train,
        "X_valid": X_valid,
        "y_train": y_train,
        "y_valid": y_valid,
        "smiles_train": smiles_train,
        "smiles_valid": smiles_valid,
        "valid_mask": valid_mask,
        "featurizer_name": featurizer_name,
        "featurizer_params": featurizer_params,
        "model_name": model_name,
        "model_params": model_params,
    }