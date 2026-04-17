import deepchem as dc
import pandas as pd


def load_tox21():
    tasks, datasets, transformers = dc.molnet.load_tox21()
    train_dataset, valid_dataset, test_dataset = datasets

    return {
        "train": train_dataset,
        "valid": valid_dataset,
        "test": test_dataset,
        "tasks": tasks
    }


def load_tox21_smiles_labels():
    data = load_tox21()

    return {
        "train_smiles": data["train"].ids,
        "train_y": data["train"].y,

        "valid_smiles": data["valid"].ids,
        "valid_y": data["valid"].y,

        "test_smiles": data["test"].ids,
        "test_y": data["test"].y,

        "tasks": data["tasks"]
    }


def load_tox21_dataframe(split="train"):
    data = load_tox21()

    dataset = data[split]

    df = pd.DataFrame({
        "smiles": dataset.ids,
    })

    # Add each task as a column
    for i, task in enumerate(data["tasks"]):
        df[task] = dataset.y[:, i]

    return df


def debug_tox21():
    data = load_tox21()
    print("Train shape:", data["train"].y.shape)
    print("Tasks:", data["tasks"])