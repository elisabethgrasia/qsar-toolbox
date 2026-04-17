import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.DataStructs import ConvertToNumpyArray


def featurize_morgan(smiles_list, radius=2, n_bits=2048):
    features = []
    valid_mask = []

    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)

        if mol is None:
            features.append(np.zeros(n_bits, dtype=np.float32))
            valid_mask.append(False)
            continue

        fp = AllChem.GetMorganFingerprintAsBitVect(
            mol,
            radius=radius,
            nBits=n_bits
        )

        arr = np.zeros((n_bits,), dtype=np.float32)
        ConvertToNumpyArray(fp, arr)

        features.append(arr)
        valid_mask.append(True)

    X = np.array(features, dtype=np.float32)
    valid_mask = np.array(valid_mask, dtype=bool)

    return X, valid_mask

def featurize_smiles(smiles_list, method="morgan", **kwargs): 
    if method == "morgan": 
        return featurize_morgan(smiles_list, **kwargs) 
    raise ValueError(f"Unknown method: {method}")