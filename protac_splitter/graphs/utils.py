from typing import Any, Optional, List

import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator

def get_fp(
    smiles: str,
    fp_generator: Optional[Any] = None,
    return_np: bool = True,
) -> Optional[np.ndarray]:
    """
    Get the Morgan fingerprint of a molecule from its SMILES representation.
    
    Parameters:
        smiles (str): The SMILES string of the molecule.
        fp_generator (Any, optional): The fingerprint generator to use. If None, a default generator is used.
        return_np (bool): Whether to return the fingerprint as a NumPy array. Defaults to True.
        
    Returns:
        Optional[np.ndarray]: The Morgan fingerprint of the molecule as a NumPy array, or None if the SMILES is invalid.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
 
    if fp_generator is None:
        fp_generator = rdFingerprintGenerator.GetMorganGenerator(
            radius=16,
            fpSize=1024,
            useBondTypes=True,
            includeChirality=True,
        )

    if return_np:
        return fp_generator.GetFingerprintAsNumPy(mol)
    else:
        return fp_generator.GetFingerprint(mol)

def max_tanimoto_similarity(
    smiles: str,
    fingerprints: List[DataStructs.ExplicitBitVect],
    morgan_fp_generator: Optional[Any] = None,
) -> float:
    """ Compute the maximum Tanimoto similarity between a query SMILES and a list
        of RDKit fingerprints.

    Parameters:
        smiles (str): SMILES string of the query molecule.
        fingerprints (list): List of RDKit fingerprint objects (e.g., ExplicitBitVect).
        morgan_fp_generator: RDKit Morgan fingerprint generator.

    Returns:
        float: Maximum Tanimoto similarity between the query and the fingerprints.
    """
    query_fp = get_fp(smiles, morgan_fp_generator, return_np=False)
    if query_fp is None:
        raise ValueError(f"Invalid SMILES string: {smiles}")
    distances = DataStructs.BulkTanimotoSimilarity(
        query_fp,
        fingerprints,
        returnDistance=False,
    )
    return np.array(distances).max()

def numpy_to_rdkit_fp(arr: np.ndarray) -> DataStructs.ExplicitBitVect:
    """ Convert a NumPy array to an RDKit ExplicitBitVect. """
    return DataStructs.CreateFromBitString(''.join(arr.astype(str)))