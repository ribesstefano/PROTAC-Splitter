import logging
import random
from typing import List, Tuple, Callable, Any, Union, Dict, Optional, Literal

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import rdchem
from rdkit import RDLogger
from rdkit.Chem import CanonSmiles

from .chemoinformatics import (
    canonize,
    get_mol,
    smiles2mol,
)

RDLogger.DisableLog("rdApp.*")


def find_atom_idx_of_map_atoms(
        mol: rdchem.Mol,
        find_poi: True,
        find_e3: True,
        poi_attachment_id: int = 1,
        e3_attachment_id: int = 2,
) -> Union[int, Tuple[int, int]]:
    """ Find the indices of the attachment points in the given molecule.

    Args:
        mol (rdkit.Chem.rdchem.Mol): The molecule.
        find_poi (bool): Whether to find the POI attachment point.
        find_e3 (bool): Whether to find the E3 attachment point.
        poi_attachment_id (int): The label of the attachment point for the POI ligand, i.e., "[*:{poi_attachment_id}]".
        e3_attachment_id (int): The label of the attachment point for the E3 binder, i.e., "[*:{e3_attachment_id}]".

    Returns:
        int | Tuple[int, int]: The index of the attachment point for the POI ligand if find_poi is True, the index of the attachment point for the E3 binder if find_e3 is True, or a tuple containing POI and E3 indices (in this order) if both find_poi and find_e3 are True.
    """
    if find_poi and find_e3:
        poi_idx = None
        e3_idx = None
        for atom in mol.GetAtoms():
            if atom.GetAtomMapNum() == poi_attachment_id:
                poi_idx = atom.GetIdx()
            elif atom.GetAtomMapNum() == e3_attachment_id:
                e3_idx = atom.GetIdx()
            if poi_idx is not None and e3_idx is not None:
                break
        return poi_idx, e3_idx
    elif find_poi:
        for atom in mol.GetAtoms():
            if atom.GetAtomMapNum() == poi_attachment_id:
                return atom.GetIdx()
    elif find_e3:
        for atom in mol.GetAtoms():
            if atom.GetAtomMapNum() == e3_attachment_id:
                return atom.GetIdx()


def reassemble_protac(
        ligands_smiles: Optional[str] = None,
        poi_smiles: Optional[str] = None,
        linker_smiles: Optional[str] = None,
        e3_smiles: Optional[str] = None,
        e3_bond_type: Literal['single', 'double', 'triple', 'rand_uniform'] = 'single',
        poi_bond_type: Literal['single', 'double', 'triple', 'rand_uniform'] = 'single',
        poi_attachment_id: int = 1,
        e3_attachment_id: int = 2,
        rand_generator = None,
) -> Tuple[str, Chem.rdchem.Mol]:
    """ Reassemble a PROTAC molecule from its substructures. The SMILES must contain attachment points.
    
    In case the bond type cannot be formed an error will be raised.

    Example of usage:

    ```python
    e3_smiles = '[*:2]NC(C(=O)N1CC(O)CC1C(=O)NCc1ccc(-c2scnc2C)cc1)C(C)(C)C'
    linker_smiles = '[*:2]C(=O)CCCCCCCCCC[*:1]'
    poi_smiles = '[*:1]CN1CCN(c2ccc(Nc3ncc4c(C)cc(=O)n(-c5cccc(NC(=O)C=C)c5)c4n3)c(OC)c2)CC1'

    merged_smiles, _ = reassemble_protac(poi_smiles, linker_smiles, e3_smiles, 'single', 'single')
    print(merged_smiles)
    ```

    Args:
        poi_smiles (str): The SMILES notation for the POI ligand.
        linker_smiles (str): The SMILES notation for the linker.
        e3_smiles (str): The SMILES notation for the E3 binder.
        e3_bond_type (str): The type of bond to be added between the E3 binder and the linker. Can be 'single', 'double', 'triple', or 'rand_uniform'.
        poi_bond_type (str): The type of bond to be added between the POI ligand and the linker. Can be 'single', 'double', 'triple', or 'rand_uniform'.
        poi_attachment_id (int): The label of the attachment point for the POI ligand, i.e., "[*:{poi_attachment_id}]".
        e3_attachment_id (int): The label of the attachment point for the E3 binder, i.e., "[*:{e3_attachment_id}]".
        rand_generator: A random number generator for 'rand_uniform' bond types. Defaults to None, i.e., standard library random.
    
    Returns:
        Tuple[str, Chem.rdchem.Mol]: The SMILES notation and RDKit molecule object for the reassembled PROTAC molecule.
    """
    if ligands_smiles is None:
        if None in [poi_smiles, linker_smiles, e3_smiles]:
            raise ValueError("Missing substructures SMILES: either provide ligands_smiles or all of poi_smiles, linker_smiles, and e3_smiles")
        ligands_smiles = f'{e3_smiles}.{linker_smiles}.{poi_smiles}'
    if None in [poi_smiles, linker_smiles, e3_smiles]:
        if ligands_smiles is None:
            raise ValueError("Missing substructures SMILES: either provide ligands_smiles or all of poi_smiles, linker_smiles, and e3_smiles")
    
    ligands_mol = canonize(smiles2mol(ligands_smiles))
    if ligands_mol is None:
        return None, None
    
    try:
        protac_mol = Chem.molzip(ligands_mol)
    except ValueError as e:
        logging.error(f"Failed to reassemble PROTAC: {e}")
        return None, None