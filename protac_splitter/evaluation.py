""" Evaluation functions for the protac_splitter package. They need to be generic to accomodate predictions coming from different models. """

from typing import List, Tuple, Callable, Any, Union, Dict, Optional, Literal

from rdkit import Chem
from rdkit import Chem, RDLogger

# Disable RDKit logging: when checking SMILES validity, we suppress warnings
RDLogger.DisableLog("rdApp.*")

from .chemoinformatics import standardize_smiles
from .protac_cheminformatics import reassemble_protac


def check_substructs(
        protac_smiles: str,
        poi_smiles: str,
        linker_smiles: str,
        e3_smiles: str,
        return_bond_types: bool = False,
        poi_attachment_id: int = 1,
        e3_attachment_id: int = 2,
) -> bool | Tuple[bool, dict[str, str]]:
    """ Check if the reassembled PROTAC is correct. 
    
    Args:
        protac_smiles (str): The SMILES notation for the PROTAC molecule.
        poi_smiles (str): The SMILES notation for the POI ligand.
        linker_smiles (str): The SMILES notation for the linker.
        e3_smiles (str): The SMILES notation for the E3 binder.
        return_bond_types (bool): If True, return the bond types used for the reassembly.
        poi_attachment_id (int): The label of the attachment point for the POI ligand, i.e., "[*:{poi_attachment_id}]".
        e3_attachment_id (int): The label of the attachment point for the E3 binder, i.e., "[*:{e3_attachment_id}]".

    Returns:
        bool | Tuple[bool, dict[str, str]]: True if the reassembled PROTAC is correct, False otherwise. If return_bond_types is True, also return the bond types used for the reassembly.
    """
    correct_substructs = False
    protac_mol = Chem.MolFromSmiles(protac_smiles)
    protac_inchi = Chem.MolToInchi(protac_mol)
    protac_smiles_canon = standardize_smiles(protac_smiles)
    bond_types = {}
    for e3_bond_type in ['single', 'double', 'triple']:
        for poi_bond_type in ['single', 'double', 'triple']:
            try:
                _, assmbl_mol = reassemble_protac(
                    poi_smiles,
                    linker_smiles,
                    e3_smiles,
                    e3_bond_type,
                    poi_bond_type,
                    poi_attachment_id,
                    e3_attachment_id,
                )
                if assmbl_mol is not None:
                    # If either the InChI or SMILES of the reassembled PROTAC is
                    # the same as the original PROTAC, then the reassembly is
                    # correct.
                    if protac_inchi == Chem.MolToInchi(assmbl_mol):
                        correct_substructs = True
                        bond_types['e3_bond_type'] = e3_bond_type
                        bond_types['poi_bond_type'] = poi_bond_type
                        break
                    if protac_smiles_canon == standardize_smiles(Chem.MolToSmiles(assmbl_mol)):
                        correct_substructs = True
                        bond_types['e3_bond_type'] = e3_bond_type
                        bond_types['poi_bond_type'] = poi_bond_type
                        break
            except:
                continue
    if return_bond_types:
        return correct_substructs, bond_types
    return correct_substructs


def is_valid_smiles(smiles: str) -> bool:
    mol = Chem.MolFromSmiles(smiles)
    return 1 if mol is not None else 0


def has_three_substructures(smiles: str) -> bool:
    return smiles.count(".") == 2


def has_all_attachment_points(smiles: str) -> bool:
    return smiles.count("[*:1]") == 2 and smiles.count("[*:2]") == 2


def is_substructure(protac_smiles: str, substruct_smiles: str) -> bool:
    """ Check if a molecule is a substructure of another molecule.
    
    Args:
        protac_smiles (str): The SMILES notation for the PROTAC molecule.
        substruct_smiles (str): The SMILES notation for the substructure molecule.

    Returns:
        bool: True if the substructure molecule is a substructure of the PROTAC molecule, False otherwise.
    """
    protac_mol = Chem.MolFromSmiles(protac_smiles)
    substruct_mol = Chem.MolFromSmarts(substruct_smiles)
    return protac_mol.HasSubstructMatch(substruct_mol)


def same_atom_counts_and_types(smiles1, smiles2, get_atoms_diff=False):
    """
    Check if two molecules have the same number and types of atoms.

    Args:
    smiles1 (str): SMILES notation for the first molecule.
    smiles2 (str): SMILES notation for the second molecule.

    Returns:
    bool: True if the molecules have the same atom counts and types, False otherwise.
    """
    if isinstance(smiles1, str):
        mol1 = Chem.MolFromSmiles(smiles1)
    else:
        mol1 = smiles1
    if isinstance(smiles2, str):
        mol2 = Chem.MolFromSmiles(smiles2)
    else:
        mol2 = smiles2
    if mol1 is None or mol2 is None:
        if get_atoms_diff:
            return False
            # raise ValueError("Invalid SMILES notation provided for one or both molecules.")
        else:
            return False
    num_atoms1 = Chem.rdMolDescriptors.CalcNumHeavyAtoms(mol1)
    num_atoms2 = Chem.rdMolDescriptors.CalcNumHeavyAtoms(mol2)
    if get_atoms_diff:
        return abs(num_atoms1 - num_atoms2)
        # tmp = {}
        # for atom in atom_counts1.keys():
        #     tmp[atom] = int(abs(atom_counts1.get(atom, 0) - atom_counts2.get(atom, 0)))
        # for atom in atom_counts2.keys():
        #     tmp[atom] = int(abs(atom_counts1.get(atom, 0) - atom_counts2.get(atom, 0)))
        # return tmp # abs(atom_counts1.get('O', 0) - atom_counts2.get('O', 0))
    else:
        atom_counts1, atom_counts2 = {}, {}
        for atom in mol1.GetAtoms():
            if '*' not in atom.GetSmarts():
                atom_counts1[atom.GetSymbol()] = atom_counts1.get(atom.GetSymbol(), 0) + 1
        for atom in mol2.GetAtoms():
            if '*' not in atom.GetSmarts():
                atom_counts2[atom.GetSymbol()] = atom_counts2.get(atom.GetSymbol(), 0) + 1
        return (atom_counts1 == atom_counts2) & (num_atoms1 == num_atoms2)