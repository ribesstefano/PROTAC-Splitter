import logging
import random
from typing import List, Tuple, Callable, Any, Union, Dict, Optional, Literal
from functools import lru_cache

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import rdchem
from rdkit import RDLogger
from rdkit.Chem import CanonSmiles

from .chemoinformatics import (
    remove_dummy_atoms,
    merge_molecules,
)

RDLogger.DisableLog("rdApp.*")


@lru_cache(maxsize=None)
def get_mol(smiles: str) -> rdchem.Mol:
    return Chem.MolFromSmiles(smiles)


def find_atom_idx_of_map_atoms(
        mol: rdchem.Mol,
        find_poi: True,
        find_e3: True,
        poi_attachment_id: int = 1,
        e3_attachment_id: int = 2,
) -> int | Tuple[int, int]:
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
        poi_smiles: str,
        linker_smiles: str,
        e3_smiles: str,
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

    # Convert SMILES to RDKit Molecule objects
    poi_mol = get_mol(poi_smiles)
    linker_mol = get_mol(linker_smiles)
    e3_mol = get_mol(e3_smiles)

    if poi_mol is None or linker_mol is None or e3_mol is None:
        raise ValueError("Invalid substructures SMILES")

    # Find the indices of the attachment points
    poi_idx = find_atom_idx_of_map_atoms(
        poi_mol,
        find_poi=True,
        find_e3=False,
        poi_attachment_id=poi_attachment_id,
        e3_attachment_id=e3_attachment_id,
    )
    linker_poi_idx, linker_e3_idx = find_atom_idx_of_map_atoms(
        linker_mol,
        find_poi=True,
        find_e3=True,
        poi_attachment_id=poi_attachment_id,
        e3_attachment_id=e3_attachment_id,
    )
    e3_idx = find_atom_idx_of_map_atoms(
        e3_mol,
        find_poi=False,
        find_e3=True,
        poi_attachment_id=poi_attachment_id,
        e3_attachment_id=e3_attachment_id,
    )

    # Ensure that each molecule has the correct number of attachment points
    if poi_idx is None or linker_poi_idx is None or linker_e3_idx is None or e3_idx is None:
        raise ValueError("Missing attachment points in one or more substructures")

    # Merge E3 with Linker
    e3_linker_mol = merge_molecules(e3_mol, linker_mol, e3_idx, linker_e3_idx, bond_type=e3_bond_type, rand_generator=rand_generator)
    linker_e3_mol_idx = find_atom_idx_of_map_atoms(
        e3_linker_mol,
        find_poi=True,
        find_e3=False,
        poi_attachment_id=poi_attachment_id,
        e3_attachment_id=e3_attachment_id,
    )

    protac_mol = merge_molecules(e3_linker_mol, poi_mol, linker_e3_mol_idx, poi_idx, bond_type=poi_bond_type, rand_generator=rand_generator)
    Chem.SanitizeMol(protac_mol)
    
    # # Reassign stereochemistry
    # Chem.AssignStereochemistry(protac_mol, force=True, cleanIt=True)

    protac_smiles = Chem.MolToSmiles(protac_mol, canonical=True)

    return protac_smiles, protac_mol


def reassemble_protac_with_reactions(
        poi_smiles: str,
        linker_smiles: str,
        e3_smiles: str,
        e3_bond_type: Literal['single', 'double', 'triple', 'rand_uniform'] = 'single',
        poi_bond_type: Literal['single', 'double', 'triple', 'rand_uniform'] = 'single',
        poi_attachment_id: int = 1,
        e3_attachment_id: int = 2,
        rand_generator=None,
) -> Tuple[str, Chem.rdchem.Mol]:
    """
    Reassemble a PROTAC molecule from its substructures. The SMILES must contain attachment points.

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
    # Map bond type strings to bond symbols
    bond_symbols = {
        'single': '-',
        'double': '=',
        'triple': '#'
    }
    possible_bonds = ['single', 'double', 'triple']

    if rand_generator is None:
        rand_generator = random

    # Determine bond symbols for E3-linker and linker-POI bonds
    if e3_bond_type == 'rand_uniform':
        e3_bond_type = rand_generator.choice(possible_bonds)
    if poi_bond_type == 'rand_uniform':
        poi_bond_type = rand_generator.choice(possible_bonds)

    # Get bond symbols
    e3_bond_symbol = bond_symbols.get(e3_bond_type)
    poi_bond_symbol = bond_symbols.get(poi_bond_type)

    if e3_bond_symbol is None or poi_bond_symbol is None:
        raise ValueError("Invalid bond type. Must be 'single', 'double', 'triple', or 'rand_uniform'.")

    # Create reaction SMARTS for E3-linker connection
    rxn_smarts_e3 = f'[*:{e3_attachment_id}].[*:{e3_attachment_id}]>>[*:{e3_attachment_id}]{e3_bond_symbol}[*:{e3_attachment_id}]'
    rxn_smarts_e3 = f'[*:{e3_attachment_id}].[*:{e3_attachment_id}]>>[*:{e3_attachment_id}]{e3_bond_symbol}[*:{e3_attachment_id}]'
    rxn_e3 = AllChem.ReactionFromSmarts(rxn_smarts_e3)

    # Create reaction SMARTS for linker-POI connection
    rxn_smarts_poi = f'[*:{poi_attachment_id}].[*:{poi_attachment_id}]>>[*:{poi_attachment_id}]{poi_bond_symbol}[*:{poi_attachment_id}]'
    rxn_smarts_poi = f'[*:{poi_attachment_id}].[*:{poi_attachment_id}]>>[*:{poi_attachment_id}]{poi_bond_symbol}[*:{poi_attachment_id}]'
    rxn_poi = AllChem.ReactionFromSmarts(rxn_smarts_poi)

    # Convert SMILES to RDKit molecules
    poi_mol = Chem.MolFromSmiles(poi_smiles)
    linker_mol = Chem.MolFromSmiles(linker_smiles)
    e3_mol = Chem.MolFromSmiles(e3_smiles)

    if not all([poi_mol, linker_mol, e3_mol]):
        raise ValueError("Invalid substructures SMILES")

    # First, connect the E3 to the linker via [*:{e3_attachment_id}]
    products = rxn_e3.RunReactants((e3_mol, linker_mol))
    if not products:
        raise ValueError("Reaction to connect E3 and linker failed")
    e3_linker_mol = products[0][0]

    # Then, connect the POI to the e3_linker_mol via [*:{poi_attachment_id}]
    products = rxn_poi.RunReactants((e3_linker_mol, poi_mol))
    if not products:
        raise ValueError("Reaction to connect POI and E3-Linker failed")
    protac_mol = products[0][0]

    # Sanitize and assign stereochemistry
    Chem.SanitizeMol(protac_mol)
    Chem.AssignStereochemistry(protac_mol, force=True, cleanIt=True)

    protac_smiles = Chem.MolToSmiles(protac_mol, canonical=True, isomericSmiles=True)

    return protac_smiles, protac_mol


def substructure_split_sort(substructure_smiles: str) -> Tuple[str, str, str]:
    """
    Splits a substructure SMILES string into three parts: poi_smile, linker_smile, and e3_smile.

    Args:
        substructure_smiles (str): The substructure SMILES string to be split.

    Returns:
        Tuple[str, str, str]: A tuple containing the poi_smile, linker_smile, and e3_smile.

    Raises:
        ValueError: If [*:1] and [*:2] are not found in any of the substructure SMILES.
    """
    if isinstance(substructure_smiles, str):
        substructure_smiles = substructure_smiles.split(".")
    for smile in substructure_smiles:
        if '[*:1]' in smile:
            if '[*:2]' in smile:
                linker_smile = smile
            else:
                poi_smile = smile
        elif '[*:2]' in smile:
            e3_smile = smile
        else:
            raise ValueError(
                f'[*:1] and [*:2] was not found in smile: {smile}')
    return poi_smile, linker_smile, e3_smile


def identify_bad_substructure_match(protac_smile: str, poi_smile: str, e3_smile: str) -> bool:
    """
    Identifies if the substructure match between the PROTAC and the POI and E3 ligands is bad.

    Args:
        protac_smile (str): The SMILES representation of the PROTAC molecule.
        poi_smile (str): The SMILES representation of the POI (Protein of Interest) molecule.
        e3_smile (str): The SMILES representation of the E3 ligand molecule.

    Returns:
        bool: True if the substructure match is bad, False otherwise.
    """
    protac_mol = Chem.MolFromSmiles(protac_smile)
    poi_mol = Chem.MolFromSmiles(poi_smile)
    e3_mol = Chem.MolFromSmiles(e3_smile)

    poi_mol = remove_dummy_atoms(poi_mol)
    e3_mol = remove_dummy_atoms(e3_mol)

    matches_poi = protac_mol.GetSubstructMatches(poi_mol)
    matches_e3 = protac_mol.GetSubstructMatches(e3_mol)

    if len(matches_poi) == 1 and len(matches_e3) == 1:
        return False
    elif len(matches_poi) == 2 and len(matches_e3) == 2:  # OBS Work in progress
        return True
    else:
        return True


def get_linker_murcko(mol: Chem.Mol) -> Chem.Mol:
    """
    Converts a linker molecule to its Murcko Scaffold representation.

    Args:
        mol (Chem.Mol): The linker molecule to be converted.

    Returns:
        Chem.Mol: The Murcko Scaffold representation of the linker molecule.
    """

    if mol.GetNumAtoms() == 2:  # only [*:1] and [*:2]
        return mol

    poi_l_attachment_point, e3_l_attachment_point = find_atom_idx_of_map_atoms(
        mol)

    emol = Chem.EditableMol(mol)

    # add one single bond between the attachment points
    try:
        emol.AddBond(
            poi_l_attachment_point[0], e3_l_attachment_point[0], Chem.Chem.rdchem.BondType.SINGLE)
    except:
        # display(mol)
        print(f'poi_l_attachment_point:{poi_l_attachment_point}')
        print(f'e3_l_attachment_point:{e3_l_attachment_point}')
        print(Chem.MolToSmiles(mol, canonical=True))
        raise ValueError("Fail add bond")

    mol_circulized = emol.GetMol()
    try:
        # Sanitize the molecule
        # Finding rings and re-perceiving aromaticity
        Chem.GetSymmSSSR(mol_circulized)
        Chem.SanitizeMol(mol_circulized)
    except:
        raise ValueError("Fail GetSymmSSSR or SanitizeMol")

    # apply MS
    mol_circulized_ms = Chem.Scaffolds.MurckoScaffold.GetScaffoldForMol(
        mol_circulized)
    ms_poi_l_attachment_point, ms_e3_l_attachment_point = find_atom_idx_of_map_atoms(
        mol_circulized_ms)
    # mol_circulized_ms.GetBondBetweenAtoms(ms_poi_l_attachment_point, ms_e3_l_attachment_point).SetBondType(Chem.Chem.rdchem.BondType.UNSPECIFIED)

    emol_circulized_ms = Chem.EditableMol(mol_circulized_ms)

    # remove the bond between the attachment points
    emol_circulized_ms.RemoveBond(
        ms_poi_l_attachment_point[0], ms_e3_l_attachment_point[0])

    mol_ms = emol_circulized_ms.GetMol()

    try:
        # Sanitize the molecule
        Chem.GetSymmSSSR(mol_ms)  # Finding rings and re-perceiving aromaticity
        Chem.SanitizeMol(mol_ms)
    except:
        raise ValueError("Fail GetSymmSSSR or SanitizeMol")

    return mol_ms


def get_murcko(smiles: str) -> str:
    """
    Get the Murcko scaffold for a given SMILES string.

    Args:
        smiles (str): The input SMILES string.

    Returns:
        str: The Murcko scaffold SMILES string.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:  # Handle invalid SMILES strings
        raise ValueError("mol is None")

    if "[*:1]" in smiles and "[*:2]" in smiles:  # is_linker = True
        mol_ms = get_linker_murcko(mol)
    else:
        mol_ms = Chem.Scaffolds.MurckoScaffold.GetScaffoldForMol(mol)

    smi_ms = Chem.MolToSmiles(mol_ms, canonical=True)

    return smi_ms