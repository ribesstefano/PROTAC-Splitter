import random
import logging
from typing import List, Tuple, Callable, Any, Union, Dict, Optional, Literal

import numpy as np
from tqdm import tqdm
from rdkit import Chem, DataStructs
from rdkit.Chem import (
    rdchem,
    AllChem,
    rdFingerprintGenerator,
    rdMolHash,
    rdFMCS,
    rdMolAlign,
)


def standardize_smiles(smiles: str, fail_on_error: bool = False) -> str:
    """
    Standardizes a given SMILES string.

    Args:
        smiles (str): The input SMILES string to be standardized.

    Returns:
        str: The standardized SMILES string.
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            return Chem.MolToSmiles(mol, isomericSmiles=False, canonical=True)
        else:
            if fail_on_error:
                raise ValueError(f'Smile returned error: {smiles}')
            else:
                logging.warning(f'Smile returned error: {smiles}')
                return float('nan')
    except Exception as e:
        if fail_on_error:
            raise e
        else:
            logging.warning(f'Smile returned error: {smiles}')
        return float('nan')


def remove_stereo(smiles: str) -> str:
    """
    Remove stereochemistry from a SMILES string.

    Args:
        smiles (str): The input SMILES string.

    Returns:
        str: The SMILES string with stereochemistry removed.
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        Chem.rdmolops.RemoveStereochemistry(mol)
        return Chem.MolToSmiles(mol)
    except:
        return float('nan')


def get_mol(smiles: str) -> Chem.Mol:
    """
    Get a molecule object from a SMILES string.

    Args:
        smiles (str): The SMILES string representing the molecule.

    Returns:
        Chem.Mol: The molecule object.
    """
    mol = Chem.MolFromSmiles(smiles)
    Chem.rdmolops.RemoveStereochemistry(mol)
    return mol


def compute_RDKitFP(
        smiles: Union[str, List[str], List[Chem.Mol]],
        maxPath: int = 7,
        fpSize: int = 2048,
) -> List[Chem.RDKFingerprint]:
    """
    Compute RDKit fingerprints for a given list of SMILES strings or RDKit molecules.

    Args:
        smiles (Union[str, List[str], List[Chem.Mol]]): A single SMILES string or a list of SMILES strings
            or a list of RDKit molecules.
        maxPath (int, optional): The maximum path length for the fingerprints. Defaults to 7.
        fpSize (int, optional): The size of the fingerprint vector. Defaults to 2048.

    Returns:
        List[Chem.RDKFingerprint]: A list of RDKit fingerprints computed from the input SMILES strings or molecules.
    """
    if isinstance(smiles[0], str):
        mols = [get_mol(smi) for smi in smiles]
    else:
        mols = smiles  # assume mols were fed instead
    rdgen = rdFingerprintGenerator.GetRDKitFPGenerator(
        maxPath=maxPath, fpSize=fpSize)
    fps = [rdgen.GetCountFingerprint(mol) for mol in mols]
    return fps


def compute_countMorgFP(
        smiles: List[str],
        radius: int = 2,
) -> List[DataStructs.cDataStructs.ExplicitBitVect]:
    """
    Compute the count-based Morgan fingerprint for a list of SMILES strings.

    Args:
        smiles (List[str]): A list of SMILES strings.
        radius (int, optional): The radius parameter for the Morgan fingerprint. Defaults to 2.

    Returns:
        List[rdkit.DataStructs.cDataStructs.ExplicitBitVect]: A list of count-based Morgan fingerprints.
    """
    if smiles is None:
        return None
    if isinstance(smiles[0], str):
        mols = [get_mol(smi) for smi in smiles]
    else:
        mols = smiles  # assume mols were fed instead
    fpgen = AllChem.GetMorganGenerator(radius=radius)
    fps = [fpgen.GetCountFingerprint(mol) for mol in mols]
    return fps


def tanimoto_similarity_matrix(fps, return_distance=False):
    """
    Calculate a symmetric Tanimoto similarity matrix for a list of fingerprints using bulk operations.

    Parameters:
    - fps: list, RDKit fingerprint objects for which to calculate similarity.

    Returns:
    - np.array, Symmetric square matrix of Tanimoto similarity.
    """
    num_fps = len(fps)
    # Initialize a square matrix of zeros
    sim_matrix = np.zeros((num_fps, num_fps))

    for i in tqdm(range(num_fps)):
        similarities = DataStructs.BulkTanimotoSimilarity(fps[i], fps)
        sim = np.array(similarities)
        sim_matrix[i, :] = sim
        # Set diagonal to 1 as the similarity to self is 1
        sim_matrix[i, i] = 1

    if return_distance:
        return 1 - sim_matrix
    return sim_matrix


def add_attachments(
        list_without_attachments: List[str],
        dict_map_without_to_with: Dict[str, List[str]],
) -> Tuple[List[str], List[Chem.Mol]]:
    """
    Adds attachments to a list of molecules.

    Args:
        list_without_attachments (List[str]): A list of SMILES strings representing molecules without attachments.
        dict_map_without_to_with (Dict[str, List[str]]): A dictionary mapping SMILES strings without attachments to a list of SMILES strings with attachments.

    Returns:
        Tuple[List[str], List[Chem.Mol]]: A tuple containing two lists:
            - smiles_with_attachment (List[str]): A list of SMILES strings representing molecules with attachments.
            - mols (List[Chem.Mol]): A list of RDKit molecule objects corresponding to the molecules with attachments.
    """
    smiles_with_attachment = []
    for smi in list_without_attachments:
        smiles_with_attachment.extend(dict_map_without_to_with[smi])
    smiles_with_attachment = list(set(smiles_with_attachment))
    mols = [Chem.MolFromSmiles(smi) for smi in smiles_with_attachment]
    return smiles_with_attachment, mols


def merge_molecules(
        mol1: Chem.rdchem.Mol,
        mol2: Chem.rdchem.Mol,
        atom_idx1: int,
        atom_idx2: int,
        bond_type: Literal['single', 'double', 'triple', 'rand_uniform'] = 'single',
) -> rdchem.Mol:
    """ Combine two molecules into a single editable molecule.
    
    Args:
        mol1 (rdkit.Chem.rdchem.Mol): The first molecule.
        mol2 (rdkit.Chem.rdchem.Mol): The second molecule.
        atom_idx1 (int): The index of the attachment point in the first molecule.
        atom_idx2 (int): The index of the attachment point in the second molecule.
        bond_type (str): The type of bond to be added between the attachment points. Can be 'single' or 'rand_uniform'.
    
    Returns:
        rdkit.Chem.rdchem.Mol: The combined molecule.
    """
    combined_mol = Chem.CombineMols(mol1, mol2)
    editable_mol = Chem.EditableMol(combined_mol)

    # Find neighbors of the attachment points
    neighbor_atom_idx1 = [nbr.GetIdx() for nbr in mol1.GetAtomWithIdx(atom_idx1).GetNeighbors() if nbr.GetAtomicNum() > 1][0]
    neighbor_atom_idx2 = [nbr.GetIdx() + mol1.GetNumAtoms() for nbr in mol2.GetAtomWithIdx(atom_idx2).GetNeighbors() if nbr.GetAtomicNum() > 1]
    
    if neighbor_atom_idx2 == []: #if linker has no length
        smi_e3_linker_with_e3_attachment = Chem.MolToSmiles(mol1, canonical=True)
        smi_e3_linker_with_poi_attachment = smi_e3_linker_with_e3_attachment.replace("[*:2]","[*:1]")
        mol_e3_linker_with_poi_attachment = Chem.MolFromSmiles(smi_e3_linker_with_poi_attachment)
        return mol_e3_linker_with_poi_attachment
    else:
        neighbor_atom_idx2 = neighbor_atom_idx2[0]

    # Add a bond between the neighboring atoms (ignoring the dummy atoms)
    if bond_type == 'single':
        editable_mol.AddBond(neighbor_atom_idx1, neighbor_atom_idx2, order=rdchem.BondType.SINGLE)
    else:
        # Get the highest allowed bond order for the neighboring atoms
        neighbor_atom1 = mol1.GetAtomWithIdx(neighbor_atom_idx1)
        neighbor_atom2 = mol2.GetAtomWithIdx(neighbor_atom_idx2-mol1.GetNumAtoms())
        max_bond_atom_idx1 = neighbor_atom1.GetTotalNumHs() + 1 # +1 for the attatchment point
        max_bond_atom_idx2 = neighbor_atom2.GetTotalNumHs() + 1
        max_bond = min([max_bond_atom_idx1, max_bond_atom_idx2])
        possible_bonds = [
            rdchem.BondType.SINGLE,
            rdchem.BondType.DOUBLE,
            rdchem.BondType.TRIPLE,
        ][0:max_bond]
        if bond_type == 'rand_uniform':
            sampled_bond = random.sample(possible_bonds, 1)[0]
            editable_mol.AddBond(neighbor_atom_idx1, neighbor_atom_idx2, order=sampled_bond)
        elif bond_type == 'double' and len(possible_bonds) > 1:
            editable_mol.AddBond(neighbor_atom_idx1, neighbor_atom_idx2, order=rdchem.BondType.DOUBLE)
        elif bond_type == 'triple' and len(possible_bonds) > 2:
            editable_mol.AddBond(neighbor_atom_idx1, neighbor_atom_idx2, order=rdchem.BondType.TRIPLE)
        else:
            raise ValueError(f"Invalid bond type requested: {bond_type}. Highest bond order allowed: {possible_bonds[-1]}.")

    # Calculate the adjusted index for the attachment point in mol2
    adjusted_atom_idx2 = atom_idx2 + mol1.GetNumAtoms()

    # Remove the dummy atoms - IMPORTANT: remove the atom with the higher index first!
    max_idx = max(atom_idx1, adjusted_atom_idx2)
    min_idx = min(atom_idx1, adjusted_atom_idx2)

    editable_mol.RemoveAtom(max_idx)
    editable_mol.RemoveAtom(min_idx)

    # Get the modified molecule
    modified_mol = editable_mol.GetMol()

    # Sanitize the molecule to ensure its chemical validity
    Chem.SanitizeMol(modified_mol)

    return modified_mol


def get_boundary_bondtype(mol: Chem.Mol, bondtype_count: Optional[Dict[str, int]] = None) -> Dict[str, int]:
    """
    Get the count of different bond types connected to dummy atoms in a molecule.

    Args:
        mol (Chem.Mol): The molecule to analyze.
        bondtype_count (Optional[Dict[str, int]]): A dictionary to store the count of different bond types.
            Defaults to None.

    Returns:
        Dict[str, int]: A dictionary containing the count of different bond types connected to dummy atoms.
            The keys are the bond types ('SINGLE', 'DOUBLE', 'TRIPLE') and the values are the corresponding counts.
    """
    if bondtype_count is None:
        bondtype_count = {'SINGLE': 0, 'DOUBLE': 0, 'TRIPLE': 0}
    bondtype_to_str = {
        Chem.rdchem.BondType.SINGLE: 'SINGLE',
        Chem.rdchem.BondType.DOUBLE: 'DOUBLE',
        Chem.rdchem.BondType.TRIPLE: 'TRIPLE',
    }

    # Find dummy atoms by symbol or index
    dummy_atoms = [atom.GetIdx() for atom in mol.GetAtoms()
                   if atom.GetSymbol() == '*']
    # reverse to avoid index shifting issues
    for dummy_atom_idx in reversed(dummy_atoms):
        # identify the order of the bond
        atom = mol.GetAtomWithIdx(dummy_atom_idx)
        neighbors = atom.GetNeighbors()

        for neighbour_atom in neighbors:
            neighbour_idx = neighbour_atom.GetIdx()
            dummy_atom_bond = mol.GetBondBetweenAtoms(
                dummy_atom_idx, neighbour_idx)
            dummy_atom_bondtype = dummy_atom_bond.GetBondType()
            bondtype_count[bondtype_to_str[dummy_atom_bondtype]] += 1

    return bondtype_count


# TODO: The following was originally called remove_dummy_atom, without the final 's'.
def remove_dummy_atoms(
        mol: str | Chem.Mol,
        output: str = "smiles",
        how: Literal['all', 'attachments'] = 'all',
) -> Union[str, Chem.Mol]:
    """
    Removes dummy atoms from a molecule and returns the modified molecule.

    Args:
        mol (Chem.Mol): The input molecule containing dummy atoms.
        output (str, optional): The output format. Defaults to "smiles".
        how (str, optional): The method to use for removing dummy atoms. Can be 'all' or 'attachments'. Defaults to 'all'.
            If 'all', removes all dummy atoms with atomic number 0 from the molecule.
            If 'attachments', removes dummy atoms that are connected to other atoms by a bond, i.e., attachment points with '*'.

    Returns:
        Union[str, Chem.Mol]: The modified molecule without dummy atoms. If output is "smiles", returns the SMILES string representation of the molecule. Otherwise, returns the modified molecule as a Chem.Mol object.
    """
    if how not in ['all', 'attachments']:
        raise ValueError("Invalid value for 'how'. Must be 'all' or 'attachments'.")

    if isinstance(mol, str):
        mol = Chem.MolFromSmiles(mol)

    # TODO: Why this?
    # if Chem.MolToSmiles(mol) == "O=C(CCCCCCCCCC[*:1])[*:2]":
    #     pass

    if how == 'attachments':
        atoms_to_remove = [atom.GetIdx() for atom in mol.GetAtoms() if atom.GetAtomicNum() == 0]
        editable_mol = Chem.EditableMol(mol)
        for idx in sorted(atoms_to_remove, reverse=True):
            editable_mol.RemoveAtom(idx)
        return editable_mol.GetMol()

    # Find dummy atoms by symbol or index
    dummy_atoms = [atom.GetIdx() for atom in mol.GetAtoms() if atom.GetSymbol() == '*']

    hydrogen_atom = Chem.rdchem.Atom(1)
    bond_to_num_Hs_to_add = {
        Chem.rdchem.BondType.SINGLE: 1,
        Chem.rdchem.BondType.DOUBLE: 2,
        Chem.rdchem.BondType.TRIPLE: 3,
    }

    # For each dummy atom:
    # 1. Identify the order of the bond
    # 2. Add that many hydrogens to the other atom the dummy atom is connected to
    # 3. Remove the dummy atom
    # 4. Remove hydrogens (RemoveHs)

    emol = Chem.RWMol(mol)
    # NOTE: Reverse to avoid index shifting issues
    for dummy_atom_idx in reversed(dummy_atoms):

        # Identify the order of the bond
        atom = mol.GetAtomWithIdx(dummy_atom_idx)
        neighbors = atom.GetNeighbors()

        for neighbour_atom in neighbors:
            neighbour_idx = neighbour_atom.GetIdx()
            dummy_atom_bond = emol.GetBondBetweenAtoms(
                dummy_atom_idx, neighbour_idx)
            dummy_atom_bondtype = dummy_atom_bond.GetBondType()
            num_Hs_to_add = bond_to_num_Hs_to_add[dummy_atom_bondtype]

            # Add that many hydrogens to the other atom the dummy atom is connected to
            for _ in range(num_Hs_to_add):
                hydrogen_atom_idx = emol.AddAtom(hydrogen_atom)
                emol.AddBond(neighbour_idx, hydrogen_atom_idx,
                             order=Chem.rdchem.BondType.SINGLE)

        # Finally, remove the dummy atom
        emol.RemoveAtom(dummy_atom_idx)

    # Remove hydrogens
    substruct_mol_wo_attach = Chem.RemoveHs(Chem.Mol(emol))

    # Sanitize and check the molecule to ensure its chemical validity
    Chem.GetSymmSSSR(substruct_mol_wo_attach)
    Chem.SanitizeMol(substruct_mol_wo_attach)

    substruct_mol_wo_attach = Chem.MolFromSmiles(Chem.MolToSmiles(
        substruct_mol_wo_attach))  # verify this can be done...

    dummy_atoms = [atom.GetIdx() for atom in substruct_mol_wo_attach.GetAtoms() if atom.GetSymbol() == '*']
    if dummy_atoms != []:
        smi = Chem.MolToSmiles(mol)
        raise ValueError(f"Dummy atoms still present for: {smi}!")

    if output == "smiles":
        return Chem.MolToSmiles(substruct_mol_wo_attach, canonical=True)
    else:
        return substruct_mol_wo_attach


def get_anonymous_mol(mol: Chem.Mol) -> str:
    """
    Get the anonymous graph representation of a molecule.

    Args:
        mol (rdkit.Chem.rdchem.Mol): The input molecule.

    Returns:
        str: The anonymous graph representation of the molecule.

    Raises:
        ValueError: If there is an error processing the molecule.
    """
    try:
        return rdMolHash.MolHash(mol, rdMolHash.HashFunction.AnonymousGraph)
    except:
        raise ValueError(
            f"Error processing molecule with rdMolHash.HashFunction.AnonymousGraph")


def get_bond_idx(smi: str, bonds_start_end_atoms: List[List[int]]) -> List[int]:
    """
    Get the indices of bonds in a molecule that match the given start and end atom indices.

    Args:
        smi (str): The SMILES representation of the molecule.
        bonds_start_end_atoms (List[List[int]]): A list of lists containing the start and end atom indices of the bonds to search for.

    Returns:
        List[int]: A list of bond indices that match the given start and end atom indices.
    """
    mol = Chem.MolFromSmiles(smi)

    bond_indices = []

    for bond in mol.GetBonds():
        begin_idx = bond.GetBeginAtomIdx()
        end_idx = bond.GetEndAtomIdx()

        if [begin_idx, end_idx] in bonds_start_end_atoms or [end_idx, begin_idx] in bonds_start_end_atoms:
            bond_indices.append(bond.GetIdx())
        elif (begin_idx, end_idx) in bonds_start_end_atoms or (end_idx, begin_idx) in bonds_start_end_atoms:
            bond_indices.append(bond.GetIdx())

    return bond_indices
