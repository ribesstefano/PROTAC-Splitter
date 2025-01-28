import random
import logging
from typing import List, Tuple, Callable, Any, Union, Dict, Optional, Literal
from multiprocessing import Process, Queue

import numpy as np
from tqdm import tqdm
from rdkit import Chem, DataStructs
from rdkit.Chem import (
    rdchem,
    AllChem,
    rdFingerprintGenerator,
    rdMolHash,
)

def GetSubstructMatchesWithTimeout(
    mol: Chem.Mol,
    substruct: Chem.Mol,
    useChirality: bool = True,
    maxMatches: int = 50,
    timeout: int | float = 10,
) -> Optional[List[List[int]]]:
    """ Get substructure matches with a timeout.

    Args:
        mol (Chem.Mol): The molecule to search for substructure matches.
        substruct (Chem.Mol): The substructure to search for in the molecule.
        useChirality (bool, optional): Whether to use chirality in the substructure search. Defaults to True.
        maxMatches (int, optional): The maximum number of matches to return. Defaults to 50.
        timeout (int | float, optional): The timeout in seconds. Defaults to 10.
    
    Returns:
        Optional[List[List[int]]]: A list of lists containing the atom indices of the substructure matches. Returns None if the search times out or failed.
    """
    def worker(q, mol, substruct, useChirality, maxMatches):
        q.put(list(mol.GetSubstructMatches(substruct, useChirality=useChirality, maxMatches=maxMatches)))

    q = Queue()
    p = Process(target=worker, args=(q, mol, substruct, useChirality, maxMatches))
    p.start()
    p.join(timeout)

    if p.is_alive():
        p.terminate()
        p.join()
        return None
    else:
        return q.get()


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
        return None


def get_mol(smiles: str, remove_stereo: bool = False) -> Chem.Mol:
    """
    Get a molecule object from a SMILES string.

    Args:
        smiles (str): The SMILES string representing the molecule.

    Returns:
        Chem.Mol: The molecule object.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is not None and remove_stereo:
        Chem.rdmolops.RemoveStereochemistry(mol)
    return mol


def canonize_smarts(smarts: str) -> str:
    """
    Cleans a SMARTS string by converting it to canonical SMARTS representation.

    NOTE: It might not work for complex patterns: https://github.com/rdkit/rdkit/discussions/6929

    Args:
        smarts (str): The input SMARTS string.

    Returns:
        str: The cleaned SMARTS string.
    """
    mol = Chem.MolFromSmarts(smarts)

    if mol is None:
        return None
    canonical_smarts = Chem.MolToSmarts(Chem.MolFromSmiles(Chem.MolToSmiles(mol), sanitize=False))
    return canonical_smarts


def smiles2mol(smiles: str) -> Chem.Mol:
    """Converts a SMILES string to an RDKit molecule object.

    Args:
        smiles (str): The input SMILES string.

    Returns:
        Chem.Mol: The RDKit molecule object.
    """
    return Chem.MolFromSmiles(smiles)


def mol2smiles(mol: Chem.Mol) -> str:
    """Converts an RDKit molecule object to a SMILES string.

    Args:
        mol (Chem.Mol): The RDKit molecule object.

    Returns:
        str: The SMILES string.
    """
    return Chem.MolToSmiles(mol)


def canonize_smiles(smiles: str) -> str:
    """ Canonizes a SMILES string by converting it to canonical SMILES representation.
    
    Args:
        smiles (str): The input SMILES string.

    Returns:
        str: The canonized SMILES string.
    """
    if smiles is None:
        return None
    try:
        mol = Chem.MolFromSmiles(smiles)
    except Exception as e:
        print(f"Error: {e}")
        return None
    if mol is None:
        return None
    try:
        return Chem.MolToSmiles(mol, canonical=True)
    except:
        return None


def canonize(x: str | Chem.Mol) -> str | Chem.Mol:
    """ Canonizes a SMILES string or RDKit molecule object.

    Args:
        x: The input SMILES string or RDKit molecule object.

    Returns:
        str | Chem.Mol: The canonized SMILES string or RDKit molecule object, according to the input type.
    """
    if x is None:
        return None
    if isinstance(x, str):
        return canonize_smiles(x)
    return Chem.MolFromSmiles(Chem.MolToSmiles(x, canonical=True))


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


def merge_molecules(
        mol1: Chem.rdchem.Mol,
        mol2: Chem.rdchem.Mol,
        atom_idx1: int,
        atom_idx2: int,
        bond_type: Literal['single', 'double', 'triple', 'rand_uniform'] = 'single',
        rand_generator = None,
) -> rdchem.Mol:
    """ Combine two molecules into a single editable molecule.
    
    Args:
        mol1 (rdkit.Chem.rdchem.Mol): The first molecule.
        mol2 (rdkit.Chem.rdchem.Mol): The second molecule.
        atom_idx1 (int): The index of the attachment point in the first molecule.
        atom_idx2 (int): The index of the attachment point in the second molecule.
        bond_type (str): The type of bond to be added between the attachment points. Can be 'single' or 'rand_uniform'.
        rand_generator: A random number generator for 'rand_uniform'. Defaults to None, i.e., standard library random.
    
    Returns:
        rdkit.Chem.rdchem.Mol: The combined molecule.
    """
    # Find neighbors of the attachment points

    neighbor_atom_idx2 = None
    for nbr in mol2.GetAtomWithIdx(atom_idx2).GetNeighbors():
        if nbr.GetAtomicNum() > 1:
            neighbor_atom_idx2 = nbr.GetIdx() + mol1.GetNumAtoms()
            break
    
    # Handle case when linker has no length
    if neighbor_atom_idx2 is None:
        smi_e3_linker_with_e3_attachment = Chem.MolToSmiles(mol1, canonical=True)
        smi_e3_linker_with_poi_attachment = smi_e3_linker_with_e3_attachment.replace("[*:2]", "[*:1]")
        mol_e3_linker_with_poi_attachment = Chem.MolFromSmiles(smi_e3_linker_with_poi_attachment)
        return mol_e3_linker_with_poi_attachment

    for nbr in mol1.GetAtomWithIdx(atom_idx1).GetNeighbors():
        if nbr.GetAtomicNum() > 1:
            neighbor_atom_idx1 = nbr.GetIdx()
            break

    editable_mol = Chem.EditableMol(Chem.CombineMols(mol1, mol2))

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
            if rand_generator is None:
                rand_generator = random
            sampled_bond = rand_generator.choice(possible_bonds)
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

    # # Reassign stereochemistry
    # Chem.AssignStereochemistry(modified_mol, force=True, cleanIt=True)

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

def dummy2query(mol: Chem.Mol) -> Chem.Mol:
    """ Converts dummy atoms to query atoms, so that a molecule with attachment points can be used in HasSubstructMatch.

    Args:
        mol: The molecule to convert.

    Returns:
        The molecule with dummy atoms converted to query atoms
    """
    if mol is None:
        return None
    p = Chem.AdjustQueryParameters.NoAdjustments()
    p.makeDummiesQueries = True
    return Chem.AdjustQueryProperties(mol, p)

def get_substr_match(
        protac_mol: Chem.Mol,
        substr: Chem.Mol,
        max_allowed_fragments: int = 1,
) -> bool:
    """ Check if a molecule contains a substructure match with a given molecule.
    Compared to RDKit HasSubstructMatch, this function also checks the number of fragments when replacing the substr in the PROTAC.
    
    Args:
        protac_mol (Chem.Mol): The PROTAC molecule.
        substr (Chem.Mol): The substructure molecule.
        max_allowed_fragments (int, optional): The maximum number of fragments allowed when replacing the substr in the PROTAC. Defaults to 1. Example when equal to 1: if removing the warhead, a single fragment should remain.

    Returns:
        bool: True if the PROTAC contains a substructure match with the given molecule and the fragments count is equal, False otherwise.
    """
    # Count the number of fragments when replacing the substr in the PROTAC
    fragments = Chem.ReplaceCore(protac_mol, dummy2query(substr), useChirality=True)
    if fragments is None:
        return False
    try:
        fragments = Chem.GetMolFrags(fragments)
    except Exception as e:
        print(e)
        return False
    return len(fragments) == max_allowed_fragments


def remove_attach_atom(mol: Chem.Mol, attach_id: int, sanitize: bool = False) -> Chem.Mol:
    """ Removes the atom with the specified attachment id from the molecule.

    Example:
    
    >>> remove_attach_atom(Chem.MolFromSmiles('CC[*:1]'), 1)
    CC

    There are no checks on the molecule, so it is assumed it is not None.

    Args:
        mol (Chem.Mol): The molecule.
        attach_id (int): The attachment id of the atom to remove.
        sanitize (bool, optional): Whether to sanitize the molecule after removing the atom. When used in `fix_prediction` function, it is used to "remove" substructures, so there is no need to have them sanitized. Default: False.

    Returns:
        (Chem.Mol) The molecule with the atom removed.
    """
    atoms_to_remove = []
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() == 0:  # Dummy atom
            map_num = atom.GetAtomMapNum()
            if map_num == attach_id:  # Targeting only [*:attach_id]
                atoms_to_remove.append(atom.GetIdx())

    # Remove atoms using an EditableMol
    editable_mol = Chem.EditableMol(mol)
    for idx in sorted(atoms_to_remove, reverse=True):  # Remove from highest index to avoid shifting
        editable_mol.RemoveAtom(idx)

    # Convert back to a molecule
    new_mol = editable_mol.GetMol()
    if sanitize:
        Chem.SanitizeMol(new_mol)
    return new_mol


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
