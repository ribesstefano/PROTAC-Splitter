import logging
from typing import List, Union, Optional, Literal
from multiprocessing import Process, Queue
from hashlib import sha256

from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator


def GetSubstructMatchesWorker(q, mol, substruct, useChirality, maxMatches):
    q.put(list(mol.GetSubstructMatches(substruct, useChirality=useChirality, maxMatches=maxMatches)))


def GetSubstructMatchesWithTimeout(
    mol: Chem.Mol,
    substruct: Chem.Mol,
    useChirality: bool = True,
    maxMatches: int = 50,
    timeout: Union[int, float] = 10,
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
    q = Queue()
    p = Process(
        target=GetSubstructMatchesWorker,
        args=(q, mol, substruct, useChirality, maxMatches),
    )
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


def canonize(x: Union[str, Chem.Mol]) -> Union[str, Chem.Mol]:
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


def remove_dummy_atoms(mol: Union[str, Chem.Mol], canonical=True) -> Union[str, Chem.Mol]:
    """
    Removes all dummy atoms (attachment points) from a molecule.
    
    Args:
        mol: RDKit Mol object with dummy atoms.

    Returns:
        A new RDKit Mol object without dummy atoms.
    """
    return_smiles = False
    if isinstance(mol, str):
        return_smiles = True
        mol = Chem.MolFromSmiles(mol)
    
    if mol is None:
        return None
    
    # Remove all dummy atoms with a query
    mol_no_dummy = Chem.DeleteSubstructs(mol, Chem.MolFromSmarts('[#0]'))

    if mol_no_dummy is None:
        # --------------------------------------------------------------------------
        # Other approach: editing molecule and removing dummy atoms
        # --------------------------------------------------------------------------
        # Create an editable molecule to remove atoms
        editable_mol = Chem.EditableMol(mol)
        
        # List of atoms to remove (dummy atoms have atomic number 0)
        dummy_atoms = [atom.GetIdx() for atom in mol.GetAtoms() if atom.GetAtomicNum() == 0]
        
        # Remove dummy atoms
        for atom_idx in sorted(dummy_atoms, reverse=True):  # Remove from the highest index to avoid index shifts
            editable_mol.RemoveAtom(atom_idx)
        
        if editable_mol is None:
            return None

        # Return the modified molecule
        if return_smiles:
            return Chem.MolToSmiles(editable_mol.GetMol())
        editable_mol = editable_mol.GetMol()
        editable_mol.UpdatePropertyCache()
        return editable_mol
        # --------------------------------------------------------------------------

    # Return the modified molecule
    if return_smiles:
        return Chem.MolToSmiles(mol_no_dummy, canonical=canonical)
    return mol_no_dummy


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
        replace: Literal['core', 'sidechains'] = 'core',
        useChirality: bool = True,
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
    if replace == 'core':
        fragments = Chem.ReplaceCore(protac_mol, dummy2query(substr), useChirality=useChirality)
    elif replace == 'sidechains':
        fragments = Chem.ReplaceSidechains(protac_mol, dummy2query(substr), useChirality=useChirality)
    else:
        raise ValueError(f"replace argument should be either 'core' or 'sidechains', provided: {replace}")
    # Check if the number of fragments is equal to the max allowed fragments
    if fragments is None:
        return False
    try:
        fragments = Chem.GetMolFrags(fragments, sanitizeFrags=False)
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


def get_mol_id(smiles: str) -> str | None:
    """ Get the Hash of a given SMILES string.
    
    Args:
        smiles (str): The SMILES string to hash.

    Returns:
        str | None: The Hash of the SMILES string. None if the function failed.
    """
    if smiles is None:
        print("Error: SMILES is None.")
        return None
    try:
        mol = Chem.MolFromSmiles(smiles)
    except Exception as e:
        print(f"Error: {e}")
        print(f"SMILES: {smiles}")
        return None
    if mol is None:
        return None
    # Remove stereochemistry from the molecule
    try:
        Chem.RemoveStereochemistry(mol)
    except:
        return None
    # Get the InChIKey for the molecule
    inchi_key = Chem.MolToInchiKey(mol)
    smiles = Chem.MolToSmiles(mol, canonical=True)
    return sha256((inchi_key + smiles).encode()).hexdigest()


def get_atom_idx_at_attachment(
        protac: Chem.Mol,
        substruct: Chem.Mol,
        linker: Optional[Chem.Mol] = None,
        timeout: Optional[Union[int, float]] = None,
        return_dict: bool = False,
        verbose: int = 0,
) -> List[int]:
    """ Get the atom index of the attachment point of a substructure in the PROTAC molecule.

    Args:
        protac: The PROTAC molecule.
        substruct: The substructure of the PROTAC that contains the attachment point, e.g., the POI or E3 ligase.
        linker: The linker molecule.
        verbose: Verbosity level.
    
    Returns:
        List[int]: The two atom indices at the attachment point.
    """
    if linker is None:
        # Get the "other" substructure, i.e., replace side chain of PROTAC using the substruct
        linker = Chem.DeleteSubstructs(protac, remove_dummy_atoms(substruct), useChirality=True)
        if timeout is None:
            timeout = 60
            logging.warning(f'No timeout set when linker is not provided, using default value of {timeout} seconds.')

    substruct_match = set(protac.GetSubstructMatch(dummy2query(substruct), useChirality=True))
    if verbose:
        print(f'Substruct match: {substruct_match}')
    
    linker_no_dummy = remove_dummy_atoms(linker)
    if verbose:
        print(f'Linker without dummy atoms found.')

    max_matches = 2
    linker_match = set()
    shared_atoms = set()

    # NOTE: The following is a hacky way to speed up the search for linker
    # matches. In fact, the linker can be quite short, so it might match in
    # multiple places of the PROTAC molecule.
    # If the number of max matches in GetSubstructMatches is low, then the
    # search tends to be faster, but imprecise. However, we are interested in
    # the interesection of the matches, so we can progressively increase the
    # number of max matches until we find a single atom in common.
    while len(shared_atoms) != 1 and max_matches <= 50:
        if timeout is None:
            linker_matches = list(protac.GetSubstructMatches(linker_no_dummy, useChirality=True, maxMatches=max_matches))
        else:
            linker_matches = GetSubstructMatchesWithTimeout(protac, linker_no_dummy, useChirality=True, maxMatches=max_matches, timeout=timeout)
        if verbose:
            print(f'Linker matches: {linker_matches}')

        if not linker_matches:
            # return None
            linker_match = set()
            shared_atoms = set()
            max_matches += 1
            continue

        for match in linker_matches:
            shared_atoms = set(match) & set(substruct_match)
            linker_match = match
            if len(shared_atoms) == 1:
                if verbose:
                    print(f'Shared atoms: {list(shared_atoms)}')
                break

        if len(shared_atoms) != 1:
            linker_match = set()
            shared_atoms = set()
            max_matches += 1

    if not shared_atoms:
        if verbose:
            print('No shared atoms found.')
        return None

    attachment_idx = list(shared_atoms)
    attachments = {'substruct': attachment_idx[0]}

    # Get the other atom at the attachment point that is NOT in the linker
    for neighbor in protac.GetAtomWithIdx(attachment_idx[0]).GetNeighbors():
        if neighbor.GetIdx() not in linker_match:
            attachment_idx.append(neighbor.GetIdx())
            attachments['linker'] = neighbor.GetIdx()
            break
    
    if return_dict:
        return attachments
    return attachment_idx


