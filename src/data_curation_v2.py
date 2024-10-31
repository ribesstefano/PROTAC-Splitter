# %% [markdown]
# # Data Curation for PROTAC-Splitter

# %% [markdown]
# ## Setup and Imports

# %%
import os
import sys
import re
from typing import Optional, List, ClassVar, Any, Tuple, Dict
from collections import Counter

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs, MACCSkeys, rdFMCS, Draw
from rdkit import RDLogger
from rdkit import rdBase

if 'ipykernel' in sys.modules:
    from tqdm.auto import tqdm  # for notebooks
    from IPython.display import display
else:
    from tqdm import tqdm

def safe_display(*args):
    """Displays content only if running in a Jupyter notebook."""
    if 'ipykernel' in sys.modules:
        display(*args)
    else:
        print(*args)

# Disable the RDKit warnings that pop up when RDKit fails to create molecules
RDLogger.DisableLog("rdApp.*")
blocker = rdBase.BlockLogs()

data_dir = os.path.join(os.getcwd(), '..', 'data')

# %%
# Enable debug logging
import logging

logging.basicConfig(level=logging.INFO)

# %% [markdown]
# ## Utility Functions

# %% [markdown]
# When splitting a PROTAC into its substructures, we are interested in labelling the "directionality" of the sub-structures, so that we can easily tell which substructure is the "warhead" (binding to the POI) and which is the "E3 ligand".
# 
# To this end, we now define and fix constant the IDs of the two attachment points:

# %%
POI_ATTACHMENT_ID = 1
E3_ATTACHMENT_ID = 2

# %% [markdown]
# Utility functions:
# 
# - `clean_smarts`: UNUSED.
# - `canonicalize_smiles`: Canonize SMILES strings.
# - `check_reassembly`: Check if the reassembly of the PROTAC is successful.
# - `dummy2query`: This function is useful for getting substructure queries, _e.g._, matches, when dealing with SMILES with dummy atoms (which normally would match any atom instead).
# - `remove_dummy_atoms`: Remove dummy atoms from a SMILES string. UNUSED.

# %%
# NOTE: It might not work for complex patterns: https://github.com/rdkit/rdkit/discussions/6929
def clean_smarts(smarts: str) -> str:
    """
    Cleans a SMARTS string by converting it to canonical SMARTS representation.

    Args:
        smarts (str): The input SMARTS string.

    Returns:
        str: The cleaned SMARTS string.
    """
    mol = Chem.MolFromSmarts(smarts)

    if mol is not None:
        canonical_smarts = Chem.MolToSmarts(Chem.MolFromSmiles(Chem.MolToSmiles(mol)))
        return canonical_smarts
    else:
        # Handle the case when the input SMARTS is invalid
        return np.nan

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
    mol = Chem.MolFromSmiles(smiles)
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
    if isinstance(x, str):
        return canonize_smiles(x)
    return Chem.MolFromSmiles(Chem.MolToSmiles(x, canonical=True))

def check_reassembly(protac_smiles: str, substructs_smiles: str, stats=None, verbose=0) -> bool:
    """Check if the reassembled PROTAC matches the original PROTAC SMILES.

    Args:
        protac_smiles (str): The original PROTAC SMILES.
        substructs_smiles (str): The substructure SMILES.

    Returns:
        bool: True if the reassembled PROTAC matches the original PROTAC SMILES, False otherwise.
    """
    substructs_mol = Chem.MolFromSmiles(canonize_smiles(substructs_smiles), sanitize=True)
    if substructs_mol is None:
        return False
    try:
        reassembled_mol = Chem.molzip(substructs_mol)
    except:
        if stats is not None:
            stats['molzip failed'] += 1
        if verbose:
            print('ERROR: molzip failed')
        return False
    reassembled_smiles = Chem.MolToSmiles(reassembled_mol, canonical=True)
    if verbose and reassembled_smiles != protac_smiles:
        print('ERROR. Not equal:')
        print('\tOriginal:   ', protac_smiles)
        print('\tReassembled:', reassembled_smiles)
    return reassembled_smiles == protac_smiles

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

def remove_dummy_atoms(mol: str | Chem.Mol, canonical=True) -> str | Chem.Mol:
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
    mol = Chem.DeleteSubstructs(mol, Chem.MolFromSmarts('[#0]'))

    # Return the modified molecule
    if return_smiles:
        return Chem.MolToSmiles(mol, canonical=canonical)
    return mol

    # # Create an editable molecule to remove atoms
    # editable_mol = Chem.EditableMol(mol)
    
    # # List of atoms to remove (dummy atoms have atomic number 0)
    # dummy_atoms = [atom.GetIdx() for atom in mol.GetAtoms() if atom.GetAtomicNum() == 0]
    
    # # Remove dummy atoms
    # for atom_idx in sorted(dummy_atoms, reverse=True):  # Remove from the highest index to avoid index shifts
    #     editable_mol.RemoveAtom(atom_idx)

    # # Return the modified molecule
    # if return_smiles:
    #     return Chem.MolToSmiles(editable_mol.GetMol())
    # return editable_mol.GetMol()

# %% [markdown]
# ### Split PROTAC when knowing mapped linker

# %% [markdown]
# The function `get_substructs_from_mapped_linker` will return the substructures given a linker with directionality, _i.e._, with the two attachment points mapped:

# %%
def get_substructs_from_mapped_linker(
        protac_smiles: str,
        linker_smiles: str,
        verbose: int = 0,
        e3_attachment_id: int = E3_ATTACHMENT_ID,
        poi_attachment_id: int = POI_ATTACHMENT_ID
) -> Dict[str, str]:
    """ Get the substructures of a PROTAC molecule from a mapped linker SMILES.
    
    Args:
        protac_smiles: The SMILES of the PROTAC molecule.
        linker_smiles: The SMILES of the linker molecule. Must have attachment points.
        verbose: Verbosity level.
    
    Returns:
        A dictionary with the substructure names as keys ('e3', 'linker', and 'poi') and their SMILES as values. None if the matching fails.
    """
    protac_smiles = canonize_smiles(protac_smiles)
    linker_smiles = canonize_smiles(linker_smiles)

    protac_mol = Chem.MolFromSmiles(protac_smiles)
    linker_mol = Chem.MolFromSmiles(linker_smiles)

    # Check if the linker is a substructure of the PROTAC
    if not protac_mol.HasSubstructMatch(dummy2query(linker_mol), useChirality=True):
        return None

    # Split the big molecule into the two fragments
    frags = Chem.ReplaceCore(protac_mol, dummy2query(linker_mol), labelByIndex=True, replaceDummies=False)
    if frags is None:
        return None
    try:
        groups = Chem.GetMolFrags(frags, asMols=True, sanitizeFrags=True)
    except Exception as e:
        print(e)
        return None

    if verbose:
        safe_display(protac_mol)
        safe_display(linker_mol)

    linker_idx2map = {}
    for atom in linker_mol.GetAtoms():
        if atom.GetAtomicNum() == 0:
            linker_idx2map[atom.GetIdx()] = atom.GetAtomMapNum()
    if verbose:
        print(f'linker indexes: {linker_idx2map}')
        print('-' * 80)

    substructs = {'linker': linker_smiles}

    for i, side_mol in enumerate(groups):

        side_smiles = Chem.MolToSmiles(side_mol, canonical=True)

        # Use a regex to get the number in the pattern, e.g., [9*], in the SMILES
        attachment_point = re.findall(r'\[(\d+)\*\]', side_smiles)
        if attachment_point:
            attachment_point = int(attachment_point[0])
        else:
            attachment_point = None
        
        if verbose:
            print(f'Side {i + 1} SMILES: {side_smiles}')
            print(f'Attachment point: {attachment_point}')
            safe_display(side_mol)

        # Get the map from the linker
        linker_attachment_point = linker_idx2map.get(attachment_point)

        # Modify the SMILES to include the map number
        if linker_attachment_point is not None:
            side_smiles = re.sub(r'\[(\d+)\*\]', f'[*:{linker_attachment_point}]', side_smiles)
            if f'[*:{e3_attachment_id}]' in side_smiles:
                substructs['e3'] = canonize_smiles(side_smiles)
            elif f'[*:{poi_attachment_id}]' in side_smiles:
                substructs['poi'] = canonize_smiles(side_smiles)

            if verbose:
                print(f'Modified SMILES: {side_smiles}')
                safe_display(Chem.MolFromSmiles(side_smiles))

    # Check that the reassembled PROTAC matches the original PROTAC
    if not check_reassembly(protac_smiles, '.'.join(substructs.values())):
        return None

    return substructs

# %% [markdown]
# ### Split PROTAC when knowing unmapped POI and unmapped E3
# 
# The function `get_substructs_from_unmapped_poi_e3` will return the substructures given a PROTAC and its unmapped POI and E3 ligand substructures, _i.e._, they do not need to have the attachment points in their SMILES strings:

# %%
def get_attachment_bonds(mol, match_atoms) -> List[int]:
    """ Get the bonds to break to separate the substructure from the PROTAC or R-groups molecule.
    
    Args:
        mol: The molecule to break.
        match_atoms: The atoms matched in the molecule, from the GetSubstructMatch function.
    
    Returns:
        List[int]: The bond indices to break.
    """
    bonds_to_break = []
    for idx in match_atoms:
        atom = mol.GetAtomWithIdx(idx)
        # Skip non-heavy atoms
        if atom.GetAtomicNum() == 1:
            continue
        for bond in atom.GetBonds():
            neighbor_idx = bond.GetOtherAtomIdx(idx)
            # Skip if the neighbor atom if non-heavy
            if mol.GetAtomWithIdx(neighbor_idx).GetAtomicNum() == 1:
                continue
            if neighbor_idx not in match_atoms:
                bonds_to_break.append(bond.GetIdx())
                # If more than one bond is found, e.g., if the substructure is
                # connected to the PROTAC/R-groups in multiple places like in a
                # ring, reset list of bonds and go to the next atom.
                if len(bonds_to_break) > 1:
                    bonds_to_break = []
                    break
    return bonds_to_break

def get_substructs_from_unmapped_e3_poi(
        protac_smiles: str,
        mol_protac: Chem.Mol,
        mol_poi: Chem.Mol,
        mol_e3: Chem.Mol,
        poi_attachment_id: int = POI_ATTACHMENT_ID,
        e3_attachment_id: int = E3_ATTACHMENT_ID,
        verbose: int = 0,
        stats: Counter = None,
) -> Dict[str, str] | None:
    """ Get the matches of the POI, E3, and linker in the PROTAC molecule.
    
    Args:
        mol_protac: The PROTAC molecule.
        mol_poi: The POI ligand molecule. Must NOT contain the attachment point.
        mol_e3: The E3 binder molecule. Must NOT contain the attachment point.
        verbose: The verbosity level.

    Returns:
        Dict: The matches of the POI, E3, and linker in the PROTAC molecule. None if no match is found.
    """
    if verbose:
        safe_display(mol_protac)

    poi_match = mol_protac.GetSubstructMatch(mol_poi, useChirality=True)

    # Get bonds to break to separate the POI ligand
    bonds_to_break_poi = get_attachment_bonds(mol_protac, poi_match)

    # Return if no bonds are found
    if len(bonds_to_break_poi) != 1:
        if stats is not None:
            stats['multiple POI attachment bonds'] += 1
        if verbose:
            print('ERROR: Multiple POI attachment bonds')
        return None

    # Break the bonds to isolate the POI ligand
    frag_mol_poi = Chem.FragmentOnBonds(mol_protac, bonds_to_break_poi, addDummies=True, dummyLabels=[(poi_attachment_id, poi_attachment_id)])

    # Get the fragments resulting from bond breaking
    try:
        frags = Chem.GetMolFrags(frag_mol_poi, asMols=True, sanitizeFrags=True)
    except Exception as e:
        print(e)
        return None

    # Identify the POI ligand fragment
    poi_fragment = None
    for frag in frags:
        if frag.HasSubstructMatch(mol_poi):
            poi_fragment = frag
            break
    if poi_fragment is None:
        if stats is not None:
            stats['POI fragment not found'] += 1
        if verbose:
            print('ERROR: POI fragment not found')
        return None

    # Combine the remaining fragments to get the R-groups
    # TODO: Check that the length of frags is 1, otherwise, there are multiple fragments
    r_group_mol = [frag for frag in frags if frag != poi_fragment]
    if len(r_group_mol) != 1:
        if stats is not None:
            stats['multiple POI fragments'] += 1
        if verbose:
            for frag in frags:
                safe_display(frag)
            print('ERROR: Multiple POI fragments')
        return None
    r_group_mol = r_group_mol[0]

    if verbose:
        print('POI:', Chem.MolToSmiles(poi_fragment, canonical=True))
        safe_display(poi_fragment)

    e3_match = r_group_mol.GetSubstructMatch(mol_e3, useChirality=True)

    # Get bonds to break to isolate the E3 binder
    bonds_to_break_e3 = get_attachment_bonds(r_group_mol, e3_match)

    # Return if no bonds are found
    if len(bonds_to_break_e3) != 1:
        if stats is not None:
            stats['multiple E3 attachment bonds'] += 1
        if verbose:
            safe_display(r_group_mol)
            print('ERROR: Multiple E3 attachment bonds')
        return None

    # Break the bonds to isolate the E3 binder
    frag_mol_e3 = Chem.FragmentOnBonds(r_group_mol, bonds_to_break_e3, addDummies=True, dummyLabels=[(e3_attachment_id, e3_attachment_id)])

    # Get fragments after breaking bonds in R-groups
    try:
        frags = Chem.GetMolFrags(frag_mol_e3, asMols=True, sanitizeFrags=True)
    except Exception as e:
        print(e)
        return None

    # Identify the E3 binder fragment
    e3_fragment = None
    for frag in frags:
        if frag.HasSubstructMatch(mol_e3):
            e3_fragment = frag
            break
    if e3_fragment is None:
        if stats is not None:
            stats['E3 fragment not found'] += 1
        if verbose:
            print('ERROR: E3 fragment not found')
        return None

    if verbose:
        print('E3:', Chem.MolToSmiles(e3_fragment, canonical=True))
        safe_display(e3_fragment)

    # The remaining fragment is the linker
    # TODO: Check that the length of frags is 1, otherwise, there are multiple fragments
    linker_mol = [frag for frag in frags if frag != e3_fragment]
    if len(linker_mol) != 1:
        if stats is not None:
            stats['multiple E3 fragments'] += 1
        if verbose:
            for frag in frags:
                safe_display(frag)
            print('ERROR: Multiple E3 fragments')
        return None
    linker_mol = linker_mol[0]

    poi_smiles = Chem.MolToSmiles(poi_fragment, canonical=True).replace(f'[{poi_attachment_id}*]', f'[*:{poi_attachment_id}]')
    e3_smiles = Chem.MolToSmiles(e3_fragment, canonical=True).replace(f'[{e3_attachment_id}*]', f'[*:{e3_attachment_id}]')
    linker_smiles = Chem.MolToSmiles(linker_mol, canonical=True).replace(f'[{poi_attachment_id}*]', f'[*:{poi_attachment_id}]').replace(f'[{e3_attachment_id}*]', f'[*:{e3_attachment_id}]')

    if verbose:
        print('Linker:', Chem.MolToSmiles(linker_mol, canonical=True))
        safe_display(linker_mol)

    if check_reassembly(protac_smiles, f'{e3_smiles}.{linker_smiles}.{poi_smiles}', stats, verbose):
        return {'poi': poi_smiles, 'e3': e3_smiles, 'linker': linker_smiles}
    
    if stats is not None:
        stats['reassembling failed'] += 1
    if verbose:
        print('ERROR: Reassembling failed')
    return None

# %% [markdown]
# ### Get matching substructure from a non-matching one
# 
# Sometimes the substructure we have is not a _perfect_ substructure of the PROTAC, _i.e._, it will generate more than two fragments when trying to replace the PROTAC core with it. In this case, the function `get_substructure_from_non_perfect_match` will perform the following steps:
# 
# 1. Get the largest fragment by trying to replace the PROTAC core with the substructure. This largest fragment will be the other substructure plus the linker.
# 2. We can now remove the largest fragment from the PROTAC to get the "original" substructure without the smaller dangling fragments.

# %%
def get_substructure_from_non_perfect_match(
        protac_mol: Chem.Mol,
        substruct_mol: Chem.Mol,
        attachment_id: int,
        verbose: int = 0,
) -> Tuple[Chem.Mol, str]:
    """ Extract the correct substructure from a PROTAC molecule, given the SMILES of a wrong substructure resulting in many fragments and matches.

    Args:
        protac_mol (Chem.Mol): The PROTAC molecule.
        substruct_smiles (Chem.Mol): The molecule of the wrong substructure.
        attachment_id (int): The attachment ID.

    Returns:
        Chem.Mol: The extracted substructure molecule.
        str: The extracted substructure SMILES. If failing, it will return None, None.
    """
    # Remove the substructure, even if there are "dangling" fragments, to obtain: PROTAC - substruct = (POI + Linker) + remainders
    linker_and_other_mol = Chem.DeleteSubstructs(protac_mol, substruct_mol, useChirality=True)

    # Get the largest fragment, i.e., the PROTAC - substruct = POI + Linker
    try:
        largest_fragment = max(Chem.GetMolFrags(linker_and_other_mol, asMols=True), key=lambda x: x.GetNumAtoms())
    except Exception as e:
        if verbose:
            print(e)
        return None, None

    # Get the match of the largest fragment in the PROTAC molecule
    largest_match = protac_mol.GetSubstructMatch(largest_fragment, useChirality=True)

    # Get bonds to break to isolate the substructure, i.e., the opposite of the POI + Linker
    bonds_to_break = get_attachment_bonds(protac_mol, largest_match)

    if len(bonds_to_break) != 1:
        if verbose:
            print(f'ERROR. The bond to break is not a single one: {bonds_to_break}')
        return None, None

    # Break the bonds to isolate the substructure
    frag_mol_substruct = Chem.FragmentOnBonds(protac_mol, bonds_to_break, addDummies=True, dummyLabels=[(attachment_id, attachment_id)])

    # Get fragments after breaking bonds, i.e., the POI + Linker and the substructure without "remainders"
    try:
        frags = Chem.GetMolFrags(frag_mol_substruct, asMols=True, sanitizeFrags=True)
    except Exception as e:
        if verbose:
            print(e)
        return None, None

    # Get the smallest between the substructure and the POI+Linker fragments
    substruct_mol = min(frags, key=lambda x: x.GetNumAtoms())

    # Check that the substructure matches in the PROTAC molecule
    if not protac_mol.HasSubstructMatch(substruct_mol, useChirality=True):
        if verbose:
            print('ERROR. Substructure does not match in PROTAC molecule:')
            safe_display(protac_mol)
            safe_display(substruct_mol)
        return None, None

    substruct_smiles = Chem.MolToSmiles(substruct_mol, canonical=True).replace(f'[{attachment_id}*]', f'[*:{attachment_id}]')
    return substruct_mol, substruct_smiles


# %% [markdown]
# ## Fix Functional Groups

# %%
def display_protac_substructures(
        protac_smiles: str,
        poi_smiles: str,
        linker_smiles: str,
        e3_smiles: str,
        compound_id: Optional[int] = None,
        print_smiles: bool = False,
):
    protac_mol = Chem.MolFromSmiles(canonize_smiles(protac_smiles))
    e3_mol = Chem.MolFromSmiles(canonize_smiles(e3_smiles))
    poi_mol = Chem.MolFromSmiles(canonize_smiles(poi_smiles))
    linker_mol = Chem.MolFromSmiles(canonize_smiles(linker_smiles))

    legends = [
        f'ID: {compound_id} - {protac_smiles}',
        f'POI - {poi_smiles}',
        f'{linker_smiles}',
        f'E3 - {e3_smiles}',
    ]
    img = Draw.MolsToGridImage([protac_mol, poi_mol, linker_mol, e3_mol], molsPerRow=4, subImgSize=(1000, 500), legends=legends)

    if print_smiles:
        print(f'ID: {compound_id}')
        print(f'PROTAC: {protac_smiles}')
        print(f'POI: {poi_smiles}')
        print(f'Linker: {linker_smiles}')
        print(f'E3: {e3_smiles}')
    safe_display(img)

# %% [markdown]
# ### Amide Groups

# %% [markdown]
# ```python
# # Check if the amide bond (N-C=O) is in the substructure
# if "N-C(=O)" in substruct: 
#     if neighbor("N-C(=O)") == "[*:substruct]": 
#         # If the neighboring atom of the amide bond is a dummy atom, i.e., attachment point
#         mark_protac_as_wrong("[PROTAC]")
# 
#         # Identify the bond to split, i.e., the nitrogen-carbon bond, and split
#         "[*:substruct]-[<optional neighboring atom>]-N-[*:tmp]", "[*:tmp]-C(=O)-[rest of the PROTAC]" = split_PROTAC_at("N-C")
# 
#         "[Linker]-N-[*:tmp]" = join("[Linker]-[*:substruct]", "[*:substruct]-N-[*:tmp]")
#         
#         rename_attachment_point("[*:tmp]-C(=O)-[rest of the PROTAC]")
#         rename_attachment_point("[Linker]-N-[*:tmp]")
#     
#     elif neighbor(neighbor("N-C(=O)")) == "[*:substruct]":
#         # If the second neighbor of athe amide bond is a dummy atom, i.e., attachment point
#         mark_protac_as_wrong("[PROTAC]")
# 
#         # Do as above
#         # Identify the bond to split, i.e., the nitrogen-carbon bond, and split
#         "[*:substruct]-N-[*:tmp]", "[*:tmp]-C(=O)-[rest of the PROTAC]" = split_PROTAC_at("N-C")
# 
#         "[Linker]-N-[*:tmp]" = join("[Linker]-[*:substruct]", "[*:substruct]-N-[*:tmp]")
#         
#         rename_attachment_point("[*:tmp]-C(=O)-[rest of the PROTAC]")
#         rename_attachment_point("[Linker]-N-[*:tmp]")
# ```

# %%
def adjust_amide_bond(substruct: Chem.Mol, linker: Chem.Mol, substruct_attachment_id: int, verbose: int = 0) -> Tuple[Chem.Mol, Chem.Mol]:
    """
    Adjust the amide bond between the substruct and linker substructure.
    Handles the case when neighboring atoms of the amide bond are dummy atoms, which represent attachment points.
    
    Args:
        substruct: The substructure of the substruct (protein of interest) that contains the amide bond.
        linker: The linker molecule that connects substruct to the E3 ligase.
        substruct_attachment_id: The attachment point ID in the substruct substructure. E.g., 1 for the POI, as in "[*:1]".
    
    Returns:
        Tuple[Chem.Mol, Chem.Mol]: The adjusted substruct and linker molecules, in that order.
    """
    # Convert dummy atoms in substruct to query atoms for substructure search
    query_substruct = dummy2query(substruct)

    # Identify amide bond (N-C=O) in substruct substructure
    amide_pattern = Chem.MolFromSmarts("[NX3][CX3](=[OX1])")
    amide_matches = query_substruct.GetSubstructMatches(amide_pattern)

    if not amide_matches:
        return substruct, linker  # No amide bond found, return the original substruct
    
    side_atom = None
    nitrogen_idx_found, carbonyl_idx_found = None, None
    for match in amide_matches:
        nitrogen_idx, carbonyl_idx = match[0], match[1]
        nitrogen_atom = query_substruct.GetAtomWithIdx(nitrogen_idx)
        carbonyl_atom = query_substruct.GetAtomWithIdx(carbonyl_idx)

        for amide_atom in [nitrogen_atom, carbonyl_atom]:
            # Check neighboring atoms for attachment points
            # NOTE: The dummy atom representing an attachment point have atomic number 0
            for neighbor in amide_atom.GetNeighbors():
                if neighbor.GetAtomicNum() == 0:
                    nitrogen_idx_found = nitrogen_idx
                    carbonyl_idx_found = carbonyl_idx
                    side_atom = "N" if amide_atom == nitrogen_atom else "C"
                    break

            # If previous search failed, check the neighbors of the neighboring
            # atoms (second-order neighbors)
            if nitrogen_idx_found is None or carbonyl_idx_found is None:
                for neighbor in amide_atom.GetNeighbors():
                    for second_neighbor in neighbor.GetNeighbors():
                        if second_neighbor.GetIdx() == carbonyl_idx or second_neighbor.GetIdx() == nitrogen_idx:
                            continue  # Skip the opposite atom from the amide bond

                        if second_neighbor.GetAtomicNum() == 0:
                            nitrogen_idx_found = nitrogen_idx
                            carbonyl_idx_found = carbonyl_idx
                            side_atom = "N" if amide_atom == nitrogen_atom else "C"
                            break
            else:
                break

    if nitrogen_idx_found is None or carbonyl_idx_found is None or side_atom is None:
        return substruct, linker

    # Split the amide bond and adjust
    dummy_labels = [(3, 3)] # The E3 and substruct will have 1 and 2, so we need a third one
    amid_bond_idx = query_substruct.GetBondBetweenAtoms(nitrogen_idx_found, carbonyl_idx_found).GetIdx()
    fragments = Chem.FragmentOnBonds(query_substruct, [amid_bond_idx], addDummies=True, dummyLabels=dummy_labels)

    # Get the fragments resulting from bond breaking
    try:
        mol_frags = Chem.GetMolFrags(fragments, asMols=True, sanitizeFrags=True)
    except Exception as e:
        print(e)
        return substruct, linker

    # Identify the "[*:substruct][<optional neighboring atom>]N[3*]" fragment, the other one will be the "truncated" substruct
    amide_fragment_pattern = Chem.MolFromSmarts(f"[*:{substruct_attachment_id}][{side_atom}][3*]")
    amide_fragment = None
    substruct_fixed = None

    for frag in mol_frags:
        if frag.HasSubstructMatch(dummy2query(amide_fragment_pattern)):
            amide_fragment = frag
        else:
            substruct_fixed = frag
    
    if amide_fragment is None or substruct_fixed is None:
        return substruct, linker

    # Rename the "[3*]" attachment point on the amide fragment to "[*:3]"
    amide_fragment_smiles = Chem.MolToSmiles(amide_fragment, canonical=True)
    amide_fragment_smiles = amide_fragment_smiles.replace('[3*]', f'[*:3]')
    amide_fragment = Chem.MolFromSmiles(amide_fragment_smiles)

    # Use molzip to join the linker and the fragment at the original attachment point
    linker_fixed = Chem.molzip(linker, amide_fragment)

    # Rename the "[*:3]" attachment point back to the original attachment point on the linker
    linker_fixed_smiles = Chem.MolToSmiles(linker_fixed, canonical=True)
    linker_fixed_smiles = linker_fixed_smiles.replace('[*:3]', f'[*:{substruct_attachment_id}]')
    linker_fixed = Chem.MolFromSmiles(linker_fixed_smiles)

    # Rename the "[3*]" attachment point back to the original attachment point on the substruct
    substruct_fixed_smiles = Chem.MolToSmiles(substruct_fixed, canonical=True)
    substruct_fixed_smiles = substruct_fixed_smiles.replace('[3*]', f'[*:{substruct_attachment_id}]')
    substruct_fixed = Chem.MolFromSmiles(substruct_fixed_smiles)

    return substruct_fixed, linker_fixed



# %% [markdown]
# ### Ester Group

# %%
def adjust_ester_bond(substruct: Chem.Mol, linker: Chem.Mol, substruct_attachment_id: int, verbose: int = 0) -> Tuple[Chem.Mol, Chem.Mol]:
    """
    Adjust the amide bond between the substruct and linker substructure.
    Handles the case when neighboring atoms of the amide bond are dummy atoms, which represent attachment points.
    
    Args:
        substruct: The substructure of the substruct (protein of interest) that contains the amide bond.
        linker: The linker molecule that connects substruct to the E3 ligase.
        substruct_attachment_id: The attachment point ID in the substruct substructure. E.g., 1 for the POI, as in "[*:1]".
    
    Returns:
        Tuple[Chem.Mol, Chem.Mol]: The adjusted substruct and linker molecules, in that order.
    """
    # Convert dummy atoms in substruct to query atoms for substructure search
    query_substruct = dummy2query(substruct)

    # Identify ester group (COOR) in substruct substructure
    ester_pattern = Chem.MolFromSmarts("[OX2][CX3](=[OX1])")

    ester_matches = query_substruct.GetSubstructMatches(ester_pattern)

    if not ester_matches:
        return substruct, linker  # No amide bond found, return the original substruct
    
    side_atom = None
    oxygen_idx_found, carbonyl_idx_found = None, None
    for match in ester_matches:
        oxygen_idx, carbonyl_idx = match[0], match[1]
        oxygen_atom = query_substruct.GetAtomWithIdx(oxygen_idx)
        carbonyl_atom = query_substruct.GetAtomWithIdx(carbonyl_idx)

        for ester_atom in [oxygen_atom, carbonyl_atom]:
            # Check neighboring atoms for attachment points
            # NOTE: The dummy atom representing an attachment point have atomic number 0
            for neighbor in ester_atom.GetNeighbors():
                if neighbor.GetAtomicNum() == 0:
                    oxygen_idx_found = oxygen_idx
                    carbonyl_idx_found = carbonyl_idx
                    side_atom = "O" if ester_atom == oxygen_atom else "C"
                    break

            # If previous search failed, check the neighbors of the neighboring
            # atoms (second-order neighbors)
            if oxygen_idx_found is None or carbonyl_idx_found is None:
                for neighbor in ester_atom.GetNeighbors():
                    for second_neighbor in neighbor.GetNeighbors():
                        if second_neighbor.GetIdx() == carbonyl_idx or second_neighbor.GetIdx() == oxygen_idx:
                            continue  # Skip the opposite atom from the amide bond

                        if second_neighbor.GetAtomicNum() == 0:
                            oxygen_idx_found = oxygen_idx
                            carbonyl_idx_found = carbonyl_idx
                            side_atom = "O" if ester_atom == oxygen_atom else "C"
                            break
            else:
                break

    if oxygen_idx_found is None or carbonyl_idx_found is None or side_atom is None:
        return substruct, linker

    # Split the amide bond and adjust
    dummy_labels = [(3, 3)] # The E3 and substruct will have 1 and 2, so we need a third one
    amid_bond_idx = query_substruct.GetBondBetweenAtoms(oxygen_idx_found, carbonyl_idx_found).GetIdx()
    fragments = Chem.FragmentOnBonds(query_substruct, [amid_bond_idx], addDummies=True, dummyLabels=dummy_labels)

    # Get the fragments resulting from bond breaking
    try:
        mol_frags = Chem.GetMolFrags(fragments, asMols=True, sanitizeFrags=True)
    except Exception as e:
        print(e)
        return substruct, linker

    # Identify the "[*:substruct][<optional neighboring atom>]N[3*]" fragment, the other one will be the "truncated" substruct
    ester_fragment_pattern = Chem.MolFromSmarts(f"[*:{substruct_attachment_id}][{side_atom}][3*]")
    ester_fragment = None
    substruct_fixed = None

    for frag in mol_frags:
        if frag.HasSubstructMatch(dummy2query(ester_fragment_pattern)):
            ester_fragment = frag
        else:
            substruct_fixed = frag
    
    if ester_fragment is None or substruct_fixed is None:
        return substruct, linker

    # Rename the "[3*]" attachment point on the amide fragment to "[*:3]"
    ester_fragment_smiles = Chem.MolToSmiles(ester_fragment, canonical=True)
    ester_fragment_smiles = ester_fragment_smiles.replace('[3*]', f'[*:3]')
    ester_fragment = Chem.MolFromSmiles(ester_fragment_smiles)

    # Use molzip to join the linker and the fragment at the original attachment point
    linker_fixed = Chem.molzip(linker, ester_fragment)

    # Rename the "[*:3]" attachment point back to the original attachment point on the linker
    linker_fixed_smiles = Chem.MolToSmiles(linker_fixed, canonical=True)
    linker_fixed_smiles = linker_fixed_smiles.replace('[*:3]', f'[*:{substruct_attachment_id}]')
    linker_fixed = Chem.MolFromSmiles(linker_fixed_smiles)

    # Rename the "[3*]" attachment point back to the original attachment point on the substruct
    substruct_fixed_smiles = Chem.MolToSmiles(substruct_fixed, canonical=True)
    substruct_fixed_smiles = substruct_fixed_smiles.replace('[3*]', f'[*:{substruct_attachment_id}]')
    substruct_fixed = Chem.MolFromSmiles(substruct_fixed_smiles)

    return substruct_fixed, linker_fixed


# %% [markdown]
# ## Identify Functional Groups Close to Attachment Points

# %%

# %%
def find_functional_groups_near_attachment(substruct: Chem.Mol, attachment_id: int, verbose: int = 0) -> List[str]:
    """
    Find common functional groups close to the attachment point of a substruct.
    The attachment point is marked as "[*:1]" in the SMILES.
    A functional group should be considered "close" if the attachment point is its neighboring atom,
    or if it is one atom away (similar to the adjust_amide_bond function).
    
    Args:
        substruct: The molecule substruct to analyze.
        attachment_id: The attachment point ID in the substruct (e.g., 1 for "[*:1]").
        verbose: Verbosity level for logging.
    
    Returns:
        List[str]: A list of functional groups identified near the attachment point.
    """
    functional_groups_smarts = {
        "ester": "[CX3](=O)[OX2]",  # Ester group (COOR)
        "ether": "[OD2]([#6])[#6]",  # Ether group (R-O-R)
        "azo_compound": "[NX2]=[NX2]",  # Azo compound (R-N=N-R')
        "acid_anhydride": "[CX3](=O)[OX2][CX3](=O)",  # Acid anhydride (R-CO-O-CO-R)
        "thiol": "[SX2H]",  # SH group
        # ----------------------------------------------------------------------
        # The following are either on a side chain only, or not very common...
        # ----------------------------------------------------------------------
        # "hydroxyl": "[OX2H]",  # OH group
        # "carbonyl": "[CX3]=[OX1]",  # C=O
        "amine": "[NX3;H2,H1,H0;!$(NC=O)]",  # Primary, secondary, tertiary amines excluding amides
        # "carboxyl": "[CX3](=O)[OX2H1]",  # COOH group
        "aromatic_carbon": "c",  # Aromatic carbon
        "aromatic_nitrogen": "n",  # Aromatic nitrogen
        # "alkene": "[CX3]=[CX3]",  # Alkene (C=C)
        # "alkyne": "[CX2]#[CX2]",  # Alkyne (C#C)
        # "nitrile": "[CX2]#[NX1]",  # Nitrile group (C#N)
        # "sulfonyl": "[SX4](=O)(=O)[#6]",  # Sulfonyl group (SO2)
        # "phosphate": "[PX4](=O)([OX2H0])[OX2H1]",  # Phosphate group (PO4)
        # "aldehyde": "[CX3H1](=O)",  # Aldehyde group (CHO)
        # "alkane": "[CX4]",  # Alkane (R-H)
        # "epoxide": "[OX2r3]",  # Epoxide (three-membered cyclic ether)
        # "haloalkane": "[F,Cl,Br,I]",  # Haloalkane (R-X)
        # "acyl_halide": "[CX3](=O)[F,Cl,Br,I]",  # Acyl halide (R-CO-X)
        # "imine": "[NX2]=[CX3]",  # Imine (R-N=CR2)
        # "isocyanate": "[NX2]=[CX2]=[OX1]",  # Isocyanate (R-N=C=O)
    }

    # Define the attachment point in SMARTS notation
    attachment_point_smarts = f"[{attachment_id}*]"

    # Replace the current attachment point in substruct to make the query work
    fragment_smiles = Chem.MolToSmiles(substruct, canonical=True)
    fragment_smiles = fragment_smiles.replace(f'[*:{attachment_id}]', f'[{attachment_id}*]')
    substruct = Chem.MolFromSmiles(fragment_smiles)
    
    # Get the atom index of the attachment point
    attachment_atom = substruct.GetSubstructMatch(Chem.MolFromSmarts(attachment_point_smarts))
    if not attachment_atom:
        return []  # No attachment point found
    attachment_idx = attachment_atom[0]
    functional_groups_found = []

    # Iterate over each functional group pattern
    for group_name, group_smarts in functional_groups_smarts.items():
        group_pattern = Chem.MolFromSmarts(group_smarts)
        for match in substruct.GetSubstructMatches(group_pattern):
            for idx in match:
                # print(f'[DEBUG] {group_name} - {idx=}, {attachment_idx=}')
                # Check if the attachment point is neighboring or one atom away
                if idx == attachment_idx:
                    functional_groups_found.append(group_name)
                    break

                atom = substruct.GetAtomWithIdx(idx)
                for neighbor in atom.GetNeighbors():
                    if neighbor.GetIdx() == attachment_idx:
                        functional_groups_found.append(group_name)
                        break
                else:
                    # Check second-order neighbors
                    for neighbor in atom.GetNeighbors():
                        for second_neighbor in neighbor.GetNeighbors():
                            if second_neighbor.GetIdx() == attachment_idx:
                                functional_groups_found.append(group_name)
                                break
                        else:
                            continue
                        break

    return list(set(functional_groups_found))

# %%
def map_functional_group_to_protac(row):
    poi_smiles = row['POI Ligand SMILES with direction']
    linker_smiles = row['Linker SMILES with direction']
    e3_smiles = row['E3 Binder SMILES with direction']

    poi_mol = Chem.MolFromSmiles(poi_smiles)
    linker_mol = Chem.MolFromSmiles(linker_smiles)
    e3_mol = Chem.MolFromSmiles(e3_smiles)

    functional_groups = {
        'poi': find_functional_groups_near_attachment(poi_mol, POI_ATTACHMENT_ID),
        'e3': find_functional_groups_near_attachment(e3_mol, E3_ATTACHMENT_ID),
        'linker_poi': find_functional_groups_near_attachment(linker_mol, POI_ATTACHMENT_ID),
        'linker_e3': find_functional_groups_near_attachment(linker_mol, E3_ATTACHMENT_ID),
    }

    # Add a True/False column for each functional group (add a suffix per substructure)
    for substr, fg in functional_groups.items():
        for group in fg:
            row[f'fg_{substr}_{group}'] = True
    
    return row

# mapped_df = pd.read_csv(os.path.join(data_dir, 'processed', 'mapped_protacs.csv')).reset_index(drop=True)
# mapped_df = mapped_df.apply(map_functional_group_to_protac, axis=1)
# # Fill NaN values with False of columns starting with 'fg_'
# mapped_df.loc[:, mapped_df.columns.str.startswith('fg_')] = mapped_df.loc[:, mapped_df.columns.str.startswith('fg_')].fillna(False)

# mapped_df.head()

# %%
# import matplotlib.pyplot as plt
# import matplotlib.ticker as mtick

# # Get value counts of all columns starting with 'fg_'
# fg_columns = mapped_df.columns[mapped_df.columns.str.startswith('fg_')]
# fg_counts = mapped_df.loc[:, fg_columns].apply(pd.Series.value_counts).T
# # Plot the value counts as percentages and as stacked bar plots (horizontal)
# fg_counts = fg_counts.div(fg_counts.sum(axis=1), axis=0) * 100
# fg_counts.plot(kind='barh', stacked=True)
# # plt.title('Functional Groups Distribution in mapped PROTACs substructures')
# plt.xlabel('')
# plt.ylabel('')
# # Rename the y-labels
# def clean_label(s):
#     s = s.replace('fg_', '')
#     if 'linker_poi_' in s:
#         return s.split('linker_poi_')[-1].capitalize().replace('_', ' ') + ' (Linker-POI)'
#     elif 'linker_e3_' in s:
#         return s.split('linker_e3_')[-1].capitalize().replace('_', ' ') + ' (Linker-E3)'
#     elif 'poi_' in s:
#         return s.split('poi_')[-1].capitalize().replace('_', ' ') + ' (POI)'
#     elif 'e3_' in s:
#         return s.split('e3_')[-1].capitalize().replace('_', ' ') + ' (E3)'
#     return s
# plt.yticks(ticks=range(len(fg_counts)), labels=[clean_label(label) for label in fg_counts.index])

# # Show the percentage of present functional groups at the top of the bars
# for i, (index, row) in enumerate(fg_counts.iterrows()):
#     for j, value in enumerate(row):
#         x = value - 5 if value > 50 else 5
#         plt.text(x, i, f'{value:.1f}%', ha='center', va='center', color='white')

# # Show x-ticks as percentages
# plt.gca().xaxis.set_major_formatter(mtick.PercentFormatter())

# # Show the legend at the bottom with two columns, rename True/False to Present/Absent
# plt.legend(title='Functional Group', labels=['Absent', 'Present'], loc='lower center', ncol=2, bbox_to_anchor=(0.5, -0.22))
# plt.show()

# %%
# mapped_df.filter(regex='fg_linker_e3_').sum().sort_values(ascending=False)
# mapped_df[mapped_df['fg_linker_e3_azo_compound']].head()

# def display_protac_in_row(row):
#     display_protac_substructures(
#         row['PROTAC SMILES'],
#         row['POI Ligand SMILES with direction'],
#         row['Linker SMILES with direction'],
#         row['E3 Binder SMILES with direction'],
#         compound_id=row['PROTAC ID'],
#         print_smiles=True,
#     )

# # for fg_group in ['fg_e3_ester', 'fg_poi_ester', 'fg_linker_e3_azo_compound']:
# #     print(f'Functional group: {fg_group}')
# #     mapped_df[mapped_df[fg_group]].apply(display_protac_in_row, axis=1)
# #     print('-' * 80)

# # mapped_df[mapped_df['fg_poi_ether'] & (mapped_df['fg_poi_aromatic_carbon'] | mapped_df['fg_poi_aromatic_nitrogen'])].iloc[:10].apply(display_protac_in_row, axis=1)
# # print('-' * 80)
# # mapped_df[mapped_df['fg_e3_ether'] & mapped_df['fg_e3_aromatic_carbon']].iloc[:10].apply(display_protac_in_row, axis=1)
# print('-' * 80)
# mapped_df[mapped_df['fg_e3_ether'] & (mapped_df['fg_e3_ether'] | mapped_df['fg_e3_aromatic_carbon']) & (~mapped_df['fg_e3_amine']) & (~mapped_df['fg_linker_e3_amine'])].iloc[:10].apply(display_protac_in_row, axis=1)

# %% [markdown]
# ## Load Raw Datasets

# %% [markdown]
# From [PROTAC-DB paper](https://academic.oup.com/nar/article/49/D1/D1381/5917660?login=false#:~:text=For%20linkers%2C%20only%20the%202D%20structures%2C%20compound%20IDs%20and%20targeted%20proteins%20are%20shown%20in%20the%20datasheet.%20The%20%E2%80%98R1%E2%80%99%20and%20%E2%80%98R2%E2%80%99%20in%20the%20structures%20represent%20the%20sites%20that%20conjugate%20warheads%20and%20E3%20ligands%2C%20respectively.):
# 
# > For linkers, only the 2D structures, compound IDs and targeted proteins are shown in the datasheet. The ‘R1’ and ‘R2’ in the structures represent the sites that conjugate warheads and E3 ligands, respectively.

# %%


# %% [markdown]
# The following is an attempt to map the provided PROTAC-DB linkers to the PROTACs in the dataset. Only a very small fraction of the linkers could be mapped to the PROTACs in the dataset... So we skip it for now.

# %%
# tmp = protac_db_df.copy()
# tmp['PROTAC SMILES'] = tmp['PROTAC SMILES'].apply(canonize_smiles)
# tmp = tmp.merge(protac_db_linker_df, on='Compound ID', how='left', suffixes=('', '_linker'))
# tmp = tmp[['PROTAC SMILES', 'Linker SMARTS']]
# print(len(tmp))
# # Replace "[R1]" in the linker SMILES with "[*:2]"
# tmp['Linker SMARTS'] = tmp['Linker SMARTS'].str.replace('[R1]', f'[*:{POI_ATTACHMENT_ID}]')
# tmp['Linker SMARTS'] = tmp['Linker SMARTS'].str.replace('[R2]', f'[*:{E3_ATTACHMENT_ID}]')
# tmp = tmp.dropna()
# print(len(tmp))

# def get_substructs_df(row: pd.Series) -> Dict[str, str]:
#     substructs = get_substructs_from_mapped_linker(row['PROTAC SMILES'], row['Linker SMARTS'])
#     if substructs is not None:
#         if check_reassembly(canonize_smiles(row['PROTAC SMILES']), '.'.join(substructs.values())):
#             for key, value in substructs.items():
#                 row[key] = value
#     return row

# tqdm.pandas(desc='Extracting substructures')
# tmp = tmp.progress_apply(get_substructs_df, axis=1)

# tmp = tmp.dropna().drop_duplicates(subset=['PROTAC SMILES', 'linker', 'e3', 'poi'])
# safe_display(tmp)
# print(len(tmp))

# for i, row in tmp.sample(10).iterrows():
#     protac_mol = Chem.MolFromSmiles(row['PROTAC SMILES'])
#     poi_mol = Chem.MolFromSmiles(row['poi'])
#     e3_mol = Chem.MolFromSmiles(row['e3'])
#     linker_mol = Chem.MolFromSmiles(row['linker'])
#     img = Draw.MolsToGridImage([protac_mol, poi_mol, linker_mol, e3_mol], molsPerRow=4, subImgSize=(1000, 500), legends=['PROTAC', 'POI', 'Linker', 'E3'])
#     safe_display(img)

# print(f'Number of unique POI: {len(tmp["poi"].unique())}')
# print(f'Number of unique E3: {len(tmp["e3"].unique())}')
# print(f'Number of unique linker: {len(tmp["linker"].unique())}')

# %% [markdown]
# ## Generate Dictionaries

# %%
from collections import defaultdict



# %% [markdown]
# Remove stereochemistry and update dictionaries with those changes:

# %%
def remove_stereo(smiles: str) -> str:
    """ Removes stereochemistry from a SMILES string.
    
    Args:
        smiles: The input SMILES string.

    Returns:
        The SMILES string without stereochemistry.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    try:
        Chem.rdmolops.RemoveStereochemistry(mol)
        return Chem.MolToSmiles(mol)
    except Exception as e:
        return np.nan



# %% [markdown]
# ## Extra Substructures (Manually Curated)

# %%


def update_dictionary(dictionary: pd.DataFrame, substr_to_add: list, verbose: int = 0) -> pd.DataFrame:
    """ Updates a dictionary with a list of additional substructures.

    Args:
        dictionary: The input dictionary dataframe.
        substr_to_add: The list of additional substructures.

    Returns:
        The updated dictionary dataframe.
    """
    # Canonize the SMILES strings
    substr_to_add = [canonize_smiles(smiles) for smiles in substr_to_add if smiles is not None]

    # Remove entries already in the dictionary
    for smiles in substr_to_add:
        if smiles in dictionary[f'SMILES'].values:
            if verbose > 1:
                print(f'\tWARNING. SMILES already in the dictionary: {smiles}')
            # Remove it from the list
            substr_to_add.remove(smiles)

    # Get the current maximum ID
    max_id = dictionary['ID'].max()
    new_entries = []
    for smiles in substr_to_add:
        max_id += 1
        mol = Chem.MolFromSmiles(smiles)
        # Remove entries that result in invalid molecules
        if mol is None:
            continue
        new_entries.append({
            'SMILES': smiles,
            'Molecule': mol,
            'ID': max_id,
        })
        # Try adding its no-stereochemistry version as well
        smiles_nostereo = remove_stereo(smiles)
        if smiles_nostereo is not None and smiles_nostereo != smiles:
            new_entries.append({
                'SMILES': canonize_smiles(smiles_nostereo),
                'Molecule': Chem.MolFromSmiles(smiles_nostereo),
                'ID': max_id,
            })
    new_entries = pd.DataFrame(new_entries)

    if len(new_entries) > 0 and verbose:
        print(f'Number of substructures added to the dictionary: {len(new_entries)}')

    # Return the updated dictionary
    return pd.concat([dictionary, pd.DataFrame(new_entries)], axis=0).reset_index(drop=True)


# %% [markdown]
# ### Example of dictionary usage

# %%
def get_id_from_dict(
        substr_smiles: str,
        dictionary: pd.DataFrame,
        update_dict_if_not_found: bool = False,
        verbose: int = 0,
) -> Optional[int]:
    """ Get the ID of a substructure from a dictionary.

    Args:
        substr_smiles: The SMILES of the substructure.
        dictionaries: The dictionaries containing the substructures.
        substr_name: The name of the substructure.
        update_dict_if_not_found: Whether to update the dictionary if the substructure is not found.

    Returns:
        The ID of the substructure, if present, None otherwise.
    """
    substr_smiles = canonize_smiles(substr_smiles)
    # if substr_smiles is None:
    #     return None
    substr_id = None
    if substr_smiles in dictionary['SMILES'].values:
        substr_id = dictionary[dictionary['SMILES'] == substr_smiles]['ID'].values[0]
    elif update_dict_if_not_found:
        dictionary = update_dictionary(dictionary, [substr_smiles], verbose)
        substr_id = dictionary[dictionary['SMILES'] == substr_smiles]['ID'].values[0]
    return substr_id


# %% [markdown]
# ## Get Substructures From Dictionaries
# 
# TODO: For some matched protacs, the removal of the attachment points, _i.e._, dummy atoms, fails, despite their SMILES with attachment point being correct.

# %%
import time
import multiprocessing

def adjust_amide_bonds_in_substructs(substructs: Dict[str, str], protac_smiles: str) -> Dict[str, str]:
    poi_mol = Chem.MolFromSmiles(substructs['poi'])
    e3_mol = Chem.MolFromSmiles(substructs['e3'])
    linker_mol = Chem.MolFromSmiles(substructs['linker'])

    # Fix the amide group on the POI ligand
    poi_mol, linker_mol = adjust_amide_bond(poi_mol, linker_mol, POI_ATTACHMENT_ID)
    poi_smiles = Chem.MolToSmiles(poi_mol, canonical=True)
    linker_smiles = Chem.MolToSmiles(linker_mol, canonical=True)
    e3_smiles = substructs['e3']
    if not check_reassembly(protac_smiles, '.'.join([poi_smiles, linker_smiles, e3_smiles])):
        return substructs

    # Fix the amide group on the E3 binder
    e3_mol, linker_mol = adjust_amide_bond(e3_mol, linker_mol, E3_ATTACHMENT_ID)
    e3_smiles = Chem.MolToSmiles(e3_mol, canonical=True)
    linker_smiles = Chem.MolToSmiles(linker_mol, canonical=True)
    if not check_reassembly(protac_smiles, '.'.join([poi_smiles, linker_smiles, e3_smiles])):
        return substructs
    
    # Fix the amide group on the linker, E3 side
    linker_mol, e3_mol = adjust_amide_bond(linker_mol, e3_mol, E3_ATTACHMENT_ID)
    e3_smiles = Chem.MolToSmiles(e3_mol, canonical=True)
    linker_smiles = Chem.MolToSmiles(linker_mol, canonical=True)
    if not check_reassembly(protac_smiles, '.'.join([poi_smiles, linker_smiles, e3_smiles])):
        return substructs
    
    # Fix the amide group on the linker, POI side
    linker_mol, poi_mol = adjust_amide_bond(linker_mol, poi_mol, POI_ATTACHMENT_ID)
    poi_smiles = Chem.MolToSmiles(poi_mol, canonical=True)
    linker_smiles = Chem.MolToSmiles(linker_mol, canonical=True)
    if not check_reassembly(protac_smiles, '.'.join([poi_smiles, linker_smiles, e3_smiles])):
        return substructs

    substructs['poi'] = poi_smiles
    substructs['e3'] = e3_smiles
    substructs['linker'] = linker_smiles
    return substructs

def adjust_ester_bonds_in_substructs(substructs: Dict[str, str], protac_smiles: str) -> Dict[str, str]:
    poi_mol = Chem.MolFromSmiles(substructs['poi'])
    e3_mol = Chem.MolFromSmiles(substructs['e3'])
    linker_mol = Chem.MolFromSmiles(substructs['linker'])

    # Fix the amide group on the POI ligand
    poi_mol, linker_mol = adjust_ester_bond(poi_mol, linker_mol, POI_ATTACHMENT_ID)
    poi_smiles = Chem.MolToSmiles(poi_mol, canonical=True)
    linker_smiles = Chem.MolToSmiles(linker_mol, canonical=True)
    e3_smiles = substructs['e3']
    if not check_reassembly(protac_smiles, '.'.join([poi_smiles, linker_smiles, e3_smiles])):
        return substructs

    # Fix the amide group on the E3 binder
    e3_mol, linker_mol = adjust_ester_bond(e3_mol, linker_mol, E3_ATTACHMENT_ID)
    e3_smiles = Chem.MolToSmiles(e3_mol, canonical=True)
    linker_smiles = Chem.MolToSmiles(linker_mol, canonical=True)
    if not check_reassembly(protac_smiles, '.'.join([poi_smiles, linker_smiles, e3_smiles])):
        return substructs
    
    # Fix the amide group on the linker, E3 side
    linker_mol, e3_mol = adjust_ester_bond(linker_mol, e3_mol, E3_ATTACHMENT_ID)
    e3_smiles = Chem.MolToSmiles(e3_mol, canonical=True)
    linker_smiles = Chem.MolToSmiles(linker_mol, canonical=True)
    if not check_reassembly(protac_smiles, '.'.join([poi_smiles, linker_smiles, e3_smiles])):
        return substructs
    
    # Fix the amide group on the linker, POI side
    linker_mol, poi_mol = adjust_ester_bond(linker_mol, poi_mol, POI_ATTACHMENT_ID)
    poi_smiles = Chem.MolToSmiles(poi_mol, canonical=True)
    linker_smiles = Chem.MolToSmiles(linker_mol, canonical=True)
    if not check_reassembly(protac_smiles, '.'.join([poi_smiles, linker_smiles, e3_smiles])):
        return substructs

    substructs['poi'] = poi_smiles
    substructs['e3'] = e3_smiles
    substructs['linker'] = linker_smiles
    return substructs

def apply_function(row, dictionaries, biggest_matches_first, max_iter_on_linkers):
    protac_smiles = row['SMILES']
    protac_mol = row['Molecule']

    # Get all substructure matches in the POI dictionary
    poi_matches = dictionaries['POI Ligand']['Molecule'].apply(lambda x: protac_mol.HasSubstructMatch(x, useChirality=True))
    pois = dictionaries['POI Ligand'][poi_matches]

    # Get all substructure matches in the E3 dictionary
    e3_matches = dictionaries['E3 Binder']['Molecule'].apply(lambda x: protac_mol.HasSubstructMatch(x, useChirality=True))
    e3s = dictionaries['E3 Binder'][e3_matches]

    # Sort the matches by the number of atoms in the molecule
    ascending = False if biggest_matches_first else True
    pois = pois.sort_values(by='Molecule', key=lambda s: s.apply(lambda m: m.GetNumAtoms()), ascending=ascending)
    e3s = e3s.sort_values(by='Molecule', key=lambda s: s.apply(lambda m: m.GetNumAtoms()), ascending=ascending)

    for _, poi in pois.iterrows():
        for _, e3 in e3s.iterrows():
            substructs = get_substructs_from_unmapped_e3_poi(protac_smiles, protac_mol, poi['Molecule'], e3['Molecule'])

            # If the substructure is not found, try to get it from a non-perfect match
            if substructs is None:
                fixed_poi, _ = get_substructure_from_non_perfect_match(protac_mol, poi['Molecule'], POI_ATTACHMENT_ID)
                fixed_e3, _ = get_substructure_from_non_perfect_match(protac_mol, e3['Molecule'], E3_ATTACHMENT_ID)
                fixed_poi = poi['Molecule'] if fixed_poi is None else fixed_poi
                fixed_e3 = e3['Molecule'] if fixed_e3 is None else fixed_e3
                if fixed_poi is not None and fixed_e3 is not None:
                    substructs = get_substructs_from_unmapped_e3_poi(protac_smiles, protac_mol, fixed_poi, fixed_e3)

            if substructs is not None:
                # Fix the bonds close to amide and ester groups, if necessary
                substructs = adjust_amide_bonds_in_substructs(substructs, protac_smiles)
                substructs = adjust_ester_bonds_in_substructs(substructs, protac_smiles)
                # Add the mapped PROTAC to the final list
                mapped_row = get_mapped_row(row, dictionaries, substructs, poi['SMILES'], e3['SMILES'])
                mapped_row['Notes'] = 'Obtained from non-mapped POI and E3 found in the dictionary.'
                return mapped_row

    # Get all substructure matches in the Linker with direction dictionary
    if max_iter_on_linkers > 0:
        matches = dictionaries['Linker with direction']['Molecule'].apply(lambda x: protac_mol.HasSubstructMatch(dummy2query(x), useChirality=True))
        linkers = dictionaries['Linker with direction'][matches]

        # Sort all the matches by the number of atoms in the linker, the biggest first
        linkers = linkers.sort_values(by='Molecule', key=lambda s: s.apply(lambda m: m.GetNumAtoms()), ascending=False)

        for j, row_linker in linkers.iterrows():
            if j >= max_iter_on_linkers and max_iter_on_linkers > 0:
                return row
            linker_smiles = row_linker['SMILES']
            substructs = get_substructs_from_mapped_linker(row['SMILES'], linker_smiles)
            if substructs is not None:
                # Fix the bonds close to amide and ester groups, if necessary
                substructs = adjust_amide_bonds_in_substructs(substructs, protac_smiles)
                substructs = adjust_ester_bonds_in_substructs(substructs, protac_smiles)
                mapped_row = get_mapped_row(row, dictionaries, substructs)
                mapped_row['Notes'] = f'Obtained from mapped linker found in the dictionary ({max_iter_on_linkers=}).'
                return mapped_row
    return row

def get_mapped_row(row, dictionaries, substructs, poi_smiles_no_dummy=None, e3_smiles_no_dummy=None, update_dict_if_ids_not_found=False):
    mapped_row = {}
    mapped_row['PROTAC SMILES'] = canonize_smiles(row['SMILES'])
    mapped_row['PROTAC ID'] = row['ID']
    mapped_row['POI Ligand SMILES with direction'] = substructs['poi']
    mapped_row['E3 Binder SMILES with direction'] = substructs['e3']
    mapped_row['Linker SMILES with direction'] = substructs['linker']
    mapped_row['POI Ligand SMILES'] = remove_dummy_atoms(substructs['poi']) if poi_smiles_no_dummy is None else poi_smiles_no_dummy
    mapped_row['E3 Binder SMILES'] = remove_dummy_atoms(substructs['e3']) if e3_smiles_no_dummy is None else e3_smiles_no_dummy
    mapped_row['Linker SMILES'] = remove_dummy_atoms(substructs['linker'])

    # Get the IDs and update the dictionaries with new substructures
    mapped_row['POI Ligand ID'] = get_id_from_dict(mapped_row['POI Ligand SMILES'], dictionaries['POI Ligand'], update_dict_if_not_found=update_dict_if_ids_not_found)
    mapped_row['E3 Binder ID'] = get_id_from_dict(mapped_row['E3 Binder SMILES'], dictionaries['E3 Binder'], update_dict_if_not_found=update_dict_if_ids_not_found)
    mapped_row['Linker ID'] = get_id_from_dict(mapped_row['Linker SMILES with direction'], dictionaries['Linker with direction'], update_dict_if_not_found=update_dict_if_ids_not_found)

    return mapped_row

from functools import partial

def map_protacs(
        protac_df: pd.DataFrame,
        dictionaries: Dict[str, pd.DataFrame] = None,
        max_iter_on_linkers: int = 5,
        biggest_matches_first: bool = True,
        update_dict_if_ids_not_found: bool = False,
        use_multiprocessing: bool = False,
) -> pd.DataFrame:
    """ Maps PROTACs to their substructures.

    Args:
        protac_df: The input PROTAC dataframe.
        max_iter_on_linkers: The maximum number of matching linkers to iterate over. If zero, there will be no attempt to match linkers in the dictionary. If negative, iterate over all matched linkers. Default is 5.
        dictionaries: The input dictionaries.

    Returns:
        The mapped PROTAC dataframe.
    """

    if use_multiprocessing:
        with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
            results = pool.map(partial(apply_function, dictionaries=dictionaries, biggest_matches_first=biggest_matches_first, max_iter_on_linkers=max_iter_on_linkers), protac_df.copy().to_dict(orient='records'))
        
        mapped_protacs = pd.DataFrame(results)
        mapped_protacs = mapped_protacs.dropna(subset=['POI Ligand SMILES with direction', 'E3 Binder SMILES with direction', 'Linker SMILES with direction'])
        return mapped_protacs

    mapped_protacs = []
    for i, row in (pbar := tqdm(protac_df.iterrows(), total=len(protac_df))):
        pbar.set_description(f'PROTAC n.{i:4d}')

        protac_smiles = row['SMILES']
        protac_mol = row['Molecule']

        # Get all substructure matches in the POI dictionary
        poi_matches = dictionaries['POI Ligand']['Molecule'].apply(lambda x: protac_mol.HasSubstructMatch(x, useChirality=True))
        pois = dictionaries['POI Ligand'][poi_matches]

        # Get all substructure matches in the E3 dictionary
        e3_matches = dictionaries['E3 Binder']['Molecule'].apply(lambda x: protac_mol.HasSubstructMatch(x, useChirality=True))
        e3s = dictionaries['E3 Binder'][e3_matches]

        # Sort the matches by the number of atoms in the molecule
        ascending = False if biggest_matches_first else True
        pois = pois.sort_values(by='Molecule', key=lambda s: s.apply(lambda m: m.GetNumAtoms()), ascending=ascending)
        e3s = e3s.sort_values(by='Molecule', key=lambda s: s.apply(lambda m: m.GetNumAtoms()), ascending=ascending)

        for _, poi in pois.iterrows():
            for _, e3 in e3s.iterrows():
                substructs = get_substructs_from_unmapped_e3_poi(protac_smiles, protac_mol, poi['Molecule'], e3['Molecule'])

                # If the substructure is not found, try to get it from a non-perfect match
                if substructs is None:
                    fixed_poi, _ = get_substructure_from_non_perfect_match(protac_mol, poi['Molecule'], POI_ATTACHMENT_ID)
                    fixed_e3, _ = get_substructure_from_non_perfect_match(protac_mol, e3['Molecule'], E3_ATTACHMENT_ID)
                    fixed_poi = poi['Molecule'] if fixed_poi is None else fixed_poi
                    fixed_e3 = e3['Molecule'] if fixed_e3 is None else fixed_e3
                    if fixed_poi is not None and fixed_e3 is not None:
                        substructs = get_substructs_from_unmapped_e3_poi(protac_smiles, protac_mol, fixed_poi, fixed_e3)

                if substructs is not None:
                    # Fix the amide bond in the linker, if necessary
                    substructs = adjust_amide_bonds_in_substructs(substructs, protac_smiles)
                    # Add the mapped PROTAC to the final list
                    mapped_row = get_mapped_row(row, dictionaries, substructs, poi['SMILES'], e3['SMILES'])
                    mapped_row['Notes'] = 'Obtained from non-mapped POI and E3 found in the dictionary.'
                    mapped_protacs.append(mapped_row)
                    break
            if substructs is not None:
                break

        if substructs is not None:
            pbar.set_postfix({'len mapped': len(mapped_protacs), 'perc mapped': f'{len(mapped_protacs) / len(protac_df):.1%}'})
            continue

        # Get all substructure matches in the Linker with direction dictionary
        if max_iter_on_linkers > 0:
            matches = dictionaries['Linker with direction']['Molecule'].apply(lambda x: protac_mol.HasSubstructMatch(dummy2query(x), useChirality=True))
            linkers = dictionaries['Linker with direction'][matches]

            # Sort all the matches by the number of atoms in the linker, the biggest first
            linkers = linkers.sort_values(by='Molecule', key=lambda s: s.apply(lambda m: m.GetNumAtoms()), ascending=False)

            for j, row_linker in linkers.iterrows():
                if j >= max_iter_on_linkers and max_iter_on_linkers > 0:
                    break
                linker_smiles = row_linker['SMILES']
                substructs = get_substructs_from_mapped_linker(row['SMILES'], linker_smiles)
                if substructs is not None:
                    mapped_row = get_mapped_row(row, dictionaries, substructs)
                    mapped_row['Notes'] = f'Obtained from mapped linker found in the dictionary ({max_iter_on_linkers=}).'
                    mapped_protacs.append(mapped_row)
                    break

        pbar.set_postfix({'len mapped': len(mapped_protacs), 'perc mapped': f'{len(mapped_protacs) / len(protac_df):.1%}'})

    mapped_protacs = pd.DataFrame(mapped_protacs)
    return mapped_protacs


def main():

    protac_pedia_df = pd.read_csv(os.path.join(data_dir, 'raw', 'PROTAC-Pedia.csv')).reset_index(drop=True)
    protac_db_df = pd.read_csv(os.path.join(data_dir, 'raw', 'PROTAC-DB.csv')).reset_index(drop=True)
    protac_db_linker_df = pd.read_csv(os.path.join(data_dir, 'raw', 'PROTAC-DB-Linkers.csv')).reset_index(drop=True)
    protac_db_e3_df = pd.read_csv(os.path.join(data_dir, 'raw', 'PROTAC-DB-E3-Ligands.csv')).reset_index(drop=True)
    protac_db_poi_df = pd.read_csv(os.path.join(data_dir, 'raw', 'PROTAC-DB-Warheads.csv')).reset_index(drop=True)

    # Rename columns to make them consistent the PROTAC-Pedia columns
    protac_pedia_df = protac_pedia_df.rename(columns={
        'Linker': 'Linker SMILES',
        'Ligand SMILES': 'POI Ligand SMILES',
    })
    protac_db_df = protac_db_df.rename(columns={'Smiles': 'PROTAC SMILES'})
    protac_db_linker_df = protac_db_linker_df.rename(columns={'Smiles': 'Linker SMILES', 'Smiles_R': 'Linker SMARTS'})
    protac_db_e3_df = protac_db_e3_df.rename(columns={'Smiles': 'E3 Binder SMILES'})
    protac_db_poi_df = protac_db_poi_df.rename(columns={'Smiles': 'POI Ligand SMILES'})

    # Replace the attachment points in the linker SMARTS with the ones we defined
    protac_db_linker_df['Linker SMARTS'] = protac_db_linker_df['Linker SMARTS'].str.replace('[R1]', f'[*:{POI_ATTACHMENT_ID}]')
    protac_db_linker_df['Linker SMARTS'] = protac_db_linker_df['Linker SMARTS'].str.replace('[R2]', f'[*:{E3_ATTACHMENT_ID}]')

    # Canonize all the SMILES strings
    tqdm.pandas(desc='Canonizing SMILES')
    protac_pedia_df['PROTAC SMILES'] = protac_pedia_df['PROTAC SMILES'].progress_apply(canonize_smiles)
    protac_db_df['PROTAC SMILES'] = protac_db_df['PROTAC SMILES'].progress_apply(canonize_smiles)
    protac_db_linker_df['Linker SMILES'] = protac_db_linker_df['Linker SMILES'].progress_apply(canonize_smiles)
    protac_db_linker_df['Linker SMARTS'] = protac_db_linker_df['Linker SMARTS'].progress_apply(canonize_smiles)
    protac_db_e3_df['E3 Binder SMILES'] = protac_db_e3_df['E3 Binder SMILES'].progress_apply(canonize_smiles)
    protac_db_poi_df['POI Ligand SMILES'] = protac_db_poi_df['POI Ligand SMILES'].progress_apply(canonize_smiles)

    # Drop rows with invalid SMILES
    protac_db_df = protac_db_df.dropna(subset=['PROTAC SMILES'])
    protac_db_linker_df = protac_db_linker_df.dropna(subset=['Linker SMILES', 'Linker SMARTS'], how='all')
    protac_db_e3_df = protac_db_e3_df.dropna(subset=['E3 Binder SMILES'])
    protac_db_poi_df = protac_db_poi_df.dropna(subset=['POI Ligand SMILES'])

    print('')
    print('Listing unique compound IDs according to PROTAC-DB:')
    print(f'protac_db_df:        {len(protac_db_df['Compound ID'].unique()):,}')
    print(f'protac_db_poi_df:    {len(protac_db_poi_df['Compound ID'].unique()):,}')
    print(f'protac_db_linker_df: {len(protac_db_linker_df['Compound ID'].unique()):,}')
    print(f'protac_db_e3_df:     {len(protac_db_e3_df['Compound ID'].unique()):,}')

    print('')
    print(f'Number of unique PROTAC SMILES in PROTAC-DB:     {len(protac_db_df["PROTAC SMILES"].unique()):,}')
    print(f'Number of unique POI Ligand SMILES in PROTAC-DB: {len(protac_db_poi_df["POI Ligand SMILES"].unique()):,}')
    print(f'Number of unique E3 Binder SMILES in PROTAC-DB:  {len(protac_db_e3_df["E3 Binder SMILES"].unique()):,}')
    print(f'Number of unique Linker SMILES in PROTAC-DB:     {len(protac_db_linker_df["Linker SMILES"].unique()):,}')
    print(f'Number of unique Linker SMARTS in PROTAC-DB:     {len(protac_db_linker_df["Linker SMARTS"].unique()):,}')

    print('')
    print(f'Number of unique PROTAC SMILES in PROTAC-Pedia:     {len(protac_pedia_df["PROTAC SMILES"].unique()):,}')
    print(f'Number of unique POI Ligand SMILES in PROTAC-Pedia: {len(protac_pedia_df["POI Ligand SMILES"].unique()):,}')
    print(f'Number of unique E3 Binder SMILES in PROTAC-Pedia:  {len(protac_pedia_df["E3 Binder SMILES"].unique()):,}')
    print(f'Number of unique Linker SMILES in PROTAC-Pedia:     {len(protac_pedia_df["Linker SMILES"].unique()):,}')


    dictionaries_orig = defaultdict(list)

    # Get a dictionary of all the available PROTACs
    protac_smiles = pd.concat([protac_pedia_df['PROTAC SMILES'], protac_db_df['PROTAC SMILES']]).dropna().unique()
    for idx, protac in enumerate(protac_smiles):
        dictionaries_orig['PROTAC'].append({'SMILES': protac, 'ID': idx})

    # Get dictionaries_orig for the POI ligands, linkers, and E3 binders
    for substr_name in ['POI Ligand', 'Linker', 'E3 Binder', 'Linker with direction']:
        if substr_name == 'POI Ligand':
            substruct_smiles = pd.concat([protac_pedia_df['POI Ligand SMILES'], protac_db_poi_df['POI Ligand SMILES']]).dropna().unique()
        elif substr_name == 'Linker':
            substruct_smiles = pd.concat([protac_pedia_df['Linker SMILES'], protac_db_linker_df['Linker SMILES']]).dropna().unique()
        elif substr_name == 'E3 Binder':
            substruct_smiles = pd.concat([protac_pedia_df['E3 Binder SMILES'], protac_db_e3_df['E3 Binder SMILES']]).dropna().unique()
        elif substr_name == 'Linker with direction':
            substruct_smiles = pd.concat([protac_db_linker_df['Linker SMARTS'], protac_db_linker_df['Linker SMARTS']]).dropna().unique()
        for idx, smiles in enumerate(substruct_smiles):
            dictionaries_orig[substr_name].append({f'SMILES': smiles, f'ID': idx})

    # Convert the dictionaries_orig to dataframes
    for key, dictionary in dictionaries_orig.items():
        dictionaries_orig[key] = pd.DataFrame(dictionary)

    # Reporting
    for key, dictionary in dictionaries_orig.items():
        print(f'{key}: {len(dictionary):,}')

    dictionaries_nostereo = {}
    # Remove stereochemistry from all the SMILES strings in the dictionaries
    for key, value in dictionaries_orig.items():
        dictionaries_nostereo[key] = value.copy()
        tqdm.pandas(desc=f'Removing stereochemistry from {key}')
        dictionaries_nostereo[key]['SMILES'] = value['SMILES'].progress_apply(remove_stereo)
        # Drop rows with invalid SMILES
        dictionaries_nostereo[key] = dictionaries_nostereo[key].dropna(subset=['SMILES'])

    # Reporting
    print('\nDictionaries with no stereochemistry:')
    for key, value in dictionaries_nostereo.items():
        print(f'{key}: {len(value):,}')

    # Concatenate the respective dictionaries
    dictionaries_no_extra = {}
    for key, value in dictionaries_nostereo.items():
        smiles_col = [col for col in value.columns if 'SMILES' in col][0]
        dictionaries_no_extra[key] = pd.concat([dictionaries_orig[key], value.copy()], axis=0).drop_duplicates(subset=smiles_col).reset_index(drop=True)

    # Reporting
    print('\nFinal dictionaries_no_extra:')
    for key, dictionary in dictionaries_no_extra.items():
        print(f'{key}: {len(dictionary):,}')
        print(f'\tNumber of unique SMILES: {len(dictionary["SMILES"].unique()):,}')
        print(f'\tNumber of unique IDs: {len(dictionary["ID"].unique()):,}')

    # %% [markdown]
    # Precompute all molecules in all dictionaries:

    # %%
    for key, d in dictionaries_no_extra.items():
        tqdm.pandas(desc=f'Creating molecules for {key}')
        dictionaries_no_extra[key]['Molecule'] = d['SMILES'].progress_apply(Chem.MolFromSmiles)
        # Drop rows with invalid molecules
        dictionaries_no_extra[key] = dictionaries_no_extra[key].dropna(subset=['Molecule'])

    # Reporting
    print('\nDictionaries with molecules after removing invalid molecules:')
    for key, dictionary in dictionaries_no_extra.items():
        print(f'{key}: {len(dictionary):,}')
        print(f'\tNumber of unique SMILES: {len(dictionary["SMILES"].unique()):,}')
        print(f'\tNumber of unique IDs: {len(dictionary["ID"].unique()):,}')

    additional_e3 = [
        'CC(C)C[C@H](NC(=O)[C@@H](O)[C@H](N)Cc1ccccc1)C(N)=O',
        'COc1cc(ccc1NC(=O)[C@@H]1N[C@@H](CC(C)(C)C)[C@@](C#N)([C@H]1c1cccc(Cl)c1F)c1ccc(Cl)cc1F)C(N)=O',
        'N[C@@H](CCCNC(N)=N)C(N)=O',
        'N[C@@H](Cc1c[nH]cn1)C(N)=O',
        'CC(C)[C@H](NC(C)=O)C(=O)N1C[C@@H](O)C[C@@H]1C(=O)N[C@@H](CC=O)c1ccccc1',
        'CSC(C)(C)[C@H](N)C(=O)N1C[C@H](O)C[C@H]1C(=O)NCc1ccc(cc1)-c1scnc1C',
        'Nc1cccc2C(=O)N(C3CCC(=O)NC3=O)C(=O)c12',
        'Cc1cccc2C(=O)N(Cc12)C1CCC(=O)NC1=O',
        'Nc1ccc2C(=O)N(Cc2c1)C1CCC(=O)NC1=O',
        # --------------------------------------------------------------------------
        # The following are manually added entries
        # --------------------------------------------------------------------------
        # 'N[C@H](C(=O)N1C[C@H](O)C[C@H]1C(=O)NCc1ccc(-c2scnc2C)cc1)C(C)(C)C', #TODO: This is coming from a trained model, so it might be wrong
        # --------------------------------------------------------------------------
        # 'C[C@H](NC(=O)[C@@H]1C[C@@H](O)CN1C(=O)[C@@H](N)C(C)(C)C)c1ccc(cc1)-c1scnc1C',
        # 'Cc1ncsc1-c1ccc(CNC(=O)[C@@H]2C[C@@H](O)CN2C(=O)[C@@H](N)C(C)(C)C)cc1',
        # 'NC(=O)c1c(N)n(nc1-c1ccc(Oc2ccc(F)cc2F)cc1)[C@@H]1CCCNC1',
        # 'Fc1ccc(Cc2n[nH]c(=O)c3ccccc23)cc1C(=O)N1CCNCC1',
        # 'NC(=O)c1c(N)n(nc1-c1ccc(Oc2ccc(F)cc2F)cc1)[C@@H]1CCCNC1',
        # 'Cc1ccccc1-n1c(C)cc(OCc2ccc(F)cc2F)c(Br)c1=O',
        # 'CC(C)c1cnn2c(NCc3ccc(N)cc3)nc(OC3CCN(C)CC3)nc12',
        # 'CCNc1nc(N2CCN(CC2)C(=O)CC)c2cc(Cl)c(c(F)c2n1)-c1cc(O)cc2ccccc12',
        # 'Fc1c(Cl)cccc1[C@H]1[C@@H](NC2(CCCCC2)[C@@]11C(=O)Nc2cc(Cl)ccc12)C(=O)Nc1ccccc1',
    ]
    additional_poi = [
        'CNCC[C@H](CSc1ccccc1)Nc1ccc(cc1S(=O)(=O)C(F)(F)F)S(=O)(=O)NC(=O)c1ccc(cc1)N1CCN(CC2=C(CCC(C)(C)C2)c2ccc(Cl)cc2)CC1',
        'CNCC[C@H](CSc1ccccc1)Nc1ccc(cc1S(=O)(=O)C(F)(F)F)S(=O)(=O)NC(=O)c1ccc(cc1)N1CCN(CC2=C(CCC(C)(C)C2)c2ccc(Cl)cc2)CC1',
        'CN(C)CC[C@H](CSc1ccccc1)Nc1ccc(cc1S(=O)(=O)C(F)(F)F)S(=O)(=O)NC(=O)c1ccc(cc1)N1CCN(CC2=C(CCC(C)(C)C2)c2ccc(Cl)cc2)CC1',
        'CC(=O)c1c(C)c2cnc(N)nc2n(C2CCCC2)c1=O',
        # --------------------------------------------------------------------------
        # The following are manually added entries
        # --------------------------------------------------------------------------
        # 'CC(=O)N[C@H](C(=O)N1C[C@H](O)C[C@H]1C(=O)NCc1ccc(cc1)-c1scnc1C)C(C)(C)C', # TODO: Double check this one! I manually fixed it in https://www.rcsb.org/chemical-sketch
        # 'Oc1ccc(N2C(=S)N(c3ccc(C#N)c(C(F)(F)F)c3)C(=O)C2(C)C)cc1', #TODO: This is coming from a trained model, so it might be wrong
        # 'N1CCC(n2nc(-c3ccc(Oc4ccccc4)cc3)c3c(N)ncnc32)CC1', #TODO: This is coming from a trained model, so it might be wrong
        # --------------------------------------------------------------------------
        # 'CCNCCOc1ccc(cc1)C(=O)c1c(sc2cc(O)ccc12)-c1ccc(O)cc1',
        # 'CCNCCOc1ccc(Cn2c(c(C)c3cc(O)ccc23)-c2ccc(O)cc2)cc1',
        # 'CCNCCOc1ccc(cc1)[C@H]1[C@H](CCc2cc(O)ccc12)c1ccccc1',
        # 'CC(C)c1cnn2c(NCc3ccc(N)cc3)nc(OC3CCN(C)CC3)nc12',
        # 'CCCS(=O)(=O)Nc1ccc(F)c(c1F)-n1cc(-c2cncnc2)c2nc(ccc12)N(C)C1CCNCC1',
        # 'NC(=O)CC[C@H](NC(=O)[C@@H]1CC[C@@H]2CCNC[C@H](NC(=O)c3cc4cc(ccc4[nH]3)C(F)(F)P(O)(O)=O)C(=O)N12)C(=O)NC(c1ccccc1)c1ccccc1',
        # 'Cn1cc2-c3cc(CS(C)(=O)=O)ccc3N(Cc3c[nH]c(c23)c1=O)c1ncc(F)cc1F',
        # 'C[C@@H]1N=C(c2c(C)c(C)sc2-n2c(C)nnc12)c1ccc(Cl)cc1',
        # 'Cn1cc2-c3cc(CS(C)(=O)=O)ccc3N(Cc3c[nH]c(c23)c1=O)c1ncc(F)cc1F',
    ]

    dictionaries = {}
    for key, value in dictionaries_no_extra.items():
        dictionaries[key] = value.copy()
        if key == 'E3 Binder':
            dictionaries[key] = update_dictionary(dictionaries[key], additional_e3)
        elif key == 'POI Ligand':
            dictionaries[key] = update_dictionary(dictionaries[key], additional_poi)

    final_df = None
    non_mapped_protacs = dictionaries['PROTAC'].copy()

    for max_iter_on_linkers in [0, 50, 200, -1]:
        print(f'Max iterations on linkers: {max_iter_on_linkers}')
        mapped_protacs = map_protacs(
            non_mapped_protacs,
            dictionaries=dictionaries,
            max_iter_on_linkers=max_iter_on_linkers,
            biggest_matches_first=True,
            use_multiprocessing=True,
        )

        # Update the final dataframe
        if final_df is None:
            final_df = mapped_protacs
        else:
            final_df = pd.concat([final_df, mapped_protacs], axis=0)
        final_df.to_csv(os.path.join(data_dir, 'processed', 'mapped_protacs.csv'), index=False)

        # Reporting
        mapped_perc = len(mapped_protacs) / len(non_mapped_protacs)
        total_mapped_perc = len(final_df) / len(dictionaries['PROTAC'])
        print(f'Number of mapped PROTACs:     {len(mapped_protacs)} ({mapped_perc:.2%})')
        print(f'Total num. of mapped PROTACs: {len(final_df)} ({total_mapped_perc:.2%})')
        print('-' * 50)

        # Update all dictionaries with the substructures of the mapped PROTACs
        for substruct in ['POI Ligand', 'E3 Binder', 'Linker with direction']:
            if substruct == 'Linker with direction':
                smiles_list = final_df[f'Linker SMILES with direction'].unique()
            else:
                smiles_list = final_df[f'{substruct} SMILES'].unique()
                smiles_list = [canonize_smiles(remove_dummy_atoms(smiles)) for smiles in smiles_list]
            dictionaries[substruct] = update_dictionary(dictionaries[substruct], smiles_list)

        # Get the non-mapped PROTACs yet
        non_mapped_protacs = dictionaries['PROTAC'][~dictionaries['PROTAC']['SMILES'].isin(final_df['PROTAC SMILES'])].copy()

        final_df.to_csv(os.path.join(data_dir, 'processed', 'mapped_protacs.csv'), index=False)
        non_mapped_protacs[['SMILES', 'ID']].to_csv(os.path.join(data_dir, 'processed', 'non_mapped_protacs.csv'), index=False)

    print('All done!')

if __name__ == '__main__':
    main()