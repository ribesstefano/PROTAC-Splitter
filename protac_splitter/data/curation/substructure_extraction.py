import re
from typing import Any, Dict, List, Optional, Union
from collections import Counter

from rdkit import Chem
from rdkit.Chem import Draw

from protac_splitter.chemoinformatics import (
    dummy2query,
    remove_dummy_atoms,
    canonize,
    canonize_smiles,
    GetSubstructMatchesWithTimeout,
)
from protac_splitter.display_utils import (
    safe_display,
    display_mol,
)
from protac_splitter.evaluation import check_reassembly


def get_substructs_from_mapped_linker(
        protac_smiles: str,
        linker_smiles: str,
        e3_attachment_id: int = 2,
        poi_attachment_id: int = 1,
        verbose: int = 0,
) -> Dict[str, str]:
    """ Get the substructures of a PROTAC molecule from a mapped linker SMILES.
    
    This function will return the substructures given a linker with
    directionality, _i.e._, with the two attachment points mapped.
    
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
        frags = Chem.GetMolFrags(frags, asMols=True, sanitizeFrags=True)
    except Exception as e:
        # print(e)
        return None

    if verbose:
        safe_display(protac_mol)
        safe_display(linker_mol)

    # The linker has a map number at its attachment points: the following is a
    # dictionary that maps the atom index of the attachment points to their
    # respective map numbers, i.e., the attachment IDs.
    linker_idx2map = {}
    for atom in linker_mol.GetAtoms():
        if atom.GetAtomicNum() == 0:
            linker_idx2map[atom.GetIdx()] = atom.GetAtomMapNum()
    if verbose:
        print(f'linker indexes: {linker_idx2map}')
        print('-' * 80)

    substructs = {'linker': linker_smiles}

    # After splitting the PROTAC with ReplaceCore, the fragments will have as
    # attachment points the same atom indexes as the linker. We can then use the
    # map numbers from the linker to identify the attachment points in the
    # PROTAC fragments and assign the correct map number to them, i.e., the
    # attachment ID.
    for i, side_mol in enumerate(frags):

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
        linker_attachment_point = linker_idx2map.get(attachment_point, None)

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

    # Canonize the substructures SMILES
    substructs = {k: canonize_smiles(v) for k, v in substructs.items()}

    # Check that the reassembled PROTAC matches the original PROTAC
    if not check_reassembly(protac_smiles, '.'.join(substructs.values())):
        return None

    return substructs


def get_attachment_bonds(mol: Chem.Mol, match_atoms: List[int]) -> List[int]:
    """ Get the bonds to break to separate the substructure from the PROTAC or R-groups molecule.
    
    Args:
        mol: The molecule to break, i.e., the PROTAC.
        match_atoms: The atoms matched in the PROTAC molecule, from the GetSubstructMatch function.
    
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
        poi_attachment_id: int = 1,
        e3_attachment_id: int = 2,
        verbose: int = 0,
        stats: Counter = None,
) -> Optional[Dict[str, str]]:
    """ Get the matches of the POI, E3, and linker in the PROTAC molecule.

    This function will return the substructures given a PROTAC and its unmapped
    POI and E3 ligand substructures, _i.e._, they do not need to have the
    attachment points in their SMILES strings.

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

    # Get the substructure names and canonize their SMILES
    substructs = {'poi': poi_smiles, 'e3': e3_smiles, 'linker': linker_smiles}
    substructs = {k: canonize_smiles(v) for k, v in substructs.items()}    

    if verbose:
        print('Linker:', Chem.MolToSmiles(linker_mol, canonical=True))
        safe_display(linker_mol)

    # Check that the reassembled PROTAC matches the original PROTAC
    if check_reassembly(protac_smiles, '.'.join(substructs.values())):
        return substructs
    
    if stats is not None:
        stats['reassembling failed'] += 1
    if verbose:
        print('ERROR: Reassembling failed')
    return None


def get_substructure_from_non_perfect_match(
        protac_mol: Chem.Mol,
        substruct_mol: Chem.Mol,
        attachment_id: int,
        verbose: int = 0,
) -> Chem.Mol:
    """ Extract the correct substructure from a PROTAC molecule, given the
    SMILES of a wrong substructure resulting in many fragments and matches.
    
    Sometimes the substructure we have is not a _perfect_ substructure of the
    PROTAC, _i.e._, it will generate more than two fragments when trying to
    replace the PROTAC core with it. In this case, this function will perform
    the following steps:

    1. Get the largest fragment by trying to replace the PROTAC core with the
       substructure. This largest fragment will be the other substructure plus
       the linker.
    2. We can now remove the largest fragment from the PROTAC to get the
       "original" substructure without the smaller dangling fragments.

    Args:
        protac_mol (Chem.Mol): The PROTAC molecule.
        substruct_smiles (Chem.Mol): The molecule of the wrong substructure, either the POI ligand or the E3 binder.
        attachment_id (int): The attachment ID.

    Returns:
        Chem.Mol: The extracted substructure molecule. If failing, it will return None.
    """
    # Remove the substructure, even if there are "dangling" fragments, to obtain: PROTAC - substruct = (POI + Linker) + remainders
    linker_and_other_mol = Chem.DeleteSubstructs(protac_mol, substruct_mol, useChirality=True)

    # Get the largest fragment, i.e., the PROTAC - substruct = POI + Linker
    try:
        fragments = Chem.GetMolFrags(linker_and_other_mol, asMols=True)
    except Exception as e:
        if verbose:
            print(e)
        return None
    
    if len(fragments) == 1:
        if verbose:
            print("WARNING. There are no small fragments, there's only one fragment.")

    if not fragments:
        if verbose:
            print('ERROR. No fragments found.')
        return None
    largest_fragment = max(fragments, key=lambda x: x.GetNumAtoms())

    # Get the match of the largest fragment in the PROTAC molecule
    largest_match = protac_mol.GetSubstructMatch(largest_fragment, useChirality=True)

    # Get bonds to break to isolate the substructure, i.e., the opposite of the POI + Linker
    bonds_to_break = get_attachment_bonds(protac_mol, largest_match)

    if len(bonds_to_break) != 1:
        if verbose:
            print(f'ERROR. The bond to break is not a single one: {bonds_to_break}')
        return None

    # Break the bonds to isolate the substructure
    frag_mol_substruct = Chem.FragmentOnBonds(protac_mol, bonds_to_break, addDummies=True, dummyLabels=[(attachment_id, attachment_id)])

    # Get fragments after breaking bonds, i.e., the POI + Linker and the substructure without "remainders"
    try:
        frags = Chem.GetMolFrags(frag_mol_substruct, asMols=True, sanitizeFrags=True)
    except Exception as e:
        if verbose:
            print(e)
        return None

    # Get the smallest between the substructure and the POI+Linker fragments
    substruct_mol = min(frags, key=lambda x: x.GetNumAtoms())
    substruct_smiles = Chem.MolToSmiles(substruct_mol, canonical=True).replace(f'[{attachment_id}*]', f'[*:{attachment_id}]')
    substruct_mol = Chem.MolFromSmiles(canonize(substruct_smiles))

    # Check that the substructure matches in the PROTAC molecule
    if not protac_mol.HasSubstructMatch(dummy2query(substruct_mol), useChirality=True):
        if verbose:
            print('ERROR. Substructure does not match in PROTAC molecule:')
            print('PROTAC molecule:')
            safe_display(protac_mol)
            print('Substructure molecule:')
            safe_display(substruct_mol)
        return None

    return substruct_mol


def get_mapped_substr_from_protac(
        protac: Chem.Mol,
        substr: Chem.Mol,
        attachment_id: int = 1,
) -> Optional[Chem.Mol]:
    """ Get the mapped substructure from a PROTAC molecule and an unmapped substructure.
    
    Args:
        protac: The PROTAC molecule.
        substr: The unmapped substructure.
        attachment_id: The attachment point ID to be assigned to the substructure.
    
    Returns:
        The mapped substructure molecule. None if the function fails to find the substructure.
    """
    num_matches = len(protac.GetSubstructMatches(substr, useChirality=True))
    if num_matches != 1:
        return None
    other_substr = Chem.ReplaceCore(protac, substr, labelByIndex=False, replaceDummies=False)
    if other_substr is None:
        return None
    mapped_substr = Chem.ReplaceCore(protac, remove_dummy_atoms(other_substr), labelByIndex=False, replaceDummies=False)
    if mapped_substr is None:
        return None
    mapped_smiles = Chem.MolToSmiles(mapped_substr, canonical=True)
    # Replace "[1*]" or "[2*]" with the correct attachment point with a regex
    mapped_smiles = re.sub(r'\[(\d+)\*\]', f'[*:{attachment_id}]', mapped_smiles)
    mapped_smiles = canonize(mapped_smiles)
    if mapped_smiles is None:
        return None
    return Chem.MolFromSmiles(mapped_smiles)


def get_substructs_from_substr_and_linker(
        protac_smiles: str,
        protac: Chem.Mol,
        substr: Chem.Mol,
        linker: Chem.Mol,
        attachment_id: int = 1,
        poi_attachment_id: int = 1,
        e3_attachment_id: int = 2,
        verbose: int = 0,
        stats: Counter = None,
) -> Optional[Dict[str, str]]:
    """ Get the substructures of a PROTAC molecule from an unmapped substructure and linker.

    Args:
        protac_smiles: The SMILES of the PROTAC molecule.
        protac: The RDKit molecule object of the PROTAC.
        substr: The RDKit molecule object of the currently matching substructure. Should be UNMAPPED.
        linker: The RDKit molecule object of the linker.
        attachment_id: The attachment point ID of the currently matching substructure.
        verbose: The verbosity level.
    
    Returns:
        Dict: The substructures of the PROTAC molecule. None if the function fails to find the substructures.
    """
    if attachment_id not in [poi_attachment_id, e3_attachment_id]:
        raise ValueError('Attachment ID must be either 1 or 2')
    
    if substr is None:
        return None
    
    subr_matches = list(protac.GetSubstructMatches(substr, useChirality=True))
    if len(subr_matches) != 1:
        if stats is not None:
            stats['multiple substructure matches'] += 1
        if verbose:
            print('ERROR: Multiple substructure matches')
        return None
    subr_match = subr_matches[0]

    mapped_substr = get_mapped_substr_from_protac(protac, substr, attachment_id)
    if mapped_substr is None:
        if stats is not None:
            stats['mapped substructure not found'] += 1
        if verbose:
            print('ERROR: Mapped substructure not found')
        return None

    linker_matches = protac.GetSubstructMatches(remove_dummy_atoms(linker), useChirality=True)
    for linker_match in linker_matches:
        # Check that the intersection between the substructure and the linker
        # matches is only one atom, i.e., the attachment point
        if len(set(subr_match).intersection(linker_match)) == 1:
            linker_match = linker_match
            break

    # Based on the linker match found, remove it from the PROTAC
    emol = Chem.EditableMol(protac)

    # Remove atoms in descending order of their indices
    for idx in sorted(linker_match, reverse=True):
        emol.RemoveAtom(idx)
    # Get the modified molecule
    try:
        protac_fragments = emol.GetMol()
    except Exception as e:
        if verbose:
            print(e)
        return None
    try:
        Chem.SanitizeMol(protac_fragments)
    except Exception as e:
        if verbose:
            print(e)
        return None
    if verbose:
        img = Draw.MolToImage(protac_fragments, highlightAtoms=linker_match, size=(800, 300))
        safe_display(img)

    # Get the fragments after removing the linker
    try:
        fragments = Chem.GetMolFrags(protac_fragments, asMols=True, sanitizeFrags=True)
    except Exception as e:
        if verbose:
            print(e)
        return None

    if len(fragments) != 2:
        if stats is not None:
            stats['multiple fragments after removing the linker'] += 1
        if verbose:
            for frag in fragments:
                safe_display(frag)
            print('ERROR: Multiple fragments after removing the linker')
        return None
    
    substructs = {}
    substructs['linker'] = Chem.MolToSmiles(linker, canonical=True)
    for frag in fragments:
        if frag.HasSubstructMatch(substr, useChirality=True):
            label = 'e3' if attachment_id == e3_attachment_id else 'poi'
            substructs[label] = Chem.MolToSmiles(mapped_substr, canonical=True)
            # Replace "[1*]" or "[2*]" with the correct attachment point with a regex
            substructs[label] = re.sub(r'\[(\d+)\*\]', f'[*:{attachment_id}]', substructs[label])
            if verbose:
                print(f'Found {label.capitalize()} fragment.')
                img = Draw.MolToImage(Chem.MolFromSmiles(substructs[label]), size=(800, 300))
                safe_display(img)
        else:
            label = 'e3' if attachment_id == poi_attachment_id else 'poi'
            other_attachment_id = e3_attachment_id if label == 'e3' else poi_attachment_id

            other_substr = get_mapped_substr_from_protac(protac, frag, other_attachment_id)
            if other_substr is None:
                return None
            substructs[label] = Chem.MolToSmiles(other_substr, canonical=True)

            if verbose:
                print(f'Found {label.capitalize()} fragment.')
                img = Draw.MolToImage(Chem.MolFromSmiles(substructs[label]), size=(800, 300))
                safe_display(img)
    # Canonicalize the SMILES strings
    substructs = {k: canonize(v) for k, v in substructs.items()}

    # Check that the reassembled PROTAC matches the original PROTAC
    if not check_reassembly(protac_smiles, '.'.join(substructs.values()), stats, verbose):
        return None

    return substructs


def swap_attachment_points(
        s: str,
        poi_attachment_id: int = 1,
        e3_attachment_id: int = 2,
) -> str:
    """ Swaps the attachment points in a SMARTS string.
    
    Args:
        s: The input SMARTS string.

    Returns:
        The SMARTS string with the attachment points swapped.
    """
    tmp_e3_id = '^^^^E3^^^^'
    tmp_poi_id = '^^^^POI^^^^'
    s = s.replace(f'[*:{poi_attachment_id}]', f'[*:{tmp_poi_id}]')
    s = s.replace(f'[*:{e3_attachment_id}]', f'[*:{tmp_e3_id}]')
    s = s.replace(f'[*:{tmp_poi_id}]', f'[*:{e3_attachment_id}]')
    s = s.replace(f'[*:{tmp_e3_id}]', f'[*:{poi_attachment_id}]')
    return canonize(s)