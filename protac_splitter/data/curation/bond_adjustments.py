""" Adjusts amide and ester bonds in PROTAC substructures. """
from typing import Tuple, Dict

from rdkit import Chem

from protac_splitter.chemoinformatics import dummy2query
from protac_splitter.display_utils import display_mol
from protac_splitter.evaluation import check_reassembly


def adjust_amide_bond(
        substruct: Chem.Mol,
        linker: Chem.Mol,
        substruct_attachment_id: int,
        verbose: int = 0,
) -> Tuple[Chem.Mol, Chem.Mol]:
    """
    Adjust the amide bond between the substruct and linker substructure.
    Handles the case when neighboring atoms of the amide bond are dummy atoms, which represent attachment points.
    The linker will be modified with the required additional atoms.

    Args:
        substruct: The substructure of the substruct (protein of interest) that contains the amide bond.
        linker: The linker molecule that connects substruct to the E3 ligase.
        substruct_attachment_id: The attachment point ID in the substruct substructure. E.g., 1 for the POI, as in "[*:1]".

    Returns:
        Tuple[Chem.Mol, Chem.Mol]: The adjusted substruct and linker molecules, in that order.
    """

    # Pseudo-code of the algorithm:
    """
    ```python
    # Check if the amide bond (N-C=O) is in the substructure
    if "N-C(=O)" in substruct:
        if neighbor("N-C(=O)") == "[*:substruct]":
            # If the neighboring atom of the amide bond is a dummy atom, i.e., attachment point
            mark_protac_as_wrong("[PROTAC]")

            # Identify the bond to split, i.e., the nitrogen-carbon bond, and split
            "[*:substruct]-[<optional neighboring atom>]-N-[*:tmp]", "[*:tmp]-C(=O)-[rest of the PROTAC]" = split_PROTAC_at("N-C")

            "[Linker]-N-[*:tmp]" = join("[Linker]-[*:substruct]", "[*:substruct]-N-[*:tmp]")

            rename_attachment_point("[*:tmp]-C(=O)-[rest of the PROTAC]")
            rename_attachment_point("[Linker]-N-[*:tmp]")

        elif neighbor(neighbor("N-C(=O)")) == "[*:substruct]":
            # If the second neighbor of athe amide bond is a dummy atom, i.e., attachment point
            mark_protac_as_wrong("[PROTAC]")

            # Do as above
            # Identify the bond to split, i.e., the nitrogen-carbon bond, and split
            "[*:substruct]-N-[*:tmp]", "[*:tmp]-C(=O)-[rest of the PROTAC]" = split_PROTAC_at("N-C")

            "[Linker]-N-[*:tmp]" = join("[Linker]-[*:substruct]", "[*:substruct]-N-[*:tmp]")

            rename_attachment_point("[*:tmp]-C(=O)-[rest of the PROTAC]")
            rename_attachment_point("[Linker]-N-[*:tmp]")
    ```
    """

    # Convert dummy atoms in substruct to query atoms for substructure search
    query_substruct = dummy2query(substruct)

    # Identify amide bond (N-C=O) in substruct substructure
    amide_pattern = Chem.MolFromSmarts("[NX3][CX3](=[OX1])")
    amide_matches = query_substruct.GetSubstructMatches(amide_pattern, useChirality=True)

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
    dummy_label = 3
    dummy_labels = [(dummy_label, dummy_label)] # The E3 and substruct will have 1 and 2, so we need a third one
    amid_bond_idx = query_substruct.GetBondBetweenAtoms(nitrogen_idx_found, carbonyl_idx_found).GetIdx()
    fragments = Chem.FragmentOnBonds(query_substruct, [amid_bond_idx], addDummies=True, dummyLabels=dummy_labels)

    # Get the fragments resulting from bond breaking
    try:
        mol_frags = Chem.GetMolFrags(fragments, asMols=True, sanitizeFrags=True)
    except Exception as e:
        print(e)
        return substruct, linker

    # Identify the "[*:substruct][<optional neighboring atom>]N[3*]" fragment, the other one will be the "truncated" substruct
    amide_fragment_pattern = Chem.MolFromSmarts(f"[*:{substruct_attachment_id}][{side_atom}][{dummy_label}*]")
    amide_fragment = None
    substruct_fixed = None

    if verbose:
        print(f'Attachment point: *:{substruct_attachment_id}')
        print('Substruct:')
        display_mol(substruct)
        print('Linker:')
        display_mol(linker)

    for frag in mol_frags:
        if frag.HasSubstructMatch(dummy2query(amide_fragment_pattern)):
            amide_fragment = frag
            if verbose:
                print('Amide fragment:')
                display_mol(frag)
        else:
            if verbose:
                print('Substruct fragment:')
                display_mol(frag)
            substruct_fixed = frag

    if amide_fragment is None or substruct_fixed is None:
        return substruct, linker

    # In order for the function to be used "on linkers", we need to make sure
    # that the amide fragment contains the attachment point of the substruct.
    # If not, there's nothing to do.
    if f'[*:{substruct_attachment_id}]' not in Chem.MolToSmiles(amide_fragment, canonical=True):
        return substruct, linker

    # Rename the "[3*]" attachment point on the amide fragment to "[*:3]"
    amide_fragment_smiles = Chem.MolToSmiles(amide_fragment, canonical=True)
    amide_fragment_smiles = amide_fragment_smiles.replace(f'[{dummy_label}*]', f'[*:{dummy_label}]')
    amide_fragment = Chem.MolFromSmiles(amide_fragment_smiles)

    # Use molzip to join the linker and the fragment at the original attachment point
    linker_fixed = Chem.molzip(linker, amide_fragment)

    # Rename the "[*:3]" attachment point back to the original attachment point on the linker
    linker_fixed_smiles = Chem.MolToSmiles(linker_fixed, canonical=True)
    linker_fixed_smiles = linker_fixed_smiles.replace(f'[*:{dummy_label}]', f'[*:{substruct_attachment_id}]')
    linker_fixed = Chem.MolFromSmiles(linker_fixed_smiles)

    # Rename the "[3*]" attachment point back to the original attachment point on the substruct
    substruct_fixed_smiles = Chem.MolToSmiles(substruct_fixed, canonical=True)
    substruct_fixed_smiles = substruct_fixed_smiles.replace(f'[{dummy_label}*]', f'[*:{substruct_attachment_id}]')
    substruct_fixed = Chem.MolFromSmiles(substruct_fixed_smiles)

    return substruct_fixed, linker_fixed


def adjust_amide_bonds_in_substructs(
        substructs: Dict[str, str],
        protac_smiles: str,
        poi_attachment_id: int = 1,
        e3_attachment_id: int = 2,
) -> Dict[str, str]:
    """ Adjusts the amide bonds in the substructures of a PROTAC. Just a wrapper function to apply it to multiple substructures.

    Args:
        substructs: The substructures of the PROTAC. A dictionary of SMILES with keys 'poi', 'linker', and 'e3'.
        protac_smiles: The SMILES of the PROTAC for checking reassembly.

    Returns:
        The updated substructures dictionary.
    """
    poi_mol = Chem.MolFromSmiles(substructs['poi'])
    e3_mol = Chem.MolFromSmiles(substructs['e3'])
    linker_mol = Chem.MolFromSmiles(substructs['linker'])

    # Fix the amide group on the POI ligand
    poi_mol, linker_mol = adjust_amide_bond(poi_mol, linker_mol, poi_attachment_id)
    poi_smiles = Chem.MolToSmiles(poi_mol, canonical=True)
    linker_smiles = Chem.MolToSmiles(linker_mol, canonical=True)
    e3_smiles = substructs['e3']
    if not check_reassembly(protac_smiles, '.'.join([poi_smiles, linker_smiles, e3_smiles])):
        return substructs

    # Fix the amide group on the E3 binder
    e3_mol, linker_mol = adjust_amide_bond(e3_mol, linker_mol, e3_attachment_id)
    e3_smiles = Chem.MolToSmiles(e3_mol, canonical=True)
    linker_smiles = Chem.MolToSmiles(linker_mol, canonical=True)
    if not check_reassembly(protac_smiles, '.'.join([poi_smiles, linker_smiles, e3_smiles])):
        return substructs

    # Fix the amide group on the linker, E3 side
    linker_mol, e3_mol = adjust_amide_bond(linker_mol, e3_mol, e3_attachment_id)
    e3_smiles = Chem.MolToSmiles(e3_mol, canonical=True)
    linker_smiles = Chem.MolToSmiles(linker_mol, canonical=True)
    if not check_reassembly(protac_smiles, '.'.join([poi_smiles, linker_smiles, e3_smiles])):
        return substructs

    # Fix the amide group on the linker, POI side
    linker_mol, poi_mol = adjust_amide_bond(linker_mol, poi_mol, poi_attachment_id)
    poi_smiles = Chem.MolToSmiles(poi_mol, canonical=True)
    linker_smiles = Chem.MolToSmiles(linker_mol, canonical=True)
    if not check_reassembly(protac_smiles, '.'.join([poi_smiles, linker_smiles, e3_smiles])):
        return substructs

    substructs['poi'] = poi_smiles
    substructs['e3'] = e3_smiles
    substructs['linker'] = linker_smiles
    return substructs


def adjust_ester_bond(
        substruct: Chem.Mol,
        linker: Chem.Mol,
        substruct_attachment_id: int,
        verbose: int = 0,
) -> Tuple[Chem.Mol, Chem.Mol]:
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
    dummy_label = 3
    dummy_labels = [(dummy_label, dummy_label)] # The E3 and substruct will have 1 and 2, so we need a third one
    amid_bond_idx = query_substruct.GetBondBetweenAtoms(oxygen_idx_found, carbonyl_idx_found).GetIdx()
    fragments = Chem.FragmentOnBonds(query_substruct, [amid_bond_idx], addDummies=True, dummyLabels=dummy_labels)

    # Get the fragments resulting from bond breaking
    try:
        mol_frags = Chem.GetMolFrags(fragments, asMols=True, sanitizeFrags=True)
    except Exception as e:
        if verbose:
            print(e)
        return substruct, linker

    # Identify the "[*:substruct][<optional neighboring atom>]N[3*]" fragment, the other one will be the "truncated" substruct
    ester_fragment_pattern = Chem.MolFromSmarts(f"[*:{substruct_attachment_id}][{side_atom}][{dummy_label}*]")
    ester_fragment = None
    substruct_fixed = None

    for frag in mol_frags:
        if frag.HasSubstructMatch(dummy2query(ester_fragment_pattern)):
            ester_fragment = frag
        else:
            substruct_fixed = frag

    if ester_fragment is None or substruct_fixed is None:
        return substruct, linker

    # In order for the function to be used "on linkers", we need to make sure
    # that the ester fragment contains the attachment point of the substruct.
    # If not, there's nothing to do.
    if f'[*:{substruct_attachment_id}]' not in Chem.MolToSmiles(ester_fragment, canonical=True):
        return substruct, linker

    # Rename the "[3*]" attachment point on the amide fragment to "[*:3]"
    ester_fragment_smiles = Chem.MolToSmiles(ester_fragment, canonical=True)
    ester_fragment_smiles = ester_fragment_smiles.replace(f'[{dummy_label}*]', f'[*:{dummy_label}]')
    ester_fragment = Chem.MolFromSmiles(ester_fragment_smiles)

    # Use molzip to join the linker and the fragment at the original attachment point
    linker_fixed = Chem.molzip(linker, ester_fragment)

    # Rename the "[*:3]" attachment point back to the original attachment point on the linker
    linker_fixed_smiles = Chem.MolToSmiles(linker_fixed, canonical=True)
    linker_fixed_smiles = linker_fixed_smiles.replace(f'[*:{dummy_label}]', f'[*:{substruct_attachment_id}]')
    linker_fixed = Chem.MolFromSmiles(linker_fixed_smiles)

    # Rename the "[3*]" attachment point back to the original attachment point on the substruct
    substruct_fixed_smiles = Chem.MolToSmiles(substruct_fixed, canonical=True)
    substruct_fixed_smiles = substruct_fixed_smiles.replace(f'[{dummy_label}*]', f'[*:{substruct_attachment_id}]')
    substruct_fixed = Chem.MolFromSmiles(substruct_fixed_smiles)

    return substruct_fixed, linker_fixed


def adjust_ester_bonds_in_substructs(
        substructs: Dict[str, str],
        protac_smiles: str,
        poi_attachment_id: int = 1,
        e3_attachment_id: int = 2,
) -> Dict[str, str]:
    """ Adjusts the ester bonds in the substructures of a PROTAC. Just a wrapper function to apply it to multiple substructures.

    Args:
        substructs: The substructures of the PROTAC. A dictionary of SMILES with keys 'poi', 'linker', and 'e3'.
        protac_smiles: The SMILES of the PROTAC for checking reassembly.

    Returns:
        The updated substructures dictionary.
    """
    poi_mol = Chem.MolFromSmiles(substructs['poi'])
    e3_mol = Chem.MolFromSmiles(substructs['e3'])
    linker_mol = Chem.MolFromSmiles(substructs['linker'])

    # Fix the amide group on the POI ligand
    poi_mol, linker_mol = adjust_ester_bond(poi_mol, linker_mol, poi_attachment_id)
    poi_smiles = Chem.MolToSmiles(poi_mol, canonical=True)
    linker_smiles = Chem.MolToSmiles(linker_mol, canonical=True)
    e3_smiles = substructs['e3']
    if not check_reassembly(protac_smiles, '.'.join([poi_smiles, linker_smiles, e3_smiles])):
        return substructs

    # Fix the amide group on the E3 binder
    e3_mol, linker_mol = adjust_ester_bond(e3_mol, linker_mol, e3_attachment_id)
    e3_smiles = Chem.MolToSmiles(e3_mol, canonical=True)
    linker_smiles = Chem.MolToSmiles(linker_mol, canonical=True)
    if not check_reassembly(protac_smiles, '.'.join([poi_smiles, linker_smiles, e3_smiles])):
        return substructs

    # Fix the amide group on the linker, E3 side
    linker_mol, e3_mol = adjust_ester_bond(linker_mol, e3_mol, e3_attachment_id)
    e3_smiles = Chem.MolToSmiles(e3_mol, canonical=True)
    linker_smiles = Chem.MolToSmiles(linker_mol, canonical=True)
    if not check_reassembly(protac_smiles, '.'.join([poi_smiles, linker_smiles, e3_smiles])):
        return substructs

    # Fix the amide group on the linker, POI side
    linker_mol, poi_mol = adjust_ester_bond(linker_mol, poi_mol, poi_attachment_id)
    poi_smiles = Chem.MolToSmiles(poi_mol, canonical=True)
    linker_smiles = Chem.MolToSmiles(linker_mol, canonical=True)
    if not check_reassembly(protac_smiles, '.'.join([poi_smiles, linker_smiles, e3_smiles])):
        return substructs

    substructs['poi'] = poi_smiles
    substructs['e3'] = e3_smiles
    substructs['linker'] = linker_smiles
    return substructs