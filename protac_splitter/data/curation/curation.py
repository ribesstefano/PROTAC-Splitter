import os
import re
from typing import Any, Dict, Optional, Union, Callable
from joblib import Parallel, delayed
from collections import Counter

from rdkit import Chem
from rdkit.Chem import DataStructs
import pandas as pd
import numpy as np
from tqdm import tqdm

from protac_splitter.chemoinformatics import (
    canonize,
    remove_dummy_atoms,
    canonize_smiles,
    get_mol_id,
    get_substr_match,
)
from protac_splitter.evaluation import check_reassembly
from .substructure_extraction import (
    get_substructure_from_non_perfect_match,
    get_substructs_from_unmapped_e3_poi,
    get_substructs_from_substr_and_linker,
    get_substructs_from_mapped_linker,
    swap_attachment_points,
)
from .bond_adjustments import (
    adjust_amide_bonds_in_substructs,
    adjust_ester_bonds_in_substructs,
)
from .mapping_utils import update_dictionary


def check_substructs_size(
        protac_mol: Chem.Mol,
        substructs: Dict[str, str],
        size_perc_threshold: float = 0.8,
) -> bool:
    """ Check the size of the substructures in the PROTAC. If any of them is too big, return False.
    
    Args:
        protac_mol: The PROTAC molecule.
        substructs: The substructures to check against.
    
    Returns:
        False if any of the substructures is too big. True otherwise.
    """
    num_protac_atoms = protac_mol.GetNumAtoms()
    for key, smiles in substructs.items():
        substruct = Chem.MolFromSmiles(smiles)
        num_substruct_atoms = substruct.GetNumAtoms()
        if num_substruct_atoms / num_protac_atoms > size_perc_threshold:
            # print(f'Error: {key.upper()} is too big in the PROTAC ({num_substruct_atoms} / {num_protac_atoms} = {num_substruct_atoms / num_protac_atoms:.2%} > {size_perc_threshold:.2%})')
            # display_mol(substruct)
            # display_mol(protac_mol)
            return False
    return True


def check_linker_similarity(
        linker_smiles: str,
        pois: Union[pd.DataFrame, str],
        e3s: Union[pd.DataFrame, str],
        linkers: Optional[Union[pd.DataFrame, str]] = None,
        pois_similarity_threshold: float = 0.7,
        e3s_similarity_threshold: float = 0.7,
        linkers_similarity_threshold: float = 0.6,
        morgan_fp_generator: Optional[Callable] = None,
) -> bool:
    """ Check the similarity of the linker with all the matching POIs and E3s. If too similar to any of them, return False.
    
    Args:
        linker_smiles: The linker SMILES.
        pois: The POI ligands. Must have a 'FP' column with the Morgan fingerprints.
        e3s: The E3 binders. Must have a 'FP' column with the Morgan fingerprints.
        pois_similarity_threshold: The similarity threshold for the POIs.
        e3s_similarity_threshold: The similarity threshold for the E3s.
        morgan_fp_generator: The Morgan fingerprint generator.

    Returns:
        False if the linker is too similar to any of the POIs or E3s. True otherwise.
    """

    # Get the linker fingerprint
    if morgan_fp_generator is None:
        morgan_fp_generator = Chem.rdFingerprintGenerator.GetMorganGenerator(
            radius=2,
            fpSize=2048,
            useBondTypes=True,
            includeChirality=True,
        )

    linker = Chem.MolFromSmiles(linker_smiles)
    linker_fp = morgan_fp_generator.GetFingerprint(linker)

    # Check the similarity of the linker with the POIs and E3s (use BulkTanimotoSimilarity)
    if isinstance(e3s, str):
        # Create a one-element list with the E3 fingerprint
        e3s_fps = [morgan_fp_generator.GetFingerprint(Chem.MolFromSmiles(e3s))]
    else:
        e3s_fps = e3s['FP'].to_list()
    e3s_similarities = DataStructs.BulkTanimotoSimilarity(linker_fp, e3s_fps)
    if (np.array(e3s_similarities) > e3s_similarity_threshold).any():
        print(f'WARNING: Linker {linker_smiles} is too similar to an E3 binder')
        # display_mol(linker)
        # display_mol(Chem.MolFromSmiles(e3s[e3s_similarities.argmax()]))
        return False

    # Check if the linker is similar to any of the POIs or E3s
    if isinstance(pois, str):
        # Create a one-element list with the POI fingerprint
        pois_fps = [morgan_fp_generator.GetFingerprint(Chem.MolFromSmiles(pois))]
    else:
        pois_fps = pois['FP'].to_list()
    pois_similarities = DataStructs.BulkTanimotoSimilarity(linker_fp, pois_fps)
    if (np.array(pois_similarities) > pois_similarity_threshold).any():
        # print(f'Error: Linker {linker_smiles} is too similar to a POI ligand')
        # display_mol(linker)
        # display_mol(Chem.MolFromSmiles(pois[pois_similarities.argmax()]))
        return False
    
    # Check if the linker is NOT similar to any of the linkers
    if linkers is not None:
        if isinstance(linkers, str):
            # Create a one-element list with the linker fingerprint
            linkers_fps = [morgan_fp_generator.GetFingerprint(Chem.MolFromSmiles(linkers))]
        else:
            linkers_fps = linkers['FP'].to_list()
        linkers_similarities = DataStructs.BulkTanimotoSimilarity(linker_fp, linkers_fps)
        if not (np.array(linkers_similarities) > linkers_similarity_threshold).all():
            print(f'WARNING: Linker {linker_smiles} is too similar to a linker')
            # display_mol(linker)
            # display_mol(Chem.MolFromSmiles(linkers[linkers_similarities.argmax()]))
            return False

    return True


def check_substructs_similarity(
        protac: Union[np.ndarray, str, Chem.Mol],
        substructs: Dict[str, str],
        similarity_threshold: float = 0.7,
        similarity_thresholds : Dict[str, float] = None,
        morgan_fp_generator: Optional[Callable] = None,
) -> bool:
    """ Check the similarity of the PROTAC with the substructures. If too similar to any of them, return False.

    Args:
        protac: The PROTAC molecule or its SMILES.
        substructs: The substructures to check against.
        similarity_threshold: The similarity threshold.
        similarity_thresholds: The similarity thresholds for the substructures.
        morgan_fp_generator: The Morgan fingerprint generator.
    
    Returns:
        False if the PROTAC is too similar to any of the substructures. True otherwise.
    """

    if morgan_fp_generator is None:
        morgan_fp_generator = Chem.rdFingerprintGenerator.GetMorganGenerator(
            radius=2,
            fpSize=2048,
            useBondTypes=True,
            includeChirality=True,
        )
    
    if isinstance(protac, str):
        protac = Chem.MolFromSmiles(protac)
        protac_fp = morgan_fp_generator.GetFingerprint(protac)
    elif isinstance(protac, Chem.Mol):
        protac_fp = morgan_fp_generator.GetFingerprint(protac)
    else:
        protac_fp = protac
    
    for key, smiles in substructs.items():
        substr_fp = morgan_fp_generator.GetFingerprint(Chem.MolFromSmiles(smiles))
        threshold = similarity_thresholds[key] if similarity_thresholds is not None else similarity_threshold
        if DataStructs.TanimotoSimilarity(protac_fp, substr_fp) > threshold:
            print(f'WARNING: {key.upper()} is too similar to the PROTAC, similarity: {DataStructs.TanimotoSimilarity(protac_fp, substr_fp):.4f} > {threshold}')
            # display_mol(Chem.MolFromSmiles(smiles))
            return False
    
    return True


def get_split_row(
        row: pd.Series,
        substructs: Dict[str, str],
        poi_smiles_no_dummy: Optional[str] = None,
        e3_smiles_no_dummy: Optional[str] = None,
) -> Dict[str, Any]:
    """ Update the fields of a row with the substructures and their IDs.

    Args:
        row: The input row.
        dictionaries: The dictionaries containing the substructures.
        substructs: The substructures found in the PROTAC.
        poi_smiles_no_dummy: The POI ligand SMILES without the dummy atoms.
        e3_smiles_no_dummy: The E3 binder SMILES without the dummy atoms.
        update_dict_if_ids_not_found: Whether to update the dictionary if the substructure IDs are not found.

    Returns:
        The updated row.
    """
    mapped_row = {}
    mapped_row['PROTAC SMILES'] = canonize_smiles(row['SMILES'])
    mapped_row['POI Ligand SMILES with direction'] = substructs['poi']
    mapped_row['E3 Binder SMILES with direction'] = substructs['e3']
    mapped_row['Linker SMILES with direction'] = substructs['linker']
    mapped_row['POI Ligand SMILES'] = remove_dummy_atoms(substructs['poi']) if poi_smiles_no_dummy is None else poi_smiles_no_dummy
    mapped_row['E3 Binder SMILES'] = remove_dummy_atoms(substructs['e3']) if e3_smiles_no_dummy is None else e3_smiles_no_dummy
    mapped_row['Linker SMILES'] = remove_dummy_atoms(substructs['linker'])

    # Get the IDs and update the dictionaries with new substructures
    mapped_row['PROTAC ID'] = get_mol_id(mapped_row['PROTAC SMILES'])
    mapped_row['POI Ligand ID'] = get_mol_id(mapped_row['POI Ligand SMILES with direction'])
    mapped_row['E3 Binder ID'] = get_mol_id(mapped_row['E3 Binder SMILES with direction'])
    mapped_row['Linker ID'] = get_mol_id(mapped_row['Linker SMILES with direction'])

    return mapped_row


def split_single_protac(
        row: pd.Series,
        dictionaries: Dict[str, pd.DataFrame],
        biggest_matches_first: bool = True,
        max_iter_on_linkers: int = 0,
        split_with_substr_and_linker_matching: bool = False,
        similarity_threshold: float = 0.65,
        morgan_radius: Optional[int] = None,
        morgan_fp_size: Optional[int] = None,
        morgan_fp_generator: Optional[Callable] = None,
        poi_attachment_id: int = 1,
        e3_attachment_id: int = 2,
) -> Dict[str, Any]:
    """ Map a PROTAC row to the substructures in the dictionaries.

    Args:
        row: The input row, containing the PROTAC SMILES, ID, and molecule.
        dictionaries: The dictionaries containing the substructures.
        biggest_matches_first: Whether to sort the matches by the number of atoms in the molecule.
        max_iter_on_linkers: The maximum number of iterations to perform on the linkers.

    Returns:
        The mapped row. None if the mapping was not successful.
    """
    # # Disable the RDKit warnings that pop up when RDKit fails to create molecules
    # # NOTE: The following is done to avoid warning messages during multiprocessing
    # RDLogger.DisableLog("rdApp.*")
    # blocker = rdBase.BlockLogs()

    protac_smiles = row['SMILES']
    protac_mol = row['Molecule']

    if morgan_fp_generator is None:
        morgan_radius = 2 if morgan_radius is None else morgan_radius
        morgan_fp_size = 2048 if morgan_fp_size is None else morgan_fp_size
        morgan_fp_generator = Chem.rdFingerprintGenerator.GetMorganGenerator(
            radius=morgan_radius,
            fpSize=morgan_fp_size,
            useBondTypes=True,
            includeChirality=True,
        )
    else:
        morgan_radius = 'None'
        morgan_fp_size = 'None'
    protac_fp = morgan_fp_generator.GetFingerprint(protac_mol)

    notes = f'({max_iter_on_linkers=})({split_with_substr_and_linker_matching=})({morgan_radius=})({morgan_fp_size=})'

    # Get all substructure matches in the POI dictionary
    # poi_matches = dictionaries['POI Ligand']['Molecule'].apply(lambda x: get_substr_match(protac_mol, x, num_allowed_fragments=1))
    poi_matches = dictionaries['POI Ligand']['Molecule'].apply(lambda x: protac_mol.HasSubstructMatch(x))
    pois = dictionaries['POI Ligand'][poi_matches].drop_duplicates(subset=['SMILES'])

    # Get all substructure matches in the E3 dictionary
    # e3_matches = dictionaries['E3 Binder']['Molecule'].apply(lambda x: get_substr_match(protac_mol, x, num_allowed_fragments=1))
    e3_matches = dictionaries['E3 Binder']['Molecule'].apply(lambda x: protac_mol.HasSubstructMatch(x))
    e3s = dictionaries['E3 Binder'][e3_matches].drop_duplicates(subset=['SMILES'])

    # # Sort the matches by the number of atoms in the molecule
    # ascending = False if biggest_matches_first else True
    # pois = pois.sort_values(by='Molecule', key=lambda s: s.apply(lambda m: m.GetNumAtoms()), ascending=True)
    # e3s = e3s.sort_values(by='Molecule', key=lambda s: s.apply(lambda m: m.GetNumAtoms()), ascending=True)

    # Get the POI median, then re-arrenge the pois dataframe so that the median is the first element
    poi_median = pois['Molecule'].apply(lambda x: x.GetNumAtoms()).median()
    pois = pois.sort_values(by='Molecule', key=lambda s: s.apply(lambda m: m.GetNumAtoms()), ascending=True)
    pois = pois.iloc[np.abs(pois['Molecule'].apply(lambda x: x.GetNumAtoms()) - poi_median).argsort()]

    # Get the E3 median, then re-arrenge the e3s dataframe so that the median is the first element
    e3_median = e3s['Molecule'].apply(lambda x: x.GetNumAtoms()).median()
    e3s = e3s.sort_values(by='Molecule', key=lambda s: s.apply(lambda m: m.GetNumAtoms()), ascending=True)
    e3s = e3s.iloc[np.abs(e3s['Molecule'].apply(lambda x: x.GetNumAtoms()) - e3_median).argsort()]

    # If any of the substructures is not found, get the matching linkers to be
    # used later (do it only once).
    linkers = None
    if len(pois) == 0 or len(e3s) == 0 or split_with_substr_and_linker_matching:
        matches = dictionaries['Linker with direction']['Molecule'].apply(lambda x: get_substr_match(protac_mol, x, num_allowed_fragments=2))
        linkers = dictionaries['Linker with direction'][matches]
        linkers = linkers.sort_values(by='Molecule', key=lambda s: s.apply(lambda m: m.GetNumAtoms()), ascending=False)

    # dummy_attachment_id = 1
    # mapping_found = False
    # for _, linker in linkers.iterrows():
    #     if mapping_found:
    #         break
    #     for _, poi in pois.iterrows():
    #         if mapping_found:
    #             break
    #         for _, e3 in e3s.iterrows():
    #             if mapping_found:
    #                 break
    #             # Get the replace side chain
    #             e3_mapped = Chem.ReplaceSidechains(protac_mol, e3['Molecule'], useChirality=True)
    #             e3_mapped = rename_attachment_id(e3_mapped, dummy_attachment_id, e3_attachment_id)
    #             if e3_mapped is None:
    #                 continue

    #             poi_mapped = Chem.ReplaceSidechains(protac_mol, poi['Molecule'], useChirality=True)
    #             poi_mapped = rename_attachment_id(poi_mapped, dummy_attachment_id, poi_attachment_id)
    #             if poi_mapped is None:
    #                 continue

    #             # Join the substructures as fragments
    #             protac_candidate = canonize('.'.join([linker['SMILES'], e3_mapped, poi_mapped]))
    #             protac_candidate = Chem.MolFromSmiles(protac_candidate)
    #             protac_candidate = canonize(Chem.molzip(protac_candidate))
    #             if check_reassembly(protac_mol, protac_candidate):
    #                 print('Found a match!')
    #                 mapping_found = True

    #                 # substructs = {
    #                 #     'linker': linker['Molecule'],
    #                 #     'e3': e3['Molecule'],
    #                 #     'poi': poi['Molecule'],
    #                 # }
    #                 # mapped_row = get_split_row(row, dictionaries, substructs, poi['SMILES'], e3['SMILES'])
    #                 # mapped_row['Notes'] = 'Obtained from matching E3, POI, and Linker found in dictionaries.'
    #                 # return mapped_row


    # TODO: Add a variable to get mapped ligands even if the checks failed... add a note when it happens
    best_substructs_candidate = None

    # There were matching E3s and matching POIs: try to recover the linker from
    # an unmapped E3 and an unmapped POI.
    if len(e3s) > 0 and len(pois) > 0:
        for _, poi in pois.iterrows():
            for _, e3 in e3s.iterrows():
                additional_notes = '(matching_poi=True)(matching_e3=True)(matching_linker=None)'
                substructs = get_substructs_from_unmapped_e3_poi(protac_smiles, protac_mol, poi['Molecule'], e3['Molecule'])

                # If the substructure is not found, try to get it from a non-perfect match
                if substructs is None:
                    fixed_poi = get_substructure_from_non_perfect_match(protac_mol, poi['Molecule'], poi_attachment_id)
                    fixed_e3 = get_substructure_from_non_perfect_match(protac_mol, e3['Molecule'], e3_attachment_id)
                    fixed_poi = poi['Molecule'] if fixed_poi is None else fixed_poi
                    fixed_e3 = e3['Molecule'] if fixed_e3 is None else fixed_e3
                    if fixed_poi is not None and fixed_e3 is not None:
                        substructs = get_substructs_from_unmapped_e3_poi(protac_smiles, protac_mol, fixed_poi, fixed_e3)
                        if Chem.MolToSmiles(fixed_e3) != e3['SMILES']:
                            additional_notes += '(non_perfect_e3_match=True)'
                        else:
                            additional_notes += '(non_perfect_e3_match=False)'
                        
                        if Chem.MolToSmiles(fixed_poi) != poi['SMILES']:
                            additional_notes += '(non_perfect_poi_match=True)'
                        else:
                            additional_notes += '(non_perfect_poi_match=False)'

                if substructs is not None:
                    size_check = check_substructs_size(protac_mol, substructs, size_perc_threshold=0.7)

                    # Check if the linker is too similar to any of the matching POIs or E3s (use the bulk Tanimoto similarity)
                    if not check_linker_similarity(substructs['linker'], pois, e3s, morgan_fp_generator=morgan_fp_generator, e3s_similarity_threshold=similarity_threshold, pois_similarity_threshold=similarity_threshold):
                        best_substructs_candidate = substructs
                        continue

                    if not size_check and not check_substructs_similarity(protac_fp, substructs, similarity_threshold=similarity_threshold, morgan_fp_generator=morgan_fp_generator):
                        best_substructs_candidate = substructs
                        # display_mol(protac_mol)
                        continue

                    # Fix the bonds close to amide and ester groups, if necessary
                    substructs_copy = substructs.copy()
                    substructs = adjust_amide_bonds_in_substructs(substructs, protac_smiles)
                    # Check and report if any SMILES was changed
                    if substructs['linker'] != substructs_copy['linker']:
                        additional_notes += '(amide_bonds_fixed=True)'
                    else:
                        additional_notes += '(amide_bonds_fixed=False)'

                    substructs_copy = substructs.copy()
                    substructs = adjust_ester_bonds_in_substructs(substructs, protac_smiles)
                    # Check and report if any SMILES was changed
                    if substructs['linker'] != substructs_copy['linker']:
                        additional_notes += '(ester_bonds_fixed=True)'
                    else:
                        additional_notes += '(ester_bonds_fixed=False)'

                    # Add the mapped PROTAC to the final list
                    mapped_row = get_split_row(row, substructs)
                    mapped_row['Notes'] = notes + additional_notes
                    return mapped_row

    # There were no matching POIs, but some E3s and linkers matched: try to
    # recover the E3 from an unmapped POI and a mapped Linker
    if len(e3s) > 0 and split_with_substr_and_linker_matching: # len(pois) == 0 and 
        # NOTE: Only take the largest linker(s) into account
        if max_iter_on_linkers:
            selected_linkers = linkers.iloc[:max_iter_on_linkers, :]
        else:
            selected_linkers = linkers.iloc[:1, :]

        for _, e3 in e3s.iterrows():
            # Adjust the E3 molecule if it is not a perfect match
            e3_mol_fixed = get_substructure_from_non_perfect_match(protac_mol, e3['Molecule'], e3_attachment_id)
            e3_mol = e3['Molecule'] if e3_mol_fixed is None else e3_mol_fixed
            e3_mol = remove_dummy_atoms(e3_mol)
            if Chem.MolToSmiles(e3_mol) != e3['SMILES']:
                non_perfect_e3_match = True
            else:
                non_perfect_e3_match = False

            for _, linker in selected_linkers.iterrows():
                additional_notes = f'(matching_poi=False)(matching_e3=True)(matching_linker=True)({non_perfect_e3_match=})'

                substructs = get_substructs_from_substr_and_linker(
                    protac_smiles=protac_smiles,
                    protac=protac_mol,
                    substr=e3_mol,
                    linker=linker['Molecule'],
                    attachment_id=e3_attachment_id,
                )
                if substructs is not None:
                    size_check = check_substructs_size(protac_mol, substructs, size_perc_threshold=0.7)

                    if not check_linker_similarity(substructs['linker'], substructs['poi'], e3s, morgan_fp_generator=morgan_fp_generator, e3s_similarity_threshold=similarity_threshold, pois_similarity_threshold=similarity_threshold):
                        best_substructs_candidate = substructs
                        continue

                    if not size_check and not check_substructs_similarity(protac_fp, substructs, similarity_threshold=similarity_threshold, morgan_fp_generator=morgan_fp_generator):
                        best_substructs_candidate = substructs
                        # display_mol(protac_mol)
                        continue

                    # Fix the bonds close to amide and ester groups, if necessary
                    substructs_copy = substructs.copy()
                    substructs = adjust_amide_bonds_in_substructs(substructs, protac_smiles)
                    if substructs['linker'] != substructs_copy['linker']:
                        additional_notes += '(amide_bonds_fixed=True)'
                    else:
                        additional_notes += '(amide_bonds_fixed=False)'
                    substructs_copy = substructs.copy()
                    substructs = adjust_ester_bonds_in_substructs(substructs, protac_smiles)
                    if substructs['linker'] != substructs_copy['linker']:
                        additional_notes += '(ester_bonds_fixed=True)'
                    else:
                        additional_notes += '(ester_bonds_fixed=False)'

                    mapped_row = get_split_row(row, substructs)
                    mapped_row['Notes'] = notes + additional_notes
                    return mapped_row
                
                # Swap the attachment points on the linker and try again
                linker_swapped = swap_attachment_points(linker['SMILES'])
                substructs = get_substructs_from_substr_and_linker(
                    protac_smiles=protac_smiles,
                    protac=protac_mol,
                    substr=e3_mol,
                    linker=Chem.MolFromSmiles(linker_swapped),
                    attachment_id=e3_attachment_id,
                )
                additional_notes += '(attachment_points_swapped_in_linker=True)'
                if substructs is not None:

                    size_check = check_substructs_size(protac_mol, substructs, size_perc_threshold=0.7)

                    if not check_linker_similarity(substructs['linker'], substructs['poi'], e3s, morgan_fp_generator=morgan_fp_generator, e3s_similarity_threshold=similarity_threshold, pois_similarity_threshold=similarity_threshold):
                        continue

                    if not size_check and not check_substructs_similarity(protac_fp, substructs, similarity_threshold=similarity_threshold, morgan_fp_generator=morgan_fp_generator):
                        # display_mol(protac_mol)
                        continue

                    # Fix the bonds close to amide and ester groups, if necessary
                    substructs_copy = substructs.copy()
                    substructs = adjust_amide_bonds_in_substructs(substructs, protac_smiles)
                    if substructs['linker'] != substructs_copy['linker']:
                        additional_notes += '(amide_bonds_fixed=True)'
                    else:
                        additional_notes += '(amide_bonds_fixed=False)'
                    substructs_copy = substructs.copy()
                    substructs = adjust_ester_bonds_in_substructs(substructs, protac_smiles)
                    if substructs['linker'] != substructs_copy['linker']:
                        additional_notes += '(ester_bonds_fixed=True)'

                    mapped_row = get_split_row(row, substructs)
                    mapped_row['Notes'] = notes + additional_notes
                    return mapped_row

    # There were no matching E3s, but some POIs and linkers matched: try to
    # recover the POI from an unmapped E3 and a mapped Linker
    if len(pois) > 0 and split_with_substr_and_linker_matching: # and len(e3s) == 0
        # NOTE: Only take the largest linker(s) into account
        if max_iter_on_linkers:
            selected_linkers = linkers.iloc[:max_iter_on_linkers, :]
        else:
            selected_linkers = linkers.iloc[:1, :]

        for _, poi in pois.iterrows():
            poi_mol = get_substructure_from_non_perfect_match(protac_mol, poi['Molecule'], poi_attachment_id)
            poi_mol = poi['Molecule'] if poi_mol is None else poi_mol
            poi_mol = remove_dummy_atoms(poi_mol)
            if Chem.MolToSmiles(poi_mol) != poi['SMILES']:
                non_perfect_poi_match = True
            else:
                non_perfect_poi_match = False

            for _, linker in selected_linkers.iterrows():
                additional_notes = f'(matching_poi=True)(matching_e3=False)(matching_linker=True)({non_perfect_poi_match=})'

                substructs = get_substructs_from_substr_and_linker(
                    protac_smiles=protac_smiles,
                    protac=protac_mol,
                    substr=poi_mol,
                    linker=linker['Molecule'],
                    attachment_id=poi_attachment_id,
                )
                if substructs is not None:
                    size_check = check_substructs_size(protac_mol, substructs, size_perc_threshold=0.7)

                    if not check_linker_similarity(substructs['linker'], pois, substructs['e3'], morgan_fp_generator=morgan_fp_generator, e3s_similarity_threshold=similarity_threshold, pois_similarity_threshold=similarity_threshold):
                        best_substructs_candidate = substructs
                        continue

                    if not size_check and not check_substructs_similarity(protac_fp, substructs, similarity_threshold=similarity_threshold, morgan_fp_generator=morgan_fp_generator):
                        best_substructs_candidate = substructs
                        # display_mol(protac_mol)
                        continue

                    # Fix the bonds close to amide and ester groups, if necessary
                    substructs_copy = substructs.copy()
                    substructs = adjust_amide_bonds_in_substructs(substructs, protac_smiles)
                    if substructs['linker'] != substructs_copy['linker']:
                        additional_notes += '(amide_bonds_fixed=True)'
                    substructs_copy = substructs.copy()
                    substructs = adjust_ester_bonds_in_substructs(substructs, protac_smiles)
                    if substructs['linker'] != substructs_copy['linker']:
                        additional_notes += '(ester_bonds_fixed=True)'

                    mapped_row = get_split_row(row, substructs)
                    mapped_row['Notes'] = notes + additional_notes
                    return mapped_row

                # Swap the attachment points on the linker and try again
                linker_swapped = swap_attachment_points(linker['SMILES'])
                substructs = get_substructs_from_substr_and_linker(
                    protac_smiles=protac_smiles,
                    protac=protac_mol,
                    substr=poi_mol,
                    linker=Chem.MolFromSmiles(linker_swapped),
                    attachment_id=poi_attachment_id,
                )
                additional_notes += '(attachment_points_swapped_in_linker=True)'
                if substructs is not None:

                    size_check = check_substructs_size(protac_mol, substructs, size_perc_threshold=0.7)

                    if not check_linker_similarity(substructs['linker'], substructs['poi'], e3s, morgan_fp_generator=morgan_fp_generator, e3s_similarity_threshold=similarity_threshold, pois_similarity_threshold=similarity_threshold):
                        best_substructs_candidate = substructs
                        continue

                    if not size_check and not check_substructs_similarity(protac_fp, substructs, similarity_threshold=similarity_threshold, morgan_fp_generator=morgan_fp_generator):
                        best_substructs_candidate = substructs
                        # display_mol(protac_mol)
                        continue

                    # Fix the bonds close to amide and ester groups, if necessary
                    substructs_copy = substructs.copy()
                    substructs = adjust_amide_bonds_in_substructs(substructs, protac_smiles)
                    if substructs['linker'] != substructs_copy['linker']:
                        additional_notes += '(amide_bonds_fixed=True)'
                    substructs_copy = substructs.copy()
                    substructs = adjust_ester_bonds_in_substructs(substructs, protac_smiles)
                    if substructs['linker'] != substructs_copy['linker']:
                        additional_notes += '(ester_bonds_fixed=True)'

                    mapped_row = get_split_row(row, substructs)
                    mapped_row['Notes'] = notes + additional_notes
                    return mapped_row


    # Get all substructure matches in the Linker with direction dictionary
    # NOTE: This code is repeated here for performance reasons, to avoid
    # calculating the matches if not needed.
    if linkers is None and max_iter_on_linkers:
        matches = dictionaries['Linker with direction']['Molecule'].apply(lambda x: get_substr_match(protac_mol, x, num_allowed_fragments=2))
        linkers = dictionaries['Linker with direction'][matches]
        # Sort all the matches by the number of atoms in the linker, the biggest first
        linkers = linkers.sort_values(by='Molecule', key=lambda s: s.apply(lambda m: m.GetNumAtoms()), ascending=False)

    # for j, (_, linker) in enumerate(linkers.iterrows()):
    #     additional_notes = '(matching_poi=False)(matching_e3=False)(matching_linker=True)'
    #     if j >= max_iter_on_linkers or max_iter_on_linkers == 0:
    #         return None

    for j in range(max_iter_on_linkers):
        additional_notes = '(matching_poi=False)(matching_e3=False)(matching_linker=True)'
        linker = linkers.iloc[j, :]
        substructs = get_substructs_from_mapped_linker(protac_smiles, linker['SMILES'])

        if substructs is not None:
            if not check_linker_similarity(substructs['linker'], substructs['poi'], substructs['e3'], morgan_fp_generator=morgan_fp_generator, e3s_similarity_threshold=similarity_threshold, pois_similarity_threshold=similarity_threshold):
                best_substructs_candidate = substructs
                continue

            size_check = check_substructs_size(protac_mol, substructs, size_perc_threshold=0.7)
            if not size_check and not check_substructs_similarity(protac_fp, substructs, similarity_threshold=similarity_threshold, morgan_fp_generator=morgan_fp_generator):
                best_substructs_candidate = substructs
                # display_mol(protac_mol)
                continue

            # Fix the bonds close to amide and ester groups, if necessary
            substructs_copy = substructs.copy()
            substructs = adjust_amide_bonds_in_substructs(substructs, protac_smiles)
            if substructs['linker'] != substructs_copy['linker']:
                additional_notes += '(amide_bonds_fixed=True)'
            substructs_copy = substructs.copy()
            substructs = adjust_ester_bonds_in_substructs(substructs, protac_smiles)
            if substructs['linker'] != substructs_copy['linker']:
                additional_notes += '(ester_bonds_fixed=True)'

            if not check_substructs_size(protac_mol, substructs, size_perc_threshold=0.95):
                best_substructs_candidate = substructs
                continue

            mapped_row = get_split_row(row, substructs)
            mapped_row['Notes'] = notes + additional_notes
            return mapped_row

    # If we are here, it means that the substructures found in the above loops
    # failed the similarity checks. We add a note and return the best
    # substructure candidate found.
    if best_substructs_candidate is not None:
        substructs_copy = substructs.copy()
        substructs = adjust_amide_bonds_in_substructs(best_substructs_candidate, protac_smiles)
        if substructs['linker'] != best_substructs_candidate['linker']:
            notes += '(amide_bonds_fixed=True)'
        substructs_copy = substructs.copy()
        substructs = adjust_ester_bonds_in_substructs(substructs, protac_smiles)
        if substructs['linker'] != substructs_copy['linker']:
            notes += '(ester_bonds_fixed=True)'
        mapped_row = get_split_row(row, substructs)
        mapped_row['Notes'] = notes + '(similarity_checks_failed=True)'
        return mapped_row

    return None


def split_protacs(
        protac_df: pd.DataFrame,
        dictionaries: Dict[str, pd.DataFrame],
        max_iter_on_linkers: int = 0,
        split_with_substr_and_linker_matching: bool = False,
        biggest_matches_first: bool = True,
        update_dict_if_ids_not_found: bool = False,
        use_multiprocessing: bool = False,
) -> pd.DataFrame:
    """ Maps PROTACs to their substructures.

    Args:
        protac_df: The input PROTAC dataframe.
        dictionaries: The input dictionaries.
        max_iter_on_linkers: The maximum number of matching linkers to iterate over. If zero, there will be no attempt to match linkers in the dictionary. If negative, iterate over all matched linkers. Default is 0.
        biggest_matches_first: Whether to sort the matches by the number of atoms in the molecule. Default is True.
        update_dict_if_ids_not_found: DEPRECATED. Whether to update the dictionary if the substructure IDs are not found. Default is False.
        use_multiprocessing: Whether to use multiprocessing. Default is False.

    Returns:
        The mapped PROTAC dataframe.
    """
    # if use_multiprocessing:
    #     global split_single_protac

    #     with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
    #         results = pool.map(partial(split_single_protac, dictionaries=dictionaries, biggest_matches_first=biggest_matches_first, max_iter_on_linkers=max_iter_on_linkers), protac_df.copy().to_dict(orient='records'))
        
    #     mapped_protacs = pd.DataFrame(results)
    #     mapped_protacs = mapped_protacs.dropna(subset=['POI Ligand SMILES with direction', 'E3 Binder SMILES with direction', 'Linker SMILES with direction'])
    #     return mapped_protacs

    if use_multiprocessing:
        # TODO: The following does run in parallel, but it gives wrong results. I don't know why. I will have to investigate further.
        results = Parallel(n_jobs=-1)(delayed(split_single_protac)(row, dictionaries=dictionaries, biggest_matches_first=biggest_matches_first, max_iter_on_linkers=max_iter_on_linkers) for _, row in protac_df.iterrows())
        mapped_protacs = pd.DataFrame([r for r in results if r is not None])
        return mapped_protacs

    mapped_protacs = []
    for i, row in (pbar := tqdm(protac_df.iterrows(), total=len(protac_df))):
        pbar.set_description(f'PROTAC n.{i:4d}')

        r = split_single_protac(
            row,
            dictionaries,
            biggest_matches_first=biggest_matches_first,
            max_iter_on_linkers=max_iter_on_linkers,
            split_with_substr_and_linker_matching=split_with_substr_and_linker_matching,
        )
        if r is not None:
            mapped_protacs.append(r)
            tmp = pd.DataFrame(mapped_protacs)
            pbar.set_postfix({'len_mapped': len(tmp), 'perc_mapped': f'{len(tmp) / len(protac_df):.1%}'})

    mapped_protacs = pd.DataFrame(mapped_protacs)
    return mapped_protacs


def parse_notes(notes: str) -> Dict[str, Any]:
    # Define the regex pattern to match key-value pairs within parentheses
    pattern = r'\(([^=]+)=([^\)]+)\)'
    
    # Find all matches in the string
    matches = re.findall(pattern, notes)
    
    # Initialize an empty dictionary to store the parsed key-value pairs
    parsed_dict = {}
    
    # Iterate over the matches and add them to the dictionary
    for key, value in matches:
        # Convert the value to the appropriate type (int, bool, None, or str)
        if value.isdigit():
            parsed_dict[key] = int(value)
        elif value.lower() == 'true':
            parsed_dict[key] = True
        elif value.lower() == 'false':
            parsed_dict[key] = False
        elif value.lower() == 'none':
            parsed_dict[key] = None
        else:
            parsed_dict[key] = value
    
    return parsed_dict


def iterative_protac_splitting(
        dictionaries: Dict[str, pd.DataFrame],
        data_dir: str,
) -> Dict[str, pd.DataFrame]:
    """ Map PROTACs to their substructures in an iterative way.
    
    Args:
        dictionaries: The input dictionaries. The same format as the output of the `update_dictionary` function. 
        data_dir: The directory where the output data is stored.

    Returns:
        The final mapped PROTAC dataframe.
    """
    
    final_df = None
    non_mapped_protacs = dictionaries['PROTAC'].copy()

    start_from_beginning = True # Re-map all PROTACs ignoring loading previous results
    step = -1
    max_iter_on_linkers = 0
    split_with_substr_and_linker_matching = False

    while True:
        if max_iter_on_linkers == -1 or non_mapped_protacs.empty or step >= 50:
            break

        if max_iter_on_linkers == 5:
            max_iter_on_linkers = -1 # Iterate over all linkers

        step += 1
        print('-' * 100)
        print(f'Step n.{step}')
        print(f'Max iterations on linkers: {max_iter_on_linkers}')
        print(f'Map with substr and linker matching: {split_with_substr_and_linker_matching}')
        print('-' * 50)

        step_filename = os.path.join(data_dir, f'mapped_protacs_{step=}.csv')
        final_filename = os.path.join(data_dir, 'mapped_protacs.csv')
        non_mapped_filename = os.path.join(data_dir, 'non_mapped_protacs.csv')

        if os.path.exists(step_filename) and not start_from_beginning:
            # Check if all lines of the file are empty
            with open(step_filename, 'r') as f:
                lines = f.readlines()
                if all([len(line.strip()) == 0 for line in lines]):
                    mapped_protacs = pd.DataFrame()
                else:
                    mapped_protacs = pd.read_csv(step_filename)
        else:
            mapped_protacs = split_protacs(
                non_mapped_protacs,
                dictionaries=dictionaries,
                split_with_substr_and_linker_matching=split_with_substr_and_linker_matching,
                max_iter_on_linkers=max_iter_on_linkers,
                biggest_matches_first=False,
                use_multiprocessing=False,
            )
            # Add a string at the end of the strings in the 'Notes' column
            if not mapped_protacs.empty:
                mapped_protacs['Notes'] = mapped_protacs['Notes'].apply(lambda x: f'{x}({step=})')
            mapped_protacs.to_csv(step_filename, index=False)

        # Update the final dataframe and save it to file
        if final_df is None:
            final_df = mapped_protacs
        else:
            final_df = pd.concat([final_df, mapped_protacs], axis=0).drop_duplicates(subset=['PROTAC SMILES'])
        final_df.to_csv(final_filename, index=False)
        print(f'All mapped PROTACs saved to: {final_filename}')

        # Reporting information
        mapped_perc = len(mapped_protacs) / len(non_mapped_protacs)
        total_mapped_perc = len(final_df) / len(dictionaries['PROTAC'])
        print(f'Number of mapped PROTACs:     {len(mapped_protacs)} ({mapped_perc:.2%})')
        print(f'Total num. of mapped PROTACs: {len(final_df)} ({total_mapped_perc:.2%})')
        print('-' * 50)
        print(final_df['Notes'].value_counts())
        print('-' * 50)

        # Get the non-mapped PROTACs yet and save them to file
        non_mapped_protacs = dictionaries['PROTAC'][~dictionaries['PROTAC']['SMILES'].isin(final_df['PROTAC SMILES'])].copy()
        non_mapped_protacs[['SMILES', 'ID']].to_csv(non_mapped_filename, index=False)
        print(f'Non-mapped PROTACs saved to: {non_mapped_filename}')

        # Control logic for breaking the loop
        if mapped_protacs.empty:
            if max_iter_on_linkers == 0 and not split_with_substr_and_linker_matching:
                split_with_substr_and_linker_matching = True
                continue
            else:
                max_iter_on_linkers += 1
            continue
        else:
            # Using only the linker to map the PROTACs can be unreliable, so if we
            # found new PROTACs, we should the max_iter_on_linkers to zero and try
            # to map the PROTACs again with the newly found substructures.
            max_iter_on_linkers = 0
            split_with_substr_and_linker_matching = False
    
        # Update all dictionaries with the substructures of the mapped PROTACs
        smiles_list = mapped_protacs['Linker SMILES with direction'].unique()
        smiles_list = [canonize(smiles) for smiles in smiles_list]
        dictionaries['Linker with direction'] = update_dictionary(dictionaries['Linker with direction'], smiles_list)

        # Avoid adding POIs that are in the E3 dictionary!
        smiles_list = mapped_protacs['POI Ligand SMILES'].unique()
        smiles_list = [canonize(smiles) for smiles in smiles_list]
        smiles_list = [s for s in smiles_list if s not in dictionaries['E3 Binder']['SMILES'].values]
        
        smiles_list = [remove_dummy_atoms(s) for s in smiles_list if s is not None]

        # Use Tanimoto similarity to prevent adding POIs too similar to E3s
        similarity_threshold = 0.5
        radius = 2
        nbits = 2048
        morgan_fp_generator = Chem.rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=nbits, useBondTypes=True, includeChirality=True)

        pois_to_add = []
        for poi_smiles in smiles_list:
            poi_mol = Chem.MolFromSmiles(poi_smiles)
            poi_fp = morgan_fp_generator.GetFingerprint(poi_mol)
            similarities = DataStructs.BulkTanimotoSimilarity(poi_fp, dictionaries['E3 Binder']['FP'].to_list())
            skip_poi = False
            for sim in similarities:
                if sim >= similarity_threshold:
                    skip_poi = True
                    break
            if not skip_poi:
                pois_to_add.append(poi_smiles)

        dictionaries['POI Ligand'] = update_dictionary(dictionaries['POI Ligand'], smiles_list)

        # Avoid adding E3s that are in the POI dictionary!
        smiles_list = mapped_protacs['E3 Binder SMILES'].unique()
        smiles_list = [canonize(smiles) for smiles in smiles_list]
        smiles_list = [s for s in smiles_list if s not in dictionaries['POI Ligand']['SMILES'].values]
        smiles_list = [remove_dummy_atoms(s) for s in smiles_list if s is not None]
        dictionaries['E3 Binder'] = update_dictionary(dictionaries['E3 Binder'], smiles_list)

        # Save all dictionaries to file
        for key, dictionary in dictionaries.items():
            filename = os.path.join(data_dir, f'dictionary_{key.lower().replace(" ", "_")}.csv')
            dictionary[['ID', 'SMILES']].to_csv(filename, index=False)
            print(f'Dictionary saved to: {filename}')
        
        return dictionaries