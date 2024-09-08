from .gnn.chemoinformatics import (
    remove_dummy_atom_from_smiles,
    reassemble_protac,
)

import random
from collections import defaultdict
from typing import Literal, Tuple

from tqdm import tqdm


def shuffle_dict_keys(original_dict: dict, seed: int = 42) -> dict:
    """
    Shuffles the keys of a dictionary.

    Args:
        original_dict (dict): The original dictionary.
        seed (int, optional): The seed value for random shuffling. Defaults to 42.

    Returns:
        dict: A new dictionary with the keys shuffled.
    """
    # Extract keys and values
    keys = list(original_dict.keys())
    values = list(original_dict.values())

    random.seed(seed)

    # Shuffle the keys
    random.shuffle(keys)

    # Reconstruct the dictionary with shuffled keys
    shuffled_dict = dict(zip(keys, values))
    return shuffled_dict


def select_combination(clusters_dict: dict, unused_substr: dict) -> tuple:
    """
    Selects a combination of cluster index, substructure index, and attachment point index.

    Args:
        clusters_dict (dict): A dictionary containing clusters of substructures.
        unused_substr (dict): A dictionary indicating which substructures have not been used yet.

    Returns:
        tuple: A tuple containing the selected combination of cluster index, substructure index, and attachment point index.
    """
    selected_combination = []
    for substructure_type, clusters in clusters_dict.items():
        if unused_substr[substructure_type]:
            # Prioritize unused substructures if any
            cluster_idx, substructure_wo_attachment_idx, attachment_point_idx = unused_substr[substructure_type].pop(
            )
        else:
            # Random selection if all substructures have been used
            cluster_idx = random.choice(
                list(clusters.keys()))  # Select a cluster
            cluster = clusters[cluster_idx]
            substructure_wo_attachment_idx = random.choice(
                list(cluster.keys()))  # Select a variant within the cluster
            substructure_wo_attachment = cluster[substructure_wo_attachment_idx]
            attachment_point_idx = random.choice(
                list(substructure_wo_attachment.keys()))  # Select an attachment point

        selected_combination.append(
            (cluster_idx, substructure_wo_attachment_idx, attachment_point_idx))
    return tuple(selected_combination)


def generate_protacs_indices(
        clusters_dict: dict,
        num_protacs: int = None,
        num_protacs_factor: float = 1,
        factor_list: list = [1, 1, 1],
        force_use_all_attachment_points: bool = True,
) -> Tuple[list, dict]:
    """
    Generate indices for augmented PROTACs.

    Args:
        clusters_dict (dict): A dictionary containing clusters of substructures.
        num_protacs (int, optional): The number of augmented PROTACs to generate. Defaults to None.
        num_protacs_factor (float, optional): A factor to multiply the number of PROTACs. Defaults to 1.
        factor_list (list, optional): A list of factors to adjust the number of substructures for each type. Defaults to [1, 1, 1].
        force_use_all_attachment_points (bool, optional): Whether to force the use of all attachment points. Defaults to True.

    Returns:
        tuple: A tuple containing the indices for augmented PROTACs and the updated clusters dictionary.
    """
    used_combinations = set()
    augmented_protac_substr_idx = []
    # Initialize tracking for unused substructures
    if force_use_all_attachment_points:

        clusters_dict_shuffled = clusters_dict.copy()

        for substructure_type, clusters in clusters_dict.items():
            for cluster_idx, cluster in clusters.items():
                for substructure_idx, substructure in cluster.items():
                    clusters_dict_shuffled[substructure_type][cluster_idx][substructure_idx] = shuffle_dict_keys(
                        substructure)  # shuffle attachmentpoints
                clusters_dict_shuffled[substructure_type][cluster_idx] = shuffle_dict_keys(
                    cluster)
            clusters_dict_shuffled[substructure_type] = shuffle_dict_keys(
                clusters)

        unused_substr = {
            substructure_type: {
                (cluster_idx, substructure_idx, attachment_point_idx)
                for cluster_idx, cluster in clusters.items()
                for substructure_idx, substructure in cluster.items()
                for attachment_point_idx in substructure.keys()
            }
            for substructure_type, clusters in clusters_dict_shuffled.items()
        }

        n_substr_w_attach_for_each_type = [len(
            unused_substr[substructure_type]) for substructure_type in clusters_dict_shuffled.keys()]
        # factor_list is to allow to ignore training substructures, when they are mixed with test substructures
        n_substr_w_attach_for_each_type = [
            f * val for f, val in zip(factor_list, n_substr_w_attach_for_each_type)]
        substr_class_w_most_unique_substr_w_attach = n_substr_w_attach_for_each_type.index(
            max(n_substr_w_attach_for_each_type))
        max_n_substr_w_attach = n_substr_w_attach_for_each_type[
            substr_class_w_most_unique_substr_w_attach]

        if num_protacs is None:
            num_protacs = max_n_substr_w_attach
        num_protacs = int(num_protacs * num_protacs_factor)

        if num_protacs < max_n_substr_w_attach:
            print(
                f"Number of PROTACs ({num_protacs}) is less than some substructures max-count: {n_substr_w_attach_for_each_type}")

        clusters_dict_out = clusters_dict_shuffled

    else:
        unused_substr = {substructure_type: ()
                         for substructure_type, clusters in clusters_dict.items()}
        clusters_dict_out = clusters_dict

    while len(augmented_protac_substr_idx) < num_protacs:
        combination = select_combination(clusters_dict_out, unused_substr)
        if combination not in used_combinations:
            augmented_protac_substr_idx.append(combination)
            used_combinations.add(combination)

    return augmented_protac_substr_idx, clusters_dict_out


def get_cluster_substructure_attachment_counts(augmented_protac_substr_idx: list) -> tuple:
    """
    Calculate the counts of cluster, substructure, and attachment point occurrences in a list of protacs.

    Args:
        augmented_protac_substr_idx (list): A list of tuples representing protacs. Each tuple contains 3 elements 
                                            (for POI, linker, E3), where each element is a tuple itself containing 
                                            indices for cluster, substructure, and attachment point.

    Returns:
        tuple: A tuple containing three dictionaries. The first dictionary represents the counts of cluster occurrences 
               for each substructure type. The second dictionary represents the counts of substructure occurrences for 
               each substructure type and cluster. The third dictionary represents the counts of attachment point 
               occurrences for each substructure type, substructure, and cluster.
    """
    # protacs is a list of tuples, where each tuple contains 3 elements (for POI, linker, E3)
    # Each element is a tuple itself, containing indices for cluster, substructure, and attachment point

    # Initialize structures to count occurrences
    cluster_counts = defaultdict(lambda: defaultdict(int))
    substructure_counts = defaultdict(lambda: defaultdict(int))
    attachment_point_counts = defaultdict(lambda: defaultdict(int))

    # Process protacs to fill the structures
    for protac in augmented_protac_substr_idx:
        for idx, substructure_type in enumerate(["POI", "Linker", "E3"]):
            cluster_idx, substructure_idx, attachment_point_idx = protac[idx]

            # Count occurrences
            cluster_counts[substructure_type][(cluster_idx,)] += 1
            substructure_counts[substructure_type][(
                cluster_idx, substructure_idx)] += 1
            attachment_point_counts[substructure_type][(
                cluster_idx, substructure_idx, attachment_point_idx)] += 1

    return cluster_counts, substructure_counts, attachment_point_counts


def generate_protac_from_indices_list(
        clusters_substr_attach_dict: dict,
        augmented_protac_substr_idx: list,
        bond_type: Literal['single', 'rand_uniform'] = 'rand_uniform',
) -> dict:
    """
    Generate PROTAC molecules from a list of indices.

    Args:
        clusters_substr_attach_dict (dict): A dictionary containing attachment points for different substrates.
        augmented_protac_substr_idx (list): A list of tuples representing the indices of substrates for generating PROTAC molecules.
        bond_type (str): The type of bond to use for linking the substrates. Can be 'single' or 'rand_uniform'.

    Returns:
        dict: A dictionary containing the generated PROTAC SMILES and related information.
    """

    smiles_dict = {'PROTAC SMILES': [], 'POI SMILES': [], 'LINKER SMILES': [], 'E3 SMILES': [
    ], 'POI SMILES WITHOUT ATTACHMENT': [], 'LINKER SMILES WITHOUT ATTACHMENT': [], 'E3 SMILES WITHOUT ATTACHMENT': []}

    for augmented_protac_tuple in tqdm(augmented_protac_substr_idx):
        poi_tuple = augmented_protac_tuple[0]
        linker_tuple = augmented_protac_tuple[1]
        e3_tuple = augmented_protac_tuple[2]

        poi_w_attach = clusters_substr_attach_dict["POI"][poi_tuple[0]
                                                          ][poi_tuple[1]][poi_tuple[2]]
        linker_w_attach = clusters_substr_attach_dict["LINKER"][linker_tuple[0]
                                                                ][linker_tuple[1]][linker_tuple[2]]
        e3_w_attach = clusters_substr_attach_dict["E3"][e3_tuple[0]
                                                        ][e3_tuple[1]][e3_tuple[2]]

        protac_smiles, _ = reassemble_protac(
            poi_w_attach,
            linker_w_attach,
            e3_w_attach,
            bond_type=bond_type,
        )

        smiles_dict['PROTAC SMILES'].append(protac_smiles)
        smiles_dict['POI SMILES'].append(poi_w_attach)
        smiles_dict['LINKER SMILES'].append(linker_w_attach)
        smiles_dict['E3 SMILES'].append(e3_w_attach)

        poi_wout_attach = remove_dummy_atom_from_smiles(
            poi_w_attach, output="smiles")
        linker_wout_attach = remove_dummy_atom_from_smiles(
            linker_w_attach, output="smiles")
        e3_wout_attach = remove_dummy_atom_from_smiles(
            e3_w_attach, output="smiles")

        smiles_dict['POI SMILES WITHOUT ATTACHMENT'].append(poi_wout_attach)
        smiles_dict['LINKER SMILES WITHOUT ATTACHMENT'].append(
            linker_wout_attach)
        smiles_dict['E3 SMILES WITHOUT ATTACHMENT'].append(e3_wout_attach)

    return smiles_dict
