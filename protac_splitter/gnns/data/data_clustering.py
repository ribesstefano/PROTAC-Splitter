from .gnn.chemoinformatics import (
    tanimoto_similarity_matrix,
    compute_countMorgFP,
    compute_RDKitFP,
)
from .gnn.data_processing import (
    get_unique_perserve_order,
)
from .gnn.plotting import plot_hists, plot_butina_clustering

from collections import Counter
from itertools import repeat, chain
from typing import Optional, Dict, List

import numpy as np
from rdkit import Chem, DataStructs
from rdkit.ML.Cluster import Butina
from sklearn.cluster import HDBSCAN


def perform_hdbscan(
        distance_matrix: np.array,
        tanimoto_distance_cutoff: float,
        min_cluster_size: int = 2,
        min_samples: int = None,
) -> HDBSCAN:
    """
    Cluster data using the HDBSCAN algorithm on a precomputed distance matrix.

    Args:
        distance_matrix (np.array): Precomputed distance matrix where distances[i, j] is the distance between i and j.
        tanimoto_distance_cutoff (float): The distance threshold. Clusters below this value will be merged.
        min_cluster_size (int, optional): The minimum size of clusters; defaults to 2.
        min_samples (int, optional): The number of samples in a neighborhood for a point to be considered as a core point.

    Returns:
        clusterer (HDBSCAN object): Fitted HDBSCAN model.
    """
    clusterer = HDBSCAN(metric='precomputed',
                        min_cluster_size=min_cluster_size,
                        min_samples=min_samples,
                        cluster_selection_epsilon=tanimoto_distance_cutoff,
                        allow_single_cluster=True)
    clusterer.fit(distance_matrix)
    return clusterer


def get_hdbscan_clusters(
        smi_list: list[str],
        fp_function: callable,
        max_allowed_tanimoto_similarity: float = 0.4,
) -> list[int]:
    """
    Get the HDBSCAN clusters for a list of SMILES strings.

    Args:
        smi_list (list): A list of SMILES strings.
        fp_function (function): A function that calculates the molecular fingerprints for the SMILES strings.
        max_allowed_tanimoto_similarity (float, optional): The maximum allowed Tanimoto similarity between fingerprints.
            Defaults to 0.4.

    Returns:
        list: The cluster labels assigned by HDBSCAN.

    """
    fps = fp_function(smi_list)
    distance_matrix = tanimoto_similarity_matrix(fps=fps, return_distance=True)

    tanimoto_distance_cutoff = 1 - max_allowed_tanimoto_similarity
    clusterer = perform_hdbscan(
        distance_matrix=distance_matrix, tanimoto_distance_cutoff=tanimoto_distance_cutoff)

    return clusterer.labels_


def butina_clustering(smi_list: list[str], cutoff: float, fp_function: callable = compute_countMorgFP) -> list:
    """
    Perform Butina clustering on a list of SMILES strings.

    Args:
        smi_list (list): A list of SMILES strings.
        cutoff (float): The clustering cutoff value.
        fp_function (function, optional): The fingerprint function to use. Defaults to compute_countMorgFP.

    Returns:
        list: A list of clusters, where each cluster is a list of indices referring to the input SMILES strings.

    """
    fps_list = fp_function(smi_list)
    dists = []
    nfps = len(fps_list)
    for i in range(1, nfps):
        sims = DataStructs.BulkTanimotoSimilarity(fps_list[i], fps_list[:i])
        dists.extend([1-x for x in sims])
    clusters = Butina.ClusterData(dists, nfps, cutoff, isDistData=True)
    return clusters


def butina_clustering_substructures_with_fixed_cutoff(
    smi_sets_without_attachment: list,
    cutoff: float,
    plot_top_clusters: Optional[list] = None,
    plot: bool = False,
    yscale: str = 'log',
    test_or_trainval: str = '',
    save_fig: bool = False
):
    """
    Perform Butina clustering on substructures with a fixed cutoff.

    Args:
        smi_sets_without_attachment (list): List of substructure sets without attachment.
        cutoff (float): The cutoff value for clustering.
        plot_top_clusters (list, optional): List of indices of top clusters to plot. Defaults to None.
        plot (bool, optional): Whether to plot the clustering results. Defaults to False.
        yscale (str, optional): The scale of the y-axis in the plot. Defaults to 'log'.
        test_or_trainval (str, optional): The type of data (test or trainval). Defaults to ''.
        save_fig (bool, optional): Whether to save the plot as a figure. Defaults to False.

    Returns:
        dict: A dictionary containing the clusters for each substructure type.
    """
    clusters_for_substructures = {}
    for substructure, substructure_str in zip(smi_sets_without_attachment, ["POI", "LINKER", "E3"]):
        if substructure_str == "LINKER":
            fp_function = compute_RDKitFP
        else:
            fp_function = compute_countMorgFP
        clusters = butina_clustering(
            substructure, cutoff=cutoff, fp_function=fp_function)
        clusters_for_substructures[substructure_str] = clusters

        print(f'{substructure_str}:')
        print(f'Number of unique substructures: {len(substructure)}')
        if plot:
            fig = plot_butina_clustering(
                substructure, clusters, cutoff, yscale)

            if plot_top_clusters is not None:
                clusters_sorted_decending = sorted(
                    clusters, key=lambda x: -len(x))
                for cluster_idx in plot_top_clusters:
                    cluster = clusters_sorted_decending[cluster_idx]
                    for substructure_idx in cluster:
                        smi = substructure[substructure_idx]
                        # display(Chem.MolFromSmiles(smi))

            if save_fig:
                fig.savefig(
                    f'fig_butinaclustering/cutoff{cutoff}_{test_or_trainval}_{substructure_str}.svg', format='svg', dpi=1200, bbox_inches='tight')
                fig.savefig(
                    f'fig_butinaclustering/cutoff{cutoff}_{test_or_trainval}_{substructure_str}.png', format='png', dpi=1200, bbox_inches='tight')

    return clusters_for_substructures


def distribute_clusters(clusters: dict, num_sets_out: int = 3) -> dict:
    """
    Distributes clusters into multiple sets based on their size.

    Args:
        clusters (dict): A dictionary containing cluster labels as keys and their corresponding sizes as values.
        num_sets_out (int, optional): The number of sets to distribute the clusters into. Defaults to 3.

    Returns:
        dict: A dictionary containing the sets of clusters, where the keys are set indices and the values are dictionaries
              with 'cluster_ids' and 'total_size' keys. 'cluster_ids' contains the labels of clusters in the set, and
              'total_size' contains the sum of sizes of clusters in the set.
    """
    # Sort clusters by size in descending order
    sorted_clusters = sorted(
        clusters.items(), key=lambda x: x[1], reverse=True)

    # Initialize three sets with labels and total size
    sets = [{'cluster_ids': [], 'total_size': 0} for _ in range(num_sets_out)]

    # Assign each cluster to the set with the smallest total size
    for label, size in sorted_clusters:
        # Find the set with the minimum total size
        min_set = min(sets, key=lambda x: x['total_size'])
        # Add the cluster to this set
        min_set['cluster_ids'].append(label)
        min_set['total_size'] += size

    sets_dict = {idx: set for idx, set in enumerate(sets)}

    return sets_dict


def get_clusters_substructures_attachments_dict(
    clusters: Dict[str, List[List[int]]],
    smi_set: List[List[str]],
    substructures_dict: Dict[str, Dict[str, List[str]]]
) -> Dict[str, Dict[int, Dict[int, List[str]]]]:
    """
    Generate a dictionary that maps clusters, substructures, and attachments.

    Args:
        clusters (dict): A dictionary containing clusters for different substructures.
        smi_set (list): A list of sets containing substructures without attachments.
        substructures_dict (dict): A dictionary mapping substructures without attachments to substructures with attachments.

    Returns:
        dict: A dictionary that maps clusters, substructures, and attachments.

    """
    # POI: {cluster_idx: {variant_idx: (attachment_idx, ...), ... }, ...}
    clusters_substructures_attachments_dict = {
        'POI': {}, 'LINKER': {}, 'E3': {}}

    for i, substruct_str in enumerate(["POI", "LINKER", "E3"]):
        substructures = clusters[substruct_str]

        for cluster_idx, cluster in enumerate(substructures):
            clusters_substructures_attachments_dict[substruct_str][cluster_idx] = {
            }
            for substructure_idx, substructure_idx_without_attachment in enumerate(cluster):
                clusters_substructures_attachments_dict[substruct_str][cluster_idx][substructure_idx] = {
                }
                substructure_without_attachment = smi_set[
                    i][substructure_idx_without_attachment]
                substructure_plural_with_attachment = substructures_dict[
                    substruct_str][substructure_without_attachment]

                for attachment_idx, substructure_with_attachment in enumerate(substructure_plural_with_attachment):
                    clusters_substructures_attachments_dict[substruct_str][cluster_idx][
                        substructure_idx][attachment_idx] = substructure_with_attachment

    return clusters_substructures_attachments_dict


def sort_by_most_common_elements(y: List[int]) -> List[int]:
    """
    Sorts a list of integers by the most common elements.

    Args:
        y (List[int]): The list of integers to be sorted.

    Returns:
        List[int]: The sorted list of integers.
    """
    return list(chain.from_iterable(repeat(i, c) for i, c in Counter(y).most_common()))


def get_test_clusters(clusterer_labels, test_set_minfraction=0.10, test_set_maxfraction=0.20):
    # TODO: Add docstring

    # test_set_minfraction = 0.10 #stop inclding more clusters when this fraction is succeeded
    # test_set_maxfraction = 0.20 #if the addition of the final cluster goes from below minmum to above this max, undo final addition

    num_mols = len(clusterer_labels)

    cumulative_count_mols_to_test = 0
    cluster_ids_for_test = []
    cluster_ids_for_test_dict = {}

    # count the number of molecules in each cluster
    cluster_counts = Counter(clusterer_labels)

    # want a list of clusters sorted by size
    labels_sorted_mostcommon = sort_by_most_common_elements(clusterer_labels)
    # each mol has a label, the label is a cluster. The number of labels is the number of mol in the cluster
    ids_largest_clusters = get_unique_perserve_order(labels_sorted_mostcommon)

    # itterativly add clusters to test set, until min set test fraction is reached.
    for cluster_id in reversed(ids_largest_clusters):
        num_mol_in_cluster = cluster_counts[cluster_id]
        cumulative_count_mols_to_test += num_mol_in_cluster
        cluster_ids_for_test.append(cluster_id)
        cluster_ids_for_test_dict[cluster_id] = num_mol_in_cluster
        test_set_fraction = cumulative_count_mols_to_test/num_mols
        if test_set_fraction > test_set_minfraction:
            break

    # if max test fraction is succeded, remove last cluster that made it tip over
    if test_set_fraction > test_set_maxfraction:
        cluster_ids_for_test.remove(cluster_id)
        del cluster_ids_for_test_dict[cluster_id]
        cumulative_count_mols_to_test -= num_mol_in_cluster

    print(
        f'test_set_fraction: {round(cumulative_count_mols_to_test/num_mols*100,2)} % ({cumulative_count_mols_to_test} molecules)')
    return cluster_ids_for_test, cluster_ids_for_test_dict


def validate_test_set(test_set, training_set, cutoff, fp_function=compute_countMorgFP, plot_highest_sim_match=False, save_fig=False, save_title=""):
    # TODO: Add better docstring

    # OBS! Does not take into aaccount if cutoff and cutoff_MS are different!
    """
    Validates that no molecule in the test setf has a Tanimoto similarity above the cutoff
    with any molecule in the training set. Returns a list of maximum similarities for test set molecules.
    """

    test_fps = fp_function(test_set)
    train_fps = fp_function(training_set)
    max_similarities = []

    highest_similarity = 0

    below_cutoff = True
    for test_idx, test_fp in enumerate(test_fps):
        similarities = DataStructs.BulkTanimotoSimilarity(test_fp, train_fps)
        max_similarity = max(similarities)
        max_similarities.append(max_similarity)

        if plot_highest_sim_match and max_similarity > highest_similarity:
            highest_similarity = max_similarity
            train_idx = similarities.index(max_similarity)
            highest_similarity_indicies = [test_idx, train_idx]

        if max_similarity > cutoff:
            below_cutoff = False

    # TODO: The following was intended to be run in a notebook, but it should be refactored to work in a script.
    # if plot_highest_sim_match:
    #     test_idx = highest_similarity_indicies[0]
    #     train_idx = highest_similarity_indicies[1]
    #     test_smi = test_set[test_idx]
    #     train_smi = training_set[train_idx]

    #     test_mol = Chem.MolFromSmiles(test_smi)
    #     train_mol = Chem.MolFromSmiles(train_smi)

    #     print(
    #         f"The most similar test molecule (right) {test_smi} to any other training molecule:")
    #     display(test_mol)
    #     print(
    #         f"It is similar to following train molecule (left) {train_smi}, with similarity of {highest_similarity}")
    #     display(train_mol)

    #     if save_fig:
    #         # AllChem.Compute2DCoords(train_mol)
    #         # test_mol = align_molecules_by_coordinates(train_mol, test_mol)

    #         train_svg = draw_molecule_to_svg(
    #             train_mol, scale=30)  # Adjust scale as needed
    #         test_svg = draw_molecule_to_svg(test_mol, scale=30)

    #         filename = f"fig_method/most_similar_train_test_{save_title}"
    #         combine_svgs(svgs=[train_svg, test_svg],
    #                      output_filename=f'{filename}.svg')
    #         display(SVG(filename=f'{filename}.svg'))

    return below_cutoff, max_similarities


def get_test_trainval_smi(smi_list, fp_function, max_allowed_tanimoto_similarity=0.4, test_set_minfraction=0.10, test_set_maxfraction=0.20, binwidth_plural=[5, 1], substructure_type="", save_fig=False):

    # TODO: Add docstring
    # TODO: Shorten variable names

    # get test and trainval clusters
    substructure_wo_attachent_clusters = get_hdbscan_clusters(
        smi_list=smi_list, fp_function=fp_function, max_allowed_tanimoto_similarity=max_allowed_tanimoto_similarity)
    if type(substructure_wo_attachent_clusters) is np.ndarray:
        substructure_wo_attachent_clusters = substructure_wo_attachent_clusters.tolist()
    plot_hists(values=substructure_wo_attachent_clusters,
               binwidth_plural=binwidth_plural, title="HDBSCAN cluster sizes")

    substructure_wo_attachment_test_clusters, substructure_wo_attachment_test_clusters_dict = get_test_clusters(
        substructure_wo_attachent_clusters, test_set_minfraction=test_set_minfraction, test_set_maxfraction=test_set_maxfraction)
    substructure_wo_attachment_trainval_clusters = list(set(
        substructure_wo_attachent_clusters) - set(substructure_wo_attachment_test_clusters))

    # Create an empty dictionary to store the clusters
    map_cluster_label_to_unique_substructures_without_attachment = {}

    # Iterate through both lists simultaneously
    for smi, cluster_label in zip(smi_list, substructure_wo_attachent_clusters):
        if cluster_label not in map_cluster_label_to_unique_substructures_without_attachment:
            map_cluster_label_to_unique_substructures_without_attachment[cluster_label] = [
            ]
        map_cluster_label_to_unique_substructures_without_attachment[cluster_label].append(
            smi)

    substructure_wo_attachment_test_clusters_subsplit_withSize = distribute_clusters(
        substructure_wo_attachment_test_clusters_dict, num_sets_out=3)
    for testset_idx, testset_clusters_and_sizes in substructure_wo_attachment_test_clusters_subsplit_withSize.items():
        print(
            f"Testset {testset_idx} has {testset_clusters_and_sizes['total_size']} entries ({round(testset_clusters_and_sizes['total_size']/len(smi_list)*100,2)} %)")

    # check if any testsplit contains the -1 group (defined as "noise" by HDBSCAN)
    # if -1 exists, then get how many molecules is in the "noise" group
    # subtract that number from the split with -1 and delete the key -1 from that test split
    # Then get all smiles for the -1 group
    # itterativly designate one SMILES to the testsplit with the fewest SMILES
    # Store in the form of a dictionary and then add onto the test_unique_substructure_without_attachment for that split (after test_unique_substructure_without_attachment has been given its smiles)

    # get number of molecules in splits
    split_sizes = [substructure_wo_attachment_test_clusters_subsplit_withSize[i]["total_size"]
                   for i in range(len(substructure_wo_attachment_test_clusters_subsplit_withSize))]
    smi_to_testsplits = {idx: [] for idx in range(len(split_sizes))}

    # check if any testsplit contains the -1 group (defined as "noise" by HDBSCAN)
    for split_idx, split_dict in substructure_wo_attachment_test_clusters_subsplit_withSize.items():
        if -1 in split_dict['cluster_ids']:
            # if -1 exists, then get how many molecules is in the "noise" group
            num_noisy_mols = substructure_wo_attachent_clusters.count(-1)
            # subtract that number from the split with -1 and delete the key -1 from that test split
            substructure_wo_attachment_test_clusters_subsplit_withSize[
                split_idx]["total_size"] -= num_noisy_mols
            split_sizes[split_idx] -= num_noisy_mols
            # remove the group from the split
            substructure_wo_attachment_test_clusters_subsplit_withSize[split_idx]["cluster_ids"].remove(
                -1)

            # Then get all smiles for the -1 group
            noise_substructure_without_attachment = map_cluster_label_to_unique_substructures_without_attachment[-1]

            for smi in noise_substructure_without_attachment:
                min_size = min(split_sizes)
                idx_min_size = split_sizes.index(min_size)
                smi_to_testsplits[idx_min_size].append(smi)
                split_sizes[idx_min_size] += 1

    # dict with split idx to cluster idx
    substructure_wo_attachment_test_clusters_subsplit = {}
    for split_idx, split_dict in substructure_wo_attachment_test_clusters_subsplit_withSize.items():
        substructure_wo_attachment_test_clusters_subsplit[split_idx] = split_dict["cluster_ids"]

    trainval_unique_substructure_without_attachment = [
        map_cluster_label_to_unique_substructures_without_attachment[trainval_cluster] for trainval_cluster in substructure_wo_attachment_trainval_clusters]
    trainval_unique_substructure_without_attachment = [
        smi for sub_list in trainval_unique_substructure_without_attachment for smi in sub_list]  # unpack listlist

    # get smiles
    test_unique_substructure_without_attachment_splits = {}
    for split_idx, substructure_wo_attachment_test_clusters in substructure_wo_attachment_test_clusters_subsplit.items():
        test_unique_substructure_without_attachment = [
            map_cluster_label_to_unique_substructures_without_attachment[test_cluster] for test_cluster in substructure_wo_attachment_test_clusters]
        test_unique_substructure_without_attachment = [
            smi for sub_list in test_unique_substructure_without_attachment for smi in sub_list]  # unpack listlist

        test_unique_substructure_without_attachment += smi_to_testsplits[split_idx]

        # sanity check
        if set(test_unique_substructure_without_attachment) & set(trainval_unique_substructure_without_attachment):
            raise ValueError("A test SMILES is in the training set!")
        else:
            print(
                f"All good, split was successfull for split idx: {split_idx}")

        # plot max tanimoto similarity between the trainval and test sets
        is_valid, max_similarities = validate_test_set(test_set=test_unique_substructure_without_attachment,
                                                       training_set=trainval_unique_substructure_without_attachment,
                                                       cutoff=max_allowed_tanimoto_similarity,
                                                       fp_function=fp_function,
                                                       plot_highest_sim_match=True)
        print(f'count max_similarities: {len(max_similarities)}')

        # TODO: The following plotting code was intended to be run in a notebook, but it should be refactored to work in a script.
        # fig, ax = plt.subplots()
        # # Plotting the maximum similarity for all test molecules, against the trainset
        # plt.hist(max_similarities, bins=30, alpha=0.75)
        # print(
        #     f'All most similar pairs are below cutoff ({max_allowed_tanimoto_similarity}) is {is_valid}')
        # plt.xlabel('Maximum Tanimoto Similarity')
        # plt.ylabel('Frequency')
        # print(f'Distribution of Maximum Tanimoto Similarity (Test vs. Train)')
        # plt.show()
        # if save_fig:
        #     fig.savefig(
        #         f'fig_similarity_test_trainval/simtesttrain{substructure_type}_{split_idx}.svg', format='svg', dpi=1200, bbox_inches='tight')
        #     fig.savefig(
        #         f'fig_similarity_test_trainval/simtesttrain{substructure_type}_{split_idx}.png', format='png', dpi=1200, bbox_inches='tight')

        if is_valid:
            print(
                f"SMILES successfully split into trainval and test set without any pair with similarity above cutoff ({max_allowed_tanimoto_similarity})")
            test_unique_substructure_without_attachment_splits[
                split_idx] = test_unique_substructure_without_attachment

    return {'test': test_unique_substructure_without_attachment_splits, 'trainval': trainval_unique_substructure_without_attachment}
