
from .gnn.data_augmentation import get_cluster_substructure_attachment_counts

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


def plot_hists(values, binwidth_plural=[0.1, 0.05, 0.01], title=""):

    value_min = min(values)
    value_max = max(values)

    num_plots = len(binwidth_plural)
    fig, axs = plt.subplots(nrows=1, ncols=num_plots, figsize=(
        8*num_plots, 6), sharex=True, sharey=False)
    for ax_id, (binwidth, ax) in enumerate(zip(binwidth_plural, axs)):
        bins = list(np.arange(value_min, value_max + binwidth, binwidth))
        ax.hist(values, range=(value_min, value_max), bins=bins)
        hist_title = f'{title} (Binwidth = {binwidth})'
        ax.set_title(hist_title)
    fig.supylabel('Y value')
    fig.supxlabel('Count')
    plt.show()


def plot_butina_clustering(smi_list, clusters, cutoff, yscale='log', substructure_str=''):
    substructure_unique_smi = list(set(smi_list))
    clusters_sorted = sorted(clusters, key=lambda x: len(x))

    num_substructures_butina = 0
    for cluster in clusters:
        num_substructures_butina += len(cluster)

    if num_substructures_butina != len(substructure_unique_smi):
        raise ValueError(
            f"Number of substructures butina got does not match the number of unique smiles for {substructure_str}")

    clusters_smi = []
    for cluster in clusters_sorted:
        temp_smi_list = []
        for substructure_idx in cluster:
            smi = substructure_unique_smi[substructure_idx]
            temp_smi_list.append(smi)
        temp_smi_set = set(temp_smi_list)
        temp_smi_list = list(temp_smi_set)
        clusters_smi.append(temp_smi_list)

    print(f'Number of clusters: {len(clusters_smi)}')

    clusters_sorted_decending = sorted(clusters, key=lambda x: -len(x))
    clusters_sizes = [len(cluster) for cluster in clusters_sorted_decending]
    fig, ax = plt.subplots()
    x_ticks = list(range(len(clusters_sizes)))
    bar_colors = ['tab:blue']*(len(clusters_sizes))
    ax.bar(x_ticks, clusters_sizes, color=bar_colors)
    ax.set_ylabel('Substructure count')
    ax.set_xlabel('Cluster ID')
    # ax.set_title(f'Butina clustering of {substructure_str} with cutoff at {cutoff}')
    plt.yscale(yscale)
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))

    plt.show()
    return fig


def plot_counts_with_patterns_and_sorting(counts, title, colors=None, ylabel=''):
    """Plots the counts with hatching patterns to differentiate clusters, sorted and without a border, with slightly overlapping bars."""
    # Define colors and hatches
    if colors is None:
        colors = ['darkgreen', 'lightgreen']  # Alternating colors for clusters

    fig, ax = plt.subplots()

    # Sort the counts dictionary
    # Sorting first by cluster ID, then by substructure ID, then by attachment point ID
    sorted_counts = sorted(counts.items(), key=lambda item: item[0])

    # Variables to manage colors and hatches based on cluster
    current_cluster = None
    color_index = -1

    # Increase the bar width by 5% to make them overlap
    bar_width = 1.05  # Slightly increase bar width to ensure overlap and eliminate gaps

    for i, ((cluster_idx, *rest), value) in enumerate(sorted_counts):
        # Change color at the start of a new cluster
        if current_cluster != cluster_idx and colors is not None:
            current_cluster = cluster_idx
            # Cycle through colors
            color_index = (color_index + 1) % len(colors)

        # Draw the bar with specific color, no edge color, and increased width for overlap
        ax.bar(i, value, width=bar_width,
               color=colors[color_index], edgecolor='none')

    # ax.set_title(title)
    plt.xlabel(ylabel)  # , fontsize=18)
    plt.ylabel('Substructure count')

    # if len(sorted_counts)<=5:
    #    fig.gca().xaxis.set_major_locator(mticker.MultipleLocator(1))

    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    plt.tight_layout()
    plt.show()

    return fig


def barplot_from_protac_substructureindices_list(augmented_protac_substructureindices_list, dataset_name='', save_figs=False):

    cluster_counts, substructure_counts, attachment_point_counts = get_cluster_substructure_attachment_counts(
        augmented_protac_substructureindices_list=augmented_protac_substructureindices_list)
    for substructure_type in ["POI", "Linker", "E3"]:
        print(f"{substructure_type} Cluster distribution")
        fig = plot_counts_with_patterns_and_sorting(
            cluster_counts[substructure_type], f'{substructure_type} - Cluster Distribution', colors=["darkgreen"], ylabel="Cluster ID")

        if save_figs:
            fig.savefig(f'fig_substructuredist/{dataset_name}_clustercounts_{substructure_type}.svg',
                        format='svg', dpi=1200, bbox_inches='tight')
            fig.savefig(f'fig_substructuredist/{dataset_name}_clustercounts_{substructure_type}.png',
                        format='png', dpi=1200, bbox_inches='tight')

        # print(f"{substructure_type} Distribution of Substructures without attachmentpoints, within clusters")
        # plot_counts_with_patterns_and_sorting(substructure_counts[substructure_type], f'{substructure_type} - Substructure Distribution', colors=['darkgreen', 'lightgreen'], ylabel="Substructure without attachment ID")

        print(
            f"{substructure_type} Distribution of Substructures with attachmentpoints, within clusters")
        fig = plot_counts_with_patterns_and_sorting(
            attachment_point_counts[substructure_type], f'{substructure_type} - Attachment Point Distribution', colors=['darkgreen', 'lightgreen'], ylabel="Substructure ID")

        if save_figs:
            fig.savefig(f'fig_substructuredist/{dataset_name}_substructurecounts_{substructure_type}.svg',
                        format='svg', dpi=1200, bbox_inches='tight')
            fig.savefig(f'fig_substructuredist/{dataset_name}_substructurecounts_{substructure_type}.png',
                        format='png', dpi=1200, bbox_inches='tight')
