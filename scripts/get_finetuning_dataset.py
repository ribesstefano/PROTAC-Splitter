import os
import argparse
from typing import List

import pandas as pd
import numpy as np
from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
import seaborn as sns
from matplotlib import pyplot as plt
from scipy.stats import skew
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from datasets import load_dataset


def get_fp(smiles: str, fp_generator) -> np.ndarray:
    """
    Get the Morgan fingerprint of a molecule from its SMILES representation.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    fp = fp_generator.GetFingerprint(mol)
    return np.array(fp)


def get_umap_clusters_fp(fp_list: List[str], n_clusters: int = 7) -> np.ndarray:
    """
    Cluster a list of SMILES strings using the umap clustering algorithm.
    From Scaffold Splits Overestimate Virtual Screening Performance
    https://arxiv.org/abs/2406.00873

    :param fp_list: List of SMILES strings
    :param n_clusters: The number of clusters to use for clustering
    :return: Array of cluster labels corresponding to each SMILES string in the input list.
    """
    ac = AgglomerativeClustering(n_clusters=n_clusters)
    ac.fit_predict(np.stack(fp_list))
    return ac.labels_


def get_kmeans_clusters_fp(fp_list: List[str], n_clusters: int = 10, return_centroids: bool = False) -> np.ndarray:
    """
    Cluster a list of SMILES strings using the KMeans clustering algorithm.

    :param fp_list: List of SMILES strings
    :param n_clusters: The number of clusters to use for clustering
    :return: Array of cluster labels corresponding to each SMILES string in the input list.
    """
    km = KMeans(n_clusters=n_clusters, n_init='auto', random_state=42, max_iter=1000)
    if return_centroids:
        km.fit(np.stack(fp_list))
        return km.labels_, km.cluster_centers_
    return km.fit_predict(np.stack(fp_list))


def evaluate_clusters(X, clusters):
    """Compute clustering metrics and assess cluster size distribution."""
    
    unique_clusters = list(set(clusters))
    
    if len(unique_clusters) < 2:  # Avoid single-cluster issues
        return {
            "silhouette": -1,
            "davies_bouldin": float("inf"),
            "calinski_harabasz": -1,
            "avg_cluster_size": len(X),
            "avg_cluster_data_ratio": 1,
            "std_cluster_size": 0,
            "min_cluster_size": len(X),
            "median_cluster_size": len(X),
            "max_cluster_size": len(X),
            "cluster_size_skewness": 0,
            "num_clusters": 1,
        }

    # Compute standard clustering metrics
    silhouette = silhouette_score(X, clusters)
    davies_bouldin = davies_bouldin_score(X, clusters)
    calinski_harabasz = calinski_harabasz_score(X, clusters)

    # Compute cluster size statistics
    cluster_sizes = [len(np.where(clusters == i)[0]) for i in np.unique(clusters)]
    avg_cluster_size = np.mean(cluster_sizes)
    avg_cluster_data_ratio = avg_cluster_size / len(X)
    std_cluster_size = np.std(cluster_sizes)
    median_cluster_size = np.median(cluster_sizes)
    min_cluster_size = np.min(cluster_sizes)
    max_cluster_size = np.max(cluster_sizes)
    cluster_size_skewness = skew(cluster_sizes, nan_policy="omit")  # Indicates imbalance in cluster sizes

    return {
        "silhouette": silhouette,
        "davies_bouldin": davies_bouldin,
        "calinski_harabasz": calinski_harabasz,
        "avg_cluster_size": avg_cluster_size,
        "avg_cluster_data_ratio": avg_cluster_data_ratio,
        "std_cluster_size": std_cluster_size,
        "min_cluster_size": min_cluster_size,
        "median_cluster_size": median_cluster_size,
        "max_cluster_size": max_cluster_size,
        "cluster_size_skewness": cluster_size_skewness,
        "num_clusters": len(unique_clusters),
    }


def get_text_labels(x):
    e3_smiles = x['E3 Binder SMILES with direction']
    linker_smiles = x['Linker SMILES with direction']
    poi_smiles = x['POI Ligand SMILES with direction']
    return {
        'text': x['PROTAC SMILES'],
        'labels': '.'.join([e3_smiles, linker_smiles, poi_smiles]),
    }


def process_dataframe(df):
    processed = df.apply(get_text_labels, axis=1, result_type='expand')
    return processed[['text', 'labels']]


def main(
    filename_held_out_df: str,
    ds_root: str = 'finetuning_dataset',
    show_plots: bool = False,
):
    held_out_df = pd.read_csv(filename_held_out_df)

    # Get the fingerprints:
    morgan_fp_generator = rdFingerprintGenerator.GetMorganGenerator(
        radius=16,
        fpSize=1024,
        useBondTypes=True,
        includeChirality=True,
    )

    fp_dict = {smi: get_fp(smi, morgan_fp_generator) for smi in held_out_df['PROTAC SMILES']}
    fp_list = list(fp_dict.values())
    fp2smiles = {fp.tobytes(): smi for smi, fp in fp_dict.items() if fp is not None}
    num_smiles = held_out_df['PROTAC SMILES'].nunique()
    num_fp = len(fp2smiles)
    print(f"Number of unique SMILES: {num_smiles:,}")
    print(f"Number of unique fingerprints: {num_fp:,}")
    # Check if the number of unique fingerprints is equal to the number of unique SMILES
    if num_smiles != num_fp:
        print("WARNING. The number of unique fingerprints is not equal to the number of unique SMILES, you may want to adjust how fingerprints are generated.")

    centroids_dict = {}
    clusters_dict = {}
    metrics_df = []
    for n_clusters in tqdm([10, 25, 50], desc="Clustering and evaluating"):
        clusters, centroids = get_kmeans_clusters_fp(fp_list, n_clusters=n_clusters, return_centroids=True)
        metrics = evaluate_clusters(fp_list, clusters)
        clusters_dict[f'kmeans_n{n_clusters}'] = clusters.copy()
        centroids_dict[n_clusters] = centroids.copy()

        metrics['num_clusters'] = n_clusters
        metrics['cluster_algorithm'] = 'kmeans'
        metrics_df.append(metrics.copy())
        
        clusters = get_umap_clusters_fp(fp_list, n_clusters=n_clusters)
        metrics = evaluate_clusters(fp_list, clusters)
        clusters_dict[f'umap_n{n_clusters}'] = clusters.copy()

        metrics['num_clusters'] = n_clusters
        metrics['cluster_algorithm'] = 'umap'
        metrics_df.append(metrics.copy())

    metrics_df = pd.DataFrame(metrics_df)

    if show_plots:
        # Bar-plot silhouette score, Davies-Bouldin score, and Calinski-Harabasz score, hue by the clustering algorithm
        _, axs = plt.subplots(1, 3, figsize=(20, 5))
        sns.barplot(data=metrics_df, x='num_clusters', y='silhouette', hue='cluster_algorithm', ax=axs[0])
        axs[0].set_title('Silhouette Score')
        sns.barplot(data=metrics_df, x='num_clusters', y='davies_bouldin', hue='cluster_algorithm', ax=axs[1])
        axs[1].set_title('Davies-Bouldin Score')
        sns.barplot(data=metrics_df, x='num_clusters', y='calinski_harabasz', hue='cluster_algorithm', ax=axs[2])
        axs[2].set_title('Calinski-Harabasz Score')
        plt.tight_layout()
        for ax in axs:
            ax.grid(axis='y', alpha=0.5)
        plt.show()

    # Create a finetuning dataset, with one different configuration for each cluster
    os.makedirs(ds_root, exist_ok=True)

    for n_clusters, centroids in centroids_dict.items():
        print('-' * 80)
        print(f'Number of clusters: {n_clusters}')
        print('-' * 80)
        # Get the cluster labels for the centroids
        clusters = np.array(clusters_dict[f'kmeans_n{n_clusters}'])
        finetune_samples = []
        for label, centroid in enumerate(centroids):
            # Isolate the FP with the same label as the centroid
            fp_cluster = np.array(fp_list)[clusters == label]
            # Get the closest FP for the centroid, use euclidean distance
            distances = np.linalg.norm(fp_cluster - centroid, axis=1)
            closest_fp = np.argmin(distances)
            # To get the SMILES from the FP, use the fp2smiles dictionary
            closest_smiles = fp2smiles[fp_cluster[closest_fp].tobytes()]
            # Print the closest SMILES
            print(f'Closest FP to centroid n.{label}: {closest_smiles}')
            sample = held_out_df.loc[held_out_df['PROTAC SMILES'] == closest_smiles]
            finetune_samples.append(sample)

        finetune_samples_df = pd.concat(finetune_samples)
        train_df = process_dataframe(finetune_samples_df)

        # Remove the isolated samples from the held-out dataset to get a test set
        test_df = held_out_df[~held_out_df['PROTAC SMILES'].isin(finetune_samples_df['PROTAC SMILES'])].copy()
        test_df = process_dataframe(test_df)

        os.makedirs(os.path.join(ds_root, f'n{n_clusters}'), exist_ok=True)
        train_df.to_csv(os.path.join(ds_root, f'n{n_clusters}', 'train.csv'), index=False)
        test_df.to_csv(os.path.join(ds_root, f'n{n_clusters}', 'test.csv'), index=False)
        print(f'Dataset configuration saved to {os.path.join(ds_root, f"n{n_clusters}")}')

        # Load the dataset to check if it was saved correctly
        ds = load_dataset(ds_root, data_dir=f'n{n_clusters}')
        print(ds)

    print('-' * 80)
    ds = load_dataset(ds_root, data_dir='n10') # Final check
    
    print('-' * 80)
    print('Example of usage of the dataset, the `data_dir` will correspond to different configurations:')
    print('')
    print(f'>>> load_dataset("{ds_root}", data_dir="n10")')
    print(f'>>> load_dataset("{ds_root}", data_dir="n25")')
    print(f'>>> load_dataset("{ds_root}", data_dir="n50")')
    print('-' * 80)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Get finetuning dataset.")
    parser.add_argument(
        "--filename_held_out_df",
        type=str,
        default="data/held_out_df.csv",
        help="Path to the held-out dataset CSV file.",
    )
    parser.add_argument(
        "--ds_root",
        type=str,
        default="finetuning_dataset",
        help="Root directory to save the finetuning dataset.",
    )
    parser.add_argument(
        "--show_plots",
        action="store_true",
        help="Whether to show plots on cluster metrics. Default: False",
    )
    args = parser.parse_args()

    main(args.filename_held_out_df, args.ds_root)