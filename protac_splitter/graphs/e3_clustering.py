from typing import List, Optional, Tuple, Any, Dict
import functools

import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.cluster import AgglomerativeClustering, KMeans
from scipy.stats import skew
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator

from protac_splitter.graphs.utils import get_fp, numpy_to_rdkit_fp
from protac_splitter.chemoinformatics import remove_dummy_atoms


def get_umap_clusters_fp(fp_list: List[str], n_clusters: int = 7) -> np.ndarray:
    """
    Cluster a list of SMILES strings using the umap clustering algorithm.
    From Scaffold Splits Overestimate Virtual Screening Performance
    https://arxiv.org/abs/2406.00873

    Args:
        fp_list (List[str]): List of SMILES strings.
        n_clusters (int): The number of clusters to use for clustering.

    Returns:
        np.ndarray: Array of cluster labels corresponding to each SMILES string in the input list.
    """
    ac = AgglomerativeClustering(n_clusters=n_clusters)
    ac.fit_predict(np.stack(fp_list))
    return ac.labels_

def get_kmeans_clusters_fp(fp_list: List[str], n_clusters: int = 10, return_centroids: bool = False) -> np.ndarray:
    """
    Cluster a list of SMILES strings using the KMeans clustering algorithm.

    Args:
        fp_list (List[str]): List of SMILES strings.
        n_clusters (int): The number of clusters to use for clustering.
        return_centroids (bool): If True, return the cluster centroids as well.

    Returns:
        np.ndarray: Array of cluster labels corresponding to each SMILES string in the input list.
    """
    km = KMeans(n_clusters=n_clusters, n_init='auto', random_state=42, max_iter=1000)
    if return_centroids:
        km.fit(np.stack(fp_list))
        return km.labels_, km.cluster_centers_
    return km.fit_predict(np.stack(fp_list))

def evaluate_clusters(X: np.array, clusters: np.ndarray) -> Dict[str, float]:
    """ Compute clustering metrics and assess cluster size distribution.
    
    Args:
        X (np.array): The input data used for clustering.
        clusters (np.ndarray): The cluster labels for each data point in X.
        
    Returns:
        Dict[str, float]: A dictionary containing various clustering metrics:
            - silhouette: Silhouette score of the clustering.
            - davies_bouldin: Davies-Bouldin index of the clustering.
            - calinski_harabasz: Calinski-Harabasz index of the clustering.
            - avg_cluster_size: Average size of clusters.
            - avg_cluster_data_ratio: Ratio of average cluster size to total data size.
            - std_cluster_size: Standard deviation of cluster sizes.
            - min_cluster_size: Minimum size of clusters.
            - median_cluster_size: Median size of clusters.
            - max_cluster_size: Maximum size of clusters.
            - cluster_size_skewness: Skewness of cluster sizes indicating imbalance.
            - num_clusters: Number of unique clusters found.
    """
    
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

def get_representative_e3s(
    train_df: pd.DataFrame,
    fp_generator: Optional[Any] = None,
    n_clusters_candidates: List[int] = [10, 25, 50, 100, 150],
    e3_column: str = 'E3 Binder SMILES with direction',
) -> Tuple[List[str], List[Any], int, pd.DataFrame]:
    """
    Get representative E3 ligands from a DataFrame of training data by clustering their fingerprints.
    This function computes Morgan fingerprints for unique E3 ligands, clusters them using KMeans and UMAP,
    evaluates the clusters using silhouette, Davies-Bouldin, and Calinski-Harabasz scores, and identifies
    the optimal number of clusters based on these metrics.
    It returns the representative E3 ligands, their fingerprints, the best number of clusters, and a DataFrame
    containing the clustering metrics.
    
    Parameters:
        train_df (pd.DataFrame): DataFrame containing training data with E3 ligands.
        fp_generator (Optional[Any]): RDKit fingerprint generator. If None, a default Morgan fingerprint generator with 1024 bits and radius 6 is used.
        n_clusters_candidates (List[int]): List of candidate numbers of clusters to evaluate.
        e3_column (str): The column name in the DataFrame that contains the E3 ligand SMILES strings.
        
    Returns:
        Tuple[List[str], List[Any], int, pd.DataFrame]: A tuple containing:
            - List of representative E3 ligand SMILES strings.
            - List of RDKit fingerprints corresponding to the representative E3 ligands.
            - The best number of clusters determined from the clustering metrics.
            - DataFrame containing clustering metrics for each candidate number of clusters.
    """
    if e3_column not in train_df.columns:
        raise ValueError(f"Column '{e3_column}' not found in the DataFrame.")

    if fp_generator is None:
        fp_generator = rdFingerprintGenerator.GetMorganGenerator(
            radius=16,
            fpSize=1024,
            useBondTypes=True,
            includeChirality=True,
        )

    fp_dict = {}
    for smi in tqdm(train_df[e3_column].unique()):
        fp = get_fp(remove_dummy_atoms(smi), fp_generator)
        if fp is not None:
            fp_dict[smi] = fp

    fp_list = list(fp_dict.values())
    fp2smiles = {fp.tobytes(): smi for smi, fp in fp_dict.items() if fp is not None}

    centroids_dict = {}
    clusters_dict = {}
    metrics_df = []
    for n_clusters in tqdm(n_clusters_candidates, desc="Clustering and evaluating"):
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

    # Get the sweet spot for the number of clusters
    # Flip davies_bouldin so that all metrics are to be maximized
    metrics_df['-davies_bouldin'] = -metrics_df['davies_bouldin']

    # Normalize all three metrics (by group if you want per algorithm)
    metrics = ['silhouette', '-davies_bouldin', 'calinski_harabasz']
    df_norm = metrics_df.copy()
    df_norm[metrics] = df_norm.groupby('cluster_algorithm')[metrics].transform(
        lambda x: (x - x.min()) / (x.max() - x.min())
    )

    # Measure divergence: standard deviation of normalized metrics per row
    df_norm['metric_divergence'] = df_norm[metrics].std(axis=1)

    # Pick the point with lowest divergence, possibly applying constraints (e.g. not too many clusters)
    sweet_spots = df_norm.loc[df_norm.groupby('cluster_algorithm')['metric_divergence'].idxmin()]

    best_n_clusters = sweet_spots[['num_clusters']]['num_clusters'].unique()[0]

    # Get the centroids of the clusters
    centroids = centroids_dict[best_n_clusters]

    # Get the cluster labels for the centroids
    clusters = np.array(clusters_dict[f'kmeans_n{n_clusters}'])
    representative_e3s = []
    representative_e3s_fp = []
    for label, centroid in enumerate(centroids):
        # Isolate the FP with the same label as the centroid
        fp_cluster = np.array(fp_list)[clusters == label]
        # Get the closest FP for the centroid, use euclidean distance
        distances = np.linalg.norm(fp_cluster - centroid, axis=1)
        closest_fp = np.argmin(distances)
        # To get the SMILES from the FP, use the fp2smiles dictionary
        closest_smiles = fp2smiles[fp_cluster[closest_fp].tobytes()]
        # Append the closest SMILES to the representative_e3s list
        representative_e3s.append(closest_smiles)
        representative_e3s_fp.append(fp_cluster[closest_fp])

    # Convert the representative E3s to RDKit fingerprints
    representative_e3s_fp = [numpy_to_rdkit_fp(fp) for fp in representative_e3s_fp]
    
    return representative_e3s, representative_e3s_fp, best_n_clusters, metrics_df


DEFAULT_REPRESENTATIVE_E3S = [
    'Cc1ncsc1-c1ccc(CNC(=O)[C@@H]2C[C@@H](O)CN2C(=O)CN[*:2])cc1',
    'O=C1CCC(N2Cc3c(N=[*:2])cccc3C2=O)C(=O)N1',
    'CC(=O)NC(C(=O)N1CC(O)CC1C(=O)[*:2])C(C)(C)C',
    'CN[C@@H](C)C(=O)N[C@H](C(=O)N1C[C@@H](Oc2ccccc2[*:2])C[C@H]1C(=O)N[C@@H]1CCCc2ccccc21)C1CCCCC1',
    'Cc1ncsc1-c1ccc(CNC(=O)C2CC(O)CN2C(=O)C(NC(=O)CCO[*:2])C(C)(C)C)cc1',
    'O=C1CCC(N2Cc3ccc([*:2])cc3C2=O)C(=O)N1',
    'COc1ccc(C2=N[C@@H](c3ccc(Cl)cc3)[C@@H](c3ccc(Cl)cc3)N2C(=O)N2CCN(CC(=O)[*:2])C(=O)C2)c(OC(C)C)c1',
    'CC(NC(=O)C1CC(O)CN1C(=O)C(N[*:2])C(C)(C)C)c1ccc(C2CC2)cc1',
    'CCOc1cc(C(C)(C)C)ccc1C1=NC(c2ccc(Cl)cc2)C(c2ccc(Cl)cc2)N1C(=O)N1CCN(CCCC[*:2])CC1',
    'CNC(C)C(=O)NC(C(=O)N1CCCC1c1cncc(C(=O)c2cccc([*:2])c2)c1)C1CCCCC1',
    'CN[C@@H](C)C(=O)N[C@H](C(=O)N1CCC[C@H]1c1nc(C(=O)c2ccc([*:2])cc2)cs1)C1CCCCC1',
    'O=C1CCC(N2C(=O)c3cccc(OC[*:2])c3C2=O)C(=O)N1',
    'CCOc1cc(C(C)(C)C)ccc1C1=NC(c2ccc(Cl)cc2)C(c2ccc(Cl)cc2)N1C(=O)N1CCN([*:2])CC1',
    'Cc1ncsc1-c1ccc(CNC(=O)[C@H]2C[C@H](O)CN2C(=O)C(N[*:2])C(C)(C)C)cc1',
    'Cc1ncsc1-c1ccc([C@H](C)NC(=O)[C@@H]2C[C@@H](O)CN2C(=O)[C@@H](N[*:2])C(C)(C)C)cc1',
    'CN[C@@H](C)C(=O)N[C@H](C(=O)N1CCC[C@H]1c1cncc(C(=O)c2cccc([*:2])c2)c1)C1CCCCC1',
    'Cc1ncsc1-c1ccc(CNC(=O)[C@@H]2C[C@@H](O)CN2C(=O)[C@@H](N[*:2])C(C)(C)C)c(OC2CCNCC2)c1',
    'CNC(C)C(=O)NC(C(=O)N1CC(Oc2ccc([*:2])cc2)CC1C(=O)NC1CCCc2ccccc21)C1CCCCC1',
    'C[C@H](NC(=O)[C@@H]1C[C@@H](O)CN1C(=O)[C@@H](N[*:2])C(C)(C)C)c1ccc(C(C)(C)C)cc1',
    'CNC(C)C(=O)NC(C(=O)N1CCCC1c1nc(C(=O)c2ccc([*:2])cc2)cs1)C1CCCCC1',
    'CC(=O)NC(C(=O)N1CC(O)CC1C(=O)NCc1ccc(-c2scnc2C)cc1[*:2])C(C)(C)C',
    'Cc1ncsc1-c1ccc(CNC(=O)[C@@H]2C[C@@H](O)CN2C(=O)[C@@H](NC(=O)C2(F)CC2)C(C)(C)C)c([*:2])c1',
    'CCOc1cc(C(C)(C)C)ccc1C1=NC(C)(c2ccc(Cl)cc2)C(C)(c2ccc(Cl)cc2)N1C(=O)N1CCN(CC(=O)[*:2])CC1',
    'COc1ccc(C(=O)[*:2])cc1N1CCC(=O)NC1=O',
    'CN[C@@H](C)C(=O)N[C@H](C(=O)N[C@H]1C[C@H]2CC[C@@H]1N(CCc1ccc([*:2])cc1)C2)C1CCCCC1',
    'CNC(C)C(=O)NC(C(=O)N1CC(N[*:2])CC1C(=O)NC1CCCc2ccccc21)C1CCCCC1',
    'CN[C@@H](C)C(=O)N[C@@H](CCCCN[*:2])C(=O)N1CCC[C@H]1C(=O)Nc1snnc1-c1ccccc1',
    'CNC(C)C(=O)NC(C(=O)NC1CC2CCC1N(CCc1cccc([*:2])c1)C2)C1CCCCC1',
    'O=C1CCC(N2C(=O)c3ccc(N[*:2])cc3C2=O)C(=O)N1',
    'CNC(C)C(=O)NC(C(=O)N1CC(NC(=O)CC[*:2])CC1C(=O)Nc1c(F)cccc1F)C(C)(C)C',
    'Cc1ncsc1-c1ccc(CNC(=O)[C@@H]2C[C@@H](O)CN2C(=O)[C@H](N[*:2])C(C)(C)C)cc1',
    'Cc1nc[nH]c1-c1ccc(CNC(=O)C2CC(O)CN2C(=O)C(N[*:2])C(C)(C)C)cc1',
    'Cc1ncsc1-c1ccc(C(C)NC(=O)C2CC(O)CN2C(=O)C(N[*:2])C(C)(C)C)cc1',
    'Cc1ncsc1-c1ccc(CNC(=O)[C@@H]2C[C@@H](O)CN2C(=O)[C@@H](N[*:2])C(C)(C)C)cc1',
    'O=C1CCC(c2cccc([*:2])c2)C(=O)N1',
    'CC(=O)N[C@H](C(=O)N1C[C@@H](O)C[C@@H]1C(=O)N[C@@H](CC(=O)N1CCC([*:2])CC1)c1ccccc1)C(C)C',
    'O=C(CCl)[*:2]',
    'CC[C@@H](NC(=O)[C@@H]1C[C@H](N[*:2])CN1C(=O)[C@@H](NC(=O)[C@H](C)NC)C(C)(C)C)c1ccccc1',
    'CN[C@H](C)C(=O)N[C@@H]1CCO[C@@H]2CC(C)(C)[C@H](C(=O)N[C@@H]3CCCc4cc([*:2])ccc43)N2C1=O',
    'CN[C@@H](C)C(=O)N[C@H](C(=O)N1CCC[C@H]1c1nc(C(=O)c2ccc(F)cc2)cs1)C1CCN(C[*:2])CC1',
    'Cc1ncsc1-c1ccc(CNC(=O)C2CC(O)CN2C(=O)C(N[*:2])C(C)(C)C)cc1',
    'CNC(C)C(=O)NC(CCCCN[*:2])C(=O)N1CCCC1C(=O)Nc1snnc1-c1ccccc1',
    'O=C1CCC(N2C(=O)c3cccc([*:2])c3C2=O)C(=O)O1',
    'COc1ccc(C2=N[C@@H](c3ccc(Cl)cc3)[C@@H](c3ccc(Cl)cc3)N2C(=O)N2CCN(CC(=O)[*:2])C(=O)C2)cc1OC(C)C',
    'Cc1ncsc1-c1ccc(CNC(=O)C2CC(O)CN2C(=O)C(N[*:2])C(C)(C)C)c(OC2CCNCC2)c1',
    'CNC(C)C(=O)NC(C(=O)N1CCCC1c1cncc(-n2ccc3c(C(=O)[*:2])cccc32)c1)C(C)C',
    'CCN1CCN(Cc2ccc(NC(=O)c3cccc(-c4ccc5nc(N[*:2])sc5n4)c3)cc2C(F)(F)F)CC1',
    'CN[C@@H](C)C(=O)N[C@H](C(=O)N1C[C@@H](NC(=O)CC[*:2])C[C@H]1C(=O)Nc1c(F)cccc1F)C(C)(C)C',
    'CNC(C)C(=O)NC(C(=O)N1CCCC1C(=O)NC(C(=O)[*:2])C(c1ccccc1)c1ccccc1)C1CCCCC1',
    'CC(=O)NCC(C(=O)N1CC(O)CC1C(=O)NC(CC(=O)N1CCC(N2CCC([*:2])CC2)CC1)c1ccccc1)C(C)C',
]


@functools.lru_cache(maxsize=1, typed=False)
def get_representative_e3s_fp(
    e3_list: Optional[List[str]] = None,
    fp_generator: Optional[Any] = None,
    verbose: int = 0,
) -> List[DataStructs.ExplicitBitVect]:
    """
    Generate Morgan fingerprints for a list of E3 ligands. If no list is provided,
    it uses a default list of representative E3 ligands.
    
    Parameters:
        e3_list (Optional[List[str]]): List of SMILES strings for E3 ligands. If None, uses a default list.
        fp_generator (Optional[Any]): RDKit fingerprint generator. If None, a default Morgan fingerprint generator is used.
        
    Returns:
        List[DataStructs.ExplicitBitVect]: List of RDKit Morgan fingerprints for the E3 ligands.
    """
    representative_e3s_fp = []
    if verbose > 0:
        iterable = tqdm(e3_list or DEFAULT_REPRESENTATIVE_E3S, desc="Generating fingerprints for E3 ligands")
    else:
        iterable = e3_list or DEFAULT_REPRESENTATIVE_E3S
    for smi in iterable:
        # Get the Morgan fingerprint for the SMILES string
        fp = get_fp(remove_dummy_atoms(smi), fp_generator, return_np=False)
        if fp is not None:
            representative_e3s_fp.append(fp)
        else:
            print(f"Warning: Invalid SMILES string '{smi}' encountered, skipping.")
    if not representative_e3s_fp:
        raise ValueError("No valid E3 ligands found in the provided list.")
    return representative_e3s_fp
    