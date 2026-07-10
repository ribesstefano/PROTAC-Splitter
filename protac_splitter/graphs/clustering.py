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
    'O=C1CCC(N2C(=O)c3ccc(C(=O)[*:2])cc3C2=O)C(=O)N1',
    'CC(NC(=O)C1CC(O)CN1C(=O)C(N[*:2])C(C)(C)C)c1ccc(Cl)cc1',
    'CNC(C)C(=O)NC(C(=O)N1CC(Oc2ccccc2[*:2])CC1C(=O)NC1CCCc2ccccc21)C1CCCCC1',
    'O=C1CCN(c2cc(C(=O)[*:2])ccc2Cl)C(=O)N1',
    'CNC(C)C(=O)NC(C(=O)N1CC(N[*:2])CC1C(=O)Nc1c(F)cccc1F)C(C)(C)C',
    'CNC(C)C(=O)NC(C(=O)N1CCCC1c1nc(C(=O)c2ccc(F)cc2)cs1)C1CCN(C[*:2])CC1',
    'CC(C)(C)C(NC(=O)C1(F)CC1)C(=O)N1CC(O)CC1C(=O)[*:2]',
    'CNC(C)C(=O)NC(C(=O)NC1CC2CCC1N(CCc1ccc([*:2])cc1)C2)C1CCCCC1',
    'CC1CN(CC(=O)N2CC(C)(C)c3c2cc(Cc2ccc(F)cc2)c(=O)n3C)C(CN2CCN([*:2])CC2C)CN1',
    'O=C1CCC(N2C(=O)c3ccc([*:2])cc3C2=O)C(=O)N1',
    'CC(=O)NC(C(=O)N1CC(O)CC1C(=O)NC(CC(=O)N1CCC([*:2])CC1)c1ccccc1)C(C)C',
    'CNC(C)C(=O)NC(C(=O)N1Cc2cc([*:2])ccc2CC1C(=O)NC1CCCc2ccccc21)C(C)(C)C',
    'CN[C@H](C)C(=O)N[C@@H](C(=O)N1C[C@H](N[*:2])C[C@@H]1C(=O)N[C@@H]1CCCc2ccccc21)C1CCCCC1',
    'CNC(C)C(=O)NC(C(=O)N1CCCC1c1nc(C(=O)c2ccc([*:2])cc2)cs1)C1CCCCC1',
    'CCOc1cc(C(C)(C)C)ccc1C1=N[C@@](C)(c2ccc(Cl)cc2)[C@@](C)(c2ccc(Cl)cc2)N1C(=O)N1CCN(CC(=O)[*:2])CC1',
    'CNC(C)C(=O)NC(CCCCN[*:2])C(=O)N1CCCC1C(=O)Nc1snnc1-c1ccccc1',
    'COC(=O)CC1C2(C)C3=C(C)C(c4ccoc4[*:2])CC3OC2C2OC(=O)C3(C)C=CC(=O)C1(C)C23',
    'CNC(C)C(=O)NC(C(=O)N1CCCC1c1cncc(-c2ccc(F)c(C(=O)[*:2])c2)c1)C1CCCCC1',
    'CNC(C)C(=O)NC1CN([*:2])CCC2CCC(C(=O)NC3CCCc4ccccc43)N2C1=O',
    'CC(C)C[C@@H](NC(=O)[C@H](O)[C@@H](N)Cc1ccccc1)C(=O)[*:2]',
    'COc1cc(C(=O)[*:2])ccc1NC(=O)C1NC(CC(C)(C)C)C(C#N)(c2ccc(Cl)cc2F)C1c1cccc(Cl)c1F',
    'O=C1CC[C@H](N2Cc3cc([*:2])ccc3C2=O)C(=O)N1',
    'CNC(C)C(=O)NC(C(=O)N1CCCC1c1nc2c(-c3ccccc3)nccc2s1)C1CCN([*:2])CC1',
    'Cc1ncsc1-c1ccc(CNC(=O)C2C(F)C(O)CN2C(=O)C(N[*:2])C(C)(C)C)cc1',
    'CNC(C)C(=O)NC(C(=O)N1CCCC1c1cncc(-n2ccc3c(C(=O)[*:2])cccc32)c1)C(C)C',
    'CC1CN(CC(=O)N2CC(C)(C)c3ncc(Cc4ccc(F)cc4)cc32)C(CN2CCN([*:2])CC2C)CN1',
    'Cc1cc(C(C(=O)N2CC(O)CC2C(=O)[*:2])C(C)C)on1',
    'CC(=O)NCC(C(=O)N1CC(O)CC1C(=O)NC(CC(=O)N1CCC(N2CCC([*:2])CC2)CC1)c1ccccc1)C(C)C',
    'CNC(C)C(=O)NC1CCOC2CC(C)(C)C(C(=O)NC3CCCc4cc([*:2])ccc43)N2C1=O',
    'CCC(NC(=O)C1CC(N[*:2])CN1C(=O)C(NC(=O)C(C)NC)C(C)(C)C)c1ccccc1',
    'CN[C@@H](C)C(=O)N[C@H](C(=O)N1CCC[C@H]1C(=O)N[C@H](C(=O)[*:2])C(c1ccccc1)c1ccccc1)C1CCCCC1',
    'CC(C=CC1=C(C)C(=N[*:2])CCC1(C)C)=CC=CC(C)=CC(=O)O',
]

DEFAULT_REPRESENTATIVE_WHS = [
    'O=C(CCl)N1CCCc2cc([*:1])ccc21',
    'Nc1ncnc2c1c(-c1ccc(Oc3ccccc3)cc1)nn2[C@H]1CC[C@H]([*:1])CC1',
    'NC(Cc1ccc(O)c(O)c1)C(=O)[*:1]',
    'COc1cc([*:1])ccc1Nc1ncc(Cl)c(Nc2ccccc2S(=O)(=O)C(C)C)n1',
    'CC1(C)CCC(c2ccc(Cl)cc2)=C(CN2CCN(c3ccc(C(=O)NS(=O)(=O)c4ccc(NC(CSc5ccccc5)C[*:1])c(S(=O)(=O)C(F)(F)F)c4)cc3)CC2)C1',
    'CC(C)c1cnn2c(NCc3cccc([*:1])c3N(C)C)nc(OC3CCN(C)CC3)nc12',
    'CC1(C[*:1])CCC(c2ccc(Cl)cc2)=C(CN2CCN(c3ccc(C(=O)NS(=O)(=O)c4ccc(NC(CCN5CCOCC5)CSc5ccccc5)c(S(=O)(=O)C(F)(F)F)c4)cc3)CC2)C1',
    'Nc1nc2c(ncn2C2OC(COP(=O)(O)O[*:1])C(O)C2O)c(=O)[nH]1',
    'NC(=O)CCC(NC(=O)C1CCC2CCN([*:1])CC(NC(=O)c3cc4cc(OP(=O)(O)O)ccc4[nH]3)C(=O)N21)C(=O)NC(c1ccccc1)c1ccccc1',
    'CCCNc1nc(Nc2ccc(C#N)cc2)ncc1C#CCCCNC(=O)C(C)N(C)C(=O)CC[*:1]',
    'C=CC(=O)N[C@H](CCC(=O)[*:1])C(=O)Nc1cccc(Nc2ncc(NC(=O)c3cc(NC(=O)c4cccc(C)c4)ccc3C)cn2)c1',
    'C=CC(=O)Nc1cc(Nc2nccc(-c3cn(C)c4ccccc34)n2)c(OC)cc1N(CCN(C)C)C[*:1]',
    'Nc1nc2c(c(=O)[nH]1)[n+](Cc1ccccc1)cn2C1OC(COP(=O)(O)O[*:1])C(O)C1O',
    'COc1ccc(Cl)c(S(=O)(=O)Nc2ccc(-c3nc(OCC4CN([*:1])CCO4)c4cn[nH]c4n3)cc2)c1',
    'COc1cc2ncnc(Nc3ccc(F)c(Cl)c3)c2cc1O[*:1]',
    'CC1Cc2c([nH]c3ccccc23)C(c2c(F)cc([*:1])cc2F)N1CC(C)(C)F',
    'CCC1c2nnc(C)n2-c2cnc(Nc3ccc(C(=O)NC4CCN(C[*:1])CC4)cc3OC)nc2N1CC1CCCCC1',
    'CC(=O)NC(C(=O)N1CC(O)CC1C(=O)NCc1ccc(-c2scnc2C)cc1[*:1])C(C)(C)C',
    'C=CC(=O)N1CCCC(n2c(Nc3ccccc3)nc3cnc(Nc4ccc(N5CCN(C[*:1])CC5)cc4)nc32)C1',
    'C=C(F)C(=O)N1CCN(c2nc(OCC3CCCN3C[*:1])nc3c2CCN(c2cccc4cccc(Cl)c24)C3)CC1CC#N',
    'NC(=O)c1c(-c2ccc(Oc3ccc(F)cc3F)cc2)nn(C2CCCN([*:1])C2)c1N',
    'CN(CCCNc1nc(Nc2ccc([*:1])cc2)ncc1C1CC1)C(=O)C1CCC1',
    'CCN(c1ccc(C#N)c(Cl)c1)C1CCC(NC(=O)c2ccc([*:1])cc2)CC1',
    'CCC1NC(=O)C(C(O)C(C)CC=CC[*:1])N(C)C(=O)C(C(C)C)N(C)C(=O)C(CC(C)C)N(C)C(=O)C(CC(C)C)N(C)C(=O)C(C)NC(=O)C(C)NC(=O)C(CC(C)C)N(C)C(=O)C(C(C)C)NC(=O)C(CC(C)C)N(C)C(=O)CN(C)C1=O',
    'Cc1sc2c(c1C)C(c1ccc(Cl)cc1)=NC(CC(=O)[*:1])c1nnc(C)n1-2',
    'CCC(C(=O)N1CCCCC1C(=O)OC(CCc1ccc(OC)c(OC)c1)c1ccccc1[*:1])c1cc(OC)c(OC)c(OC)c1',
    'CCCn1cc(-c2nc([*:1])nc3[nH]ccc23)cn1',
    'COC1CC2CCC(C)C(O)(O2)C(=O)C(=O)N2CCCCC2C(=O)OC(C(C)CC2CCC(O[*:1])C(OC)C2)CC(=O)C(C)C=C(C)C(O)C(OC)C(=O)C(C)CC(C)C=CC=CC=C1C',
    'N#Cc1ccc(OC2CC(N[*:1])C2)cc1Cl',
    'CN1CCN(c2ccc(-c3cccc(C(=O)[*:1])c3)cc2NC(=O)c2c[nH]c(=O)cc2C(F)(F)F)CC1',
    'Cc1nc(Nc2ncc(C(=O)Nc3c(C)cccc3Cl)s2)cc(N2CCN(C[*:1])CC2)n1',
    'CC1(C)C(NC(=O)c2ccc(O[*:1])cc2)C(C)(C)C1Oc1ccc(C#N)c(Cl)c1',
    'NC(=O)CCC(NC(=O)C1CCC2CCN(C(=O)[*:1])CC(NC(=O)c3cc4cc(C(F)(F)P(=O)(O)O)ccc4[nH]3)C(=O)N21)C(=O)NC(c1ccccc1)c1ccccc1',
    'CCCS(=O)(=O)Nc1ccc(F)c(C(=O)c2c[nH]c3ncc(-c4ccc([*:1])cc4)cc23)c1F',
    'O=C(c1ccc(OCCN(C[*:1])C2CCC2)cc1)c1c(-c2ccc(O)cc2)sc2cc(O)ccc12',
    'CC1CC(O)c2ncnc(N3CCN(C(=O)C(CN[*:1])c4ccc(Cl)cc4)CC3)c21',
    'Cc1c(C#N)c(-c2ccc(C#N)cc2)c(C)n1Cc1ccccc1[*:1]',
    'CC1(N)CCN(c2cnc(Sc3cccc(N[*:1])c3Cl)c(N)n2)CC1',
    'CC(C)c1cnn2c(NCc3ccc([*:1])cc3)cc(NCCCCCCN)nc12',
    'CCOc1cc(C(C)(C)C)ccc1C1=N[C@H](c2ccc(Cl)cc2)[C@H](c2ccc(Cl)cc2)N1C(=O)N1CCN(CC(=O)[*:1])C(=O)C1',
    'c1cc([*:1])ccc1-c1csc(N2CCOCC2)n1',
    'CCCS(=O)(=O)Nc1ccc(F)c(-n2cc(-c3cncnc3)c3nc(N(C)C4CCN(CC[*:1])CC4)ccc32)c1F',
    'COc1cc2c(OCC3CCC(=O)N3)ncc([*:1])c2cc1C(N)=O',
    'Cn1cc(-c2ccccc2)c2cc(C(=O)[*:1])[nH]c2c1=O',
    'CCN(c1cc(-c2ccc(C[*:1])cc2)cc(C(=O)NCc2c(C)cc(C)[nH]c2=O)c1C)C1CCOCC1',
    'Cc1ccc(C(=O)Nc2ccc(CN3CCN(C[*:1])CC3)c(C(F)(F)F)c2)cc1C#Cc1cnc2cccnn12',
    'NC(=O)CCC(NC(=O)C1CCC2CCN([*:1])CC(NC(=O)c3cc4cc(C(F)(F)P(=O)(O)O)ccc4[nH]3)C(=O)N21)C(=O)NC(c1ccccc1)c1ccccc1',
    'O=C(CCCN[*:1])N1CCN(C(=O)c2cc(Cc3n[nH]c(=O)c4ccccc34)ccc2F)CC1',
    'Cc1ncsc1-c1ccc(CNC(=O)C2CC(O)CN2C(=O)C(NC(=O)CO[*:1])C(C)(C)C)c(F)c1',
]


@functools.lru_cache(maxsize=16, typed=False)
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

@functools.lru_cache(maxsize=16, typed=False)
def get_representative_whs_fp(
    wh_list: Optional[List[str]] = None,
    fp_generator: Optional[Any] = None,
    verbose: int = 0,
) -> List[DataStructs.ExplicitBitVect]:
    """
    Generate Morgan fingerprints for a list of warheads. If no list is provided,
    it uses a default list of representative warheads.
    
    Parameters:
        wh_list (Optional[List[str]]): List of SMILES strings for warheads. If None, uses a default list.
        fp_generator (Optional[Any]): RDKit fingerprint generator. If None, a default Morgan fingerprint generator is used.
        
    Returns:
        List[DataStructs.ExplicitBitVect]: List of RDKit Morgan fingerprints for the warheads.
    """
    representative_e3s_fp = []
    if verbose > 0:
        iterable = tqdm(wh_list or DEFAULT_REPRESENTATIVE_WHS, desc="Generating fingerprints for warheads")
    else:
        iterable = wh_list or DEFAULT_REPRESENTATIVE_WHS
    for smi in iterable:
        # Get the Morgan fingerprint for the SMILES string
        fp = get_fp(remove_dummy_atoms(smi), fp_generator, return_np=False)
        if fp is not None:
            representative_e3s_fp.append(fp)
        else:
            print(f"Warning: Invalid SMILES string '{smi}' encountered, skipping.")
    if not representative_e3s_fp:
        raise ValueError("No valid warheads found in the provided list.")
    return representative_e3s_fp