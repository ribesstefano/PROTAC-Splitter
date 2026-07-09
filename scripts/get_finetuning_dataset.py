"""Generate a clustered finetuning dataset from a held-out PROTAC CSV.

Selects representative PROTACs via K-means and Agglomerative clustering for
cluster sizes [10, 25, 50, 100], saving train/test splits under ``ds_root``.

Usage:
    python scripts/get_finetuning_dataset.py --help
    python scripts/get_finetuning_dataset.py --filename-held-out-df data/held_out.csv
"""
from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import tyro
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from scipy.stats import skew
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score
from tqdm import tqdm

from scripts.common import ensure_output_dir


@dataclasses.dataclass
class Args:
    """Generate a finetuning dataset via K-means clustering of PROTAC fingerprints."""

    filename_held_out_df: str
    """Path to the held-out dataset CSV file."""

    ds_root: str = "finetuning_dataset"
    """Root directory to save the finetuning dataset."""

    show_plots: bool = False
    """Show cluster metric bar-plots (requires seaborn + matplotlib)."""

    protac_smiles_col: str = "PROTAC SMILES"
    poi_smiles_col: str = "POI Ligand SMILES with direction"
    linker_smiles_col: str = "Linker SMILES with direction"
    e3_smiles_col: str = "E3 Binder SMILES with direction"


def _get_fp(smiles: str, fp_generator) -> np.ndarray | None:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return np.array(fp_generator.GetFingerprint(mol))


def _get_umap_clusters(fp_list: List[np.ndarray], n_clusters: int) -> np.ndarray:
    ac = AgglomerativeClustering(n_clusters=n_clusters)
    ac.fit_predict(np.stack(fp_list))
    return ac.labels_


def _get_kmeans_clusters(fp_list, n_clusters, return_centroids=False):
    km = KMeans(n_clusters=n_clusters, n_init="auto", random_state=42, max_iter=1000)
    if return_centroids:
        km.fit(np.stack(fp_list))
        return km.labels_, km.cluster_centers_
    return km.fit_predict(np.stack(fp_list))


def _evaluate_clusters(X, clusters) -> dict:
    unique = list(set(clusters))
    if len(unique) < 2:
        return {"silhouette": -1, "davies_bouldin": float("inf"), "calinski_harabasz": -1,
                "num_clusters": 1}
    sizes = [len(np.where(clusters == i)[0]) for i in np.unique(clusters)]
    return {
        "silhouette": silhouette_score(X, clusters),
        "davies_bouldin": davies_bouldin_score(X, clusters),
        "calinski_harabasz": calinski_harabasz_score(X, clusters),
        "avg_cluster_size": float(np.mean(sizes)),
        "std_cluster_size": float(np.std(sizes)),
        "min_cluster_size": int(np.min(sizes)),
        "max_cluster_size": int(np.max(sizes)),
        "cluster_size_skewness": float(skew(sizes, nan_policy="omit")),
        "num_clusters": len(unique),
    }


def _get_text_labels(row, protac_col, e3_col, linker_col, poi_col) -> dict:
    return {
        "text": row[protac_col],
        "labels": ".".join([row[e3_col], row[linker_col], row[poi_col]]),
    }


def _process_df(df, protac_col, e3_col, linker_col, poi_col) -> pd.DataFrame:
    processed = df.apply(
        lambda r: _get_text_labels(r, protac_col, e3_col, linker_col, poi_col),
        axis=1,
        result_type="expand",
    )
    return processed[["text", "labels"]]


def main(args: Args) -> None:
    from datasets import load_dataset

    held_out_df = pd.read_csv(args.filename_held_out_df)
    ds_root = Path(args.ds_root)

    morgan_fp_gen = rdFingerprintGenerator.GetMorganGenerator(
        radius=16, fpSize=1024, useBondTypes=True, includeChirality=True,
    )

    fp_dict = {smi: _get_fp(smi, morgan_fp_gen) for smi in held_out_df[args.protac_smiles_col]}
    fp_list = [fp for fp in fp_dict.values() if fp is not None]
    fp2smiles = {fp.tobytes(): smi for smi, fp in fp_dict.items() if fp is not None}
    print(f"Unique SMILES: {held_out_df[args.protac_smiles_col].nunique():,}")
    print(f"Unique fingerprints: {len(fp2smiles):,}")

    centroids_dict, clusters_dict, metrics_rows = {}, {}, []
    for n in tqdm([10, 25, 50, 100], desc="Clustering"):
        clusters, centroids = _get_kmeans_clusters(fp_list, n, return_centroids=True)
        clusters_dict[f"kmeans_n{n}"] = clusters.copy()
        centroids_dict[n] = centroids.copy()
        metrics_rows.append({"num_clusters": n, "cluster_algorithm": "kmeans",
                             **_evaluate_clusters(fp_list, clusters)})
        clusters = _get_umap_clusters(fp_list, n)
        clusters_dict[f"umap_n{n}"] = clusters.copy()
        metrics_rows.append({"num_clusters": n, "cluster_algorithm": "umap",
                             **_evaluate_clusters(fp_list, clusters)})

    if args.show_plots:
        import seaborn as sns
        from matplotlib import pyplot as plt
        metrics_df = pd.DataFrame(metrics_rows)
        _, axs = plt.subplots(1, 3, figsize=(20, 5))
        for ax, metric in zip(axs, ["silhouette", "davies_bouldin", "calinski_harabasz"]):
            sns.barplot(data=metrics_df, x="num_clusters", y=metric, hue="cluster_algorithm", ax=ax)
            ax.set_title(metric.replace("_", " ").title())
            ax.grid(axis="y", alpha=0.5)
        plt.tight_layout()
        plt.show()

    ensure_output_dir(str(ds_root))
    for n, centroids in centroids_dict.items():
        print(f"\n{'─' * 60}\nClusters: {n}\n{'─' * 60}")
        clusters = np.array(clusters_dict[f"kmeans_n{n}"])
        finetune_samples = []
        for label, centroid in enumerate(centroids):
            fp_cluster = np.array(fp_list)[clusters == label]
            distances = np.linalg.norm(fp_cluster - centroid, axis=1)
            closest_smiles = fp2smiles[fp_cluster[np.argmin(distances)].tobytes()]
            print(f"  Centroid {label}: {closest_smiles}")
            finetune_samples.append(held_out_df.loc[held_out_df[args.protac_smiles_col] == closest_smiles])

        finetune_df = pd.concat(finetune_samples)
        train_df = _process_df(finetune_df, args.protac_smiles_col, args.e3_smiles_col,
                               args.linker_smiles_col, args.poi_smiles_col)
        test_raw = held_out_df[~held_out_df[args.protac_smiles_col].isin(finetune_df[args.protac_smiles_col])].copy()
        test_df = _process_df(test_raw, args.protac_smiles_col, args.e3_smiles_col,
                              args.linker_smiles_col, args.poi_smiles_col)

        config_dir = ds_root / f"n{n}"
        ensure_output_dir(str(config_dir))
        train_df.to_csv(config_dir / "train.csv", index=False)
        test_df.to_csv(config_dir / "test.csv", index=False)
        print(f"  Saved to {config_dir}")
        print(load_dataset(str(ds_root), data_dir=f"n{n}"))

    print(f"\nExample usage:\n  load_dataset('{args.ds_root}', data_dir='n10')")


if __name__ == "__main__":
    main(tyro.cli(Args))
