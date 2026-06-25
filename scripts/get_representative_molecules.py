"""Get N most representative E3 ligases and warheads from two input CSV files.

Each input CSV must have a 'SMILES' column. The script clusters molecules by
fingerprint similarity, finds the best number of clusters via silhouette /
Davies-Bouldin / Calinski-Harabasz metric agreement, and extracts
cluster-centroid nearest-neighbours as representatives. A cross-dissimilarity
filter then drops any representative that is too similar to a molecule in the
opposite class, ensuring E3 ligases and warheads remain chemically distinct.

Output: a single CSV with columns 'SMILES' and 'is_warhead' (bool).

Usage:
    python scripts/get_representative_molecules.py \\
        --e3-csv data/e3_ligases.csv \\
        --warhead-csv data/warheads.csv \\
        --output-csv data/representatives.csv
"""
from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import tyro
from rdkit import DataStructs
from rdkit.Chem import rdFingerprintGenerator
from tqdm import tqdm

from protac_splitter.graphs.e3_clustering import (
    evaluate_clusters,
    get_kmeans_clusters_fp,
    get_umap_clusters_fp,
)
from protac_splitter.graphs.utils import get_fp
from protac_splitter.chemoinformatics import remove_dummy_atoms


@dataclasses.dataclass
class Args:
    """Get representative E3 ligases and warheads from two input CSV files."""

    e3_csv: str
    """CSV file with E3 ligase SMILES (must have a 'SMILES' column)."""

    warhead_csv: str
    """CSV file with warhead SMILES (must have a 'SMILES' column)."""

    output_csv: str = "data/representative_molecules.csv"
    """Path to write the output CSV (columns: SMILES, is_warhead)."""

    smiles_column: str = "SMILES"
    """Column name in both input CSVs that holds SMILES strings."""

    n_clusters_candidates: List[int] = dataclasses.field(
        default_factory=lambda: [10, 25, 50, 100, 150]
    )
    """Candidate cluster counts evaluated for both molecule sets."""

    cross_similarity_threshold: float = 0.4
    """Max Tanimoto similarity allowed between an E3 rep and a warhead rep.
    Representatives exceeding this threshold vs. the other class are removed."""

    fp_radius: int = 16
    """Morgan fingerprint radius."""

    fp_size: int = 1024
    """Morgan fingerprint bit size."""

    verbose: bool = True
    """Print progress information."""


def _build_fp_generator(radius: int, fp_size: int):
    return rdFingerprintGenerator.GetMorganGenerator(
        radius=radius,
        fpSize=fp_size,
        useBondTypes=True,
        includeChirality=True,
    )


def _compute_fingerprints(
    smiles_list: List[str],
    fp_generator,
    desc: str = "Computing fingerprints",
) -> Tuple[List[np.ndarray], List[str]]:
    """Return parallel (fp_array, smiles) lists for valid molecules only."""
    fp_list, smi_list = [], []
    for smi in tqdm(smiles_list, desc=desc, leave=False):
        fp = get_fp(remove_dummy_atoms(smi), fp_generator)
        if fp is not None:
            fp_list.append(fp)
            smi_list.append(smi)
    return fp_list, smi_list


def _select_best_n_clusters(
    fp_list: List[np.ndarray],
    candidates: List[int],
    desc: str,
) -> Tuple[int, dict, dict]:
    """
    Sweep over candidate cluster counts, return (best_n, clusters_dict, centroids_dict).

    Sweet-spot heuristic: normalize silhouette, -davies_bouldin, and
    calinski_harabasz per algorithm then pick the n with minimum std across the
    three normalized scores (i.e. all three metrics agree the most).
    """
    max_k = len(fp_list) - 1
    valid_candidates = [k for k in candidates if 1 < k <= max_k]
    if not valid_candidates:
        valid_candidates = [max(2, max_k)]

    centroids_dict: dict = {}
    clusters_dict: dict = {}
    metrics_rows: list = []

    for n_clusters in tqdm(valid_candidates, desc=desc, leave=True):
        clusters, centroids = get_kmeans_clusters_fp(
            fp_list, n_clusters=n_clusters, return_centroids=True
        )
        clusters_dict[f"kmeans_n{n_clusters}"] = clusters.copy()
        centroids_dict[n_clusters] = centroids.copy()
        m = evaluate_clusters(fp_list, clusters)
        m["num_clusters"] = n_clusters
        m["cluster_algorithm"] = "kmeans"
        metrics_rows.append(m)

        clusters_umap = get_umap_clusters_fp(fp_list, n_clusters=n_clusters)
        clusters_dict[f"umap_n{n_clusters}"] = clusters_umap.copy()
        m_umap = evaluate_clusters(fp_list, clusters_umap)
        m_umap["num_clusters"] = n_clusters
        m_umap["cluster_algorithm"] = "umap"
        metrics_rows.append(m_umap)

    metrics_df = pd.DataFrame(metrics_rows)
    metrics_df["-davies_bouldin"] = -metrics_df["davies_bouldin"]
    metric_cols = ["silhouette", "-davies_bouldin", "calinski_harabasz"]
    df_norm = metrics_df.copy()
    df_norm[metric_cols] = df_norm.groupby("cluster_algorithm")[metric_cols].transform(
        lambda x: (x - x.min()) / (x.max() - x.min() + 1e-12)
    )
    df_norm["metric_divergence"] = df_norm[metric_cols].std(axis=1)
    sweet_spots = df_norm.loc[
        df_norm.groupby("cluster_algorithm")["metric_divergence"].idxmin()
    ]
    best_n = int(sweet_spots["num_clusters"].mode().iloc[0])
    return best_n, clusters_dict, centroids_dict


def get_representatives(
    smiles_list: List[str],
    fp_generator,
    n_clusters_candidates: List[int],
    label: str = "molecules",
) -> List[str]:
    """Cluster molecules and return the nearest-to-centroid SMILES per cluster."""
    fp_list, smi_list = _compute_fingerprints(
        smiles_list, fp_generator, desc=f"Fingerprints ({label})"
    )
    if not fp_list:
        raise ValueError(f"No valid SMILES found in the {label} input.")

    fp2smi = {fp.tobytes(): smi for fp, smi in zip(fp_list, smi_list)}

    best_n, clusters_dict, centroids_dict = _select_best_n_clusters(
        fp_list, n_clusters_candidates, desc=f"Clustering ({label})"
    )

    centroids = centroids_dict[best_n]
    clusters = np.array(clusters_dict[f"kmeans_n{best_n}"])
    fp_arr = np.array(fp_list)

    representatives: List[str] = []
    for label_idx, centroid in enumerate(centroids):
        mask = clusters == label_idx
        if not mask.any():
            continue
        fp_cluster = fp_arr[mask]
        distances = np.linalg.norm(fp_cluster - centroid, axis=1)
        closest = fp_cluster[int(np.argmin(distances))]
        smi = fp2smi.get(closest.tobytes())
        if smi is not None:
            representatives.append(smi)

    return representatives


def cross_dissimilarity_filter(
    e3_reps: List[str],
    warhead_reps: List[str],
    fp_generator,
    threshold: float,
    verbose: bool = True,
) -> Tuple[List[str], List[str]]:
    """Remove representatives that are too similar across the two classes.

    First removes E3 reps too similar to any warhead rep, then removes warhead
    reps too similar to the (already-filtered) E3 reps.
    """
    def _rdkit_fps(smiles, gen):
        return [(smi, get_fp(smi, gen, return_np=False)) for smi in smiles]

    e3_with_fps = _rdkit_fps(e3_reps, fp_generator)
    wh_with_fps = _rdkit_fps(warhead_reps, fp_generator)

    wh_fps_valid = [fp for _, fp in wh_with_fps if fp is not None]

    filtered_e3: List[str] = []
    removed_e3 = 0
    for smi, fp in e3_with_fps:
        if fp is None:
            continue
        if not wh_fps_valid:
            filtered_e3.append(smi)
            continue
        sims = DataStructs.BulkTanimotoSimilarity(fp, wh_fps_valid)
        if max(sims) <= threshold:
            filtered_e3.append(smi)
        else:
            removed_e3 += 1

    e3_fps_filtered = [fp for _, fp in _rdkit_fps(filtered_e3, fp_generator) if fp is not None]

    filtered_wh: List[str] = []
    removed_wh = 0
    for smi, fp in wh_with_fps:
        if fp is None:
            continue
        if not e3_fps_filtered:
            filtered_wh.append(smi)
            continue
        sims = DataStructs.BulkTanimotoSimilarity(fp, e3_fps_filtered)
        if max(sims) <= threshold:
            filtered_wh.append(smi)
        else:
            removed_wh += 1

    if verbose:
        print(
            f"Cross-dissimilarity filter (threshold={threshold}): "
            f"removed {removed_e3} E3 rep(s), {removed_wh} warhead rep(s)."
        )

    return filtered_e3, filtered_wh


def main(args: Args) -> None:
    e3_path = Path(args.e3_csv)
    wh_path = Path(args.warhead_csv)

    if not e3_path.exists():
        raise FileNotFoundError(f"E3 CSV not found: {e3_path}")
    if not wh_path.exists():
        raise FileNotFoundError(f"Warhead CSV not found: {wh_path}")

    e3_df = pd.read_csv(e3_path)
    wh_df = pd.read_csv(wh_path)

    for df, name in [(e3_df, "E3"), (wh_df, "warhead")]:
        if args.smiles_column not in df.columns:
            raise ValueError(
                f"Column '{args.smiles_column}' not found in {name} CSV. "
                f"Available columns: {list(df.columns)}"
            )

    e3_smiles = e3_df[args.smiles_column].dropna().unique().tolist()
    wh_smiles = wh_df[args.smiles_column].dropna().unique().tolist()

    if args.verbose:
        print(f"Loaded {len(e3_smiles)} unique E3 SMILES, {len(wh_smiles)} unique warhead SMILES.")

    fp_generator = _build_fp_generator(args.fp_radius, args.fp_size)

    if args.verbose:
        print("Finding representative E3 ligases...")
    e3_reps = get_representatives(
        e3_smiles, fp_generator, args.n_clusters_candidates, label="E3"
    )

    if args.verbose:
        print(f"Found {len(e3_reps)} E3 representatives.")
        print("Finding representative warheads...")
    wh_reps = get_representatives(
        wh_smiles, fp_generator, args.n_clusters_candidates, label="warheads"
    )

    if args.verbose:
        print(f"Found {len(wh_reps)} warhead representatives.")
        print("Applying cross-dissimilarity filter...")

    e3_reps_final, wh_reps_final = cross_dissimilarity_filter(
        e3_reps, wh_reps, fp_generator, args.cross_similarity_threshold, verbose=args.verbose
    )

    if args.verbose:
        print(
            f"Final counts: {len(e3_reps_final)} E3 representatives, "
            f"{len(wh_reps_final)} warhead representatives."
        )

    rows = (
        [{"SMILES": s, "is_warhead": False} for s in e3_reps_final]
        + [{"SMILES": s, "is_warhead": True} for s in wh_reps_final]
    )
    out_df = pd.DataFrame(rows)

    out_path = Path(args.output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)

    if args.verbose:
        print(f"Saved {len(out_df)} representatives to {out_path}")


if __name__ == "__main__":
    main(tyro.cli(Args))
