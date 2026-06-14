"""Plot the chemical space of PROTACs using PCA on Morgan fingerprints.

Usage:
    python scripts/plot_chemical_space.py --help
    python scripts/plot_chemical_space.py --protac-db-path data/protacdb.csv --protac-pedia-path data/protacpedia.csv
"""
from __future__ import annotations

import dataclasses
from collections import defaultdict
from pathlib import Path
from typing import Optional

import matplotlib
import numpy as np
import pandas as pd
import tyro
from rdkit import Chem
from sklearn.decomposition import PCA
from tqdm import tqdm

from protac_splitter.chemoinformatics import canonize
from scripts.common import ensure_output_dir

matplotlib.rcParams.update({"font.size": 12, "font.family": "serif"})


@dataclasses.dataclass
class Args:
    """Plot PCA of PROTAC chemical space from PROTAC-DB and PROTAC-Pedia."""

    protac_db_path: str
    """Path to the PROTAC-DB v3.0 CSV file."""

    protac_pedia_path: str
    """Path to the PROTAC-Pedia CSV file."""

    output_dir: str = "chemical_space_plots"
    """Directory to save output plots."""

    internal_data_path: Optional[str] = None
    """Optional path to internal PROTAC-Splitter dataset CSV."""

    num_proc: int = 2
    """Processes for HuggingFace dataset mapping."""

    num_proc_fp_gen: int = 8
    """Threads for fingerprint generation."""

    test_internal_data: bool = False
    """Add noise to internal data fingerprints (for testing)."""


def main(args: Args) -> None:
    import seaborn as sns
    from datasets import load_dataset
    from matplotlib import pyplot as plt

    out_dir = ensure_output_dir(args.output_dir)

    def get_substructs(row):
        return {
            "PROTAC SMILES": row["text"],
            "POI Ligand SMILES with direction": row["labels"].split(".")[2],
            "Linker SMILES with direction": row["labels"].split(".")[1],
            "E3 Binder SMILES with direction": row["labels"].split(".")[0],
        }

    import os
    ds = load_dataset("ailab-bio/PROTAC-Splitter-Dataset", "clustered",
                      token=os.getenv("HF_TOKEN"))
    ds = ds.map(get_substructs, num_proc=args.num_proc, remove_columns=["text", "labels"])
    train_df = ds["train"].to_pandas()
    val_df = ds["validation"].to_pandas()
    test_df = ds["test"].to_pandas()
    held_out_df = ds["held_out"].to_pandas()

    protacdb_df = pd.read_csv(args.protac_db_path, low_memory=False).dropna(subset=["Smiles"])
    protacpedia_df = pd.read_csv(args.protac_pedia_path, low_memory=False).dropna(subset=["PROTAC SMILES"])
    protacdb_df = protacdb_df.rename(columns={"Smiles": "PROTAC SMILES"}).drop_duplicates(subset=["PROTAC SMILES"])
    protacpedia_df = protacpedia_df.drop_duplicates(subset=["PROTAC SMILES"])

    tqdm.pandas(desc="Canonizing PROTAC-DB")
    protacdb_df["PROTAC SMILES"] = protacdb_df["PROTAC SMILES"].progress_apply(canonize)
    tqdm.pandas(desc="Canonizing PROTAC-Pedia")
    protacpedia_df["PROTAC SMILES"] = protacpedia_df["PROTAC SMILES"].progress_apply(canonize)

    def map_protac_smiles(row):
        smi = row["PROTAC SMILES"]
        match = held_out_df[held_out_df["PROTAC SMILES"] == smi]
        if len(match):
            return match.iloc[0].to_dict()
        return {"PROTAC SMILES": smi, "POI Ligand SMILES with direction": None,
                "Linker SMILES with direction": None, "E3 Binder SMILES with direction": None}

    tqdm.pandas(desc="Mapping PROTAC-DB")
    protacdb_df = protacdb_df.progress_apply(map_protac_smiles, axis=1, result_type="expand")
    tqdm.pandas(desc="Mapping PROTAC-Pedia")
    protacpedia_df = protacpedia_df.progress_apply(map_protac_smiles, axis=1, result_type="expand")

    cols = ["POI Ligand SMILES with direction", "Linker SMILES with direction", "E3 Binder SMILES with direction"]
    protacdb_df = protacdb_df.dropna(subset=cols)
    protacpedia_df = protacpedia_df.dropna(subset=cols)

    if args.internal_data_path is not None:
        internal_df = pd.read_csv(args.internal_data_path, low_memory=False)
        if "labels" in internal_df.columns and "text" in internal_df.columns:
            tqdm.pandas(desc="Extracting internal substructures")
            internal_df = internal_df.progress_apply(get_substructs, axis=1, result_type="expand")

    ligands: dict = {s: defaultdict(list) for s in ["train", "val", "test", "held_out", "protacdb", "protacpedia"]}
    if args.internal_data_path is not None:
        ligands["internal"] = defaultdict(list)

    split_dfs = {"train": train_df, "val": val_df, "test": test_df, "held_out": held_out_df,
                 "protacdb": protacdb_df, "protacpedia": protacpedia_df}
    if args.internal_data_path is not None:
        split_dfs["internal"] = internal_df

    column_names = ["PROTAC SMILES", "E3 Binder SMILES with direction", "POI Ligand SMILES with direction", "Linker SMILES with direction"]
    for col in column_names:
        for split, df in split_dfs.items():
            for smi in df[col].unique():
                ligands[split][col].append(smi)

    fp_gen = Chem.rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=512, useBondTypes=True, includeChirality=True)

    def bitvect_to_numpy(bv):
        return np.frombuffer(bv.ToBitString().encode(), "u1") - ord("0")

    ligands_fp: dict = {}
    for split, ligand_dict in ligands.items():
        ligands_fp[split] = defaultdict(list)
        for col, ligand_list in ligand_dict.items():
            fname = out_dir / f"{split}_{col.split(' ')[0].lower()}_fp.npy"
            if fname.exists():
                print(f"Loading cached FP: {fname}")
                ligands_fp[split][col] = np.load(fname)
                continue
            print(f"Generating {split} {col} FPs...")
            fps = fp_gen.GetFingerprints([Chem.MolFromSmiles(s) for s in ligand_list], numThreads=args.num_proc_fp_gen)
            arr = [bitvect_to_numpy(fp) for fp in fps]
            ligands_fp[split][col] = arr
            np.save(fname, np.array(arr))
        print()

    colors = {
        "Train (Synthetic)": "#FFD700", "Validation (Synthetic)": "#EE82EE",
        "Test (Synthetic)": "#94ED67", "PROTAC-DB v3.0": "#83B8FE",
        "PROTAC-Pedia": "#FF7F50", "Internal Data": "#8A2BE2",
    }

    for col in column_names:
        ligand_name = col.split(" ")[0].lower()
        ligand_name_ext = " ".join(col.split(" ")[:1 if "linker" in ligand_name else 2])

        if col == "PROTAC SMILES":
            fp_list = [ligands_fp[s][col] for s in ("train", "val", "test", "protacdb", "protacpedia")]
        else:
            fp_list = [ligands_fp["protacdb"][col], ligands_fp["protacpedia"][col]]
        if args.internal_data_path is not None:
            fps = np.array(ligands_fp["internal"][col])
            if args.test_internal_data:
                fps = fps + np.random.normal(0, 0.5, fps.shape)
            fp_list.append(fps.tolist())

        all_embeddings = np.vstack(fp_list)
        pca_file = out_dir / f"all_{ligand_name}_embeddings_pca.npy"
        if pca_file.exists():
            print(f"Loading cached PCA: {pca_file}")
            all_embeddings_pca = np.load(pca_file)
        else:
            pca = PCA(n_components=2, random_state=42)
            all_embeddings_pca = pca.fit_transform(all_embeddings)
            np.save(pca_file, all_embeddings_pca)

        df_embeddings = pd.DataFrame(all_embeddings_pca, columns=["x", "y"])
        split_labels: list = []
        if col == "PROTAC SMILES":
            for split, label in [("train", "Train (Synthetic)"), ("val", "Validation (Synthetic)"),
                                  ("test", "Test (Synthetic)"), ("protacdb", "PROTAC-DB v3.0"), ("protacpedia", "PROTAC-Pedia")]:
                split_labels += [label] * len(ligands_fp[split][col])
            if args.internal_data_path is not None:
                split_labels += ["Internal Data"] * len(ligands_fp["internal"][col])
        else:
            split_labels = [f"{ligand_name_ext}s - PROTAC-DB v3.0"] * len(ligands_fp["protacdb"][col]) + \
                           [f"{ligand_name_ext}s - PROTAC-Pedia"] * len(ligands_fp["protacpedia"][col])
            if args.internal_data_path is not None:
                split_labels += [f"{ligand_name_ext}s - Internal"] * len(ligands_fp["internal"][col])
        df_embeddings["split"] = split_labels

        if col == "PROTAC SMILES":
            palette = list(colors.values())[:df_embeddings["split"].nunique()]
            plt.figure(figsize=(8, 8))
            sns.scatterplot(data=df_embeddings, x="x", y="y", hue="split", alpha=0.6, palette=palette, s=12, edgecolor="black", linewidth=0.1, rasterized=True)
            plt.legend(title="", loc="upper left", fontsize=12, markerscale=2.2)
        else:
            palette = ["#83B8FE", "#FF7FFF"] + (["#94ED67"] if args.internal_data_path else [])
            plt.figure(figsize=(6, 6))
            sns.scatterplot(data=df_embeddings, x="x", y="y", hue="split", alpha=0.6, palette=palette, edgecolor="black")
            plt.legend(title="", loc="lower left", fontsize=12, markerscale=2.2)

        plt.xlabel("PCA Component 1", fontdict={"fontsize": 12})
        plt.ylabel("PCA Component 2", fontdict={"fontsize": 12})
        plt.title("")
        plt.grid(alpha=0.5)
        plt.tight_layout()
        plot_path = out_dir / f"pca_{ligand_name}.pdf"
        plt.savefig(plot_path, bbox_inches="tight")
        plt.clf()
        plt.close()
        print(f"Saved: {plot_path}")


if __name__ == "__main__":
    main(tyro.cli(Args))
