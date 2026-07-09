"""Curate PROTAC molecules from PROTAC-DB / PROTAC-Pedia CSV files.

Builds substructure dictionaries from E3, warhead, and linker SMILES columns,
then runs iterative splitting to assign ``[*:1]``/``[*:2]`` attachment points.

Usage:
    python scripts/curate_data.py --help
    python scripts/curate_data.py --protac-csv data/protacdb.csv --output-csv data/curated.csv
"""
from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Optional

import pandas as pd
import tyro

from protac_splitter.data.curation.mapping_utils import update_dictionary
from protac_splitter.data.curation.curation import iterative_protac_splitting
from scripts.common import ensure_output_dir


@dataclasses.dataclass
class Args:
    """Curate PROTAC data from a CSV file by iterative substructure splitting."""

    protac_csv: str
    """Path to input CSV file (PROTAC-DB or PROTAC-Pedia format)."""

    output_csv: str
    """Path to save the curated output CSV."""

    output_dir: str = "data/curation_steps"
    """Directory to store intermediate splitting step CSVs."""

    protac_smiles_col: str = "PROTAC SMILES"
    e3_smiles_col: str = "E3 Binder SMILES"
    poi_smiles_col: str = "POI Ligand SMILES"
    linker_smiles_col: Optional[str] = "Linker SMILES"
    """Linker SMILES column (optional; leave empty if not available)."""

    push_to_hub: bool = False
    hub_dataset_id: str = "ailab-bio/PROTAC-Splitter-Dataset"
    hub_token: Optional[str] = None


def main(args: Args) -> None:
    from rdkit.Chem import rdFingerprintGenerator
    from scripts.common import get_hub_token

    df = pd.read_csv(args.protac_csv, low_memory=False)
    print(f"Loaded {len(df)} rows from {args.protac_csv}")

    # Validate required columns
    for col in (args.protac_smiles_col, args.e3_smiles_col, args.poi_smiles_col):
        if col not in df.columns:
            raise ValueError(f"Column not found: '{col}'. Available: {list(df.columns)}")

    ensure_output_dir(args.output_dir)

    morgan_fp_gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=512)

    # Build substructure dictionaries from unique SMILES values in the CSV
    print("Building substructure dictionaries...")
    empty_dict = pd.DataFrame(columns=["SMILES", "Molecule", "ID", "FP"])

    e3_dict = update_dictionary(
        empty_dict.copy(),
        substr_to_add=df[args.e3_smiles_col].dropna().unique().tolist(),
        morgan_fp_generator=morgan_fp_gen,
        verbose=1,
    )
    poi_dict = update_dictionary(
        empty_dict.copy(),
        substr_to_add=df[args.poi_smiles_col].dropna().unique().tolist(),
        morgan_fp_generator=morgan_fp_gen,
        verbose=1,
    )
    linker_smiles = []
    if args.linker_smiles_col and args.linker_smiles_col in df.columns:
        linker_smiles = df[args.linker_smiles_col].dropna().unique().tolist()
    linker_dict = update_dictionary(
        empty_dict.copy(),
        substr_to_add=linker_smiles,
        morgan_fp_generator=morgan_fp_gen,
        verbose=1,
    )

    print(f"E3 dictionary: {len(e3_dict)} entries")
    print(f"POI dictionary: {len(poi_dict)} entries")
    print(f"Linker dictionary: {len(linker_dict)} entries")

    # Build the PROTAC dictionary for iterative_protac_splitting
    protac_dict = update_dictionary(
        empty_dict.copy(),
        substr_to_add=df[args.protac_smiles_col].dropna().unique().tolist(),
        morgan_fp_generator=morgan_fp_gen,
    )

    dictionaries = {
        "PROTAC": df[[args.protac_smiles_col]].rename(columns={args.protac_smiles_col: "PROTAC SMILES"}),
        "E3 Binder": e3_dict,
        "POI Ligand": poi_dict,
        "Linker": linker_dict,
    }

    print("\nRunning iterative PROTAC splitting...")
    result = iterative_protac_splitting(dictionaries, data_dir=args.output_dir)

    # The function returns a dict; combine the final result
    if isinstance(result, dict):
        final_df = pd.concat(list(result.values()), ignore_index=True) if result else pd.DataFrame()
    else:
        final_df = result

    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    final_df.to_csv(output_path, index=False)
    print(f"\nCurated dataset saved: {output_path} ({len(final_df)} rows)")

    if args.push_to_hub:
        from datasets import Dataset
        token = get_hub_token(args.hub_token)
        Dataset.from_pandas(final_df).push_to_hub(args.hub_dataset_id, token=token)
        print(f"Pushed to HuggingFace Hub: {args.hub_dataset_id}")


if __name__ == "__main__":
    main(tyro.cli(Args))
