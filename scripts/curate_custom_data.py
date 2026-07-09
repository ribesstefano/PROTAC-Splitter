"""Curate a custom PROTAC dataset by splitting into E3 / linker / POI components.

Applies iterative substructure splitting using dictionaries built from the
E3, warhead, and linker columns in the input CSV.

Usage:
    python scripts/curate_custom_data.py --help
    python scripts/curate_custom_data.py --input-csv my_protacs.csv --output-csv curated.csv
"""
from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Optional

import pandas as pd
import tyro

from protac_splitter.data.curation.curation import iterative_protac_splitting, split_protacs
from protac_splitter.data.curation.mapping_utils import update_dictionary
from scripts.common import ensure_output_dir


@dataclasses.dataclass
class Args:
    """Split a custom PROTAC CSV into E3 / linker / POI substructures."""

    input_csv: str
    """Input CSV with PROTAC, E3, POI, and (optionally) linker SMILES columns."""

    output_csv: str
    """Path to save the split output CSV."""

    output_dir: str = "data/custom_curation_steps"
    """Directory to store intermediate splitting steps."""

    protac_col: str = "PROTAC SMILES"
    e3_col: str = "E3 Binder SMILES"
    poi_col: str = "POI Ligand SMILES"
    linker_col: Optional[str] = None
    """Linker SMILES column (optional)."""

    iterative: bool = True
    """Use iterative splitting (recommended) vs. single-pass."""


def main(args: Args) -> None:
    from rdkit.Chem import rdFingerprintGenerator

    df = pd.read_csv(args.input_csv, low_memory=False)
    print(f"Loaded {len(df)} rows from {args.input_csv}")

    for col in (args.protac_col, args.e3_col, args.poi_col):
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found. Available: {list(df.columns)}")

    ensure_output_dir(args.output_dir)
    morgan_fp_gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=512)
    empty = pd.DataFrame(columns=["SMILES", "Molecule", "ID", "FP"])

    e3_dict = update_dictionary(empty.copy(), df[args.e3_col].dropna().unique().tolist(), morgan_fp_gen, verbose=1)
    poi_dict = update_dictionary(empty.copy(), df[args.poi_col].dropna().unique().tolist(), morgan_fp_gen, verbose=1)
    linker_smiles = []
    if args.linker_col and args.linker_col in df.columns:
        linker_smiles = df[args.linker_col].dropna().unique().tolist()
    linker_dict = update_dictionary(empty.copy(), linker_smiles, morgan_fp_gen, verbose=1)

    print(f"Dictionaries — E3: {len(e3_dict)}, POI: {len(poi_dict)}, Linker: {len(linker_dict)}")

    protac_rows = df[[args.protac_col]].rename(columns={args.protac_col: "PROTAC SMILES"})
    dictionaries = {
        "PROTAC": protac_rows,
        "E3 Binder": e3_dict,
        "POI Ligand": poi_dict,
        "Linker": linker_dict,
    }

    if args.iterative:
        print("\nRunning iterative splitting...")
        result = iterative_protac_splitting(dictionaries, data_dir=args.output_dir)
        final_df = pd.concat(list(result.values()), ignore_index=True) if isinstance(result, dict) else result
    else:
        print("\nRunning single-pass splitting...")
        final_df = split_protacs(protac_rows.rename(columns={"PROTAC SMILES": args.protac_col}), dictionaries)

    out = Path(args.output_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    final_df.to_csv(out, index=False)
    print(f"\nSaved {len(final_df)} curated PROTACs to {out}")


if __name__ == "__main__":
    main(tyro.cli(Args))
