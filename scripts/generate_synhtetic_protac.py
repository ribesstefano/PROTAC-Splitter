"""Generate synthetic PROTAC training data from a curated dataset.

Computes functional-group distributions from curated E3/linker/WH SMILES
with attachment-point direction labels, then samples N synthetic PROTACs
via the ``generate_protacs`` generator.

Usage:
    python scripts/generate_training_data.py --help
    python scripts/generate_training_data.py --curated-csv data/curated.csv --output-dir data/generated
"""
from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Optional

import pandas as pd
import tyro

from protac_splitter.data.generation.functional_groups import get_functional_groups_distributions
from protac_splitter.data.generation.generation import generate_protacs
from scripts.common import ensure_output_dir, get_hub_token


@dataclasses.dataclass
class Args:
    """Generate synthetic PROTAC training data from a curated CSV."""

    curated_csv: str
    """Input curated CSV (output of curate_data.py) with directional SMILES columns."""

    output_dir: str = "data/generated"
    """Directory to save generated datasets and distribution files."""

    num_samples: int = 1_000_000
    """Number of synthetic PROTACs to generate."""

    random_state: int = 42
    batch_size: int = 1000
    max_workers: int = 4

    push_to_hub: bool = False
    hub_dataset_id: str = "ailab-bio/PROTAC-Splitter-Dataset"
    hub_token: Optional[str] = None

    protac_smiles_col: str = "PROTAC SMILES"
    e3_smiles_col: str = "E3 Binder SMILES with direction"
    linker_smiles_col: str = "Linker SMILES with direction"
    poi_smiles_col: str = "POI Ligand SMILES with direction"

    reload_distributions: bool = True
    """Load precomputed distributions from file if available."""

    verbose: int = 0


def main(args: Args) -> None:
    out_dir = ensure_output_dir(args.output_dir)

    df = pd.read_csv(args.curated_csv, low_memory=False)
    print(f"Loaded {len(df)} curated PROTACs from {args.curated_csv}")

    for col in (args.protac_smiles_col, args.e3_smiles_col, args.linker_smiles_col, args.poi_smiles_col):
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found. Available: {list(df.columns)}")

    distributions_file = out_dir / "fg_distributions.json"
    mappings_file = out_dir / "fg_mappings.json"
    df_fg_file = out_dir / "df_with_functional_groups.csv"

    print("\nComputing functional group distributions...")
    fg_data = get_functional_groups_distributions(
        df,
        load_from_file=args.reload_distributions,
        filename_distributions=str(distributions_file),
        filename_mappings=str(mappings_file),
        filename_df_with_functional_groups=str(df_fg_file),
        verbose=args.verbose,
    )

    # Save distributions for reuse
    distributions_file.write_text(json.dumps({k: v for k, v in fg_data.items() if "distr" in k}, default=list))
    mappings_file.write_text(json.dumps({k: v for k, v in fg_data.items() if "distr" not in k}, default=list))
    print(f"Distributions saved: {distributions_file}, {mappings_file}")

    generated_csv = out_dir / "generated_protacs.csv"
    print(f"\nGenerating {args.num_samples:,} synthetic PROTACs...")
    generated_df = generate_protacs(
        poi_fg_distr=fg_data["poi_fg_distr"],
        e3_fg_distr=fg_data["e3_fg_distr"],
        substr_fg_2_linker=fg_data["substr_fg_2_linker"],
        poi_fg_2_substr=fg_data["poi_fg_2_substr"],
        e3_fg_2_substr=fg_data["e3_fg_2_substr"],
        num_samples=args.num_samples,
        random_state=args.random_state,
        batch_size=args.batch_size,
        max_workers=args.max_workers,
        original_df=df,
        filename_generated_df=str(generated_csv),
        base_data_dir=str(out_dir),
    )
    print(f"Generated {len(generated_df):,} PROTACs → {generated_csv}")

    if args.push_to_hub:
        from datasets import Dataset
        token = get_hub_token(args.hub_token)
        Dataset.from_pandas(generated_df).push_to_hub(args.hub_dataset_id, token=token)
        print(f"Pushed to HuggingFace Hub: {args.hub_dataset_id}")


if __name__ == "__main__":
    main(tyro.cli(Args))
