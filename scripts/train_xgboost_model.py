"""Train the XGBoost graph edge classifier for PROTAC splitting.

Downloads the PROTAC-Splitter dataset from HuggingFace Hub unless CSVs are
already present in the cache directory.

Usage:
    python scripts/train_xgboost_model.py --help
    python scripts/train_xgboost_model.py --output-model-path models/my_model.joblib
"""
from __future__ import annotations

import dataclasses
from pathlib import Path

import tyro

from scripts.common import ensure_output_dir


@dataclasses.dataclass
class Args:
    """Train an XGBoost edge classifier for PROTAC splitting."""

    output_model_path: str = "./models/PROTAC-Splitter-XGBoost.joblib"
    """Path to save the trained edge classifier model."""

    graph_datasets_cache_dir: str = "./data/graph_based/"
    """Directory to cache the extracted graph feature datasets."""

    hub_token: str = ""
    """HuggingFace token (defaults to HF_TOKEN in .env)."""

    num_proc: int = 8
    """Number of parallel processes for dataset mapping."""


def main(args: Args) -> None:
    import pandas as pd
    from datasets import load_dataset
    from protac_splitter.graphs.edge_classifier import train_edge_classifier

    cache_dir = Path(args.graph_datasets_cache_dir)
    ensure_output_dir(str(cache_dir))
    ensure_output_dir(str(Path(args.output_model_path).parent))

    train_csv = cache_dir / "train.csv"
    val_csv = cache_dir / "val.csv"
    test_csv = cache_dir / "test.csv"

    if train_csv.exists() and val_csv.exists() and test_csv.exists():
        print(f"Loading cached graph datasets from {cache_dir}")
        train_df = pd.read_csv(train_csv)
        val_df = pd.read_csv(val_csv)
        test_df = pd.read_csv(test_csv)
    else:
        print("Downloading PROTAC-Splitter dataset from HuggingFace Hub...")
        from scripts.common import get_hub_token
        token = get_hub_token(args.hub_token) if args.hub_token else None
        ds = load_dataset("ailab-bio/PROTAC-Splitter-Dataset", "clustered", token=token)

        def get_substructs(row: dict) -> dict:
            text = row["text"]
            parts = row["labels"].split(".")
            return {
                "PROTAC SMILES": text,
                "E3 Binder SMILES with direction": parts[0],
                "Linker SMILES with direction": parts[1],
                "POI Ligand SMILES with direction": parts[2],
            }

        ds = ds.map(get_substructs, num_proc=args.num_proc, remove_columns=["text", "labels"])
        train_df = ds["train"].to_pandas()
        val_df = ds["validation"].to_pandas()
        test_df = ds["test"].to_pandas()

    train_edge_classifier(
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        model_filename=args.output_model_path,
        cache_dir=str(cache_dir),
    )
    print(f"Edge classifier model saved to: {args.output_model_path}")


if __name__ == "__main__":
    main(tyro.cli(Args))
