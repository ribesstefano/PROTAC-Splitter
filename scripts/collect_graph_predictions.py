"""Collect graph-based (XGBoost or heuristic) predictions for PROTAC splitting.

Usage:
    python scripts/collect_graph_predictions.py --help
    python scripts/collect_graph_predictions.py --input-csv data/test.csv --output-csv logs/graph-preds.csv
"""
from __future__ import annotations

import dataclasses
from typing import Optional

import pandas as pd
import tyro
from datasets import Dataset

from protac_splitter.graphs.edge_classifier import GraphEdgeClassifier
from protac_splitter.graphs.clustering import get_representative_e3s_fp
from protac_splitter.graphs.splitting_algorithms import split_protac_graph_based


@dataclasses.dataclass
class Args:
    """Split PROTACs using graph-based algorithms and save predictions as CSV."""

    input_csv: str
    """Input CSV file with PROTAC SMILES."""

    output_csv: str
    """Output CSV file for predictions."""

    smiles_column: str = "text"
    labels_column: str = "labels"

    classifier_model: Optional[str] = None
    """Path to the edge classifier model (required when use_classifier=True)."""

    use_classifier: bool = True
    """Use the XGBoost edge classifier (default). Set to False for heuristic mode."""

    use_capacity_weight: bool = False
    betweenness_threshold: float = 0.4

    n_jobs: int = 1
    batch_size: int = 1


def main(args: Args) -> None:
    df = pd.read_csv(args.input_csv)
    print(f"Splitting {len(df)} PROTACs with graph-based algorithms...")

    if args.use_classifier and args.classifier_model is None:
        raise ValueError("--classifier-model is required when --use-classifier is set (default).")

    representative_e3s_fp = get_representative_e3s_fp(verbose=1)

    classifier = None
    if args.use_classifier:
        classifier = GraphEdgeClassifier.load(args.classifier_model)
        model_name = "GraphEdgeClassifier"
        print(f"Using edge classifier: {args.classifier_model}")
    else:
        model_name = "GraphHeuristic"
        print("Using heuristic algorithm without edge classifier.")

    def mapping_func(example):
        ret = split_protac_graph_based(
            protac_smiles=example[args.smiles_column],
            use_classifier=args.use_classifier,
            classifier=classifier,
            representative_e3s_fp=representative_e3s_fp,
            morgan_fp_generator=None,
            use_capacity_weight=args.use_capacity_weight,
            betweenness_threshold=args.betweenness_threshold,
        )
        return {
            "protac_smiles": example[args.smiles_column],
            "label_smiles": example[args.labels_column],
            "default_pred_n0": f"{ret['e3']}.{ret['linker']}.{ret['poi']}",
            "model_name": model_name,
        }

    out_df = (
        Dataset.from_pandas(df)
        .map(mapping_func, remove_columns=[args.smiles_column, args.labels_column], num_proc=args.n_jobs)
        .to_pandas()
    )
    out_df.to_csv(args.output_csv, index=False)
    print(f"Saved to: {args.output_csv}")


if __name__ == "__main__":
    main(tyro.cli(Args))
