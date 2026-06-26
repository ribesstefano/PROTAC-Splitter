"""Collect combined Transformer + graph predictions for PROTAC splitting.

Uses the Transformer as primary predictor; falls back to graph-based splitting
when the Transformer output fails cheminformatics reassembly checks.

Usage:
    python scripts/collect_combined_predictions.py --help
"""
from __future__ import annotations

import dataclasses
import logging
from typing import Optional

import tyro

from scripts.common import ensure_output_dir, get_hub_token, load_dataset_or_csv


@dataclasses.dataclass
class Args:
    """Collect combined Transformer + graph fallback predictions."""

    output_csv: str = "combined_models-preds.csv"

    input_csv: Optional[str] = None
    """Local CSV file (skips HuggingFace Hub if provided)."""

    smiles_column: str = "text"
    labels_column: str = "labels"

    model_name: str = "ailab-bio/PROTAC-Splitter-EncoderDecoder-lr_reduce-rand-smiles"
    """HuggingFace seq2seq model name."""

    hub_token: Optional[str] = None
    cache_dir: Optional[str] = None

    dataset_dir: str = "ailab-bio/PROTAC-Splitter-Dataset"
    dataset_config: str = "clustered"
    dataset_test_split: str = "held_out"

    use_classifier: bool = False
    """Use XGBoost edge classifier as fallback (requires --classifier-model)."""

    classifier_model: Optional[str] = None
    use_capacity_weight: bool = False
    betweenness_threshold: float = 0.5

    batch_size: int = 1
    is_causal_language_model: bool = False
    num_proc: int = 1


def main(args: Args) -> None:
    logging.basicConfig(level=logging.ERROR)

    from protac_splitter.evaluation import check_reassembly
    from protac_splitter.graphs.clustering import get_representative_e3s_fp
    from protac_splitter.graphs.edge_classifier import GraphEdgeClassifier
    from protac_splitter.graphs.splitting_algorithms import split_protac_graph_based
    from protac_splitter.llms.model_utils import get_pipeline, run_pipeline
    from protac_splitter.protac_splitter import fix_prediction

    token = get_hub_token(args.hub_token)

    print("Loading dataset...")
    test_ds = load_dataset_or_csv(
        args.input_csv,
        hub_dataset_id=args.dataset_dir,
        hub_config=args.dataset_config,
        hub_split=args.dataset_test_split,
        hub_token=token,
        cache_dir=args.cache_dir,
    )

    pipe = get_pipeline(model_name=args.model_name, token=token, is_causal_language_model=args.is_causal_language_model)
    print(f"Pipeline loaded: {args.model_name}")

    preds = run_pipeline(pipe, test_ds, args.batch_size, args.is_causal_language_model, args.smiles_column)
    test_ds = test_ds.add_column("predictions", preds)

    representative_e3s_fp = get_representative_e3s_fp(verbose=1)

    classifier = None
    if args.use_classifier:
        if args.classifier_model is None:
            raise ValueError("--classifier-model is required when --use-classifier is set.")
        classifier = GraphEdgeClassifier.load(args.classifier_model)
        model_name = "TransformerAndGraphEdgeClassifier"
        print(f"Using edge classifier: {args.classifier_model}")
    else:
        model_name = "TransformerAndGraphHeuristic"
        print("Using heuristic fallback.")

    def mapping_func(example):
        protac_smiles = example[args.smiles_column]
        pred_smiles = example["predictions"]["pred_n0"]
        fixed_smiles = fix_prediction(protac_smiles, pred_smiles)
        if not check_reassembly(protac_smiles, fixed_smiles):
            ret = split_protac_graph_based(
                protac_smiles=protac_smiles,
                use_classifier=args.use_classifier,
                classifier=classifier,
                representative_e3s_fp=representative_e3s_fp,
                morgan_fp_generator=None,
                use_capacity_weight=args.use_capacity_weight,
                betweenness_threshold=args.betweenness_threshold,
            )
            fixed_smiles = f"{ret['e3']}.{ret['linker']}.{ret['poi']}"
        return {
            "protac_smiles": protac_smiles,
            "label_smiles": example[args.labels_column],
            "default_pred_n0": fixed_smiles,
            "model_name": model_name,
        }

    print("Applying graph fallback to failed Transformer predictions...")
    out_df = (
        test_ds.map(
            mapping_func,
            remove_columns=[args.smiles_column, args.labels_column, "predictions"],
            num_proc=args.num_proc,
        ).to_pandas()
    )
    out_df.to_csv(args.output_csv, index=False)
    print(f"Saved to: {args.output_csv}")


if __name__ == "__main__":
    main(tyro.cli(Args))
