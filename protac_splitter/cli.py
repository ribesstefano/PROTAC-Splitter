"""Command-line interface for PROTAC-Splitter.

Examples:
    protac-splitter --smiles "CC(C)..."
    protac-splitter --smiles "..." --model transformer
    protac-splitter --smiles "..." --model heuristic --betweenness-threshold 0.5
    protac-splitter --input-csv data.csv --smiles-col SMILES --output-csv out.csv
    protac-splitter --smiles-list "smi1" "smi2" --model heuristic --output-format csv
"""
from __future__ import annotations

import sys
import dataclasses
from typing import Optional, List, Literal

import tyro

SplittingModel = Literal["xgboost", "transformer", "transformer+xgboost", "heuristic"]

_MODEL_KWARGS = {
    "xgboost":             {"use_transformer": False, "use_xgboost": True},
    "transformer":         {"use_transformer": True,  "use_xgboost": False},
    "transformer+xgboost": {"use_transformer": True,  "use_xgboost": True},
    "heuristic":           {"use_transformer": False, "use_xgboost": False},
}


@dataclasses.dataclass
class SplitArgs:
    """Split one or more PROTAC SMILES strings into E3 / linker / POI substructures."""

    # --- Input: exactly one of these must be provided ---
    smiles: Optional[str] = None
    """Single PROTAC SMILES string to split."""

    smiles_list: Optional[List[str]] = None
    """Multiple PROTAC SMILES strings to split."""

    input_csv: Optional[str] = None
    """Path to a CSV file containing PROTAC SMILES. Use together with --smiles-col."""

    smiles_col: str = "text"
    """Column name for SMILES in the input CSV (default: 'text')."""

    output_csv: Optional[str] = None
    """Path to write output CSV (required when --input-csv is used)."""

    # --- Model selection ---
    model: SplittingModel = "xgboost"
    """Splitting strategy to use:

      xgboost             — XGBoost graph edge classifier (default; no GPU needed;
                            model is downloaded automatically on first use).
      transformer         — Seq2seq Transformer model hosted on HuggingFace
                            (requires the [transformer] extra; GPU recommended).
      transformer+xgboost — Transformer model with XGBoost as fallback when
                            Transformer predictions fail reassembly.
      heuristic           — Betweenness-centrality graph algorithm (no model needed).
    """

    # --- Transformer-specific options (ignored for xgboost / heuristic) ---
    fix_predictions: bool = True
    """Apply cheminformatics post-processing to Transformer predictions."""

    batch_size: int = 1
    """Inference batch size (Transformer only)."""

    beam_size: int = 5
    """Number of beam-search candidates to generate (Transformer only). Higher
    values may yield better results at the cost of more computation."""

    device: Optional[str] = None
    """Torch device string ('cpu', 'cuda', 'cuda:0'). Auto-detected if not set."""

    # --- Heuristic / graph-algorithm parameters ---
    betweenness_threshold: float = 0.4
    """Betweenness-centrality threshold for identifying bond split points
    (heuristic and XGBoost fallback). Range [0, 1]: higher values are more
    conservative and will produce fewer, larger fragments. Default: 0.4."""

    use_capacity_weight: bool = False
    """Weight graph edges by bond order (capacity) when computing betweenness
    centrality (heuristic algorithm only). May improve results for molecules
    with many aromatic rings."""

    betweenness_approx_frac: float = None
    """Fraction of nodes (0.0–1.0) to sample when approximating betweenness
    centrality. None (default) uses the exact algorithm. Lower values are
    faster but less accurate — e.g. 0.5 samples half the nodes."""

    # --- General ---
    n_jobs: int = 1
    """Number of parallel worker processes for XGBoost and heuristic splitting.
    Set to -1 to use all available CPUs. Default 1 (sequential)."""

    num_proc: int = 1
    """Number of parallel worker processes for the Transformer path."""

    verbose: int = 0
    """Verbosity level (0 = silent, 1 = info, 2 = debug)."""

    output_format: Literal["table", "csv"] = "table"
    """Output format when printing to stdout:
      table — human-readable aligned output.
      csv   — machine-parseable comma-separated values.
    """


def _print_result(result: dict, fmt: str) -> None:
    from protac_splitter.evaluation import split_prediction
    pred = result.get("default_pred_n0")
    parts = split_prediction(pred) if pred else {"e3": None, "linker": None, "poi": None}
    smiles_key = next(k for k in result if k not in ("default_pred_n0", "model_name"))
    if fmt == "csv":
        print(
            f"{result[smiles_key]},"
            f"{parts.get('e3')},"
            f"{parts.get('linker')},"
            f"{parts.get('poi')},"
            f"{result['model_name']}"
        )
    else:
        print(f"PROTAC : {result[smiles_key]}")
        print(f"Model  : {result['model_name']}")
        print(f"E3     : {parts.get('e3')}")
        print(f"Linker : {parts.get('linker')}")
        print(f"POI    : {parts.get('poi')}")
        print()


def main() -> None:
    args = tyro.cli(SplitArgs)

    from protac_splitter import split_protac
    import pandas as pd

    kwargs = dict(
        **_MODEL_KWARGS[args.model],
        fix_predictions=args.fix_predictions,
        batch_size=args.batch_size,
        beam_size=args.beam_size,
        device=args.device,
        n_jobs=args.n_jobs,
        num_proc=args.num_proc,
        verbose=args.verbose,
        betweenness_threshold=args.betweenness_threshold,
        use_capacity_weight=args.use_capacity_weight,
        betweenness_approx_frac=args.betweenness_approx_frac,
    )

    if args.input_csv is not None:
        df = pd.read_csv(args.input_csv)
        result_df = split_protac(df, protac_smiles_col=args.smiles_col, **kwargs)
        if args.output_csv:
            result_df.to_csv(args.output_csv, index=False)
            print(f"Results saved to {args.output_csv}")
        else:
            print(result_df.to_string(index=False))
    elif args.smiles_list is not None:
        if args.output_format == "csv":
            print("smiles,e3,linker,poi,model_name")
        for r in split_protac(args.smiles_list, **kwargs):
            _print_result(r, args.output_format)
    elif args.smiles is not None:
        if args.output_format == "csv":
            print("smiles,e3,linker,poi,model_name")
        _print_result(split_protac(args.smiles, **kwargs), args.output_format)
    else:
        print(
            "Error: provide one of --smiles, --smiles-list, or --input-csv.",
            file=sys.stderr,
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
