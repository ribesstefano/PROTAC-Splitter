"""Command-line interface for PROTAC-Splitter.

Examples:
    protac-splitter --smiles "CC(C)..."
    protac-splitter --smiles "..." --model transformer
    protac-splitter --smiles "..." --model heuristic --betweenness-threshold 0.5
    protac-splitter --smiles "..." --model xgboost->heuristic
    protac-splitter --input-csv data.csv --smiles-col SMILES --output-csv out.csv
    protac-splitter --smiles-list "smi1" "smi2" --model heuristic --output-format csv
    protac-splitter serve --model heuristic --port 8778   # needs the [serve] extra
"""
from __future__ import annotations

import sys
import socket
import logging
import dataclasses
from pathlib import Path
from typing import Optional, List, Literal

import tyro

SplittingModel = Literal[
    "xgboost",
    "heuristic",
    "transformer",
    "transformer->xgboost",
    "xgboost->heuristic",
    "heuristic->xgboost",
    "xgboost+heuristic",
    "heuristic+xgboost",
    "adaptive",
]


@dataclasses.dataclass
class ModelArgs:
    """Which splitting strategy to run, and how. Shared by splitting and ``serve``."""

    # --- Model selection ---
    model: SplittingModel = "adaptive"
    """Splitting strategy to use:

      adaptive            — (default) QC-gated escalation: a heuristic parameter
                            grid, then XGBoost, then (if --adaptive-use-transformer)
                            the Transformer — each stage only runs on molecules the
                            previous stage left flagged. See --adaptive-use-*.
      xgboost             — XGBoost graph edge classifier (no GPU needed; model is
                            downloaded automatically on first use).
      heuristic           — Betweenness-centrality graph algorithm (no model needed).
      transformer         — Seq2seq Transformer model hosted on HuggingFace
                            (requires the [transformer] extra; GPU recommended).
      transformer->xgboost — Transformer first; XGBoost replaces failed predictions.
      xgboost->heuristic  — XGBoost first; heuristic replaces failed predictions.
      heuristic->xgboost  — Heuristic first; XGBoost replaces failed predictions.
      xgboost+heuristic   — Run both and pick the best result (not yet implemented).
      heuristic+xgboost   — Reserved for future use.
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

    # --- Adaptive-mode options (only used when --model adaptive) ---
    adaptive_use_xgboost: bool = True
    """Whether the XGBoost stage runs on molecules the heuristic grid left flagged."""

    adaptive_use_transformer: bool = False
    """Whether the Transformer stage runs on molecules still flagged after XGBoost.
    Default False: requires the [transformer] extra and a GPU is recommended. The
    custom heuristic-grid override (a list of (threshold, capacity) pairs) is
    Python-API only — use protac_splitter.split_protac(..., adaptive_heuristic_grid=...)
    for that."""

    # --- General ---
    num_proc: int = 1
    """Number of parallel worker processes."""

    verbose: int = 0
    """Verbosity level (0 = silent, 1 = info, 2 = debug)."""

    def split_kwargs(self) -> dict:
        """Keyword arguments for ``split_protac`` (everything but the input)."""
        return dict(
            model=self.model,
            fix_predictions=self.fix_predictions,
            batch_size=self.batch_size,
            beam_size=self.beam_size,
            device=self.device,
            num_proc=self.num_proc,
            verbose=self.verbose,
            betweenness_threshold=self.betweenness_threshold,
            use_capacity_weight=self.use_capacity_weight,
            betweenness_approx_frac=self.betweenness_approx_frac,
            adaptive_use_xgboost=self.adaptive_use_xgboost,
            adaptive_use_transformer=self.adaptive_use_transformer,
        )


@dataclasses.dataclass
class SplitArgs(ModelArgs):
    """Split one or more PROTAC SMILES strings into E3 / linker / POI substructures.

    Run `protac-splitter serve --help` for the HTTP server.
    """

    # --- Input: exactly one of these must be provided ---
    smiles: Optional[str] = None
    """Single PROTAC SMILES string to split."""

    smiles_list: Optional[List[str]] = None
    """Multiple PROTAC SMILES strings to split."""

    input_csv: Optional[str] = None
    """Path to a CSV file containing PROTAC SMILES. Use together with --smiles-col."""

    smiles_col: str = "SMILES"
    """Column name for SMILES in the input CSV (default: 'SMILES')."""

    output_csv: Optional[str] = None
    """Path to write output CSV (required when --input-csv is used)."""

    output_format: Literal["table", "csv"] = "table"
    """Output format when printing to stdout:
      table — human-readable aligned output.
      csv   — machine-parseable comma-separated values.
    """


@dataclasses.dataclass
class ServeArgs(ModelArgs):
    """Serve the chosen splitting strategy over HTTP (needs the [serve] extra).

    Takes the same strategy options as splitting; the models are loaded once at
    start-up and every request uses them.
    """

    host: str = "127.0.0.1"
    """Bind address. Loopback keeps the endpoint off the network: there is no
    authentication, so on a shared node 0.0.0.0 would expose it to every other user."""

    port: int = 8778
    """Bind port. Pass 0 to let the OS pick a free one, and combine with --port-file to
    tell the client which."""

    port_file: Optional[str] = None
    """Write the bound port to this file once listening (removed on shutdown). This is
    the handshake for --port 0, so several jobs on one node need no port bookkeeping."""

    log_level: str = "info"
    """uvicorn log level."""


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
        # Only present for model="adaptive" — see evaluation.score_split/count_flags.
        if "n_flags" in result:
            reasons = result.get("review_reasons") or "none"
            params = result.get("heuristic_params")
            print(f"Flags  : {result['n_flags']} ({reasons})" + (f" [{params}]" if params else ""))
        print()


def serve(argv: List[str]) -> None:
    """Run the HTTP server: ``protac-splitter serve [options]``."""
    args = tyro.cli(ServeArgs, prog="protac-splitter serve", args=argv)

    try:
        import uvicorn
        from protac_splitter.serve.app import create_app, load_service
    except ImportError as exc:
        raise SystemExit(
            f"protac-splitter serve needs the 'serve' extra ({exc}). Install it with:\n"
            '    pip install "protac-splitter[serve]"'
        ) from exc

    logging.basicConfig(level=args.log_level.upper(),
                        format="%(levelname)s:     %(name)s: %(message)s")
    try:
        service = load_service(args.split_kwargs())
    except (RuntimeError, ImportError) as exc:
        raise SystemExit(f"protac-splitter serve: {exc}") from exc
    app = create_app(service)
    # One worker: a second would load its own copy of every model.
    config = uvicorn.Config(app, log_level=args.log_level, workers=1)

    # Bind first so --port 0 resolves to the real port before it is published; uvicorn
    # alone would only log it.
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        sock.bind((args.host, args.port))
    except OSError as exc:
        raise SystemExit(
            f"Could not bind {args.host}:{args.port} ({exc}). Another server may already "
            "hold it; pass --port 0 --port-file PATH to pick a free one."
        ) from exc
    # Listen before publishing the port: bind() alone refuses connections, so a client
    # that reacts to the file immediately would get ECONNREFUSED. Connections made before
    # uvicorn starts wait in the backlog and are served once it does.
    sock.listen(socket.SOMAXCONN)
    port = sock.getsockname()[1]
    if args.port_file:
        port_path = Path(args.port_file)
        port_path.parent.mkdir(parents=True, exist_ok=True)
        port_path.write_text(str(port))
    print(f"Serving model={args.model} on http://{args.host}:{port} (docs at /docs)",
          flush=True)
    try:
        uvicorn.Server(config).run(sockets=[sock])
    finally:
        if args.port_file:
            Path(args.port_file).unlink(missing_ok=True)


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] == "serve":
        serve(sys.argv[2:])
        return

    args = tyro.cli(SplitArgs)

    from protac_splitter import split_protac
    import pandas as pd

    kwargs = args.split_kwargs()

    try:
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
    except ImportError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
