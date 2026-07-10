"""Flag suspect rows in a PROTAC-Splitter dataset / prediction CSV for manual review.

See `protac_splitter.data.curation.dataset_qc` for what is actually checked:
structural validity, chemical plausibility (BRENK, leaving groups, linker
topology, fragment size), and split-correctness proxies (similarity to known
E3/warhead ligands, agreement with the heuristic splitter, XGBoost edge
confidence). Nothing is deleted — rows get a `n_flags` count and a
`review_reasons` string, and the output CSV is sorted worst-first.

Usage:
    python scripts/qc_dataset.py --help
    python scripts/qc_dataset.py --input-csv tack_smiles_split_fast.csv --limit 200
    python scripts/qc_dataset.py --input-csv tack_smiles_split.csv --n-jobs 8
"""
from __future__ import annotations

import dataclasses
import logging
from pathlib import Path
from typing import Optional

import pandas as pd
import tyro
from joblib import Parallel, delayed
from tqdm import tqdm

from protac_splitter.data.curation.dataset_qc import qc_row, warm_caches


@dataclasses.dataclass
class Args:
    """Flag suspect rows (foreign molecules, unstable/artefact groups, implausible splits) for manual review."""

    input_csv: str = "tack_smiles_split_fast.csv"
    output_csv: Optional[str] = None
    """Defaults to `<input_csv stem>.qc.csv` next to the input file."""

    smiles_col: str = "SMILES"
    pred_col: str = "default_pred_n0"

    limit: Optional[int] = None
    """Only process the first N rows — useful for a quick smoke test."""

    n_jobs: int = 1

    run_heuristic_agreement: bool = True
    """Re-split every PROTAC with the betweenness-centrality heuristic and flag disagreement with the given split."""

    run_xgboost_confidence: bool = True
    """Score the XGBoost edge classifier's own decision margin on every PROTAC (downloads the model on first use)."""

    e3_similarity_threshold: float = 0.2
    poi_similarity_threshold: float = 0.2
    betweenness_threshold: float = 0.4
    agreement_similarity_threshold: float = 0.6
    """Below this per-fragment Tanimoto similarity to the heuristic split, flag `flag_method_disagreement`."""

    xgb_margin_threshold: float = 0.15


def main(args: Args) -> None:
    logging.basicConfig(level=logging.ERROR)

    input_path = Path(args.input_csv)
    if not input_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_path}")
    output_path = Path(args.output_csv) if args.output_csv else input_path.with_suffix("").with_suffix(".qc.csv")

    df = pd.read_csv(input_path)
    if args.limit is not None:
        df = df.head(args.limit)

    if args.smiles_col not in df.columns or args.pred_col not in df.columns:
        raise ValueError(f"CSV must contain columns '{args.smiles_col}' and '{args.pred_col}'. Found: {list(df.columns)}")

    # Warm the caches once in the main process before any forking, so worker
    # processes inherit them (copy-on-write) instead of each downloading /
    # rebuilding the E3 & warhead reference fingerprints and the XGBoost model.
    warm_caches(load_xgboost=args.run_xgboost_confidence)

    rows = list(zip(df[args.smiles_col], df[args.pred_col]))
    common_kwargs = dict(
        e3_sim_threshold=args.e3_similarity_threshold,
        poi_sim_threshold=args.poi_similarity_threshold,
        betweenness_threshold=args.betweenness_threshold,
        agreement_similarity_threshold=args.agreement_similarity_threshold,
        xgb_margin_threshold=args.xgb_margin_threshold,
        run_heuristic_agreement=args.run_heuristic_agreement,
        run_xgboost_confidence=args.run_xgboost_confidence,
    )

    if args.n_jobs > 1:
        results = Parallel(n_jobs=args.n_jobs)(
            delayed(qc_row)(smi, pred, **common_kwargs) for smi, pred in tqdm(rows, desc="QC")
        )
    else:
        results = [qc_row(smi, pred, **common_kwargs) for smi, pred in tqdm(rows, desc="QC")]

    qc_df = pd.DataFrame(results)
    out_df = pd.concat([df.reset_index(drop=True), qc_df], axis=1)
    out_df = out_df.sort_values("n_flags", ascending=False)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(output_path, index=False)

    print(f"\nQC'd {len(out_df):,} rows -> {output_path}")
    print(f"Rows with >=1 flag: {(out_df['n_flags'] > 0).sum():,} / {len(out_df):,}")
    print("\nFlag frequency:")
    flag_cols = [c for c in out_df.columns if c.startswith("flag_")]
    counts = out_df[flag_cols].sum().sort_values(ascending=False)
    for name, count in counts.items():
        print(f"  {name:<32} {int(count):>6}  ({count / len(out_df):.1%})")


if __name__ == "__main__":
    main(tyro.cli(Args))
