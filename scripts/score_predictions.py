"""Score PROTAC-Splitter predictions stored as CSV files in a logs directory.

Reads all ``*preds.csv`` files in the log directory, scores each prediction
column, and writes matching ``*scores.csv`` files.

Usage:
    python scripts/score_predictions.py --help
    python scripts/score_predictions.py --log-dir logs
"""
from __future__ import annotations

import dataclasses
import logging
from pathlib import Path

import pandas as pd
import tyro
from datasets import Dataset

from protac_splitter.chemoinformatics import canonize
from protac_splitter.evaluation import score_prediction
from protac_splitter.protac_splitter import fix_prediction


@dataclasses.dataclass
class Args:
    """Score all ``*preds.csv`` files found in a logs directory."""

    log_dir: str = "logs"
    """Directory containing ``*preds.csv`` prediction files."""

    num_proc: int = 16
    skip_if_log_exists: bool = False
    """Skip scoring when the corresponding ``*scores.csv`` already exists."""


def main(args: Args) -> None:
    logging.basicConfig(level=logging.ERROR)

    logs_dir = Path(args.log_dir)
    if not logs_dir.exists():
        raise FileNotFoundError(f"Log directory not found: {logs_dir}")

    predictions_files = list(logs_dir.glob("*preds.csv"))
    if not predictions_files:
        raise FileNotFoundError(f"No predictions CSV files found in {logs_dir}")

    print(f"Found {len(predictions_files)} predictions CSV files:")
    for f in predictions_files:
        print(f"  - {f}")
    print()

    for predictions_path in predictions_files:
        print("-" * 80)
        print(f"Model: {predictions_path.stem}")
        print("-" * 80)
        scores_path = predictions_path.with_name(predictions_path.stem.replace("preds", "scores") + ".csv")

        if args.skip_if_log_exists and scores_path.exists():
            print(f"Skipping — scores already exist: {scores_path}")
            continue
        print(f"Scoring {predictions_path} → {scores_path}")

        df = pd.read_csv(predictions_path)
        df = df[~df["label_smiles"].str.contains(r"\.\[Cl-\]\.")]
        df["protac_smiles"] = df["protac_smiles"].apply(canonize)
        df["label_smiles"] = df["label_smiles"].apply(canonize)

        ds = Dataset.from_pandas(df, preserve_index=False)

        def score_multiple_predictions(row: dict) -> dict:
            scores = {"protac_smiles": row["protac_smiles"], "label_smiles": row["label_smiles"]}
            protac_smiles = row["protac_smiles"]
            label_smiles = row["label_smiles"]
            for pred_name, pred_smiles in row.items():
                if pred_name in ("protac_smiles", "label_smiles", "model_name") or "pred_n" not in pred_name:
                    continue
                curr = score_prediction(
                    protac_smiles=protac_smiles,
                    label_smiles=label_smiles,
                    pred_smiles=pred_smiles,
                    compute_graph_metrics=True,
                    graph_edit_kwargs={"timeout": 0.1},
                )
                metric_names = list(curr.keys())
                curr = {f"{pred_name}_{m}": v for m, v in curr.items()}
                if pred_smiles == label_smiles:
                    curr.update({f"{k}_fixed": v for k, v in curr.items()})
                    curr[f"{pred_name}_is_fixed"] = True
                    scores.update(curr)
                    continue
                fixed = fix_prediction(protac_smiles, pred_smiles)
                curr.update({f"{pred_name}_{m}_fixed": curr[f"{pred_name}_{m}"] for m in metric_names})
                if fixed is None:
                    curr[f"{pred_name}_is_fixed"] = False
                elif fixed == pred_smiles:
                    curr[f"{pred_name}_is_fixed"] = True
                else:
                    fixed_scores = score_prediction(
                        protac_smiles=protac_smiles,
                        label_smiles=label_smiles,
                        pred_smiles=fixed,
                        compute_graph_metrics=True,
                        graph_edit_kwargs={"timeout": 0.1},
                    )
                    curr.update({f"{pred_name}_{m}_fixed": v for m, v in fixed_scores.items()})
                    curr[f"{pred_name}_is_fixed"] = True
                curr["model_name"] = row["model_name"]
                scores.update(curr)
            return {k: v for k, v in scores.items() if "tanimoto" not in k}

        scores = ds.map(score_multiple_predictions, num_proc=args.num_proc)
        pd.DataFrame(scores).to_csv(scores_path, index=False)
        print(f"Scores saved: {scores_path}\n")


if __name__ == "__main__":
    main(tyro.cli(Args))
