"""Score PROTAC-Splitter predictions stored as CSV files.

Two modes:

* Logs mode (default): reads all ``*preds.csv`` files in the log directory,
  which already carry a ``label_smiles`` column, scores each prediction
  column, and writes matching ``*scores.csv`` files.
* Reference mode (``--predictions``): scores prediction CSVs written by the
  ``protac-splitter`` CLI (``PROTAC SMILES``, ``default_pred_n0``,
  ``model_name``) against a reference label CSV (``PROTAC SMILES``,
  ``Warhead SMILES``, ``Linker SMILES``, ``E3 Ligase Ligand SMILES``), writes
  one ``*_scores.csv`` per prediction file, and summarizes all models into
  two top-1 tables: overall metrics and metrics per substructure.

Usage:
    python scripts/score_predictions.py --help
    python scripts/score_predictions.py --log-dir logs
    python scripts/score_predictions.py \\
        --labels Datasets/smiles/dataset-curated-held-out.csv \\
        --predictions outputs/predictions_xgboost.csv outputs/predictions_heuristic.csv outputs/predictions_chemprop.csv \\
        --rename predictions_transformer_nofix=Transformer predictions_transformer=Transformer-Δ
"""
from __future__ import annotations

import dataclasses
import logging
from pathlib import Path
from typing import Optional

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

    predictions: tuple[str, ...] = ()
    """Prediction CSVs from the ``protac-splitter`` CLI. When set, score them
    against ``--labels`` instead of scanning ``--log-dir``."""

    labels: str = "Datasets/smiles/dataset-curated-held-out.csv"
    """Reference label CSV, used with ``--predictions``."""

    output_dir: Optional[str] = None
    """Where to write scores and summary tables in reference mode. Defaults to
    the directory of the first prediction file."""

    pred_column: str = "default_pred_n0"
    """Column holding the top-1 prediction in each prediction CSV."""

    rename: tuple[str, ...] = ()
    """Display names for the summary tables, as ``file_stem=Name`` entries, e.g.
    ``predictions_transformer=Transformer-Δ``. Files without an entry keep the
    ``model_name`` column of their CSV. Needed when two files share a
    ``model_name``, such as the Transformer with and without ``--fix-predictions``."""


SUBSTRUCTURES = {"poi": "Warhead", "linker": "Linker", "e3": "E3 Ligand"}


def add_merge_key(df: pd.DataFrame) -> pd.DataFrame:
    """Add an occurrence index disambiguating repeated canonical PROTAC SMILES.

    Canonicalization standardizes salts and charges, so two dataset rows that
    differ only in protonation collapse to the same ``protac_smiles`` while
    still carrying different reference splits. Pairing labels with predictions
    on the SMILES alone is then ambiguous, so repeated SMILES are matched in
    file order, which is how the CLI writes its predictions.
    """
    df["occurrence"] = df.groupby("protac_smiles").cumcount()
    return df


def load_labels(labels_path: str) -> pd.DataFrame:
    """Return ``protac_smiles``/``label_smiles`` keyed by canonical PROTAC SMILES."""
    labels = pd.read_csv(labels_path)
    labels["protac_smiles"] = labels["PROTAC SMILES"].apply(canonize)
    labels["label_smiles"] = (
        labels["E3 Ligase Ligand SMILES"] + "." + labels["Linker SMILES"] + "." + labels["Warhead SMILES"]
    ).apply(canonize)
    n_ambiguous = labels["protac_smiles"].duplicated().sum()
    if n_ambiguous:
        print(
            f"WARNING: {n_ambiguous} labelled PROTACs share a canonical SMILES with an earlier row "
            f"(same molecule after standardization, different reference split); matched in file order."
        )
    return add_merge_key(labels)[["protac_smiles", "occurrence", "label_smiles"]]


def score_against_labels(predictions_path: Path, labels: pd.DataFrame, args: Args) -> pd.DataFrame:
    """Score the top-1 predictions of one CLI prediction CSV against the labels."""
    preds = pd.read_csv(predictions_path)
    preds["protac_smiles"] = preds["PROTAC SMILES"].apply(canonize)
    preds = add_merge_key(preds)
    # Left merge on the labels so every model is scored over the same set of
    # PROTACs: a PROTAC a model produced no row for counts as a failed split,
    # exactly like the empty cells the CLI writes for splits it could not make.
    df = labels.merge(preds, on=["protac_smiles", "occurrence"], how="left", validate="one_to_one")
    n_unlabelled = (~preds["protac_smiles"].isin(labels["protac_smiles"])).sum()
    if n_unlabelled:
        print(f"WARNING: {n_unlabelled} PROTACs in {predictions_path} have no reference label and are skipped.")
    n_missing = df[args.pred_column].isna().sum()
    if n_missing:
        print(f"WARNING: {n_missing} labelled PROTACs have no prediction in {predictions_path}; scored as failures.")
    renames = dict(entry.split("=", 1) for entry in args.rename)
    if predictions_path.stem in renames:
        model_name = renames[predictions_path.stem]
    else:
        model_name = preds["model_name"].iloc[0] if "model_name" in preds.columns else predictions_path.stem
    # Failed splits are empty cells in the CLI output; score them as missing predictions
    df["pred_smiles"] = df[args.pred_column].where(df[args.pred_column].notna(), None)
    df = df[["protac_smiles", "label_smiles", "pred_smiles"]]

    def score_row(row: dict) -> dict:
        scores = score_prediction(
            protac_smiles=row["protac_smiles"],
            label_smiles=row["label_smiles"],
            pred_smiles=row["pred_smiles"],
        )
        return {k: v for k, v in scores.items() if "tanimoto" not in k and "graph_edit" not in k}

    ds = Dataset.from_pandas(df, preserve_index=False)
    scores = pd.DataFrame(ds.map(score_row, num_proc=args.num_proc))
    scores.insert(0, "model_name", model_name)
    return scores


def summarize(scores: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build the top-1 overall and per-substructure tables, one column per model.

    Extra atoms in the overall table is the mean number of PROTAC heavy atoms
    missing from the reassembled prediction, averaged only over predictions
    that fail reassembly (0 when none fail). Per substructure, it is the mean
    absolute heavy-atom difference between predicted and reference
    substructure over all predictions.
    """
    overall, per_sub = {}, {}
    for model_name, df in scores.groupby("model_name", sort=False):
        failed = df[~df["reassembly"].astype(bool)]
        overall[model_name] = {
            ("", "Validity"): df["valid"].mean(),
            ("", "Reassembly"): df["reassembly"].mean(),
            ("", "Exact-match"): df["all_ligands_equal"].mean(),
            ("", "Extra atoms"): failed["heavy_atoms_difference"].mean() if len(failed) else 0.0,
        }
        per_sub[model_name] = {}
        for sub, sub_name in SUBSTRUCTURES.items():
            per_sub[model_name].update({
                (sub_name, "Validity"): df[f"{sub}_valid"].mean(),
                (sub_name, "Exact-match"): df[f"{sub}_equal"].mean(),
                (sub_name, "Extra atoms"): df[f"{sub}_heavy_atoms_difference"].abs().mean(),
            })
    overall_df = pd.DataFrame(overall).droplevel(0)
    per_sub_df = pd.DataFrame(per_sub)
    per_sub_df.index.names = ["Substructure", "Metric"]
    return overall_df, per_sub_df


def format_table(table: pd.DataFrame) -> pd.DataFrame:
    """Render rates as percentages and extra atoms as plain numbers."""
    def fmt(metric: str, value: float) -> str:
        return f"{value:.2f}" if metric == "Extra atoms" else f"{value:.2%}"

    metrics = table.index.get_level_values(-1)
    return pd.DataFrame(
        [[fmt(m, v) for v in row] for m, row in zip(metrics, table.to_numpy())],
        index=table.index,
        columns=table.columns,
    )


def main_reference(args: Args) -> None:
    prediction_paths = [Path(p) for p in args.predictions]
    for p in prediction_paths:
        if not p.exists():
            raise FileNotFoundError(f"Predictions file not found: {p}")
    output_dir = Path(args.output_dir) if args.output_dir else prediction_paths[0].parent
    output_dir.mkdir(parents=True, exist_ok=True)

    labels = load_labels(args.labels)
    n_labels = len(labels)
    print(f"Loaded {n_labels} reference labels from {args.labels}\n")

    all_scores = []
    for predictions_path in prediction_paths:
        scores_path = output_dir / f"{predictions_path.stem}_scores.csv"
        if args.skip_if_log_exists and scores_path.exists():
            print(f"Loading existing scores: {scores_path}")
            all_scores.append(pd.read_csv(scores_path))
            continue
        print(f"Scoring {predictions_path} → {scores_path}")
        scores = score_against_labels(predictions_path, labels, args)
        scores.to_csv(scores_path, index=False)
        all_scores.append(scores)

    overall, per_sub = summarize(pd.concat(all_scores, ignore_index=True))
    overall.to_csv(output_dir / "scores_summary_overall.csv")
    per_sub.to_csv(output_dir / "scores_summary_substructures.csv")

    dataset_name = f"{Path(args.labels).stem} ({n_labels:,})"
    print(f"\nTop-1 evaluation metrics on {dataset_name}:\n")
    print(format_table(overall).rename_axis("Metric").to_markdown())
    print(f"\nTop-1 evaluation metrics per substructure on {dataset_name}:\n")
    print(format_table(per_sub).reset_index().to_markdown(index=False))
    print(f"\nSummary tables saved to {output_dir}/scores_summary_{{overall,substructures}}.csv")


def main(args: Args) -> None:
    logging.basicConfig(level=logging.ERROR)

    if args.predictions:
        main_reference(args)
        return

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
