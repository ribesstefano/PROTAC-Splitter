"""Plot PROTAC-Splitter prediction scores from a scores CSV file.

Usage:
    python scripts/plot_scores.py --help
    python scripts/plot_scores.py --score-file logs/model-scores.csv
"""
from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import seaborn as sns
import tyro
from matplotlib import pyplot as plt
from matplotlib import ticker as mtick
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from protac_splitter.display_utils import get_mapped_protac_img
from protac_splitter.evaluation import split_prediction
from scripts.common import ensure_output_dir


@dataclasses.dataclass
class Args:
    """Plot prediction score metrics from a scores CSV file."""

    score_file: str
    """Path to the scores CSV file (``*scores.csv`` from score_predictions.py)."""

    best_strategy: str = "beam_search_decoding"
    """Generation strategy to highlight in top-5 plots."""

    img_dir: Optional[str] = None
    """Directory to save plot images. If None, shows plots interactively."""

    failed_metric_to_display: str = "default_pred_n0_reassembly_fixed"
    """Column to filter for failed-prediction molecule images."""

    print_failed_predictions: bool = False


def process_score_file(csv_file: str, top_metric: str = "reassembly") -> pd.DataFrame:
    df = pd.read_csv(csv_file)
    model_name = df["model_name"].iloc[0] if "model_name" in df.columns else csv_file
    strategies = set(col.split("_pred_n")[0] for col in df.columns if "_pred_n" in col)
    rows = []
    for strategy in strategies:
        strategy_columns = [col for col in df.columns if col.startswith(strategy + "_pred_n")]
        num_preds = len(set(col.split("_pred_n")[1][0] for col in strategy_columns))
        metric_names = set(col.split("_pred_n")[1][2:] for col in strategy_columns)
        metric_names.discard("")
        if "search" in strategy or num_preds > 1:
            best_pred = df[[f"{strategy}_pred_n{i}_{top_metric}" for i in range(num_preds)]].idxmax(axis=1)
            df["best_pred_idx"] = best_pred.apply(lambda x: int(x.split("_pred_n")[-1][0]))
        for metric in metric_names:
            top_1 = df[f"{strategy}_pred_n0_{metric}"]
            top_n = df.apply(lambda x: x[f"{strategy}_pred_n{x['best_pred_idx']}_{metric}"], axis=1) if num_preds > 1 else None
            base = {"model_name": model_name, "strategy": strategy, "metric": metric, "fixed": "fixed" in metric}
            for red in ("mean", "max", "min", "std"):
                rows.append({**base, "top_n": 1, "reduce": red, "value": top_1.dropna().agg(red)})
            if top_n is not None:
                for red in ("mean", "max", "min", "std"):
                    rows.append({**base, "top_n": num_preds, "reduce": red, "value": top_n.dropna().agg(red)})
    return pd.DataFrame(rows)


def _save_or_show(img_dir: Optional[Path], filename: str) -> None:
    if img_dir:
        path = img_dir / filename
        plt.savefig(path, bbox_inches="tight")
        plt.clf()
        plt.close()
        print(f"Saved: {path}")
    else:
        plt.show()


def main(args: Args) -> None:
    img_dir = ensure_output_dir(args.img_dir) if args.img_dir else None

    scores_df = pd.read_csv(args.score_file)
    processed_df = process_score_file(args.score_file)

    if processed_df["strategy"].nunique() > 1:
        print("-" * 80 + "\nCompare generation strategies\n" + "-" * 80)
        for metric in ("valid", "reassembly", "reassembly_nostereo", "all_ligands_equal", "heavy_atoms_difference"):
            df = processed_df.copy()
            df = df[(df["metric"] == metric) | (df["metric"] == metric + "_fixed")]
            df = df[(df["reduce"] == "mean") & (df["top_n"] == 1)]
            df["value"] = df["value"].astype(float)
            df["strategy"] = df["strategy"].str.split("_").str.join(" ").str.title()
            df["fixed"] = df["fixed"].replace({True: "Fixed", False: "Not Fixed"})
            plt.figure(figsize=(8, 6))
            ax = sns.barplot(data=df, y="strategy", x="value", hue="fixed")
            plt.title(f"{' '.join(metric.split('_')).title()} - Top-1")
            plt.ylabel("")
            plt.xlabel("")
            plt.grid(axis="x", alpha=0.5)
            for p in ax.patches:
                if p.get_width() == 0:
                    continue
                text = f"{p.get_width():.2%}" if metric != "heavy_atoms_difference" else f"{p.get_width():.2f}"
                ax.annotate(text, (0.05, p.get_y() + p.get_height() / 2), ha="left", va="center", fontsize=10, color="white")
            if metric != "heavy_atoms_difference":
                plt.xlim(0, 1.05)
                plt.gca().xaxis.set_major_formatter(mtick.PercentFormatter(1))
            plt.legend(title=None, loc="lower center", bbox_to_anchor=(0.5, -0.15), ncol=2)
            _save_or_show(img_dir, f"plot_{metric}_top1.pdf")

    print("-" * 80 + "\nTop-1 mean metrics\n" + "-" * 80)
    for ligand in ("protac", "e3", "poi", "linker"):
        df = scores_df.copy()
        cols = [c for c in df.columns if c.startswith("default_pred_n")]
        df = df[cols]
        df.columns = df.columns.str.replace("default_pred_n0_", "")
        df = df[df.columns[~df.columns.str.contains("num_fragments|has_three_substructures|is_fixed|graph")]]
        df = df[df.columns[~df.columns.str.contains("default_pred_n0")]]
        if ligand == "protac":
            df = df[df.columns[~df.columns.str.contains("e3|poi|linker")]]; title = "Top-1 Mean PROTAC Metrics"
        elif ligand == "e3":
            df = df[df.columns[df.columns.str.contains("e3_", regex=False)]]; title = "Top-1 Mean E3 Metrics"
        elif ligand == "poi":
            df = df[df.columns[df.columns.str.contains("poi_", regex=False)]]; title = "Top-1 Mean Warhead Metrics"
        else:
            df = df[df.columns[df.columns.str.contains("linker_", regex=False)]]; title = "Top-1 Mean Linker Metrics"
        df.columns = df.columns.str.split("_").str.join(" ").str.title().str.replace("Fixed", "(fixed)")
        df = df.reindex(sorted(df.columns, key=lambda x: (x.split(" ")[-1] != "(fixed)", x), reverse=True), axis=1)
        df = df.astype(float)
        ax = sns.barplot(data=df, orient="h", errorbar="ci", color="C0")
        plt.xlabel(""); plt.ylabel(""); plt.title(title); plt.xlim(0, 1.1); plt.grid(axis="x", alpha=0.5)
        for p in ax.patches:
            ax.annotate(f"{p.get_width():.4f}", (0.05, p.get_y() + p.get_height() / 2), ha="left", va="center", fontsize=10, color="black")
        plt.gca().xaxis.set_major_formatter(mtick.PercentFormatter(1))
        _save_or_show(img_dir, f"plot_{ligand}_top1.pdf")

    if not processed_df[processed_df["top_n"] == 5].empty:
        print("-" * 80 + "\nTop-5 mean metrics\n" + "-" * 80)
        for ligand in ("protac", "e3", "poi", "linker"):
            df = processed_df[processed_df["strategy"] == args.best_strategy].copy()
            df = df[(df["reduce"] == "mean") & (df["top_n"] == 5)]
            df = df[~df["metric"].str.contains("graph|is_fixed|num_fragments|has_three_substructures|has_all_attachment_points", regex=True)]
            if ligand == "protac":
                df = df[~(df["metric"].str.contains("e3_|poi_|linker"))]; title = "Top-5 Mean PROTAC Metrics"
            elif ligand == "e3":
                df = df[df["metric"].str.contains("e3_")]; title = "Top-5 Mean E3 Metrics"
            elif ligand == "poi":
                df = df[df["metric"].str.contains("poi_")]; title = "Top-5 Mean Warhead Metrics"
            else:
                df = df[df["metric"].str.contains("linker_")]; title = "Top-5 Mean Linker Metrics"
            df["metric"] = df["metric"].str.split("_").str.join(" ").str.title().str.replace("Fixed", "(fixed)")
            df["value"] = df["value"].astype(float)
            df = df.sort_values("metric", key=lambda x: x.str.contains("(fixed)", regex=False))
            ax = sns.barplot(data=df, orient="h", color="C0", x="value", y="metric")
            plt.xlabel(""); plt.ylabel(""); plt.title(title); plt.xlim(0, 1.1); plt.grid(axis="x", alpha=0.5)
            for p in ax.patches:
                ax.annotate(f"{p.get_width():.4f}", (0.05, p.get_y() + p.get_height() / 2), ha="left", va="center", fontsize=10, color="black")
            plt.gca().xaxis.set_major_formatter(mtick.PercentFormatter(1))
            _save_or_show(img_dir, f"plot_{ligand}_top5.pdf")

    if "default_prob_n0" in scores_df.columns:
        print("-" * 80 + "\nPerplexity scores\n" + "-" * 80)
        for x_metric in ("default_prob_n0", "default_perplexity_n0"):
            for hue_metric in ("default_pred_n0_reassembly", "default_pred_n0_all_ligands_equal"):
                x_label = "Perplexity" if "perplexity" in x_metric else "Probability"
                hue_label = "Reassembly" if "reassembly" in hue_metric else "All Ligands Equal"
                plt.figure(figsize=(10, 5))
                sns.histplot(data=scores_df, x=x_metric, hue=hue_metric, kde=True)
                plt.xlabel(x_label); plt.ylabel("Frequency"); plt.title(f"Histogram: {x_label} vs {hue_label}"); plt.grid(alpha=0.5)
                _save_or_show(img_dir, f"plot_{x_metric}_vs_{hue_metric}.pdf")
                plt.figure(figsize=(10, 5))
                sns.histplot(data=scores_df, x=x_metric, hue=hue_metric, kde=True)
                plt.xlabel(x_label); plt.ylabel("Frequency"); plt.ylim(0, 300)
                _save_or_show(img_dir, f"plot_{x_metric}_vs_{hue_metric}_zoom.pdf")
                plt.figure(figsize=(10, 5))
                sns.scatterplot(data=scores_df, x=x_metric, y=hue_metric)
                plt.xlabel(x_label); plt.grid(alpha=0.5)
                _save_or_show(img_dir, f"plot_{x_metric}_vs_{hue_metric}_scatter.pdf")

        X = scores_df["default_prob_n0"].values.reshape(-1, 1)
        y = scores_df["default_pred_n0_reassembly"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        model = LogisticRegressionCV(cv=5, random_state=42, max_iter=1000, tol=1e-7, Cs=100).fit(X_train, y_train)
        dummy = DummyClassifier(strategy="most_frequent").fit(X_train, y_train)
        print(f"Accuracy: {model.score(X_test, y_test):.2f} (dummy: {dummy.score(X_test, y_test):.2f})")
        print(f"F1:       {f1_score(y_test, model.predict(X_test)):.2f} (dummy: {f1_score(y_test, dummy.predict(X_test)):.2f})")
        print(f"ROC-AUC:  {roc_auc_score(y_test, model.predict(X_test)):.2f}")
        threshold = -model.intercept_[0] / model.coef_[0][0]
        print(f"Confidence threshold: {threshold:.4f}")

    if img_dir is None:
        return

    def split_row(row):
        ligands = split_prediction(row["default_pred_n0"])
        label_ligands = split_prediction(row["label_smiles"])
        return {**row.to_dict(), "e3": ligands["e3"], "linker": ligands["linker"], "poi": ligands["poi"],
                "e3_label": label_ligands["e3"], "linker_label": label_ligands["linker"], "poi_label": label_ligands["poi"]}

    failed_df = scores_df[~scores_df[args.failed_metric_to_display]]
    tqdm.pandas(desc="Splitting failed PROTACs")
    failed_df = failed_df.progress_apply(split_row, axis=1, result_type="expand")

    for i, row in tqdm(failed_df.sample(n=len(failed_df), random_state=42).iterrows(), total=len(failed_df)):
        ligands = split_prediction(row["label_smiles"])
        if args.print_failed_predictions:
            print(f"{i:5d} PROTAC:  {row['protac_smiles']}")
            print(f"{i:5d} Ligands: {row['label_smiles']}")
            print(f"{i:5d} Pred:    {row['default_pred_n0']}")
        svg = get_mapped_protac_img(
            row["protac_smiles"], e3_smiles=ligands["e3"], linker_smiles=ligands["linker"],
            poi_smiles=ligands["poi"], w=1000, h=600, legend="", useSVG=True, display_image=False,
        )
        out_file = img_dir / f"image_{args.failed_metric_to_display}_n{i}.svg"
        if svg:
            out_file.write_text(svg)


if __name__ == "__main__":
    main(tyro.cli(Args))
