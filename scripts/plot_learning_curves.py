"""Plot comparison heatmaps of multiple PROTAC-Splitter model score files.

Reads all ``*scores.csv`` files from a directory and generates comparison
plots (heatmaps, bar charts) across models and metrics.

Usage:
    python scripts/plot_learning_curves.py --help
    python scripts/plot_learning_curves.py --scores-dir logs --output-dir plots
"""
from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import tyro

from scripts.common import ensure_output_dir


@dataclasses.dataclass
class Args:
    """Plot model comparison heatmaps from ``*scores.csv`` files."""

    scores_dir: str = "logs"
    """Directory containing ``*scores.csv`` files."""

    output_dir: str = "plots"
    """Directory to save output plot files."""

    metrics: Optional[str] = None
    """Comma-separated metric names to plot (default: reassembly, valid, all_ligands_equal)."""

    strategy: str = "default"
    """Generation strategy prefix to extract (e.g. 'default', 'beam_search_decoding')."""

    plot_heatmap: bool = True
    plot_bar: bool = True
    plot_venn: bool = False
    """Plot Venn diagram of correctly split PROTACs across models (requires matplotlib-venn)."""


def _load_scores(scores_dir: Path, strategy: str) -> pd.DataFrame:
    frames = []
    for f in sorted(scores_dir.glob("*scores.csv")):
        df = pd.read_csv(f)
        model_name = df["model_name"].iloc[0] if "model_name" in df.columns else f.stem
        pred_col = f"{strategy}_pred_n0"
        metric_cols = [c for c in df.columns if c.startswith(pred_col + "_") and "_fixed" not in c and "pred_n" not in c[len(pred_col) + 1:]]
        row = {"model": model_name}
        for col in metric_cols:
            metric = col[len(pred_col) + 1:]
            row[metric] = df[col].mean()
        frames.append(row)
    return pd.DataFrame(frames).set_index("model")


def main(args: Args) -> None:
    import seaborn as sns
    from matplotlib import pyplot as plt

    scores_dir = Path(args.scores_dir)
    if not scores_dir.exists():
        raise FileNotFoundError(f"Scores directory not found: {scores_dir}")
    out_dir = ensure_output_dir(args.output_dir)

    print(f"Loading score files from {scores_dir} (strategy='{args.strategy}')...")
    df = _load_scores(scores_dir, args.strategy)

    if df.empty:
        print("No score files found or no matching strategy columns.")
        return

    print(f"Found {len(df)} models, {len(df.columns)} metrics.")

    if args.metrics:
        keep = [m.strip() for m in args.metrics.split(",")]
        df = df[[c for c in df.columns if any(k in c for k in keep)]]

    if args.plot_heatmap:
        fig, ax = plt.subplots(figsize=(max(10, len(df.columns) * 0.8), max(6, len(df) * 0.6)))
        # Normalize each metric column to [0, 1] for uniform color scale
        norm_df = (df - df.min()) / (df.max() - df.min()).replace(0, 1)
        sns.heatmap(norm_df, annot=df.round(3), fmt="g", cmap="RdYlGn", ax=ax,
                    linewidths=0.5, cbar_kws={"label": "Normalized score"})
        ax.set_title("Model Comparison Heatmap")
        ax.set_xlabel("")
        ax.set_ylabel("")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        path = out_dir / "heatmap_model_comparison.pdf"
        plt.savefig(path, bbox_inches="tight")
        plt.close()
        print(f"Saved: {path}")

    if args.plot_bar:
        for metric in df.columns:
            fig, ax = plt.subplots(figsize=(8, max(4, len(df) * 0.5)))
            values = df[metric].sort_values(ascending=False)
            ax.barh(values.index, values.values, color="steelblue")
            for i, v in enumerate(values):
                ax.text(0.02, i, f"{v:.4f}", va="center", color="white", fontsize=9)
            ax.set_title(metric.replace("_", " ").title())
            ax.set_xlim(0, 1.05)
            ax.grid(axis="x", alpha=0.4)
            plt.tight_layout()
            safe_name = metric.replace("/", "-").replace(" ", "_")
            path = out_dir / f"bar_{safe_name}.pdf"
            plt.savefig(path, bbox_inches="tight")
            plt.close()
        print(f"Bar plots saved to {out_dir}")

    if args.plot_venn and len(df) >= 2:
        try:
            from matplotlib_venn import venn2, venn3
        except ImportError:
            print("matplotlib-venn not installed; skipping Venn diagram.")
            return

        score_files = list(scores_dir.glob("*scores.csv"))
        metric = "reassembly_fixed"
        sets = {}
        for f in score_files[:3]:
            raw = pd.read_csv(f)
            model = raw["model_name"].iloc[0] if "model_name" in raw.columns else f.stem
            col = f"{args.strategy}_pred_n0_{metric}"
            if col in raw.columns:
                sets[model] = set(raw[raw[col] == 1].index.tolist())

        if len(sets) == 2:
            names = list(sets.keys())
            plt.figure()
            venn2([sets[names[0]], sets[names[1]]], set_labels=names)
        elif len(sets) == 3:
            names = list(sets.keys())
            plt.figure()
            venn3([sets[n] for n in names], set_labels=names)
        else:
            print("Need 2 or 3 models for Venn diagram.")
            return
        path = out_dir / f"venn_{metric}.pdf"
        plt.savefig(path, bbox_inches="tight")
        plt.close()
        print(f"Saved Venn diagram: {path}")


if __name__ == "__main__":
    main(tyro.cli(Args))
