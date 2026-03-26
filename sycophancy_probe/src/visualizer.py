"""
visualizer.py — Plot per-layer probe metrics and sycophancy breakdowns.

Outputs:
  - results/probe_accuracy.png   — accuracy + AUROC vs. layer depth
  - results/breakdown_*.png      — sycophancy rate by domain, tone, trigger layer
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns

logger = logging.getLogger(__name__)

# Use a clean non-interactive backend
import matplotlib
matplotlib.use("Agg")


def plot_probe_metrics(metrics_df: pd.DataFrame, output_dir: Path) -> None:
    """
    Line plot of accuracy and AUROC vs. layer index.
    Shaded bands show ± 1 std dev.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Linear Probe Performance per Transformer Layer", fontsize=14, fontweight="bold")

    layers = metrics_df["layer"].values

    for ax, metric, color in zip(
        axes,
        [("accuracy_mean", "accuracy_std"), ("auroc_mean", "auroc_std")],
        ["steelblue", "darkorange"],
    ):
        mean_col, std_col = metric
        mean = metrics_df[mean_col].values
        std = metrics_df[std_col].values

        ax.plot(layers, mean, color=color, linewidth=2, label=mean_col.split("_")[0].upper())
        ax.fill_between(layers, mean - std, mean + std, alpha=0.2, color=color)
        ax.axhline(0.5, color="gray", linestyle="--", linewidth=1, label="Chance (0.5)")

        ax.set_xlabel("Layer", fontsize=11)
        ax.set_ylabel(mean_col.split("_")[0].capitalize(), fontsize=11)
        ax.set_title(f"{mean_col.split('_')[0].upper()} vs. Layer", fontsize=12)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(layers[0], layers[-1])
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))

    plt.tight_layout()
    out_path = output_dir / "probe_accuracy.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved probe accuracy plot → {out_path}")


def plot_sycophancy_breakdown(
    breakdown_df: pd.DataFrame,
    group_key: str,
    output_dir: Path,
    title_suffix: str = "",
) -> None:
    """Bar plot of sycophancy rate grouped by a categorical variable."""
    if breakdown_df.empty:
        logger.warning(f"No data to plot for breakdown: {group_key}")
        return

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(max(6, len(breakdown_df) * 1.2), 5))

    colors = sns.color_palette("Set2", n_colors=len(breakdown_df))
    bars = ax.bar(
        breakdown_df[group_key].astype(str),
        breakdown_df["sycophancy_rate"],
        color=colors,
        edgecolor="white",
        linewidth=0.8,
    )

    # Annotate counts above bars
    for bar, count in zip(bars, breakdown_df["count"]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"n={count}",
            ha="center",
            va="bottom",
            fontsize=8,
            color="gray",
        )

    ax.set_xlabel(group_key.replace("_", " ").title(), fontsize=11)
    ax.set_ylabel("Sycophancy Rate", fontsize=11)
    ax.set_title(
        f"Sycophancy Rate by {group_key.replace('_', ' ').title()}{title_suffix}",
        fontsize=12,
        fontweight="bold",
    )
    ax.set_ylim(0, 1.1)
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=1, alpha=0.5)
    ax.grid(True, axis="y", alpha=0.3)
    plt.xticks(rotation=30, ha="right")

    plt.tight_layout()
    out_path = output_dir / f"breakdown_{group_key}.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved breakdown plot → {out_path}")


def plot_all_breakdowns(labeled_samples, output_dir: Path) -> None:
    """Generate breakdown plots for domain, tone, and trigger layers."""
    from src.probe_trainer import compute_breakdown_stats

    for key in ["domain", "tone", "layer1", "layer2", "layer3", "context_type"]:
        df = compute_breakdown_stats(labeled_samples, key)
        plot_sycophancy_breakdown(df, key, output_dir)


def print_summary(metrics_df: pd.DataFrame, labeled_samples) -> None:
    """Print a concise summary to stdout."""
    valid = [s for s in labeled_samples if s.label is not None]
    syco = [s for s in valid if s.label == 1]
    unparseable = [s for s in labeled_samples if s.label is None]

    print("\n" + "=" * 60)
    print("SYCOPHANCY PROBE SUMMARY")
    print("=" * 60)
    print(f"Total samples:      {len(labeled_samples)}")
    print(f"Valid (labeled):    {len(valid)}")
    print(f"Unparseable:        {len(unparseable)}")
    print(f"Sycophantic (1):    {len(syco)} ({100*len(syco)/max(len(valid),1):.1f}%)")
    print(f"Non-syco    (0):    {len(valid)-len(syco)} ({100*(len(valid)-len(syco))/max(len(valid),1):.1f}%)")
    print()

    best_acc_row = metrics_df.loc[metrics_df["accuracy_mean"].idxmax()]
    best_auc_row = metrics_df.loc[metrics_df["auroc_mean"].idxmax()]
    print(f"Best accuracy:  Layer {int(best_acc_row['layer'])}  acc={best_acc_row['accuracy_mean']:.3f}±{best_acc_row['accuracy_std']:.3f}")
    print(f"Best AUROC:     Layer {int(best_auc_row['layer'])}  auc={best_auc_row['auroc_mean']:.3f}±{best_auc_row['auroc_std']:.3f}")
    print("=" * 60 + "\n")
