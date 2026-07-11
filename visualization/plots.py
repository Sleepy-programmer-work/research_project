"""
visualization/plots.py — Benchmark figure generation.

Produces Figs 1-8 from the per-video benchmark CSV.

Fig 1-6: Grouped bar charts (metric × pipeline config × caption mode)
Fig 7:   Quality/latency Pareto scatter
Fig 8:   Frame-reduction box plot for SSIM variants (see plot_frame_reduction)
"""
import logging
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from config.settings import settings
from visualization.plot_utils import (
    PALETTE, apply_theme, save_figure, annotate_scatter_with_anticollision
)
from visualization.plot_frame_reduction import plot_frame_reduction

logger = logging.getLogger(__name__)

_BAR_METRICS = {
    "cider":             ("fig1_cider",  "Mean CIDEr Score"),
    "bleu4":             ("fig2_bleu4",  "Mean BLEU-4 Score"),
    "processing_time_s": ("fig3_time",   "Average Processing Time (s)"),
    "peak_vram_mb":      ("fig4_vram",   "Peak VRAM Footprint (MB)"),
    "peak_ram_delta_mb": ("fig5_ram",    "Peak RAM Delta Footprint (MB)"),
    "frames_selected":   ("fig6_frames", "Selected Frame Count"),
    "semantic_yield":    ("fig9_yield",  "Semantic Yield (CIDEr / Frame)"),
}


def generate_plots(csv_path: str, out_dir: Path = None, frame_selection_dir: Path = None):
    """Generate all benchmark figures from the raw per-video CSV."""
    df = pd.read_csv(csv_path)
    if df.empty:
        return

    if out_dir is None:
        out_dir = Path(settings.experiment.get("output_dir", "./results"))

    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    df["pipeline_config"] = df["sampling_method"] + " + " + df["aggregation_method"]
    apply_theme()

    _plot_bar_charts(df, plots_dir)
    _plot_scatter_pareto(df, plots_dir)

    if frame_selection_dir is not None:
        plot_frame_reduction(frame_selection_dir=frame_selection_dir, output_dir=plots_dir)


def _plot_bar_charts(df: pd.DataFrame, plots_dir: Path) -> None:
    """Figs 1-6: grouped ablation-study bar charts."""
    for metric, (filename, ylabel) in _BAR_METRICS.items():
        if metric not in df.columns:
            continue
        plt.figure(figsize=(10, 5), dpi=300)
        sns.barplot(
            data=df, x="pipeline_config", y=metric, hue="caption_mode",
            palette=PALETTE, capsize=0.05, errorbar=("ci", 95),
            edgecolor="black", linewidth=0.7,
        )
        plt.xticks(rotation=25, ha="right")
        plt.title(f"Comparative Architectural Metrics: {ylabel}", pad=12, weight="bold")
        plt.xlabel("Upstream Frame Aggregation Pipeline", labelpad=6)
        plt.ylabel(ylabel)
        sns.despine(left=True, bottom=False)
        plt.legend(title="Inference Framework", loc="upper right", frameon=True)
        plt.tight_layout()
        save_figure(plt, plots_dir / filename)
        plt.close()


def _plot_scatter_pareto(df: pd.DataFrame, plots_dir: Path) -> None:
    """Fig 7: Quality (CIDEr) vs. Latency Pareto frontier scatter."""
    if "processing_time_s" not in df.columns or "cider" not in df.columns:
        return
    agg_df = (
        df.groupby(["sampling_method", "aggregation_method", "caption_mode"])[
            ["processing_time_s", "cider"]
        ]
        .mean()
        .reset_index()
        .sort_values(by=["processing_time_s", "cider"])
        .reset_index(drop=True)
    )
    fig, ax = plt.subplots(figsize=(10, 6.5), dpi=300)
    sns.scatterplot(
        ax=ax, data=agg_df, x="processing_time_s", y="cider",
        hue="caption_mode", palette=PALETTE, s=120,
        edgecolor="black", alpha=0.9, zorder=3,
    )
    annotate_scatter_with_anticollision(ax, agg_df)
    ax.set_title("Quality (CIDEr) vs. System Efficiency (Latency Frontier)", pad=15, weight="bold")
    ax.set_xlabel("Average Pipeline Processing Latency (seconds)", labelpad=6)
    ax.set_ylabel("Mean Corpus CIDEr Evaluation Score", labelpad=6)
    ax.set_xlim(-0.5, agg_df["processing_time_s"].max() + 1.5)
    ax.set_ylim(-0.02, agg_df["cider"].max() + 0.04)
    sns.despine(ax=ax)
    ax.legend(title="Framework Configuration", loc="center right", frameon=True)
    plt.tight_layout()
    save_figure(fig, plots_dir / "fig7_scatter")
    plt.close(fig)
