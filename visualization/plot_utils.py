"""
visualization/plot_utils.py — Shared style and chart helpers for benchmark plots.

Extracted from plots.py to keep every file under 200 LOC.
"""
import matplotlib.pyplot as plt
import seaborn as sns


# Publication-quality theme used across all figures
PALETTE = {"vlm_only": "#E76F51", "vlm_plus_llm": "#2A9D8F"}

RC_PARAMS = {
    "font.family": "sans-serif",
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "figure.titlesize": 14,
}


def apply_theme() -> None:
    """Apply the shared academic publication theme to matplotlib."""
    sns.set_theme(style="whitegrid", context="paper")
    plt.rcParams.update(RC_PARAMS)


def save_figure(fig_or_plt, path_no_ext, dpi: int = 300) -> None:
    """Save a figure as both PNG and PDF at publication quality."""
    for fmt in ("png", "pdf"):
        kwargs = {"dpi": dpi, "bbox_inches": "tight"}
        if hasattr(fig_or_plt, "savefig"):
            fig_or_plt.savefig(f"{path_no_ext}.{fmt}", **kwargs)
        else:
            fig_or_plt.savefig(f"{path_no_ext}.{fmt}", **kwargs)


def annotate_scatter_with_anticollision(ax, agg_df) -> None:
    """Annotate scatter points with collision-aware vertical stacking."""
    registry: dict = {}
    for _, row in agg_df.iterrows():
        x, y = row["processing_time_s"], row["cider"]
        label = f"{row['sampling_method']}_{row['aggregation_method']}"
        if row["caption_mode"] == "vlm_plus_llm":
            label += " (+LLM)"
        slot = round(x, 1)
        offset = 0
        if slot in registry:
            for last_y in registry[slot]:
                if abs(y - last_y) < 0.02:
                    offset += 1
            registry[slot].append(y)
        else:
            registry[slot] = [y]
        ax.annotate(
            label, (x, y),
            xytext=(8, 3 + offset * 12),
            textcoords="offset points",
            fontsize=8.5,
            alpha=0.85,
            weight="bold" if "LLM" in label else "normal",
        )
