"""
visualization/plot_frame_reduction.py — Fig 8: Frame-reduction box plot.

Extracted from plots.py to keep every file under 200 LOC.

Reads per-video JSON artefacts from results/frame_selection/ written by
save_frame_selection_meta() and plots reduction % by sampling method.
"""
import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from visualization.plot_utils import apply_theme

logger = logging.getLogger(__name__)

# Colour palette for benchmark methods
_METHOD_COLOURS = {
    "phash":         "#E76F51",
    "tass_adaptive": "#2A9D8F",
    "tass_fixed":    "#264653",
    "fps1":          "#ADB5BD",
    "fps2":          "#868E96",
    "random":        "#CED4DA",
}
_DEFAULT_COLOUR = "#ADB5BD"


def plot_frame_reduction(frame_selection_dir: Path, output_dir: Path) -> None:
    """Fig 8: Frame reduction % by sampling method (box plot)."""
    if not frame_selection_dir.exists():
        logger.warning(f"frame_selection directory not found: {frame_selection_dir} — skipping Fig 8")
        return

    records = _load_records(frame_selection_dir)
    if not records:
        logger.warning("No frame_selection metadata found — skipping Fig 8.")
        return

    df = pd.DataFrame(records)
    method_order = _sort_methods(df["method"].unique())
    data_by_method = [df[df["method"] == m]["reduction_pct"].values for m in method_order]

    apply_theme()
    fig, ax = plt.subplots(figsize=(10, 5))

    bp = ax.boxplot(
        data_by_method, labels=method_order, patch_artist=True,
        medianprops={"color": "black", "linewidth": 1.5},
    )
    _colour_boxes(bp["boxes"], method_order)

    ax.set_ylabel("Frame reduction (%)")
    ax.set_xlabel("Sampling method")
    ax.set_title(
        "Fig 8: Frame Reduction % by Sampling Method\n(per-method vs. original video frame count)",
        weight="bold", pad=12,
    )
    ax.set_ylim(bottom=0)
    sns.despine(ax=ax, left=True, bottom=False)
    plt.tight_layout()

    for fmt in ("png", "pdf"):
        fig.savefig(output_dir / f"fig8_frame_reduction.{fmt}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Fig 8 saved: {len(records)} records across {len(method_order)} methods ({method_order}).")


def _load_records(frame_selection_dir: Path) -> list:
    raw_metas = []
    video_orig_map = {}

    # First pass: load json files and build map of video_id -> original_frame_count
    for json_file in sorted(frame_selection_dir.glob("*.json")):
        try:
            meta = json.loads(json_file.read_text())
            raw_metas.append(meta)
            v_id = meta.get("video_id")
            orig = meta.get("original_frame_count")
            if v_id and orig and orig > 0:
                video_orig_map[v_id] = orig
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning(f"Could not read {json_file.name}: {exc}")
            continue

    records = []
    for meta in raw_metas:
        v_id = meta.get("video_id")
        orig = meta.get("original_frame_count") or video_orig_map.get(v_id)
        selected = meta.get("selected_frame_count")
        
        red_pct = meta.get("reduction_pct")
        if red_pct is None and orig and orig > 0 and selected is not None:
            red_pct = round((1.0 - selected / orig) * 100.0, 2)

        if red_pct is not None:
            records.append({
                "method": meta["sampler"],
                "reduction_pct": red_pct,
                "video_id": v_id,
            })
    return records


def _sort_methods(methods) -> list:
    """Sort methods: fps variants first, then phash, then tass, then others."""
    priority = {"fps1": 0, "fps2": 1, "random": 2, "phash": 3, "tass_fixed": 4, "tass_adaptive": 5}
    return sorted(methods, key=lambda m: (priority.get(m, 99), m))


def _colour_boxes(boxes, method_order: list) -> None:
    for patch, label in zip(boxes, method_order):
        colour = _METHOD_COLOURS.get(label, _DEFAULT_COLOUR)
        patch.set_facecolor(colour)
        patch.set_alpha(0.75)
