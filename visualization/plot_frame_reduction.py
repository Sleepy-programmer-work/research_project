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

_SSIM_COLOURS = ["#E76F51", "#2A9D8F", "#264653"]


def plot_frame_reduction(frame_selection_dir: Path, output_dir: Path) -> None:
    """Fig 8: Frame reduction % by SSIM method (box plot)."""
    if not frame_selection_dir.exists():
        logger.warning(f"frame_selection directory not found: {frame_selection_dir} — skipping Fig 8")
        return

    records = _load_records(frame_selection_dir)
    if not records:
        logger.warning("No SSIM frame_selection metadata found — skipping Fig 8.")
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
        "Fig 8: Frame Reduction % by Sampling Method\n(SSIM variants vs. original video frame count)",
        weight="bold", pad=12,
    )
    ax.set_ylim(bottom=0)
    sns.despine(ax=ax, left=True, bottom=False)
    plt.tight_layout()

    for fmt in ("png", "pdf"):
        fig.savefig(output_dir / f"fig8_frame_reduction.{fmt}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Fig 8 saved: {len(records)} records, {len(method_order)} methods.")


def _load_records(frame_selection_dir: Path) -> list:
    records = []
    for json_file in sorted(frame_selection_dir.glob("*.json")):
        try:
            meta = json.loads(json_file.read_text())
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning(f"Could not read {json_file.name}: {exc}")
            continue
        if meta.get("reduction_pct") is not None:
            records.append({
                "method": meta["sampler"],
                "reduction_pct": meta["reduction_pct"],
                "video_id": meta.get("video_id", json_file.stem),
            })
    return records


def _sort_methods(methods) -> list:
    return sorted(
        methods,
        key=lambda m: (0, float(m.split("_")[1]) / 100) if m.startswith("ssim_") else (1, m),
    )


def _colour_boxes(boxes, method_order: list) -> None:
    ssim_idx = 0
    for patch, label in zip(boxes, method_order):
        colour = _SSIM_COLOURS[ssim_idx % len(_SSIM_COLOURS)] if label.startswith("ssim_") else "#ADB5BD"
        if label.startswith("ssim_"):
            ssim_idx += 1
        patch.set_facecolor(colour)
        patch.set_alpha(0.75)
