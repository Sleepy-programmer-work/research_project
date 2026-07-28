"""
evaluation/caption_io.py — File I/O for raw caption results.

metrics.py computes scores; caption_io.py writes caption artefacts to disk.
"""
import json
from pathlib import Path
from config.settings import settings


def save_raw_caption(
    video_id: str,
    method: str,
    agg: str,
    mode: str,
    gen: str,
    gt: list[str],
) -> None:
    """Write per-video raw caption and ground-truth to results/captions/."""
    out_dir = Path(settings.experiment.get("output_dir", "./results")) / "captions"
    out_dir.mkdir(parents=True, exist_ok=True)

    data = {
        "video_id": video_id,
        "method": method,
        "aggregation": agg,
        "mode": mode,
        "generated_caption": gen,
        "ground_truth": gt,
    }
    with open(out_dir / f"{video_id}_{method}_{agg}_{mode}.json", "w") as f:
        json.dump(data, f, indent=2)
