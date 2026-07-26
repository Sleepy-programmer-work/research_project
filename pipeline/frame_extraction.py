import json
from pathlib import Path
from typing import List
import numpy as np
from samplers.base_sampler import BaseSampler
from config.settings import settings

def extract_frames(video_path: str, video_id: str, sampler: BaseSampler) -> List[np.ndarray]:
    """Extract frames using the given sampler and write frame-selection metadata.

    Writes results/frame_selection/{video_id}_{sampler_name}.json with the
    actual video frame indices returned by the sampler.  Index[i] is the
    zero-based position in the source video file of frames[i] — NOT a
    sequential placeholder.

    Note: PHashSampler and TASSSampler are handled via their own
    sample_with_metadata() paths in run_benchmark.py and do not go through
    this function.  This function is only called for FPS1/FPS2/Random.
    """
    frames = sampler.sample(video_path)

    # Retrieve actual video frame indices populated by sampler.sample().
    # BaseSampler.get_last_sampled_indices() returns [] as a fail-safe if a
    # sampler subclass does not populate _last_sampled_indices — this produces
    # an empty list in the JSON rather than a misleading 0,1,2,... sequence.
    actual_indices = sampler.get_last_sampled_indices()

    # Sanity check: indices and frames must be parallel lists.
    if actual_indices and len(actual_indices) != len(frames):
        import logging
        logging.getLogger(__name__).warning(
            f"[{sampler.get_name()}] Index/frame count mismatch for {video_id}: "
            f"{len(actual_indices)} indices vs {len(frames)} frames. "
            f"Falling back to empty indices list."
        )
        actual_indices = []

    out_dir = Path(settings.experiment.get("output_dir", "./results")) / "frame_selection"
    out_dir.mkdir(parents=True, exist_ok=True)

    data = {
        "video_id": video_id,
        "method": sampler.get_name(),
        "selected_frames": actual_indices,  # actual video frame positions
        "selected_frame_count": len(frames),
    }

    with open(out_dir / f"{video_id}_{sampler.get_name()}.json", "w") as f:
        json.dump(data, f, indent=2)

    return frames
