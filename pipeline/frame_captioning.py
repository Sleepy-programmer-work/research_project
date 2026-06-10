import json
import logging
from pathlib import Path
from typing import List, Tuple
from PIL import Image
import numpy as np
from models.vlm_loader import VLMLoader
from config.settings import settings

logger = logging.getLogger(__name__)

def caption_frames(
    video_id: str,
    frames: List[np.ndarray],
    vlm: VLMLoader,
    method_name: str,
    vlm_model_id: str = "",
    vlm_revision: str = "",
) -> Tuple[List[str], bool]:
    """Generate or load cached per-frame captions for a video.

    Cache key format: {video_id}_{method_name}_{model_safe}_{revision_safe}.json
    The cache key includes the VLM model ID and revision so that switching model
    versions (e.g. moondream2@2024-08-26 → 2024-09-01) never silently reuses
    stale captions.

    Args:
        video_id:      Unique video identifier (used as cache key prefix).
        frames:        BGR numpy frames to caption.
        vlm:           Loaded VLMLoader instance.
        method_name:   Sampler name (e.g. 'fps1', 'ssim_090').
        vlm_model_id:  VLM model identifier (e.g. 'vikhyatk/moondream2').
                       Defaults to the value in settings.models['vlm']['name'].
        vlm_revision:  VLM revision/commit hash (e.g. '2024-08-26').
                       Defaults to the value in settings.models['vlm']['revision'].
                       If empty string, the revision component is omitted from the key.
    """
    cache_dir = Path(settings.experiment.get("cache_dir", "./cache")) / "frame_captions"
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Resolve model info from settings if not provided by caller.
    vlm_cfg = settings.models.get("vlm", {})
    if not vlm_model_id:
        vlm_model_id = vlm_cfg.get("name", "unknown_model")
    if not vlm_revision:
        vlm_revision = vlm_cfg.get("revision", "")

    # Sanitize for use in filenames (replace path-unsafe characters).
    model_safe = vlm_model_id.replace("/", "_").replace(":", "_").replace(" ", "_")
    revision_safe = vlm_revision.replace("/", "_").replace(":", "_").replace(" ", "_")

    # Build cache key — include revision only if one is specified.
    if revision_safe:
        cache_filename = f"{video_id}_{method_name}_{model_safe}_{revision_safe}.json"
    else:
        cache_filename = f"{video_id}_{method_name}_{model_safe}.json"

    cache_file = cache_dir / cache_filename

    if cache_file.exists():
        logger.debug(
            f"Loading cached frame captions for {video_id} ({method_name}) "
            f"from {cache_filename}"
        )
        with open(cache_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data["captions"], False  # cached — no new OOM recovery

    logger.debug(
        f"Generating frame captions for {video_id} ({method_name}) — "
        f"model={vlm_model_id}, revision={vlm_revision or 'none'}"
    )

    pil_images = []
    for frame in frames:
        rgb_frame = frame[:, :, ::-1]  # BGR → RGB
        pil_images.append(Image.fromarray(rgb_frame))

    batch_size = vlm_cfg.get("batch_size", 4)
    captions = vlm.generate_captions(pil_images, batch_size=batch_size)
    oom_triggered = vlm.oom_recovery_triggered

    data = {
        "video_id":     video_id,
        "method":       method_name,
        "vlm_model_id": vlm_model_id,
        "vlm_revision": vlm_revision,
        "captions":     captions,
    }
    with open(cache_file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    return captions, oom_triggered
