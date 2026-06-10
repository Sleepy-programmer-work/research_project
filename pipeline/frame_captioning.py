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

    The cache key is intentionally complete — it includes both the model ID and
    the pinned revision.  Omitting either field recreates the original bug where
    switching model versions silently reuses stale captions.

    STARTUP GUARD: If vlm_revision resolves to an empty string this function
    raises RuntimeError immediately rather than building a key without a revision
    component.  An incomplete key is indistinguishable from any other run that
    happened to use the same model name without a revision — exactly the bug
    we are fixing.

    Args:
        video_id:      Unique video identifier (used as cache key prefix).
        frames:        BGR numpy frames to caption.
        vlm:           Loaded VLMLoader instance.
        method_name:   Sampler name (e.g. 'fps1', 'ssim_090').
        vlm_model_id:  Override VLM model identifier. Defaults to settings.vlm_model_id.
        vlm_revision:  Override VLM revision/commit hash. Defaults to settings.vlm_revision.
                       Must never be empty — raises RuntimeError if it is.
    """
    cache_dir = Path(settings.experiment.get("cache_dir", "./cache")) / "frame_captions"
    cache_dir.mkdir(parents=True, exist_ok=True)

    vlm_cfg = settings.models.get("vlm", {})

    # Resolve model info — prefer explicit args, fall back to settings properties.
    # settings.vlm_model_id and settings.vlm_revision have hardcoded fallbacks
    # so they are always non-empty even if the YAML field is accidentally missing.
    if not vlm_model_id:
        vlm_model_id = settings.vlm_model_id
    if not vlm_revision:
        vlm_revision = settings.vlm_revision  # raises RuntimeError if empty

    # Guard: revision must be present — an empty revision corrupts the cache key.
    if not vlm_revision:
        raise RuntimeError(
            f"vlm_revision is empty for {video_id}/{method_name}. "
            f"Add 'revision: <commit-hash>' under models.vlm in benchmark.yaml. "
            f"An empty revision would produce cache keys identical to any other run, "
            f"causing silent reuse of stale captions across model versions."
        )

    # Sanitize for use in filenames (replace path-unsafe characters).
    model_safe = vlm_model_id.replace("/", "_").replace(":", "_").replace(" ", "_")
    revision_safe = vlm_revision.replace("/", "_").replace(":", "_").replace(" ", "_")

    # Cache key always includes both model and revision — never omit either.
    cache_filename = f"{video_id}_{method_name}_{model_safe}_{revision_safe}.json"
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
        f"model={vlm_model_id}@{vlm_revision}"
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
