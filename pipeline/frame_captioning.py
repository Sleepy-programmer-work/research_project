import json
import logging
from pathlib import Path
from typing import List, Tuple
from PIL import Image
import numpy as np
from models.vlm_loader import VLMLoader
from config.settings import settings

logger = logging.getLogger(__name__)

def caption_frames(video_id: str, frames: List[np.ndarray], vlm: VLMLoader, method_name: str) -> Tuple[List[str], bool]:
    cache_dir = Path(settings.experiment.get("cache_dir", "./cache")) / "frame_captions"
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    cache_file = cache_dir / f"{video_id}_{method_name}.json"
    
    if cache_file.exists():
        logger.debug(f"Loading cached frame captions for {video_id} ({method_name})")
        with open(cache_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data["captions"], False # cached implies no new OOM recovery
            
    logger.debug(f"Generating frame captions for {video_id} ({method_name})")
    
    pil_images = []
    for frame in frames:
        rgb_frame = frame[:, :, ::-1] # BGR to RGB
        pil_images.append(Image.fromarray(rgb_frame))
        
    batch_size = settings.models.get("vlm", {}).get("batch_size", 4)
    captions = vlm.generate_captions(pil_images, batch_size=batch_size)
    oom_triggered = vlm.oom_recovery_triggered
    
    data = {
        "video_id": video_id,
        "method": method_name,
        "captions": captions
    }
    with open(cache_file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
        
    return captions, oom_triggered
