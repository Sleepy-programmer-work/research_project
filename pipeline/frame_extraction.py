import json
from pathlib import Path
from typing import List
import numpy as np
from samplers.base_sampler import BaseSampler
from config.settings import settings

def extract_frames(video_path: str, video_id: str, sampler: BaseSampler) -> List[np.ndarray]:
    frames = sampler.sample(video_path)
    
    out_dir = Path(settings.experiment.get("output_dir", "./results")) / "frame_selection"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Placeholder for actual indices, since samplers return raw frames.
    dummy_indices = [i for i in range(len(frames))] 
    
    data = {
        "video_id": video_id,
        "method": sampler.get_name(),
        "selected_frames": dummy_indices
    }
    
    with open(out_dir / f"{video_id}_{sampler.get_name()}.json", "w") as f:
        json.dump(data, f, indent=2)
        
    return frames
