import cv2
import numpy as np
import random
from typing import List
from .base_sampler import BaseSampler
from .fps1 import FPS1Sampler

class RandomSampler(BaseSampler):
    def __init__(self, seed: int = 42):
        self.seed = seed

    def get_name(self) -> str:
        return "random"

    def sample(self, video_path: str) -> List[np.ndarray]:
        """Randomly sample ceil(duration) frames at a fixed seed.

        Populates self._last_sampled_indices with the actual randomly-selected
        video frame positions so frame_extraction.py can write correct temporal
        metadata to results/frame_selection/ JSON files.
        """
        result = self.sample_with_metadata(video_path)
        return result["frames"]

    def sample_with_metadata(self, video_path: str) -> dict:
        """Randomly sample frames and return frames + metadata dict.

        Returns the same schema as TASSSampler.sample_with_metadata() so the
        benchmark loop can extract per-video frames_selected counts.
        """
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()

        if total_frames <= 0 or fps <= 0:
            # Fallback: delegate to FPS1Sampler and inherit its indices
            fps1 = FPS1Sampler()
            result = fps1.sample_with_metadata(video_path)
            self._last_sampled_indices = fps1.get_last_sampled_indices()
            return result

        target_count = max(1, int(np.ceil(total_frames / fps)))

        random.seed(self.seed)
        np.random.seed(self.seed)

        if target_count >= total_frames:
            indices = list(range(total_frames))
        else:
            indices = sorted(random.sample(range(total_frames), target_count))

        cap = cv2.VideoCapture(video_path)
        frames = []
        frame_idx = 0
        target_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if target_idx < len(indices) and frame_idx == indices[target_idx]:
                frames.append(frame)
                target_idx += 1
            frame_idx += 1
            if target_idx >= len(indices):
                break

        cap.release()
        self._last_sampled_indices = indices  # actual video frame positions
        return {
            "frames": frames,
            "indices": indices,
            "meta": {
                "frames_original": frame_idx,
                "candidate_pool_size": len(frames),
                "frames_degenerate_dropped": 0,
                "tass_stopped_early": False,
                "vlm_calls": len(frames),
                "fallback_used": False,
            },
        }

