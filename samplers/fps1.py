import cv2
import numpy as np
from typing import List
from .base_sampler import BaseSampler

class FPS1Sampler(BaseSampler):
    def get_name(self) -> str:
        return "fps1"

    def sample(self, video_path: str) -> List[np.ndarray]:
        """Select 1 frame per second using the video's native FPS.

        Populates self._last_sampled_indices with actual video frame positions
        (not sequential 0,1,2,...) so frame_extraction.py can write correct
        temporal metadata to results/frame_selection/ JSON files.
        """
        result = self.sample_with_metadata(video_path)
        return result["frames"]

    def sample_with_metadata(self, video_path: str) -> dict:
        """Select 1 fps frames and return frames + metadata dict.

        Returns the same schema as TASSSampler.sample_with_metadata() so the
        benchmark loop can extract per-video frames_selected counts.
        """
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        frame_interval = max(1, int(round(fps)))
        frames = []
        indices = []
        frame_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % frame_interval == 0:
                frames.append(frame)
                indices.append(frame_idx)  # actual video frame position
            frame_idx += 1

        cap.release()
        self._last_sampled_indices = indices
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

