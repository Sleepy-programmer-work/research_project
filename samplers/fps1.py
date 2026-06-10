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
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 30.0

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
        return frames
