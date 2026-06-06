import cv2
import numpy as np
from typing import List
from .base_sampler import BaseSampler

class FPS2Sampler(BaseSampler):
    def get_name(self) -> str:
        return "fps2"

    def sample(self, video_path: str) -> List[np.ndarray]:
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 30.0
            
        frame_interval = max(1, int(round(fps / 2.0)))
        frames = []
        frame_idx = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % frame_interval == 0:
                frames.append(frame)
            frame_idx += 1
            
        cap.release()
        return frames
