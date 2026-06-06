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
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        
        if total_frames <= 0 or fps <= 0:
            return FPS1Sampler().sample(video_path)
            
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
        return frames
