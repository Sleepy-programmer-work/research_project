import numpy as np
from typing import List
from .base_sampler import BaseSampler

class DSISSampler(BaseSampler):
    def get_name(self) -> str:
        return "dsis"

    def sample(self, video_path: str) -> List[np.ndarray]:
        # Placeholder for future Phase 2 implementation
        return []
