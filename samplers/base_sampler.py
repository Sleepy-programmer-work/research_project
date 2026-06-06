from abc import ABC, abstractmethod
from typing import List
import numpy as np

class BaseSampler(ABC):
    @abstractmethod
    def sample(self, video_path: str) -> List[np.ndarray]:
        """Return sampled frames as a list of numpy arrays (BGR, HxWxC)."""
        pass

    @abstractmethod
    def get_name(self) -> str:
        """Return a short identifier string, e.g. 'fps1', 'random'."""
        pass

    def get_frame_count(self, video_path: str) -> int:
        """Return number of frames this sampler would select."""
        return len(self.sample(video_path))
