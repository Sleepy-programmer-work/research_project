from abc import ABC, abstractmethod
from typing import List
import numpy as np

class BaseSampler(ABC):
    """Abstract base for all frame sampling strategies.

    Contract:
      - sample() returns BGR frames as a list of numpy arrays.
      - get_last_sampled_indices() returns the actual video frame indices
        (0-based position in the source video) of each frame returned by the
        most recent sample() call.  The list is parallel to sample()'s return:
        get_last_sampled_indices()[i] is the video frame index of frames[i].

    CRITICAL: Subclasses MUST keep _last_sampled_indices in sync with the
    frames returned by sample().  Using sequential 0,1,2,... placeholders is
    wrong — they corrupt temporal analysis and frame-selection JSON metadata.
    """

    # Subclasses write their actual indices here after each sample() call.
    _last_sampled_indices: List[int]

    @abstractmethod
    def sample(self, video_path: str) -> List[np.ndarray]:
        """Return sampled frames as a list of numpy arrays (BGR, HxWxC).

        Implementation requirement: populate self._last_sampled_indices with
        the actual video frame indices (position in the source file) of each
        returned frame before returning.
        """
        pass

    @abstractmethod
    def get_name(self) -> str:
        """Return a short identifier string, e.g. 'fps1', 'random'."""
        pass

    def get_last_sampled_indices(self) -> List[int]:
        """Return actual video frame indices from the most recent sample() call.

        Returns an empty list if sample() has not been called yet or if the
        subclass does not populate _last_sampled_indices (fail-safe default).
        """
        return getattr(self, "_last_sampled_indices", [])

    def get_frame_count(self, video_path: str) -> int:
        """Return number of frames this sampler would select."""
        return len(self.sample(video_path))
