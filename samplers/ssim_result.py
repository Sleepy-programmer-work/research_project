"""
samplers/ssim_result.py — SSIMSamplerResult dataclass.

Extracted from ssim.py to keep every source file under 200 LOC.
"""
from dataclasses import dataclass
from typing import List, Optional
import numpy as np


@dataclass
class SSIMSamplerResult:
    """Full output from SSIMSampler.sample_with_metadata().

    Invariants (TASS Stage 1 depends on them):
      - frames contains full-resolution BGR NumPy arrays (not thumbnails).
      - frame_indices preserves the original temporal position in the video
        so that TASS Stage 2 can reconstruct a monotonically ordered sequence.
      - len(frames) == len(frame_indices) == len(ssim_scores) always holds.
    """
    frames: List[np.ndarray]
    frame_indices: List[int]
    ssim_scores: List[Optional[float]]
    original_frame_count: int
    total_frames_meta: int          # from cv2.CAP_PROP_FRAME_COUNT (may be inaccurate)
    selected_frame_count: int
    reduction_pct: float
    average_ssim: float
    threshold_used: float
    fps: float
    fallback_used: bool = False

    @classmethod
    def _empty(cls) -> "SSIMSamplerResult":
        return cls(
            frames=[], frame_indices=[], ssim_scores=[], original_frame_count=0,
            total_frames_meta=0, selected_frame_count=0, reduction_pct=0.0,
            average_ssim=0.0, threshold_used=0.0, fps=0.0, fallback_used=False,
        )

    def __getitem__(self, key: str):
        if key == "frames":
            return self.frames
        elif key == "indices":
            return self.frame_indices
        elif key == "meta":
            return {
                "frames_original": self.original_frame_count,
                "candidate_pool_size": self.selected_frame_count,
                "frames_degenerate_dropped": 0,
                "tass_stopped_early": False,
                "vlm_calls": self.selected_frame_count,
                "fallback_used": self.fallback_used,
            }
        raise KeyError(key)

