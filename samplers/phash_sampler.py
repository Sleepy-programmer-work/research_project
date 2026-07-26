"""
samplers/phash_sampler.py — Standalone pHash benchmark sampler.

Exposes TASS Stage 1's perceptual-hash filtering as an independent benchmark
method.  Reuses the existing algorithmic helpers (is_degenerate, get_phash,
phash_distance) from tass_helpers.py — no code duplication.

Pipeline:
    Video → Frame Extraction → Degenerate Filter → pHash Filtering → Selected Frames

There is NO MobileCLIP, NO Greedy FPS, and NO Adaptive Early Stopping.
"""

import logging
import math
from pathlib import Path
from typing import List

import cv2
import numpy as np

from .base_sampler import BaseSampler
from .tass_helpers import is_degenerate, get_phash, phash_distance

logger = logging.getLogger(__name__)


class PHashSampler(BaseSampler):
    """Perceptual-hash scene-change sampler (standalone benchmark baseline).

    Accepts a frame when its pHash Hamming distance to the previously accepted
    frame exceeds a threshold (default: 1).  Operates at ~10 effective fps
    via stride-based frame skipping, matching TASS Stage 1 behaviour exactly.
    """

    _COMPARE_W = 256
    _COMPARE_H = 144
    _STAGE1_STRIDE = 3
    _MAX_CANDIDATES = 2000

    def __init__(self, hamming_threshold: int = 1) -> None:
        self.hamming_threshold = hamming_threshold
        self._compare_size = (self._COMPARE_W, self._COMPARE_H)
        logger.info(f"PHashSampler: hamming_threshold={self.hamming_threshold}")

    def get_name(self) -> str:
        return "phash"

    def sample(self, video_path: str) -> List[np.ndarray]:
        return self.sample_with_metadata(video_path)["frames"]

    def sample_with_metadata(self, video_path: str) -> dict:
        """Execute pHash-only filtering; return frames + telemetry dict."""
        path = Path(video_path)
        cap, fps, total_meta = self._open_video(path)
        if cap is None:
            return self._empty_result()

        candidates, indices, degenerate_count, frames_read = self._run_phash_filter(cap)
        cap.release()

        logger.info(
            f"[{self.get_name()}] {path.name}: {frames_read} read, "
            f"{len(candidates)} selected, {degenerate_count} degenerate."
        )

        if not candidates:
            if frames_read > 0:
                logger.warning(
                    f"[{self.get_name()}] pHash produced 0 candidates for "
                    f"{path.name} ({degenerate_count} degenerate) — "
                    f"falling back to interval sampling (fps1-style)."
                )
                candidates, indices = self._run_fallback(path, fps)
            if not candidates:
                logger.warning(f"[{self.get_name()}] No frames for {path.name}.")
                return self._empty_result(frames_read, degenerate_count)

        self._last_sampled_indices = indices

        return {
            "frames": candidates,
            "indices": indices,
            "meta": {
                "frames_original": frames_read,
                "candidate_pool_size": len(candidates),
                "frames_degenerate_dropped": degenerate_count,
                "tass_stopped_early": False,
                "vlm_calls": len(candidates),
                "fallback_used": False,
            },
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _open_video(self, path: Path):
        if not path.exists():
            logger.error(f"[{self.get_name()}] Video not found: {path}")
            return None, 30.0, 0
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            logger.error(f"[{self.get_name()}] Cannot open: {path}")
            return None, 30.0, 0
        raw_fps = cap.get(cv2.CAP_PROP_FPS)
        fps = raw_fps if raw_fps > 0 and raw_fps == raw_fps else 30.0
        total_meta = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        return cap, fps, total_meta

    def _run_phash_filter(self, cap):
        """Stream through frames at ~10 effective fps; apply degenerate + pHash."""
        candidates, indices = [], []
        prev_hash = None
        frame_idx = degenerate_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame is None or frame.size == 0:
                frame_idx += 1
                continue
            if frame_idx % self._STAGE1_STRIDE != 0:
                frame_idx += 1
                continue
            if is_degenerate(frame):
                degenerate_count += 1
                frame_idx += 1
                continue

            try:
                small = cv2.resize(frame, self._compare_size, interpolation=cv2.INTER_AREA)
                gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
            except cv2.error as exc:
                logger.warning(f"[{self.get_name()}] Resize failed at {frame_idx}: {exc}")
                frame_idx += 1
                continue

            curr_hash = get_phash(gray)

            if prev_hash is None:
                candidates.append(frame.copy())
                indices.append(frame_idx)
                prev_hash = curr_hash
                frame_idx += 1
                continue

            if phash_distance(prev_hash, curr_hash) > self.hamming_threshold:
                candidates.append(frame.copy())
                indices.append(frame_idx)
                prev_hash = curr_hash
                if len(candidates) >= self._MAX_CANDIDATES:
                    logger.warning(f"[{self.get_name()}] Hit MAX_CANDIDATES at frame {frame_idx}.")
                    while cap.read()[0]:
                        frame_idx += 1
                    break
            frame_idx += 1

        return candidates, indices, degenerate_count, frame_idx

    def _run_fallback(self, path: Path, fps: float):
        """FPS-1-style interval fallback when pHash produces zero candidates."""
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            logger.error(f"[{self.get_name()}] Fallback: cannot re-open {path}")
            return [], []

        frame_interval = max(1, int(round(fps)))
        candidates, indices = [], []
        frame_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame is None or frame.size == 0:
                frame_idx += 1
                continue
            if frame_idx % frame_interval == 0:
                candidates.append(frame.copy())
                indices.append(frame_idx)
                if len(candidates) >= self._MAX_CANDIDATES:
                    break
            frame_idx += 1

        cap.release()
        logger.info(f"[{self.get_name()}] Fallback: {len(candidates)} frames at interval={frame_interval}.")
        return candidates, indices

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _empty_result(frames_original: int = 0, degenerate_dropped: int = 0) -> dict:
        return {
            "frames": [], "indices": [],
            "meta": {
                "frames_original": frames_original,
                "candidate_pool_size": 0,
                "frames_degenerate_dropped": degenerate_dropped,
                "tass_stopped_early": False,
                "vlm_calls": 0,
                "fallback_used": False,
            },
        }
