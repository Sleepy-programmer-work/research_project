"""
samplers/tass.py — Two-Stage Adaptive Semantic Sampling (TASS) Engine.

Architecture overview:
  Stage 1 — Streaming Degenerate Purge + pHash Pre-filter  (CPU, O(N))
    1a. Degenerate Frame Detector: drops flash/fade/lens-cap frames.
    1b. pHash Perceptual Filter: accepts frames where the Hamming distance
        between perceptual hashes exceeds a threshold.

  Stage 2 — Micro-Batched MobileCLIP + Greedy Farthest-Point Sampling (CPU, O(M))
    2a. MobileCLIP-S1 encodes each Stage-1 candidate into a 512-d embedding.
    2b. Greedy FPS selects the K most semantically diverse frames.
    2c. Adaptive early stopping halts when further additions are redundant.

Modes:
  'fixed':    Selects exactly K = ceil(duration_seconds) frames.
  'adaptive': Selects until the candidate pool is exhausted or early-stop fires.

Pure algorithmic helpers (is_degenerate, get_phash, phash_distance, greedy_fps) live in
tass_helpers.py for unit-testability independent of the sampler class.
"""

import gc
import logging
import math
from pathlib import Path
from typing import List

import cv2
import numpy as np

from .base_sampler import BaseSampler
from .tass_helpers import is_degenerate, get_phash, phash_distance, greedy_fps
from models.clip_embedder import MobileCLIPEmbedder

logger = logging.getLogger(__name__)


class TASSSampler(BaseSampler):
    """Two-Stage Adaptive Semantic Sampling (TASS) sampler."""

    _COMPARE_W = 256
    _COMPARE_H = 144
    _GRID = 2
    _MAX_CANDIDATES = 2000
    # Stage 1 operates at ~10 effective fps (every 3rd frame of a typical 30fps
    # source).  This is sufficient to capture scene transitions above 100ms
    # duration while reducing I/O and pHash computation by ~3×.
    _STAGE1_STRIDE = 3

    def __init__(self, mode: str = "fixed", threshold: float = 0.90,
                 min_distance: float = 0.10, clip_batch_size: int = 16) -> None:
        if not 0.0 < threshold < 1.0:
            raise ValueError(f"TASS threshold must be in (0, 1), got {threshold}.")
        if not 0.0 <= min_distance < 1.0:
            raise ValueError(f"min_distance must be in [0, 1), got {min_distance}.")
        self.mode = mode.strip().lower()
        if self.mode not in ("fixed", "adaptive"):
            raise ValueError(f"mode must be 'fixed' or 'adaptive', got '{mode}'.")
        self.threshold = threshold
        self.min_distance = min_distance
        self.clip_batch_size = clip_batch_size
        self._compare_size = (self._COMPARE_W, self._COMPARE_H)
        self._embedder: MobileCLIPEmbedder | None = None
        logger.info(f"TASSSampler: mode={self.mode}, threshold={self.threshold}, min_distance={self.min_distance}")

    def get_name(self) -> str:
        return f"tass_{self.mode}"

    def sample(self, video_path: str) -> List[np.ndarray]:
        return self.sample_with_metadata(video_path)["frames"]

    def sample_with_metadata(self, video_path: str) -> dict:
        """Execute the full two-stage TASS pipeline; return frames + telemetry dict."""
        path = Path(video_path)
        cap, fps, total_meta, target_k = self._open_and_measure(path)
        if cap is None:
            return self._empty_result()

        candidates, indices, degenerate_count, frames_read = self._run_stage1(cap, path)
        cap.release()
        logger.info(f"[{self.get_name()}] Stage 1: {frames_read} read, "
                    f"{len(candidates)} candidates, {degenerate_count} degenerate.")

        fallback_used = False
        if not candidates:
            if frames_read > 0:
                # Stage 1 produced zero candidates — either pHash rejected all
                # non-degenerate frames (static content, Hamming distance ≤1) or
                # all strided frames were degenerate (blurry / dark throughout).
                # Fall back to fps1-style interval sampling with NO degenerate
                # filter, matching fps1 baseline behaviour exactly.
                logger.warning(
                    f"[{self.get_name()}] Stage 1 produced 0 candidates for "
                    f"{path.name} ({degenerate_count} degenerate) — falling "
                    f"back to interval sampling (fps1-style)."
                )
                candidates, indices = self._run_stage1_fallback(path, fps)
                fallback_used = True
            if not candidates:
                logger.warning(f"[{self.get_name()}] No candidates after Stage 1 for {path.name}.")
                return self._empty_result(frames_read, degenerate_count)

        final_frames, final_indices, stopped_early = self._run_stage2(candidates, indices, target_k)
        logger.info(f"[{self.get_name()}] Stage 2: {len(final_frames)} frames selected "
                    f"(stopped_early={stopped_early}, fallback={fallback_used}).")

        return {
            "frames": final_frames, "indices": final_indices,
            "meta": {
                "frames_original": frames_read,
                "candidate_pool_size": len(candidates),
                "frames_degenerate_dropped": degenerate_count,
                "tass_stopped_early": stopped_early,
                "vlm_calls": len(final_frames),
                "fallback_used": fallback_used,
            },
        }

    # ------------------------------------------------------------------
    # Stage helpers
    # ------------------------------------------------------------------

    def _open_and_measure(self, path: Path):
        if not path.exists():
            logger.error(f"[{self.get_name()}] Video not found: {path}")
            return None, 30.0, 0, 1
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            logger.error(f"[{self.get_name()}] Cannot open: {path}")
            return None, 30.0, 0, 1
        raw_fps = cap.get(cv2.CAP_PROP_FPS)
        fps = raw_fps if raw_fps > 0 and raw_fps == raw_fps else 30.0
        total_meta = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = max(1.0, total_meta / fps) if total_meta > 0 else 1.0
        target_k = max(1, math.ceil(duration))
        logger.info(f"[{self.get_name()}] {path.name}: fps={fps:.1f}, "
                    f"total={total_meta}, dur={duration:.1f}s, k={target_k} (mode={self.mode})")
        return cap, fps, total_meta, target_k

    def _run_stage1(self, cap, path: Path):
        """Stream through frames at ~10 effective fps; apply degenerate filter + pHash."""
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
            # Skip frames between stride boundaries to reduce I/O and pHash work.
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

            if phash_distance(prev_hash, curr_hash) > 1:
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

    def _run_stage1_fallback(self, path: Path, fps: float):
        """FPS-1-style interval fallback: select one frame per round(fps) frames.

        Called when Stage 1 (degenerate filter + pHash) produces zero candidates.
        Mirrors fps1 exactly: no degenerate gating, no pHash — just regular
        interval sampling.  This guarantees every non-empty video produces at
        least one frame for Stage 2.
        """
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

    def _run_stage2(self, candidates, cand_indices, target_k):
        """Encode candidates with MobileCLIP and select K diverse frames via Greedy FPS."""
        if self._embedder is None:
            self._embedder = MobileCLIPEmbedder.get()
        embeddings = self._embedder.encode_micro_batched(candidates, batch_size=self.clip_batch_size)
        k_limit = target_k if self.mode == "fixed" else len(candidates)

        pool_indices, stopped_early = greedy_fps(embeddings, k_limit, self.min_distance, self.mode)
        pool_indices.sort()  # restore temporal order

        final_frames = [candidates[i] for i in pool_indices]
        final_indices = [cand_indices[i] for i in pool_indices]

        del candidates
        gc.collect()
        return final_frames, final_indices, stopped_early

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
