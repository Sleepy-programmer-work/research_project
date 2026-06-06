"""
SSIMSampler: Adaptive temporal baseline sampler.

Uses Structural Similarity Index Measure (SSIM) to detect scene transitions
and select only frames where visual content changes significantly.

Role in research:
  - Phase 1b baseline: demonstrates content-aware (but semantic-free) sampling
  - TASS Stage 1: output (accepted_frames, ssim_scores, frame_indices) feeds
    directly into TASS's MobileCLIP semantic refinement stage

Algorithm complexity:
  - O(N) frame reads (streaming — no full video loaded into RAM)
  - O(1) SSIM computation per frame (fixed compare resolution)
  - O(K) storage where K = accepted frames (K << N for static videos)

Advantages over FPS:
  - Avoids redundant frames from static scenes
  - Automatically increases density at scene transitions
  - Very low CPU cost; zero GPU usage

Limitations:
  - Blind to semantic content — two dissimilar-looking frames may say
    the same thing ("a man walks" forward vs. backward)
  - Threshold is content-agnostic — requires tuning per video type
  - NOT the final proposed method (TASS); see README Section 17
"""

import cv2
import logging
import numpy as np
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

from skimage.metrics import structural_similarity

from .base_sampler import BaseSampler
from config.settings import settings

logger = logging.getLogger(__name__)

# Compare resolution for SSIM — fast CPU computation, retains structure.
# Read from config if available, otherwise fall back to this safe default.
def _get_ssim_compare_size() -> Tuple[int, int]:
    """Return (width, height) compare size from config or default (256, 144)."""
    ssim_cfg = settings.ssim
    if ssim_cfg:
        size = ssim_cfg.get("compare_size", [256, 144])
        return (int(size[0]), int(size[1]))
    return (256, 144)


SSIM_COMPARE_SIZE: Tuple[int, int] = (256, 144)  # (width, height) — set at import time

# Safety guards — prevent degenerate outputs
MIN_ACCEPTED_FRAMES: int = 1
MAX_ACCEPTED_FRAMES: int = 500   # hard cap for pathological videos

# Fallback to FPS-1 if SSIM acceptance rate is outside this range
ACCEPTANCE_RATE_MIN: float = 0.01   # <1%  → too aggressive, threshold too low
ACCEPTANCE_RATE_MAX: float = 0.99   # >99% → no filtering, threshold too high


@dataclass
class SSIMSamplerResult:
    """
    Full output from SSIMSampler.sample_with_metadata().
    Used by the benchmark framework for logging and by TASS Stage 1.

    Fields designed for TASS compatibility:
      - frame_indices: passed to TASS for temporal position tracking
      - ssim_scores:   passed to TASS as a pre-filter quality signal
      - frames:        passed directly to TASS Stage 2 (MobileCLIP embeddings)

    Invariants (must not be broken — TASS depends on them):
      - frames is always full-resolution BGR NumPy arrays
      - frame_indices always preserves original temporal position in the video
      - len(frames) == len(frame_indices) == len(ssim_scores)
    """
    frames: List[np.ndarray]         # BGR frames at original capture resolution
    frame_indices: List[int]         # original frame indices in the source video
    ssim_scores: List[float]         # SSIM score that triggered each frame's acceptance
    original_frame_count: int
    selected_frame_count: int
    reduction_pct: float
    average_ssim: float
    threshold_used: float
    fps: float
    fallback_used: bool = False      # True if SSIM fell back to FPS-1


class SSIMSampler(BaseSampler):
    """
    Adaptive temporal sampler using SSIM-based scene-change detection.

    Implements the BaseSampler interface — fully compatible with the benchmark
    runner, frame_extraction pipeline, and existing caching infrastructure.

    Design principle (streaming, memory-safe):
      Frames are read one at a time and discarded unless accepted.  For a
      10-minute video at 30 fps (18,000 frames) this keeps peak RAM at
      O(K) where K is the number of accepted frames, not O(N) total frames.

    Reference tracking:
      prev_gray_small is updated ONLY when a frame is accepted, not on every
      iteration.  This compares the current frame against the last accepted
      frame — making SSIM a scene-change detector rather than a motion-blur
      filter.

    TASS Stage 1 interface:
      Call sample_with_metadata() to obtain SSIMSamplerResult.  This dataclass
      is the Stage 1 → Stage 2 handoff contract.  TASS Stage 2 (MobileCLIP)
      receives it unchanged and uses frames, frame_indices, and ssim_scores
      to compute semantic embeddings and perform diversity-based selection.

    Future TASS usage (do NOT implement now — document only):
        ssim_result = ssim_sampler.sample_with_metadata(video_path)
        embeddings = mobileclip.encode(ssim_result.frames)
        diverse_indices = kmeans_diverse_select(embeddings, k=target_count)
        final_frames = [ssim_result.frames[i] for i in diverse_indices]
        final_positions = [ssim_result.frame_indices[i] for i in diverse_indices]
    """

    def __init__(self, threshold: float, name: str):
        """
        Args:
            threshold: SSIM similarity threshold in (0, 1).
                       Frames with SSIM < threshold are accepted (scene changed).
                       Typical values:
                         0.85 — aggressive (more frames, more scene changes caught)
                         0.90 — balanced (default recommendation)
                         0.95 — conservative (only major scene changes)
            name:      Unique variant identifier (e.g. 'ssim_090').
                       Used as cache key, CSV row identifier, and JSON filename.
                       Must be stable — changing this invalidates all cached
                       frame captions for this sampler variant.
        """
        if not 0.0 < threshold < 1.0:
            raise ValueError(
                f"SSIM threshold must be in open interval (0, 1), got {threshold}. "
                f"Typical values: 0.85, 0.90, 0.95."
            )
        self.threshold = threshold
        self._name = name

    def get_name(self) -> str:
        """
        Return the unique variant name (e.g. 'ssim_090').

        This is the primary identifier used throughout the pipeline:
          - Cache key in frame_captions/
          - CSV column 'sampling_method'
          - JSON filename in frame_selection/
        """
        return self._name

    def sample(self, video_path: str) -> List[np.ndarray]:
        """
        BaseSampler interface — returns selected BGR frames.

        This is the contract method used by the benchmark runner's inner loop
        (extract_frames → caption_frames).  Internally delegates to
        sample_with_metadata() and discards the metadata.

        For SSIM-specific metadata (reduction_pct, ssim_scores, etc.) call
        sample_with_metadata() directly — the benchmark runner does this for
        SSIMSampler instances via isinstance() dispatch.
        """
        result = self.sample_with_metadata(video_path)
        return result.frames

    def sample_with_metadata(self, video_path: str) -> "SSIMSamplerResult":
        """
        Full streaming SSIM sampling with complete metadata.

        Always call this from the benchmark runner for SSIMSampler instances
        so that frame_selection JSON files receive SSIM-specific fields
        (reduction_pct, average_ssim, threshold_used, fallback_used, fps).

        Returns:
            SSIMSamplerResult with all fields populated.  On any fatal error
            returns _empty_result() so the caller never receives None.
        """
        path = Path(video_path)
        if not path.exists():
            logger.error(f"[{self._name}] Video not found: {video_path}")
            return self._empty_result()

        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            logger.error(f"[{self._name}] OpenCV cannot open: {video_path}")
            return self._empty_result()

        # --- Read video metadata ---
        total_frames_meta = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0 or fps != fps:   # catches zero or NaN
            logger.warning(
                f"[{self._name}] Invalid FPS ({fps}) in {path.name} — assuming 30"
            )
            fps = 30.0
        if total_frames_meta <= 0:
            logger.warning(
                f"[{self._name}] Unknown frame count in {path.name} — streaming without limit"
            )

        # Read compare size from config (falls back to module-level constant)
        compare_size = _get_ssim_compare_size()
        win_size = settings.ssim.get("win_size", 7) if settings.ssim else 7
        max_accepted = (
            settings.ssim.get("max_accepted_frames", MAX_ACCEPTED_FRAMES)
            if settings.ssim else MAX_ACCEPTED_FRAMES
        )
        acceptance_rate_min = (
            settings.ssim.get("acceptance_rate_min", ACCEPTANCE_RATE_MIN)
            if settings.ssim else ACCEPTANCE_RATE_MIN
        )
        acceptance_rate_max = (
            settings.ssim.get("acceptance_rate_max", ACCEPTANCE_RATE_MAX)
            if settings.ssim else ACCEPTANCE_RATE_MAX
        )

        logger.debug(
            f"[{self._name}] {path.name}: total_frames_meta={total_frames_meta}, "
            f"fps={fps:.1f}, compare_size={compare_size}, win_size={win_size}"
        )

        # --- Streaming SSIM loop ---
        accepted_frames: List[np.ndarray] = []
        accepted_indices: List[int] = []
        ssim_scores_at_accept: List[float] = []
        all_ssim_scores: List[float] = []

        prev_gray_small: Optional[np.ndarray] = None
        frame_idx: int = 0
        read_errors: int = 0
        MAX_READ_ERRORS: int = 10

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Guard against corrupt / blank frames
            if frame is None or frame.size == 0:
                read_errors += 1
                logger.warning(
                    f"[{self._name}] Blank frame at index {frame_idx} in {path.name} "
                    f"({read_errors}/{MAX_READ_ERRORS})"
                )
                if read_errors >= MAX_READ_ERRORS:
                    logger.error(
                        f"[{self._name}] Too many read errors in {path.name} — stopping early"
                    )
                    break
                frame_idx += 1
                continue

            # Resize to compare resolution (CPU-only, ~1ms per frame)
            # The original full-resolution frame is kept for downstream VLM inference.
            try:
                small = cv2.resize(frame, compare_size, interpolation=cv2.INTER_AREA)
                gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
            except cv2.error as e:
                logger.warning(
                    f"[{self._name}] Frame resize/convert failed at {frame_idx}: {e}"
                )
                frame_idx += 1
                continue

            # Always accept the first valid frame — no reference to compare against
            if prev_gray_small is None:
                accepted_frames.append(frame.copy())
                accepted_indices.append(frame_idx)
                ssim_scores_at_accept.append(1.0)  # self-similarity is 1.0
                prev_gray_small = gray
                frame_idx += 1
                continue

            # Compute SSIM against last ACCEPTED frame (not last seen frame).
            # This is intentional: we measure "how different is this from the
            # last frame we kept" rather than "is this frame moving".
            try:
                score = float(structural_similarity(
                    prev_gray_small,
                    gray,
                    data_range=255,
                    win_size=win_size,
                    channel_axis=None,   # grayscale input — no channel axis
                ))
            except Exception as e:
                logger.warning(
                    f"[{self._name}] SSIM computation failed at frame {frame_idx}: {e}"
                )
                frame_idx += 1
                continue

            all_ssim_scores.append(score)

            # Accept frame if scene changed (SSIM below threshold)
            if score < self.threshold:
                if len(accepted_frames) >= max_accepted:
                    logger.warning(
                        f"[{self._name}] Hit MAX_ACCEPTED_FRAMES ({max_accepted}) "
                        f"at frame {frame_idx} in {path.name} — stopping acceptance. "
                        f"Consider raising the threshold."
                    )
                    break
                accepted_frames.append(frame.copy())
                accepted_indices.append(frame_idx)
                ssim_scores_at_accept.append(score)
                # KEY: reference updates only on acceptance — this is what makes
                # SSIM a scene-change detector rather than a motion-blur filter.
                prev_gray_small = gray

            frame_idx += 1

        cap.release()
        actual_total = frame_idx  # may differ from metadata total_frames_meta

        # --- Edge case: zero frames read ---
        if actual_total == 0:
            logger.warning(f"[{self._name}] Zero frames read from {path.name}")
            return self._empty_result()

        # --- Edge case: single-frame video ---
        if actual_total == 1 and len(accepted_frames) == 1:
            logger.info(f"[{self._name}] Single-frame video: {path.name}")

        # --- Acceptance rate sanity check → FPS-1 fallback ---
        acceptance_rate = len(accepted_frames) / max(actual_total, 1)
        fallback_used = False

        if acceptance_rate < acceptance_rate_min:
            logger.warning(
                f"[{self._name}] Acceptance rate {acceptance_rate:.1%} < "
                f"{acceptance_rate_min:.1%} for {path.name}. "
                f"Threshold {self.threshold} is too low — falling back to FPS-1 output."
            )
            accepted_frames, accepted_indices = self._fps1_fallback(str(path), fps)
            ssim_scores_at_accept = [0.0] * len(accepted_frames)
            fallback_used = True

        elif acceptance_rate > acceptance_rate_max:
            logger.warning(
                f"[{self._name}] Acceptance rate {acceptance_rate:.1%} > "
                f"{acceptance_rate_max:.1%} for {path.name}. "
                f"Threshold {self.threshold} is too high — no effective filtering. "
                f"Data retained as-is for ablation study."
            )
            # Do NOT fall back — high acceptance rate is valid data for the paper
            # (it demonstrates that this threshold is ineffective for this video type)

        # --- Edge case: still empty after fallback ---
        if not accepted_frames:
            logger.error(
                f"[{self._name}] No frames selected for {path.name} after all guards "
                f"— using first frame only"
            )
            cap2 = cv2.VideoCapture(str(path))
            ret, first = cap2.read()
            cap2.release()
            if ret and first is not None:
                accepted_frames = [first]
                accepted_indices = [0]
                ssim_scores_at_accept = [1.0]
            else:
                return self._empty_result()

        avg_ssim = float(np.mean(all_ssim_scores)) if all_ssim_scores else 1.0
        reduction = (1.0 - len(accepted_frames) / max(actual_total, 1)) * 100.0

        logger.info(
            f"[{self._name}] {path.name}: "
            f"{actual_total} frames → {len(accepted_frames)} selected "
            f"({reduction:.1f}% reduction, avg_ssim={avg_ssim:.3f}, "
            f"threshold={self.threshold}, fallback={fallback_used})"
        )

        return SSIMSamplerResult(
            frames=accepted_frames,
            frame_indices=accepted_indices,
            ssim_scores=ssim_scores_at_accept,
            original_frame_count=actual_total,
            selected_frame_count=len(accepted_frames),
            reduction_pct=reduction,
            average_ssim=avg_ssim,
            threshold_used=self.threshold,
            fps=fps,
            fallback_used=fallback_used,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _fps1_fallback(
        self, video_path: str, fps: float
    ) -> Tuple[List[np.ndarray], List[int]]:
        """
        Return FPS-1 frames as a fallback when SSIM acceptance rate is pathological.

        This mirrors FPS1Sampler.sample() exactly so the fallback output is
        directly comparable to the fps1 baseline in the benchmark matrix.
        """
        cap = cv2.VideoCapture(video_path)
        native_fps = cap.get(cv2.CAP_PROP_FPS)
        if native_fps <= 0 or native_fps != native_fps:
            native_fps = fps if fps > 0 else 30.0
        interval = max(1, round(native_fps))

        frames: List[np.ndarray] = []
        indices: List[int] = []
        idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if idx % interval == 0 and frame is not None and frame.size > 0:
                frames.append(frame.copy())
                indices.append(idx)
            idx += 1

        cap.release()
        return frames, indices

    @staticmethod
    def _empty_result() -> "SSIMSamplerResult":
        """Return a zero-count result for unrecoverable error cases."""
        return SSIMSamplerResult(
            frames=[],
            frame_indices=[],
            ssim_scores=[],
            original_frame_count=0,
            selected_frame_count=0,
            reduction_pct=0.0,
            average_ssim=0.0,
            threshold_used=0.0,
            fps=0.0,
            fallback_used=False,
        )
