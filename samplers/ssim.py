"""SSIMSampler — Adaptive temporal baseline using SSIM scene-change detection.

Phase 1b baseline (content-aware, non-semantic) and TASS Stage 1 pre-filter.
Algorithm: O(N) streaming, O(1) per frame, O(K) storage.
"""

import cv2
import logging
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from skimage.metrics import structural_similarity

from .base_sampler import BaseSampler
from .fps1 import FPS1Sampler
from .ssim_result import SSIMSamplerResult
from config.settings import settings

logger = logging.getLogger(__name__)

MAX_ACCEPTED_FRAMES: int = 500
ACCEPTANCE_RATE_MIN: float = 0.01
ACCEPTANCE_RATE_MAX: float = 0.99
_DEFAULT_COMPARE_SIZE: Tuple[int, int] = (256, 144)
_MAX_READ_ERRORS = 10


def _get_ssim_compare_size() -> Tuple[int, int]:
    cfg = settings.ssim
    if cfg:
        size = cfg.get("compare_size", list(_DEFAULT_COMPARE_SIZE))
        return int(size[0]), int(size[1])
    return _DEFAULT_COMPARE_SIZE


class SSIMSampler(BaseSampler):
    """Adaptive temporal sampler using SSIM-based scene-change detection."""

    def __init__(self, threshold: float, name: str):
        if not 0.0 < threshold < 1.0:
            raise ValueError(f"SSIM threshold must be in (0, 1), got {threshold}.")
        self.threshold = threshold
        self._name = name

    def get_name(self) -> str:
        return self._name

    def sample(self, video_path: str) -> List[np.ndarray]:
        return self.sample_with_metadata(video_path).frames

    def sample_with_metadata(self, video_path: str) -> SSIMSamplerResult:
        path = Path(video_path)
        cap, total_frames_meta, fps = self._open_video(path)
        if cap is None:
            return SSIMSamplerResult._empty()

        cfg = self._read_config()
        accepted_frames, accepted_indices, ssim_at_accept, all_ssim, actual_total = (
            self._stream_ssim_frames(cap, cfg)
        )
        cap.release()

        if actual_total == 0:
            logger.warning(f"[{self._name}] Zero frames read from {path.name}")
            return SSIMSamplerResult._empty()

        accepted_frames, accepted_indices, ssim_at_accept, fallback_used = (
            self._apply_acceptance_guard(accepted_frames, accepted_indices, ssim_at_accept,
                                         actual_total, path, fps, cfg)
        )

        if not accepted_frames:
            accepted_frames, accepted_indices, ssim_at_accept = self._recover_first_frame(path)
            if not accepted_frames:
                return SSIMSamplerResult._empty()

        avg_ssim = float(np.mean(all_ssim)) if all_ssim else 1.0
        denom = max(total_frames_meta, 1) if total_frames_meta > 0 else max(actual_total, 1)
        reduction = (1.0 - len(accepted_frames) / denom) * 100.0
        logger.info(f"[{self._name}] {path.name}: {actual_total} → {len(accepted_frames)} "
                    f"({reduction:.1f}% reduction, avg_ssim={avg_ssim:.3f}, fallback={fallback_used})")

        return SSIMSamplerResult(
            frames=accepted_frames, frame_indices=accepted_indices,
            ssim_scores=ssim_at_accept, original_frame_count=actual_total,
            total_frames_meta=total_frames_meta if total_frames_meta > 0 else actual_total,
            selected_frame_count=len(accepted_frames), reduction_pct=reduction,
            average_ssim=avg_ssim, threshold_used=self.threshold, fps=fps,
            fallback_used=fallback_used,
        )

    def _open_video(self, path: Path):
        if not path.exists():
            logger.error(f"[{self._name}] Video not found: {path}")
            return None, 0, 30.0
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            logger.error(f"[{self._name}] Cannot open: {path}")
            return None, 0, 30.0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0 or fps != fps:
            logger.warning(f"[{self._name}] Invalid FPS — assuming 30")
            fps = 30.0
        return cap, total, fps

    def _read_config(self) -> dict:
        cfg = settings.ssim or {}
        return {
            "compare_size": _get_ssim_compare_size(),
            "win_size": cfg.get("win_size", 7),
            "max_accepted": cfg.get("max_accepted_frames", MAX_ACCEPTED_FRAMES),
            "rate_min": cfg.get("acceptance_rate_min", ACCEPTANCE_RATE_MIN),
            "rate_max": cfg.get("acceptance_rate_max", ACCEPTANCE_RATE_MAX),
        }

    def _stream_ssim_frames(self, cap, cfg) -> tuple:
        accepted_frames, accepted_indices, ssim_at_accept, all_ssim = [], [], [], []
        prev_gray = None
        frame_idx = read_errors = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame is None or frame.size == 0:
                read_errors += 1
                if read_errors >= _MAX_READ_ERRORS:
                    logger.error(f"[{self._name}] Too many read errors — stopping.")
                    break
                frame_idx += 1
                continue
            small, gray = self._resize_to_gray(frame, cfg["compare_size"], frame_idx)
            if gray is None:
                frame_idx += 1
                continue
            frame_idx = self._process_one_frame(
                frame, small, gray, prev_gray, frame_idx,
                accepted_frames, accepted_indices, ssim_at_accept, all_ssim, cfg,
            )
            prev_gray = gray if gray is not None else prev_gray

        return accepted_frames, accepted_indices, ssim_at_accept, all_ssim, frame_idx

    def _process_one_frame(self, frame, small, gray, prev_gray, frame_idx,
                            frames, indices, ssim_at_accept, all_ssim, cfg) -> int:
        if prev_gray is None:
            frames.append(frame.copy())
            indices.append(frame_idx)
            ssim_at_accept.append(1.0)
            return frame_idx + 1
        score = self._compute_ssim(prev_gray, gray, cfg["win_size"], frame_idx)
        if score is not None:
            all_ssim.append(score)
            if score < self.threshold:
                if len(frames) >= cfg["max_accepted"]:
                    logger.warning(f"[{self._name}] Hit MAX_ACCEPTED_FRAMES.")
                    return frame_idx + 1
                frames.append(frame.copy())
                indices.append(frame_idx)
                ssim_at_accept.append(score)
        return frame_idx + 1

    def _resize_to_gray(self, frame, compare_size, frame_idx):
        try:
            small = cv2.resize(frame, compare_size, interpolation=cv2.INTER_AREA)
            return small, cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        except cv2.error as exc:
            logger.warning(f"[{self._name}] Resize failed at {frame_idx}: {exc}")
            return None, None

    def _compute_ssim(self, prev_gray, gray, win_size, frame_idx) -> Optional[float]:
        try:
            return float(structural_similarity(
                prev_gray, gray, data_range=255, win_size=win_size, channel_axis=None,
            ))
        except Exception as exc:
            logger.warning(f"[{self._name}] SSIM failed at {frame_idx}: {exc}")
            return None

    def _apply_acceptance_guard(self, frames, indices, ssim_scores, actual_total, path, fps, cfg):
        rate = len(frames) / max(actual_total, 1)
        if rate < cfg["rate_min"]:
            logger.warning(f"[{self._name}] Rate {rate:.1%} < min — FPS-1 fallback.")
            fps1 = FPS1Sampler()
            new_frames = fps1.sample(str(path))
            new_indices = getattr(fps1, "_last_sampled_indices", list(range(len(new_frames))))
            return new_frames, new_indices, [None] * len(new_frames), True
        if rate > cfg["rate_max"]:
            logger.warning(f"[{self._name}] Rate {rate:.1%} > max — no effective filtering.")
        return frames, indices, ssim_scores, False
    def _recover_first_frame(self, path: Path):
        cap = cv2.VideoCapture(str(path))
        ret, first = cap.read()
        cap.release()
        if ret and first is not None:
            logger.error(f"[{self._name}] No frames selected — using first frame only.")
            return [first], [0], [1.0]
        return [], [], []
