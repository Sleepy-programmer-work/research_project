"""
tests/helpers/video_factory.py — Shared synthetic video factories for sampler tests.

Provides _make_video(), _solid_frame(), _noise_frame(), _static_nondegenerate_frame(),
_scene_frame(), _white_frame(), and _expected_k() so they are never copy-pasted across
test files.
"""
import math
import tempfile
from typing import List

import cv2
import numpy as np


def make_video(
    frames: List[np.ndarray],
    fps: float = 30.0,
    width: int = 320,
    height: int = 180,
) -> str:
    """Write BGR frames to a temporary .avi file and return its path.

    The caller MUST delete the file after use via os.unlink(path).
    VideoWriter on Linux requires the file to pre-exist on disk — the
    NamedTemporaryFile is closed immediately so the writer can open it.
    """
    tmp = tempfile.NamedTemporaryFile(suffix=".avi", delete=False)
    tmp.close()
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    out = cv2.VideoWriter(tmp.name, fourcc, fps, (width, height))
    for frame in frames:
        out.write(frame)
    out.release()
    return tmp.name


def solid_frame(color=(128, 64, 32), size=(320, 180)) -> np.ndarray:
    """Return a solid-colour BGR frame of the given size."""
    return np.full((*size[::-1], 3), color, dtype=np.uint8)


def white_frame(width: int = 320, height: int = 180) -> np.ndarray:
    """All-white frame — classified as degenerate (mean > 245)."""
    return np.full((height, width, 3), 255, dtype=np.uint8)


def noise_frame(seed: int = 0, width: int = 320, height: int = 180) -> np.ndarray:
    """Random noise frame — maximally different from any other.
    Always passes the degenerate filter: mean≈128, var≈5400.
    """
    rng = np.random.default_rng(seed=seed)
    return rng.integers(0, 256, (height, width, 3), dtype=np.uint8)


def static_nondegenerate_frame(width: int = 320, height: int = 180) -> np.ndarray:
    """Spatially-varied gradient frame that:
      - Passes the degenerate filter (mean≈128, var>>80)
      - Looks identical to every other call with the same size
      - Has SSIM ≈ 1.0 with consecutive copies after XVID encoding

    Horizontal gradient from 20→230 gives mean≈125 and var≈3500,
    comfortably above the var<80 degenerate threshold.
    """
    gradient = np.linspace(20, 230, width, dtype=np.uint8)
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    frame[:, :, 0] = gradient
    frame[:, :, 1] = gradient
    frame[:, :, 2] = gradient
    return frame


def scene_frame(scene_id: int, width: int = 320, height: int = 180) -> np.ndarray:
    """Distinct-scene frame: coloured circle on gradient background.
    Each scene_id produces a visually different frame for SSIM/semantic testing.
    """
    frame = static_nondegenerate_frame(width, height).copy()
    cx = (scene_id * 73 + 50) % (width - 40) + 20
    cy = (scene_id * 47 + 40) % (height - 40) + 20
    colour = (
        (scene_id * 80) % 256,
        (scene_id * 53 + 100) % 256,
        (scene_id * 37 + 200) % 256,
    )
    cv2.circle(frame, (cx, cy), 40, colour, -1)
    return frame


def expected_k(num_frames: int, fps: float = 30.0) -> int:
    """Replicate TASSSampler's internal target_k formula for assertions."""
    duration = max(1.0, num_frames / fps)
    return max(1, math.ceil(duration))
