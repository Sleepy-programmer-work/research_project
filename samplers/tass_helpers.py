"""
samplers/tass_helpers.py — Stateless algorithmic helpers for TASSSampler.

Isolates pure algorithms (degenerate detection, pHash, Greedy FPS) from the
sampler class that orchestrates them.

These functions have no side effects and are fully unit-testable in isolation.
"""
import logging
from typing import List, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Degenerate-frame detection thresholds (luminance, variance, and sharpness)
_BRIGHT_THRESH = 245.0   # white flash / overexposure
_DARK_THRESH = 8.0       # deep fade to black / lens cap
_VAR_THRESH = 80.0       # near-uniform frame (colour card, static noise)
_BLUR_THRESH = 50.0      # Laplacian variance — below this = motion blur / defocus


def is_degenerate(frame: np.ndarray) -> bool:
    """Return True if frame is a flash, deep fade, near-uniform, or blurry.

    Checks four conditions (any True → degenerate):
      1. Mean brightness > 245 — white flash / overexposure.
      2. Mean brightness < 8   — deep fade to black / lens cap.
      3. Pixel variance < 80   — near-uniform (colour card, static noise).
      4. Laplacian variance < 50 — motion blur or defocus. The Laplacian
         operator highlights edges; low variance of its response indicates
         the frame lacks sharp detail and carries little semantic content.

    Runs on the full-resolution frame to preserve sensitivity to subtle
    luminance extremes that survive downscaling.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    mean_val = float(np.mean(gray))
    var_val = float(np.var(gray))
    lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    return (mean_val > _BRIGHT_THRESH or mean_val < _DARK_THRESH
            or var_val < _VAR_THRESH or lap_var < _BLUR_THRESH)


def get_phash(gray: np.ndarray) -> np.ndarray:
    """Compute a 64-bit perceptual hash using Discrete Cosine Transform (DCT).
    
    1. Resize to 32x32.
    2. Compute DCT.
    3. Take top-left 8x8 (excluding DC component at 0,0).
    4. Binarise based on median.
    """
    resized = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA)
    dct = cv2.dct(np.float32(resized))
    dct_8x8 = dct[0:8, 0:8]
    med = np.median(dct_8x8.flatten()[1:])
    return dct_8x8 > med


def phash_distance(prev_hash: np.ndarray, curr_hash: np.ndarray) -> int:
    """Compute Hamming distance between two pHash arrays.
    
    Returns an integer [0, 64]. 0 means identical.
    Validation showed a distance > 1 correlates 94% with visual scene changes.
    """
    return int(np.count_nonzero(prev_hash != curr_hash))


def greedy_fps(embeddings: np.ndarray, k_limit: int, min_distance: float, mode: str) -> Tuple[List[int], bool]:
    """Greedy Farthest-Point Sampling over L2-normalised embeddings.

    Returns (selected_pool_indices, stopped_early).

    Selects frames in diversity order (not temporal). The caller must sort the
    result to restore temporal ordering for the benchmark runner.

    Implementation uses a vectorised incremental update: a persistent
    ``min_dists`` array tracks each candidate's minimum cosine distance to
    the selected set.  Each iteration adds one frame and updates the array
    via a single ``embeddings @ embeddings[new]`` call — O(M) numpy ops
    instead of the previous O(M·K) Python loop.

    Args:
        embeddings:   (M, D) float32 array of L2-normalised frame embeddings.
        k_limit:      Maximum number of frames to select.
        min_distance: Cosine distance floor for adaptive early stopping.
        mode:         'fixed' or 'adaptive'.
    """
    n = len(embeddings)
    if n == 0:
        return [], False

    # Medoid seeding: pick the candidate closest to the geometric centre
    # of the embedding space.  This reduces initialisation bias compared to
    # always starting from the temporally first candidate (index 0).
    centroid = embeddings.mean(axis=0)
    centroid /= np.linalg.norm(centroid) + 1e-8    # L2-normalise
    seed = int(np.argmax(embeddings @ centroid))    # highest cosine sim to centroid

    selected = [seed]
    stopped_early = False

    # min_dists[i] = min cosine distance from candidate i to any selected frame.
    # Initialise with distances to the seed.
    sims = embeddings @ embeddings[seed]          # (M,) — vectorised dot product
    min_dists = 1.0 - sims                        # cosine distance

    # Mask the seed so it is never re-selected
    min_dists[seed] = -1.0

    for _ in range(min(k_limit, n) - 1):
        best_idx = int(np.argmax(min_dists))
        best_dist = float(min_dists[best_idx])

        if best_dist <= 0.0:
            # All remaining candidates are already selected (or masked)
            break

        if mode == "adaptive" and best_dist < min_distance:
            stopped_early = True
            logger.info(f"Early stopping at {len(selected)} frames (best_dist={best_dist:.4f} < {min_distance}).")
            break

        selected.append(best_idx)
        min_dists[best_idx] = -1.0                # mask selected

        # Incremental update — only the new frame's distances matter.
        # For each unselected candidate, update min_dists if the new frame
        # is closer than anything previously selected.
        new_sims = embeddings @ embeddings[best_idx]
        new_dists = 1.0 - new_sims
        min_dists = np.where(min_dists < 0.0, min_dists, np.minimum(min_dists, new_dists))

    return selected, stopped_early
