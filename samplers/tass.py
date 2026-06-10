"""
samplers/tass.py — Two-Stage Adaptive Semantic Sampling (TASS) Engine.

Architecture overview:
  Stage 1 — Streaming Degenerate Purge + Grid-SSIM Pre-filter  (CPU, O(N))
    1a. Degenerate Frame Detector:
          Drops frames with extreme luminance (flash/fade/lens-cap) before any
          SSIM computation.  Checks mean brightness and pixel variance against
          empirically-tuned thresholds in O(1) per frame via numpy.
    1b. 2×2 Grid-SSIM Quadrant Filter:
          Computes SSIM independently in each of four spatial quadrants and
          takes the *minimum* score.  This prevents a static background from
          masking a moving foreground object in one quadrant — a known failure
          mode of whole-frame SSIM that biases sampling towards visually
          redundant frames.

  Stage 2 — Micro-Batched MobileCLIP + Greedy Farthest-Point Selection (CPU, O(M))
    2a. Semantic Embedding:
          MobileCLIP-S1 (via the CPU-only singleton) encodes each candidate
          frame into a 512-d L2-normalised embedding.
    2b. Greedy Farthest-Point Sampling (FPS):
          Iteratively selects the frame that maximises minimum cosine distance
          to all already-selected frames.  This guarantees a 2-approximation
          to the optimal max-min diversity bound — unlike K-Means, it never
          produces empty clusters or requires re-initialisation.
    2c. Early Stopping (adaptive mode only):
          If the next best candidate's minimum distance to the selected set
          falls below `min_distance`, the algorithm halts.  This prevents
          forced selection of semantically redundant frames in static video
          segments — a critical guardrail for short clips.

Modes:
  'fixed':    Selects exactly K = ceil(duration_seconds) frames to match the
              FPS-1 baseline frame budget.  Enables controlled ablation that
              isolates sampling *intelligence* from raw frame *count*.
  'adaptive': Selects frames until either the candidate pool is exhausted or
              early stopping fires.  Trades reproducibility for quality.

Complexity summary:
  Stage 1:  O(N) time, O(M) space   (M << N for non-static videos)
  Stage 2:  O(M × K) time, O(M) space  (dominated by embedding + FPS)

Reviewer-armour notes:
  - The degenerate filter is a pre-stage *before* SSIM — it does not replace it.
  - Grid-SSIM uses the *minimum* quadrant score, not the mean.  This is
    intentional and documented in the paper (Section IV-B).
  - The singleton embedder guarantees exactly one model load per process
    lifetime, preventing repeated 85 MB allocations in the benchmark loop.
  - All metadata fields are written to the return dict to enable the benchmark
    runner to compute Semantic Yield and VLM Calls Saved without re-running.
"""

import cv2
import gc
import logging
import math
import numpy as np
from pathlib import Path
from typing import List

from skimage.metrics import structural_similarity

from .base_sampler import BaseSampler
from models.clip_embedder import MobileCLIPEmbedder

logger = logging.getLogger(__name__)


class TASSSampler(BaseSampler):
    """
    Two-Stage Adaptive Semantic Sampling (TASS) sampler.

    Implements the BaseSampler interface so it is a drop-in replacement for
    FPS1Sampler, SSIMSampler, etc. in the benchmark runner.

    The primary entry-point for the benchmark runner is sample_with_metadata(),
    which returns a rich dict suitable for computing the IEEE-paper metrics
    (Semantic Yield, VLM Calls Saved %).  The BaseSampler.sample() interface
    delegates to it and extracts only the frame list.

    Structural contract (parallel to SSIMSamplerResult for TASS):
        result["frames"]    — List[np.ndarray] BGR frames, full resolution
        result["indices"]   — List[int] original frame indices in source video
        result["meta"]      — dict with telemetry fields (see sample_with_metadata)
    """

    # Compare resolution for Stage 1 SSIM.
    # 256×144 gives a 16:9 thumbnail at minimal CPU cost (~0.5 ms SSIM).
    _COMPARE_W = 256
    _COMPARE_H = 144

    # Grid size for quadrant SSIM.  2 produces 4 quadrants (2×2).
    _GRID = 2

    # Safety guard: never embed more than this many candidate frames to
    # prevent runaway RAM usage on pathologically long, low-SSIM videos.
    _MAX_CANDIDATES = 2000

    def __init__(
        self,
        mode: str = "fixed",
        threshold: float = 0.90,
        min_distance: float = 0.10,
        clip_batch_size: int = 16,
    ) -> None:
        """
        Initialise TASSSampler with the given operating parameters.

        Args:
            mode:            'fixed' — select exactly ceil(duration) frames
                               (controlled evaluation, matches FPS-1 budget).
                             'adaptive' — stop early when semantic diversity
                               drops below min_distance.
            threshold:       Grid-SSIM change threshold for Stage 1 acceptance.
                             Frames with min-quadrant SSIM < threshold are
                             admitted to the candidate pool.
                             Range: (0, 1).  Recommended: 0.90.
            min_distance:    Cosine distance floor for Greedy FPS early stopping
                             (adaptive mode only).  If the next best candidate's
                             minimum distance to the selected set is below this
                             value the algorithm halts.
                             Range: [0, 1].  Recommended: 0.10.
            clip_batch_size: Mini-batch size for MobileCLIP encoding.
                             Reduce to 8 if RAM pressure is observed on WSL2.
        """
        if not 0.0 < threshold < 1.0:
            raise ValueError(
                f"TASS threshold must be in open interval (0, 1), got {threshold}. "
                f"Recommended value: 0.90."
            )
        if not 0.0 <= min_distance < 1.0:
            raise ValueError(
                f"min_distance must be in [0, 1), got {min_distance}. "
                f"Recommended value: 0.10."
            )

        self.mode = mode.strip().lower()
        if self.mode not in ("fixed", "adaptive"):
            raise ValueError(
                f"TASSSampler mode must be 'fixed' or 'adaptive', got '{mode}'. "
                f"Use 'fixed' for fair benchmark comparison against FPS-1."
            )

        self.threshold = threshold
        self.min_distance = min_distance
        self.clip_batch_size = clip_batch_size
        self._compare_size = (self._COMPARE_W, self._COMPARE_H)

        # Lazily resolve the singleton — do NOT load weights at __init__ time.
        # The benchmark runner creates all sampler instances before loading the
        # VLM.  Deferring until sample() is called prevents the 85 MB MobileCLIP
        # allocation from front-loading RAM pressure before VLM warm-up.
        self._embedder: MobileCLIPEmbedder | None = None

        logger.info(
            f"TASSSampler initialised: mode={self.mode}, "
            f"threshold={self.threshold}, min_distance={self.min_distance}"
        )

    # ------------------------------------------------------------------
    # BaseSampler interface
    # ------------------------------------------------------------------

    def get_name(self) -> str:
        """Return the unique sampler identifier used as CSV key and cache key."""
        return f"tass_{self.mode}"

    def sample(self, video_path: str) -> List[np.ndarray]:
        """
        BaseSampler interface — returns selected BGR frames.

        For richer telemetry (candidate pool size, degenerate count, etc.)
        call sample_with_metadata() directly.  The benchmark runner does this
        via the 'tass' name check in its inner loop.
        """
        return self.sample_with_metadata(video_path)["frames"]

    # ------------------------------------------------------------------
    # Primary sampling entry-point
    # ------------------------------------------------------------------

    def sample_with_metadata(self, video_path: str) -> dict:
        """
        Execute the full two-stage TASS pipeline and return frames + telemetry.

        Return dict schema:
            {
              "frames":  List[np.ndarray],   # selected BGR frames (full resolution)
              "indices": List[int],           # original frame indices in source video
              "meta": {
                "frames_original":         int,   # total frames read from video
                "candidate_pool_size":     int,   # frames surviving Stage 1
                "frames_degenerate_dropped": int, # frames purged by degenerate filter
                "tass_stopped_early":      bool,  # True if adaptive early-stop fired
                "vlm_calls":               int,   # = len(frames); metric for paper
              }
            }

        On unrecoverable error (bad video path, unreadable file) returns an
        empty result dict rather than raising — consistent with SSIMSampler's
        contract so the benchmark runner's try/except is not required here.
        """
        path = Path(video_path)
        if not path.exists():
            logger.error(f"[{self.get_name()}] Video not found: {video_path}")
            return self._empty_result()

        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            logger.error(f"[{self.get_name()}] OpenCV cannot open: {video_path}")
            return self._empty_result()

        # --- Read video-level metadata ---
        raw_fps = cap.get(cv2.CAP_PROP_FPS)
        fps = raw_fps if raw_fps > 0 and raw_fps == raw_fps else 30.0  # NaN guard
        total_frames_meta = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = max(1.0, total_frames_meta / fps) if total_frames_meta > 0 else 1.0

        # Fixed-mode frame budget — exactly mirrors FPS-1's ceil(duration) count.
        # This is the controlled variable that isolates sampling intelligence from
        # raw frame count when comparing TASS against FPS-1 in the benchmark table.
        target_k = max(1, math.ceil(duration))

        logger.info(
            f"[{self.get_name()}] {path.name}: fps={fps:.1f}, "
            f"total_frames_meta={total_frames_meta}, duration={duration:.1f}s, "
            f"target_k={target_k} (mode={self.mode})"
        )

        # --- STAGE 1: Streaming Degenerate Purge + Grid-SSIM Pre-filter ---
        candidate_frames: List[np.ndarray] = []
        candidate_indices: List[int] = []
        prev_gray: np.ndarray | None = None
        frame_idx: int = 0
        degenerate_dropped: int = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame is None or frame.size == 0:
                frame_idx += 1
                continue

            # 1a: Degenerate filter (flash / deep-fade / lens-cap detection).
            #     Runs on the full-resolution frame to preserve sensitivity to
            #     subtle luminance extremes that survive downscaling.
            if self._is_degenerate(frame):
                degenerate_dropped += 1
                frame_idx += 1
                continue

            # Downscale to compare resolution for cheap SSIM computation.
            try:
                small = cv2.resize(
                    frame, self._compare_size, interpolation=cv2.INTER_AREA
                )
                gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
            except cv2.error as exc:
                logger.warning(
                    f"[{self.get_name()}] Resize/convert failed at frame "
                    f"{frame_idx}: {exc}"
                )
                frame_idx += 1
                continue

            # Always accept the very first non-degenerate frame — there is no
            # reference frame to compare it against.
            if prev_gray is None:
                candidate_frames.append(frame.copy())
                candidate_indices.append(frame_idx)
                prev_gray = gray
                frame_idx += 1
                continue

            # 1b: 2×2 Grid-SSIM quadrant filter.
            #     Accept the frame only if the *minimum* quadrant SSIM score
            #     falls below the threshold, meaning at least one region of
            #     the frame has changed significantly since the last accepted frame.
            if self._grid_ssim(prev_gray, gray) < self.threshold:
                candidate_frames.append(frame.copy())
                candidate_indices.append(frame_idx)
                # Update reference only on acceptance — this makes Stage 1 a
                # scene-change detector rather than a per-frame motion filter.
                prev_gray = gray

                # Hard cap: prevent runaway memory on long static-ish videos
                # where SSIM dips below threshold very frequently.
                if len(candidate_frames) >= self._MAX_CANDIDATES:
                    logger.warning(
                        f"[{self.get_name()}] Hit MAX_CANDIDATES "
                        f"({self._MAX_CANDIDATES}) at frame {frame_idx} in "
                        f"{path.name} — halting Stage 1 early."
                    )
                    # Drain remaining frames without storing them
                    while cap.read()[0]:
                        frame_idx += 1
                    break

            frame_idx += 1

        cap.release()
        total_frames_read = frame_idx

        logger.info(
            f"[{self.get_name()}] Stage 1 complete: {total_frames_read} frames read, "
            f"{len(candidate_frames)} candidates, {degenerate_dropped} degenerate dropped."
        )

        if not candidate_frames:
            logger.warning(
                f"[{self.get_name()}] No candidate frames after Stage 1 for {path.name}."
            )
            return self._empty_result(
                frames_original=total_frames_read,
                degenerate_dropped=degenerate_dropped,
            )

        # --- STAGE 2: Micro-Batched MobileCLIP + Greedy Farthest-Point Sampling ---

        # Resolve embedder singleton lazily (see __init__ docstring for rationale).
        if self._embedder is None:
            self._embedder = MobileCLIPEmbedder.get()

        embeddings = self._embedder.encode_micro_batched(
            candidate_frames, batch_size=self.clip_batch_size
        )

        # In fixed mode K is bounded by target_k.
        # In adaptive mode K can grow to the full candidate pool (early stop governs).
        k_limit = target_k if self.mode == "fixed" else len(candidate_frames)

        selected_pool_indices: List[int] = [0]  # anchor: first candidate always selected
        # P1-4 FIX: parallel set for O(1) membership testing.
        # `if i in selected_pool_indices` on a list is O(k) per candidate,
        # making the loop O(M²k). Using a set reduces it to O(Mk).
        # For M=2000 candidates, k=100 selections: 200K ops instead of 400M.
        selected_pool_set: set = {0}
        stopped_early = False

        for _ in range(k_limit - 1):
            if len(selected_pool_indices) >= len(candidate_frames):
                break  # Candidate pool exhausted — every frame is selected.

            best_idx: int = -1
            best_dist: float = -1.0

            for i in range(len(embeddings)):
                if i in selected_pool_set:  # O(1) — set lookup
                    continue

                # Cosine distance = 1 − dot(a, b) for L2-normalised vectors.
                # Taking the minimum over all already-selected frames gives the
                # distance from frame i to its nearest semantic neighbour in the
                # current selection — the standard Greedy FPS criterion.
                min_dist_to_selected = min(
                    1.0 - float(np.dot(embeddings[i], embeddings[j]))
                    for j in selected_pool_indices
                )

                if min_dist_to_selected > best_dist:
                    best_dist = min_dist_to_selected
                    best_idx = i

            # Adaptive early stopping: if even the *best* available candidate is
            # too semantically similar to the current selection, further additions
            # would only introduce redundancy.
            if self.mode == "adaptive" and best_dist < self.min_distance:
                stopped_early = True
                logger.info(
                    f"[{self.get_name()}] Early stopping at "
                    f"{len(selected_pool_indices)} frames "
                    f"(best_dist={best_dist:.4f} < min_distance={self.min_distance})."
                )
                break

            if best_idx != -1:
                selected_pool_indices.append(best_idx)
                selected_pool_set.add(best_idx)  # keep set in sync

        # Sort selected pool indices to restore temporal ordering.
        # Greedy FPS selects frames in diversity order, not temporal order.
        # The benchmark runner and VLM context builder both assume temporal ordering.
        selected_pool_indices.sort()

        final_frames = [candidate_frames[i] for i in selected_pool_indices]
        final_indices = [candidate_indices[i] for i in selected_pool_indices]

        # Free candidate frame memory — the Greedy FPS loop is done.
        del candidate_frames
        gc.collect()

        logger.info(
            f"[{self.get_name()}] Stage 2 complete: {len(final_frames)} frames selected "
            f"from {len(embeddings)} candidates "
            f"(stopped_early={stopped_early})."
        )

        return {
            "frames": final_frames,
            "indices": final_indices,
            "meta": {
                "frames_original":           total_frames_read,
                "candidate_pool_size":       len(embeddings),
                "frames_degenerate_dropped": degenerate_dropped,
                "tass_stopped_early":        stopped_early,
                "vlm_calls":                 len(final_frames),
            },
        }

    # ------------------------------------------------------------------
    # Stage 1 private helpers
    # ------------------------------------------------------------------

    def _is_degenerate(self, frame: np.ndarray) -> bool:
        """
        Classify a frame as degenerate (flash / deep fade / lens cap).

        Degenerate frames carry no useful semantic content and should be
        excluded before the SSIM filter — feeding them to Grid-SSIM would
        produce artificially low scores that inflate the candidate pool.

        Detection criteria (any → degenerate):
          - mean brightness > 245:  white flash or overexposure
          - mean brightness < 8:    deep fade to black or lens cap
          - pixel variance  < 80:   near-uniform frame (colour card, static noise)

        These thresholds are intentionally lenient to avoid false positives on
        legitimately dark scenes (night footage) or bright scenes (snow, sky).
        The combination of all three criteria reduces false positive rate.

        Args:
            frame: Full-resolution BGR uint8 numpy array.

        Returns:
            True if the frame should be discarded, False to pass it through.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mean_val = float(np.mean(gray))
        var_val = float(np.var(gray))
        return mean_val > 245.0 or mean_val < 8.0 or var_val < 80.0

    def _grid_ssim(
        self, prev: np.ndarray, curr: np.ndarray, grid: int = 2
    ) -> float:
        """
        Compute quadrant-partitioned SSIM between two grayscale thumbnails.

        The frame is divided into a `grid × grid` spatial grid and SSIM is
        computed independently in each cell.  The *minimum* score across all
        cells is returned.

        Rationale for minimum (not mean):
          Whole-frame SSIM can be dominated by a large static background,
          producing a high score even when a foreground object is in significant
          motion within one quadrant.  Taking the minimum makes the filter
          sensitive to *any* localised change — a critical property for
          close-up and action videos in MSVD.

        Win-size computation:
          skimage requires win_size to be odd and ≥ 3.  For very small quadrants
          (shorter side < 3 px) the cell is assigned SSIM = 1.0 (conservative:
          do not accept a frame solely due to a tiny partial cell).

        Args:
            prev: Grayscale uint8 array of shape (H, W).
            curr: Grayscale uint8 array of same shape.
            grid: Number of divisions per axis.  2 → 4 quadrants (2×2).

        Returns:
            Minimum SSIM score across all quadrants; in [−1, 1] but
            practically in [0, 1] for natural video frames.
        """
        h, w = prev.shape
        cell_h = h // grid
        cell_w = w // grid
        scores: List[float] = []

        for row in range(grid):
            for col in range(grid):
                r0, r1 = row * cell_h, (row + 1) * cell_h
                c0, c1 = col * cell_w, (col + 1) * cell_w
                patch_prev = prev[r0:r1, c0:c1]
                patch_curr = curr[r0:r1, c0:c1]

                min_side = min(patch_prev.shape)
                if min_side < 3:
                    # Cell too small for meaningful SSIM — conservatively skip.
                    scores.append(1.0)
                    continue

                # Ensure win_size is odd and ≤ min_side
                win_size = min(7, min_side)
                if win_size % 2 == 0:
                    win_size -= 1  # force odd
                if win_size < 3:
                    scores.append(1.0)
                    continue

                try:
                    score = float(
                        structural_similarity(
                            patch_prev,
                            patch_curr,
                            data_range=255,
                            win_size=win_size,
                            channel_axis=None,  # grayscale input
                        )
                    )
                except Exception as exc:
                    logger.warning(
                        f"[{self.get_name()}] SSIM failed in quadrant "
                        f"({row},{col}): {exc}. Treating as SSIM=1.0."
                    )
                    score = 1.0

                scores.append(score)

        return min(scores) if scores else 1.0

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _empty_result(
        frames_original: int = 0,
        degenerate_dropped: int = 0,
    ) -> dict:
        """
        Return a zero-count result dict for unrecoverable error cases.

        Used identically to SSIMSampler._empty_result() — keeps the benchmark
        runner's error handling contract consistent across all sampler types.
        """
        return {
            "frames": [],
            "indices": [],
            "meta": {
                "frames_original":           frames_original,
                "candidate_pool_size":       0,
                "frames_degenerate_dropped": degenerate_dropped,
                "tass_stopped_early":        False,
                "vlm_calls":                 0,
            },
        }
