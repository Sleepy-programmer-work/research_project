"""
tests/test_tass_stage2.py — Stage 2 tests: Greedy FPS, fixed/adaptive modes.

Covers:
  - Fixed mode: output count == ceil(duration) when candidates >= K
  - Fixed mode: output count <= candidate pool when candidates < K
  - Adaptive mode: early stopping fires for low-diversity video
  - Adaptive mode: selects >= fixed mode for dynamic/diverse video
  - Any valid (non-all-degenerate) video yields >= 1 frame

Run with:
    PYTHONPATH=. pytest tests/test_tass_stage2.py -v
"""

import os
from samplers.tass import TASSSampler
from tests.helpers import (
    make_video, noise_frame, static_nondegenerate_frame, scene_frame, expected_k,
)


class TestTASSStage2GreedyFPS:
    """Verify greedy FPS and mode-specific behaviour."""

    def test_fixed_mode_output_respects_target_k(self):
        """Fixed mode + dynamic video → output count == ceil(duration)."""
        fps_val = 30.0
        num_frames = 300   # 10 seconds → target_k = 10
        frames = [noise_frame(seed=i) for i in range(num_frames)]
        path = make_video(frames, fps=fps_val)
        try:
            result = TASSSampler(mode="fixed", threshold=0.90).sample_with_metadata(path)
            expected = expected_k(num_frames, fps_val)
            assert len(result["frames"]) == expected, (
                f"Fixed mode must select {expected} frames, got {len(result['frames'])}"
            )
        finally:
            os.unlink(path)

    def test_fixed_mode_bounded_by_candidate_pool(self):
        """When candidate pool < target_k, output <= pool (never over-selects)."""
        fps_val = 30.0
        # 900 frames = 30 seconds → target_k = 30; static gradient → pool << 30
        base = static_nondegenerate_frame()
        frames = [base.copy() for _ in range(900)]
        path = make_video(frames, fps=fps_val)
        try:
            result = TASSSampler(mode="fixed", threshold=0.90).sample_with_metadata(path)
            pool = result["meta"]["candidate_pool_size"]
            n_selected = len(result["frames"])
            assert n_selected <= pool, (
                f"Output ({n_selected}) must not exceed pool ({pool})"
            )
            assert pool < 30, (
                f"Static gradient should produce pool < 30, got {pool}"
            )
            assert n_selected >= 1, "Must always return at least 1 frame"
        finally:
            os.unlink(path)

    def test_adaptive_mode_stops_early_for_low_diversity(self):
        """Adaptive + min_distance=0.50: near-identical candidates → early stop or pool<=2."""
        gradient = static_nondegenerate_frame()
        noise_sep = noise_frame(seed=42)
        frames = (
            [gradient.copy()] * 5
            + [noise_sep.copy()]
            + [gradient.copy()] * 5
        )
        path = make_video(frames)
        try:
            result = TASSSampler(mode="adaptive", threshold=0.90, min_distance=0.50).sample_with_metadata(path)
            pool = result["meta"]["candidate_pool_size"]
            stopped = result["meta"]["tass_stopped_early"]
            assert stopped or pool <= 2, (
                f"Must stop early or have tiny pool for near-static video. "
                f"stopped={stopped}, pool={pool}"
            )
        finally:
            os.unlink(path)

    def test_adaptive_selects_at_least_as_many_as_fixed_for_dynamic_video(self):
        """Adaptive mode selects >= fixed mode for a diverse 30-scene video."""
        fps_val = 30.0
        # 30 scenes × 10 frames = 300 total → target_k = 10
        frames = [scene_frame(scene_id) for scene_id in range(30) for _ in range(10)]
        path = make_video(frames, fps=fps_val)
        try:
            fixed = TASSSampler(mode="fixed", threshold=0.90, min_distance=0.05)
            adaptive = TASSSampler(mode="adaptive", threshold=0.90, min_distance=0.05)
            n_fixed = len(fixed.sample_with_metadata(path)["frames"])
            n_adaptive = len(adaptive.sample_with_metadata(path)["frames"])
            assert n_adaptive >= n_fixed, (
                f"Adaptive ({n_adaptive}) must select >= fixed ({n_fixed}) for diverse video"
            )
        finally:
            os.unlink(path)

    def test_always_returns_at_least_one_frame_for_valid_video(self):
        """Any non-all-degenerate video must produce >= 1 frame."""
        frames = [noise_frame(seed=i) for i in range(30)]
        path = make_video(frames)
        try:
            result = TASSSampler(mode="fixed", threshold=0.90).sample_with_metadata(path)
            assert len(result["frames"]) >= 1, "Must return at least 1 frame"
        finally:
            os.unlink(path)
