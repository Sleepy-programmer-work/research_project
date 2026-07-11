"""
tests/test_tass_stage1.py — Stage 1 tests: degenerate filter, pHash, and fallback.

Covers:
  - All-white (flash) video → fallback activates, produces frames
  - Mixed degenerate/normal → degenerate_dropped counts strided frames only
  - frames_degenerate_dropped type and range (non-negative int)
  - Static non-degenerate video → fallback activates (Laplacian ≈ 0 for gradients)
  - Dynamic noise video → large candidate pool (>5)
  - 3-block scene-change video → pool >= 2 candidates at transitions
  - Fallback metadata (fallback_used key) always present

Note: STAGE1_STRIDE=3 means only every 3rd frame is examined.  Degenerate counts
and candidate pools reflect strided-frame counts, not total video frames.

See test_tass_structural.py for output schema invariants.
See test_tass_stage2.py for Greedy FPS and mode tests.

Run with:
    PYTHONPATH=. pytest tests/test_tass_stage1.py -v
"""

import os
from samplers.tass import TASSSampler
from tests.helpers import (
    make_video, white_frame, noise_frame,
    static_nondegenerate_frame, scene_frame,
)

# TASSSampler._STAGE1_STRIDE = 3, so only every 3rd frame is examined.
_STRIDE = 3


class TestTASSStage1DegenerateFilter:
    """Verify the pre-stage degenerate frame purge."""

    def test_all_white_video_degenerate_count_matches_strided(self):
        """All-white video (mean > 245): degenerate count equals strided frame count.

        With stride=3 and 30 frames, 10 frames are examined, all white → 10 degenerate.
        Fallback activates because candidates is empty and frames_read > 0.
        """
        num_frames = 30
        frames = [white_frame()] * num_frames
        path = make_video(frames)
        try:
            result = TASSSampler(mode="fixed").sample_with_metadata(path)
            expected_examined = num_frames // _STRIDE
            assert result["meta"]["frames_degenerate_dropped"] == expected_examined, (
                f"Expected {expected_examined} degenerate (strided), got "
                f"{result['meta']['frames_degenerate_dropped']}"
            )
            # Fallback produces frames via fps1-style sampling (no degenerate filter)
            assert result["meta"]["fallback_used"] is True
            assert len(result["frames"]) >= 1, "Fallback must produce at least 1 frame"
        finally:
            os.unlink(path)

    def test_mixed_degenerate_at_least_half_strided_dropped(self):
        """Alternating white/noise frames: degenerate_dropped >= half of strided count.

        With stride=3 and 30 frames, 10 frames are examined.  White frames
        at even indices aligned with stride: 0, 6, 12, 18, 24 → 5 white.
        """
        num_frames = 30
        frames = [
            white_frame() if i % 2 == 0 else noise_frame(seed=i)
            for i in range(num_frames)
        ]
        path = make_video(frames)
        try:
            result = TASSSampler(mode="fixed").sample_with_metadata(path)
            dropped = result["meta"]["frames_degenerate_dropped"]
            strided_count = num_frames // _STRIDE
            assert dropped >= strided_count // 2, (
                f"Expected >= {strided_count // 2} degenerate drops "
                f"(from {strided_count} strided frames), got {dropped}"
            )
        finally:
            os.unlink(path)

    def test_degenerate_counter_is_non_negative_int(self):
        """frames_degenerate_dropped must be a non-negative int; 0 for noise."""
        frames = [noise_frame(seed=i) for i in range(20)]
        path = make_video(frames)
        try:
            result = TASSSampler(mode="fixed").sample_with_metadata(path)
            dropped = result["meta"]["frames_degenerate_dropped"]
            assert isinstance(dropped, int), "frames_degenerate_dropped must be int"
            assert dropped >= 0
            assert dropped == 0, f"Noise frames should never be degenerate, got {dropped}"
        finally:
            os.unlink(path)


class TestTASSStage1PHash:
    """Verify pHash-based candidate selection."""

    def test_static_gradient_video_triggers_fallback(self):
        """Identical gradient frames → all degenerate (Laplacian ≈ 0) → fallback.

        static_nondegenerate_frame() is a smooth horizontal gradient.  Its
        Laplacian variance is near zero, so the degenerate filter catches all
        strided frames.  The fallback activates and produces >= 1 frame.
        """
        base = static_nondegenerate_frame()
        frames = [base.copy() for _ in range(60)]
        path = make_video(frames)
        try:
            result = TASSSampler(mode="fixed", threshold=0.90).sample_with_metadata(path)
            assert result["meta"]["fallback_used"] is True, (
                "Gradient video should trigger fallback (Laplacian ≈ 0 → degenerate)"
            )
            assert len(result["frames"]) >= 1, (
                f"Fallback must produce >= 1 frame, got {len(result['frames'])}"
            )
        finally:
            os.unlink(path)

    def test_dynamic_video_candidate_pool_grows(self):
        """Random noise frames → large candidate pool (>5)."""
        frames = [noise_frame(seed=i) for i in range(60)]
        path = make_video(frames)
        try:
            result = TASSSampler(mode="fixed", threshold=0.90).sample_with_metadata(path)
            pool = result["meta"]["candidate_pool_size"]
            assert pool > 5, f"Dynamic video must produce >5 candidates, got {pool}"
            assert result["meta"]["fallback_used"] is False
        finally:
            os.unlink(path)

    def test_scene_change_video_admits_candidates_at_transitions(self):
        """3 distinct scene blocks → pool >= 2 (one candidate per transition)."""
        block_a = noise_frame(seed=1)
        block_b = noise_frame(seed=999)
        block_c = noise_frame(seed=12345)
        frames = (
            [block_a.copy()] * 10
            + [block_b.copy()] * 10
            + [block_c.copy()] * 10
        )
        path = make_video(frames)
        try:
            result = TASSSampler(mode="fixed", threshold=0.90).sample_with_metadata(path)
            pool = result["meta"]["candidate_pool_size"]
            assert pool >= 2, f"3-block video must produce >= 2 candidates, got {pool}"
            assert result["meta"]["fallback_used"] is False
        finally:
            os.unlink(path)


class TestTASSStage1Fallback:
    """Verify the fps1-style fallback when Stage 1 produces zero candidates."""

    def test_static_video_fallback_produces_frames(self):
        """Static non-degenerate content → fallback activates and produces frames.

        static_nondegenerate_frame() has Laplacian ≈ 0 (smooth gradient), so
        all strided frames are flagged degenerate.  The fallback (pure fps1,
        no degenerate filter) must produce at least 1 frame.
        """
        base = static_nondegenerate_frame()
        frames = [base.copy() for _ in range(90)]
        path = make_video(frames)
        try:
            result = TASSSampler(mode="fixed", threshold=0.90).sample_with_metadata(path)
            assert len(result["frames"]) >= 1, (
                f"Static video must produce >= 1 frame via fallback, got {len(result['frames'])}"
            )
            assert "fallback_used" in result["meta"], (
                "Meta must contain 'fallback_used' key"
            )
            assert result["meta"]["fallback_used"] is True
        finally:
            os.unlink(path)

    def test_all_white_video_fallback_activates(self):
        """All-white video → fallback activates (fps1-style, no degenerate filter).

        Even genuinely degenerate videos get frames via fallback, matching
        fps1 baseline behaviour.  The pipeline's VLM/captioning stages handle
        the low-quality content gracefully.
        """
        frames = [white_frame()] * 30
        path = make_video(frames)
        try:
            result = TASSSampler(mode="fixed").sample_with_metadata(path)
            assert result["meta"]["fallback_used"] is True, (
                "All-white video should trigger fallback"
            )
            assert len(result["frames"]) >= 1, (
                "Fallback must produce at least 1 frame even for degenerate content"
            )
        finally:
            os.unlink(path)

    def test_dynamic_video_no_fallback(self):
        """High-variance noise video → no fallback (pHash accepts frames normally)."""
        frames = [noise_frame(seed=i) for i in range(60)]
        path = make_video(frames)
        try:
            result = TASSSampler(mode="fixed").sample_with_metadata(path)
            assert result["meta"]["fallback_used"] is False, (
                "Dynamic video should not trigger fallback"
            )
        finally:
            os.unlink(path)

    def test_nonexistent_video_no_fallback(self):
        """Missing video → empty result, no fallback (frames_read = 0)."""
        result = TASSSampler(mode="fixed").sample_with_metadata("/nonexistent/ghost.avi")
        assert result["frames"] == []
        assert result["meta"]["fallback_used"] is False
