"""
tests/test_tass_errors.py — Error handling and constructor validation tests.

Covers:
  - Nonexistent file → empty result dict, no exception
  - Invalid mode string → ValueError at construction
  - Invalid threshold (outside (0,1)) → ValueError
  - Invalid min_distance (outside [0,1)) → ValueError
  - get_name() returns 'tass_fixed' / 'tass_adaptive' correctly
  - meta['frames_original'] approximately matches actual frame count (±3)
  - All-degenerate video → empty frames, 0 VLM calls

Run with:
    PYTHONPATH=. pytest tests/test_tass_errors.py -v
"""

import os
import pytest
from samplers.tass import TASSSampler
from tests.helpers import make_video, white_frame, noise_frame


class TestTASSErrorHandling:
    """Graceful handling of invalid inputs and edge cases."""

    def test_nonexistent_file_returns_empty_not_exception(self):
        """Missing video path must return empty result dict, never raise."""
        result = TASSSampler(mode="fixed").sample_with_metadata("/nonexistent/ghost.avi")
        assert result["frames"] == []
        assert result["indices"] == []
        assert result["meta"]["vlm_calls"] == 0

    def test_invalid_mode_raises_value_error(self):
        """An unknown mode string must raise ValueError at construction."""
        with pytest.raises(ValueError, match="mode"):
            TASSSampler(mode="turbo")

    def test_invalid_threshold_raises_value_error(self):
        """Threshold outside (0, 1) must raise ValueError."""
        for bad in (0.0, 1.0, 1.5, -0.5):
            with pytest.raises(ValueError, match="threshold"):
                TASSSampler(threshold=bad)

    def test_invalid_min_distance_raises_value_error(self):
        """min_distance outside [0, 1) must raise ValueError."""
        with pytest.raises(ValueError, match="min_distance"):
            TASSSampler(min_distance=-0.1)
        with pytest.raises(ValueError, match="min_distance"):
            TASSSampler(min_distance=1.0)

    def test_get_name_fixed(self):
        """get_name() must return 'tass_fixed'."""
        assert TASSSampler(mode="fixed").get_name() == "tass_fixed"

    def test_get_name_adaptive(self):
        """get_name() must return 'tass_adaptive'."""
        assert TASSSampler(mode="adaptive").get_name() == "tass_adaptive"

    def test_frames_original_approximately_correct(self):
        """meta['frames_original'] must be within ±3 of the actual frame count."""
        num_frames = 60
        frames = [noise_frame(seed=i) for i in range(num_frames)]
        path = make_video(frames)
        try:
            result = TASSSampler(mode="fixed").sample_with_metadata(path)
            actual = result["meta"]["frames_original"]
            assert abs(actual - num_frames) <= 3, (
                f"frames_original={actual} expected ~{num_frames} (±3)"
            )
        finally:
            os.unlink(path)

    def test_all_degenerate_video_triggers_fallback(self):
        """All-white video → fallback activates, produces frames via fps1-style sampling."""
        frames = [white_frame()] * 20
        path = make_video(frames)
        try:
            result = TASSSampler(mode="fixed").sample_with_metadata(path)
            assert result["meta"]["fallback_used"] is True
            assert len(result["frames"]) >= 1, "Fallback must produce at least 1 frame"
            assert result["meta"]["candidate_pool_size"] >= 1, (
                "Fallback candidate pool must be >= 1"
            )
        finally:
            os.unlink(path)
