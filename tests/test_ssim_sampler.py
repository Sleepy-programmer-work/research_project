"""
tests/test_ssim_sampler.py — Core acceptance & threshold tests for SSIMSampler.

Run with:
    PYTHONPATH=. pytest tests/test_ssim_sampler.py -v

All tests use synthetic in-memory videos from tests/helpers/video_factory.py
so they run without any real video dataset or network access.

See tests/test_ssim_sampler_invariants.py for structural-invariant tests.
"""

import os
import pytest
from samplers.ssim import SSIMSampler, ACCEPTANCE_RATE_MIN, ACCEPTANCE_RATE_MAX
from tests.helpers import make_video, solid_frame, noise_frame


class TestSSIMSamplerAcceptance:

    def test_always_keeps_first_frame(self):
        """Even a completely static video must always yield at least 1 frame."""
        frames = [solid_frame((0, 0, 0))] * 10
        path = make_video(frames)
        try:
            result = SSIMSampler(threshold=0.90, name="ssim_090").sample_with_metadata(path)
            assert result.selected_frame_count >= 1, "First frame must always be accepted"
        finally:
            os.unlink(path)

    def test_identical_frames_produce_one_frame(self):
        """A static video should produce exactly 1 accepted frame."""
        frames = [solid_frame((100, 100, 100))] * 30
        path = make_video(frames)
        try:
            result = SSIMSampler(threshold=0.90, name="ssim_090").sample_with_metadata(path)
            assert result.selected_frame_count == 1, (
                f"Identical frames should yield 1, got {result.selected_frame_count}"
            )
        finally:
            os.unlink(path)

    def test_all_different_frames_keep_many(self):
        """Random-noise frames should produce many accepted frames."""
        import numpy as np
        rng = np.random.default_rng(seed=0)
        frames = [rng.integers(0, 255, (180, 320, 3), dtype=np.uint8) for _ in range(30)]
        path = make_video(frames)
        try:
            result = SSIMSampler(threshold=0.90, name="ssim_090").sample_with_metadata(path)
            assert result.selected_frame_count > 5, (
                f"Random-noise video should yield many frames, got {result.selected_frame_count}"
            )
        finally:
            os.unlink(path)


class TestSSIMSamplerThreshold:

    def test_threshold_085_more_aggressive_than_095(self):
        """Higher threshold (0.95) accepts ≥ frames than lower (0.85)."""
        frames = [solid_frame((i * 8 % 255, i * 4 % 255, i * 2 % 255)) for i in range(30)]
        path = make_video(frames)
        try:
            r085 = SSIMSampler(threshold=0.85, name="ssim_085").sample_with_metadata(path)
            r095 = SSIMSampler(threshold=0.95, name="ssim_095").sample_with_metadata(path)
            assert r095.selected_frame_count >= r085.selected_frame_count, (
                f"ssim_095 ({r095.selected_frame_count}) should ≥ ssim_085 ({r085.selected_frame_count})"
            )
        finally:
            os.unlink(path)

    def test_threshold_validation_raises_on_invalid_values(self):
        """SSIMSampler.__init__ must reject thresholds outside (0, 1)."""
        for bad in (0.0, 1.0, 1.5, -0.1):
            with pytest.raises(ValueError):
                SSIMSampler(threshold=bad, name="bad")

    def test_get_name_returns_variant_name(self):
        """get_name() must return the exact name passed at construction."""
        for name in ("ssim_085", "ssim_090", "ssim_095"):
            threshold = float(f"0.{name.split('_')[1]}")
            assert SSIMSampler(threshold=threshold, name=name).get_name() == name


class TestSSIMSamplerEdgeCases:

    def test_single_frame_video(self):
        """A one-frame video should return exactly 1 frame without crashing."""
        path = make_video([solid_frame()])
        try:
            result = SSIMSampler(threshold=0.90, name="ssim_090").sample_with_metadata(path)
            assert result.selected_frame_count == 1
            assert result.original_frame_count >= 1
        finally:
            os.unlink(path)

    def test_nonexistent_video_returns_empty(self):
        """Missing file must return empty result without raising."""
        result = SSIMSampler(threshold=0.90, name="ssim_090").sample_with_metadata(
            "/nonexistent/does_not_exist.avi"
        )
        assert result.selected_frame_count == 0
        assert result.frames == []
        assert result.frame_indices == []
        assert result.ssim_scores == []

    def test_pathological_threshold_does_not_crash(self):
        """A degenerate threshold of 0.01 must not raise — result must be valid."""
        frames = [solid_frame((i, i, i)) for i in range(60)]
        path = make_video(frames)
        try:
            result = SSIMSampler(threshold=0.01, name="ssim_001").sample_with_metadata(path)
            assert result is not None
            assert result.frames is not None
        finally:
            os.unlink(path)

    def test_sample_interface_matches_sample_with_metadata(self):
        """sample() must return the same count as sample_with_metadata().frames."""
        frames = [solid_frame((i * 12 % 255, i * 5 % 255, i * 2 % 255)) for i in range(20)]
        path = make_video(frames)
        try:
            s = SSIMSampler(threshold=0.90, name="ssim_090")
            assert len(s.sample(path)) == len(s.sample_with_metadata(path).frames)
        finally:
            os.unlink(path)
