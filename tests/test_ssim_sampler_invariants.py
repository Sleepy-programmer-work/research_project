"""
tests/test_ssim_sampler_invariants.py — Structural invariant tests for SSIMSampler.

These tests verify the parallel-list invariants that TASS Stage 2 depends on:
  - len(frames) == len(frame_indices) == len(ssim_scores)
  - frame_indices are strictly monotonically increasing
  - frames are non-None numpy arrays
  - SSIMSamplerResult has the three TASS-required attributes

Run with:
    PYTHONPATH=. pytest tests/test_ssim_sampler_invariants.py -v
"""

import os
from samplers.ssim import SSIMSampler
from tests.helpers import make_video, solid_frame


class TestSSIMSamplerInvariants:

    def test_result_lists_have_equal_length(self):
        """frames, frame_indices, ssim_scores must always have the same length."""
        frames = [solid_frame((i * 10 % 255, 0, 0)) for i in range(20)]
        path = make_video(frames)
        try:
            result = SSIMSampler(threshold=0.90, name="ssim_090").sample_with_metadata(path)
            assert len(result.frames) == len(result.frame_indices), \
                "frames and frame_indices must have equal length"
            assert len(result.frames) == len(result.ssim_scores), \
                "frames and ssim_scores must have equal length"
        finally:
            os.unlink(path)

    def test_frame_indices_are_monotonically_increasing(self):
        """Frame indices must be strictly increasing (no backward seeks)."""
        frames = [solid_frame((i * 7 % 255, i * 3 % 255, 0)) for i in range(30)]
        path = make_video(frames)
        try:
            result = SSIMSampler(threshold=0.90, name="ssim_090").sample_with_metadata(path)
            if len(result.frame_indices) > 1:
                diffs = [
                    result.frame_indices[i + 1] - result.frame_indices[i]
                    for i in range(len(result.frame_indices) - 1)
                ]
                assert all(d > 0 for d in diffs), \
                    f"frame_indices must be strictly increasing, got diffs: {diffs}"
        finally:
            os.unlink(path)

    def test_ssim_result_has_tass_required_fields(self):
        """SSIMSamplerResult must expose frames, frame_indices, ssim_scores."""
        frames = [solid_frame((i * 5 % 255, i * 3 % 255, i % 255)) for i in range(15)]
        path = make_video(frames)
        try:
            result = SSIMSampler(threshold=0.90, name="ssim_090").sample_with_metadata(path)
            assert hasattr(result, "frames"), "Missing 'frames' field"
            assert hasattr(result, "frame_indices"), "Missing 'frame_indices' field"
            assert hasattr(result, "ssim_scores"), "Missing 'ssim_scores' field"
            assert all(f is not None for f in result.frames), \
                "No frame in accepted list should be None"
        finally:
            os.unlink(path)

    def test_empty_result_classmethod(self):
        """SSIMSamplerResult._empty() must return a valid zeroed result."""
        from samplers.ssim_result import SSIMSamplerResult
        empty = SSIMSamplerResult._empty()
        assert empty.frames == []
        assert empty.frame_indices == []
        assert empty.ssim_scores == []
        assert empty.selected_frame_count == 0
        assert empty.fallback_used is False
