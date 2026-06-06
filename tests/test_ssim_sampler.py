"""
tests/test_ssim_sampler.py — Edge case test suite for SSIMSampler.

Run with:
    PYTHONPATH=. pytest tests/test_ssim_sampler.py -v

All tests use synthetic in-memory videos written to temporary .avi files
so they run without any real video dataset or network access.

Test coverage:
  - First frame always kept (even for completely static video)
  - Identical frames → single frame selected (perfect SSIM = 1.0)
  - All-different frames → many frames selected
  - Threshold ordering: ssim_085 ≤ ssim_095 frames selected
  - Single-frame video
  - Nonexistent file → empty result (no crash)
  - Pathological threshold (0.01) → does not crash, returns valid result
  - frame_indices / ssim_scores length invariants
  - SSIMSamplerResult has all three TASS-required fields
"""

import numpy as np
import pytest
import tempfile
import os
import cv2
from samplers.ssim import SSIMSampler, ACCEPTANCE_RATE_MIN, ACCEPTANCE_RATE_MAX


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_video(frames: list[np.ndarray], fps: float = 30.0) -> str:
    """
    Write synthetic BGR frames to a temporary .avi file and return its path.

    The file must be manually deleted by the caller (os.unlink) after the test.
    Using delete=False because VideoWriter needs the file to exist on disk
    before it can write to it.
    """
    tmp = tempfile.NamedTemporaryFile(suffix=".avi", delete=False)
    tmp.close()  # close so VideoWriter can open it on Windows/Linux

    h, w = frames[0].shape[:2]
    out = cv2.VideoWriter(
        tmp.name,
        cv2.VideoWriter_fourcc(*"XVID"),
        fps,
        (w, h),
    )
    for f in frames:
        out.write(f)
    out.release()
    return tmp.name


def _solid_frame(color=(128, 64, 32), size=(320, 180)) -> np.ndarray:
    """Return a solid-colour BGR frame of the given size."""
    return np.full((*size[::-1], 3), color, dtype=np.uint8)


# ---------------------------------------------------------------------------
# Test class
# ---------------------------------------------------------------------------

class TestSSIMSamplerEdgeCases:

    # --- Basic acceptance ---

    def test_always_keeps_first_frame(self):
        """Even a completely static video must always yield at least 1 frame."""
        frames = [_solid_frame((0, 0, 0))] * 10
        path = _make_video(frames)
        try:
            s = SSIMSampler(threshold=0.90, name="ssim_090")
            result = s.sample_with_metadata(path)
            assert result.selected_frame_count >= 1, (
                "First frame must always be accepted regardless of threshold"
            )
        finally:
            os.unlink(path)

    def test_identical_frames_produce_one_frame(self):
        """
        A video where every frame is identical should produce exactly 1 accepted
        frame (only the first, since all subsequent SSIM scores = 1.0 ≥ threshold).
        """
        frames = [_solid_frame((100, 100, 100))] * 30
        path = _make_video(frames)
        try:
            s = SSIMSampler(threshold=0.90, name="ssim_090")
            result = s.sample_with_metadata(path)
            assert result.selected_frame_count == 1, (
                f"Identical frames should yield 1 accepted frame, got {result.selected_frame_count}"
            )
        finally:
            os.unlink(path)

    def test_all_different_frames_keep_many(self):
        """
        A video of fully random noise frames should accept nearly all of them
        because consecutive SSIM scores will be well below any reasonable threshold.
        """
        rng = np.random.default_rng(seed=0)
        frames = [
            rng.integers(0, 255, (180, 320, 3), dtype=np.uint8)
            for _ in range(30)
        ]
        path = _make_video(frames)
        try:
            s = SSIMSampler(threshold=0.90, name="ssim_090")
            result = s.sample_with_metadata(path)
            assert result.selected_frame_count > 5, (
                f"Random-noise video should yield many accepted frames, "
                f"got {result.selected_frame_count}"
            )
        finally:
            os.unlink(path)

    # --- Threshold ordering ---

    def test_threshold_085_more_aggressive_than_095(self):
        """
        A higher threshold (0.95) accepts more frames than a lower one (0.85)
        because more SSIM values fall below the higher bar.

        With a gradual colour-ramp video, both thresholds should accept different
        numbers of frames, and ssim_095 ≥ ssim_085.
        """
        frames = [
            _solid_frame((i * 8 % 255, i * 4 % 255, i * 2 % 255))
            for i in range(30)
        ]
        path = _make_video(frames)
        try:
            s085 = SSIMSampler(threshold=0.85, name="ssim_085")
            s095 = SSIMSampler(threshold=0.95, name="ssim_095")
            r085 = s085.sample_with_metadata(path)
            r095 = s095.sample_with_metadata(path)
            # Higher threshold → accepts more frames (SSIM < 0.95 is easier to satisfy)
            assert r095.selected_frame_count >= r085.selected_frame_count, (
                f"ssim_095 ({r095.selected_frame_count}) should be ≥ "
                f"ssim_085 ({r085.selected_frame_count})"
            )
        finally:
            os.unlink(path)

    # --- Edge case videos ---

    def test_single_frame_video(self):
        """A one-frame video should return exactly 1 frame with no crash."""
        path = _make_video([_solid_frame()])
        try:
            s = SSIMSampler(threshold=0.90, name="ssim_090")
            result = s.sample_with_metadata(path)
            assert result.selected_frame_count == 1
            assert result.original_frame_count >= 1
        finally:
            os.unlink(path)

    def test_nonexistent_video_returns_empty(self):
        """Missing file must return _empty_result(), not raise an exception."""
        s = SSIMSampler(threshold=0.90, name="ssim_090")
        result = s.sample_with_metadata("/nonexistent/does_not_exist.avi")
        assert result.selected_frame_count == 0
        assert result.frames == []
        assert result.frame_indices == []
        assert result.ssim_scores == []

    def test_fallback_triggers_on_pathological_threshold(self):
        """
        A threshold of 0.01 means 'accept only if SSIM < 0.01', which almost
        never happens for real video frames → near-zero acceptance rate →
        FPS-1 fallback should fire, or the result is otherwise valid.

        The test only asserts the sampler does not crash and returns a
        non-None frames list.  It does NOT assert fallback_used because
        some synthetic frame sequences might still satisfy the condition.
        """
        frames = [_solid_frame((i, i, i)) for i in range(60)]
        path = _make_video(frames)
        try:
            # Deliberately degenerate threshold
            s = SSIMSampler(threshold=0.01, name="ssim_001")
            result = s.sample_with_metadata(path)
            assert result is not None
            assert result.frames is not None  # list, possibly empty after fallback guard
        finally:
            os.unlink(path)

    # --- Structural invariants ---

    def test_result_indices_match_frames_length(self):
        """
        The three parallel lists (frames, frame_indices, ssim_scores) must
        always have the same length.  TASS Stage 2 depends on this invariant.
        """
        frames = [_solid_frame((i * 10 % 255, 0, 0)) for i in range(20)]
        path = _make_video(frames)
        try:
            s = SSIMSampler(threshold=0.90, name="ssim_090")
            result = s.sample_with_metadata(path)
            assert len(result.frames) == len(result.frame_indices), (
                "frames and frame_indices must have equal length"
            )
            assert len(result.frames) == len(result.ssim_scores), (
                "frames and ssim_scores must have equal length"
            )
        finally:
            os.unlink(path)

    def test_ssim_result_is_tass_compatible(self):
        """
        Verify SSIMSamplerResult has all three fields that TASS Stage 2 requires:
          - frames: full-resolution BGR numpy arrays
          - frame_indices: original temporal positions in the video
          - ssim_scores: SSIM score that caused each frame to be accepted

        Also checks that no frame in the accepted list is None.
        """
        frames = [_solid_frame((i * 5 % 255, i * 3 % 255, i % 255)) for i in range(15)]
        path = _make_video(frames)
        try:
            s = SSIMSampler(threshold=0.90, name="ssim_090")
            result = s.sample_with_metadata(path)
            # TASS Stage 2 requires these three attributes
            assert hasattr(result, "frames"),        "Missing 'frames' field"
            assert hasattr(result, "frame_indices"), "Missing 'frame_indices' field"
            assert hasattr(result, "ssim_scores"),   "Missing 'ssim_scores' field"
            # Every accepted frame must be a real array, not None
            assert all(f is not None for f in result.frames), (
                "No frame in accepted list should be None"
            )
        finally:
            os.unlink(path)

    def test_frame_indices_are_monotonically_increasing(self):
        """
        Frame indices must be in strictly increasing order because the sampler
        iterates forward through the video and never seeks backward.
        """
        frames = [_solid_frame((i * 7 % 255, i * 3 % 255, 0)) for i in range(30)]
        path = _make_video(frames)
        try:
            s = SSIMSampler(threshold=0.90, name="ssim_090")
            result = s.sample_with_metadata(path)
            if len(result.frame_indices) > 1:
                diffs = [
                    result.frame_indices[i + 1] - result.frame_indices[i]
                    for i in range(len(result.frame_indices) - 1)
                ]
                assert all(d > 0 for d in diffs), (
                    f"frame_indices must be strictly increasing, got diffs: {diffs}"
                )
        finally:
            os.unlink(path)

    def test_threshold_validation_raises_on_invalid_values(self):
        """
        SSIMSampler.__init__ must reject threshold values outside (0, 1).
        """
        with pytest.raises(ValueError):
            SSIMSampler(threshold=0.0, name="bad")
        with pytest.raises(ValueError):
            SSIMSampler(threshold=1.0, name="bad")
        with pytest.raises(ValueError):
            SSIMSampler(threshold=1.5, name="bad")
        with pytest.raises(ValueError):
            SSIMSampler(threshold=-0.1, name="bad")

    def test_get_name_returns_variant_name_not_ssim(self):
        """
        get_name() must return the variant name passed at construction,
        not a hardcoded 'ssim'.  This is critical for cache key correctness.
        """
        s085 = SSIMSampler(threshold=0.85, name="ssim_085")
        s090 = SSIMSampler(threshold=0.90, name="ssim_090")
        s095 = SSIMSampler(threshold=0.95, name="ssim_095")
        assert s085.get_name() == "ssim_085"
        assert s090.get_name() == "ssim_090"
        assert s095.get_name() == "ssim_095"

    def test_sample_interface_matches_sample_with_metadata(self):
        """
        sample() must return a list of frames identical to sample_with_metadata().frames.
        The BaseSampler interface must remain consistent with the metadata interface.
        """
        frames = [_solid_frame((i * 12 % 255, i * 5 % 255, i * 2 % 255)) for i in range(20)]
        path = _make_video(frames)
        try:
            s = SSIMSampler(threshold=0.90, name="ssim_090")
            simple = s.sample(path)
            result = s.sample_with_metadata(path)
            assert len(simple) == len(result.frames), (
                "sample() and sample_with_metadata().frames must yield the same count"
            )
        finally:
            os.unlink(path)
