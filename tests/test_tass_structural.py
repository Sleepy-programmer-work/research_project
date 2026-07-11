"""
tests/test_tass_structural.py — Output schema and parallel-list invariant tests.

Covers:
  - result dict schema (required keys)
  - frames == indices length invariant
  - meta['vlm_calls'] == len(frames)
  - frame indices monotonically increasing
  - sample() consistency with sample_with_metadata()
  - all frames are non-None uint8 BGR numpy arrays

Run with:
    PYTHONPATH=. pytest tests/test_tass_structural.py -v
"""

import os
import pytest
from samplers.tass import TASSSampler
from tests.helpers import make_video, scene_frame, noise_frame

_REQUIRED_META_KEYS = {
    "frames_original",
    "candidate_pool_size",
    "frames_degenerate_dropped",
    "tass_stopped_early",
    "vlm_calls",
}


class TestTASSStructuralContract:
    """Verify the output dict schema and parallel-list invariants."""

    def test_meta_keys_present(self):
        """sample_with_metadata() must always return all required meta keys."""
        frames = [scene_frame(i % 5) for i in range(30)]
        path = make_video(frames)
        try:
            result = TASSSampler(mode="fixed").sample_with_metadata(path)
            assert "meta" in result
            missing = _REQUIRED_META_KEYS - result["meta"].keys()
            assert not missing, f"Missing meta keys: {missing}"
        finally:
            os.unlink(path)

    def test_frames_and_indices_same_length(self):
        """len(result['frames']) must always equal len(result['indices'])."""
        frames = [scene_frame(i % 8) for i in range(60)]
        path = make_video(frames)
        try:
            result = TASSSampler(mode="fixed").sample_with_metadata(path)
            assert len(result["frames"]) == len(result["indices"]), (
                f"frames ({len(result['frames'])}) and indices ({len(result['indices'])}) "
                f"must be equal length"
            )
        finally:
            os.unlink(path)

    def test_vlm_calls_matches_frames_length(self):
        """meta['vlm_calls'] must equal len(result['frames'])."""
        frames = [scene_frame(i % 6) for i in range(30)]
        path = make_video(frames)
        try:
            result = TASSSampler(mode="fixed").sample_with_metadata(path)
            assert result["meta"]["vlm_calls"] == len(result["frames"])
        finally:
            os.unlink(path)

    def test_frame_indices_monotonically_increasing(self):
        """Output frame indices must be strictly increasing (temporal order preserved)."""
        frames = [scene_frame(i % 10) for i in range(60)]
        path = make_video(frames)
        try:
            result = TASSSampler(mode="fixed").sample_with_metadata(path)
            indices = result["indices"]
            if len(indices) > 1:
                diffs = [indices[i + 1] - indices[i] for i in range(len(indices) - 1)]
                assert all(d > 0 for d in diffs), f"Indices not increasing: {diffs}"
        finally:
            os.unlink(path)

    def test_sample_interface_matches_sample_with_metadata(self):
        """sample() must return the same frame count as sample_with_metadata()['frames']."""
        frames = [scene_frame(i % 5) for i in range(30)]
        path = make_video(frames)
        try:
            sampler = TASSSampler(mode="fixed")
            assert len(sampler.sample(path)) == len(sampler.sample_with_metadata(path)["frames"])
        finally:
            os.unlink(path)

    def test_frames_are_non_none_bgr_arrays(self):
        """Every frame in result['frames'] must be a non-None uint8 numpy array."""
        import numpy as np
        frames = [scene_frame(i % 4) for i in range(30)]
        path = make_video(frames)
        try:
            result = TASSSampler(mode="fixed").sample_with_metadata(path)
            for idx, f in enumerate(result["frames"]):
                assert f is not None, f"Frame {idx} is None"
                assert isinstance(f, np.ndarray), f"Frame {idx} is not ndarray"
                assert f.ndim == 3 and f.shape[2] == 3, f"Frame {idx} not BGR: {f.shape}"
        finally:
            os.unlink(path)
