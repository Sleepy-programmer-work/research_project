"""
experiments/benchmark_loop.py — Per-video pipeline execution and TASS metrics.

Responsibilities:
  - Run the full TASS pipeline for a single video (frame extraction → captioning
    → transcription → context aggregation → LLM → telemetry).
  - Write frame-selection metadata JSON artefacts.
  - Compute TASS-specific IEEE-paper metrics (VLM Calls Saved %, Semantic Yield).
"""
import json
import logging
import os
import time
from pathlib import Path

import psutil
import torch

from evaluation.telemetry import PeakResourceTracker
from pipeline import extract_frames, caption_frames, transcribe_audio, build_context, generate_final_caption
from samplers import SSIMSampler, TASSSampler
from samplers.fps1 import FPS1Sampler
from utils.gpu import flush_vram

logger = logging.getLogger("benchmark")


class CUDAContextBrokenError(RuntimeError):
    """Raised when the CUDA device enters an unrecoverable error state.

    Once a CUDA "unknown error" poisons the device context, no further
    GPU operations can succeed within this process.  Callers should catch
    this to skip remaining GPU work and save any partial results.
    """
    pass


def cuda_sync() -> None:
    """Flush CUDA queue if GPU available. Always call before and after timing.

    Tolerates a broken CUDA context so that a failed VLM call's error is
    captured by the surrounding try/except rather than being re-raised here
    and masking the real stack trace.
    """
    if torch.cuda.is_available():
        try:
            torch.cuda.synchronize()
        except RuntimeError as exc:
            logger.warning(f"cuda_sync: synchronize() failed (device may be in a broken state): {exc}")
            raise  # re-raise so the pipeline except block can log and skip this video


def _cuda_context_healthy() -> bool:
    """Return True if the CUDA context can still execute operations.

    A quick synchronize() probe — if it raises, the context is broken
    and no further CUDA calls will succeed in this process.
    """
    if not torch.cuda.is_available():
        return True
    try:
        torch.cuda.synchronize()
        return True
    except RuntimeError:
        return False


def run_video_pipeline(
    video_id: str,
    video_path: str,
    sampler,
    aggregator,
    caption_mode: str,
    vlm_loader,
    llm_loader,
    out_dir: Path,
) -> tuple[str | None, dict | None]:
    """Run all pipeline stages for one video; return (final_caption, telemetry)."""
    if not os.path.exists(video_path):
        logger.warning(f"Video {video_path} not found. Skipping.")
        return None, None

    if psutil.virtual_memory().percent > 85:
        logger.warning("System RAM > 85% — swap pressure may inflate timing.")

    logger.info(f"Running {video_id} | {sampler.get_name()} | {aggregator.get_name()} | {caption_mode}")

    try:
        cuda_sync()
        with PeakResourceTracker(device_index=0) as tracker:
            start = time.perf_counter()
            frames, tass_meta = _extract_frames(sampler, video_id, video_path, out_dir)
            raw_captions, _ = caption_frames(video_id, frames, vlm_loader, sampler.get_name())
            transcript, _, _ = transcribe_audio(video_path, video_id)
            prompt = build_context(raw_captions, transcript, aggregator, caption_mode)
            final_caption, _, _ = generate_final_caption(prompt, llm_loader, caption_mode)
            cuda_sync()
            elapsed = time.perf_counter() - start

        return final_caption, {
            "frames_selected": len(frames),
            "processing_time_s": elapsed,
            "peak_vram_mb": tracker.stats["peak_vram_mb"],
            "peak_ram_delta_mb": tracker.stats["peak_ram_delta_mb"],
            "gpu_utilization_pct": tracker.stats["peak_gpu_util_pct"],
            "tass_meta": tass_meta,
        }
    except Exception as exc:
        logger.error(f"Failed: {video_id}/{sampler.get_name()}/{caption_mode}: {exc}", exc_info=True)
        if not _cuda_context_healthy():
            raise CUDAContextBrokenError(
                f"CUDA context unrecoverable after {video_id}/{sampler.get_name()}/{caption_mode}. "
                f"Original error: {exc}"
            ) from exc
        return None, None
    finally:
        flush_vram()


def _extract_frames(sampler, video_id: str, video_path: str, out_dir: Path):
    """Dispatch to the correct sampler path; write frame-selection JSON."""
    ssim_result = tass_meta = tass_indices = None
    if hasattr(sampler, "sample_with_metadata"):
        result = sampler.sample_with_metadata(video_path)
        frames, tass_meta, tass_indices = result["frames"], result["meta"], result["indices"]
        if isinstance(sampler, SSIMSampler):
            ssim_result = result
    else:
        frames = extract_frames(video_path, video_id, sampler)
    save_frame_selection_meta(out_dir, video_id, sampler, frames, ssim_result, tass_meta, tass_indices)
    return frames, tass_meta



def save_frame_selection_meta(out_dir, video_id, sampler, frames, ssim_result=None, tass_meta=None, tass_indices=None):
    """Write per-video frame-selection metadata JSON to results/frame_selection/."""
    if ssim_result is not None:
        meta = {
            "sampler": sampler.get_name(), "video_id": video_id,
            "original_frame_count": ssim_result.original_frame_count,
            "total_frames_meta": ssim_result.total_frames_meta,
            "selected_frame_count": ssim_result.selected_frame_count,
            "frame_indices": ssim_result.frame_indices,
            "reduction_pct": round(ssim_result.reduction_pct, 2),
            "average_ssim": round(ssim_result.average_ssim, 4),
            "threshold_used": ssim_result.threshold_used,
            "fallback_used": ssim_result.fallback_used,
            "fps": ssim_result.fps,
        }
    elif tass_meta is not None:
        meta = {
            "sampler": sampler.get_name(), "video_id": video_id,
            "original_frame_count": tass_meta.get("frames_original", None), 
            "selected_frame_count": len(frames),
            "frame_indices": tass_indices if tass_indices else [], 
            "reduction_pct": None, "average_ssim": None,
            "threshold_used": getattr(sampler, 'threshold', None), 
            "fallback_used": tass_meta.get("fallback_used", False),
            "fps": None,
        }
    else:
        meta = {
            "sampler": sampler.get_name(), "video_id": video_id,
            "original_frame_count": None, "selected_frame_count": len(frames),
            "frame_indices": [], "reduction_pct": None, "average_ssim": None,
            "threshold_used": None, "fallback_used": False, "fps": None,
        }
    path = out_dir / "frame_selection" / f"{video_id}_{sampler.get_name()}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(meta, indent=2))


def compute_tass_row_metrics(
    tass_meta: dict | None,
    frames_sel: int,
    cider_score: float,
    fps1_mean_calls: int,
) -> dict:
    """Return TASS IEEE-paper metric columns for one result row."""
    if tass_meta is None:
        return {
            "tass_candidate_pool": None, "tass_degenerate_dropped": None,
            "tass_stopped_early": None, "vlm_calls_saved_pct": float("nan"),
            "semantic_yield": float("nan"),
        }
    vlm_calls = tass_meta.get("vlm_calls", frames_sel)
    return {
        "tass_candidate_pool": tass_meta.get("candidate_pool_size"),
        "tass_degenerate_dropped": tass_meta.get("frames_degenerate_dropped"),
        "tass_stopped_early": tass_meta.get("tass_stopped_early"),
        "vlm_calls_saved_pct": (fps1_mean_calls - vlm_calls) / fps1_mean_calls * 100.0,
        "semantic_yield": cider_score / max(1, vlm_calls),
    }
