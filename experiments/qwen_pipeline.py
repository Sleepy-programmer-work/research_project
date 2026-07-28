"""
experiments/qwen_pipeline.py — Per-video processing for the Qwen VL baseline.

Separates the per-video pipeline logic from the CLI orchestration.
"""
import json
import logging
import os
import time
from pathlib import Path

import cv2
import psutil

from evaluation.metrics import build_gts_res, compute_all_metrics
from evaluation.telemetry import PeakResourceTracker
from experiments.benchmark_loop import cuda_sync
from utils.gpu import flush_vram

logger = logging.getLogger("qwen_baseline")


def uniform_sample_frames(video_path: str, n_frames: int) -> list:
    """Extract exactly n_frames uniformly distributed frames from video_path.

    Returns list of BGR numpy arrays. Returns [] if video is unreadable.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        logger.warning(f"Failed to open video: {video_path}")
        return []
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        logger.warning(f"Empty or corrupt video: {video_path}")
        return []
    indices = [int(i * total / n_frames) for i in range(n_frames)]
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret and frame is not None:
            frames.append(frame)
    cap.release()
    return frames


def run_qwen_video(
    video_id: str,
    video_path: str,
    qwen,
    frames_per_video: int,
    prompt_text: str,
    reference_captions_dict: dict,
    corpus_idf: dict,
    out_dir: Path,
) -> dict | None:
    """Run the Qwen VL pipeline for one video. Returns a result dict or None on failure."""
    if not os.path.exists(video_path):
        logger.warning(f"Video {video_path} not found. Skipping.")
        return None

    if psutil.virtual_memory().percent > 85:
        logger.warning("System RAM > 85% — swap pressure may inflate timing.")

    logger.info(f"Running {video_id} with Qwen2.5-VL baseline")
    try:
        cuda_sync()
        with PeakResourceTracker(device_index=0) as tracker:
            start = time.perf_counter()
            frames = uniform_sample_frames(video_path, frames_per_video)
            if not frames:
                raise RuntimeError(f"No frames extracted from {video_path}")
            caption, actual_used, fallback = qwen.generate_video_caption(frames=frames, prompt=prompt_text)
            cuda_sync()
            elapsed = time.perf_counter() - start

        _write_caption_json(out_dir, video_id, caption, reference_captions_dict.get(video_id, []))
        scores = _score_caption(video_id, caption, reference_captions_dict, corpus_idf)

        return {
            "video_id": video_id, "benchmark_type": "single_model",
            "sampling_method": "qwen_vl_3b", "aggregation_method": "native",
            "caption_mode": "direct",
            "frames_selected": frames_per_video, "actual_frames_used": actual_used,
            "oom_fallback_triggered": fallback,
            "processing_time_s": round(elapsed, 4),
            "peak_vram_mb": round(tracker.stats["peak_vram_mb"], 2),
            "peak_ram_delta_mb": round(tracker.stats["peak_ram_delta_mb"], 2),
            "gpu_utilization_pct": round(tracker.stats["peak_gpu_util_pct"], 1),
            **scores, "generated_caption": caption,
        }
    except Exception as exc:
        logger.error(f"Failed Qwen baseline on {video_id}: {exc}", exc_info=True)
        return None
    finally:
        flush_vram()


def _write_caption_json(out_dir: Path, video_id: str, caption: str, refs: list) -> None:
    path = out_dir / "captions" / f"{video_id}_qwen_vl_3b_native_direct.json"
    path.write_text(json.dumps({
        "video_id": video_id, "method": "qwen_vl_3b",
        "generated": caption, "reference": refs,
    }, indent=2))


def _score_caption(video_id: str, caption: str, ref_dict: dict, corpus_idf: dict) -> dict:
    gts, res = build_gts_res(
        video_ids=[video_id],
        reference_captions=ref_dict,
        generated_captions={video_id: caption},
        logger=logger,
    )
    vm = compute_all_metrics(gts, res, corpus_idf).get(str(video_id), {})
    return {k: vm.get(k, 0.0) for k in ("cider", "bleu1", "bleu4", "rouge_l", "meteor")}


def check_ollama_model(endpoint: str, model_name: str) -> bool:
    """Return True if Ollama is reachable and model appears available."""
    import requests
    try:
        resp = requests.get(f"{endpoint}/api/tags")
        resp.raise_for_status()
        models = [m["name"] for m in resp.json().get("models", [])]
        logger.info(f"Ollama local models: {models}")
        found = any(
            m == model_name or m.startswith(model_name + ":") or model_name.startswith(m + ":")
            for m in models
        )
        if not found:
            logger.warning(f"Model '{model_name}' not in local models. Attempting anyway.")
        return True
    except Exception as exc:
        logger.error(f"Ollama connection check failed at {endpoint}: {exc}")
        return False
