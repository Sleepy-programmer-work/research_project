"""
experiments/run_qwen_baseline.py — CLI entry-point for Qwen2.5-VL baseline benchmark.

Delegates to shared modules for all cross-cutting concerns:
  benchmark_setup.py   — logging, GPU name, output dirs
  benchmark_data.py    — MSVD video download, reference captions
  benchmark_report.py  — Markdown report generation
  qwen_pipeline.py     — per-video frame extraction + Qwen inference + scoring
"""
import argparse
import json
import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch.multiprocessing as mp
from datasets import load_dataset
from tqdm import tqdm

from config.settings import settings
from evaluation.corpus_idf import load_corpus_idf
from evaluation.statistics import compute_statistics
from models.qwen_vl_loader import QwenVLLoader

from experiments.benchmark_setup import setup_logging, get_gpu_name, build_output_dirs
from experiments.benchmark_data import ensure_msvd_videos, build_reference_captions
from experiments.benchmark_report import generate_report
from experiments.qwen_pipeline import run_qwen_video, check_ollama_model

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)

logger = logging.getLogger("qwen_baseline")

ROW_COLUMNS = [
    "video_id", "benchmark_type", "sampling_method", "aggregation_method",
    "caption_mode", "frames_selected", "actual_frames_used", "oom_fallback_triggered",
    "processing_time_s", "peak_vram_mb", "peak_ram_delta_mb", "gpu_utilization_pct",
    "cider", "bleu1", "bleu4", "rouge_l", "meteor", "generated_caption",
]


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Qwen2.5-VL Standalone Baseline Benchmark")
    p.add_argument("--videos",     type=int, default=None)
    p.add_argument("--config",     type=str, default="configs/qwen_baseline.yaml")
    p.add_argument("--output-dir", type=str, default=None)
    p.add_argument("--log-level",  type=str, default="INFO")
    return p.parse_args()


def _load_config(args) -> tuple:
    """Return (qwen_cfg, out_dir, exp_cfg)."""
    settings.reload(args.config)
    exp = settings.config.get("experiment", {})
    qwen_cfg = settings.config.get("qwen_vl", {})
    out_dir = Path(args.output_dir or exp.get("output_dir", "./results/qwen_baseline"))
    return qwen_cfg, out_dir, exp


def _build_metadata(qwen_cfg, gpu_name, frames_per_video, mean_actual, oom_count,
                    ds_name, timestamp, results) -> dict:
    import platform
    return {
        "provider": "ollama",
        "model": qwen_cfg.get("model", "qwen2.5vl:3b"),
        "ollama_digest": qwen_cfg.get("ollama_digest", "fb90415cde1e"),
        "requested_frames": frames_per_video,
        "actual_frames_used": round(mean_actual, 2),
        "oom_fallback_triggered": oom_count > 0,
        "frame_resolution": [448, 448],
        "max_new_tokens": qwen_cfg.get("max_new_tokens", 60),
        "date": timestamp, "gpu": gpu_name, "cpu": platform.processor(),
        "videos": len(results), "dataset": ds_name,
    }


def main():
    args = _parse_args()
    qwen_cfg, out_dir, exp = _load_config(args)
    build_output_dirs(out_dir)
    setup_logging(args.log_level, out_dir / "logs")

    gpu_name = get_gpu_name()
    ds_name = exp.get("dataset", "friedrichor/MSVD")
    num_videos = args.videos or exp.get("videos", 10)
    cache_dir = Path(exp.get("cache_dir", "./cache"))

    ensure_msvd_videos(cache_dir)
    ds = load_dataset(ds_name, split="train")
    ds = ds.shuffle(seed=exp.get("seed", 42)).select(range(min(num_videos, len(ds))))

    model_name = qwen_cfg.get("model", "qwen2.5vl:3b")
    endpoint = qwen_cfg.get("ollama_endpoint", "http://localhost:11434")

    if not check_ollama_model(endpoint, model_name):
        return

    logger.info("Loading full MSVD corpus IDF for CIDEr scoring...")
    corpus_idf = load_corpus_idf(cache_dir=Path("results/cache"), force_rebuild=False)
    reference_captions = build_reference_captions(ds_name)

    qwen = QwenVLLoader.get(model_name=model_name, host=endpoint)
    frames_per_video = qwen_cfg.get("frames_per_video", 16)
    prompt_text = qwen_cfg.get("prompt", "")

    results, skipped, actual_frames_list, oom_count = [], [], [], 0
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    for row in tqdm(ds, desc="Qwen Baseline"):
        video_id = str(row.get("id", row.get("video_id", "unknown")))
        video_path = row.get("video_path") or f"./cache/{video_id}.avi"
        result = run_qwen_video(
            video_id, video_path, qwen, frames_per_video,
            prompt_text, reference_captions, corpus_idf, out_dir,
        )
        if result is None:
            skipped.append(video_id)
            continue
        actual_frames_list.append(result["actual_frames_used"])
        if result["oom_fallback_triggered"]:
            oom_count += 1
        results.append(result)

    if not results:
        logger.warning("No results generated. Exiting.")
        return

    csv_path = out_dir / "csv" / f"results_{timestamp}.csv"
    df = pd.DataFrame(results)[ROW_COLUMNS]
    df.to_csv(csv_path, index=False)
    df.to_csv(out_dir / "csv" / "results_latest.csv", index=False)
    logger.info(f"Results CSV: {csv_path}")

    stats_path = out_dir / "csv" / f"statistics_summary_{timestamp}.csv"
    compute_statistics(str(csv_path), str(stats_path))

    mean_actual = float(np.mean(actual_frames_list)) if actual_frames_list else 0.0
    metadata = _build_metadata(qwen_cfg, gpu_name, frames_per_video, mean_actual, oom_count,
                               ds_name, timestamp, results)
    (out_dir / "metadata" / f"run_info_{timestamp}.json").write_text(json.dumps(metadata, indent=2))

    report_path = out_dir / "reports" / f"benchmark_summary_{timestamp}.md"
    generate_report(str(stats_path), str(report_path), metadata)
    logger.info(f"Benchmark complete. Results in {out_dir}")


if __name__ == "__main__":
    main()
