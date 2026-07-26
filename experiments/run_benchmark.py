"""
experiments/run_benchmark.py — Entry-point for the TASS benchmark suite.

Orchestrates the benchmark run by delegating to focused modules:
  benchmark_setup.py   — logging, hardware, output directories
  benchmark_data.py    — dataset download and reference captions
  benchmark_samplers.py — sampler / aggregator factories
  benchmark_loop.py    — per-video processing and report generation

This file contains only argument parsing, high-level orchestration, and the
CSV / statistics / plot output. All business logic lives in the modules above.
"""
import argparse
import json
import logging
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch.multiprocessing as mp
from tqdm import tqdm

from config.settings import settings
from evaluation.corpus_idf import load_corpus_idf
from evaluation.metrics import build_gts_res, compute_all_metrics
from evaluation.caption_io import save_raw_caption
from evaluation.statistics import compute_statistics
from models import VLMLoader, LLMLoader
from visualization.plots import generate_plots

from experiments.benchmark_setup import setup_logging, get_gpu_name, build_output_dirs, make_run_metadata
from experiments.benchmark_data import ensure_dataset_videos
from experiments.benchmark_samplers import get_samplers, get_aggregators
from experiments.benchmark_loop import run_video_pipeline, compute_tass_row_metrics, cuda_sync, CUDAContextBrokenError
from experiments.benchmark_report import generate_report

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)

logger = logging.getLogger("benchmark")

ROW_COLUMNS = [
    "video_id", "sampling_method", "aggregation_method", "caption_mode",
    "frames_selected", "processing_time_s", "peak_vram_mb", "peak_ram_delta_mb",
    "gpu_utilization_pct", "cider", "bleu1", "bleu4", "rouge_l", "meteor",
    "tass_candidate_pool", "tass_degenerate_dropped", "tass_stopped_early",
    "vlm_calls_saved_pct", "semantic_yield",
]


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Video Captioning Benchmark")
    p.add_argument("--videos",      type=int,   default=None)
    p.add_argument("--config",      type=str,   default="configs/benchmark.yaml")
    p.add_argument("--sampler",     type=str,   nargs="+", default=None)
    p.add_argument("--aggregation", type=str,   default=None)
    p.add_argument("--model",       type=str,   default=None)
    p.add_argument("--output-dir",  type=str,   default=None)
    p.add_argument("--log-level",   type=str,   default="INFO")
    return p.parse_args()


def _load_models(args, settings):
    vlm_cfg = settings.models.get("vlm", {})
    llm_name = args.model or settings.models.get("llm", {}).get("name", "phi3:mini")
    vlm = VLMLoader(vlm_cfg.get("name"), vlm_cfg.get("fallback"))
    logger.info("Pre-loading VLM...")
    vlm.load()
    llm = LLMLoader(llm_name, settings.models.get("llm", {}).get("host", "http://localhost:11434"))
    llm.warm_model()
    return vlm, llm, vlm_cfg, llm_name


def _run_configuration(s_name, a_name, c_mode, ds, sampler, aggregator, vlm, llm,
                        out_dir, ref_caps, corpus_idf, results, skipped, timestamp):
    """Process all videos for one (sampler, aggregator, mode) triple."""
    predictions, ground_truths, telemetry = {}, {}, {}

    for row in tqdm(ds, desc=f"{s_name} | {a_name} | {c_mode}"):
        video_id = str(row.get("video_id", "unknown"))
        video_path = row.get("video_path") or f"./cache/{video_id}.mp4"
        try:
            caption, tel = run_video_pipeline(video_id, video_path, sampler, aggregator, c_mode, vlm, llm, out_dir)
        except CUDAContextBrokenError:
            skipped.append(f"{video_id}/{s_name}/{a_name}/{c_mode} (CUDA_BROKEN)")
            raise
        if caption is None:
            skipped.append(f"{video_id}/{s_name}/{a_name}/{c_mode}")
            continue
        predictions[video_id] = caption
        ground_truths[video_id] = ref_caps.get(video_id, [])
        telemetry[video_id] = tel
        save_raw_caption(video_id, s_name, a_name, c_mode, caption, ground_truths[video_id])

    if not predictions:
        logger.warning(f"No results for {s_name}|{a_name}|{c_mode} — skipping evaluation.")
        return

    gts, res = build_gts_res(list(predictions.keys()), ground_truths, predictions, logger)
    metrics = compute_all_metrics(gts, res, corpus_idf)
    fps1_calls = max(1, int(np.mean([t["frames_selected"] for t in telemetry.values()]))) if s_name == "fps1" else 1

    for vid_id, caption in predictions.items():
        tel = telemetry[vid_id]
        vm = metrics.get(vid_id, {})
        scores = {k: vm.get(k, 0.0) for k in ("cider", "bleu1", "bleu4", "rouge_l", "meteor")}
        tass_cols = compute_tass_row_metrics(tel.get("tass_meta"), tel["frames_selected"], scores["cider"], fps1_calls)
        row_data = {
            "video_id": vid_id, "sampling_method": s_name,
            "aggregation_method": a_name, "caption_mode": c_mode,
            "frames_selected": tel["frames_selected"],
            "processing_time_s": tel["processing_time_s"],
            "peak_vram_mb": tel["peak_vram_mb"],
            "peak_ram_delta_mb": tel["peak_ram_delta_mb"],
            "gpu_utilization_pct": tel["gpu_utilization_pct"],
            "semantic_yield": scores["cider"] / tel["frames_selected"] if tel["frames_selected"] > 0 else 0.0,
            **scores, **tass_cols,
        }
        results.append({col: row_data.get(col) for col in ROW_COLUMNS})


def main():
    args = _parse_args()
    settings.reload(args.config)

    out_dir = Path(args.output_dir or settings.experiment.get("output_dir", "./results"))
    build_output_dirs(out_dir)
    setup_logging(args.log_level, out_dir / "logs")

    gpu_name = get_gpu_name()
    cache_dir = Path(settings.experiment.get("cache_dir", "./cache"))
    ds_name   = settings.experiment.get("dataset", "vishnutheepb/msrvtt")
    seed      = settings.experiment.get("seed", 42)
    num_videos = args.videos or settings.experiment.get("videos", 10)

    logger.info(f"Loading MSR-VTT dataset ({num_videos} videos, seed={seed}) via Kaggle...")
    ds, ref_caps = ensure_dataset_videos(
        num_videos=num_videos,
        seed=seed,
        cache_dir=cache_dir,
    )

    vlm, llm, vlm_cfg, llm_name = _load_models(args, settings)

    logger.info("Loading MSR-VTT corpus IDF for CIDEr scoring...")
    corpus_idf = load_corpus_idf(cache_dir=Path("results/cache"), force_rebuild=False)
    samplers, aggregators = get_samplers(), get_aggregators()

    sampler_names = args.sampler or settings.pipeline.get("samplers", [])
    agg_names = [args.aggregation] if args.aggregation else settings.pipeline.get("aggregators", [])
    caption_modes = settings.pipeline.get("caption_modes", [])

    results, skipped = [], []
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    cuda_broken = False

    for s_name in sampler_names:
        if cuda_broken:
            break
        sampler = samplers.get(s_name)
        if not sampler:
            logger.warning(f"Sampler '{s_name}' not registered — skipping.")
            continue
        for a_name in agg_names:
            if cuda_broken:
                break
            aggregator = aggregators.get(a_name)
            if not aggregator:
                continue
            for c_mode in caption_modes:
                if cuda_broken:
                    break
                try:
                    _run_configuration(s_name, a_name, c_mode, ds, sampler, aggregator,
                                       vlm, llm, out_dir, ref_caps, corpus_idf,
                                       results, skipped, timestamp)
                except CUDAContextBrokenError as exc:
                    logger.critical(
                        f"CUDA context is unrecoverably broken: {exc}. "
                        f"Aborting remaining configurations and saving partial results."
                    )
                    cuda_broken = True

    if skipped:
        (out_dir / "logs" / f"skipped_{timestamp}.txt").write_text("\n".join(skipped))
        logger.warning(f"Skipped {len(skipped)} video(s). See logs/skipped_{timestamp}.txt.")

    if not results:
        logger.warning("No results generated. Exiting.")
        return

    csv_path = out_dir / "csv" / f"results_{timestamp}.csv"
    df = pd.DataFrame(results)[ROW_COLUMNS]
    df.to_csv(csv_path, index=False)

    stats_path = out_dir / "csv" / f"statistics_summary_{timestamp}.csv"
    compute_statistics(str(csv_path), str(stats_path))
    generate_plots(str(csv_path), out_dir=out_dir, frame_selection_dir=out_dir / "frame_selection")

    metadata = make_run_metadata(timestamp, gpu_name, num_videos, ds_name, vlm_cfg.get("name"), llm_name)
    (out_dir / "metadata" / f"run_info_{timestamp}.json").write_text(json.dumps(metadata, indent=2))

    report_path = out_dir / "reports" / f"benchmark_summary_{timestamp}.md"
    generate_report(str(stats_path), str(report_path), metadata)
    logger.info(f"Benchmark complete. Results in {out_dir}")


if __name__ == "__main__":
    main()
