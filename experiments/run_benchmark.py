import os
import time
import json
import logging
import argparse
from datetime import datetime
from pathlib import Path
import psutil
import torch
import torch.multiprocessing as mp
import pynvml
import pandas as pd
import numpy as np
from datasets import load_dataset
from tqdm import tqdm
from collections import defaultdict

from config.settings import settings
from utils.gpu import flush_vram
from samplers import FPS1Sampler, FPS2Sampler, RandomSampler, SSIMSampler, SSIMSamplerResult, DSISSampler
from aggregation import RawAggregator, CentroidAggregator, TemporalAggregator
from models import VLMLoader, LLMLoader
from pipeline import extract_frames, caption_frames, transcribe_audio, build_context, generate_final_caption
from evaluation.metrics import build_gts_res, compute_all_metrics, save_raw_caption
from evaluation.statistics import compute_statistics
from visualization.plots import generate_plots
from evaluation.telemetry import PeakResourceTracker
from evaluation.corpus_idf import load_corpus_idf

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)

logger = logging.getLogger("benchmark")

# ---------------------------------------------------------------------------
# SSIM variant registry
# Each key is the unique sampler name used as cache key, CSV value, and JSON
# filename.  Changing a key here invalidates all cached captions for that variant.
# ---------------------------------------------------------------------------
SSIM_VARIANTS = {
    "ssim_085": 0.85,   # aggressive — more frames, catches subtler transitions
    "ssim_090": 0.90,   # balanced   — recommended default for mixed-content video
    "ssim_095": 0.95,   # conservative — only major scene transitions
}


def setup_logging(log_level_str: str, log_dir: Path):
    log_level = getattr(logging, log_level_str.upper(), logging.INFO)
    logger.setLevel(log_level)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"benchmark_{timestamp}.log"
    
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    fh = logging.FileHandler(log_file)
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    logger.addHandler(ch)


def get_samplers(cfg=None) -> dict:
    """
    Build the sampler registry from config.

    FPS-1, FPS-2, and Random are always registered.
    SSIM variants are registered only when their name appears in
    cfg.pipeline.samplers (or settings.pipeline.samplers if cfg is None).
    Each SSIM variant gets a unique name ('ssim_085', 'ssim_090', 'ssim_095')
    which is used as the cache key, CSV identifier, and JSON filename.

    Benchmark matrix note:
      Adding ssim_085 + ssim_090 + ssim_095 expands the matrix from
      18 → 36 configurations (6 samplers × 3 aggregators × 2 modes).
      On a 50-video run this roughly doubles wall-clock time.
      Run --sampler ssim_090 --videos 10 first as a smoke test.
    """
    pipeline_cfg = cfg.pipeline if cfg is not None else settings.pipeline
    active_samplers = pipeline_cfg.get("samplers", [])

    samplers = {
        "fps1":   FPS1Sampler(),
        "fps2":   FPS2Sampler(),
        "random": RandomSampler(seed=settings.experiment.get("seed", 42)),
    }

    # Register SSIM variants that are listed in the YAML sampler list
    ssim_count = 0
    for variant_name, threshold in SSIM_VARIANTS.items():
        if variant_name in active_samplers:
            samplers[variant_name] = SSIMSampler(threshold=threshold, name=variant_name)
            ssim_count += 1

    if ssim_count > 0:
        logger.info(
            f"Registered {ssim_count} SSIM variant(s): "
            f"{[v for v in SSIM_VARIANTS if v in active_samplers]}. "
            f"Benchmark matrix expanded to "
            f"{len([s for s in active_samplers if s in samplers])} samplers. "
            f"Tip: run --videos 10 --sampler ssim_090 before a full 50-video sweep."
        )

    return samplers


def get_aggregators():
    return {
        "raw":      RawAggregator(),
        "centroid": CentroidAggregator(),
        "temporal": TemporalAggregator(),
    }


def cuda_sync():
    """Flush CUDA queue if GPU is available. Always call before timing."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def build_reference_captions(dataset_name: str = "friedrichor/MSVD") -> dict[str, list[str]]:
    """
    Load all MSVD reference captions grouped by video_id.
    Returns: { str(video_id): [caption1, caption2, ...] }

    Inspects the dataset schema dynamically so it does not break
    if the field names change between dataset versions.
    """
    from datasets import load_dataset
    ds = load_dataset(dataset_name)

    refs = defaultdict(list)

    # Inspect first row to find the correct field names
    first_split = list(ds.keys())[0]
    first_row   = ds[first_split][0]
    logger.debug(f"MSVD dataset fields: {list(first_row.keys())}")

    id_field  = None
    cap_field = None
    for f in first_row.keys():
        if f in ("video_id", "id", "vid_id"):
            id_field = f
        if f in ("caption", "text", "description", "sentence"):
            cap_field = f

    if not id_field or not cap_field:
        raise ValueError(
            f"Cannot find video_id or caption fields in MSVD dataset. "
            f"Found fields: {list(first_row.keys())}. "
            f"Update build_reference_captions() to match."
        )

    logger.info(f"Using fields: id='{id_field}', caption='{cap_field}'")

    for split_name in ds.keys():
        for row in ds[split_name]:
            vid = str(row[id_field])
            cap_val = row[cap_field]
            
            if isinstance(cap_val, list):
                for c in cap_val:
                    c_clean = str(c).strip()
                    if vid and c_clean:
                        refs[vid].append(c_clean)
            else:
                c_clean = str(cap_val).strip()
                if vid and c_clean:
                    refs[vid].append(c_clean)

    n_videos   = len(refs)
    n_caps     = sum(len(v) for v in refs.values())
    avg_caps   = n_caps / n_videos if n_videos else 0

    logger.info(
        f"Reference captions loaded: {n_videos} videos, "
        f"{n_caps} total captions, {avg_caps:.1f} avg per video."
    )

    # Sanity check — flag any videos with only 1 reference
    sparse = [v for v, caps in refs.items() if len(caps) < 2]
    if sparse:
        logger.warning(
            f"{len(sparse)} videos have fewer than 2 reference captions. "
            f"CIDEr is less reliable with single references."
        )

    return dict(refs)


def generate_report(csv_path: str, report_path: str, metadata: dict):
    df = pd.read_csv(csv_path)
    if df.empty:
        return
        
    # Fix 6: determine ranking metric safely
    RANKING_PRIORITY = ["cider_mean", "bleu1_mean", "rouge_l_mean"]
    ranking_warnings = []
    
    primary_metric = None
    for candidate in RANKING_PRIORITY:
        if candidate in df.columns and df[candidate].max() > 0.0:
            primary_metric = candidate
            break
            
    if primary_metric is None:
        raise RuntimeError("All quality metrics are zero — pipeline evaluation failed.")
        
    if primary_metric != "cider_mean":
        ranking_warnings.append(
            f"> WARNING: CIDEr is degenerate (all zero). "
            f"Ranking on `{primary_metric}` instead. "
            f"Re-run after fixing the CIDEr evaluation pipeline."
        )
        
    best_quality = df.loc[df[primary_metric].idxmax()]
    fastest = df.loc[df['processing_time_s_mean'].idxmin()]
    vram_eff = df.loc[df['peak_vram_mb_mean'].idxmin()]
    ram_eff = df.loc[df['peak_ram_delta_mb_mean'].idxmin()]
    
    report_header = "\n".join(ranking_warnings) + "\n\n" if ranking_warnings else ""
    
    report_content = f"""# Benchmark Summary
{report_header}Generated on: {metadata['date']}

## Run Metadata
- **Videos Processed**: {metadata['videos']}
- **Dataset**: {metadata['dataset']}
- **GPU**: {metadata['gpu']}
- **CPU**: {metadata['cpu']}
- **VLM**: {metadata['vlm']}
- **LLM**: {metadata['llm']}

## Highlights
- **Best Quality (Highest Mean {primary_metric.replace('_mean', '').upper()})**: {best_quality['sampling_method']} + {best_quality['aggregation_method']} ({best_quality['caption_mode']}) - {best_quality[primary_metric]:.2f}
- **Fastest Method**: {fastest['sampling_method']} + {fastest['aggregation_method']} ({fastest['caption_mode']}) - {fastest['processing_time_s_mean']:.2f}s
- **Most VRAM Efficient**: {vram_eff['sampling_method']} + {vram_eff['aggregation_method']} ({vram_eff['caption_mode']}) - {vram_eff['peak_vram_mb_mean']:.2f} MB
- **Most RAM Efficient**: {ram_eff['sampling_method']} + {ram_eff['aggregation_method']} ({ram_eff['caption_mode']}) - {ram_eff['peak_ram_delta_mb_mean']:.2f} MB

## Full Results Table
{df.to_markdown()}
"""
    with open(report_path, "w") as f:
        f.write(report_content)


def ensure_msvd_videos(cache_dir: Path):
    """Downloads MSVD_Videos.zip from Hugging Face if not present, and extracts .avi files into cache_dir."""
    import zipfile
    import shutil
    from huggingface_hub import hf_hub_download

    cache_dir.mkdir(parents=True, exist_ok=True)
    
    existing_videos = list(cache_dir.glob("*.avi"))
    if len(existing_videos) >= 10:
        logger.info(f"Found {len(existing_videos)} local video files in cache. Skipping download/extraction.")
        return

    logger.info("MSVD video files not found in cache. Checking and downloading from Hugging Face (1.8 GB)...")
    try:
        zip_path = hf_hub_download(
            repo_id="friedrichor/MSVD",
            filename="MSVD_Videos.zip",
            repo_type="dataset"
        )
        logger.info(f"Video zip archive verified at: {zip_path}. Extracting and flattening to {cache_dir}...")
        
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            members = zip_ref.infolist()
            for member in tqdm(members, desc="Extracting videos"):
                if member.is_dir():
                    continue
                filename = os.path.basename(member.filename)
                if not filename.endswith(".avi"):
                    continue
                target_path = cache_dir / filename
                if target_path.exists():
                    continue
                with zip_ref.open(member) as source, open(target_path, "wb") as target:
                    shutil.copyfileobj(source, target)
                    
        logger.info("Successfully extracted and flattened MSVD videos to cache!")
    except Exception as e:
        logger.error(f"Failed to download/extract MSVD videos: {e}")


def _save_frame_selection_meta(
    out_dir: Path,
    video_id: str,
    sampler,
    frames,
    ssim_result=None,
):
    """
    Write per-video frame selection metadata JSON to results/frame_selection/.

    For SSIMSampler instances, the full SSIMSamplerResult metadata is written
    (reduction_pct, average_ssim, threshold_used, fallback_used, fps).
    For all other samplers, non-SSIM fields are set to None.

    The file is named {video_id}_{sampler.get_name()}.json, which is the unique
    key used for cache invalidation.  Changing the sampler name without deleting
    old JSON files will leave stale metadata on disk.
    """
    if ssim_result is not None:
        frame_meta = {
            "sampler":              sampler.get_name(),
            "video_id":             video_id,
            "original_frame_count": ssim_result.original_frame_count,
            "selected_frame_count": ssim_result.selected_frame_count,
            "frame_indices":        ssim_result.frame_indices,
            "reduction_pct":        round(ssim_result.reduction_pct, 2),
            "average_ssim":         round(ssim_result.average_ssim, 4),
            "threshold_used":       ssim_result.threshold_used,
            "fallback_used":        ssim_result.fallback_used,
            "fps":                  ssim_result.fps,
        }
    else:
        frame_meta = {
            "sampler":              sampler.get_name(),
            "video_id":             video_id,
            "original_frame_count": None,
            "selected_frame_count": len(frames),
            "frame_indices":        [],
            "reduction_pct":        None,
            "average_ssim":         None,
            "threshold_used":       None,
            "fallback_used":        False,
            "fps":                  None,
        }

    meta_path = (
        out_dir / "frame_selection"
        / f"{video_id}_{sampler.get_name()}.json"
    )
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps(frame_meta, indent=2))


def main():
    parser = argparse.ArgumentParser(description="Video Captioning Benchmark")
    parser.add_argument("--videos", type=int, default=None, help="Number of videos to process")
    parser.add_argument("--config", type=str, default="configs/benchmark.yaml", help="Path to config file")
    parser.add_argument("--sampler", type=str, nargs="+", default=None,
                        help="Specific sampler(s) to run (e.g. --sampler ssim_090 fps1)")
    parser.add_argument("--aggregation", type=str, default=None, help="Specific aggregation to run")
    parser.add_argument("--model", type=str, default=None, help="Override LLM model")
    parser.add_argument("--output-dir", type=str, default=None, help="Custom output directory")
    parser.add_argument("--log-level", type=str, default="INFO", help="Logging level")
    args = parser.parse_args()
    
    settings.reload(args.config)
    
    out_dir = Path(args.output_dir if args.output_dir else settings.experiment.get("output_dir", "./results"))
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "logs").mkdir(exist_ok=True)
    (out_dir / "csv").mkdir(exist_ok=True)
    (out_dir / "metadata").mkdir(exist_ok=True)
    (out_dir / "reports").mkdir(exist_ok=True)
    (out_dir / "frame_selection").mkdir(exist_ok=True)
    
    setup_logging(args.log_level, out_dir / "logs")
    
    logger.info("Initializing hardware monitors...")
    try:
        pynvml.nvmlInit()
        gpu_name = pynvml.nvmlDeviceGetName(pynvml.nvmlDeviceGetHandleByIndex(0))
        if isinstance(gpu_name, bytes):
            gpu_name = gpu_name.decode("utf-8")
    except Exception as e:
        logger.warning(f"Could not initialize NVML for GPU name: {e}")
        gpu_name = "Unknown GPU"
        
    import platform
    cpu_name = platform.processor()
    
    num_videos = args.videos if args.videos is not None else settings.experiment.get("videos", 10)
    
    cache_dir = Path(settings.experiment.get("cache_dir", "./cache"))
    ensure_msvd_videos(cache_dir)
    
    logger.info("Loading dataset...")
    ds_name = settings.experiment.get("dataset", "friedrichor/MSVD")
    ds = load_dataset(ds_name, split="train")
    
    ds = ds.shuffle(seed=settings.experiment.get("seed", 42)).select(range(min(num_videos, len(ds))))
    
    vlm_config = settings.models.get("vlm", {})
    llm_name = args.model if args.model else settings.models.get("llm", {}).get("name", "phi3:mini")
    
    vlm_loader = VLMLoader(vlm_config.get("name"), vlm_config.get("fallback"))
    logger.info("Pre-loading VLM to ensure singleton persistent memory across configurations...")
    vlm_loader.load()
    llm_loader = LLMLoader(llm_name, settings.models.get("llm", {}).get("host", "http://localhost:11434"))
    
    try:
        llm_loader.warm_model()
    except Exception as e:
        logger.error(f"Ollama load failed. Error: {e}")
        return
        
    # Load corpus IDF once at startup
    logger.info("Loading full MSVD corpus IDF for CIDEr scoring...")
    corpus_idf = load_corpus_idf(
        cache_dir=Path("results/cache"),
        force_rebuild=False,
    )
    logger.info(
        f"Corpus IDF ready: {corpus_idf['n_docs']} videos, "
        f"{len(corpus_idf['idf']):,} n-gram entries."
    )

    # Load MSVD reference captions grouped by video_id
    logger.info("Loading MSVD reference captions...")
    reference_captions_dict = build_reference_captions(ds_name)

    samplers = get_samplers()
    aggregators = get_aggregators()
    
    # --sampler now accepts a list (e.g. --sampler ssim_090 fps1)
    if args.sampler is not None:
        sampler_names = args.sampler if isinstance(args.sampler, list) else [args.sampler]
    else:
        sampler_names = settings.pipeline.get("samplers", [])

    agg_names = [args.aggregation] if args.aggregation else settings.pipeline.get("aggregators", [])
    caption_modes = settings.pipeline.get("caption_modes", [])
    
    results = []
    skipped_videos = []
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Schema columns definer
    ROW_COLUMNS = [
        "video_id",
        "sampling_method",
        "aggregation_method",
        "caption_mode",
        "frames_selected",
        "processing_time_s",
        "peak_vram_mb",           # absolute peak VRAM during pipeline
        "peak_ram_delta_mb",      # incremental RAM above model-load baseline
        "gpu_utilization_pct",    # peak GPU core utilisation % during pipeline
        "cider",
        "bleu1",
        "bleu4",
        "rouge_l",
        "meteor",
    ]

    # Log benchmark matrix size warning for SSIM runs
    active_ssim = [n for n in sampler_names if n in SSIM_VARIANTS]
    if active_ssim:
        total_configs = len(sampler_names) * len(agg_names) * len(caption_modes)
        logger.info(
            f"⚠ Benchmark matrix with SSIM: {len(sampler_names)} samplers × "
            f"{len(agg_names)} aggregators × {len(caption_modes)} modes = "
            f"{total_configs} configurations × {num_videos} videos. "
            f"This is ~{total_configs / 18:.1f}× the baseline 18-config run. "
            f"Wall-clock time will increase proportionally."
        )
    
    for s_name in sampler_names:
        sampler = samplers.get(s_name)
        if not sampler:
            logger.warning(f"Sampler '{s_name}' not found in registry — skipping.")
            continue
            
        for a_name in agg_names:
            aggregator = aggregators.get(a_name)
            if not aggregator:
                continue
                
            for c_mode in caption_modes:
                corpus_predictions = {}
                corpus_ground_truths = {}
                telemetry_data = {}
                
                for row in tqdm(ds, desc=f"Processing videos for {s_name} | {a_name} | {c_mode}"):
                    video_id = str(row.get("id", row.get("video_id", "unknown")))
                    
                    video_path = row.get("video_path")
                    if not video_path or not os.path.exists(video_path):
                        video_path = f"./cache/{video_id}.avi" 
                        if not os.path.exists(video_path):
                            logger.warning(f"Video {video_path} not found locally. Skipping.")
                            continue
                            
                    if psutil.virtual_memory().percent > 85:
                        logger.warning("WSL2 Alert: System RAM usage is over 85%. Swap pressure may be inflating timing measurements.")

                    logger.info(f"Running {video_id} with {s_name} | {a_name} | {c_mode}")
                    
                    try:
                        cuda_sync()
                        with PeakResourceTracker(device_index=0) as tracker:
                            start_time = time.perf_counter()
                            
                            # Stage 1 — frame extraction
                            # For SSIMSampler: call sample_with_metadata() to capture
                            # SSIM-specific metadata for the frame_selection JSON.
                            # For all other samplers: use the standard extract_frames() path.
                            ssim_result = None
                            if isinstance(sampler, SSIMSampler):
                                ssim_result = sampler.sample_with_metadata(video_path)
                                frames = ssim_result.frames
                            else:
                                frames = extract_frames(video_path, video_id, sampler)

                            frames_selected = len(frames)

                            # Save frame selection metadata (SSIM-enriched or standard)
                            _save_frame_selection_meta(
                                out_dir, video_id, sampler, frames, ssim_result
                            )
                            
                            # Stage 2 — VLM captioning (includes per-frame in raw mode)
                            raw_captions, oom_triggered = caption_frames(video_id, frames, vlm_loader, s_name)
                            
                            # Stage 3 — audio transcription
                            transcript, audio_present, audio_duration = transcribe_audio(video_path, video_id)
                            
                            # Stage 4 — context aggregation
                            prompt = build_context(raw_captions, transcript, aggregator, c_mode)
                            
                            # Stage 5 — LLM final caption
                            final_caption, m_bf, m_af = generate_final_caption(prompt, llm_loader, c_mode)
                            
                            raw_gts = reference_captions_dict.get(video_id, [])
                            save_raw_caption(video_id, s_name, a_name, c_mode, final_caption, raw_gts)
                            
                            cuda_sync()
                            elapsed = time.perf_counter() - start_time
                            
                        # tracker.stats available here, after __exit__
                        peak_vram   = tracker.stats["peak_vram_mb"]
                        peak_ram    = tracker.stats["peak_ram_delta_mb"]
                        peak_util   = tracker.stats["peak_gpu_util_pct"]
                        
                        corpus_predictions[video_id] = final_caption
                        corpus_ground_truths[video_id] = raw_gts
                        
                        telemetry_data[video_id] = {
                            "frames_selected": frames_selected,
                            "processing_time_s": elapsed,
                            "peak_vram_mb": peak_vram,
                            "peak_ram_delta_mb": peak_ram,
                            "gpu_utilization_pct": peak_util,
                        }
                        
                    except Exception as e:
                        logger.error(f"Failed: {video_id} / {s_name} / {a_name} / {c_mode}: {e}", exc_info=True)
                        skipped_videos.append(f"{video_id} / {s_name} / {a_name} / {c_mode}: {e}")
                        
                    flush_vram()
                
                if not corpus_predictions:
                    logger.warning(f"No successfully processed videos for {s_name} | {a_name} | {c_mode}. Skipping evaluation.")
                    continue
                
                logger.info(f"Computing corpus-level metrics for {s_name} | {a_name} | {c_mode}...")
                gts, res = build_gts_res(
                    video_ids=list(corpus_predictions.keys()),
                    reference_captions=corpus_ground_truths,
                    generated_captions=corpus_predictions,
                    logger=logger,
                )
                
                metrics = compute_all_metrics(gts, res, corpus_idf)
                
                for video_id in corpus_predictions.keys():
                    telemetry = telemetry_data.get(video_id, {})
                    video_metrics = metrics.get(video_id, {})
                    
                    scores = {
                        "cider":   video_metrics.get("cider", 0.0),
                        "bleu1":   video_metrics.get("bleu1", 0.0),
                        "bleu4":   video_metrics.get("bleu4", 0.0),
                        "rouge_l": video_metrics.get("rouge_l", 0.0),
                        "meteor":  video_metrics.get("meteor", 0.0),
                    }
                    
                    res_row = {
                        "video_id":            video_id,
                        "sampling_method":     s_name,
                        "aggregation_method":  a_name,
                        "caption_mode":        c_mode,
                        "frames_selected":     telemetry.get("frames_selected", 0),
                        "processing_time_s":   telemetry.get("processing_time_s", 0.0),
                        "peak_vram_mb":        telemetry.get("peak_vram_mb", 0.0),
                        "peak_ram_delta_mb":   telemetry.get("peak_ram_delta_mb", 0.0),
                        "gpu_utilization_pct": telemetry.get("gpu_utilization_pct", 0.0),
                    }
                    res_row.update(scores)
                    
                    filtered_row = {col: res_row[col] for col in ROW_COLUMNS if col in res_row}
                    results.append(filtered_row)
                    
    if skipped_videos:
        logger.warning(
            f"Skipped {len(skipped_videos)} video(s) due to errors. "
            f"These are excluded from all statistics. "
            f"Details in log file."
        )
        # Also write a skipped_videos.txt so you can investigate
        skip_log = out_dir / "logs" / f"skipped_{timestamp}.txt"
        with open(skip_log, "w", encoding="utf-8") as f:
            f.write("\n".join(skipped_videos))
            
    if not results:
        logger.warning("No results were generated. Exiting.")
        return
        
    csv_path = out_dir / "csv" / f"results_{timestamp}.csv"
    df = pd.DataFrame(results)
    # Ensure correct column ordering
    df = df[ROW_COLUMNS]
    df.to_csv(csv_path, index=False)
    
    stats_csv_path = out_dir / "csv" / f"statistics_summary_{timestamp}.csv"
    compute_statistics(str(csv_path), str(stats_csv_path))
    
    generate_plots(str(csv_path), out_dir=out_dir, frame_selection_dir=out_dir / "frame_selection")
    
    metadata = {
        "date":    timestamp,
        "gpu":     gpu_name,
        "cpu":     cpu_name,
        "videos":  num_videos,
        "vlm":     vlm_config.get("name"),
        "llm":     llm_name,
        "dataset": ds_name,
    }
    
    with open(out_dir / "metadata" / f"run_info_{timestamp}.json", "w") as f:
        json.dump(metadata, f, indent=2)
        
    report_path = out_dir / "reports" / f"benchmark_summary_{timestamp}.md"
    generate_report(str(stats_csv_path), str(report_path), metadata)
    
    logger.info(f"Benchmark complete. Results saved to {out_dir}")


if __name__ == "__main__":
    main()
