import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["TORCH_USE_CUDA_DSA"] = "1"

import sys
sys.path.append(".")

from config.settings import settings
from experiments.benchmark_data import ensure_dataset_videos
from experiments.benchmark_setup import build_output_dirs
from models import VLMLoader, LLMLoader
from experiments.benchmark_samplers import get_samplers, get_aggregators
from experiments.benchmark_loop import _extract_frames, cuda_sync
from pathlib import Path
import time

def main():
    settings.reload("configs/benchmark.yaml")
    out_dir = Path("./results/debug")
    build_output_dirs(out_dir)
    
    ds, ref_caps = ensure_dataset_videos(num_videos=100, seed=42, cache_dir=Path("./cache"))
    row = next(r for r in ds if str(r["video_id"]) == "video5360")
    video_path = row.get("video_path") or f"./cache/video5360.mp4"
    
    vlm_cfg = settings.models.get("vlm", {})
    vlm = VLMLoader(vlm_cfg.get("name"), vlm_cfg.get("fallback"))
    vlm.load()
    
    # No LLM for this debug run since Ollama is not available in my session
    
    sampler = get_samplers()["tass_adaptive"]
    aggregator = get_aggregators()["temporal"]
    caption_mode = "vlm_plus_llm"
    
    from evaluation.telemetry import PeakResourceTracker
    from pipeline import caption_frames, transcribe_audio, build_context
    from experiments.benchmark_loop import flush_vram
    
    for vid_id in ["video5360", "video1721"]:
        row = next(r for r in ds if str(r["video_id"]) == vid_id)
        video_path = row.get("video_path") or f"./cache/{vid_id}.mp4"
        print(f"--- Running {vid_id}: {video_path} ---")
        try:
            cuda_sync()
            with PeakResourceTracker(device_index=0) as tracker:
                start = time.perf_counter()
                frames, tass_meta = _extract_frames(sampler, vid_id, video_path, out_dir)
                raw_captions, _ = caption_frames(vid_id, frames, vlm, sampler.get_name())
                transcript, _, _ = transcribe_audio(video_path, vid_id)
                prompt = build_context(raw_captions, transcript, aggregator, caption_mode)
                cuda_sync()
                elapsed = time.perf_counter() - start
                print(f"Pytorch pipeline done in {elapsed:.2f}s.")
            flush_vram()
        except Exception as exc:
            print("Exception caught:")
            import traceback
            traceback.print_exc()
            break



if __name__ == "__main__":
    main()
