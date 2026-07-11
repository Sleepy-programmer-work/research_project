"""
experiments/benchmark_setup.py — Logging, hardware detection, and output directories.

Responsibilities:
  - Configure file + stream logging for a benchmark run.
  - Probe GPU/CPU name via NVML / platform.
  - Create the standard output directory tree.
  - Assemble run metadata dict for the final JSON artefact.
"""
import logging
import platform
from datetime import datetime
from pathlib import Path
import pynvml

logger = logging.getLogger("benchmark")


def setup_logging(log_level_str: str, log_dir: Path) -> None:
    """Attach file and stream handlers to the benchmark logger."""
    log_level = getattr(logging, log_level_str.upper(), logging.INFO)
    logger.setLevel(log_level)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    fmt = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    fh = logging.FileHandler(log_dir / f"benchmark_{timestamp}.log")
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)


def get_gpu_name() -> str:
    """Return the GPU device name, or 'Unknown GPU' on failure."""
    try:
        pynvml.nvmlInit()
        raw = pynvml.nvmlDeviceGetName(pynvml.nvmlDeviceGetHandleByIndex(0))
        return raw.decode("utf-8") if isinstance(raw, bytes) else raw
    except Exception as exc:
        logger.warning(f"Could not initialize NVML for GPU name: {exc}")
        return "Unknown GPU"


def build_output_dirs(out_dir: Path) -> None:
    """Create the canonical subdirectory tree under out_dir."""
    for subdir in ("logs", "csv", "metadata", "reports", "frame_selection", "captions"):
        (out_dir / subdir).mkdir(parents=True, exist_ok=True)


def make_run_metadata(
    timestamp: str,
    gpu_name: str,
    num_videos: int,
    ds_name: str,
    vlm_name: str,
    llm_name: str,
) -> dict:
    """Assemble the run metadata dict written to results/metadata/run_info_*.json."""
    return {
        "date": timestamp,
        "gpu": gpu_name,
        "cpu": platform.processor(),
        "videos": num_videos,
        "vlm": vlm_name,
        "llm": llm_name,
        "dataset": ds_name,
    }
