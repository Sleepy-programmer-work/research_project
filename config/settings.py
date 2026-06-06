import yaml
from pathlib import Path
import os
import psutil
import torch

WSL2_MODE = os.path.exists("/proc/sys/fs/binfmt_misc/WSLInterop")

MAX_RAM_BUDGET_GB   = 10 if WSL2_MODE else 14
FRAME_BATCH_SIZE    = 2  if WSL2_MODE else 4
DATALOADER_WORKERS  = min(4, os.cpu_count() // 3)
VRAM_BUDGET_MB      = 5500   # leave 500 MB headroom on 6 GB card

if torch.cuda.is_available():
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False

class Settings:
    def __init__(self, config_path: str = "configs/benchmark.yaml"):
        self.config_path = Path(config_path)
        self.config = self._load_config()

    def _load_config(self) -> dict:
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found at {self.config_path}")
        with open(self.config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    def reload(self, config_path: str):
        self.config_path = Path(config_path)
        self.config = self._load_config()

    @property
    def experiment(self) -> dict:
        return self.config.get("experiment", {})

    @property
    def models(self) -> dict:
        return self.config.get("models", {})

    @property
    def pipeline(self) -> dict:
        return self.config.get("pipeline", {})

    @property
    def ssim(self) -> dict:
        """
        SSIM sampler configuration block from benchmark.yaml.

        Keys (all optional — SSIMSampler falls back to safe defaults):
          compare_size:        [width, height] for resize-before-compare (default: [256, 144])
          win_size:            SSIM window size, must be odd (default: 7)
          min_accepted_frames: minimum accepted frames guard (default: 1)
          max_accepted_frames: hard cap on accepted frames (default: 500)
          acceptance_rate_min: below this rate → FPS-1 fallback (default: 0.01)
          acceptance_rate_max: above this rate → log warning only (default: 0.99)
          variants:            dict mapping variant name → threshold float
        """
        return self.config.get("ssim", {})

# Global settings instance
settings = Settings()
