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
    def vlm_model_id(self) -> str:
        """VLM model identifier used in cache keys and HuggingFace lookups.

        Reads from models.vlm.name in the YAML.  Falls back to 'vikhyatk/moondream2'
        so the field is always non-empty — an empty model ID would corrupt cache keys.
        """
        return self.config.get("models", {}).get("vlm", {}).get("name", "vikhyatk/moondream2")

    @property
    def vlm_revision(self) -> str:
        """Pinned VLM revision/commit hash used in cache keys.

        CACHE KEY CONTRACT: This value is baked into every frame-caption cache filename
        as {video_id}_{method}_{model}_{revision}.json.  An absent or changed revision
        must result in a different filename so old captions are NEVER silently reused.

        Reads from models.vlm.revision in benchmark.yaml.  The hardcoded fallback
        '2024-08-26' matches the revision currently pinned in the YAML — it exists
        only as a safety net if the YAML key is accidentally deleted.  If you upgrade
        the model, bump BOTH the YAML value and this fallback in the same commit.
        """
        revision = self.config.get("models", {}).get("vlm", {}).get("revision", "2024-08-26")
        if not revision:
            raise RuntimeError(
                "settings.vlm_revision is empty. "
                "Add 'revision: <commit-hash>' under models.vlm in benchmark.yaml. "
                "An empty revision would corrupt frame-caption cache keys."
            )
        return revision


    @property
    def pipeline(self) -> dict:
        return self.config.get("pipeline", {})

# Global settings instance
settings = Settings()
