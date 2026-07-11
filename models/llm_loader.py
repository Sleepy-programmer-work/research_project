import logging
import requests
from typing import Optional
from utils.memory import check_ram_budget

logger = logging.getLogger("benchmark")

_GENERATE_ENDPOINT = "/api/generate"
_DEFAULT_TEMPERATURE = 0.2
_WARMUP_KEEP_ALIVE = "1h"


class LLMLoader:
    def __init__(self, model_name: str, host: str = "http://localhost:11434"):
        self.model_name = model_name
        self.host = host

    def warm_model(self):
        """Load the model into Ollama memory to prevent latency on first run."""
        check_ram_budget("LLM Loader")
        logger.info(f"Warming up Ollama model: {self.model_name}")
        try:
            response = requests.post(
                f"{self.host}{_GENERATE_ENDPOINT}",
                json={"model": self.model_name, "prompt": "", "keep_alive": _WARMUP_KEEP_ALIVE},
                timeout=30,
            )
            response.raise_for_status()
            logger.info("Ollama model warmed up successfully.")
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to warm up Ollama model. Is Ollama running? Error: {e}")
            raise RuntimeError(f"Ollama connection failed: {e}")

    def generate(self, prompt: str) -> Optional[str]:
        try:
            response = requests.post(
                f"{self.host}{_GENERATE_ENDPOINT}",
                json={
                    "model": self.model_name,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"temperature": _DEFAULT_TEMPERATURE},
                },
                timeout=120,
            )
            response.raise_for_status()
            return response.json().get("response", "").strip()
        except requests.exceptions.RequestException as e:
            logger.error(f"Ollama generation failed: {e}")
            return None
