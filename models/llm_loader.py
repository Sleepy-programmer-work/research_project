import logging
import requests
import gc
import psutil
from typing import Optional
from config.settings import MAX_RAM_BUDGET_GB

logger = logging.getLogger(__name__)

class LLMLoader:
    def __init__(self, model_name: str, host: str = "http://localhost:11434"):
        self.model_name = model_name
        self.host = host

    def warm_model(self):
        """Loads the model into Ollama memory to prevent latency on first run."""
        # RAM budget check
        current_ram_gb = psutil.virtual_memory().used / (1024 ** 3)
        if current_ram_gb >= MAX_RAM_BUDGET_GB:
            logger.warning(f"LLM Loader: RAM usage ({current_ram_gb:.2f} GB) is approaching MAX_RAM_BUDGET_GB ({MAX_RAM_BUDGET_GB} GB). Running gc.collect().")
            gc.collect()
            current_ram_gb = psutil.virtual_memory().used / (1024 ** 3)
            if current_ram_gb >= MAX_RAM_BUDGET_GB:
                logger.warning(f"LLM Loader: RAM usage remains high ({current_ram_gb:.2f} GB) after gc.collect(). Warm up may be delayed.")

        logger.info(f"Warming up Ollama model: {self.model_name}")
        url = f"{self.host}/api/generate"
        payload = {
            "model": self.model_name,
            "prompt": "",
            "keep_alive": "1h"
        }
        try:
            response = requests.post(url, json=payload, timeout=30)
            response.raise_for_status()
            logger.info("Ollama model warmed up successfully.")
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to warm up Ollama model. Is Ollama running? Error: {e}")
            raise RuntimeError(f"Ollama connection failed: {e}")

    def generate(self, prompt: str) -> Optional[str]:
        url = f"{self.host}/api/generate"
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.2
            }
        }
        try:
            response = requests.post(url, json=payload, timeout=120)
            response.raise_for_status()
            return response.json().get("response", "").strip()
        except requests.exceptions.RequestException as e:
            logger.error(f"Ollama generation failed: {e}")
            return None
