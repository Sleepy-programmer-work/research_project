import torch
import logging
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List
from utils.gpu import flush_vram
from utils.memory import check_ram_budget

logger = logging.getLogger(__name__)

_VLM_REVISION = "2024-08-26"  # Pinning revision stops dynamic cache corruption
_CAPTION_PROMPT = "Describe this image in a short sentence."
_OOM_PLACEHOLDER = "Failed to generate caption due to OOM."
_ERR_PLACEHOLDER = "Error during generation."


class VLMLoader:
    def __init__(self, model_name: str, fallback_name: str):
        self.model_name = model_name
        self.fallback_name = fallback_name
        self.model = None
        self.tokenizer = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.oom_recovery_triggered = False

    def load(self):
        check_ram_budget("VLM Loader")
        self._log_vram_info()
        logger.info(f"Attempting to load VLM: {self.model_name}")
        print("Loading Moondream2 directly into VRAM...")
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            torch_dtype=torch.float16,
            revision=_VLM_REVISION,
        ).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, revision=_VLM_REVISION
        )

    def _log_vram_info(self):
        if self.device != "cuda":
            return
        total_memory = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        logger.info(f"Total VRAM detected: {total_memory:.2f} GB")
        if total_memory < 6.0:
            logger.warning(f"VRAM is below 6 GB ({total_memory:.2f} GB). Loading may fail or OOM.")

    def generate_captions(self, images: List[Image.Image], batch_size: int = 4) -> List[str]:
        if not self.model:
            raise RuntimeError("VLMLoader.model is not loaded! Call load() before generate_captions().")
        self.oom_recovery_triggered = False
        captions: List[str] = []
        current_batch_size = batch_size
        i = 0
        while i < len(images):
            batch = images[i : i + current_batch_size]
            i, current_batch_size, captions = self._process_batch(
                batch, i, current_batch_size, captions
            )
        return captions

    def _process_batch(self, batch, i, current_batch_size, captions):
        try:
            with torch.no_grad():
                for img in batch:
                    enc = self.model.encode_image(img)
                    captions.append(self.model.answer_question(enc, _CAPTION_PROMPT, self.tokenizer))
            return i + current_batch_size, current_batch_size, captions
        except torch.cuda.OutOfMemoryError:
            return self._handle_oom(batch, i, current_batch_size, captions)
        except Exception as e:
            logger.error(f"Error generating caption: {e}")
            flush_vram()
            captions.extend([_ERR_PLACEHOLDER] * len(batch))
            return i + current_batch_size, current_batch_size, captions

    def _handle_oom(self, batch, i, current_batch_size, captions):
        self.oom_recovery_triggered = True
        logger.warning(f"OOM during VLM generation. Flushing VRAM and reducing batch size ({current_batch_size}).")
        flush_vram()
        if current_batch_size > 1:
            new_size = max(1, current_batch_size // 2)
            logger.info(f"Retrying with batch size {new_size}")
            return i, new_size, captions
        logger.error("Batch size is 1 and still OOM. Skipping batch.")
        captions.extend([_OOM_PLACEHOLDER] * len(batch))
        return i + current_batch_size, current_batch_size, captions
