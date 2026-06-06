import torch
import logging
import gc
import psutil
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer
from utils.gpu import flush_vram
from typing import List
from config.settings import MAX_RAM_BUDGET_GB

logger = logging.getLogger(__name__)

class VLMLoader:
    def __init__(self, model_name: str, fallback_name: str):
        self.model_name = model_name
        self.fallback_name = fallback_name
        self.model = None
        self.tokenizer = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.oom_recovery_triggered = False
        
    def load(self):
        # RAM budget check
        current_ram_gb = psutil.virtual_memory().used / (1024 ** 3)
        if current_ram_gb >= MAX_RAM_BUDGET_GB:
            logger.warning(f"VLM Loader: RAM usage ({current_ram_gb:.2f} GB) is approaching MAX_RAM_BUDGET_GB ({MAX_RAM_BUDGET_GB} GB). Running gc.collect().")
            gc.collect()
            current_ram_gb = psutil.virtual_memory().used / (1024 ** 3)
            if current_ram_gb >= MAX_RAM_BUDGET_GB:
                logger.warning(f"VLM Loader: RAM usage remains high ({current_ram_gb:.2f} GB) after gc.collect(). Proceeding with caution.")

        if self.device == "cuda":
            total_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            logger.info(f"Total VRAM detected: {total_memory:.2f} GB")
            if total_memory < 6.0:
                logger.warning(f"VRAM is below 6 GB ({total_memory:.2f} GB). Loading may fail or OOM.")

        logger.info(f"Attempting to load VLM: {self.model_name}")
        print("Loading Moondream2 directly into VRAM...")
        
        # 1. LOAD THE MODEL FIRST (No separate AutoConfig!)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            torch_dtype=torch.float16,
            revision="2024-08-26" # Pinning revision stops dynamic cache corruption
        ).to(self.device)
        
        # 2. LOAD THE TOKENIZER SECOND
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            revision="2024-08-26"
        )

    def generate_captions(self, images: List[Image.Image], batch_size: int = 4) -> List[str]:
        if not self.model:
            raise RuntimeError("VLMLoader.model is not loaded! Call load() explicitly before generate_captions().")

        captions = []
        current_batch_size = batch_size
        self.oom_recovery_triggered = False
        
        i = 0
        while i < len(images):
            batch = images[i:i + current_batch_size]
            try:
                for img in batch:
                    enc_image = self.model.encode_image(img)
                    answer = self.model.answer_question(enc_image, "Describe this image in a short sentence.", self.tokenizer)
                    captions.append(answer)
                i += current_batch_size
            except torch.cuda.OutOfMemoryError:
                self.oom_recovery_triggered = True
                logger.warning(f"OOM Error during VLM generation. Flushing VRAM and reducing batch size (Current: {current_batch_size}).")
                flush_vram()
                if current_batch_size > 1:
                    current_batch_size = max(1, current_batch_size // 2)
                    logger.info(f"Retrying with batch size {current_batch_size}")
                else:
                    logger.error("Batch size is 1 and still OOM. Skipping remaining images in this batch.")
                    for _ in range(len(batch)):
                        captions.append("Failed to generate caption due to OOM.")
                    i += current_batch_size
            except Exception as e:
                logger.error(f"Error generating caption: {e}")
                for _ in range(len(batch)):
                    captions.append("Error during generation.")
                i += current_batch_size

        return captions
