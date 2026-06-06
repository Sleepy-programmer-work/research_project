import logging
from models.llm_loader import LLMLoader
from utils.gpu import flush_vram

logger = logging.getLogger(__name__)

def generate_final_caption(prompt: str, llm: LLMLoader, caption_mode: str) -> tuple[str, float, float]:
    if caption_mode == "vlm_only":
        return prompt, 0.0, 0.0
        
    logger.debug("Flushing VRAM before LLM generation.")
    mem_before, mem_after = flush_vram()
    
    caption = llm.generate(prompt)
    if not caption:
        caption = "Failed to generate caption."
        
    return caption, mem_before, mem_after
