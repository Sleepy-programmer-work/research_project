import gc
import torch
import logging

logger = logging.getLogger(__name__)

def flush_vram():
    """
    Flushes PyTorch VRAM and returns memory allocated before and after.
    Forces garbage collection and empties the CUDA cache.
    """
    mem_before = 0
    mem_after = 0
    
    gc.collect()
    
    if torch.cuda.is_available():
        mem_before = torch.cuda.memory_allocated() / (1024 ** 2)
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        mem_after = torch.cuda.memory_allocated() / (1024 ** 2)
        
        logger.debug(f"VRAM Flush - Before: {mem_before:.2f} MB, After: {mem_after:.2f} MB")
        
    return mem_before, mem_after
