import gc
import torch
import logging

logger = logging.getLogger(__name__)

def flush_vram():
    """
    Flushes PyTorch VRAM and returns memory allocated before and after.
    Forces garbage collection and empties the CUDA cache.

    Tolerates a broken CUDA context (e.g. after a CUDA "unknown error") so
    that callers in finally-blocks do not raise a second exception that would
    mask the original one and crash the benchmark loop.
    """
    mem_before = 0
    mem_after = 0

    gc.collect()

    if torch.cuda.is_available():
        try:
            mem_before = torch.cuda.memory_allocated() / (1024 ** 2)
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            mem_after = torch.cuda.memory_allocated() / (1024 ** 2)
            logger.debug(f"VRAM Flush - Before: {mem_before:.2f} MB, After: {mem_after:.2f} MB")
        except RuntimeError as exc:
            logger.warning(f"flush_vram: CUDA call failed (device may be in a broken state): {exc}")

    return mem_before, mem_after
