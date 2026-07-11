"""
utils/memory.py — Shared RAM budget guard for model loaders.

Centralises the psutil + gc.collect() pattern that was copy-pasted verbatim
in VLMLoader.load() and LLMLoader.warm_model().
"""
import gc
import logging
import psutil

from config.settings import MAX_RAM_BUDGET_GB

logger = logging.getLogger(__name__)


def check_ram_budget(label: str) -> None:
    """Warn and attempt GC if RAM usage is at or above MAX_RAM_BUDGET_GB.

    Args:
        label: Caller name shown in log messages (e.g. 'VLM Loader').
    """
    current_gb = psutil.virtual_memory().used / (1024 ** 3)
    if current_gb < MAX_RAM_BUDGET_GB:
        return

    logger.warning(
        f"{label}: RAM usage ({current_gb:.2f} GB) is approaching "
        f"MAX_RAM_BUDGET_GB ({MAX_RAM_BUDGET_GB} GB). Running gc.collect()."
    )
    gc.collect()

    current_gb = psutil.virtual_memory().used / (1024 ** 3)
    if current_gb >= MAX_RAM_BUDGET_GB:
        logger.warning(
            f"{label}: RAM usage remains high ({current_gb:.2f} GB) "
            f"after gc.collect(). Proceeding with caution."
        )
