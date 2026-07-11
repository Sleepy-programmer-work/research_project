"""
Public API for frame sampling strategies.

DO NOT export DSISSampler — it is an unimplemented stub that returns [].
Accidental use would silently produce zero frames, crashing metrics computation.
"""

from .base_sampler import BaseSampler
from .fps1 import FPS1Sampler
from .fps2 import FPS2Sampler
from .random_sampler import RandomSampler
from .ssim_result import SSIMSamplerResult
from .ssim import SSIMSampler
# DSISSampler is intentionally NOT imported — it is a Phase 2 stub (returns []).
# from .dsis import DSISSampler
from .tass import TASSSampler

__all__ = [
    "BaseSampler",
    "SSIMSamplerResult",
    "FPS1Sampler",
    "FPS2Sampler",
    "RandomSampler",
    "SSIMSampler",
    "TASSSampler",
]
