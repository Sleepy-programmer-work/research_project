"""
experiments/benchmark_samplers.py — Sampler and aggregator factories.

Centralises the variant registries (TASS_VARIANTS) and the
factory functions (get_samplers, get_aggregators) that were previously
inlined in run_benchmark.main().
"""
import logging
from config.settings import settings
from samplers import FPS1Sampler, FPS2Sampler, RandomSampler, PHashSampler, TASSSampler
from aggregation import RawAggregator, CentroidAggregator, TemporalAggregator

logger = logging.getLogger("benchmark")

# ---------------------------------------------------------------------------
# Variant registries
# Each key is the unique sampler name used as cache key, CSV value, and JSON
# filename. Changing a key invalidates all cached captions for that variant.
# ---------------------------------------------------------------------------

TASS_VARIANTS: dict[str, dict] = {
    "tass_fixed":    {"mode": "fixed",    "threshold": 0.90, "min_distance": 0.10},
    "tass_adaptive": {"mode": "adaptive", "threshold": 0.90, "min_distance": 0.10},
}


def get_samplers(cfg=None) -> dict:
    """Build the sampler registry from config."""
    pipeline_cfg = cfg.pipeline if cfg is not None else settings.pipeline
    active = pipeline_cfg.get("samplers", [])

    samplers = {
        "fps1":   FPS1Sampler(),
        "fps2":   FPS2Sampler(),
        "random": RandomSampler(seed=settings.experiment.get("seed", 42)),
    }
    _register_phash(samplers, active)
    _register_tass_variants(samplers, active)
    return samplers


def _register_phash(samplers: dict, active: list) -> None:
    if "phash" in active:
        samplers["phash"] = PHashSampler()
        logger.info("Registered pHash sampler.")


def _register_tass_variants(samplers: dict, active: list) -> None:
    registered = []
    for name, kwargs in TASS_VARIANTS.items():
        if name in active:
            samplers[name] = TASSSampler(**kwargs)
            registered.append(name)
    if registered:
        logger.info(
            f"Registered {len(registered)} TASS variant(s): {registered}. "
            f"MobileCLIP-S1 (~85 MB CPU RAM) loads on first sample call."
        )


def get_aggregators() -> dict:
    return {
        "raw":      RawAggregator(),
        "centroid": CentroidAggregator(),
        "temporal": TemporalAggregator(),
    }
