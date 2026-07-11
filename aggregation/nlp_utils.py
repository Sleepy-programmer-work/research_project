"""
aggregation/nlp_utils.py — Shared NLP helpers for text-based aggregators.

Centralises the three utilities that were previously copy-pasted verbatim
between CentroidAggregator and TemporalAggregator:
  - NLTK resource bootstrapping
  - Token-set normalisation
  - Jaccard similarity between token sets

Import from here; never duplicate.
"""
import string
import logging
import nltk
from typing import Set

logger = logging.getLogger(__name__)


def ensure_nltk_resources(*resource_names: str) -> None:
    """Robustly ensure required NLTK tokenizer resources are available.

    Strategy:
      1. Try nltk.data.find() — respects all standard NLTK search paths
         (~/.nltk_data, ~/nltk_data, /usr/share/nltk_data, etc.) and any
         paths the user has appended to nltk.data.path.
      2. If not found, download to NLTK's default user data directory.
         NEVER downloads to a hardcoded ./venv/nltk_data path, which does not
         exist on non-venv systems (CI, Docker, other devs' machines).
      3. Raises RuntimeError with a clear message if download fails, so the
         benchmark halts with a useful error rather than a cryptic LookupError
         mid-run.
    """
    for resource in resource_names:
        try:
            nltk.data.find(f"tokenizers/{resource}")
        except LookupError:
            logger.info(
                f"NLTK resource '{resource}' not found in search path. "
                f"Downloading to default NLTK data directory..."
            )
            try:
                nltk.download(resource, quiet=True)
            except Exception as exc:
                raise RuntimeError(
                    f"Failed to download required NLTK resource '{resource}': {exc}. "
                    f"Run: python -c \"import nltk; nltk.download('{resource}')\" "
                    f"to install it manually before running the benchmark."
                ) from exc


def normalize_to_token_set(text: str) -> Set[str]:
    """Lowercase, strip punctuation, tokenize, and return a token set."""
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return set(nltk.word_tokenize(text))


def jaccard_similarity(set_a: Set[str], set_b: Set[str]) -> float:
    """Return Jaccard similarity (intersection / union) for two token sets."""
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    return intersection / union if union > 0 else 0.0


# Ensure punkt tokenizer is available at import time so failures surface
# immediately (at module load) rather than mid-benchmark during aggregate().
ensure_nltk_resources("punkt", "punkt_tab")
