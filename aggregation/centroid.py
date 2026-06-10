import string
import logging
import nltk
from typing import List
from .base import BaseAggregator

logger = logging.getLogger(__name__)


def _ensure_nltk_resources(*resource_names: str) -> None:
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


# Ensure punkt tokenizer is available at import time so failures surface
# immediately (at module load) rather than mid-benchmark during aggregate().
_ensure_nltk_resources("punkt", "punkt_tab")


class CentroidAggregator(BaseAggregator):
    def get_name(self) -> str:
        return "centroid"

    def _normalize(self, text: str) -> set:
        text = text.lower()
        text = text.translate(str.maketrans('', '', string.punctuation))
        tokens = nltk.word_tokenize(text)
        return set(tokens)

    def _jaccard(self, set_a: set, set_b: set) -> float:
        intersection = len(set_a.intersection(set_b))
        union = len(set_a.union(set_b))
        return intersection / union if union > 0 else 0.0

    def aggregate(self, captions: List[str]) -> str:
        if not captions:
            return ""
        if len(captions) == 1:
            return captions[0]

        token_sets = [self._normalize(c) for c in captions]
        n = len(captions)
        scores = [0.0] * n

        for i in range(n):
            for j in range(n):
                if i != j:
                    scores[i] += self._jaccard(token_sets[i], token_sets[j])

        best_idx = scores.index(max(scores))
        return captions[best_idx]
