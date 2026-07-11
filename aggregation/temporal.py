import logging
from typing import List
from .base import BaseAggregator
from .nlp_utils import normalize_to_token_set, jaccard_similarity

logger = logging.getLogger(__name__)

_DEDUP_THRESHOLD = 0.85


class TemporalAggregator(BaseAggregator):
    """Join captions in order, deduplicating adjacent near-identical ones."""

    def get_name(self) -> str:
        return "temporal"

    def aggregate(self, captions: List[str]) -> str:
        if not captions:
            return ""

        unique: List[str] = []
        last_set = None

        for caption in captions:
            current_set = normalize_to_token_set(caption)
            if last_set is not None and jaccard_similarity(current_set, last_set) > _DEDUP_THRESHOLD:
                continue
            unique.append(caption)
            last_set = current_set

        return " ".join(unique)
