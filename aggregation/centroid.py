import logging
from typing import List
from .base import BaseAggregator
from .nlp_utils import normalize_to_token_set, jaccard_similarity

logger = logging.getLogger(__name__)


class CentroidAggregator(BaseAggregator):
    """Select the caption most similar (by Jaccard) to all others."""

    def get_name(self) -> str:
        return "centroid"

    def aggregate(self, captions: List[str]) -> str:
        if not captions:
            return ""
        if len(captions) == 1:
            return captions[0]

        token_sets = [normalize_to_token_set(c) for c in captions]
        n = len(captions)
        scores = [
            sum(
                jaccard_similarity(token_sets[i], token_sets[j])
                for j in range(n) if i != j
            )
            for i in range(n)
        ]
        return captions[scores.index(max(scores))]
