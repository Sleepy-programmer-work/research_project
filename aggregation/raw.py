from typing import List
from .base import BaseAggregator

class RawAggregator(BaseAggregator):
    def get_name(self) -> str:
        return "raw"

    def aggregate(self, captions: List[str]) -> str:
        return " ".join(captions)
