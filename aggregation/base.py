from abc import ABC, abstractmethod
from typing import List

class BaseAggregator(ABC):
    @abstractmethod
    def aggregate(self, captions: List[str]) -> str:
        """Combine multiple frame captions into a single text representation."""
        pass

    @abstractmethod
    def get_name(self) -> str:
        """Return a short identifier string."""
        pass
