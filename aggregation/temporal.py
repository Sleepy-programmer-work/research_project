import string
import nltk
from typing import List
from .base import BaseAggregator

nltk.data.path.append("./venv/nltk_data")

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', download_dir='./venv/nltk_data')

try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab', download_dir='./venv/nltk_data')

class TemporalAggregator(BaseAggregator):
    def get_name(self) -> str:
        return "temporal"

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
            
        unique_captions = []
        last_set = None
        
        for caption in captions:
            current_set = self._normalize(caption)
            if last_set is not None:
                sim = self._jaccard(current_set, last_set)
                if sim > 0.85:
                    continue
            unique_captions.append(caption)
            last_set = current_set
            
        return " ".join(unique_captions)
