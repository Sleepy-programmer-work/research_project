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
