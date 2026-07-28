"""
evaluation/cider_scorer.py — Custom CIDEr scorer with precomputed corpus IDF.
"""
import math
import numpy as np
from collections import defaultdict
from pycocoevalcap.cider.cider_scorer import CiderScorer


class CustomCiderScorer(CiderScorer):
    """CiderScorer subclass that bypasses the test-size assertion and supports
    precomputed corpus IDF weights."""

    def __init__(self, df=None, n=4, sigma=6.0):
        super().__init__(n=n, sigma=sigma)
        if df is not None:
            self.document_frequency = df

    # ------------------------------------------------------------------
    # Private computation helpers
    # ------------------------------------------------------------------

    def _counts2vec(self, cnts):
        vec = [defaultdict(float) for _ in range(self.n)]
        norm = [0.0] * self.n
        length = 0
        for ngram, term_freq in cnts.items():
            n = len(ngram) - 1
            if not (0 <= n < self.n):
                continue
            vec[n][ngram] = float(term_freq) * self.document_frequency.get(ngram, 0.0)
            norm[n] += vec[n][ngram] ** 2
            if n == 1:
                length += term_freq
        norm = [np.sqrt(v) for v in norm]
        return vec, norm, length

    def _sim(self, vec_hyp, vec_ref, norm_hyp, norm_ref, length_hyp, length_ref):
        delta = float(length_hyp - length_ref)
        val = np.zeros(self.n)
        for n in range(self.n):
            for ngram, count in vec_hyp[n].items():
                val[n] += min(count, vec_ref[n][ngram]) * vec_ref[n][ngram]
            if norm_hyp[n] != 0 and norm_ref[n] != 0:
                val[n] /= norm_hyp[n] * norm_ref[n]
            assert not math.isnan(val[n])
            val[n] *= np.e ** (-(delta ** 2) / (2 * self.sigma ** 2))
        return val

    # ------------------------------------------------------------------
    # Public interface (override)
    # ------------------------------------------------------------------

    def compute_cider(self):
        scores = []
        for test, refs in zip(self.ctest, self.crefs):
            vec, norm, length = self._counts2vec(test)
            score = np.zeros(self.n)
            for ref in refs:
                vec_ref, norm_ref, length_ref = self._counts2vec(ref)
                score += self._sim(vec, vec_ref, norm, norm_ref, length, length_ref)
            score_avg = np.mean(score) / len(refs) * 10.0
            scores.append(score_avg)
        return scores

    def compute_score(self):
        score = self.compute_cider()
        return np.mean(np.array(score)), np.array(score)
