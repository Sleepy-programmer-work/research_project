"""
Builds and caches a full-corpus IDF table from all MSVD reference captions.
This is loaded by metrics.py to give CIDEr meaningful IDF weights even when
evaluating on a small subset (10–100 videos).

Usage:
    from evaluation.corpus_idf import load_corpus_idf
    df_file = load_corpus_idf(cache_dir=Path("results/cache"))
    # pass df_file path to CiderScorer(df=df_file)
"""

import pickle
import logging
from pathlib import Path
from collections import defaultdict

import numpy as np
from datasets import load_dataset

logger = logging.getLogger(__name__)

CACHE_FILENAME = "msvd_full_corpus_idf.pkl"
DATASET_NAME   = "friedrichor/MSVD"


def _tokenize(caption: str) -> list[str]:
    """Simple whitespace tokenizer matching pycocoevalcap's internal logic."""
    return caption.lower().strip().split()


def _build_ngrams(tokens: list[str], n: int) -> list[tuple]:
    return [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]


def build_msvd_idf(
    cache_dir: Path,
    ngram_range: int = 4,
    force_rebuild: bool = False,
) -> Path:
    """
    Load all MSVD reference captions from HuggingFace, compute document
    frequency for every n-gram (n=1..ngram_range), and save as a pickle
    dict compatible with pycocoevalcap's CiderScorer df format.

    Returns the path to the cached .pkl file.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / CACHE_FILENAME

    if cache_path.exists() and not force_rebuild:
        logger.info(f"Loading cached MSVD corpus IDF from {cache_path}")
        return cache_path

    logger.info("Building full MSVD corpus IDF — this runs once and is cached.")
    logger.info(f"Loading dataset: {DATASET_NAME}")

    ds = load_dataset(DATASET_NAME)

    # Collect all captions across all splits, grouped by video_id
    # The dataset may have 'train', 'val', 'test' splits
    all_captions: dict[str, list[str]] = defaultdict(list)

    for split_name in ds.keys():
        split = ds[split_name]
        for row in split:
            vid_id = str(row.get("video_id", row.get("id", "")))
            caption_val = row.get("caption", row.get("text", ""))
            
            if isinstance(caption_val, list):
                for c in caption_val:
                    c_clean = str(c).strip()
                    if vid_id and c_clean:
                        all_captions[vid_id].append(c_clean)
            else:
                c_clean = str(caption_val).strip()
                if vid_id and c_clean:
                    all_captions[vid_id].append(c_clean)

    n_videos   = len(all_captions)
    n_captions = sum(len(v) for v in all_captions.values())
    logger.info(f"Corpus: {n_videos} videos, {n_captions} total captions.")

    if n_videos < 100:
        logger.warning(
            f"Only {n_videos} videos found in corpus — IDF may still be weak. "
            f"Check dataset field names if this seems wrong."
        )

    # Build document frequency table
    # doc_freq[ngram] = number of videos that contain this ngram at least once
    doc_freq: dict[tuple, int] = defaultdict(int)

    for vid_id, captions in all_captions.items():
        ngrams_in_doc: set[tuple] = set()
        for caption in captions:
            tokens = _tokenize(caption)
            for n in range(1, ngram_range + 1):
                for ng in _build_ngrams(tokens, n):
                    ngrams_in_doc.add(ng)
        for ng in ngrams_in_doc:
            doc_freq[ng] += 1

    # Compute IDF: log((N + 1) / df) — matching CiderScorer formula
    N = n_videos
    idf: dict[tuple, float] = {}
    for ng, df in doc_freq.items():
        idf[ng] = np.log((N + 1.0) / df)

    # Also store metadata for diagnostics
    corpus_data = {
        "idf":       idf,
        "n_docs":    N,
        "n_captions": n_captions,
        "dataset":   DATASET_NAME,
        "ngram_range": ngram_range,
    }

    with open(cache_path, "wb") as f:
        pickle.dump(corpus_data, f, protocol=pickle.HIGHEST_PROTOCOL)

    logger.info(
        f"Corpus IDF built: {len(idf):,} unique n-grams across "
        f"{N} videos. Cached to {cache_path}"
    )
    return cache_path


def load_corpus_idf(cache_dir: Path, force_rebuild: bool = False) -> dict:
    """
    Return the IDF dict. Builds and caches on first call.
    Returns the corpus_data dict with keys: idf, n_docs, n_captions.
    """
    cache_path = build_msvd_idf(cache_dir, force_rebuild=force_rebuild)
    with open(cache_path, "rb") as f:
        return pickle.load(f)
