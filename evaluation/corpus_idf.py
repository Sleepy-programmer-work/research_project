"""
evaluation/corpus_idf.py — Full-corpus IDF builder for CIDEr scoring.

Builds and caches a document-frequency table from a large reference corpus so
that CIDEr IDF weights are meaningful even when evaluating on a small subset
(10–100 videos).  Computing IDF on only the evaluation subset causes degenerate
weights; using the full training corpus produces stable, semantically correct
scores.

Active dataset:  MSR-VTT  (train_val_videodatainfo.json via Kaggle)
Decoupled:       MSVD     (HuggingFace-based builder kept intact)

Usage:
    from evaluation.corpus_idf import load_corpus_idf
    corpus_idf = load_corpus_idf(cache_dir=Path("results/cache"))
    # corpus_idf is a dict with keys: idf, n_docs, n_captions, dataset
"""

import json
import logging
import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# MSR-VTT constants
# ---------------------------------------------------------------------------
_MSRVTT_CACHE_FILENAME  = "msrvtt_full_corpus_idf.pkl"
_MSRVTT_LOCAL_DIR       = Path("/home/tushar/msrvtt")
_MSRVTT_JSON_FILENAME   = "MSRVTT_data.json"

# ---------------------------------------------------------------------------
# MSVD constants (preserved)
# ---------------------------------------------------------------------------
_MSVD_CACHE_FILENAME = "msvd_full_corpus_idf.pkl"
_MSVD_DATASET_NAME   = "friedrichor/MSVD"


# ===========================================================================
# Shared n-gram utilities
# ===========================================================================

def _tokenize(caption: str) -> list[str]:
    """Simple whitespace tokenizer matching pycocoevalcap's internal logic."""
    return caption.lower().strip().split()


def _build_ngrams(tokens: list[str], n: int) -> list[tuple]:
    return [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]


def _compute_idf(all_captions: dict[str, list[str]], ngram_range: int = 4) -> dict:
    """Compute IDF from a {video_id: [caption, ...]} mapping.

    Returns a corpus_data dict with keys: idf, n_docs, n_captions, ngram_range.
    """
    n_videos   = len(all_captions)
    n_captions = sum(len(v) for v in all_captions.values())
    logger.info(f"Corpus: {n_videos} videos, {n_captions} total captions.")

    if n_videos < 100:
        logger.warning(
            f"Only {n_videos} videos in corpus — IDF weights may be weak. "
            "Consider using a larger split."
        )

    doc_freq: dict[tuple, int] = defaultdict(int)
    for captions in all_captions.values():
        ngrams_in_doc: set[tuple] = set()
        for caption in captions:
            tokens = _tokenize(caption)
            for n in range(1, ngram_range + 1):
                for ng in _build_ngrams(tokens, n):
                    ngrams_in_doc.add(ng)
        for ng in ngrams_in_doc:
            doc_freq[ng] += 1

    # IDF formula: log((N + 1) / df)  — matches CiderScorer internals
    N = n_videos
    idf: dict[tuple, float] = {ng: np.log((N + 1.0) / df) for ng, df in doc_freq.items()}

    return {
        "idf":        idf,
        "n_docs":     N,
        "n_captions": n_captions,
        "ngram_range": ngram_range,
    }


# ===========================================================================
# MSR-VTT IDF builder  (active)
# ===========================================================================

def build_msrvtt_idf(
    cache_dir: Path,
    ngram_range: int = 4,
    force_rebuild: bool = False,
) -> Path:
    """Build the MSR-VTT corpus IDF from train_val_videodatainfo.json.

    Downloads the Kaggle dataset on first run (cached by kagglehub).
    Reads every caption sentence from the JSON (~200 k sentences), groups them
    by video_id, computes document frequencies for 1-grams through 4-grams,
    and saves the result as `msrvtt_full_corpus_idf.pkl`.

    Returns the path to the cached .pkl file.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / _MSRVTT_CACHE_FILENAME

    if cache_path.exists() and not force_rebuild:
        logger.info(f"Loading cached MSR-VTT corpus IDF from {cache_path}")
        return cache_path

    logger.info("Building MSR-VTT full corpus IDF — this runs once and is cached.")

    json_path = _MSRVTT_LOCAL_DIR / _MSRVTT_JSON_FILENAME
    if not json_path.exists():
        raise FileNotFoundError(
            f"{_MSRVTT_JSON_FILENAME} not found at {json_path}."
        )

    logger.info(f"Parsing {json_path} ...")
    with open(json_path, "r", encoding="utf-8") as fh:
        raw = json.load(fh)

    # Group all caption sentences by video_id
    all_captions: dict[str, list[str]] = defaultdict(list)
    for sent in raw.get("sentences", []):
        vid = str(sent.get("video_id", "")).strip()
        cap = str(sent.get("caption",  "")).strip()
        if vid and cap:
            all_captions[vid].append(cap)

    corpus_data = _compute_idf(all_captions, ngram_range)
    corpus_data["dataset"] = "local_msrvtt"

    with open(cache_path, "wb") as f:
        pickle.dump(corpus_data, f, protocol=pickle.HIGHEST_PROTOCOL)

    logger.info(
        f"MSR-VTT corpus IDF built: {len(corpus_data['idf']):,} unique n-grams "
        f"across {corpus_data['n_docs']} videos. Cached to {cache_path}"
    )
    return cache_path


# ===========================================================================
# MSVD IDF builder  (decoupled — kept intact)
# ===========================================================================

def build_msvd_idf(
    cache_dir: Path,
    ngram_range: int = 4,
    force_rebuild: bool = False,
) -> Path:
    """[DECOUPLED] Build the MSVD corpus IDF from HuggingFace.

    Kept intact for future re-activation.  Not called when MSR-VTT is active.
    Returns the path to the cached .pkl file.
    """
    from datasets import load_dataset  # lazy import — not needed for MSR-VTT

    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / _MSVD_CACHE_FILENAME

    if cache_path.exists() and not force_rebuild:
        logger.info(f"Loading cached MSVD corpus IDF from {cache_path}")
        return cache_path

    logger.info("Building full MSVD corpus IDF — this runs once and is cached.")
    ds = load_dataset(_MSVD_DATASET_NAME)

    all_captions: dict[str, list[str]] = defaultdict(list)
    for split_name in ds.keys():
        for row in ds[split_name]:
            vid_id    = str(row.get("video_id", row.get("id", "")))
            cap_val   = row.get("caption",  row.get("text", ""))
            cap_list  = cap_val if isinstance(cap_val, list) else [cap_val]
            for c in cap_list:
                c_clean = str(c).strip()
                if vid_id and c_clean:
                    all_captions[vid_id].append(c_clean)

    corpus_data = _compute_idf(all_captions, ngram_range)
    corpus_data["dataset"] = _MSVD_DATASET_NAME

    with open(cache_path, "wb") as f:
        pickle.dump(corpus_data, f, protocol=pickle.HIGHEST_PROTOCOL)

    logger.info(
        f"MSVD corpus IDF built: {len(corpus_data['idf']):,} unique n-grams "
        f"across {corpus_data['n_docs']} videos. Cached to {cache_path}"
    )
    return cache_path


# ===========================================================================
# Public entry-point
# ===========================================================================

def load_corpus_idf(
    cache_dir: Path,
    force_rebuild: bool = False,
) -> dict:
    """Load (and build if needed) the MSR-VTT corpus IDF.

    Returns a dict with keys: idf, n_docs, n_captions, dataset, ngram_range.
    The IDF dict maps n-gram tuples -> float IDF weights compatible with
    pycocoevalcap's CiderScorer.
    """
    cache_path = build_msrvtt_idf(cache_dir, force_rebuild=force_rebuild)
    with open(cache_path, "rb") as f:
        return pickle.load(f)
