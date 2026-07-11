"""
experiments/benchmark_data.py — Dataset loading, video acquisition, and reference captions.

Responsibilities:
  - [ACTIVE]  MSR-VTT: Download via kagglehub (vishnutheepb/msrvtt), copy selected
              .mp4 files to the project cache, return video metadata + reference captions.
  - [DECOUPLED] MSVD: Original Hugging Face–based loader kept intact for future
              re-activation — no code has been deleted.

Public API:
  ensure_dataset_videos(num_videos, seed, cache_dir)
      -> tuple[list[dict], dict[str, list[str]]]   # (video_rows, ref_caps)

  build_reference_captions(dataset_name)   # retained for backward-compat (MSVD only)
"""
import json
import logging
import os
import random
import shutil
import zipfile
from collections import defaultdict
from pathlib import Path

from tqdm import tqdm

logger = logging.getLogger("benchmark")

# ---------------------------------------------------------------------------
# MSR-VTT constants
# ---------------------------------------------------------------------------
_MSRVTT_LOCAL_DIR      = Path("/home/tushar/msrvtt")
_MSRVTT_JSON_FILENAME  = "MSRVTT_data.json"
_MSRVTT_ZIP_FILENAME   = "MSRVTT_Videos.zip"

# ---------------------------------------------------------------------------
# MSVD constants (preserved — not deleted)
# ---------------------------------------------------------------------------
_MSVD_REPO     = "friedrichor/MSVD"
_ID_FIELDS     = ("video_id", "id", "vid_id")
_CAPTION_FIELDS = ("caption", "text", "description", "sentence")


# ===========================================================================
# MSR-VTT — active dataset loader
# ===========================================================================

def ensure_dataset_videos(
    num_videos: int,
    seed: int,
    cache_dir: Path,
) -> tuple[list[dict], dict[str, list[str]]]:
    """Load MSR-VTT from local storage, extract selected videos to cache, return metadata.

    Steps:
      1. Parse MSRVTT_data.json to get video metadata + captions.
      2. Deduplicate by video_id, shuffle deterministically with `seed`.
      3. Slice to exactly `num_videos` (capped at total available).
      4. Extract the selected .mp4 files from MSRVTT_Videos.zip into `cache_dir`.
      5. Return a list of row-dicts and a ref-captions dict.

    Args:
        num_videos: How many videos the benchmark should process (CLI --videos).
        seed:       Deterministic shuffle seed (from benchmark.yaml).
        cache_dir:  Project-local cache directory (videos are copied here).

    Returns:
        video_rows: list of dicts with keys video_id, video_path, caption.
        ref_caps:   dict mapping video_id -> list[str] of reference captions.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)

    json_path = _MSRVTT_LOCAL_DIR / _MSRVTT_JSON_FILENAME
    if not json_path.exists():
        raise FileNotFoundError(f"Could not find local dataset JSON at {json_path}")

    logger.info(f"Parsing {json_path} ...")
    with open(json_path, "r", encoding="utf-8") as fh:
        raw = json.load(fh)

    # Build per-video caption index from the 'sentences' array
    # Each entry: {"video_id": "video0", "caption": "...", "sen_id": 0, ...}
    captions_by_vid: dict[str, list[str]] = defaultdict(list)
    for sent in raw.get("sentences", []):
        vid = str(sent.get("video_id", "")).strip()
        cap = str(sent.get("caption", "")).strip()
        if vid and cap:
            captions_by_vid[vid].append(cap)

    # Build ordered list of unique video metadata from the 'videos' array
    seen: set[str] = set()
    unique_videos: list[dict] = []
    for v in raw.get("videos", []):
        vid = str(v.get("video_id", "")).strip()
        if vid and vid not in seen:
            seen.add(vid)
            unique_videos.append(v)

    logger.info(
        f"MSR-VTT metadata: {len(unique_videos)} unique videos, "
        f"{sum(len(c) for c in captions_by_vid.values())} total captions."
    )

    # --- 3 & 4. Deterministic shuffle + slice -------------------------------
    rng = random.Random(seed)
    rng.shuffle(unique_videos)
    selected = unique_videos[: min(num_videos, len(unique_videos))]
    logger.info(f"Selected {len(selected)} videos (num_videos={num_videos}, seed={seed}).")

    # --- 5. Extract .mp4 files to project cache -----------------------------
    zip_path = _MSRVTT_LOCAL_DIR / _MSRVTT_ZIP_FILENAME
    if not zip_path.exists():
        raise FileNotFoundError(f"Could not find local dataset ZIP at {zip_path}")

    video_rows: list[dict] = []
    ref_caps: dict[str, list[str]] = {}

    with zipfile.ZipFile(zip_path, "r") as zf:
        for v in tqdm(selected, desc="Extracting MSR-VTT videos to cache"):
            vid_id = str(v["video_id"])
            dst = cache_dir / f"{vid_id}.mp4"

            if not dst.exists():
                zip_member = f"video/{vid_id}.mp4"
                try:
                    with zf.open(zip_member) as src_file, open(dst, "wb") as dst_file:
                        shutil.copyfileobj(src_file, dst_file)
                except KeyError:
                    logger.warning(f"Video file {zip_member} not found in zip — skipping.")
                    continue

            video_rows.append({"video_id": vid_id, "video_path": str(dst)})
            ref_caps[vid_id] = captions_by_vid.get(vid_id, [])

    _log_reference_stats(ref_caps)
    logger.info(f"Videos ready in {cache_dir}: {len(video_rows)} usable.")
    return video_rows, ref_caps


# ===========================================================================
# MSVD — preserved, decoupled (not active in benchmark.yaml)
# ===========================================================================

def ensure_msvd_videos(cache_dir: Path) -> None:
    """[DECOUPLED] Download MSVD_Videos.zip from Hugging Face if not already cached.

    This function is NOT called by the main benchmark runner when MSR-VTT is
    the active dataset.  It is kept intact so the pipeline can be switched back
    to MSVD by restoring the old imports in run_benchmark.py.
    """
    from datasets import load_dataset  # lazy import — not needed for MSR-VTT path
    from huggingface_hub import hf_hub_download

    cache_dir.mkdir(parents=True, exist_ok=True)
    if len(list(cache_dir.glob("*.avi"))) >= 10:
        logger.info(f"Found {len(list(cache_dir.glob('*.avi')))} local AVI files. Skipping MSVD download.")
        return

    logger.info("MSVD video files not found. Downloading from Hugging Face (1.8 GB)...")
    try:
        zip_path = hf_hub_download(repo_id=_MSVD_REPO, filename="MSVD_Videos.zip", repo_type="dataset")
        _extract_zip_flat(zip_path, cache_dir, suffix=".avi")
        logger.info("Successfully extracted MSVD videos to cache.")
    except Exception as exc:
        logger.error(f"Failed to download/extract MSVD videos: {exc}")


def _extract_zip_flat(zip_path: str, dest: Path, suffix: str) -> None:
    """[MSVD] Extract matching files from a zip archive into dest (no subdirectories)."""
    with zipfile.ZipFile(zip_path, "r") as zf:
        for member in tqdm(zf.infolist(), desc="Extracting videos"):
            if member.is_dir():
                continue
            filename = os.path.basename(member.filename)
            if not filename.endswith(suffix):
                continue
            target = dest / filename
            if target.exists():
                continue
            with zf.open(member) as src, open(target, "wb") as dst:
                shutil.copyfileobj(src, dst)


def build_reference_captions(dataset_name: str = _MSVD_REPO) -> dict[str, list[str]]:
    """[DECOUPLED / MSVD] Load all MSVD reference captions grouped by video_id.

    Kept for backward-compatibility.  MSR-VTT reference captions are returned
    directly by ensure_dataset_videos() and do not use this function.
    """
    from datasets import load_dataset  # lazy import

    ds = load_dataset(dataset_name)
    first_split = list(ds.keys())[0]
    first_row = ds[first_split][0]
    id_field, cap_field = _detect_fields(first_row)
    logger.info(f"Using fields: id='{id_field}', caption='{cap_field}'")

    refs: dict[str, list[str]] = defaultdict(list)
    for split_name in ds.keys():
        for row in ds[split_name]:
            _append_captions(refs, row, id_field, cap_field)

    refs = dict(refs)
    _log_reference_stats(refs)
    return refs


def _detect_fields(row: dict) -> tuple[str, str]:
    """[MSVD] Detect video_id and caption field names from a sample dataset row."""
    id_field  = next((f for f in row if f in _ID_FIELDS), None)
    cap_field = next((f for f in row if f in _CAPTION_FIELDS), None)
    if not id_field or not cap_field:
        raise ValueError(
            f"Cannot find video_id or caption fields in dataset. "
            f"Found: {list(row.keys())}. Update _detect_fields() to match."
        )
    return id_field, cap_field


def _append_captions(refs: dict, row: dict, id_field: str, cap_field: str) -> None:
    """[MSVD] Append captions from one dataset row into the refs dict."""
    vid = str(row[id_field])
    cap_val = row[cap_field]
    captions = cap_val if isinstance(cap_val, list) else [cap_val]
    for c in captions:
        c_clean = str(c).strip()
        if vid and c_clean:
            refs[vid].append(c_clean)


def _log_reference_stats(refs: dict) -> None:
    """Log a summary of reference-caption coverage."""
    n_videos = len(refs)
    n_caps   = sum(len(v) for v in refs.values())
    if n_videos:
        logger.info(
            f"Reference captions: {n_videos} videos, {n_caps} total, "
            f"{n_caps / n_videos:.1f} avg/video."
        )
    else:
        logger.info("Reference captions: 0 videos.")
    sparse = [v for v, caps in refs.items() if len(caps) < 2]
    if sparse:
        logger.warning(f"{len(sparse)} videos have fewer than 2 references — CIDEr less reliable.")
