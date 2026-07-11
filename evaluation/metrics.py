import logging
import numpy as np
from typing import Optional

from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.meteor.meteor import Meteor

from evaluation.cider_scorer import CustomCiderScorer

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Diagnostic helpers
# ---------------------------------------------------------------------------

def _diagnose_cider_inputs(gts: dict, res: dict) -> None:
    """Log a structured diagnostic of gts/res at DEBUG level (safe in production)."""
    logger.debug("=== CIDEr input diagnostic ===")
    logger.debug(f"  gts keys (first 3): {list(gts.keys())[:3]}")
    logger.debug(f"  res keys (first 3): {list(res.keys())[:3]}")
    for vid_id in list(gts.keys())[:2]:
        refs, hyp = gts[vid_id], res.get(vid_id, [])
        logger.debug(f"  video '{vid_id}': refs[0]={refs[0] if refs else 'EMPTY'}, "
                     f"hyp[0]={hyp[0] if hyp else 'EMPTY'}")
        if refs and isinstance(refs[0], str):
            logger.error("  CAUSE A DETECTED: refs are plain strings, not dicts.")
        if refs and isinstance(refs[0], dict):
            empty = [r for r in refs if not r.get("caption", "").strip()]
            if empty:
                logger.error(f"  CAUSE C DETECTED: {len(empty)} empty caption strings.")
    logger.debug(f"  total gts/res videos: {len(gts)}/{len(res)}")
    if len(gts) < 50:
        logger.warning(f"  CAUSE B WARNING: only {len(gts)} videos — CIDEr IDF unreliable.")


# ---------------------------------------------------------------------------
# Format builders
# ---------------------------------------------------------------------------

def build_gts_res(
    video_ids: list[str],
    reference_captions: dict[str, list[str]],
    generated_captions: dict[str, str],
    logger: logging.Logger,
) -> tuple[dict, dict]:
    """Build gts and res dicts in the format required by pycocoevalcap."""
    gts: dict[str, list[dict]] = {}
    res: dict[str, list[dict]] = {}
    skipped = 0

    for vid_id in video_ids:
        key = str(vid_id)
        clean_refs = [
            " ".join(str(r).strip().split())
            for r in reference_captions.get(vid_id, [])
            if r and str(r).strip()
        ]
        if not clean_refs:
            logger.warning(f"video '{vid_id}' has no valid reference captions — skipping.")
            skipped += 1
            continue

        gts[key] = [{"image_id": key, "caption": ref} for ref in clean_refs]
        gen = " ".join(generated_captions.get(vid_id, "").strip().split()) or "no caption generated"
        if not gen.strip():
            logger.warning(f"video '{vid_id}' has empty generated caption — using placeholder.")
        res[key] = [{"image_id": key, "caption": gen}]

    if skipped:
        logger.warning(f"Skipped {skipped}/{len(video_ids)} videos due to missing references.")
    return gts, res


# ---------------------------------------------------------------------------
# Scorers
# ---------------------------------------------------------------------------

def compute_cider_with_corpus_idf(
    gts: dict[str, list[dict]],
    res: dict[str, list[dict]],
    corpus_idf: dict,
) -> dict[str, float]:
    """Compute CIDEr using full MSVD corpus IDF (methodologically correct for subsets)."""
    scorer = CustomCiderScorer(df=corpus_idf["idf"], n=4, sigma=6.0)
    for vid_id in sorted(gts.keys()):
        if vid_id not in res:
            logger.warning(f"No hypothesis for video '{vid_id}' — skipping CIDEr.")
            continue
        scorer += (res[vid_id][0]["caption"], [r["caption"] for r in gts[vid_id]])

    if not scorer.crefs:
        logger.error("CiderScorer has no references — returning 0.")
        return {vid_id: 0.0 for vid_id in gts}

    _, per_image = scorer.compute_score()
    return {
        vid_id: float(per_image[i])
        for i, vid_id in enumerate(sorted(gts.keys()))
        if vid_id in res
    }


def _score_string_dicts(gts: dict, res: dict) -> tuple[dict, dict]:
    """Convert pycocoevalcap caption dicts to plain string lists for BLEU/ROUGE/METEOR."""
    gts_s = {v: [r["caption"] for r in gts[v]] for v in gts}
    res_s = {v: [r["caption"] for r in res[v]] for v in res}
    return gts_s, res_s


def compute_all_metrics(
    gts: dict[str, list[dict]],
    res: dict[str, list[dict]],
    corpus_idf: dict,
) -> dict[str, dict[str, float]]:
    """Compute CIDEr, BLEU-1, BLEU-4, ROUGE-L, METEOR for every video."""
    _diagnose_cider_inputs(gts, res)
    results: dict[str, dict] = {vid: {} for vid in gts}

    cider_scores = compute_cider_with_corpus_idf(gts, res, corpus_idf)
    for vid_id, score in cider_scores.items():
        results[vid_id]["cider"] = score

    gts_s, res_s = _score_string_dicts(gts, res)
    sorted_ids = sorted(gts.keys())

    _, bleu_list = Bleu(4).compute_score(gts_s, res_s)
    for i, vid_id in enumerate(sorted_ids):
        results[vid_id]["bleu1"] = float(bleu_list[0][i])
        results[vid_id]["bleu4"] = float(bleu_list[3][i])

    _, rouge_list = Rouge().compute_score(gts_s, res_s)
    for i, vid_id in enumerate(sorted_ids):
        results[vid_id]["rouge_l"] = float(rouge_list[i])

    _, meteor_list = Meteor().compute_score(gts_s, res_s)
    for i, vid_id in enumerate(sorted_ids):
        results[vid_id]["meteor"] = float(meteor_list[i])

    return results
