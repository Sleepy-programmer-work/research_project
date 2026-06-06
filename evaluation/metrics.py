import logging
import json
import math
import numpy as np
from pathlib import Path
from typing import Optional
from collections import defaultdict

from pycocoevalcap.cider.cider_scorer import CiderScorer
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.meteor.meteor import Meteor

from evaluation.corpus_idf import load_corpus_idf
from config.settings import settings

logger = logging.getLogger(__name__)


class CustomCiderScorer(CiderScorer):
    """
    Subclass of CiderScorer that bypasses the test-size assertion check
    and supports precomputed corpus IDF weights.
    """
    def __init__(self, df=None, n=4, sigma=6.0):
        super().__init__(n=n, sigma=sigma)
        if df is not None:
            self.document_frequency = df

    def compute_cider(self):
        def counts2vec(cnts):
            vec = [defaultdict(float) for _ in range(self.n)]
            length = 0
            norm = [0.0 for _ in range(self.n)]
            for (ngram, term_freq) in cnts.items():
                n = len(ngram) - 1
                if n >= self.n or n < 0:
                    continue
                # tf (term_freq) * idf (precomputed idf) for n-grams
                vec[n][ngram] = float(term_freq) * self.document_frequency.get(ngram, 0.0)
                norm[n] += pow(vec[n][ngram], 2)

                if n == 1:
                    length += term_freq
            norm = [np.sqrt(n) for n in norm]
            return vec, norm, length

        def sim(vec_hyp, vec_ref, norm_hyp, norm_ref, length_hyp, length_ref):
            delta = float(length_hyp - length_ref)
            val = np.array([0.0 for _ in range(self.n)])
            for n in range(self.n):
                for (ngram, count) in vec_hyp[n].items():
                    val[n] += min(vec_hyp[n][ngram], vec_ref[n][ngram]) * vec_ref[n][ngram]

                if (norm_hyp[n] != 0) and (norm_ref[n] != 0):
                    val[n] /= (norm_hyp[n]*norm_ref[n])

                assert(not math.isnan(val[n]))
                val[n] *= np.e**(-(delta**2)/(2*self.sigma**2))
            return val

        scores = []
        for test, refs in zip(self.ctest, self.crefs):
            vec, norm, length = counts2vec(test)
            score = np.array([0.0 for _ in range(self.n)])
            for ref in refs:
                vec_ref, norm_ref, length_ref = counts2vec(ref)
                score += sim(vec, vec_ref, norm, norm_ref, length, length_ref)
            score_avg = np.mean(score)
            score_avg /= len(refs)
            score_avg *= 10.0
            scores.append(score_avg)
        return scores

    def compute_score(self):
        score = self.compute_cider()
        return np.mean(np.array(score)), np.array(score)


def _diagnose_cider_inputs(
    gts: dict,
    res: dict,
    logger: logging.Logger
) -> None:
    """
    Print a structured diagnostic of gts and res before scoring.
    Call once per batch. Safe to leave in production — logs at DEBUG level.
    """
    logger.debug("=== CIDEr input diagnostic ===")
    logger.debug(f"  gts keys (first 3): {list(gts.keys())[:3]}")
    logger.debug(f"  res keys (first 3): {list(res.keys())[:3]}")

    for vid_id in list(gts.keys())[:2]:
        refs = gts[vid_id]
        hyp  = res.get(vid_id, [])
        logger.debug(f"  video '{vid_id}':")
        logger.debug(f"    refs type : {type(refs)}")
        logger.debug(f"    refs[0]   : {refs[0] if refs else 'EMPTY'}")
        logger.debug(f"    hyp type  : {type(hyp)}")
        logger.debug(f"    hyp[0]    : {hyp[0] if hyp else 'EMPTY'}")

        # Cause A check
        if refs and isinstance(refs[0], str):
            logger.error(
                f"  CAUSE A DETECTED: refs are plain strings, "
                f"not dicts. pycocoevalcap requires "
                f"[{{'image_id': id, 'caption': text}}, ...]"
            )

        # Cause C check
        if refs and isinstance(refs[0], dict):
            empty = [r for r in refs if not r.get("caption", "").strip()]
            if empty:
                logger.error(
                    f"  CAUSE C DETECTED: {len(empty)} empty caption "
                    f"strings in refs for video '{vid_id}'"
                )

    logger.debug(f"  total gts videos : {len(gts)}")
    logger.debug(f"  total res videos : {len(res)}")
    if len(gts) < 50:
        logger.warning(
            f"  CAUSE B WARNING: only {len(gts)} videos in gts. "
            f"CIDEr IDF computed on this small corpus — "
            f"scores will be unreliable. Use full-corpus IDF instead."
        )


def build_gts_res(
    video_ids: list[str],
    reference_captions: dict[str, list[str]],
    generated_captions: dict[str, str],
    logger: logging.Logger
) -> tuple[dict, dict]:
    """
    Build gts and res dicts in the exact format pycocoevalcap requires.

    pycocoevalcap mandates:
      gts = {
        "video_id_as_str": [
          {"image_id": "video_id_as_str", "caption": "ref caption 1"},
          {"image_id": "video_id_as_str", "caption": "ref caption 2"},
          ...
        ]
      }
      res = {
        "video_id_as_str": [
          {"image_id": "video_id_as_str", "caption": "generated caption"}
        ]
      }

    Keys MUST be strings. Values MUST be lists of dicts.
    Plain strings in the list → CIDEr returns 0 with no error.
    """
    gts: dict[str, list[dict]] = {}
    res: dict[str, list[dict]] = {}

    skipped = 0
    for vid_id in video_ids:
        key = str(vid_id)

        # --- Build references ---
        raw_refs = reference_captions.get(vid_id, [])
        # Filter empty / None / whitespace-only strings and sanitize whitespaces/newlines
        clean_refs = []
        for r in raw_refs:
            if r and str(r).strip():
                clean_ref = " ".join(str(r).strip().split())
                if clean_ref:
                    clean_refs.append(clean_ref)

        if not clean_refs:
            logger.warning(
                f"video '{vid_id}' has no valid reference captions — skipping."
            )
            skipped += 1
            continue

        gts[key] = [
            {"image_id": key, "caption": ref}
            for ref in clean_refs
        ]

        # --- Build hypothesis ---
        gen = generated_captions.get(vid_id, "").strip()
        gen_clean = " ".join(gen.split())
        if not gen_clean:
            logger.warning(
                f"video '{vid_id}' has empty generated caption — "
                f"using placeholder to avoid scorer crash."
            )
            gen_clean = "no caption generated"

        res[key] = [{"image_id": key, "caption": gen_clean}]

    if skipped:
        logger.warning(
            f"Skipped {skipped}/{len(video_ids)} videos "
            f"due to missing reference captions."
        )

    return gts, res


def compute_cider_with_corpus_idf(
    gts: dict[str, list[dict]],
    res: dict[str, list[dict]],
    corpus_idf: dict,
) -> dict[str, float]:
    """
    Compute CIDEr using the full MSVD corpus IDF instead of the
    small evaluation subset IDF. This is the methodologically correct
    approach for subset evaluation.

    gts and res must already be in pycocoevalcap dict format.
    corpus_idf is the dict returned by load_corpus_idf().
    """
    prebuilt_idf = corpus_idf["idf"]

    scorer = CustomCiderScorer(df=prebuilt_idf, n=4, sigma=6.0)

    scores: dict[str, float] = {}

    for vid_id in sorted(gts.keys()):
        if vid_id not in res:
            logger.warning(f"No hypothesis for video '{vid_id}' — skipping CIDEr.")
            scores[vid_id] = 0.0
            continue

        hyp_caption  = res[vid_id][0]["caption"]
        ref_captions = [r["caption"] for r in gts[vid_id]]

        scorer += (hyp_caption, ref_captions)

    if not scorer.crefs:
        logger.error("CiderScorer has no references — returning 0.")
        return {vid_id: 0.0 for vid_id in gts}

    (score, per_image_scores) = scorer.compute_score()

    vid_ids = sorted(gts.keys())
    for i, vid_id in enumerate(vid_ids):
        if vid_id in res:
            scores[vid_id] = float(per_image_scores[i])

    return scores


def compute_all_metrics(
    gts: dict[str, list[dict]],
    res: dict[str, list[dict]],
    corpus_idf: dict,
) -> dict[str, dict[str, float]]:
    """
    Compute CIDEr, BLEU-1, BLEU-4, ROUGE-L, METEOR for every video.
    Returns per-video metric dict:
        { video_id: { "cider": x, "bleu1": x, "bleu4": x,
                      "rouge_l": x, "meteor": x } }
    """
    _diagnose_cider_inputs(gts, res, logger)

    results: dict[str, dict] = {vid: {} for vid in gts}

    # --- CIDEr (full-corpus IDF) ---
    cider_scores = compute_cider_with_corpus_idf(gts, res, corpus_idf)
    for vid_id, score in cider_scores.items():
        results[vid_id]["cider"] = score

    # Convert to list-of-strings format for Bleu, Rouge, Meteor
    gts_strings = {vid_id: [r["caption"] for r in gts[vid_id]] for vid_id in gts}
    res_strings = {vid_id: [r["caption"] for r in res[vid_id]] for vid_id in res}

    # --- BLEU-1 and BLEU-4 (pycocoevalcap) ---
    bleu_scorer = Bleu(4)
    _, bleu_scores_list = bleu_scorer.compute_score(gts_strings, res_strings)
    for i, vid_id in enumerate(sorted(gts.keys())):
        results[vid_id]["bleu1"] = float(bleu_scores_list[0][i])
        results[vid_id]["bleu4"] = float(bleu_scores_list[3][i])

    # --- ROUGE-L ---
    rouge_scorer = Rouge()
    _, rouge_scores_list = rouge_scorer.compute_score(gts_strings, res_strings)
    for i, vid_id in enumerate(sorted(gts.keys())):
        results[vid_id]["rouge_l"] = float(rouge_scores_list[i])

    # --- METEOR ---
    meteor_scorer = Meteor()
    _, meteor_scores_list = meteor_scorer.compute_score(gts_strings, res_strings)
    for i, vid_id in enumerate(sorted(gts.keys())):
        results[vid_id]["meteor"] = float(meteor_scores_list[i])

    return results


def save_raw_caption(video_id: str, method: str, agg: str, mode: str, gen: str, gt: list[str]):
    out_dir = Path(settings.experiment.get("output_dir", "./results")) / "captions"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    data = {
        "video_id": video_id,
        "method": method,
        "aggregation": agg,
        "mode": mode,
        "generated_caption": gen,
        "ground_truth": gt
    }
    
    with open(out_dir / f"{video_id}_{method}_{agg}_{mode}.json", "w") as f:
        json.dump(data, f, indent=2)
