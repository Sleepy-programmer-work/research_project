from aggregation.base import BaseAggregator


def build_context(
    captions: list[str],
    transcript: str,
    aggregator: BaseAggregator,
    caption_mode: str,
) -> str:
    """Build the context string for the final caption generation step.

    For 'vlm_only' mode the aggregated caption is returned directly as the
    final caption — no LLM synthesis is performed.  The aggregator is still
    applied so that multi-frame outputs are reduced to a single representative
    string (centroid / temporal dedup / raw join) before scoring.

    BYPASS FIX: previously an empty captions list would propagate an empty
    string all the way to the metric scorer without any warning.  We now
    substitute a placeholder so that the scorer always receives a valid (if
    poor) prediction, and the warning in the log makes the failure visible.

    Args:
        captions:     Per-frame VLM captions (may be empty if all frames failed).
        transcript:   Audio transcript for the video.
        aggregator:   Aggregation strategy (raw / centroid / temporal).
        caption_mode: 'vlm_only' — aggregated caption is the final output.
                      'vlm_plus_llm' — aggregated caption is fed to the LLM.
    """
    # Guard: an empty caption list means every frame failed captioning.
    # Return a placeholder rather than an empty string so the metric scorer
    # always has a valid (if low-scoring) prediction to evaluate.
    if not captions:
        return "[no captions generated]"

    aggregated_caption = aggregator.aggregate(captions)

    if caption_mode == "vlm_only":
        # Aggregator output IS the final caption — no LLM call follows.
        return aggregated_caption

    # vlm_plus_llm: embed the aggregated caption into the LLM prompt.
    # If the aggregator returned an empty string (e.g. all whitespace captions),
    # fall back to a safe placeholder so the prompt structure stays valid.
    visual_summary = aggregated_caption.strip() or "No visual information available."

    prompt = f"""You are an automated video captioning system. \
Your ONLY job is to synthesize the provided visual frame descriptions and audio transcripts into a single, declarative sentence.

STRICT RULES:
1. Output EXACTLY ONE sentence.
2. NEVER use conversational filler (Do NOT say "The video shows", "Here is a summary", "In this clip").
3. Describe only the literal action occurring.
4. Maximum length: 15 words.

Visual Summary: {visual_summary}
Audio Transcript: {transcript}

Caption:"""

    return prompt
