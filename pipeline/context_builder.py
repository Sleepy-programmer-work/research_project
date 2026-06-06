from aggregation.base import BaseAggregator

def build_context(captions: list[str], transcript: str, aggregator: BaseAggregator, caption_mode: str) -> str:
    aggregated_caption = aggregator.aggregate(captions)
    
    if caption_mode == "vlm_only":
        return aggregated_caption
        
    prompt = f"""You are an automated video captioning system. 
Your ONLY job is to synthesize the provided visual frame descriptions and audio transcripts into a single, declarative sentence.

STRICT RULES:
1. Output EXACTLY ONE sentence.
2. NEVER use conversational filler (Do NOT say "The video shows", "Here is a summary", "In this clip").
3. Describe only the literal action occurring.
4. Maximum length: 15 words.

Visual Summary: {aggregated_caption}
Audio Transcript: {transcript}

Caption:"""
    
    return prompt

