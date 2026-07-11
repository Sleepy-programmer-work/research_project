"""
QwenVLLoader: Singleton loader for Qwen2.5-VL-3B via Ollama.

Architecture role:
  Standalone video captioning baseline — receives multiple uniformly-sampled
  frames, encodes them, and sends a request to the local Ollama instance
  to generate a single holistic video caption in a single call.
"""

import base64
import logging
from typing import List, Optional, Tuple
import cv2
import numpy as np
import requests

logger = logging.getLogger(__name__)

class QwenVLLoader:
    """
    Singleton loader for Qwen2.5-VL-3B via local Ollama API.
    Call QwenVLLoader.get() to retrieve the shared instance.
    """
    _instance: Optional["QwenVLLoader"] = None

    @classmethod
    def get(cls, model_name: str = "qwen2.5vl:3b", host: str = "http://172.31.64.1:11434") -> "QwenVLLoader":
        if cls._instance is None:
            cls._instance = cls(model_name=model_name, host=host)
        return cls._instance

    def __init__(self, model_name: str = "qwen2.5vl:3b", host: str = "http://172.31.64.1:11434") -> None:
        self.model_name = model_name
        self.host = host
        logger.info(f"Initialized QwenVLLoader for model {self.model_name} at {self.host}")

    def frame_to_base64(self, frame: np.ndarray, target_size: Tuple[int, int] = (448, 448)) -> str:
        """
        Resizes the frame to a safe target size to prevent VRAM spikes/OOMs,
        then encodes it to JPEG and converts to a base64 string.
        """
        resized = cv2.resize(frame, target_size, interpolation=cv2.INTER_LANCZOS4)
        success, buffer = cv2.imencode(".jpg", resized)
        if not success:
            raise ValueError("Could not encode frame to JPEG format.")
        return base64.b64encode(buffer).decode("utf-8")

    def generate_video_caption(
        self,
        frames: List[np.ndarray],
        prompt: str = (
            "You are a video understanding system.\n\n"
            "Analyze all provided frames from the same video.\n\n"
            "Generate a concise caption describing the primary action and context of the video.\n\n"
            "Output exactly one sentence.\n\n"
            "Do not use conversational phrases."
        ),
    ) -> Tuple[str, int, bool]:
        """
        Generate one caption for the entire video from multiple frames using Ollama /api/chat.
        Automatically falls back to fewer frames (16 -> 12 -> 8) if memory pressure
        or server errors are encountered.

        Args:
            frames:          List of BGR numpy arrays (uniform sample from video).
            prompt:          Instruction text sent with the images.

        Returns:
            Tuple of (caption, actual_frames_used, oom_fallback_triggered)
        """
        if not frames:
            logger.warning("QwenVLLoader: empty frame list — returning placeholder.")
            return "No visual content available.", 0, False

        original_count = len(frames)
        budgets = [16, 12, 8]
        
        # Determine candidate budgets dynamically based on original_count
        candidate_budgets = []
        for b in budgets:
            if b <= original_count:
                candidate_budgets.append(b)
        if not candidate_budgets:
            candidate_budgets = [original_count]
        
        # Sort descending
        candidate_budgets = sorted(list(set(candidate_budgets)), reverse=True)

        for budget in candidate_budgets:
            # Sample uniformly from frames list
            if len(frames) == budget:
                sampled = frames
            else:
                indices = [int(i * len(frames) / budget) for i in range(budget)]
                sampled = [frames[idx] for idx in indices]
            
            try:
                base64_images = [self.frame_to_base64(f) for f in sampled]
            except Exception as e:
                logger.error(f"QwenVLLoader: failed to process/encode frames for budget {budget}: {e}")
                if budget == candidate_budgets[-1]:
                    raise e
                continue

            fallback_triggered = (budget < original_count)
            if fallback_triggered:
                logger.warning(
                    f"QwenVLLoader: OOM or connection failure fallback triggered. "
                    f"Reducing frame budget from {original_count} to {budget}."
                )

            # Build user message with images
            payload = {
                "model": self.model_name,
                "messages": [
                    {
                        "role": "user",
                        "content": prompt,
                        "images": base64_images
                    }
                ],
                "stream": False,
                "options": {
                    "num_ctx": 32768,
                    "temperature": 0.2,
                    "num_predict": 60
                }
            }

            try:
                response = requests.post(f"{self.host}/api/chat", json=payload, timeout=180)
                response.raise_for_status()
                res_data = response.json()
                
                if "error" in res_data:
                    error_msg = res_data["error"]
                    logger.warning(f"Ollama returned error for budget {budget}: {error_msg}")
                    if budget == candidate_budgets[-1]:
                        raise RuntimeError(f"Ollama execution error: {error_msg}")
                    continue

                caption = res_data.get("message", {}).get("content", "").strip()
                return caption, budget, fallback_triggered

            except Exception as e:
                logger.warning(f"QwenVLLoader: Failed caption generation at budget {budget}. Error: {e}")
                if budget == candidate_budgets[-1]:
                    raise RuntimeError(f"All Ollama frame budgets failed. Last error: {e}")
                # Clear python memory / gc between retries just in case
                import gc
                gc.collect()

        raise RuntimeError("QwenVLLoader: caption generation failed across all budgets.")
