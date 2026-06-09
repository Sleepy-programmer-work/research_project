"""
models/clip_embedder.py — Singleton MobileCLIP-S1 CPU embedder for TASS Stage 2.

Design constraints (enforced, not optional):
  - Model ALWAYS runs on CPU.  This is a hard architectural requirement:
    the 6 GB VRAM pool is reserved exclusively for Moondream2 VLM inference.
    Any accidental GPU allocation here will trigger OOM on the RTX 4050.
  - Weights are loaded ONCE via the Singleton pattern and reused across all
    sampler calls within a benchmark run (no repeated disk I/O or RAM churn).
  - Encoding is chunked into mini-batches of ≤16 frames to stay within the
    12 GB WSL2 RAM cap even for pathologically long videos.

Memory footprint:
  MobileCLIP-S1 weights:     ~85 MB RAM  (CPU tensors)
  Per-batch tensor overhead:  ~16 frames × 3 × 256 × 256 × 4 bytes ≈ 50 MB
  Total peak RAM per call:   ~135 MB (well within 12 GB ceiling)

Output embedding dimension: 512 (L2-normalised float32 vectors).
This matches the cosine-distance space used by the Greedy FPS stage in TASS.
"""

import gc
import logging
import numpy as np
import open_clip
import torch
from PIL import Image

logger = logging.getLogger(__name__)


class MobileCLIPEmbedder:
    """
    Singleton MobileCLIP-S1 image embedder for TASS Stage 2 semantic diversity.

    All forward passes are executed on CPU to guarantee zero VRAM consumption,
    preserving the full 6 GB VRAM budget for Moondream2.

    Usage:
        embedder = MobileCLIPEmbedder.get()   # first call loads weights
        embedder = MobileCLIPEmbedder.get()   # subsequent calls reuse singleton
        embeddings = embedder.encode_micro_batched(frames, batch_size=16)

    Thread safety:
        The singleton is NOT thread-safe.  The benchmark runner is single-process
        (mp.set_start_method("spawn") is for the VLM subprocess only), so this
        is acceptable.  If parallel sampling is ever added, wrap get() in a lock.
    """

    MODEL_NAME = "MobileCLIP-S1"
    PRETRAINED = "datacompdr"
    DEVICE = "cpu"

    _instance: "MobileCLIPEmbedder | None" = None

    @classmethod
    def get(cls) -> "MobileCLIPEmbedder":
        """
        Return the singleton embedder, loading weights on first call.

        Subsequent calls return the cached instance with zero overhead.
        This pattern prevents redundant disk I/O during the benchmark's
        inner loop (one video per iteration, many iterations per run).
        """
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self) -> None:
        """
        Load MobileCLIP-S1 weights onto CPU and set model to eval mode.

        Called exactly once per process lifetime (enforced by the singleton).
        Raising here is intentional — a missing model should halt the benchmark
        rather than silently produce empty embeddings.
        """
        logger.info(
            f"Loading {self.MODEL_NAME} (pretrained='{self.PRETRAINED}') "
            f"onto {self.DEVICE}. Expected RAM footprint: ~85 MB."
        )
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            self.MODEL_NAME,
            pretrained=self.PRETRAINED,
            device=self.DEVICE,
        )
        self.model.eval()
        # Explicit guard: if open_clip ever changes defaults, this assertion
        # will catch an accidental GPU allocation before VRAM is exhausted.
        assert next(self.model.parameters()).device.type == "cpu", (
            f"MobileCLIP-S1 must reside on CPU but was found on "
            f"{next(self.model.parameters()).device}. "
            f"This violates the VRAM isolation contract for RTX 4050 edge hardware."
        )
        logger.info(
            f"{self.MODEL_NAME} loaded successfully. "
            f"Embedding dimension: 512. Device: {self.DEVICE}."
        )

    def encode_micro_batched(
        self,
        frames: list[np.ndarray],
        batch_size: int = 16,
    ) -> np.ndarray:
        """
        Encode a list of BGR numpy frames into L2-normalised 512-d embeddings.

        Frames are processed in chunks of `batch_size` to cap the per-call RAM
        overhead at ~50 MB per batch (safe under the 12 GB WSL2 constraint).
        Tensors are explicitly deleted after each chunk to assist the Python
        garbage collector on systems without a CUDA memory manager.

        Args:
            frames:     List of BGR uint8 numpy arrays (H×W×3).
                        Height and width may vary; the CLIP preprocessor
                        handles resizing to the model's native 256×256 input.
            batch_size: Number of frames to process per forward pass.
                        Default 16 is tuned for the 12 GB RAM ceiling.
                        Reduce to 8 if RAM pressure is observed in monitoring.

        Returns:
            Float32 numpy array of shape (N, 512) with L2-normalised rows.
            Returns shape (0, 512) for an empty input list — callers should
            guard against this before indexing into the result.

        Design note on BGR→RGB conversion:
            OpenCV stores frames as BGR but PIL and CLIP expect RGB.
            The slice `f[:, :, ::-1]` reverses the channel axis in-place
            without copying the array data (numpy view semantics), making
            it O(1) in memory.  The subsequent PIL.Image.fromarray() call
            then copies into RGB PIL format as required by open_clip's
            preprocessing pipeline.
        """
        if not frames:
            return np.empty((0, 512), dtype=np.float32)

        all_embeddings: list[np.ndarray] = []

        for start in range(0, len(frames), batch_size):
            chunk = frames[start : start + batch_size]

            # BGR (OpenCV) → RGB (PIL) channel reversal — numpy view, zero-copy
            pil_images = [Image.fromarray(f[:, :, ::-1]) for f in chunk]

            # Apply CLIP preprocessing (resize, centre-crop, normalise)
            tensors = torch.stack(
                [self.preprocess(img) for img in pil_images]
            ).to(self.DEVICE)

            with torch.no_grad():
                features = self.model.encode_image(tensors)
                # L2 normalisation projects embeddings onto the unit hypersphere.
                # This is required for cosine similarity to equal the dot product,
                # which is the operation used in TASSSampler's Greedy FPS step.
                features = features / features.norm(dim=-1, keepdim=True)

            all_embeddings.append(features.cpu().numpy().astype(np.float32))

            # Explicit tensor deallocation.  On CPU there is no CUDA cache to
            # flush, but releasing the reference allows Python's reference
            # counter to reclaim the memory before the next batch is allocated,
            # preventing a brief double-allocation spike.
            del tensors, features
            gc.collect()

        return np.vstack(all_embeddings)
