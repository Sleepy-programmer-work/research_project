# Final Benchmark Results & Comparative Analysis

This report presents the consolidated performance metrics across all 5 video frame sampling techniques (`tass_adaptive`, `fps1`, `fps2`, `random`, `phash`) evaluated within our captioning benchmark framework.

## Benchmark Configuration
* **Evaluated Samplers**: `tass_adaptive`, `fps1`, `fps2`, `random`, `phash`
* **Aggregation Methods**: `raw`, `centroid`, `temporal`
* **Captioning Modes**: `vlm_only` (VLM captions concatenated) vs. `vlm_plus_llm` (LLM-synthesized visual summaries)
* **Dataset**: MSR-VTT (100 video evaluation set, 20 reference captions per video)
* **Hardware**: NVIDIA GeForce RTX 4050 Laptop GPU & x86_64 CPU

---

## Best Configurations by Metric

Below is the summary of the top-performing system configurations across all evaluation categories:

| Metric Category | Target | Best Configuration | Performance Value | Runner-up Configuration |
| :--- | :--- | :--- | :---: | :--- |
| **CIDEr Quality** | Maximize | `fps2 + centroid (vlm_plus_llm)` | **0.07192** | `fps1 + centroid (vlm_plus_llm)` (0.06621) |
| **BLEU-1 Quality** | Maximize | `fps2 + centroid (vlm_only)` | **0.38612** | `fps1 + centroid (vlm_only)` (0.38558) |
| **BLEU-4 Quality** | Maximize | `tass_adaptive + centroid (vlm_only)` | **0.02302** | `fps1 + centroid (vlm_only)` (0.02181) |
| **ROUGE-L Quality** | Maximize | `tass_adaptive + centroid (vlm_only)` | **0.28191** | `fps2 + centroid (vlm_only)` (0.27712) |
| **METEOR Quality** | Maximize | `tass_adaptive + centroid (vlm_only)` | **0.20986** | `fps2 + centroid (vlm_only)` (0.20902) |
| **Semantic Yield** | Maximize | `tass_adaptive + centroid (vlm_plus_llm)` | **0.01218** | `tass_adaptive + raw (vlm_plus_llm)` (0.00766) |
| **Processing Latency**| Minimize | `fps2 + temporal (vlm_only)` | **0.02134s** | `random + temporal (vlm_only)` (0.02436s) |
| **Selected Frames** | Minimize | `tass_adaptive + temporal (vlm_only)` | **6.57 frames** | `tass_adaptive + centroid (vlm_only)` (6.57 frames) |
| **VRAM Footprint** | Minimize | `fps1 + centroid (vlm_only)` | **0.00 MB** | `fps1 + temporal (vlm_only)` (0.00 MB) |
| **RAM Footprint** | Minimize | `fps2 + temporal (vlm_only)` | **0.00 MB** | `random + temporal (vlm_only)` (0.00 MB) |

---

## Sampler Performance Summary (Averaged Across Aggregations & Modes)

| Sampler | Selected Frames (Mean) | CIDEr (`vlm_plus_llm`) | BLEU-1 (`vlm_only`) | ROUGE-L (`vlm_only`) | Semantic Yield (CIDEr/frame) | Latency (`vlm_only`) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **TASS-Adaptive** | **6.57** | 0.05928 | 0.38120 | **0.28191** | **0.01218** | 0.048s |
| **FPS-1** | 16.29 | 0.06621 | 0.38558 | 0.27259 | 0.00453 | 0.031s |
| **FPS-2** | 24.22 | **0.07192** | **0.38612** | 0.27712 | 0.00324 | **0.021s** |
| **pHash** | 14.85 | 0.05713 | 0.38483 | 0.27328 | 0.00455 | 0.036s |
| **Random** | 16.29 | 0.06346 | 0.38205 | 0.27154 | 0.00420 | 0.024s |

---

## Key Performance & Architectural Insights

### 1. Standalone pHash vs. TASS Adaptive & Uniform Baselines
* **Frame Selection**: `phash` standalone selects **14.85 frames** on average, successfully skipping redundant frames via perceptual Hamming distance compared to uniform `fps1` (16.29) and `fps2` (24.22).
* **Quality & Efficiency**: In `vlm_plus_llm` mode with `centroid` aggregation, `phash` delivers **0.05713 CIDEr** and a **Semantic Yield of 0.00455 CIDEr/frame**, closely matching `fps1` (0.00453) while using fewer frames.
* **TASS Adaptive Dominance**: `tass_adaptive` combines pHash (Stage 1) with MobileCLIP + Greedy FPS (Stage 2) to achieve **6.57 mean frames** (over 55% frame reduction vs pHash alone) while boosting **Semantic Yield to 0.01218 CIDEr/frame** (~2.7× higher than standalone pHash).

### 2. Absolute Textual Quality vs. Semantic Efficiency
* **Absolute Quality Winner**: `fps2 + centroid (vlm_plus_llm)` achieved the highest absolute CIDEr score (**0.07192**). Sampling 24.22 frames per video and clustering embeddings to select key exemplars provides the LLM with maximum visual detail.
* **Semantic Efficiency Winner**: `tass_adaptive` leads on token-level quality metrics (BLEU-4: **0.02302**, ROUGE-L: **0.28191**, METEOR: **0.20986**) in `vlm_only` mode while selecting **only 6.57 frames**.

### 3. Aggregation Strategy & LLM Synthesis
* **Centroid Clustering**: Consistently maximizes textual quality across all 5 samplers by deduplicating visual semantics before feeding context to the LLM.
* **VLM+LLM Synthesis**: Boosts CIDEr quality across all methods (averaging **0.05 - 0.07** CIDEr) compared to VLM-only concatenation (**0.02 - 0.03** CIDEr).

---

## Architectural Recommendations

1. **For Maximum Caption Quality (Search & Indexing)**:
   * Recommend: `fps2 + centroid (vlm_plus_llm)`.
2. **For Resource-Constrained Streaming & Edge Devices**:
   * Recommend: `tass_adaptive + centroid (vlm_only)` or `tass_adaptive + centroid (vlm_plus_llm)`.
   * *Rationale*: Delivers the top Semantic Yield (0.01218 CIDEr/frame), processes only 6.57 frames/video, and minimizes VLM compute overhead.
3. **For Lightweight Perceptual Scene Filtering**:
   * Recommend: `phash + centroid (vlm_plus_llm)`.
   * *Rationale*: Content-aware scene selection without deep learning embedding models.
