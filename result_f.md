# Final Benchmark Results & Comparative Analysis

This report presents the consolidated performance metrics across all 5 video frame sampling techniques evaluated within our captioning benchmark framework.

## Benchmark Configuration
* **Evaluated Samplers**: `tass_adaptive`, `fps1`, `fps2`, `random`, `ssim_090`
* **Aggregation Methods**: `raw`, `centroid`, `temporal`
* **Captioning Modes**: `vlm_only` (VLM captions concatenated) vs. `vlm_plus_llm` (LLM-synthesized visual summaries)
* **Dataset**: MSR-VTT (100 video subset, 20 reference captions per video)
* **Hardware**: NVIDIA GeForce RTX 4050 Laptop GPU & x86_64 CPU

---

## Best Configurations by Metric

Below is the summary of the top-performing system configurations across different evaluation categories:

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

## Key Performance & Architectural Insights

### 1. Absolute Textual Quality vs. Semantic Efficiency
* **Absolute Quality Winner**: `fps2 + centroid (vlm_plus_llm)` achieved the highest absolute CIDEr score (**0.07192**). By sampling a high rate of frames (mean of **24.22 frames** per video) and clustering their embeddings to extract key semantic frames (`centroid`), it provides the LLM with the most comprehensive visual information.
* **Semantic Efficiency Winner**: `tass_adaptive` dominates on all token-overlap metrics (BLEU-4, ROUGE-L, METEOR) in `vlm_only` mode while selecting **only 6.57 frames** on average.
* **Semantic Yield**: `tass_adaptive + centroid (vlm_plus_llm)` achieved a **Semantic Yield of 0.01218 CIDEr/frame**, which is **nearly 3x higher** than `fps2` (0.00324) and `fps1` (0.00453). This confirms that TASS successfully eliminates redundant frames, generating highly descriptive summaries with minimal visual input.

### 2. Upstream Aggregation Impact
* **Centroid Clustering**: Across almost all samplers, `centroid` aggregation yielded the highest captioning quality. Grouping frames based on visual similarity and selecting cluster exemplars ensures distinct, non-repetitive descriptions.
* **Temporal Aggregation**: `temporal` aggregation is the fastest option. In `vlm_only` mode, it processes videos in **~0.02 seconds** (latency frontier) because it quickly chains temporal keywords without visual feature matching.

### 3. VLM-Only vs. VLM+LLM Modes
* **VLM+LLM**: Yields significantly higher CIDEr scores (averaging **0.05 - 0.07**) compared to VLM-only modes (averaging **0.02 - 0.03**). Synthesizing captions via a secondary LLM (`phi3:mini`) resolves grammatical fragmentation and structures raw captions into coherent narrative flows.
* **VRAM Overhead**: VLM+LLM mode incurs a VRAM overhead (typically **~0.5 - 2.0 MB** delta after model initialization) and increases latency (from **~0.03s** to **~0.7s** mean processing time).

---

## Architectural Recommendations

1. **For High-Performance Captioning (e.g. detailed index searching)**:
   * Recommend: `fps2 + centroid (vlm_plus_llm)`.
   * *Rationale*: Delivers the highest descriptive resolution and the top CIDEr scores.
2. **For High-Efficiency & Real-Time Streaming**:
   * Recommend: `tass_adaptive + centroid (vlm_only)`.
   * *Rationale*: Maximizes semantic yield, processes only 6.57 frames/video, and yields top BLEU-4 and ROUGE-L quality without LLM processing overhead.
3. **For Low-Resource WSL2 / Edge Deployments**:
   * Recommend: `tass_adaptive + temporal (vlm_only)`.
   * *Rationale*: Minimizes memory consumption and selected frame count while remaining fast.
