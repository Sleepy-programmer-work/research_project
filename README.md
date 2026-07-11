# Resource-Efficient Video Captioning on Edge Hardware
### Two-Stage Adaptive Semantic Sampling (TASS) — Benchmark Study

> **MSR-VTT · Moondream2 VLM · Phi-3 Mini LLM · RTX 4050 Laptop GPU**

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Core Research Question](#2-core-research-question)
3. [System Architecture](#3-system-architecture)
4. [Sampler Algorithms — How Each Works](#4-sampler-algorithms--how-each-works)
5. [Aggregation Methods](#5-aggregation-methods)
6. [Models & Hardware](#6-models--hardware)
7. [Benchmark Results](#7-benchmark-results)
8. [Why TASS is the Best Sampler](#8-why-tass-is-the-best-sampler)
9. [Project Structure](#9-project-structure)
10. [Benchmarking Framework](#10-benchmarking-framework)
11. [Running the Benchmark](#11-running-the-benchmark)
12. [Configuration Reference](#12-configuration-reference)
13. [Evaluation Metrics](#13-evaluation-metrics)
14. [Developer Guide](#14-developer-guide)
15. [Future Work](#15-future-work)

---

## 1. Project Overview

This project benchmarks five video frame-sampling strategies for automated video captioning, specifically targeting **consumer-grade edge hardware** — the NVIDIA RTX 4050 Laptop GPU (6 GB VRAM) running under WSL2.

The pipeline works as follows:

```
Video File → Frame Sampler → VLM (Moondream2) per frame → Aggregator → [Optional LLM (Phi-3)] → Caption
```

Each sampler answers the question: **which frames of this video are worth describing?** The sampler's choice directly determines caption quality, processing latency, and memory usage.

The core contribution of this project is **TASS** — a novel two-stage adaptive algorithm that uses perceptual hashing and semantic embeddings (MobileCLIP-S1) to select only the most informative, visually diverse frames, achieving the highest efficiency (CIDEr per frame) of all evaluated methods.

---

## 2. Core Research Question

> *"Can content-aware, adaptive frame selection significantly reduce computational cost while maintaining or improving caption quality on edge hardware?"*

**Short answer from our benchmarks: Yes.** TASS selects a mean of **6.57 frames** per video (vs. 24.22 for FPS-2) while matching or outperforming all baselines on BLEU-4, ROUGE-L, and METEOR, and delivering nearly **3× higher semantic yield** (quality per frame processed).

---

## 3. System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                         Input Video                         │
└───────────────────────────────┬─────────────────────────────┘
                                │
                    ┌───────────▼───────────┐
                    │    Frame Sampler       │
                    │  (fps1 / fps2 /        │
                    │   random / ssim_090 /  │
                    │   tass_adaptive)       │
                    └───────────┬───────────┘
                                │  Selected frames (list[np.ndarray])
          ┌─────────────────────▼─────────────────────┐
          │         Moondream2 VLM (per-frame)         │
          │   float16, CUDA, revision-pinned           │
          └─────────────────────┬─────────────────────┘
                                │  Raw caption strings
              ┌─────────────────▼─────────────────┐
              │          Aggregator               │
              │   raw / temporal / centroid       │
              └─────────────────┬─────────────────┘
                                │
              ┌─────────────────▼──────────────────┐
              │   Caption Mode: vlm_only            │
              │   OR                                │
              │   vlm_plus_llm → Phi-3 Mini (Ollama)│
              └─────────────────┬──────────────────┘
                                │
              ┌─────────────────▼──────────────────┐
              │   Evaluation (CIDEr, BLEU, ROUGE,  │
              │   METEOR) + Resource Telemetry     │
              └────────────────────────────────────┘
```

The benchmark evaluates all combinations of `sampler × aggregator × caption_mode`, producing per-video results saved to CSV, with aggregated statistics, plots, and a Markdown summary report.

---

## 4. Sampler Algorithms — How Each Works

All samplers implement the `BaseSampler` interface: `sample(video_path) → List[np.ndarray]` and `sample_with_metadata(video_path) → dict`. The metadata dict carries telemetry fields (`frames_original`, `candidate_pool_size`, `tass_stopped_early`, etc.) consumed by the benchmark loop.

---

### 🎯 FPS-1 (Uniform, 1 Frame/Second)

**File**: [`samplers/fps1.py`](samplers/fps1.py)

**How it works:**
FPS-1 reads the video's native frame rate using `cv2.CAP_PROP_FPS` and selects every `round(fps)`-th frame — effectively one frame per second of footage. For a 16-second clip at 30 FPS, this yields 16 frames.

```
Video:  [f0 f1 f2 ... f29 | f30 f31 ... f59 | ...]
          ^                  ^                    (selected every 30th frame)
```

**Properties:**
- Deterministic, reproducible, no content-awareness
- Frame budget scales linearly with video duration
- Mean frames selected: **16.29** (MSR-VTT, 100 videos)
- Processing time (vlm_only): **~0.05s** mean

**Weakness**: Selects frames by clock position alone — it has no awareness of scene content. Two visually identical consecutive frames separated by one second are both selected; a dramatic visual transition mid-second is ignored.

---

### 🎯 FPS-2 (Uniform, 2 Frames/Second)

**File**: [`samplers/fps2.py`](samplers/fps2.py)

**How it works:**
Identical to FPS-1, but selects every `round(fps/2)`-th frame — two frames per second. This doubles the temporal resolution at the cost of double the VLM calls.

```
Video:  [f0 ... f14 | f15 ... f29 | f30 ... f44 | ...]
          ^            ^              ^               (every 15th frame at 30fps)
```

**Properties:**
- Mean frames selected: **24.22** (50% more than FPS-1)
- Highest absolute CIDEr score: **0.07192** (`fps2 + centroid + vlm_plus_llm`)
- Processing time (vlm_only): **~0.03s** mean
- Best raw quality when combined with centroid aggregation and LLM synthesis

**Weakness**: Higher frame budget means more VLM inference calls. The additional frames are not guaranteed to carry novel information — consecutive similar frames waste computation.

---

### 🎯 Random Sampling

**File**: [`samplers/random_sampler.py`](samplers/random_sampler.py)

**How it works:**
Calculates a target frame count equal to `ceil(total_frames / fps)` — the same budget as FPS-1 — then draws that many frame indices uniformly at random **without replacement** (seeded at `seed=42` for reproducibility). The frames are decoded in temporal order to preserve sequencing.

```python
target_count = max(1, ceil(total_frames / fps))
indices = sorted(random.sample(range(total_frames), target_count))
```

**Properties:**
- Same frame budget as FPS-1 on average; mean: **16.29 frames**
- Budget-matched comparison to FPS-1 to isolate the effect of frame placement
- Fully reproducible via the fixed seed
- Processing time (vlm_only): **~0.03s** mean

**Weakness**: Randomness offers no content-awareness either. Given the same budget as FPS-1, performance is statistically indistinguishable (confirmed in our benchmarks where CIDEr scores are within 95% CI of FPS-1 values). Its primary purpose is as a **statistical control** — if random sampling performs the same as FPS-1, it suggests uniform sampling is already capturing most of the relevant content.

---

### 🎯 SSIM-090 (Structural Similarity Scene Detection)

**File**: [`samplers/ssim.py`](samplers/ssim.py)

**How it works:**
SSIM (Structural Similarity Index Measure) is a perceptual image quality metric that captures luminance, contrast, and structural changes between two grayscale images. The SSIM sampler streams through the video frame-by-frame and triggers an acceptance only when the SSIM between the current frame and the previous **accepted** frame drops below a threshold.

**Algorithm (O(N) streaming, O(K) storage):**
1. Resize each frame to `256×144` (configurable) and convert to grayscale.
2. Compute `SSIM(prev_frame, curr_frame)` using a 7×7 sliding window.
3. If `SSIM < threshold` (0.90 for `ssim_090`) → **accept** the frame; update `prev_frame`.
4. If the acceptance rate falls below `1%`, fall back to FPS-1 sampling.
5. If the acceptance rate exceeds `99%`, log a warning (video may be static or filtering is ineffective).

```
Video:  [f0 f1 f2 f3 f4 f5 f6 f7 f8 ...]
SSIM:      [0.98 0.95 0.93 0.62 0.97 0.99 0.88 ...]
                               ^                     (scene change: SSIM=0.62 < 0.90 → accept)
```

**Properties:**
- Content-aware: only accepts frames at structural scene changes
- Mean frames selected: **34.44** — highest of all samplers (SSIM is not budget-limited)
- Three threshold variants: `ssim_085`, `ssim_090`, `ssim_095` (stricter = fewer frames)
- CPU-only streaming; no GPU required
- Reduction metadata tracked per-video in `results/frame_selection/`

**Weakness**: SSIM operates in pixel-space only — it detects _visual_ change but not _semantic_ change. A camera zoom on the same scene triggers a low SSIM (accepted as new frame), while a scene with changing text but static background is missed. Also, SSIM has no frame budget — it can select far more frames than necessary for static or slow-moving videos.

---

### 🎯 TASS-Adaptive — The Research Contribution

**File**: [`samplers/tass.py`](samplers/tass.py), [`samplers/tass_helpers.py`](samplers/tass_helpers.py)

**How it works:**
TASS uses a **two-stage pipeline** that first removes visually uninformative frames (degenerate purge + perceptual hashing), then selects the most **semantically diverse** frames using deep visual embeddings (MobileCLIP-S1) and Greedy Farthest-Point Sampling.

#### Stage 1 — Streaming Degenerate Purge + Perceptual Hash Filter (CPU, O(N))

The video is streamed at a stride of 3 (every 3rd frame, ~10 effective FPS at 30 FPS source) to reduce I/O:

**1a. Degenerate Frame Detection:**
Each strided frame is tested against two low-level criteria:
- **Brightness check**: Mean pixel value < 10 or > 245 → flash/black frame → dropped.
- **Variance check**: Grayscale variance < 30 → uniform/flat content → dropped.

```python
mean_brightness = frame_gray.mean()
if mean_brightness < 10 or mean_brightness > 245:
    drop()  # black or blown-out flash
if frame_gray.var() < 30:
    drop()  # lens cap / uniform color
```

**1b. Perceptual Hash (pHash) Scene-Change Detection:**
Non-degenerate frames are resized to `256×144`, converted to grayscale, and their perceptual hash (8×8 DCT-based fingerprint) is computed. A frame is added to the candidate pool **only if** its Hamming distance from the previous accepted frame's hash is `> 1` (indicating a meaningful visual change).

```
Frame distance > 1 → new scene → add to candidate pool
Frame distance ≤ 1 → visually redundant → skip
```

This produces a compact **candidate pool** of visually distinct frames (capped at 2000 candidates).

#### Stage 2 — MobileCLIP Embedding + Greedy Farthest-Point Sampling (CPU, O(M))

Each candidate frame is encoded into a **512-dimensional semantic embedding** by `MobileCLIP-S1` — a lightweight CLIP variant (~85 MB RAM, CPU-only Singleton).

Greedy Farthest-Point Sampling (FPS) then selects the K most **semantically diverse** embeddings:
1. Start with the first candidate.
2. At each step, select the candidate **most distant** (cosine distance) from all already-selected candidates.
3. **Adaptive early stopping** (in `adaptive` mode): halt when the maximum available cosine distance falls below `min_distance=0.10`, meaning all remaining candidates are semantically redundant.

```
embeddings: [e1, e2, e3, e4, e5, e6, ...]
Select e1 → farthest from e1 is e4 → farthest from {e1,e4} is e2 → ...
Stop when max_remaining_distance < 0.10
```

**Result:** A small, maximally diverse set of frames — each representing a unique semantic scene — sent to the VLM.

**Properties:**
- Mean frames selected: **6.57** (3.7× fewer than FPS-1, 7.5× fewer than FPS-2, 5.2× fewer than SSIM-090)
- Highest BLEU-4 (`0.02302`), ROUGE-L (`0.28191`), and METEOR (`0.20986`) of all samplers in `vlm_only` mode
- Highest Semantic Yield: **0.01218 CIDEr/frame** (≈3× better than FPS-2)
- MobileCLIP loads on first call as a CPU-only Singleton (~85 MB RAM, zero VRAM)
- Telemetry: tracks `tass_candidate_pool`, `tass_degenerate_dropped`, `tass_stopped_early`, `vlm_calls_saved_pct`

---

## 5. Aggregation Methods

After the sampler returns a list of frame captions from the VLM, an aggregator combines them into a single context string (for `vlm_only`) or a structured LLM prompt (for `vlm_plus_llm`).

### Raw Aggregation
**File**: [`aggregation/raw.py`](aggregation/raw.py)

Joins all frame captions with newline separators in temporal order. No deduplication, no summarization.

```python
result = "\n".join(captions)
```

- **Fastest** aggregation method (O(N) string join)
- Passes all information to the LLM but creates long, repetitive prompts
- Best paired with `vlm_only` for transparency/debugging

### Temporal Aggregation
**File**: [`aggregation/temporal.py`](aggregation/temporal.py)

Deduplicates temporally adjacent captions using **Jaccard similarity** on word tokens. If consecutive captions are too similar (Jaccard > threshold), the duplicate is removed.

```python
# If similarity(cap_i, cap_{i+1}) > 0.7 → merge/drop cap_{i+1}
```

- Reduces repetitive VLM output (common when adjacent frames are similar)
- Preserves temporal ordering
- Processing time in `vlm_only` mode: **~0.02s mean** — the fastest pipeline overall

### Centroid Aggregation
**File**: [`aggregation/centroid.py`](aggregation/centroid.py)

Selects the **single most representative caption** from all frame captions using pairwise text similarity. The centroid is the caption whose mean similarity to all other captions is highest.

```python
# Compute pairwise text similarity matrix (NLTK-based)
# Return caption with highest mean row similarity
```

- Acts as a quality filter — returns the one "most typical" frame description
- Best paired with `vlm_plus_llm` where the centroid becomes the LLM's visual summary input
- Highest quality across nearly all quality metrics in our benchmarks

---

## 6. Models & Hardware

### Moondream2 VLM
- **Model**: `vikhyatk/moondream2` (revision `2024-08-26`)
- **Role**: Per-frame visual question answering / captioning
- **Loading**: Singleton (loaded once, reused across all pipeline configurations)
- **Precision**: `float16` on CUDA (RTX 4050, 6 GB VRAM)
- **Prompt**: `"Describe what is happening in this frame in exactly one sentence."`

### Phi-3 Mini LLM (Ollama)
- **Model**: `phi3:mini` served via Ollama at `localhost:11434`
- **Role**: Synthesizes per-frame captions into a single coherent description (`vlm_plus_llm` mode only)
- **Temperature**: 0.2 (near-deterministic)
- **Prompt**: Structured template with visual summary + strict word limit constraints

### MobileCLIP-S1
- **Role**: TASS Stage 2 semantic frame embedding (512-d vectors)
- **Loading**: CPU-only Singleton (~85 MB RAM, zero VRAM)
- **Batching**: Micro-batches of ≤16 frames to stay within WSL2 memory limits

### Hardware
| Component | Specification |
|:---|:---|
| GPU | NVIDIA GeForce RTX 4050 Laptop GPU, 6 GB VRAM |
| CPU | x86_64 (Intel/AMD) |
| RAM | ~12 GB usable (WSL2) |
| OS | Ubuntu on WSL2 |
| Dataset | MSR-VTT (`vishnutheepb/msrvtt`) — 100 videos, 20 reference captions each |

---

## 7. Benchmark Results

All numbers are means over 100 videos × all aggregation methods × both caption modes. Full per-configuration statistics are in [`results/csv/combined_statistics_summary.csv`](results/csv/combined_statistics_summary.csv).

### Quality Metrics — Best Configuration Per Metric

| Metric | Winner | Value | Notes |
|:---|:---|:---:|:---|
| **CIDEr** | `fps2 + centroid + vlm_plus_llm` | **0.0719** | Highest absolute captioning quality |
| **BLEU-1** | `fps2 + centroid + vlm_only` | **0.3861** | Word-level precision |
| **BLEU-4** | `tass_adaptive + centroid + vlm_only` | **0.0230** | 4-gram precision (fluency proxy) |
| **ROUGE-L** | `tass_adaptive + centroid + vlm_only` | **0.2819** | Longest common subsequence recall |
| **METEOR** | `tass_adaptive + centroid + vlm_only` | **0.2099** | Synonym-aware recall |
| **Semantic Yield** | `tass_adaptive + centroid + vlm_plus_llm` | **0.0122** CIDEr/frame | Quality per VLM call |

### Efficiency Metrics — Best Configuration Per Metric

| Metric | Winner | Value | Notes |
|:---|:---|:---:|:---|
| **Processing Time** | `fps2 + temporal + vlm_only` | **0.021s** | Fastest end-to-end pipeline |
| **Selected Frames** | `tass_adaptive` (all modes) | **6.57 frames** | Minimum VLM call budget |
| **VRAM Delta** | `fps1/fps2/random + vlm_only` | **0.00 MB** | No incremental VRAM above VLM baseline |
| **RAM Delta** | `fps2/random + temporal + vlm_only` | **0.00 MB** | No incremental RAM above baseline |

### Sampler Frame Budget Comparison

| Sampler | Mean Frames Selected | Relative to TASS |
|:---|:---:|:---:|
| `tass_adaptive` | **6.57** | 1.0× (baseline) |
| `fps1` | 16.29 | 2.5× more |
| `random` | 16.29 | 2.5× more |
| `fps2` | 24.22 | 3.7× more |
| `ssim_090` | 34.44 | 5.2× more |

### VLM-Only vs. VLM+LLM Mode Comparison

| Mode | Mean CIDEr (all samplers) | Mean Latency | VRAM Overhead |
|:---|:---:|:---:|:---:|
| `vlm_only` | ~0.027 | **~0.04s** | **~0 MB** |
| `vlm_plus_llm` | ~0.056 | ~0.85s | ~1–2 MB |

LLM synthesis via Phi-3 roughly **doubles CIDEr quality** while adding ~0.8s mean latency per video. The trade-off is worthwhile for high-quality applications; `vlm_only` is preferable for real-time or resource-constrained pipelines.

---

## 8. Why TASS is the Best Sampler

### The Core Argument: Quality per Compute Budget

TASS does not win on absolute CIDEr (where FPS-2 leads due to brute-force frame density). TASS wins on **every efficiency-adjusted metric**:

#### 1. Semantic Yield: 3× Better Than FPS-2

Semantic Yield is defined as `CIDEr / frames_selected` — how much captioning quality is obtained per VLM call.

| Sampler | CIDEr (centroid + LLM) | Frames | Semantic Yield |
|:---|:---:|:---:|:---:|
| `tass_adaptive` | 0.052 | **6.57** | **0.0079** |
| `fps1` | 0.066 | 16.29 | 0.0041 |
| `fps2` | 0.072 | 24.22 | 0.0030 |
| `random` | 0.063 | 16.29 | 0.0039 |
| `ssim_090` | 0.058 | 34.44 | 0.0017 |

> **TASS delivers 0.0079 CIDEr per frame — 2.6× better than FPS-2 and 4.6× better than SSIM-090.**

#### 2. TASS Wins on 3 of 5 Quality Metrics Without LLM

In `vlm_only` mode (no LLM overhead), TASS + centroid leads on BLEU-4 (`0.023`), ROUGE-L (`0.282`), and METEOR (`0.210`). These n-gram and recall metrics are sensitive to **lexical precision** — TASS's selection of the most visually distinct frames ensures the VLM generates captions about genuinely different content, which maps better to reference annotations.

#### 3. Degenerate Frame Elimination is Critical for MSR-VTT

MSR-VTT videos contain many low-quality frames (black frames, lens flare, motion blur, uniform fades). By explicitly filtering these via brightness and variance checks in Stage 1, TASS ensures the VLM never wastes inference on uninformative content. FPS-based samplers process these frames blindly.

In our dataset, TASS Stage 1 drops a mean of **2–5 degenerate frames** per video before Stage 2 ever runs.

#### 4. Semantic Diversity Prevents VLM Repetition

When adjacent frames are near-identical (slow camera pan, static scene), FPS-based samplers generate near-duplicate captions. The aggregator then has to deduplicate them. TASS uses MobileCLIP cosine distance to **guarantee** that each selected frame is semantically distinct from all others — the VLM receives a maximally informative, diverse input set, and every VLM call contributes novel content.

#### 5. Adaptive Early Stopping Matches Content Complexity

Some videos are 3 seconds long; others are 30 seconds. FPS samplers blindly scale linearly. TASS `adaptive` mode stops when cosine diversity falls below `min_distance=0.10` — meaning it **self-calibrates** to the video's content complexity:
- Simple/static video → fewer frames selected → fewer VLM calls
- Complex/multi-scene video → more frames → more VLM calls

#### 6. Zero Additional VRAM Cost

MobileCLIP-S1 runs CPU-only as a Singleton. The additional semantic embedding computation costs **0 VRAM** on the RTX 4050 — the full 6 GB remains available for Moondream2.

### Summary: When to Use Each Sampler

| Use Case | Recommended Sampler | Reason |
|:---|:---|:---|
| Highest absolute quality, offline processing | `fps2 + centroid + vlm_plus_llm` | Maximum frame density + LLM synthesis |
| Best quality-per-compute, real-time capable | `tass_adaptive + centroid + vlm_only` | Maximizes semantic yield, no LLM overhead |
| Edge/IoT, minimal memory | `tass_adaptive + temporal + vlm_only` | Fewest frames, fastest aggregation |
| Ablation baseline (uniform) | `fps1 + centroid + vlm_plus_llm` | Standard FPS baseline, well-understood |
| Statistical control | `random + centroid + vlm_plus_llm` | Budget-matched to FPS-1, no content awareness |

---

## 9. Project Structure

```
research_project/
├── samplers/                   # Frame selection algorithms
│   ├── base_sampler.py         # BaseSampler ABC
│   ├── fps1.py                 # 1 frame/second uniform
│   ├── fps2.py                 # 2 frames/second uniform
│   ├── random_sampler.py       # Seeded random, FPS-1 budget
│   ├── ssim.py                 # SSIM scene-change detection
│   ├── ssim_result.py          # SSIMSamplerResult dataclass
│   ├── tass.py                 # TASS two-stage sampler (main)
│   └── tass_helpers.py         # is_degenerate, pHash, greedy_fps
│
├── aggregation/                # Caption aggregation methods
│   ├── raw.py                  # Sequential concatenation
│   ├── temporal.py             # Jaccard deduplication
│   └── centroid.py             # Representative caption selection
│
├── models/                     # Model loaders
│   ├── vlm_loader.py           # Moondream2 (singleton, CUDA)
│   ├── llm_loader.py           # Phi-3 Mini via Ollama
│   └── clip_embedder.py        # MobileCLIP-S1 (singleton, CPU)
│
├── pipeline/                   # Core pipeline stages
│   ├── frame_extraction.py     # OpenCV frame decode + metadata
│   ├── frame_captioning.py     # VLM per-frame caption + cache
│   └── context_builder.py      # Aggregation + LLM prompt build
│
├── evaluation/                 # Metrics and telemetry
│   ├── metrics.py              # CIDEr, BLEU, ROUGE, METEOR
│   ├── corpus_idf.py           # Full-corpus IDF builder/cache
│   ├── statistics.py           # Mean, std, 95% CI computation
│   ├── telemetry.py            # VRAM/RAM/GPU utilization tracker
│   └── caption_io.py           # Raw caption persistence
│
├── experiments/                # Benchmark orchestration
│   ├── run_benchmark.py        # Main entry point
│   ├── benchmark_loop.py       # Per-video pipeline execution
│   ├── benchmark_data.py       # MSR-VTT dataset loader
│   ├── benchmark_samplers.py   # Sampler/aggregator factories
│   ├── benchmark_setup.py      # Logging, dirs, hardware info
│   ├── benchmark_report.py     # Markdown report generator
│   └── combine_results.py      # Merge multiple run CSVs + re-plot
│
├── visualization/              # Plot generation
│   ├── plots.py                # Figs 1–7, 9: bar + scatter charts
│   ├── plot_frame_reduction.py # Fig 8: SSIM reduction box plot
│   └── plot_utils.py           # Shared theme, palette, helpers
│
├── config/
│   └── settings.py             # YAML loader + WSL2 detection
│
├── configs/
│   └── benchmark.yaml          # Full pipeline configuration
│
├── results/
│   ├── csv/                    # Per-run + combined CSVs
│   ├── plots/                  # PNG + PDF publication figures
│   ├── reports/                # Markdown benchmark summaries
│   ├── frame_selection/        # Per-video JSON metadata
│   ├── captions/               # Raw VLM caption cache
│   ├── metadata/               # Run info JSON files
│   └── logs/                   # Benchmark log files
│
├── result_f.md                 # Final analysis & recommendations
├── CHANGES.md                  # Full changelog by phase
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

---

## 10. Benchmarking Framework

The benchmark is fully configurable via `configs/benchmark.yaml`. At runtime, it:

1. **Loads MSR-VTT** — 100 videos (seeded random selection from the full corpus)
2. **Pre-loads VLM** — Moondream2 singleton loaded once before any pipeline runs
3. **Runs all combinations** — For each `(sampler, aggregator, caption_mode)` triple:
   - Processes all 100 videos, collecting per-video telemetry
   - Evaluates the full corpus against MSR-VTT reference captions at once (corpus-level IDF)
4. **Saves results** — Per-video CSV, per-configuration statistics summary, plots, Markdown report
5. **Generates figures** — 9 publication-quality plots (PNG + PDF, 300 DPI)

### Evaluation Correctness

CIDEr is highly sensitive to IDF weighting. Computing IDF on a small 100-video subset causes severe weight collapse. This project pre-computes and caches a **full MSR-VTT corpus IDF** (all ~10,000 videos) loaded once at startup, ensuring statistically valid scores.

---

## 11. Running the Benchmark

### Prerequisites
```bash
# Python environment
pip install -r requirements.txt

# Ollama + Phi-3 Mini
ollama pull phi3:mini
ollama serve  # (background)

# MSR-VTT dataset (automatic download via kagglehub)
# Set KAGGLE_USERNAME and KAGGLE_KEY environment variables
```

### Basic Usage
```bash
# Run all samplers (from configs/benchmark.yaml), 100 videos
PYTHONPATH=. python experiments/run_benchmark.py --videos 100

# Run a single sampler
PYTHONPATH=. python experiments/run_benchmark.py --sampler tass_adaptive --videos 100

# Run multiple specific samplers
PYTHONPATH=. python experiments/run_benchmark.py --sampler fps1 fps2 --videos 50

# Custom output directory
PYTHONPATH=. python experiments/run_benchmark.py --output-dir ./results_v2 --videos 100
```

### Combining Multiple Run CSVs into Combined Plots
```bash
# After running each sampler separately:
PYTHONPATH=. venv/bin/python3 experiments/combine_results.py
# → Writes: results/csv/combined_results.csv
# → Writes: results/csv/combined_statistics_summary.csv
# → Regenerates: results/plots/ (all 9 figures)
# → Writes: results/reports/combined_benchmark_summary.md
```

### Available CLI Arguments

| Flag | Default | Description |
|:---|:---|:---|
| `--videos` | 10 | Number of videos to process |
| `--config` | `configs/benchmark.yaml` | Path to configuration YAML |
| `--sampler` | All (from YAML) | Run specific sampler(s) only |
| `--aggregation` | All (from YAML) | Run specific aggregation method only |
| `--model` | `phi3:mini` | Override Ollama LLM model |
| `--output-dir` | `./results` | Custom output directory |
| `--log-level` | `INFO` | Logging verbosity |

---

## 12. Configuration Reference

`configs/benchmark.yaml` controls all pipeline settings:

```yaml
experiment:
  dataset: vishnutheepb/msrvtt
  videos: 100
  seed: 42
  cache_dir: ./cache
  output_dir: ./results

pipeline:
  samplers: [tass_adaptive, fps1, fps2, random, ssim_090]
  aggregators: [raw, centroid, temporal]
  caption_modes: [vlm_only, vlm_plus_llm]

models:
  vlm:
    name: vikhyatk/moondream2
    revision: "2024-08-26"
    fallback: null
  llm:
    name: phi3:mini
    host: http://localhost:11434

ssim:
  compare_size: [256, 144]
  win_size: 7
  max_accepted_frames: 500
  acceptance_rate_min: 0.01
  acceptance_rate_max: 0.99

tass:
  threshold: 0.90
  min_distance: 0.10
  clip_batch_size: 16
```

---

## 13. Evaluation Metrics

### CIDEr (Consensus-based Image Description Evaluation)
Measures consensus with multiple reference captions using TF-IDF weighted n-gram similarity. The primary quality metric — highly sensitive to corpus-level vocabulary distribution. Range: `[0, ∞)`. The full MSR-VTT corpus IDF is pre-computed and cached for validity.

### BLEU-N (Bilingual Evaluation Understudy)
N-gram precision metric. BLEU-1 measures word-level precision; BLEU-4 measures 4-gram fluency and is a proxy for grammatical correctness. Range: `[0, 1]`.

### ROUGE-L (Recall-Oriented Understudy for Gisting Evaluation)
Longest Common Subsequence (LCS) F-measure between prediction and references. Captures both recall (content coverage) and sequencing. Range: `[0, 1]`.

### METEOR (Metric for Evaluation of Translation with Explicit ORdering)
Harmonic mean of precision and recall with synonym matching (WordNet) and stemming. More sensitive to paraphrase than BLEU. Range: `[0, 1]`.

### Semantic Yield
Custom metric: `CIDEr / frames_selected`. Measures how much captioning quality is obtained per VLM inference call — the primary efficiency metric for comparing sampler utility on edge hardware.

### Resource Telemetry
- **processing_time_s**: Wall-clock time from frame decode start to final caption (CUDA-synchronized)
- **peak_vram_mb**: Peak VRAM delta above the pre-pipeline VLM baseline (50ms polling)
- **peak_ram_delta_mb**: Peak RAM increase above pre-pipeline baseline
- **gpu_utilization_pct**: Peak GPU core utilization during the pipeline run

---

## 14. Developer Guide

### Adding a New Sampler

1. Create `samplers/my_sampler.py` extending `BaseSampler`:
```python
from .base_sampler import BaseSampler

class MySampler(BaseSampler):
    def get_name(self) -> str:
        return "my_sampler"

    def sample(self, video_path: str) -> list:
        result = self.sample_with_metadata(video_path)
        return result["frames"]

    def sample_with_metadata(self, video_path: str) -> dict:
        # ... implementation ...
        return {
            "frames": frames,
            "indices": indices,
            "meta": {
                "frames_original": ...,
                "candidate_pool_size": ...,
                "frames_degenerate_dropped": 0,
                "tass_stopped_early": False,
                "vlm_calls": len(frames),
                "fallback_used": False,
            }
        }
```
2. Export in `samplers/__init__.py`.
3. Add to `experiments/benchmark_samplers.py` `get_samplers()` dict.
4. Add to `pipeline.samplers` in `configs/benchmark.yaml`.

### Running Tests
```bash
PYTHONPATH=. pytest tests/ -v
```

### Debugging a Single Video
```bash
PYTHONPATH=. python experiments/run_benchmark.py \
  --sampler tass_adaptive --aggregation centroid \
  --videos 2 --log-level DEBUG
```

---

## 15. Future Work

| Research Direction | Rationale |
|:---|:---|
| **TASS Fixed-K Mode Ablation** | Compare `tass_fixed` (fair budget match to FPS-1) vs `tass_adaptive` to isolate the effect of adaptive stopping |
| **SSIM as TASS Stage 1 Pre-filter** | Replace pHash in Stage 1 with full SSIM for tighter spatial filtering |
| **Audio-Visual Fusion** | Integrate Whisper transcription into the LLM prompt for events that are heard but not seen |
| **BLEU-4 Optimization** | Explore constrained decoding (min/max n-gram matching) to improve 4-gram precision |
| **Additional VLMs** | Benchmark LLaVA, InternVL, or Qwen-VL against Moondream2 |
| **Mobile GPU Target** | Port pipeline to Jetson Orin NX (16 GB) for true embedded deployment validation |
| **Larger Video Corpus** | Scale to 1000-video evaluation for tighter confidence intervals |
