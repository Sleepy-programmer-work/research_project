# Changes Log — TASS Benchmark Framework

This file documents all critical bug fixes, architectural changes, and research-phase milestones applied to the video captioning benchmark framework, in reverse-chronological order.

---

## Phase 10 — SSIM Removal & Standalone pHash Benchmark Integration (July 2026)

### 1. `samplers/phash_sampler.py` [NEW] — Standalone pHash Benchmark Sampler
- **Feature**: Extracted TASS Stage 1's pHash scene-change filtering as an independent, standalone benchmark sampler (`PHashSampler`).
- **Pipeline**: `Video → Frame Extraction → Degenerate Filter → pHash Filtering → Selected Frames`.
- **Implementation**: Reuses existing `get_phash()`, `phash_distance()`, and `is_degenerate()` functions from `tass_helpers.py` without code duplication. Operates at ~10 effective FPS via stride-based frame reading, matching TASS Stage 1 behavior.

### 2. Complete Removal of SSIM
- **Refactoring**: Searched and removed all remaining SSIM references across samplers, experiment runners, configuration files, visualization scripts, tests, requirements, and documentation.
- **Deletions**: Deleted `samplers/ssim.py`, `samplers/ssim_result.py`, `tests/test_ssim_sampler.py`, and `tests/test_ssim_sampler_invariants.py`. Removed `scikit-image` from `requirements.txt`.
- **Cleaned Data**: Deleted obsolete SSIM cached captions and frame-selection JSONs; removed SSIM rows from all benchmark CSVs and report summaries.

### 3. `visualization/plot_frame_reduction.py` — Dynamic Reduction Percentage Computation
- **Fix**: Updated `_load_records()` to build a map of `video_id -> original_frame_count` across all JSON files and compute `reduction_pct` dynamically when `null`.
- **Effect**: Enabled Fig 8 box plot to load and visualize all 5 benchmark samplers (`fps1`, `fps2`, `random`, `phash`, `tass_adaptive`) together in a single plot.

### 4. Consolidated Benchmark Run & Documentation Updates
- Executed 100-video evaluation run for `phash` and ran `experiments/combine_results.py` to regenerate all 8 combined publication figures, statistics CSVs, and markdown reports.
- Updated `README.md` and `result_f.md` with final consolidated benchmark numbers and comparative insights across all 5 frame sampling techniques.

---

## Phase 9 — Infrastructure Fixes & TASS Sampler Activation (July 2026)

### 1. `pipeline/context_builder.py` — Empty Captions Aggregation Guard
- **Bug**: If all frames in a video failed captioning, the empty `captions` list propagated as an empty string to the metric scorer, corrupting per-video rows.
- **Fix**: Added `if not captions:` guard returning `"[no captions generated]"` before `aggregator.aggregate()`, ensuring the metric scorer always receives a valid prediction string.

### 2. `evaluation/telemetry.py` — VRAM Delta Peak Initialization Fix
- **Bug**: `PeakResourceTracker` initialized `_peak_vram_mb` to `0.0` in `__enter__()`. Because baseline VRAM is non-zero (VLM pre-loaded), the first poll always reported absolute VRAM rather than the incremental pipeline delta.
- **Fix**: Initialized `_peak_vram_mb` to `self._baseline_vram_mb` in `__enter__()`, ensuring peak tracking is strictly relative to the pre-pipeline baseline level.

### 3. `configs/benchmark.yaml` — Activation of `tass_adaptive` Sampler
- **Change**: Added `tass_adaptive` to the `pipeline.samplers` list to execute and evaluate the adaptive early-stopping TASS configuration.

---

## Phase 8 — Benchmark Pipeline Quality & Robustness Fixes (June 2026)

### 1. `samplers/__init__.py` & `experiments/run_benchmark.py` — DSISSampler Exclusion
- **Bug**: The inactive `DSISSampler` stub (returns empty list `[]`) was exported in the sampler registry `__all__`, causing potential silent failures.
- **Fix**: Removed `DSISSampler` from exports and removed its direct import from `run_benchmark.py`.

### 2. `samplers/` & `pipeline/frame_extraction.py` — True Video Frame Index Tracking
- **Bug**: `frame_extraction.py` saved sequential dummy indices `[0, 1, 2, ...]` instead of the actual frame indices selected by the sampler, causing incorrect temporal metadata.
- **Fix**: Extended the sampler API by adding a default `get_last_sampled_indices()` method to `BaseSampler`. Updated `FPS1Sampler`, `FPS2Sampler`, and `RandomSampler` to track and return actual video frame indices. Updated `frame_extraction.py` to save these actual indices via `sampler.get_last_sampled_indices()`.

### 3. `samplers/` [REMOVED] — Historical: Legacy Sampler Fallback Scores & Metadata Fields
- **Bug**: When the legacy perceptual sampler fell back or failed, it returned `0.0` scores (`[0.0] * n`), leading to silent metrics corruption and misleading data.
- **Fix**: Changed fallback scores from `0.0` to `None` to clearly represent the absence of scores. *(Note: This sampler has since been replaced by pHash-based perceptual filtering.)*

### 4. `pipeline/frame_captioning.py` & `configs/benchmark.yaml` — Cache Key VLM Model & Revision Pinning
- **Bug**: The VLM captioning cache key did not include the model name or revision. If the model changed, stale cached captions were silently reused.
- **Fix**: Updated the cache key format to: `{video_id}_{method}_{model_safe}_{revision_safe}.json`. Added `vlm_model_id` and `vlm_revision` properties to `config/settings.py` enforcing a hardcoded default revision of `"2024-08-26"`. Pinned `revision: "2024-08-26"` under `models.vlm` in `configs/benchmark.yaml`.

### 5. `samplers/` [REMOVED] — Historical: Legacy Sampler Frame Reduction Denominator Correction
- **Bug**: The legacy perceptual sampler's frame reduction percentage formula used the physically decoded frame count (`actual_total`) as the denominator, skewing results for truncated videos.
- **Fix**: Changed the denominator to use the video container's metadata frame count (`total_frames_meta`). *(Note: This sampler has since been replaced by pHash-based perceptual filtering.)*

### 6. `models/clip_embedder.py` — Cosine Distance Norm Assertion
- **Bug**: Cosine distance in TASS Stage 2 assumes L2-normalized embeddings but had no runtime assertion checking this invariant.
- **Fix**: Added `assert np.allclose(norms, 1.0, atol=1e-5)` after pooling embeddings to guarantee the L2-norm invariant holds.

### 7. `aggregation/temporal.py` & `aggregation/centroid.py` — NLTK Path Robustness
- **Bug**: A hardcoded NLTK path (`./venv/nltk_data`) caused crashes on environments where the virtual environment name differed.
- **Fix**: Replaced the hardcoded path with a robust `_ensure_nltk_resources()` helper using standard NLTK search paths, dynamically downloading resources to default folders.

### 8. `samplers/tass.py` — TASS Stage 2 Greedy FPS Efficiency
- **Bug**: Greedy FPS selection in TASS Stage 2 had O(M²k) time complexity due to redundant list membership checks on `selected_pool_indices`.
- **Fix**: Added a `selected_pool_set` hash set alongside the list to enable O(1) lookup, reducing the overall complexity to O(Mk).

---

## Phase 7 — TASS & MobileCLIP-S1 Integration (June 2026)

### 1. `samplers/tass.py` [NEW]
Implemented the core research contribution: Two-Stage Adaptive Semantic Sampling (TASS).
- **Stage 1**: Streaming degenerate frame purge (numpy-based brightness/variance checks) followed by perceptual hash (pHash) scene-change detection operating at ~10 effective FPS (every 3rd frame of a 30 FPS source).
- **Stage 2**: Generates 512-d embeddings via MobileCLIP-S1 and performs Greedy Farthest-Point Sampling (FPS) to select the K most semantically diverse frames.
- **Modes**: `fixed` (fair budget comparison to FPS-1 baseline) and `adaptive` (early stopping based on a cosine distance floor of `min_distance`).

### 2. `models/clip_embedder.py` [NEW]
Implemented `MobileCLIPEmbedder` using a CPU-only Singleton pattern.
- **CPU-only Constraint**: Ensures zero GPU VRAM consumption to isolate model loading alongside Moondream2 on the RTX 4050 6 GB GPU.
- **Micro-batching**: Batches ≤16 frames to stay within WSL2 12 GB memory limits.

### 3. `experiments/run_benchmark.py`
- Integrated `TASSSampler` into the runner pipeline and sampler mapping.
- Added TASS-specific columns: `tass_candidate_pool`, `tass_degenerate_dropped`, `tass_stopped_early`, `vlm_calls_saved_pct`, and `semantic_yield`.

### 4. `samplers/__init__.py` & `requirements.txt`
- Exposed `TASSSampler` in the samplers API.
- Added `open-clip-torch>=2.24.0` dependency to support MobileCLIP.

---

## Phase 6 — VLM Singleton Loading & Stability Patches (May 2026)

### 1. `experiments/run_benchmark.py` & `models/vlm_loader.py`
- **Bug**: Benchmark crashed on configuration transitions with a `<class 'HfConfig'>` mismatch error when the model was re-instantiated mid-process.
- **Fix**: Enforced strict **Singleton VLM Loading** — added an explicit `vlm_loader.load()` before the nested loops. Removed lazy loading from `VLMLoader.generate_captions()`, instead raising `RuntimeError` if the model isn't pre-loaded.

### 2. `models/vlm_loader.py` — Monkey-Patch Revert & Library Downgrade
- **Bug**: Monkey-patched fallback logic (catching exceptions and falling back to `BlipProcessor`) created caching conflicts causing fatal `HfConfig` double-import collisions.
- **Fix**: Reverted to standard `AutoModelForCausalLM.from_pretrained(..., trust_remote_code=True)`. Fixed the underlying compatibility issue by pinning `transformers==4.46.3`.

### 3. Revision Pinning & Loading Sequence
- **Bug**: Dynamic Hugging Face updates mid-run triggered duplicate `HfConfig` compilations.
- **Fix**: Added `revision="2024-08-26"` revision pins to both model and tokenizer. Enforced strict loading order (model first, then tokenizer).

---

## Phase 5 — Post-Audit Critical Bug Fixes (May 2026)

### 1. `evaluation/telemetry.py` — VRAM Measurement Fix
- **Bug**: `PeakResourceTracker` reported **absolute GPU-wide memory usage** (`nvmlDeviceGetMemoryInfo().used`) rather than pipeline-specific allocation, making VRAM comparisons meaningless (every row reported the identical value ~5432 MB).
- **Fix**: Added `_baseline_vram_mb` snapshot in `__enter__()`. `__exit__()` now reports `peak_vram_mb = max(0, peak_observed - baseline)`.

### 2. `pipeline/context_builder.py` — Aggregation Bypass in `vlm_plus_llm` Mode
- **Bug**: In `build_context()`, `aggregator.aggregate(captions)` was computed but **discarded** for `vlm_plus_llm` mode. The LLM prompt was built from raw `captions` directly — meaning centroid, temporal, and raw aggregation all produced the **identical LLM prompt**, making all aggregation comparisons scientifically invalid.
- **Fix**: The LLM prompt now uses `aggregated_caption` as the `Visual Summary` field.

---

## Phase 4 — Evaluation Loop Restructure (Corpus-Level Scoring) (April 2026)

### 1. `experiments/run_benchmark.py` [RESTORED]
- **Bug**: `pycocoevalcap` warnings and incorrect CIDEr TF-IDF scores when evaluating video-by-video within nested loops.
- **Fix**: Restructured loop hierarchy: configuration loops (Sampler → Aggregator → Mode) on the outside, video dataset loop on the inside. Extracted metrics computation to run once per configuration over the full corpus, then joined per-video telemetry with batch-evaluated NLP metrics.

---

## Phase 3 — CIDEr Evaluation & Scoring Correctness (April 2026)

### 1. `evaluation/corpus_idf.py` [NEW]
- Implemented a complete, cached full-corpus IDF builder for MSR-VTT, preventing IDF weight collapse on small evaluation subsets.
- Caches processed IDF weights as a pickle file under `results/cache/`, preventing repeated overhead on subsequent runs.

### 2. `evaluation/metrics.py` [REWRITTEN]
- Created `CustomCiderScorer` subclassing `CiderScorer` to bypass standard `ctest` size assertion checks.
- Direct injection of precomputed full-corpus IDF dictionaries into the scorer.
- Overhauled `build_gts_res` to compile predictions/references into correct `image_id`/`caption` format.

### 3. `experiments/run_benchmark.py` [REWRITTEN]
- Implemented `build_reference_captions` with dynamic schema inspection for Hugging Face dataset versions.
- Integrated one-time startup loading of the precomputed full-corpus IDF file.

---

## Phase 2 — Telemetry & Resource Patches (March 2026)

### 1. `config/settings.py` — WSL2-Specific Resource Configuration
- Implemented automatic WSL2 environment detection via `/proc/sys/fs/binfmt_misc/WSLInterop`.
- Added WSL2-specific constants: `MAX_RAM_BUDGET_GB=10`, `FRAME_BATCH_SIZE=2`, and capped `DATALOADER_WORKERS`.
- Disabled `torch.backends.cudnn.benchmark` and enabled `deterministic` mode.

### 2. `evaluation/telemetry.py` — Polling Interval & RAM Delta
- Rewrote `PeakResourceTracker` as a context manager. Shrunk `POLL_INTERVAL_S` to `0.05` (50ms). Rewrote RAM monitoring to track `delta_mb` (incremental load) instead of absolute RSS.

### 3. `evaluation/statistics.py` — CI Clamping
- Added `compute_ci95` helper using `scipy.stats`. Added zero-std and low-count guards. Implemented boundary clamping for probability metrics to `[0, 1]` and CIDEr/timing/memory to `[0, ∞)`.

---

## Phase 1 — Prompt & Evaluation Patches (March 2026)

### 1. `pipeline/context_builder.py` — LLM Verbosity Constraint
- Replaced relaxed task instruction with a structured `SYSTEM_PROMPT` enforcing exactly one sentence, ≤15 words, literal action focus, no conversational filler.

### 2. `evaluation/metrics.py` — `pycocoevalcap` Dictionary Key Mismatch Fix
- Added explicit `str(k)` casting for all prediction and ground-truth dictionary keys before passing to the COCO evaluator.

### 3. `experiments/run_benchmark.py` — VRAM/RAM Telemetry Initial Fix
- Implemented the first threaded polling tracker class utilizing `pynvml` and `psutil`.

### 4. End-to-End Pipeline Timing Fix
- Hoisted OpenCV frame decoding, transcription, and captioning above the aggregation loop, ensuring `processing_time_s` measures start-to-finish duration.

### 5. `evaluation/statistics.py` — `NaN` Protection in Statistics
- Added safe standard deviation checks returning the mean when `std == 0.0` to prevent division-by-zero.
