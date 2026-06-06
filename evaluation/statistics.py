import pandas as pd
import numpy as np
from scipy import stats as scipy_stats
import warnings

PROBABILITY_METRICS = {"bleu1", "bleu4", "rouge_l", "meteor"}
CIDER_METRIC = {"cider"}

def compute_ci95(data: np.ndarray, col_name: str) -> tuple[float, float]:
    """
    Compute 95% confidence interval for a metric column.
    Clamps bounds to valid range for probability and CIDEr metrics.
    Returns (ci_lower, ci_upper).
    """
    n = len(data)
    mean = float(np.mean(data))
    std  = float(np.std(data, ddof=1)) if n > 1 else 0.0

    # Guard: zero std → CI equals the mean (no spread)
    if std == 0.0 or n < 2:
        return mean, mean

    se = std / np.sqrt(n)
    try:
        ci_lower, ci_upper = scipy_stats.t.interval(0.95, df=n - 1, loc=mean, scale=se)
    except Exception:
        ci_lower, ci_upper = mean, mean

    col = col_name.lower()

    # Clamp probability metrics to [0, 1]
    if any(m in col for m in PROBABILITY_METRICS):
        ci_lower = max(0.0, ci_lower)
        ci_upper = min(1.0, ci_upper)

    # Clamp CIDEr to [0, ∞) — cannot be negative
    if any(m in col for m in CIDER_METRIC):
        ci_lower = max(0.0, ci_lower)

    # Clamp processing time and memory to [0, ∞)
    if any(m in col for m in {"processing_time", "vram", "ram", "util"}):
        ci_lower = max(0.0, ci_lower)

    return float(ci_lower), float(ci_upper)

def compute_statistics(csv_path: str, out_path: str):
    df = pd.read_csv(csv_path)
    if df.empty:
        return
        
    group_cols = ["sampling_method", "aggregation_method", "caption_mode"]
    metrics = [
        "processing_time_s", "peak_vram_mb", "peak_ram_delta_mb", "gpu_utilization_pct", 
        "cider", "bleu1", "bleu4", "rouge_l", "meteor"
    ]
    
    results = []
    
    for name, group in df.groupby(group_cols):
        res = {
            "sampling_method": name[0],
            "aggregation_method": name[1],
            "caption_mode": name[2],
            "count": len(group)
        }
        for m in metrics:
            if m not in group.columns:
                continue
            data = group[m].dropna().to_numpy()
            if len(data) == 0:
                continue
                
            res[f"{m}_mean"] = float(np.mean(data))
            res[f"{m}_median"] = float(np.median(data))
            res[f"{m}_std"] = float(np.std(data, ddof=1)) if len(data) > 1 else 0.0
            res[f"{m}_min"] = float(np.min(data))
            res[f"{m}_max"] = float(np.max(data))
            
            ci_lower, ci_upper = compute_ci95(data, m)
            res[f"{m}_ci95_lower"] = ci_lower
            res[f"{m}_ci95_upper"] = ci_upper
            
        results.append(res)
        
    out_df = pd.DataFrame(results)
    out_df.to_csv(out_path, index=False)
