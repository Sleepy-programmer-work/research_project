"""
experiments/benchmark_report.py — Markdown report generation from benchmark CSV.
"""
import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger("benchmark")

_RANKING_PRIORITY = ["cider_mean", "bleu1_mean", "rouge_l_mean"]


def generate_report(csv_path: str, report_path: str, metadata: dict) -> None:
    """Write a Markdown benchmark summary from the statistics CSV."""
    df = pd.read_csv(csv_path)
    if df.empty:
        return
    primary_metric, warnings = _select_ranking_metric(df)
    best = df.loc[df[primary_metric].idxmax()]
    fastest = df.loc[df["processing_time_s_mean"].idxmin()]
    vram_eff = df.loc[df["peak_vram_mb_mean"].idxmin()]
    ram_eff = df.loc[df["peak_ram_delta_mb_mean"].idxmin()]

    header = "\n".join(warnings) + "\n\n" if warnings else ""
    content = _format_report(header, metadata, primary_metric, best, fastest, vram_eff, ram_eff, df)
    Path(report_path).write_text(content)


def _select_ranking_metric(df: pd.DataFrame) -> tuple[str, list[str]]:
    warnings = []
    for candidate in _RANKING_PRIORITY:
        if candidate in df.columns and df[candidate].max() > 0.0:
            if candidate != "cider_mean":
                warnings.append(f"> WARNING: CIDEr degenerate. Ranking on `{candidate}`.")
            return candidate, warnings
    raise RuntimeError("All quality metrics are zero — pipeline evaluation failed.")


def _format_report(header, meta, metric, best, fastest, vram_eff, ram_eff, df) -> str:
    m = metric.replace("_mean", "").upper()
    return (
        f"# Benchmark Summary\n{header}"
        f"Generated on: {meta['date']}\n\n"
        f"## Run Metadata\n"
        f"- **Videos**: {meta['videos']} · **Dataset**: {meta['dataset']}\n"
        f"- **GPU**: {meta['gpu']} · **CPU**: {meta['cpu']}\n"
        f"- **VLM**: {meta['vlm']} · **LLM**: {meta['llm']}\n\n"
        f"## Highlights\n"
        f"- **Best ({m})**: {best['sampling_method']} + {best['aggregation_method']} "
        f"({best['caption_mode']}) — {best[metric]:.2f}\n"
        f"- **Fastest**: {fastest['sampling_method']} + {fastest['aggregation_method']} "
        f"({fastest['caption_mode']}) — {fastest['processing_time_s_mean']:.2f}s\n"
        f"- **Min VRAM**: {vram_eff['sampling_method']} + {vram_eff['aggregation_method']} "
        f"({vram_eff['caption_mode']}) — {vram_eff['peak_vram_mb_mean']:.2f} MB\n"
        f"- **Min RAM**: {ram_eff['sampling_method']} + {ram_eff['aggregation_method']} "
        f"({ram_eff['caption_mode']}) — {ram_eff['peak_ram_delta_mb_mean']:.2f} MB\n\n"
        f"## Full Results\n{df.to_markdown()}\n"
    )
