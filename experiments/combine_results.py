"""
experiments/combine_results.py — Utility to combine all 5 sampler benchmark runs,
compute aggregated statistics, regenerate all figures, and output a combined report.
"""
import json
import logging
from pathlib import Path
import pandas as pd

from evaluation.statistics import compute_statistics
from visualization.plots import generate_plots
from experiments.benchmark_report import generate_report

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("combine_results")

def main():
    csv_dir = Path("results/csv")
    out_dir = Path("results")
    
    # Find all results_*.csv files (excluding any combined files)
    csv_files = [f for f in csv_dir.glob("results_*.csv") if "combined" not in f.name]
    
    if not csv_files:
        logger.error("No benchmark results CSV files found to combine!")
        return
        
    logger.info(f"Found {len(csv_files)} benchmark CSV files to combine:")
    for f in csv_files:
        logger.info(f"  - {f.name}")
        
    dfs = []
    for f in csv_files:
        logger.info(f"Reading {f.name}...")
        dfs.append(pd.read_csv(f))
        
    combined_df = pd.concat(dfs, ignore_index=True)
    
    combined_csv_path = csv_dir / "combined_results.csv"
    combined_df.to_csv(combined_csv_path, index=False)
    logger.info(f"Saved combined results to {combined_csv_path}")
    
    # Compute combined statistics
    combined_stats_path = csv_dir / "combined_statistics_summary.csv"
    compute_statistics(str(combined_csv_path), str(combined_stats_path))
    logger.info(f"Saved combined statistics to {combined_stats_path}")
    
    # Generate combined plots
    logger.info("Generating combined plots from the unified dataset...")
    generate_plots(
        str(combined_csv_path),
        out_dir=out_dir,
        frame_selection_dir=out_dir / "frame_selection"
    )
    logger.info("Combined plots generated successfully in results/plots/.")
    
    # Load metadata for combined run report
    meta_dir = Path("results/metadata")
    meta_files = sorted(list(meta_dir.glob("run_info_*.json")))
    metadata = {}
    if meta_files:
        try:
            with open(meta_files[-1]) as f:
                metadata = json.load(f)
        except Exception as exc:
            logger.warning(f"Could not load metadata from {meta_files[-1].name}: {exc}")
            
    # Fallback or update metadata fields
    metadata["date"] = "Combined Run (All 5 Samplers)"
    metadata["videos"] = len(combined_df["video_id"].unique())
    
    # Save combined metadata
    combined_meta_path = meta_dir / "combined_run_info.json"
    combined_meta_path.write_text(json.dumps(metadata, indent=2))
    logger.info(f"Saved combined run info to {combined_meta_path}")
    
    # Generate combined report
    combined_report_path = out_dir / "reports" / "combined_benchmark_summary.md"
    generate_report(str(combined_stats_path), str(combined_report_path), metadata)
    logger.info(f"Saved combined report to {combined_report_path}")

if __name__ == "__main__":
    main()
