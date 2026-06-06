import json
import logging
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from config.settings import settings

logger = logging.getLogger(__name__)


def generate_plots(csv_path: str, out_dir: Path = None, frame_selection_dir: Path = None):
    """
    Generate all benchmark figures from the raw per-video CSV.

    Args:
        csv_path:             Path to results_{timestamp}.csv
        out_dir:              Output root directory (plots/ subdirectory is created here).
                              Defaults to settings.experiment.output_dir / plots.
        frame_selection_dir:  Path to results/frame_selection/ directory containing
                              per-video JSON files.  Required for Fig 8.
                              If None, Fig 8 is skipped.
    """
    df = pd.read_csv(csv_path)
    if df.empty:
        return

    if out_dir is None:
        out_dir = Path(settings.experiment.get("output_dir", "./results"))

    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    # Create unified configuration ticks for the X-axis
    df['pipeline_config'] = df['sampling_method'] + ' + ' + df['aggregation_method']
    
    # Establish a clean, professional theme for academic publication
    sns.set_theme(style="whitegrid", context="paper")
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.size': 10,
        'axes.labelsize': 11,
        'axes.titlesize': 12,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'figure.titlesize': 14
    })
    
    # Define a high-contrast theme color scheme
    # Salmon/Coral for baseline, Deep Teal for your framework's enhancements
    palette = {"vlm_only": "#E76F51", "vlm_plus_llm": "#2A9D8F"}
    
    metrics = {
        'cider':             ('fig1_cider',  'Mean CIDEr Score'),
        'bleu4':             ('fig2_bleu4',  'Mean BLEU-4 Score'),
        'processing_time_s': ('fig3_time',   'Average Processing Time (s)'),
        'peak_vram_mb':      ('fig4_vram',   'Peak VRAM Footprint (MB)'),
        'peak_ram_delta_mb': ('fig5_ram',    'Peak RAM Delta Footprint (MB)'),
        'frames_selected':   ('fig6_frames', 'Selected Frame Count'),
    }
    
    # --- Generate Grouped Ablation Study Bar Charts (Figs 1-6) ---
    for metric, (filename, ylabel) in metrics.items():
        if metric not in df.columns:
            continue
            
        plt.figure(figsize=(10, 5), dpi=300)
        
        # Plot side-by-side grouped bars using 'hue' to avoid long x-axis strings
        ax = sns.barplot(
            data=df, 
            x='pipeline_config', 
            y=metric, 
            hue='caption_mode', 
            palette=palette,
            capsize=0.05, 
            errorbar=('ci', 95),
            edgecolor='black',
            linewidth=0.7
        )
        
        plt.xticks(rotation=25, ha='right')
        plt.title(f'Comparative Architectural Metrics: {ylabel}', pad=12, weight='bold')
        plt.xlabel('Upstream Frame Aggregation Pipeline', labelpad=6)
        plt.ylabel(ylabel)
        
        # Aesthetic polishing
        sns.despine(left=True, bottom=False)
        plt.legend(title="Inference Framework", loc="upper right", frameon=True)
        plt.tight_layout()
        
        plt.savefig(plots_dir / f"{filename}.png", dpi=300, bbox_inches='tight')
        plt.savefig(plots_dir / f"{filename}.pdf", bbox_inches='tight')
        plt.close()
        
    # --- Generate Optimized Scatter Pareto Frontier Plot (Fig 7) ---
    if 'processing_time_s' in df.columns and 'cider' in df.columns:
        plt.figure(figsize=(10, 6.5), dpi=300)
        
        # Group by all tracking keys to preserve categorical color styling in scatter points
        agg_df = df.groupby(['sampling_method', 'aggregation_method', 'caption_mode'])[['processing_time_s', 'cider']].mean().reset_index()
        
        # Sort values to ensure programmatic label text anti-collision calculations work sequentially
        agg_df = agg_df.sort_values(by=['processing_time_s', 'cider']).reset_index(drop=True)
        
        sns.scatterplot(
            data=agg_df, 
            x='processing_time_s', 
            y='cider', 
            hue='caption_mode',
            palette=palette,
            s=120,
            edgecolor='black',
            alpha=0.9,
            zorder=3
        )
        
        # Anti-collision layout tracker for text labels
        vertical_stack_registry = {}
        
        for i, row in agg_df.iterrows():
            x = row['processing_time_s']
            y = row['cider']
            
            label_text = f"{row['sampling_method']}_{row['aggregation_method']}"
            if row['caption_mode'] == 'vlm_plus_llm':
                label_text += " (+LLM)"
            
            # Divide processing time into slots to register overlapping clusters
            grid_slot = round(x, 1)
            y_offset_multiplier = 0
            
            if grid_slot in vertical_stack_registry:
                # If points collide on the horizontal timeline, stack labels vertically
                for last_y in vertical_stack_registry[grid_slot]:
                    if abs(y - last_y) < 0.02:
                        y_offset_multiplier += 1
                vertical_stack_registry[grid_slot].append(y)
            else:
                vertical_stack_registry[grid_slot] = [y]
            
            # Annotate with a dynamic layout shift
            plt.annotate(
                label_text, 
                (x, y), 
                xytext=(8, 3 + (y_offset_multiplier * 12)), 
                textcoords='offset points',
                fontsize=8.5,
                alpha=0.85,
                weight='bold' if 'LLM' in label_text else 'normal'
            )
                 
        plt.title('Quality (CIDEr) vs. System Efficiency (Latency Operating Frontier)', pad=15, weight='bold')
        plt.xlabel('Average Pipeline Processing Latency (seconds)', labelpad=6)
        plt.ylabel('Mean Corpus CIDEr Evaluation Score', labelpad=6)
        
        # Give safety bounds so annotations do not cut off at the grid borders
        plt.xlim(-0.5, agg_df['processing_time_s'].max() + 1.5)
        plt.ylim(-0.02, agg_df['cider'].max() + 0.04)
        
        sns.despine()
        plt.legend(title="Framework Configuration", loc="center right", frameon=True)
        plt.tight_layout()
        
        plt.savefig(plots_dir / "fig7_scatter.png", dpi=300, bbox_inches='tight')
        plt.savefig(plots_dir / "fig7_scatter.pdf", bbox_inches='tight')
        plt.close()

    # --- Fig 8: Frame reduction % by SSIM method ---
    if frame_selection_dir is not None:
        plot_frame_reduction(frame_selection_dir=frame_selection_dir, output_dir=plots_dir)


def plot_frame_reduction(frame_selection_dir: Path, output_dir: Path) -> None:
    """
    Fig 8: Frame reduction percentage for each SSIM variant vs. FPS methods.

    FPS-1 and FPS-2 have fixed frame counts determined by video duration, so
    their reduction_pct is None (not written to JSON).  This plot therefore
    shows only SSIM variants, which is intentional — it answers the question:
    "How aggressively does each SSIM threshold reduce frame count compared
    to the original video frame count?"

    Reads per-video JSON files from results/frame_selection/ written by
    _save_frame_selection_meta() in experiments/run_benchmark.py.

    Output:
        results/plots/fig8_frame_reduction.{png,pdf}

    When to call:
        Called automatically from generate_plots() when frame_selection_dir
        is provided.  Also callable standalone for post-hoc regeneration:

        from visualization.plots import plot_frame_reduction
        from pathlib import Path
        plot_frame_reduction(
            frame_selection_dir=Path("results/frame_selection"),
            output_dir=Path("results/plots"),
        )
    """
    if not frame_selection_dir.exists():
        logger.warning(
            f"frame_selection directory not found: {frame_selection_dir} — skipping Fig 8"
        )
        return

    records = []
    for json_file in sorted(frame_selection_dir.glob("*.json")):
        try:
            meta = json.loads(json_file.read_text())
        except (json.JSONDecodeError, OSError) as e:
            logger.warning(f"Could not read {json_file.name}: {e}")
            continue

        if meta.get("reduction_pct") is not None:
            records.append({
                "method":        meta["sampler"],
                "reduction_pct": meta["reduction_pct"],
                "video_id":      meta.get("video_id", json_file.stem),
            })

    if not records:
        logger.warning(
            "No SSIM frame_selection metadata found (reduction_pct is None for "
            "non-SSIM samplers) — skipping Fig 8. "
            "Run with at least one SSIM variant to generate this figure."
        )
        return

    df = pd.DataFrame(records)

    # Sort methods for consistent left-to-right ordering:
    # ssim_085 < ssim_090 < ssim_095 (then any others alphabetically)
    method_order = sorted(
        df["method"].unique(),
        key=lambda m: (
            # SSIM variants first, ordered by threshold ascending
            (0, float(m.split("_")[1]) / 100) if m.startswith("ssim_") else (1, m)
        )
    )
    data_by_method = [
        df[df["method"] == m]["reduction_pct"].values for m in method_order
    ]

    fig, ax = plt.subplots(figsize=(10, 5))

    sns.set_theme(style="whitegrid", context="paper")
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.size': 10,
        'axes.labelsize': 11,
        'axes.titlesize': 12,
    })

    # Grouped box plot: reduction % by method
    bp = ax.boxplot(
        data_by_method,
        labels=method_order,
        patch_artist=True,
        medianprops={"color": "black", "linewidth": 1.5},
    )

    # Colour boxes: SSIM variants get a gradient from coral → teal by threshold
    ssim_colours = ["#E76F51", "#2A9D8F", "#264653"]
    ssim_idx = 0
    for patch, label in zip(bp["boxes"], method_order):
        if label.startswith("ssim_"):
            colour = ssim_colours[ssim_idx % len(ssim_colours)]
            ssim_idx += 1
        else:
            colour = "#ADB5BD"  # neutral grey for non-SSIM methods
        patch.set_facecolor(colour)
        patch.set_alpha(0.75)

    ax.set_ylabel("Frame reduction (%)")
    ax.set_xlabel("Sampling method")
    ax.set_title(
        "Fig 8: Frame Reduction % by Sampling Method\n"
        "(SSIM variants vs. original video frame count)",
        weight="bold",
        pad=12,
    )
    ax.set_ylim(bottom=0)
    sns.despine(ax=ax, left=True, bottom=False)
    plt.tight_layout()

    for fmt in ("png", "pdf"):
        fig.savefig(output_dir / f"fig8_frame_reduction.{fmt}", dpi=300, bbox_inches="tight")

    plt.close(fig)
    logger.info(
        f"Fig 8 saved: frame reduction % "
        f"({len(records)} records across {len(method_order)} methods)"
    )
