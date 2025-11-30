"""Offline multiscale analysis for optimal scaling policy selection.

This tool analyzes bbox statistics across multiple scaling policies to predict
optimal training scales before running expensive training experiments.

Key features:
- Compares aspect-ratio preserving (AR) and square (SQ) resize policies
- FPN-aware analysis using D-FINE strides [8, 16, 32]
- Cell-based coverage metrics (how many FPN cells per bbox)
- Scoring heuristic to rank policies by detection friendliness
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from tqdm import tqdm

from doc_obj_detect.config import TrainConfig
from doc_obj_detect.data import DatasetFactory, collate_fn
from doc_obj_detect.models import ModelFactory

logger = logging.getLogger(__name__)


# ============================================================================
# Policy Definitions
# ============================================================================


@dataclass
class ScalingPolicy:
    """Definition of a scaling policy for comparison."""

    name: str
    policy_type: Literal["aspect", "square", "stretch"]
    short_side: int | None = None  # For aspect-ratio and stretch policies
    long_cap: int | None = None  # For aspect-ratio policies
    square_size: int | None = None  # For square policies
    stretch_ratio: float | None = None  # For stretch policies (e.g., 1.5, 2.0, 2.5)

    def __str__(self):
        if self.policy_type == "aspect":
            return f"{self.name} (short={self.short_side}, cap={self.long_cap})"
        elif self.policy_type == "stretch":
            return f"{self.name} (short={self.short_side}, stretch={self.stretch_ratio}×)"
        else:
            return f"{self.name} (square={self.square_size}×{self.square_size})"


# Default policy set for comparison (limited to ≤640 for VRAM constraints)
DEFAULT_POLICIES = [
    # Aspect-ratio policies (224-640, every 32 pixels)
    ScalingPolicy("AR-224", "aspect", short_side=224, long_cap=928),
    ScalingPolicy("AR-256", "aspect", short_side=256, long_cap=928),
    ScalingPolicy("AR-288", "aspect", short_side=288, long_cap=928),
    ScalingPolicy("AR-320", "aspect", short_side=320, long_cap=928),
    ScalingPolicy("AR-352", "aspect", short_side=352, long_cap=928),
    ScalingPolicy("AR-384", "aspect", short_side=384, long_cap=928),
    ScalingPolicy("AR-416", "aspect", short_side=416, long_cap=928),
    ScalingPolicy("AR-448", "aspect", short_side=448, long_cap=928),
    ScalingPolicy("AR-480", "aspect", short_side=480, long_cap=928),
    ScalingPolicy("AR-512", "aspect", short_side=512, long_cap=928),
    ScalingPolicy("AR-544", "aspect", short_side=544, long_cap=928),
    ScalingPolicy("AR-576", "aspect", short_side=576, long_cap=928),
    ScalingPolicy("AR-608", "aspect", short_side=608, long_cap=928),
    ScalingPolicy("AR-640", "aspect", short_side=640, long_cap=928),
    # Square policies (224-640, every 32 pixels)
    ScalingPolicy("SQ-224", "square", square_size=224),
    ScalingPolicy("SQ-256", "square", square_size=256),
    ScalingPolicy("SQ-288", "square", square_size=288),
    ScalingPolicy("SQ-320", "square", square_size=320),
    ScalingPolicy("SQ-352", "square", square_size=352),
    ScalingPolicy("SQ-384", "square", square_size=384),
    ScalingPolicy("SQ-416", "square", square_size=416),
    ScalingPolicy("SQ-448", "square", square_size=448),
    ScalingPolicy("SQ-480", "square", square_size=480),
    ScalingPolicy("SQ-512", "square", square_size=512),
    ScalingPolicy("SQ-544", "square", square_size=544),
    ScalingPolicy("SQ-576", "square", square_size=576),
    ScalingPolicy("SQ-608", "square", square_size=608),
    ScalingPolicy("SQ-640", "square", square_size=640),
]


# ============================================================================
# FPN Cell Coverage Analysis
# ============================================================================


@dataclass
class BboxCellStats:
    """FPN cell statistics for a single bbox."""

    cells_s8: float  # Cells on stride-8 feature map
    cells_s16: float  # Cells on stride-16 feature map
    cells_s32: float  # Cells on stride-32 feature map
    best_level: int  # Which stride gives optimal cells (8, 16, or 32)
    best_cells: float  # Number of cells on best level
    good_any: bool  # True if any level has cells in [GOOD_MIN, GOOD_MAX]
    tiny_all: bool  # True if all levels have cells < GOOD_MIN
    huge_all: bool  # True if all levels have cells > GOOD_MAX


# FPN cell thresholds (3-12 cells is sweet spot)
GOOD_MIN_CELLS = 3.0
GOOD_MAX_CELLS = 12.0
OPTIMAL_CELLS = 7.5  # Center of sweet spot


def compute_bbox_cells(
    bbox_h: float, bbox_w: float, strides: list[int] | None = None
) -> BboxCellStats:
    """Compute FPN cell coverage for a bbox.

    For elongated document boxes (text lines, columns), we check BOTH dimensions
    independently. A bbox is detectable if EITHER dimension has good cell coverage.

    Args:
        bbox_h: Bbox height in pixels (after resizing)
        bbox_w: Bbox width in pixels (after resizing)
        strides: FPN strides (default: D-FINE [8, 16, 32])

    Returns:
        BboxCellStats with cell counts and flags
    """
    if strides is None:
        strides = [8, 16, 32]
    # Compute cells for BOTH dimensions on each level
    cells_h = [bbox_h / s for s in strides]
    cells_w = [bbox_w / s for s in strides]

    # For reporting, use the max dimension (most detectable)
    cells = [max(ch, cw) for ch, cw in zip(cells_h, cells_w, strict=False)]
    cells_s8, cells_s16, cells_s32 = cells

    # Find best level (closest to OPTIMAL_CELLS, using max dimension)
    distances = [abs(c - OPTIMAL_CELLS) for c in cells]
    best_idx = np.argmin(distances)
    best_level = strides[best_idx]
    best_cells = cells[best_idx]

    # Flag coverage quality - bbox is good if EITHER dimension works
    good_any = any(GOOD_MIN_CELLS <= ch <= GOOD_MAX_CELLS for ch in cells_h) or any(
        GOOD_MIN_CELLS <= cw <= GOOD_MAX_CELLS for cw in cells_w
    )

    # Bbox is tiny only if BOTH dimensions are too small on ALL levels
    tiny_all = all(ch < GOOD_MIN_CELLS for ch in cells_h) and all(
        cw < GOOD_MIN_CELLS for cw in cells_w
    )

    # Bbox is huge only if BOTH dimensions are too large on ALL levels
    huge_all = all(ch > GOOD_MAX_CELLS for ch in cells_h) and all(
        cw > GOOD_MAX_CELLS for cw in cells_w
    )

    return BboxCellStats(
        cells_s8=cells_s8,
        cells_s16=cells_s16,
        cells_s32=cells_s32,
        best_level=best_level,
        best_cells=best_cells,
        good_any=good_any,
        tiny_all=tiny_all,
        huge_all=huge_all,
    )


# ============================================================================
# Bbox Collection
# ============================================================================


def collect_bboxes_for_policy(
    config: TrainConfig,
    policy: ScalingPolicy,
    split: str = "val",
    max_samples: int = 5000,
) -> dict:
    """Collect bbox statistics for a single scaling policy.

    Args:
        config: Training configuration
        policy: Scaling policy to analyze
        split: Dataset split (default: "val")
        max_samples: Maximum number of samples to analyze

    Returns:
        Dictionary with:
            - widths: List of bbox widths (pixels)
            - heights: List of bbox heights (pixels)
            - class_ids: List of class IDs
            - cell_stats: List of BboxCellStats objects
    """
    logger.info(f"Collecting bboxes for policy: {policy}")

    # Build model artifacts to get processor
    dfine_cfg = config.dfine.model_dump()
    artifacts = ModelFactory.from_config(
        config.model,
        dfine_cfg,
        image_size=config.data.image_size,
    ).build()

    # Configure processor for this policy
    processor = artifacts.processor

    if policy.policy_type == "aspect":
        eval_size = {"shortest_edge": policy.short_side}
        if policy.long_cap:
            eval_size["longest_edge"] = policy.long_cap
    elif policy.policy_type == "stretch":
        # Stretch policy: short_side × stretch_ratio
        # E.g., 384 × 1.5 = 576, so resize to 384×576 (stretched)
        long_side = int(policy.short_side * policy.stretch_ratio)
        eval_size = {"height": policy.short_side, "width": long_side}
    else:  # square
        eval_size = {"height": policy.square_size, "width": policy.square_size}

    processor.do_resize = True
    processor.size = eval_size
    processor.do_pad = True

    # Build dataset
    pad_stride = max(config.dfine.feat_strides)
    dataset_factory = DatasetFactory(
        dataset_name=config.data.dataset,
        image_processor=processor,
        pad_stride=pad_stride,
        cache_dir=config.data.cache_dir,
        augmentation_config=None,
    )
    dataset, _ = dataset_factory.build(
        split=split,
        max_samples=max_samples,
        apply_augmentation=False,
    )

    # Collect bboxes
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=16,
        shuffle=False,
        num_workers=4,
        collate_fn=collate_fn,
    )

    widths = []
    heights = []
    class_ids = []
    cell_stats_list = []

    for batch in tqdm(dataloader, desc=f"Processing {policy.name}"):
        labels_list = batch["labels"]

        for lbl in labels_list:
            boxes = lbl["boxes"].cpu().numpy()  # [N, 4] normalized YOLO format
            classes = lbl["class_labels"].cpu().numpy()  # [N]

            # For DocLayNet: use original PDF dimensions if available in metadata
            # For other datasets: use the resized dimensions
            if "original_height" in lbl and "original_width" in lbl:
                # DocLayNet case: bboxes are normalized to COCO size (1025×1025),
                # but we want to analyze them at their original aspect ratio
                orig_h = float(lbl["original_height"])
                orig_w = float(lbl["original_width"])

                # Calculate what the resize dimensions WOULD BE for the original size
                # using the same logic as infer_resized_shape
                if policy.policy_type == "square":
                    # Square policy: fit to square, pad if needed
                    target_h = policy.square_size
                    target_w = policy.square_size
                elif policy.policy_type == "stretch":
                    # Stretch policy: force to specific dimensions
                    target_h = policy.short_side
                    target_w = int(policy.short_side * policy.stretch_ratio)
                else:  # aspect-ratio preserving
                    short_side = min(orig_h, orig_w)
                    scale = policy.short_side / short_side
                    target_h = int(orig_h * scale)
                    target_w = int(orig_w * scale)
                    if policy.long_cap:
                        long_side = max(target_h, target_w)
                        if long_side > policy.long_cap:
                            cap_scale = policy.long_cap / long_side
                            target_h = int(target_h * cap_scale)
                            target_w = int(target_w * cap_scale)

                # Bboxes are normalized to COCO size, denormalize to original, then scale to target
                coco_h = 1025.0  # DocLayNet COCO render size
                coco_w = 1025.0
                height_px_scale = target_h * (orig_h / coco_h)
                width_px_scale = target_w * (orig_w / coco_w)
            else:
                # Non-DocLayNet: use resized dimensions directly
                size = lbl["size"].cpu().numpy()  # [height, width] AFTER resize
                height_px_scale = float(size[0])
                width_px_scale = float(size[1] if size.shape[0] > 1 else size[0])

            if boxes.size == 0:
                continue

            for box, cls in zip(boxes, classes, strict=False):
                # YOLO format: [cx, cy, w, h] normalized
                bbox_w = box[2] * width_px_scale
                bbox_h = box[3] * height_px_scale

                widths.append(bbox_w)
                heights.append(bbox_h)
                class_ids.append(int(cls))

                # Compute FPN cell stats
                cell_stats = compute_bbox_cells(bbox_h, bbox_w)
                cell_stats_list.append(cell_stats)

    logger.info(f"Collected {len(widths)} bboxes for {policy.name}")

    return {
        "widths": widths,
        "heights": heights,
        "class_ids": class_ids,
        "cell_stats": cell_stats_list,
    }


# ============================================================================
# Policy Scoring & Ranking
# ============================================================================


@dataclass
class PolicyMetrics:
    """Aggregated metrics for a scaling policy."""

    policy_name: str
    n_bboxes: int
    frac_good: float  # Fraction with good cells on any level
    frac_tiny: float  # Fraction too small on all levels
    frac_huge: float  # Fraction too large on all levels
    mean_best_cells: float  # Average cells on best level
    stride8_pct: float  # % best detected at stride 8
    stride16_pct: float  # % best detected at stride 16
    stride32_pct: float  # % best detected at stride 32
    score: float  # Overall score


# Scoring weights
ALPHA_TINY = 4.0  # Strong penalty for too-small boxes
BETA_HUGE = 1.0  # Mild penalty for giants
GAMMA_SIZE = 0.05  # Mild penalty for oversized even on best level


def compute_policy_metrics(policy: ScalingPolicy, bbox_data: dict) -> PolicyMetrics:
    """Compute aggregated metrics for a policy.

    Args:
        policy: Scaling policy
        bbox_data: Output from collect_bboxes_for_policy

    Returns:
        PolicyMetrics with scores and statistics
    """
    cell_stats_list = bbox_data["cell_stats"]
    n_bboxes = len(cell_stats_list)

    if n_bboxes == 0:
        logger.warning(f"No bboxes for policy {policy.name}")
        return PolicyMetrics(
            policy_name=policy.name,
            n_bboxes=0,
            frac_good=0.0,
            frac_tiny=0.0,
            frac_huge=0.0,
            mean_best_cells=0.0,
            stride8_pct=0.0,
            stride16_pct=0.0,
            stride32_pct=0.0,
            score=0.0,
        )

    # Aggregate flags
    n_good = sum(s.good_any for s in cell_stats_list)
    n_tiny = sum(s.tiny_all for s in cell_stats_list)
    n_huge = sum(s.huge_all for s in cell_stats_list)

    frac_good = n_good / n_bboxes
    frac_tiny = n_tiny / n_bboxes
    frac_huge = n_huge / n_bboxes

    # Average cells on best level
    mean_best_cells = np.mean([s.best_cells for s in cell_stats_list])

    # Stride distribution
    stride_counts = {8: 0, 16: 0, 32: 0}
    for s in cell_stats_list:
        stride_counts[s.best_level] += 1

    stride8_pct = 100.0 * stride_counts[8] / n_bboxes
    stride16_pct = 100.0 * stride_counts[16] / n_bboxes
    stride32_pct = 100.0 * stride_counts[32] / n_bboxes

    # Compute score
    score = (
        frac_good - ALPHA_TINY * frac_tiny - BETA_HUGE * frac_huge - GAMMA_SIZE * mean_best_cells
    )

    return PolicyMetrics(
        policy_name=policy.name,
        n_bboxes=n_bboxes,
        frac_good=frac_good,
        frac_tiny=frac_tiny,
        frac_huge=frac_huge,
        mean_best_cells=mean_best_cells,
        stride8_pct=stride8_pct,
        stride16_pct=stride16_pct,
        stride32_pct=stride32_pct,
        score=score,
    )


def rank_policies(metrics_list: list[PolicyMetrics]) -> pd.DataFrame:
    """Rank policies by score and return DataFrame.

    Args:
        metrics_list: List of PolicyMetrics

    Returns:
        DataFrame sorted by score (descending)
    """
    data = []
    for m in metrics_list:
        data.append(
            {
                "Policy": m.policy_name,
                "N Bboxes": m.n_bboxes,
                "Good %": f"{100 * m.frac_good:.1f}",
                "Tiny %": f"{100 * m.frac_tiny:.1f}",
                "Huge %": f"{100 * m.frac_huge:.1f}",
                "Mean Cells": f"{m.mean_best_cells:.2f}",
                "S8 %": f"{m.stride8_pct:.1f}",
                "S16 %": f"{m.stride16_pct:.1f}",
                "S32 %": f"{m.stride32_pct:.1f}",
                "Score": f"{m.score:.4f}",
            }
        )

    df = pd.DataFrame(data)
    df = df.sort_values("Score", ascending=False).reset_index(drop=True)
    df.index += 1  # 1-indexed ranking
    return df


# ============================================================================
# Visualization
# ============================================================================


def plot_comparison_histograms(
    policies: list[ScalingPolicy],
    bbox_data_dict: dict,
    output_path: Path,
):
    """Plot overlaid histograms for all policies.

    Args:
        policies: List of scaling policies
        bbox_data_dict: Dict mapping policy.name -> bbox_data
        output_path: Where to save the plot
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Color map
    colors = plt.cm.tab10(np.linspace(0, 1, len(policies)))

    # Width histogram
    for policy, color in zip(policies, colors, strict=False):
        widths = bbox_data_dict[policy.name]["widths"]
        ax1.hist(
            widths,
            bins=50,
            alpha=0.4,
            label=policy.name,
            color=color,
            range=(0, 500),
        )
    ax1.set_xlabel("Bbox Width (pixels)")
    ax1.set_ylabel("Count")
    ax1.set_title("Bbox Width Distribution Across Policies")
    ax1.legend(loc="upper right", fontsize=8)
    ax1.grid(alpha=0.3)

    # Height histogram
    for policy, color in zip(policies, colors, strict=False):
        heights = bbox_data_dict[policy.name]["heights"]
        ax2.hist(
            heights,
            bins=50,
            alpha=0.4,
            label=policy.name,
            color=color,
            range=(0, 500),
        )
    ax2.set_xlabel("Bbox Height (pixels)")
    ax2.set_ylabel("Count")
    ax2.set_title("Bbox Height Distribution Across Policies")
    ax2.legend(loc="upper right", fontsize=8)
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved comparison histograms to {output_path}")


def plot_stride_coverage_heatmap(
    metrics_list: list[PolicyMetrics],
    output_path: Path,
):
    """Plot heatmap of stride coverage across policies.

    Args:
        metrics_list: List of PolicyMetrics
        output_path: Where to save the plot
    """
    # Build matrix: rows = policies, cols = strides
    policy_names = [m.policy_name for m in metrics_list]
    stride_data = np.array([[m.stride8_pct, m.stride16_pct, m.stride32_pct] for m in metrics_list])

    fig, ax = plt.subplots(figsize=(8, len(policy_names) * 0.5 + 2))
    sns.heatmap(
        stride_data,
        annot=True,
        fmt=".1f",
        cmap="YlGnBu",
        xticklabels=["Stride 8", "Stride 16", "Stride 32"],
        yticklabels=policy_names,
        cbar_kws={"label": "% of Bboxes"},
        ax=ax,
    )
    ax.set_title("FPN Stride Coverage by Policy")
    ax.set_ylabel("Policy")
    ax.set_xlabel("Best Detection Stride")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved stride coverage heatmap to {output_path}")


def plot_score_breakdown(
    metrics_list: list[PolicyMetrics],
    output_path: Path,
):
    """Plot score components as stacked bar chart.

    Args:
        metrics_list: List of PolicyMetrics
        output_path: Where to save the plot
    """
    # Sort by score
    metrics_list = sorted(metrics_list, key=lambda m: m.score, reverse=True)

    policy_names = [m.policy_name for m in metrics_list]
    good = [m.frac_good for m in metrics_list]
    tiny_penalty = [-ALPHA_TINY * m.frac_tiny for m in metrics_list]
    huge_penalty = [-BETA_HUGE * m.frac_huge for m in metrics_list]
    size_penalty = [-GAMMA_SIZE * m.mean_best_cells for m in metrics_list]

    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(len(policy_names))
    width = 0.6

    # Stacked bars
    ax.bar(x, good, width, label="Good Coverage", color="green", alpha=0.7)
    ax.bar(x, tiny_penalty, width, bottom=good, label="Tiny Penalty", color="red", alpha=0.7)
    ax.bar(
        x,
        huge_penalty,
        width,
        bottom=np.array(good) + np.array(tiny_penalty),
        label="Huge Penalty",
        color="orange",
        alpha=0.7,
    )
    ax.bar(
        x,
        size_penalty,
        width,
        bottom=np.array(good) + np.array(tiny_penalty) + np.array(huge_penalty),
        label="Size Penalty",
        color="blue",
        alpha=0.7,
    )

    # Add total score markers
    total_scores = [m.score for m in metrics_list]
    ax.scatter(x, total_scores, color="black", s=100, zorder=10, label="Total Score")

    ax.set_ylabel("Score")
    ax.set_xlabel("Policy")
    ax.set_title("Policy Score Breakdown")
    ax.set_xticks(x)
    ax.set_xticklabels(policy_names, rotation=45, ha="right")
    ax.legend()
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved score breakdown to {output_path}")


# ============================================================================
# Main Analysis Pipeline
# ============================================================================


def run_multiscale_analysis(
    config: TrainConfig,
    output_dir: Path,
    split: str = "val",
    max_samples: int = 5000,
    policies: list[ScalingPolicy] | None = None,
):
    """Run complete multiscale analysis pipeline.

    Args:
        config: Training configuration
        output_dir: Where to save outputs
        split: Dataset split to analyze
        max_samples: Maximum samples to process
        policies: List of policies to compare (default: DEFAULT_POLICIES)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create subdirectories
    stats_dir = output_dir / "stats"
    stats_dir.mkdir(exist_ok=True)

    if policies is None:
        policies = DEFAULT_POLICIES

    logger.info(f"Starting multiscale analysis with {len(policies)} policies")
    logger.info(f"Output directory: {output_dir}")

    # Step 1: Collect bboxes for all policies
    bbox_data_dict = {}
    for policy in policies:
        bbox_data = collect_bboxes_for_policy(config, policy, split, max_samples)
        bbox_data_dict[policy.name] = bbox_data

        # Save raw stats
        stats_path = stats_dir / f"{policy.name}.npz"
        np.savez(
            stats_path,
            widths=bbox_data["widths"],
            heights=bbox_data["heights"],
            class_ids=bbox_data["class_ids"],
        )
        logger.info(f"Saved stats to {stats_path}")

    # Step 2: Compute metrics for all policies
    metrics_list = []
    for policy in policies:
        metrics = compute_policy_metrics(policy, bbox_data_dict[policy.name])
        metrics_list.append(metrics)

    # Step 3: Rank policies
    ranking_df = rank_policies(metrics_list)
    ranking_path = output_dir / "policy_ranking.csv"
    ranking_df.to_csv(ranking_path)
    logger.info(f"Saved policy ranking to {ranking_path}")

    # Print ranking to console
    print("\n" + "=" * 80)
    print("POLICY RANKING")
    print("=" * 80)
    print(ranking_df.to_string(index=True))
    print("=" * 80 + "\n")

    # Step 4: Generate visualizations
    plot_comparison_histograms(policies, bbox_data_dict, output_dir / "comparison_histograms.png")
    plot_stride_coverage_heatmap(metrics_list, output_dir / "stride_coverage_heatmap.png")
    plot_score_breakdown(metrics_list, output_dir / "score_breakdown.png")

    # Step 5: Generate recommendation report
    best_policy = metrics_list[0]  # Already sorted by score in rank_policies
    recommendation_path = output_dir / "recommendation.txt"

    with open(recommendation_path, "w") as f:
        f.write("=" * 80 + "\n")
        f.write("MULTISCALE ANALYSIS RECOMMENDATION\n")
        f.write("=" * 80 + "\n\n")

        f.write(f"Dataset: {config.data.dataset}\n")
        f.write(f"Split: {split}\n")
        f.write(f"Samples analyzed: {max_samples}\n")
        f.write(f"Policies compared: {len(policies)}\n\n")

        f.write("TOP 3 POLICIES:\n")
        f.write("-" * 80 + "\n")
        for i, m in enumerate(sorted(metrics_list, key=lambda x: x.score, reverse=True)[:3], 1):
            f.write(f"{i}. {m.policy_name} (score: {m.score:.4f})\n")
            f.write(f"   - Good coverage: {100 * m.frac_good:.1f}%\n")
            f.write(f"   - Tiny boxes: {100 * m.frac_tiny:.1f}%\n")
            f.write(f"   - Huge boxes: {100 * m.frac_huge:.1f}%\n")
            f.write(
                f"   - Stride distribution: S8={m.stride8_pct:.1f}%, S16={m.stride16_pct:.1f}%, S32={m.stride32_pct:.1f}%\n"
            )
            f.write("\n")

        f.write("RECOMMENDATION:\n")
        f.write("-" * 80 + "\n")
        f.write(f"Based on FPN-aware analysis, {best_policy.policy_name} is recommended.\n\n")

        # Compare to AR-640 if it's not the best
        ar640_metrics = next((m for m in metrics_list if m.policy_name == "AR-640"), None)
        if ar640_metrics and ar640_metrics.policy_name != best_policy.policy_name:
            score_diff = best_policy.score - ar640_metrics.score
            f.write(f"Current policy (AR-640) score: {ar640_metrics.score:.4f}\n")
            f.write(f"Recommended policy score: {best_policy.score:.4f}\n")
            f.write(
                f"Score improvement: {score_diff:+.4f} ({100 * score_diff / abs(ar640_metrics.score):+.1f}%)\n\n"
            )

        f.write("KEY INSIGHTS:\n")
        f.write("-" * 80 + "\n")

        # Check if square policies dominate
        sq_policies = [m for m in metrics_list if m.policy_name.startswith("SQ")]
        ar_policies = [m for m in metrics_list if m.policy_name.startswith("AR")]

        if sq_policies:
            best_sq = max(sq_policies, key=lambda m: m.score)
            best_ar = max(ar_policies, key=lambda m: m.score)

            if best_sq.score > best_ar.score:
                f.write(
                    f"- Square policies ({best_sq.policy_name}) outperform aspect-ratio policies\n"
                )
                f.write("  This suggests uniform scaling reduces FPN coverage gaps\n")
            else:
                f.write(
                    f"- Aspect-ratio policies ({best_ar.policy_name}) outperform square policies\n"
                )
                f.write(
                    "  This suggests preserving aspect ratios is more important than uniform scaling\n"
                )

        # Check tiny box problem
        worst_tiny = max(metrics_list, key=lambda m: m.frac_tiny)
        if worst_tiny.frac_tiny > 0.1:
            f.write(
                f"- {worst_tiny.policy_name} has {100 * worst_tiny.frac_tiny:.1f}% tiny boxes (<3 cells on all strides)\n"
            )
            f.write("  Consider avoiding small scales for this dataset\n")

        f.write("\n" + "=" * 80 + "\n")

    logger.info(f"Saved recommendation to {recommendation_path}")
    logger.info("Multiscale analysis complete!")
