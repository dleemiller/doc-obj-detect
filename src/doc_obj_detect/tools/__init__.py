"""Utility analysis tools."""

from .bbox_stats import (
    collect_bbox_samples,
    compute_stride_ranges,
    generate_bbox_histograms,
    load_bbox_data,
    plot_bbox_histograms,
    save_bbox_data,
)
from .cleanup import CleanupSummary, cleanup_outputs

__all__ = [
    "collect_bbox_samples",
    "generate_bbox_histograms",
    "plot_bbox_histograms",
    "compute_stride_ranges",
    "save_bbox_data",
    "load_bbox_data",
    "cleanup_outputs",
    "CleanupSummary",
]
