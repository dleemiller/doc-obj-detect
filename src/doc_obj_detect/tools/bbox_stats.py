"""Analyze bounding-box scales and visualize histograms with stride overlays."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from rich.progress import BarColumn, Progress, TextColumn, TimeRemainingColumn
from torch.utils.data import DataLoader

from doc_obj_detect.config import TrainConfig, load_train_config
from doc_obj_detect.data import DatasetFactory, collate_fn
from doc_obj_detect.models import ModelFactory

logger = logging.getLogger(__name__)


def _configure_processors(config: TrainConfig, target_short_side: int | None = None):
    dfine_cfg = config.dfine.model_dump()
    factory = ModelFactory.from_config(config.model, dfine_cfg, config.data.image_size)
    artifacts = factory.build()
    processor = artifacts.processor

    eval_short = target_short_side or config.data.image_size
    if target_short_side is None and config.augmentation and config.augmentation.multi_scale_sizes:
        eval_short = max(config.augmentation.multi_scale_sizes)
    eval_size = {"shortest_edge": eval_short}
    if config.augmentation and config.augmentation.max_long_side:
        eval_size["longest_edge"] = config.augmentation.max_long_side

    processor.do_resize = True
    processor.size = eval_size
    processor.do_pad = True
    return processor


def _collect_bbox_samples(
    dataset_factory: DatasetFactory,
    split: str,
    max_samples: int | None,
) -> tuple[np.ndarray, np.ndarray]:
    dataset, _ = dataset_factory.build(
        split=split,
        max_samples=max_samples,
        apply_augmentation=False,
    )

    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)
    widths: list[float] = []
    heights: list[float] = []
    total_samples = len(dataset)
    logger.info("Collecting bounding boxes from %s split (%s samples)...", split, total_samples)

    progress = Progress(
        TextColumn("[bold blue]Processing[/]"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TimeRemainingColumn(),
        transient=True,
    )
    task_id = progress.add_task(f"BBox {split}", total=total_samples)
    with progress:
        for batch in dataloader:
            labels_list = batch["labels"]
            for lbl in labels_list:
                boxes = lbl["boxes"].cpu().numpy()
                orig = lbl["orig_size"].cpu().numpy()
                if boxes.size == 0:
                    continue
                height_px_scale = float(orig[0])
                width_px_scale = float(orig[1] if orig.shape[0] > 1 else orig[0])
                width_px = boxes[:, 2] * width_px_scale
                height_px = boxes[:, 3] * height_px_scale
                widths.extend(width_px.tolist())
                heights.extend(height_px.tolist())
            progress.update(task_id, advance=1)
    logger.info("Collected %s bounding boxes from %s split.", len(widths), split)
    return np.array(widths, dtype=np.float32), np.array(heights, dtype=np.float32)


def compute_stride_ranges(strides: Sequence[int]) -> dict[int, tuple[float, float]]:
    ranges = {}
    for stride in strides:
        # Heuristic: each stride handles objects roughly 2x to 8x the stride (similar to FPN anchors)
        ranges[stride] = (stride * 2.0, stride * 8.0)
    return ranges


def collect_bbox_samples(
    config_path: str | Path,
    split: str = "val",
    max_samples: int | None = None,
    target_short_side: int | None = None,
) -> tuple[np.ndarray, np.ndarray, Sequence[int]]:
    """Collect bbox widths/heights (in pixels) for a dataset split."""

    config = load_train_config(config_path)
    processor = _configure_processors(config, target_short_side=target_short_side)
    pad_stride = max(config.dfine.feat_strides)
    dataset_factory = DatasetFactory(
        dataset_name=config.data.dataset,
        image_processor=processor,
        pad_stride=pad_stride,
        cache_dir=config.data.cache_dir,
        augmentation_config=None,
    )

    widths, heights = _collect_bbox_samples(dataset_factory, split, max_samples)
    return widths, heights, config.dfine.feat_strides


def plot_bbox_histograms(
    widths: np.ndarray,
    heights: np.ndarray,
    stride_ranges: dict[int, tuple[float, float]],
    strides: Sequence[int],
    bins: int = 80,
    output_path: str | Path = "outputs/bbox_hist.png",
    figsize: tuple[int, int] = (14, 5),
) -> Path:
    """Plot histograms with stride coverage overlays."""

    fig, axes = plt.subplots(1, 2, figsize=figsize)
    width_ax, height_ax = axes

    width_ax.hist(widths, bins=bins, color="#4ECDC4", alpha=0.7, edgecolor="black")
    width_ax.set_title("Bounding Box Widths (px)")
    width_ax.set_xlabel("Width (pixels)")
    width_ax.set_ylabel("Count")

    height_ax.hist(heights, bins=bins, color="#FFA07A", alpha=0.7, edgecolor="black")
    height_ax.set_title("Bounding Box Heights (px)")
    height_ax.set_xlabel("Height (pixels)")
    height_ax.set_ylabel("Count")

    colors = ["#FF6B6B", "#45B7D1", "#98D8C8", "#F7DC6F", "#BB8FCE"]
    for idx, stride in enumerate(strides):
        rng = stride_ranges[stride]
        color = colors[idx % len(colors)]
        for ax in axes:
            ax.axvspan(rng[0], rng[1], color=color, alpha=0.12, label=f"Stride {stride} range")

    handles, labels = [], []
    for idx, stride in enumerate(strides):
        color = colors[idx % len(colors)]
        handles.append(plt.Rectangle((0, 0), 1, 1, color=color, alpha=0.2))
        labels.append(
            f"Stride {stride}: {stride_ranges[stride][0]:.0f}-{stride_ranges[stride][1]:.0f}px"
        )
    width_ax.legend(handles, labels, loc="upper right", fontsize=8)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def save_bbox_data(
    path: str | Path, widths: np.ndarray, heights: np.ndarray, strides: Sequence[int]
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, widths=widths, heights=heights, strides=np.asarray(strides))
    return path


def load_bbox_data(path: str | Path) -> tuple[np.ndarray, np.ndarray, Sequence[int]]:
    data = np.load(path)
    strides = data["strides"].tolist()
    return data["widths"], data["heights"], strides


def generate_bbox_histograms(
    config_path: str | Path,
    split: str = "val",
    output_path: str | Path = "outputs/bbox_hist.png",
    bins: int = 80,
    max_samples: int | None = None,
    target_short_side: int | None = None,
) -> Path:
    """Generate width/height histograms with stride overlays."""

    widths, heights, strides = collect_bbox_samples(
        config_path=config_path,
        split=split,
        max_samples=max_samples,
        target_short_side=target_short_side,
    )
    if widths.size == 0:
        raise RuntimeError("No bounding boxes found in the dataset subset.")

    stride_ranges = compute_stride_ranges(strides)
    for stride, rng in stride_ranges.items():
        logger.info("Stride %s covers approx %.1fpx - %.1fpx", stride, rng[0], rng[1])

    return plot_bbox_histograms(
        widths,
        heights,
        stride_ranges,
        strides,
        bins=bins,
        output_path=output_path,
    )
