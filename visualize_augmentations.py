#!/usr/bin/env python3
"""Generate a concise overview of the document augmentations."""

from __future__ import annotations

import random
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from math import ceil
from pathlib import Path

import albumentations as A
import cv2
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from datasets import load_dataset
from PIL import Image, ImageDraw

from doc_obj_detect.data.constants import PUBLAYNET_CLASSES

MOSAIC_METADATA_KEY = "mosaic_metadata"
OUTPUT_DIR = Path("outputs/augmentation_visualizations")

COLOR_MAP = {
    "text": "cyan",
    "title": "blue",
    "list": "magenta",
    "table": "orange",
    "figure": "green",
    "caption": "red",
    "unknown": "gray",
}


@dataclass
class BoundingBox:
    x_min: float
    y_min: float
    x_max: float
    y_max: float
    label: str

    def to_pascal(self) -> list[float]:
        return [float(self.x_min), float(self.y_min), float(self.x_max), float(self.y_max)]


@dataclass
class Sample:
    image: np.ndarray
    boxes: list[BoundingBox]


def load_publaynet_samples(num_samples: int = 4) -> list[Sample]:
    """Stream PubLayNet and return samples with at least three annotations."""
    print(f"Loading {num_samples} PubLayNet pages …")
    try:
        dataset = load_dataset("shunk031/PubLayNet", split="train", streaming=True)
        samples: list[Sample] = []
        dataset_iter = iter(dataset)
        attempts = 0

        while len(samples) < num_samples and attempts < num_samples * 6:
            attempts += 1
            try:
                sample = next(dataset_iter)
            except StopIteration:  # pragma: no cover - streaming fallback
                break

            image = sample["image"]
            if isinstance(image, Image.Image):
                image = np.array(image.convert("RGB"))

            annotations = sample["annotations"]
            if not annotations or "bbox" not in annotations:
                continue

            boxes: list[BoundingBox] = []
            for bbox, category_id in zip(
                annotations["bbox"], annotations["category_id"], strict=False
            ):
                x_min, y_min, width, height = bbox
                label = PUBLAYNET_CLASSES.get(category_id, "unknown")
                boxes.append(
                    BoundingBox(
                        x_min=x_min,
                        y_min=y_min,
                        x_max=x_min + width,
                        y_max=y_min + height,
                        label=label,
                    )
                )

            if len(boxes) >= 3:
                samples.append(Sample(image=image, boxes=boxes))
                print(f"  ✓ Sample {len(samples)}: {image.shape}, {len(boxes)} boxes")

        if samples:
            return samples
        raise RuntimeError("No valid samples collected")
    except Exception as exc:  # pragma: no cover - fallback path
        print(f"Falling back to synthetic pages: {exc}")
        return create_synthetic_samples(num_samples)


def create_synthetic_samples(num_samples: int) -> list[Sample]:
    """Create synthetic document pages when PubLayNet is unavailable."""
    samples: list[Sample] = []
    for idx in range(num_samples):
        width = 640 + idx * 40
        height = 840 + idx * 60
        image = Image.new("RGB", (width, height), color="white")
        draw = ImageDraw.Draw(image)

        draw.rectangle([40, 40, width - 40, 110], fill="lightblue", outline="blue", width=3)
        draw.rectangle([40, 130, width - 40, 360], fill="lightgray", outline="black")
        draw.rectangle([40, 380, width - 40, 560], fill="lemonchiffon", outline="orange", width=3)
        draw.rectangle(
            [40, 590, width // 2, height - 80], fill="lightgreen", outline="green", width=3
        )
        draw.rectangle([width // 2 + 10, 590, width - 40, height - 80], outline="black", width=2)

        boxes = [
            BoundingBox(40, 40, width - 40, 110, "title"),
            BoundingBox(40, 130, width - 40, 360, "text"),
            BoundingBox(40, 380, width - 40, 560, "table"),
            BoundingBox(40, 590, width // 2, height - 80, "figure"),
        ]
        samples.append(Sample(image=np.array(image), boxes=boxes))
    return samples


def visualize_with_boxes(ax, image: np.ndarray, boxes: Sequence[BoundingBox], title: str) -> None:
    ax.imshow(image)
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.axis("off")
    for box in boxes:
        rect = patches.Rectangle(
            (box.x_min, box.y_min),
            box.x_max - box.x_min,
            box.y_max - box.y_min,
            linewidth=2.5,
            edgecolor=COLOR_MAP.get(box.label, "black"),
            facecolor="none",
        )
        ax.add_patch(rect)
        ax.text(
            box.x_min + 4,
            box.y_min + 14,
            box.label,
            fontsize=8,
            color="white",
            weight="bold",
            bbox={"facecolor": COLOR_MAP.get(box.label, "black"), "alpha": 0.7, "pad": 1},
        )


def convert_for_albumentations(
    boxes: Sequence[BoundingBox],
) -> tuple[list[list[float]], list[str]]:
    coords = [box.to_pascal() for box in boxes]
    labels = [box.label for box in boxes]
    return coords, labels


def boxes_from_albumentations(
    coords: Iterable[Sequence[float]],
    labels: Iterable[str],
) -> list[BoundingBox]:
    output: list[BoundingBox] = []
    for (x_min, y_min, x_max, y_max), label in zip(coords, labels, strict=False):
        output.append(BoundingBox(x_min=x_min, y_min=y_min, x_max=x_max, y_max=y_max, label=label))
    return output


def apply_transform(sample: Sample, transform: A.Compose, name: str) -> Sample:
    bboxes, labels = convert_for_albumentations(sample.boxes)
    try:
        augmented = transform(image=sample.image, bboxes=bboxes, class_labels=labels)
        return Sample(
            image=augmented["image"],
            boxes=boxes_from_albumentations(augmented["bboxes"], augmented["class_labels"]),
        )
    except Exception as exc:  # pragma: no cover - visualization fallback
        print(f"Warning: {name} failed ({exc}); returning original image")
        return sample


def apply_mosaic_preview(primary: Sample, pool: Sequence[Sample], target_size: int = 640) -> Sample:
    if len(pool) < 3:
        return primary

    others = random.sample(pool, 3)
    metadata = []
    for item in others:
        bboxes, labels = convert_for_albumentations(item.boxes)
        metadata.append({"image": item.image, "bboxes": bboxes, "class_labels": labels})

    mosaic = A.Compose(
        [
            A.Mosaic(
                grid_yx=(2, 2),
                target_size=(target_size, target_size),
                cell_shape=(target_size, target_size),
                center_range=(0.35, 0.65),
                fit_mode="cover",
                interpolation=cv2.INTER_LINEAR,
                mask_interpolation=cv2.INTER_NEAREST,
                metadata_key=MOSAIC_METADATA_KEY,
                p=1.0,
            )
        ],
        bbox_params=A.BboxParams(
            format="pascal_voc", label_fields=["class_labels"], min_visibility=0.3
        ),
    )

    primary_boxes, primary_labels = convert_for_albumentations(primary.boxes)
    augmented = mosaic(
        image=primary.image,
        bboxes=primary_boxes,
        class_labels=primary_labels,
        mosaic_metadata=metadata,
    )
    return Sample(
        image=augmented["image"],
        boxes=boxes_from_albumentations(augmented["bboxes"], augmented["class_labels"]),
    )


def build_standard_transforms() -> dict[str, A.Compose]:
    bbox_params = A.BboxParams(
        format="pascal_voc", label_fields=["class_labels"], min_visibility=0.3
    )
    return {
        "Original": A.Compose([A.NoOp()], bbox_params=bbox_params),
        "Horizontal Flip": A.Compose([A.HorizontalFlip(p=1.0)], bbox_params=bbox_params),
        "Vertical Flip": A.Compose([A.VerticalFlip(p=1.0)], bbox_params=bbox_params),
        "Random Crop": A.Compose(
            [A.RandomResizedCrop(size=(768, 640), scale=(0.6, 0.95), p=1.0)],
            bbox_params=bbox_params,
        ),
        "Rotate ±5°": A.Compose([A.Rotate(limit=5, border_mode=0, p=1.0)], bbox_params=bbox_params),
        "Brightness/Contrast": A.Compose(
            [A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=1.0)],
            bbox_params=bbox_params,
        ),
        "Gaussian Blur": A.Compose([A.GaussianBlur(blur_limit=3, p=1.0)], bbox_params=bbox_params),
        "Gaussian Noise": A.Compose(
            [A.GaussNoise(std_range=(0.0, 0.01), p=1.0)], bbox_params=bbox_params
        ),
        "Perspective": A.Compose(
            [A.Perspective(scale=(0.02, 0.05), p=1.0)],
            bbox_params=bbox_params,
        ),
        "Elastic Transform": A.Compose(
            [A.ElasticTransform(alpha=30, sigma=5, border_mode=0, p=1.0)],
            bbox_params=bbox_params,
        ),
    }


def render_augmentation_grid(samples: Sequence[Sample]) -> None:
    base_sample = samples[0]
    transforms = build_standard_transforms()

    panels: list[tuple[str, Sample]] = []
    for name, transform in transforms.items():
        panels.append((name, apply_transform(base_sample, transform, name)))

    if len(samples) >= 4:
        panels.append(("Mosaic", apply_mosaic_preview(base_sample, samples[1:])))

    n_cols = 4
    n_rows = ceil(len(panels) / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.5 * n_cols, 4.5 * n_rows))
    axes = axes.flatten()

    for idx, (title, sample) in enumerate(panels):
        visualize_with_boxes(axes[idx], sample.image, sample.boxes, title)

    for idx in range(len(panels), len(axes)):
        axes[idx].axis("off")

    fig.suptitle("Document Augmentations", fontsize=16, fontweight="bold", y=0.995)
    fig.tight_layout()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / "augmentation_grid.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"✓ Wrote augmentation grid to {path}")


def render_dataset_grid(samples: Sequence[Sample], count: int = 3) -> None:
    subset = samples[:count]
    if not subset:
        return

    fig, axes = plt.subplots(1, len(subset), figsize=(6 * len(subset), 6))
    if len(subset) == 1:
        axes = [axes]  # type: ignore[list-item]

    for ax, sample in zip(axes, subset, strict=False):
        visualize_with_boxes(ax, sample.image, sample.boxes, "PubLayNet Page")

    fig.suptitle("PubLayNet Reference Pages", fontsize=15, fontweight="bold", y=0.98)
    fig.tight_layout()
    path = OUTPUT_DIR / "publaynet_triptych.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"✓ Wrote dataset overview to {path}")


def main() -> None:
    samples = load_publaynet_samples(num_samples=4)
    render_augmentation_grid(samples)
    render_dataset_grid(samples, count=min(3, len(samples)))
    print(f"All visualizations stored in {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
