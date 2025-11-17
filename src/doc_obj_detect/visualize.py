"""Visualization utilities for document augmentations."""

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from doc_obj_detect.config import AugmentationConfig
from doc_obj_detect.data import (
    get_augmentation_transform,
    load_doclaynet,
    load_publaynet,
)

_COLORS = [
    "#FF6B6B",
    "#4ECDC4",
    "#45B7D1",
    "#FFA07A",
    "#98D8C8",
    "#F7DC6F",
    "#BB8FCE",
    "#85C1E2",
    "#F8B739",
    "#52B788",
    "#F06292",
]


def visualize_augmentations(
    dataset_name: str,
    num_samples: int = 4,
    output_dir: str = "outputs/augmentation_samples",
    cache_dir: str | None = None,
) -> None:
    """Generate side-by-side original vs augmented document samples."""

    dataset_name = dataset_name.lower()
    if dataset_name == "publaynet":
        dataset, class_labels = load_publaynet("val", cache_dir)
    elif dataset_name == "doclaynet":
        dataset, class_labels = load_doclaynet("val", cache_dir)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Single-scale transform for clarity
    aug_config = AugmentationConfig(multi_scale_sizes=[512]).model_dump()
    transform = get_augmentation_transform(aug_config, batch_scale=512)

    print(f"Generating {num_samples} augmentation samples...")
    for idx in range(num_samples):
        sample = dataset[idx]
        image = sample["image"]
        annotations = sample["annotations"]

        image_np = np.array(image.convert("RGB"))
        bboxes = [ann["bbox"] for ann in annotations]
        category_ids = [ann["category_id"] for ann in annotations]

        augmented = transform(
            image=image_np,
            bboxes=bboxes,
            category_ids=category_ids,
        )

        original_with_boxes = _draw_bboxes(image_np, bboxes, category_ids, class_labels)
        augmented_with_boxes = _draw_bboxes(
            augmented["image"],
            augmented["bboxes"],
            augmented["category_ids"],
            class_labels,
        )

        comparison = _compose_side_by_side(original_with_boxes, augmented_with_boxes)
        output_file = output_path / f"sample_{idx:02d}.png"
        comparison.save(output_file)
        print(f"Saved: {output_file}")

    print(f"\nSamples saved to {output_path}")


def _draw_bboxes(
    image_np: np.ndarray,
    boxes: list[list[float]],
    labels: list[int],
    class_labels: dict[int, str],
) -> np.ndarray:
    """Render bounding boxes and labels onto an image."""
    image_draw = Image.fromarray(image_np)
    draw = ImageDraw.Draw(image_draw)

    for bbox, label in zip(boxes, labels, strict=False):
        x, y, w, h = bbox
        color = _COLORS[int(label) % len(_COLORS)]
        draw.rectangle([x, y, x + w, y + h], outline=color, width=3)

        label_name = class_labels.get(int(label), str(int(label)))
        font = _load_font(size=16)
        text_bbox = draw.textbbox((x, y - 20), label_name, font=font)
        draw.rectangle(text_bbox, fill=color)
        draw.text((x, y - 20), label_name, fill="white", font=font)

    return np.array(image_draw)


def _compose_side_by_side(
    original: np.ndarray,
    augmented: np.ndarray,
) -> Image.Image:
    """Create a labeled side-by-side comparison image."""
    h, w = original.shape[:2]
    comparison = np.zeros((h, w * 2, 3), dtype=np.uint8)
    comparison[:, :w] = original
    comparison[:, w:] = augmented

    comparison_image = Image.fromarray(comparison)
    draw = ImageDraw.Draw(comparison_image)
    font = _load_font(size=24)
    draw.text(
        (20, 20),
        "Original",
        fill="white",
        font=font,
        stroke_width=2,
        stroke_fill="black",
    )
    draw.text(
        (w + 20, 20),
        "Augmented",
        fill="white",
        font=font,
        stroke_width=2,
        stroke_fill="black",
    )
    return comparison_image


def _load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """Load a truetype font with fallback to default."""
    try:
        return ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            size,
        )
    except Exception:
        return ImageFont.load_default()


def main() -> None:
    """CLI entry point for augmentation visualization."""
    parser = argparse.ArgumentParser(description="Visualize document augmentations.")
    parser.add_argument(
        "dataset",
        choices=["publaynet", "doclaynet"],
        help="Dataset to sample from.",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=4,
        help="Number of samples to generate.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/augmentation_samples",
        help="Directory to save comparison images.",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="Optional cache directory for datasets.",
    )
    args = parser.parse_args()

    visualize_augmentations(
        dataset_name=args.dataset,
        num_samples=args.num_samples,
        output_dir=args.output_dir,
        cache_dir=args.cache_dir,
    )


if __name__ == "__main__":
    main()
