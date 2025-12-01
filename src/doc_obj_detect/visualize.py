"""Visualization utilities for document augmentations."""

import argparse
import logging
from pathlib import Path

import cv2
import numpy as np
import yaml
from PIL import Image, ImageDraw, ImageFont

from doc_obj_detect.config import AugmentationConfig
from doc_obj_detect.data.augmentor import AlbumentationsAugmentor
from doc_obj_detect.data.datasets import DatasetLoader

logger = logging.getLogger(__name__)
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
    mode: str = "simple",
    config_path: str | None = None,
) -> None:
    """Generate augmentation visualization samples.

    Args:
        dataset_name: Dataset to use ('publaynet' or 'doclaynet')
        num_samples: Number of samples to generate
        output_dir: Directory to save visualizations
        cache_dir: Optional cache directory for datasets
        mode: Visualization mode:
            - 'simple': Original vs augmented (default)
            - 'comparison': Original vs photometric vs augraphy (3-way comparison)
        config_path: Path to config YAML for augmentation settings (optional)
    """
    dataset_name = dataset_name.lower()
    if dataset_name == "publaynet":
        dataset, class_labels = DatasetLoader.load_publaynet("val", cache_dir)
    elif dataset_name == "doclaynet":
        dataset, class_labels = DatasetLoader.load_doclaynet("val", cache_dir)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if mode == "simple":
        _visualize_simple(dataset, class_labels, num_samples, output_path, config_path)
    elif mode == "comparison":
        _visualize_comparison(dataset, class_labels, num_samples, output_path, config_path)
    else:
        raise ValueError(f"Unknown mode: {mode}. Choose 'simple' or 'comparison'")

    logger.info("Samples saved to %s", output_path)


def _visualize_simple(
    dataset,
    class_labels: dict[int, str],
    num_samples: int,
    output_path: Path,
    config_path: str | None,
) -> None:
    """Generate simple original vs augmented comparisons."""
    # Load config if provided, otherwise use defaults
    if config_path:
        with open(config_path) as f:
            config = yaml.safe_load(f)
        aug_config = config.get("augmentation", {})
    else:
        aug_config = AugmentationConfig(multi_scale_sizes=[512]).model_dump()

    augmentor = AlbumentationsAugmentor(aug_config)
    transform = augmentor.build_transform(batch_scale=512)

    logger.info("Generating %s augmentation samples (simple mode)...", num_samples)
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
        logger.info("Saved: %s", output_file)


def _visualize_comparison(
    dataset,
    class_labels: dict[int, str],
    num_samples: int,
    output_path: Path,
    config_path: str | None,
) -> None:
    """Generate 3-way comparison: original, photometric, augraphy."""
    # Load config
    if config_path:
        with open(config_path) as f:
            config = yaml.safe_load(f)
        base_aug_config = config.get("augmentation", {})
    else:
        # Use default config with augraphy enabled
        base_aug_config = AugmentationConfig(multi_scale_sizes=[512]).model_dump()
        base_aug_config["augraphy"] = {"enabled": True}

    # Create two augmentors: photometric vs augraphy
    aug_config_photometric = base_aug_config.copy()
    aug_config_photometric["augraphy"] = {
        "enabled": True,
        "choice_probability": 0.0,  # Always use photometric
    }

    aug_config_augraphy = base_aug_config.copy()
    aug_config_augraphy["augraphy"] = {
        "enabled": True,
        "choice_probability": 1.0,  # Always use augraphy
        "ink_probability": 0.6,
        "paper_probability": 0.7,
        "post_probability": 0.5,
    }

    augmentor_photometric = AlbumentationsAugmentor(aug_config_photometric)
    augmentor_augraphy = AlbumentationsAugmentor(aug_config_augraphy)

    logger.info("Generating %s augmentation comparisons (photometric vs augraphy)...", num_samples)
    logger.info("  1. Original image (with bboxes)")
    logger.info("  2. Geometric + Photometric")
    logger.info("  3. Geometric + Augraphy")

    for idx in range(num_samples):
        sample = dataset[idx]
        image = sample["image"]
        annotations = sample["annotations"]

        image_np = np.array(image.convert("RGB"))
        bboxes = [ann["bbox"] for ann in annotations]
        category_ids = [ann["category_id"] for ann in annotations]

        # Clean bboxes
        clean_bboxes = []
        clean_cats = []
        for bbox, cat in zip(bboxes, category_ids, strict=False):
            x, y, w, h = bbox[:4]
            if w > 1e-3 and h > 1e-3:
                clean_bboxes.append([float(x), float(y), float(w), float(h)])
                clean_cats.append(int(cat))

        if not clean_bboxes:
            logger.warning("Sample %d has no valid bboxes, skipping", idx)
            continue

        # Save original with bboxes
        original_with_boxes = _draw_bboxes_cv2(
            image_np.copy(), clean_bboxes, clean_cats, class_labels
        )
        original_path = output_path / f"sample_{idx:03d}_1_original.jpg"
        cv2.imwrite(str(original_path), cv2.cvtColor(original_with_boxes, cv2.COLOR_RGB2BGR))

        # Apply photometric augmentation
        try:
            examples_photo = {
                "image": [image_np.copy()],
                "annotations": [{"bbox": clean_bboxes.copy(), "category_id": clean_cats.copy()}],
            }
            augmented_photo = augmentor_photometric.augment(examples_photo)

            aug_image_photo = augmented_photo["image"][0]
            aug_anns_photo = augmented_photo["annotations"][0]
            aug_bboxes_photo = aug_anns_photo["bbox"]
            aug_cats_photo = aug_anns_photo["category_id"]

            photo_with_boxes = _draw_bboxes_cv2(
                aug_image_photo.copy(), aug_bboxes_photo, aug_cats_photo, class_labels
            )
            photo_path = output_path / f"sample_{idx:03d}_2_photometric.jpg"
            cv2.imwrite(str(photo_path), cv2.cvtColor(photo_with_boxes, cv2.COLOR_RGB2BGR))
        except Exception as e:
            logger.error("Error with photometric augmentation on sample %d: %s", idx, e)
            continue

        # Apply augraphy augmentation
        try:
            examples_aug = {
                "image": [image_np.copy()],
                "annotations": [{"bbox": clean_bboxes.copy(), "category_id": clean_cats.copy()}],
            }
            augmented_aug = augmentor_augraphy.augment(examples_aug)

            aug_image_aug = augmented_aug["image"][0]
            aug_anns_aug = augmented_aug["annotations"][0]
            aug_bboxes_aug = aug_anns_aug["bbox"]
            aug_cats_aug = aug_anns_aug["category_id"]

            aug_with_boxes = _draw_bboxes_cv2(
                aug_image_aug.copy(), aug_bboxes_aug, aug_cats_aug, class_labels
            )
            aug_path = output_path / f"sample_{idx:03d}_3_augraphy.jpg"
            cv2.imwrite(str(aug_path), cv2.cvtColor(aug_with_boxes, cv2.COLOR_RGB2BGR))

            logger.info("Saved sample %d triplet", idx)
        except Exception as e:
            logger.error("Error with augraphy augmentation on sample %d: %s", idx, e)
            continue


def _draw_bboxes(
    image_np: np.ndarray,
    boxes: list[list[float]],
    labels: list[int],
    class_labels: dict[int, str],
) -> np.ndarray:
    """Render bounding boxes and labels onto an image using PIL."""
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


def _draw_bboxes_cv2(
    image: np.ndarray,
    bboxes: list,
    category_ids: list,
    labels: dict[int, str],
) -> np.ndarray:
    """Draw bounding boxes on image for visualization using OpenCV."""
    # Ensure RGB for consistent colors
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.ndim == 3 and image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)

    pil_image = Image.fromarray(image)
    draw = ImageDraw.Draw(pil_image)

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
    except Exception:
        font = ImageFont.load_default()

    # Define colors (RGB)
    colors = [
        (255, 0, 0),  # Red
        (0, 255, 0),  # Green
        (0, 0, 255),  # Blue
        (255, 255, 0),  # Yellow
        (255, 0, 255),  # Magenta
    ]

    for bbox, cat_id in zip(bboxes, category_ids, strict=False):
        x, y, w, h = float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])
        cat_id = int(cat_id)
        color = colors[cat_id % len(colors)]
        draw.rectangle([x, y, x + w, y + h], outline=color, width=2)
        label = labels.get(cat_id, f"Class {cat_id}")
        draw.text((x, max(0, y - 20)), label, fill=color, font=font)

    return np.array(pil_image)


def _compose_side_by_side(
    original: np.ndarray,
    augmented: np.ndarray,
) -> Image.Image:
    """Create a labeled side-by-side comparison image."""
    orig_h, orig_w = original.shape[:2]
    aug_h, aug_w = augmented.shape[:2]
    canvas_height = max(orig_h, aug_h)
    comparison = np.zeros((canvas_height, orig_w + aug_w, 3), dtype=np.uint8)
    comparison[:orig_h, :orig_w] = original
    comparison[:aug_h, orig_w : orig_w + aug_w] = augmented

    comparison_image = Image.fromarray(comparison)
    draw = ImageDraw.Draw(comparison_image)
    font = _load_font(size=24)
    draw.text((20, 20), "Original", fill="white", font=font, stroke_width=2, stroke_fill="black")
    draw.text(
        (orig_w + 20, 20),
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
    parser.add_argument(
        "--mode",
        type=str,
        choices=["simple", "comparison"],
        default="simple",
        help="Visualization mode: 'simple' for original vs augmented, "
        "'comparison' for original vs photometric vs augraphy",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to config YAML for augmentation settings (optional)",
    )
    args = parser.parse_args()

    visualize_augmentations(
        dataset_name=args.dataset,
        num_samples=args.num_samples,
        output_dir=args.output_dir,
        cache_dir=args.cache_dir,
        mode=args.mode,
        config_path=args.config,
    )


if __name__ == "__main__":
    main()
