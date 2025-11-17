"""Data loading, preprocessing, and augmentation for document object detection."""

from pathlib import Path

import albumentations as A
import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoImageProcessor

# Dataset class labels
# PubLayNet original IDs: 1=text, 2=title, 3=list, 4=table, 5=figure
# We remap to 0-indexed for the model
PUBLAYNET_CLASSES = {
    0: "text",
    1: "title",
    2: "list",
    3: "table",
    4: "figure",
}

# Mapping from PubLayNet dataset IDs to model IDs (0-indexed)
PUBLAYNET_ID_MAPPING = {
    1: 0,  # text
    2: 1,  # title
    3: 2,  # list
    4: 3,  # table
    5: 4,  # figure
}

DOCLAYNET_CLASSES = {
    0: "Caption",
    1: "Footnote",
    2: "Formula",
    3: "List-item",
    4: "Page-footer",
    5: "Page-header",
    6: "Picture",
    7: "Section-header",
    8: "Table",
    9: "Text",
    10: "Title",
}


def get_augmentation_transform(config: dict, batch_scale: int | None = None) -> A.Compose:
    """Create document-specific augmentation pipeline.

    Augmentations simulate realistic document capture conditions:
    - Multi-scale training: resize to specified scale (same per batch)
    - Perspective transforms: camera angles and scanning perspectives
    - Elastic transforms: paper warping and book spine curvature
    - Blur: poor quality scans or camera shake
    - Compression: different JPEG quality levels
    - Lighting: shadows and uneven illumination

    Args:
        config: Augmentation configuration dict with keys:
            - multi_scale_sizes: list of image sizes for multi-scale training (default: [512])
            - rotate_limit: rotation angle limit in degrees (default: 5)
            - brightness_contrast: brightness/contrast variation limit (default: 0.2)
            - noise_std: gaussian noise standard deviation (default: 0.01)
        batch_scale: If provided, use this scale instead of choosing randomly
                    (required for multi-scale to work with batching)

    Returns:
        Albumentations composition with bbox transforms
    """
    # Multi-scale training: resize to batch_scale (chosen once per batch)
    # All images in a batch must have the same size for torch.stack()
    multi_scale_sizes = config.get("multi_scale_sizes", [512])

    if batch_scale is not None:
        # Use provided batch scale
        resize_size = batch_scale
    elif len(multi_scale_sizes) == 1:
        # Single scale
        resize_size = multi_scale_sizes[0]
    else:
        # Should not happen - batch_scale should always be provided for multi-scale
        # Fall back to first size
        resize_size = multi_scale_sizes[0]

    transforms = [A.Resize(resize_size, resize_size, p=1.0)]

    # Add remaining augmentations
    transforms.extend(
        [
            # Perspective distortion from camera angles or scanning
            A.Perspective(scale=(0.02, 0.05), p=0.3),
            # Elastic deformation for paper warping and book spine curvature
            A.ElasticTransform(
                alpha=30,
                sigma=5,
                p=0.2,
            ),
            # Slight rotation for skewed documents
            A.Rotate(
                limit=config.get("rotate_limit", 5),
                border_mode=0,
                p=0.5,
            ),
            # Lighting variations
            A.RandomBrightnessContrast(
                brightness_limit=config.get("brightness_contrast", 0.2),
                contrast_limit=config.get("brightness_contrast", 0.2),
                p=0.5,
            ),
            # Blur from poor scan quality or motion
            A.OneOf(
                [
                    A.MotionBlur(blur_limit=3, p=1.0),
                    A.GaussianBlur(blur_limit=3, p=1.0),
                ],
                p=0.3,
            ),
            # JPEG compression artifacts
            A.ImageCompression(quality_range=(75, 100), p=0.3),
            # Scanner noise
            A.GaussNoise(
                std_range=(0.0, config.get("noise_std", 0.01)),
                p=0.2,
            ),
        ]
    )

    return A.Compose(
        transforms,
        bbox_params=A.BboxParams(
            format="coco",
            label_fields=["category_ids"],
            min_visibility=0.3,
        ),
    )


def format_annotations_for_detr(
    image_id: int,
    annotations: list[dict],
) -> dict:
    """Format COCO annotations for DETR model.

    Args:
        image_id: Unique image identifier
        annotations: List of annotation dicts with bbox, category_id, area, iscrowd

    Returns:
        Dict with image_id and formatted annotations
    """
    return {
        "image_id": image_id,
        "annotations": annotations,
    }


def apply_augmentations(
    examples: dict,
    config: dict,
) -> dict:
    """Apply augmentations to a batch of examples.

    For multi-scale training, chooses one scale randomly per batch
    to ensure all images in the batch have the same size (required for torch.stack).

    Args:
        examples: Batch dict with 'image' and 'annotations' keys
        config: Augmentation config dict (includes multi_scale_sizes)

    Returns:
        Augmented batch with same structure
    """
    import random

    # Choose scale once per batch for multi-scale training
    multi_scale_sizes = config.get("multi_scale_sizes", [512])
    if len(multi_scale_sizes) > 1:
        batch_scale = random.choice(multi_scale_sizes)
    else:
        batch_scale = multi_scale_sizes[0]

    # Create transform with chosen batch scale
    transform = get_augmentation_transform(config, batch_scale=batch_scale)

    images = []
    annotations = []

    for image, anns in zip(examples["image"], examples["annotations"], strict=False):
        # Convert PIL image to numpy
        image_np = np.array(image.convert("RGB"))

        # Extract bboxes and labels
        bboxes = [ann["bbox"] for ann in anns]
        category_ids = [ann["category_id"] for ann in anns]

        # Apply augmentations
        transformed = transform(
            image=image_np,
            bboxes=bboxes,
            category_ids=category_ids,
        )

        # Update annotations with transformed bboxes
        transformed_anns = []
        for bbox, _cat_id, orig_ann in zip(
            transformed["bboxes"],
            transformed["category_ids"],
            anns,
            strict=False,
        ):
            ann_copy = orig_ann.copy()
            ann_copy["bbox"] = list(bbox)
            # Recalculate area
            ann_copy["area"] = bbox[2] * bbox[3]
            transformed_anns.append(ann_copy)

        images.append(transformed["image"])
        annotations.append(transformed_anns)

    examples["image"] = images
    examples["annotations"] = annotations
    return examples


def load_publaynet(
    split: str = "train",
    cache_dir: str | None = None,
) -> tuple:
    """Load PubLayNet dataset.

    Args:
        split: Dataset split ('train' or 'val')
        cache_dir: Optional cache directory path

    Returns:
        Tuple of (dataset, class labels dict)
    """
    dataset = load_dataset(
        "shunk031/PubLayNet",
        split=split,
        cache_dir=cache_dir,
        trust_remote_code=True,
    )
    return dataset, PUBLAYNET_CLASSES


def load_doclaynet(
    split: str = "train",
    cache_dir: str | None = None,
) -> tuple:
    """Load DocLayNet dataset.

    Args:
        split: Dataset split ('train', 'val', or 'test')
        cache_dir: Optional cache directory path

    Returns:
        Tuple of (dataset, class labels dict)
    """
    dataset = load_dataset(
        "ds4sd/DocLayNet",
        split=split,
        cache_dir=cache_dir,
        trust_remote_code=True,
    )
    return dataset, DOCLAYNET_CLASSES


def collate_fn(batch: list[dict]) -> dict:
    """Collate function for batching DETR inputs.

    Args:
        batch: List of dicts with 'pixel_values', 'pixel_mask', 'labels'

    Returns:
        Batched dict with stacked tensors
    """
    pixel_values = torch.stack([item["pixel_values"] for item in batch])
    pixel_mask = torch.stack([item["pixel_mask"] for item in batch])
    labels = [item["labels"] for item in batch]

    return {
        "pixel_values": pixel_values,
        "pixel_mask": pixel_mask,
        "labels": labels,
    }


def prepare_dataset_for_training(
    dataset_name: str,
    split: str,
    image_processor: AutoImageProcessor,
    augmentation_config: dict | None = None,
    cache_dir: str | None = None,
    max_samples: int | None = None,
) -> tuple:
    """Prepare dataset for training with preprocessing and augmentation.

    Args:
        dataset_name: 'publaynet' or 'doclaynet'
        split: Dataset split to load
        image_processor: HuggingFace image processor for DETR
        augmentation_config: Optional augmentation configuration (only used for 'train')
        cache_dir: Optional cache directory
        max_samples: If set and split != 'train', limit dataset to this many samples.

    Returns:
        Tuple of (processed dataset, class labels dict)
    """
    # Load dataset
    if dataset_name.lower() == "publaynet":
        dataset, class_labels = load_publaynet(split, cache_dir)
    elif dataset_name.lower() == "doclaynet":
        dataset, class_labels = load_doclaynet(split, cache_dir)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    # Optional: limit number of samples for non-train splits
    if max_samples is not None and split != "train":
        # If you want a random subset instead of the first N, shuffle here:
        # dataset = dataset.shuffle(seed=42)
        n = min(max_samples, len(dataset))
        dataset = dataset.select(range(n))

    # Apply augmentations if provided (train only)
    if augmentation_config and split == "train":
        dataset = dataset.with_transform(
            lambda examples: apply_augmentations(examples, augmentation_config)
        )

    def preprocess_batch(examples):
        """Preprocess batch with image processor."""
        images = [img.convert("RGB") for img in examples["image"]]

        # Format annotations for DETR - convert from dataset format to COCO format
        annotations = []
        for img_id, anns_dict in zip(examples["image_id"], examples["annotations"], strict=False):
            # Convert dict of lists to list of dicts
            num_objects = len(anns_dict["bbox"])
            anns_list = []
            for i in range(num_objects):
                # Remap category IDs from dataset format to 0-indexed model format
                dataset_cat_id = anns_dict["category_id"][i]
                model_cat_id = PUBLAYNET_ID_MAPPING.get(dataset_cat_id, dataset_cat_id - 1)

                anns_list.append(
                    {
                        "bbox": anns_dict["bbox"][i],
                        "category_id": model_cat_id,
                        "area": anns_dict["area"][i],
                        "iscrowd": anns_dict["iscrowd"][i],
                    }
                )

            annotations.append(
                {
                    "image_id": img_id,
                    "annotations": anns_list,
                }
            )

        # Process with HuggingFace processor
        encoding = image_processor(
            images=images,
            annotations=annotations,
            return_tensors="pt",
        )

        # Convert labels to list format for batching
        labels = list(encoding["labels"])

        return {
            "pixel_values": encoding["pixel_values"],
            "pixel_mask": encoding["pixel_mask"],
            "labels": labels,
        }

    # Apply preprocessing
    dataset = dataset.with_transform(preprocess_batch)

    return dataset, class_labels


def visualize_augmentations(
    dataset_name: str,
    num_samples: int = 4,
    output_dir: str = "outputs/augmentation_samples",
    cache_dir: str | None = None,
) -> None:
    """Generate sample images showing augmentation effects.

    Args:
        dataset_name: 'publaynet' or 'doclaynet'
        num_samples: Number of samples to generate
        output_dir: Directory to save visualization images
        cache_dir: Optional cache directory for datasets
    """
    from PIL import Image, ImageDraw, ImageFont

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load dataset (use val split which is smaller for faster loading)
    if dataset_name.lower() == "publaynet":
        dataset, class_labels = load_publaynet("val", cache_dir)
    elif dataset_name.lower() == "doclaynet":
        dataset, class_labels = load_doclaynet("val", cache_dir)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    # Create augmentation transform (use single scale for visualization)
    aug_config = {
        "rotate_limit": 5,
        "brightness_contrast": 0.2,
        "noise_std": 0.01,
        "multi_scale_sizes": [512],  # Single scale for vis
    }
    transform = get_augmentation_transform(aug_config, batch_scale=512)

    print(f"Generating {num_samples} augmentation samples...")

    for idx in range(num_samples):
        # Get sample
        sample = dataset[idx]
        image = sample["image"]
        annotations = sample["annotations"]

        # Convert to numpy for augmentation
        image_np = np.array(image.convert("RGB"))

        # Extract bboxes and labels
        bboxes = [ann["bbox"] for ann in annotations]
        category_ids = [ann["category_id"] for ann in annotations]

        # Apply augmentations
        augmented = transform(
            image=image_np,
            bboxes=bboxes,
            category_ids=category_ids,
        )

        # Draw bboxes on both original and augmented
        def draw_bboxes(img_np, boxes, labels):
            img_draw = Image.fromarray(img_np)
            draw = ImageDraw.Draw(img_draw)

            colors = [
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

            for bbox, label in zip(boxes, labels, strict=False):
                x, y, w, h = bbox
                color = colors[int(label) % len(colors)]

                # Draw rectangle
                draw.rectangle([x, y, x + w, y + h], outline=color, width=3)

                # Draw label
                label_name = class_labels.get(int(label), str(int(label)))
                try:
                    font = ImageFont.truetype(
                        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16
                    )
                except Exception:
                    font = ImageFont.load_default()

                # Draw label background
                text_bbox = draw.textbbox((x, y - 20), label_name, font=font)
                draw.rectangle(text_bbox, fill=color)
                draw.text((x, y - 20), label_name, fill="white", font=font)

            return np.array(img_draw)

        # Draw bboxes
        original_with_boxes = draw_bboxes(image_np, bboxes, category_ids)
        augmented_with_boxes = draw_bboxes(
            augmented["image"], augmented["bboxes"], augmented["category_ids"]
        )

        # Create side-by-side comparison
        h, w = image_np.shape[:2]
        comparison = np.zeros((h, w * 2, 3), dtype=np.uint8)
        comparison[:, :w] = original_with_boxes
        comparison[:, w:] = augmented_with_boxes

        # Add labels
        comparison_img = Image.fromarray(comparison)
        draw = ImageDraw.Draw(comparison_img)
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 24)
        except Exception:
            font = ImageFont.load_default()

        draw.text(
            (20, 20), "Original", fill="white", font=font, stroke_width=2, stroke_fill="black"
        )
        draw.text(
            (w + 20, 20), "Augmented", fill="white", font=font, stroke_width=2, stroke_fill="black"
        )

        # Save
        output_file = output_path / f"sample_{idx:02d}.png"
        comparison_img.save(output_file)
        print(f"Saved: {output_file}")

    print(f"\nSamples saved to {output_path}")


def preprocess_cli() -> None:
    """CLI entry point for data preprocessing."""
    import argparse

    parser = argparse.ArgumentParser(description="Preprocess datasets for training")
    parser.add_argument(
        "command",
        choices=["load", "visualize"],
        help="Command to run: 'load' to verify dataset loading, 'visualize' to generate augmentation samples",
    )
    parser.add_argument(
        "dataset",
        choices=["publaynet", "doclaynet"],
        help="Dataset to use",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="Cache directory for datasets",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=4,
        help="Number of samples to generate (for visualize command)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/augmentation_samples",
        help="Output directory for visualization samples",
    )
    args = parser.parse_args()

    if args.command == "load":
        print(f"Loading {args.dataset} dataset...")
        if args.dataset == "publaynet":
            train_ds, labels = load_publaynet("train", args.cache_dir)
            val_ds, _ = load_publaynet("val", args.cache_dir)
        else:
            train_ds, labels = load_doclaynet("train", args.cache_dir)
            val_ds, _ = load_doclaynet("val", args.cache_dir)

        print("\nDataset loaded successfully!")
        print(f"Train samples: {len(train_ds)}")
        print(f"Val samples: {len(val_ds)}")
        print(f"Classes: {labels}")

    elif args.command == "visualize":
        visualize_augmentations(
            args.dataset,
            num_samples=args.num_samples,
            output_dir=args.output_dir,
            cache_dir=args.cache_dir,
        )
