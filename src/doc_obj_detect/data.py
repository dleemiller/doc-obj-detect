"""Data loading, preprocessing, and augmentation for document object detection."""

import math
import random

import albumentations as A
import numpy as np
import torch
from datasets import load_dataset
from PIL import Image
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
    - Perspective & elastic warps
    - Rotation, brightness/contrast, noise, blur, compression

    Args:
        config: Augmentation configuration dict.
        batch_scale: Optional override when multi-scale selection is handled externally.

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

    force_square = config.get("force_square_resize", False)
    max_long_side = config.get("max_long_side")

    if force_square:
        transforms = [A.Resize(resize_size, resize_size, p=1.0)]
    else:
        transforms = [
            A.SmallestMaxSize(max_size=resize_size, p=1.0),
        ]
        if max_long_side:
            transforms.append(A.LongestMaxSize(max_size=max_long_side, p=1.0))

    # Add remaining augmentations
    perspective_cfg = (config or {}).get("perspective", {})
    perspective_prob = perspective_cfg.get("probability", 0.3)
    if perspective_prob > 0:
        transforms.append(
            A.Perspective(
                scale=(
                    perspective_cfg.get("scale_min", 0.02),
                    perspective_cfg.get("scale_max", 0.05),
                ),
                p=perspective_prob,
            )
        )

    elastic_cfg = (config or {}).get("elastic", {})
    elastic_prob = elastic_cfg.get("probability", 0.2)
    if elastic_prob > 0:
        transforms.append(
            A.ElasticTransform(
                alpha=elastic_cfg.get("alpha", 30),
                sigma=elastic_cfg.get("sigma", 5),
                p=elastic_prob,
            )
        )

    rotate_prob = (config or {}).get("rotate_prob", 0.5)
    if rotate_prob > 0:
        transforms.append(
            A.Rotate(
                limit=(config or {}).get("rotate_limit", 5),
                border_mode=0,
                p=rotate_prob,
            )
        )

    brightness_cfg = (config or {}).get("brightness_contrast", {})
    brightness_prob = brightness_cfg.get("probability", 0.5)
    if brightness_prob > 0:
        limit = brightness_cfg.get("limit", 0.2)
        transforms.append(
            A.RandomBrightnessContrast(
                brightness_limit=limit,
                contrast_limit=limit,
                p=brightness_prob,
            )
        )

    blur_cfg = (config or {}).get("blur", {})
    blur_prob = blur_cfg.get("probability", 0.3)
    if blur_prob > 0:
        blur_limit = blur_cfg.get("blur_limit", 3)
        transforms.append(
            A.OneOf(
                [
                    A.MotionBlur(blur_limit=blur_limit, p=1.0),
                    A.GaussianBlur(blur_limit=blur_limit, p=1.0),
                ],
                p=blur_prob,
            )
        )

    compression_cfg = (config or {}).get("compression", {})
    compression_prob = compression_cfg.get("probability", 0.3)
    if compression_prob > 0:
        transforms.append(
            A.ImageCompression(
                quality_range=(
                    compression_cfg.get("quality_min", 75),
                    compression_cfg.get("quality_max", 100),
                ),
                p=compression_prob,
            )
        )

    noise_cfg = (config or {}).get("noise", {})
    noise_prob = noise_cfg.get("probability", 0.2)
    if noise_prob > 0:
        transforms.append(
            A.GaussNoise(
                std_range=(
                    noise_cfg.get("std_min", 0.0),
                    noise_cfg.get("std_max", 0.01),
                ),
                p=noise_prob,
            )
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

    Supports both:
      - annotations as dict-of-lists (HF detection format)
      - annotations as list-of-dicts (older / custom formats)

    Returns examples with:
      - image: list of augmented images (np arrays)
      - annotations: list of dict-of-lists with keys: bbox, category_id, area, iscrowd
    """
    # Choose scale once per batch for multi-scale training
    multi_scale_sizes = config.get("multi_scale_sizes", [512])
    if len(multi_scale_sizes) > 1:
        batch_scale = random.choice(multi_scale_sizes)
    else:
        batch_scale = multi_scale_sizes[0]

    # Create transform with chosen batch scale
    transform = get_augmentation_transform(config, batch_scale=batch_scale)

    images_out: list[np.ndarray] = []
    annotations_out: list[dict] = []

    for image, anns in zip(examples["image"], examples["annotations"], strict=False):
        # Convert PIL image to numpy
        image_np = np.array(image.convert("RGB"))

        # --- Normalize annotations to simple lists for Albumentations ---
        if isinstance(anns, dict):
            # HF detection format: dict-of-lists
            bboxes = anns.get("bbox", [])
            category_ids = anns.get("category_id", [])
        else:
            # list-of-dicts format
            bboxes = [ann["bbox"] for ann in anns]
            category_ids = [ann["category_id"] for ann in anns]

        # --- Filter obviously degenerate input boxes (zero/negative size) ---
        filtered_bboxes = []
        filtered_category_ids = []
        for bbox, cat_id in zip(bboxes, category_ids, strict=False):
            x, y, w, h = bbox[:4]
            # drop boxes with (almost) zero width/height
            if w <= 1e-3 or h <= 1e-3:
                continue
            filtered_bboxes.append(bbox)
            filtered_category_ids.append(cat_id)

        bboxes = filtered_bboxes
        category_ids = filtered_category_ids

        # Albumentations can handle empty bbox lists, so that's fine.
        # Now apply transforms, but be defensive about rare degenerate cases.
        try:
            transformed = transform(
                image=image_np,
                bboxes=bboxes,
                category_ids=category_ids,
            )
        except ValueError as e:
            # Clamp COCO boxes into the valid range if Albumentations complains
            if "Expected x_max" in str(e) or "Expected y_max" in str(e):
                clamped_bboxes = []
                for bbox in bboxes:
                    x_min, y_min, width, height = bbox[:4]
                    x_min = max(0.0, min(1.0, x_min))
                    y_min = max(0.0, min(1.0, y_min))
                    max_width = max(1e-6, 1.0 - x_min - 1e-6)
                    max_height = max(1e-6, 1.0 - y_min - 1e-6)
                    width = max(0.0, min(max_width, width))
                    height = max(0.0, min(max_height, height))
                    clamped_bboxes.append([x_min, y_min, width, height])

                try:
                    transformed = transform(
                        image=image_np,
                        bboxes=clamped_bboxes,
                        category_ids=category_ids,
                    )
                except ValueError as retry_error:
                    print(
                        "[Albumentations] Skipping bbox transform after clamping; "
                        f"original error: {e}; retry error: {retry_error}"
                    )
                    transformed = {
                        "image": image_np,
                        "bboxes": clamped_bboxes,
                        "category_ids": category_ids,
                    }
            else:
                print(f"[Albumentations] Skipping bbox transform for one sample due to error: {e}")
                transformed = {
                    "image": image_np,
                    "bboxes": bboxes,
                    "category_ids": category_ids,
                }

        # --- Rebuild annotations as dict-of-lists (what preprocess_batch expects) ---
        new_anns = {
            "bbox": [],
            "category_id": [],
            "area": [],
            "iscrowd": [],
        }

        for bbox, cat_id in zip(
            transformed["bboxes"],
            transformed["category_ids"],
            strict=False,
        ):
            bbox = list(bbox)
            new_anns["bbox"].append(bbox)
            new_anns["category_id"].append(int(cat_id))
            new_anns["area"].append(float(bbox[2] * bbox[3]))  # w*h in COCO format
            new_anns["iscrowd"].append(0)

        images_out.append(transformed["image"])
        annotations_out.append(new_anns)

    examples["image"] = images_out
    examples["annotations"] = annotations_out
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
    pad_stride: int = 32,
) -> tuple:
    """Prepare dataset for training with preprocessing and augmentation.

    Args:
        dataset_name: 'publaynet' or 'doclaynet'
        split: Dataset split to load
        image_processor: HuggingFace image processor for DETR/DFine
        augmentation_config: Optional augmentation configuration (only used for 'train')
        cache_dir: Optional cache directory
        max_samples: If set and split != 'train', limit dataset to this many samples.
        pad_stride: Align padding to this stride (match detector feature-map stride).

    Returns:
        Tuple of (processed dataset, class labels dict)
    """
    # Load dataset
    if dataset_name.lower() == "publaynet":
        dataset, class_labels = load_publaynet(split, cache_dir)
        is_publaynet = True
    elif dataset_name.lower() == "doclaynet":
        dataset, class_labels = load_doclaynet(split, cache_dir)
        is_publaynet = False
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    # Optional: limit number of samples for non-train splits
    if max_samples is not None and split != "train":
        n = min(max_samples, len(dataset))
        dataset = dataset.select(range(n))

    def preprocess_batch(examples: dict) -> dict:
        """Preprocess batch with image processor."""
        # examples["image"] may contain PIL Images (no aug) or np.ndarray (after aug)
        images = []
        for img in examples["image"]:
            if isinstance(img, np.ndarray):
                # Ensure uint8 for images coming from Albumentations
                if img.dtype != np.uint8:
                    img = np.clip(img, 0, 255).astype(np.uint8)
                img = Image.fromarray(img)
            # Now img is PIL.Image.Image
            images.append(img.convert("RGB"))

        # Format annotations for DETR - convert from dataset format to COCO format
        annotations = []
        for img_id, anns_dict in zip(examples["image_id"], examples["annotations"], strict=False):
            # anns_dict is dict-of-lists: "bbox", "category_id", "area", "iscrowd"
            num_objects = len(anns_dict["bbox"])
            anns_list = []
            for i in range(num_objects):
                dataset_cat_id = anns_dict["category_id"][i]

                if is_publaynet:
                    # Remap category IDs from dataset format to 0-indexed model format
                    model_cat_id = PUBLAYNET_ID_MAPPING.get(dataset_cat_id, dataset_cat_id - 1)
                else:
                    # DocLayNet is (effectively) already 0-indexed
                    model_cat_id = dataset_cat_id

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

        def infer_resized_shape(height: int, width: int, size_dict: dict) -> tuple[int, int]:
            height = max(1, int(height))
            width = max(1, int(width))
            if not isinstance(size_dict, dict):
                return height, width
            if "height" in size_dict and "width" in size_dict:
                return max(1, int(size_dict["height"])), max(1, int(size_dict["width"]))
            if "shortest_edge" in size_dict:
                target_short = size_dict["shortest_edge"]
                short, long = sorted((height, width))
                scale = float(target_short) / max(1.0, float(short))
                new_height = max(1, int(round(height * scale)))
                new_width = max(1, int(round(width * scale)))
                max_long = size_dict.get("longest_edge")
                if max_long is not None:
                    current_long = max(new_height, new_width)
                    if current_long > max_long:
                        long_scale = float(max_long) / float(current_long)
                        new_height = max(1, int(round(new_height * long_scale)))
                        new_width = max(1, int(round(new_width * long_scale)))
                return new_height, new_width
            if "max_height" in size_dict and "max_width" in size_dict:
                return (
                    min(max(1, int(size_dict["max_height"])), height),
                    min(max(1, int(size_dict["max_width"])), width),
                )
            return height, width

        processor_size = getattr(image_processor, "size", None)
        do_resize = getattr(image_processor, "do_resize", False)
        candidate_sizes: list[tuple[int, int]] = []
        for img in images:
            h = getattr(img, "height", 1)
            w = getattr(img, "width", 1)
            if do_resize and isinstance(processor_size, dict):
                candidate_sizes.append(infer_resized_shape(h, w, processor_size))
            else:
                candidate_sizes.append((max(1, h), max(1, w)))

        target_height = max((h for h, _ in candidate_sizes), default=1)
        target_width = max((w for _, w in candidate_sizes), default=1)

        # Pad to stride-aligned shapes (match detector strides)
        stride = max(1, pad_stride)
        pad_height = max(1, math.ceil(target_height / stride) * stride)
        pad_width = max(1, math.ceil(target_width / stride) * stride)

        encoding = image_processor(
            images=images,
            annotations=annotations,
            return_tensors="pt",
            pad_size={"height": pad_height, "width": pad_width},
        )

        labels = list(encoding["labels"])

        return {
            "pixel_values": encoding["pixel_values"],
            "pixel_mask": encoding["pixel_mask"],
            "labels": labels,
        }

    # Combine (optional) augmentation + preprocessing into one transform
    if augmentation_config is not None and split == "train":

        def full_transform(examples: dict) -> dict:
            augmented = apply_augmentations(examples, augmentation_config)
            return preprocess_batch(augmented)
    else:

        def full_transform(examples: dict) -> dict:
            return preprocess_batch(examples)

    dataset = dataset.with_transform(full_transform)

    return dataset, class_labels
