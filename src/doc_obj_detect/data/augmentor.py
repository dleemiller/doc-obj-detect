"""Albumentations-based augmentation pipeline with custom mosaic support.

Following albumentations documentation, Mosaic is a batch-based augmentation that
requires extra metadata management beyond the standard per-image transforms.
https://albumentations.ai/docs/3-basic-usage/choosing-augmentations/#beyond-albumentations-batch-based-augmentations

We use official albumentations for per-image transforms (flips, rotations, etc.)
and integrate Mosaic as a batch-level augmentation.
"""

from __future__ import annotations

import random
from typing import Any

import albumentations as A
import cv2
import numpy as np


class AlbumentationsAugmentor:
    """Augmentation pipeline using albumentations for per-image transforms and mosaic."""

    def __init__(self, config: dict[str, Any] | None):
        self.config = config or {}
        self.current_epoch = 0
        # Cache for batch-level augmentations (mosaic)
        self.sample_cache: list[tuple[np.ndarray, list, list]] = []
        self.cache_size = 100
        self.mosaic_metadata_key = "mosaic_metadata"

    def set_epoch(self, epoch: int) -> None:
        """Set current epoch for epoch-dependent augmentations."""
        self.current_epoch = epoch

    def sample_scale(self) -> int:
        """Sample a scale from multi_scale_sizes config."""
        sizes = self.config.get("multi_scale_sizes", [512])
        if not sizes:
            return 512
        if len(sizes) == 1:
            return sizes[0]
        return random.choice(sizes)

    def build_transform(self, batch_scale: int | None = None) -> A.Compose:
        """Build albumentations transform pipeline for per-image augmentations."""
        cfg = self.config
        resize_size = batch_scale or self.sample_scale()
        force_square = cfg.get("force_square_resize", False)
        max_long_side = cfg.get("max_long_side")

        transforms: list[A.BasicTransform] = []

        # Resize (per-image operation)
        if force_square:
            transforms.append(A.Resize(resize_size, resize_size, p=1.0))
        else:
            transforms.append(A.SmallestMaxSize(max_size=resize_size, p=1.0))
            if max_long_side:
                transforms.append(A.LongestMaxSize(max_size=max_long_side, p=1.0))

        # Perspective transform
        perspective_cfg = cfg.get("perspective", {})
        if perspective_cfg.get("probability", 0) > 0:
            transforms.append(
                A.Perspective(
                    scale=(
                        perspective_cfg.get("scale_min", 0.02),
                        perspective_cfg.get("scale_max", 0.05),
                    ),
                    p=perspective_cfg["probability"],
                )
            )

        # Elastic transform
        elastic_cfg = cfg.get("elastic", {})
        if elastic_cfg.get("probability", 0) > 0:
            transforms.append(
                A.ElasticTransform(
                    alpha=elastic_cfg.get("alpha", 30),
                    sigma=elastic_cfg.get("sigma", 5),
                    p=elastic_cfg["probability"],
                )
            )

        # Flip augmentations (official albumentations)
        hflip_prob = cfg.get("horizontal_flip", 0.0)
        if hflip_prob > 0:
            transforms.append(A.HorizontalFlip(p=hflip_prob))

        vflip_prob = cfg.get("vertical_flip", 0.0)
        if vflip_prob > 0:
            transforms.append(A.VerticalFlip(p=vflip_prob))

        # Rotation
        rotate_prob = cfg.get("rotate_prob", 0.5)
        if rotate_prob > 0:
            transforms.append(
                A.Rotate(
                    limit=cfg.get("rotate_limit", 5),
                    border_mode=0,
                    p=rotate_prob,
                )
            )

        # Photometric augmentations
        brightness_cfg = cfg.get("brightness_contrast", {})
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

        blur_cfg = cfg.get("blur", {})
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

        compression_cfg = cfg.get("compression", {})
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

        noise_cfg = cfg.get("noise", {})
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

    def build_mosaic_transform(self, target_size: int) -> A.Compose:
        """Construct Albumentations mosaic transform for the provided target size."""
        mosaic_cfg = self.config.get("mosaic", {})
        grid_yx = tuple(mosaic_cfg.get("grid_yx", (2, 2)))
        center_range = tuple(mosaic_cfg.get("center_range", (0.3, 0.7)))
        fit_mode = mosaic_cfg.get("fit_mode", "cover")
        fill = mosaic_cfg.get("fill", 0)
        fill_mask = mosaic_cfg.get("fill_mask", 0)
        interpolation = mosaic_cfg.get("interpolation", cv2.INTER_LINEAR)
        mask_interpolation = mosaic_cfg.get("mask_interpolation", cv2.INTER_NEAREST)

        return A.Compose(
            [
                A.Mosaic(
                    grid_yx=grid_yx,
                    target_size=(target_size, target_size),
                    cell_shape=(target_size, target_size),
                    center_range=center_range,
                    fit_mode=fit_mode,
                    fill=fill,
                    fill_mask=fill_mask,
                    interpolation=interpolation,
                    mask_interpolation=mask_interpolation,
                    metadata_key=self.mosaic_metadata_key,
                    p=1.0,
                )
            ],
            bbox_params=A.BboxParams(
                format="coco",
                label_fields=["category_ids"],
                min_visibility=0.3,
            ),
        )

    def prepare_mosaic_metadata(self, num_samples: int = 3) -> list[dict[str, Any]]:
        """Build albumentations metadata payload from cached samples."""
        if len(self.sample_cache) < num_samples:
            return []
        indices = random.sample(range(len(self.sample_cache)), num_samples)
        metadata: list[dict[str, Any]] = []
        for idx in indices:
            cached_image, cached_bboxes, cached_cats = self.sample_cache[idx]
            # Validate and clean cached bboxes before mosaic
            valid_boxes = []
            valid_cats = []
            for box, cat in zip(cached_bboxes, cached_cats, strict=False):
                # Ensure box is exactly 4 elements [x, y, w, h]
                if len(box) >= 4:
                    x, y, w, h = float(box[0]), float(box[1]), float(box[2]), float(box[3])
                    # Filter out degenerate boxes
                    if w > 1e-3 and h > 1e-3:
                        valid_boxes.append([x, y, w, h])
                        valid_cats.append(int(cat))

            if valid_boxes:  # Only add metadata if there are valid boxes
                metadata.append(
                    {
                        "image": cached_image.copy(),
                        "bboxes": valid_boxes,
                        "category_ids": valid_cats,
                    }
                )
        return metadata

    def augment(self, examples: dict) -> dict:
        """Apply augmentation to a HF batch with optional mosaic before per-image transforms."""
        mosaic_cfg = self.config.get("mosaic", {})
        mosaic_prob = mosaic_cfg.get("probability", 0.0)
        disable_after_epoch = mosaic_cfg.get("disable_after_epoch", float("inf"))

        # Disable mosaic after specified epoch
        if self.current_epoch >= disable_after_epoch:
            mosaic_prob = 0.0

        transform = self.build_transform()
        images_out: list[np.ndarray] = []
        annotations_out: list[dict] = []

        for image, anns in zip(examples["image"], examples["annotations"], strict=False):
            # Handle both PIL images and numpy arrays
            if isinstance(image, np.ndarray):
                image_np = image
            else:
                image_np = np.array(image.convert("RGB"))

            if isinstance(anns, dict):
                bboxes = anns.get("bbox", [])
                category_ids = anns.get("category_id", [])
            else:
                bboxes = [ann["bbox"] for ann in anns]
                category_ids = [ann["category_id"] for ann in anns]

            # Clean invalid boxes
            clean_boxes = []
            clean_labels = []
            for bbox, cat in zip(bboxes, category_ids, strict=False):
                x, y, w, h = bbox[:4]
                if w <= 1e-3 or h <= 1e-3:
                    continue
                clean_boxes.append([float(x), float(y), float(w), float(h)])
                clean_labels.append(int(cat))

            # Apply mosaic (before per-image transforms)
            aug_roll = random.random()
            if aug_roll < mosaic_prob and len(self.sample_cache) >= 3:
                metadata = self.prepare_mosaic_metadata(3)
                if metadata:
                    target_size = self.sample_scale()
                    mosaic_transform = self.build_mosaic_transform(target_size)
                    mosaic_kwargs = {
                        "image": image_np,
                        "bboxes": clean_boxes,
                        "category_ids": clean_labels,
                        self.mosaic_metadata_key: metadata,
                    }
                    try:
                        mosaic_augmented = mosaic_transform(**mosaic_kwargs)
                        image_np = mosaic_augmented["image"]

                        # Validate and filter mosaic output boxes
                        mosaic_boxes = []
                        mosaic_cats = []
                        img_h, img_w = image_np.shape[:2]
                        for box, cat in zip(
                            mosaic_augmented["bboxes"],
                            mosaic_augmented["category_ids"],
                            strict=False,
                        ):
                            if len(box) >= 4:
                                x, y, w, h = (
                                    float(box[0]),
                                    float(box[1]),
                                    float(box[2]),
                                    float(box[3]),
                                )
                                # Strict validation: box must be within image and have valid dimensions
                                if (
                                    w > 2.0
                                    and h > 2.0
                                    and x >= 0
                                    and y >= 0
                                    and x + w <= img_w
                                    and y + h <= img_h
                                ):
                                    mosaic_boxes.append([x, y, w, h])
                                    mosaic_cats.append(int(cat))

                        clean_boxes = mosaic_boxes
                        clean_labels = mosaic_cats
                    except ValueError:
                        # If mosaic fails, use original boxes
                        # Log the error for debugging
                        pass

            # Update cache (before per-image transforms)
            if len(self.sample_cache) >= self.cache_size:
                self.sample_cache.pop(0)
            self.sample_cache.append((image_np.copy(), clean_boxes.copy(), clean_labels.copy()))

            # Apply per-image transforms (albumentations)
            try:
                augmented = transform(
                    image=image_np,
                    bboxes=clean_boxes,
                    category_ids=clean_labels,
                )
            except ValueError as exc:
                # Handle edge cases (invalid bboxes after augmentation)
                augmented = {
                    "image": image_np,
                    "bboxes": clean_boxes,
                    "category_ids": clean_labels,
                }
                # Try clamping bboxes if needed
                if "Expected x_max" in str(exc) or "Expected y_max" in str(exc):
                    h, w = image_np.shape[:2]
                    clamped = []
                    for bbox in clean_boxes:
                        x_min, y_min, width, height = bbox[:4]
                        x_min = max(0.0, min(float(w - 1), x_min))
                        y_min = max(0.0, min(float(h - 1), y_min))
                        width = max(1.0, min(float(w - x_min), width))
                        height = max(1.0, min(float(h - y_min), height))
                        clamped.append([x_min, y_min, width, height])
                    try:
                        augmented = transform(
                            image=image_np,
                            bboxes=clamped,
                            category_ids=clean_labels,
                        )
                    except ValueError:
                        augmented["bboxes"] = clamped

            # Prepare output annotations
            new_anns = {
                "bbox": [],
                "category_id": [],
                "area": [],
                "iscrowd": [],
            }
            for bbox, cat in zip(augmented["bboxes"], augmented["category_ids"], strict=False):
                bbox = list(bbox)
                new_anns["bbox"].append(bbox)
                new_anns["category_id"].append(int(cat))
                new_anns["area"].append(float(bbox[2] * bbox[3]))
                new_anns["iscrowd"].append(0)

            images_out.append(augmented["image"])
            annotations_out.append(new_anns)

        examples["image"] = images_out
        examples["annotations"] = annotations_out
        return examples
