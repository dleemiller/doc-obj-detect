"""Albumentations-based augmentation pipeline with custom batch-level augmentations.

Following albumentations documentation, Mosaic and MixUp are batch-based augmentations
that require custom implementation (beyond albumentations scope).
https://albumentations.ai/docs/3-basic-usage/choosing-augmentations/#beyond-albumentations-batch-based-augmentations

We use official albumentations for per-image transforms (flips, rotations, etc.)
and implement Mosaic/MixUp as custom batch-level augmentations.
"""

from __future__ import annotations

import random
from typing import Any

import albumentations as A
import numpy as np
from PIL import Image


class AlbumentationsAugmentor:
    """Augmentation pipeline using albumentations for per-image transforms.

    Custom implementations for batch-based augmentations (Mosaic, MixUp) that
    are beyond albumentations' scope per their documentation.
    """

    def __init__(self, config: dict[str, Any] | None):
        self.config = config or {}
        self.current_epoch = 0
        # Cache for batch-level augmentations (mosaic/mixup)
        self.sample_cache: list[tuple[np.ndarray, list, list]] = []
        self.cache_size = 100

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

    def apply_mosaic(
        self,
        image: np.ndarray,
        bboxes: list,
        category_ids: list,
        target_size: int,
    ) -> tuple[np.ndarray, list, list]:
        """
        Apply mosaic augmentation by combining 4 images into a 2x2 grid.

        Custom implementation as recommended by albumentations documentation
        for batch-based augmentations.

        Args:
            image: Primary image (H, W, 3)
            bboxes: List of bboxes in COCO format [x, y, w, h]
            category_ids: List of category IDs
            target_size: Output size (square)

        Returns:
            Tuple of (mosaic_image, mosaic_bboxes, mosaic_category_ids)
        """
        # Need 3 more images from cache
        if len(self.sample_cache) < 3:
            return image, bboxes, category_ids

        # Sample 3 additional images
        indices = random.sample(range(len(self.sample_cache)), 3)
        images = [image] + [self.sample_cache[i][0] for i in indices]
        all_bboxes = [bboxes] + [self.sample_cache[i][1] for i in indices]
        all_category_ids = [category_ids] + [self.sample_cache[i][2] for i in indices]

        # Random center point (0.4-0.6 for balanced quadrants)
        cx = int(target_size * random.uniform(0.4, 0.6))
        cy = int(target_size * random.uniform(0.4, 0.6))

        # Create mosaic canvas
        mosaic_img = np.zeros((target_size, target_size, 3), dtype=np.uint8)
        mosaic_bboxes = []
        mosaic_category_ids = []

        # Placement coordinates for 4 quadrants: top-left, top-right, bottom-left, bottom-right
        placements = [
            (0, 0, cx, cy),  # top-left
            (cx, 0, target_size, cy),  # top-right
            (0, cy, cx, target_size),  # bottom-left
            (cx, cy, target_size, target_size),  # bottom-right
        ]

        for idx, (img, boxes, cats) in enumerate(
            zip(images, all_bboxes, all_category_ids, strict=False)
        ):
            x1, y1, x2, y2 = placements[idx]
            quad_w = x2 - x1
            quad_h = y2 - y1

            # Resize image to fit quadrant
            h, w = img.shape[:2]
            scale = min(quad_w / w, quad_h / h)
            new_w = int(w * scale)
            new_h = int(h * scale)

            if new_w > 0 and new_h > 0:
                img_resized = np.array(Image.fromarray(img).resize((new_w, new_h), Image.BILINEAR))

                # Place in quadrant (centered)
                offset_x = x1 + (quad_w - new_w) // 2
                offset_y = y1 + (quad_h - new_h) // 2
                mosaic_img[offset_y : offset_y + new_h, offset_x : offset_x + new_w] = img_resized

                # Transform bboxes
                for bbox, cat in zip(boxes, cats, strict=False):
                    x_min, y_min, box_w, box_h = bbox[:4]

                    # Scale and offset bbox
                    new_x = (x_min * scale) + offset_x
                    new_y = (y_min * scale) + offset_y
                    new_box_w = box_w * scale
                    new_box_h = box_h * scale

                    # Clip to mosaic boundaries
                    new_x = max(0, min(new_x, target_size))
                    new_y = max(0, min(new_y, target_size))
                    new_box_w = min(target_size - new_x, new_box_w)
                    new_box_h = min(target_size - new_y, new_box_h)

                    # Only keep if box is large enough (at least 2x2 pixels)
                    if new_box_w >= 2 and new_box_h >= 2:
                        mosaic_bboxes.append([new_x, new_y, new_box_w, new_box_h])
                        mosaic_category_ids.append(cat)

        return mosaic_img, mosaic_bboxes, mosaic_category_ids

    def apply_mixup(
        self,
        image: np.ndarray,
        bboxes: list,
        category_ids: list,
        alpha: float = 0.2,
    ) -> tuple[np.ndarray, list, list]:
        """
        Apply mix-up augmentation by blending two images.

        Custom implementation as MixUp is beyond albumentations scope (batch-based).

        Args:
            image: Primary image (H, W, 3)
            bboxes: List of bboxes in COCO format [x, y, w, h]
            category_ids: List of category IDs
            alpha: Beta distribution parameter for mixing

        Returns:
            Tuple of (mixed_image, mixed_bboxes, mixed_category_ids)
        """
        if len(self.sample_cache) < 1:
            return image, bboxes, category_ids

        # Sample one cached sample
        idx = random.randint(0, len(self.sample_cache) - 1)
        image2, bboxes2, category_ids2 = self.sample_cache[idx]

        # Sample mixing coefficient from Beta distribution
        lam = np.random.beta(alpha, alpha) if alpha > 0 else 0.5

        # Resize images to same size if needed
        h1, w1 = image.shape[:2]
        h2, w2 = image2.shape[:2]
        if (h1, w1) != (h2, w2):
            image2 = np.array(Image.fromarray(image2).resize((w1, h1), Image.BILINEAR))

        # Blend images
        mixed_image = (lam * image + (1 - lam) * image2).astype(np.uint8)

        # Combine bboxes from both images (keep all)
        mixed_bboxes = bboxes + bboxes2
        mixed_category_ids = category_ids + category_ids2

        return mixed_image, mixed_bboxes, mixed_category_ids

    def augment(self, examples: dict) -> dict:
        """Apply augmentation to a HF batch.

        Applies batch-level augmentations (mosaic/mixup) first, then per-image
        transforms via albumentations.
        """
        mosaic_cfg = self.config.get("mosaic", {})
        mosaic_prob = mosaic_cfg.get("probability", 0.0)
        disable_after_epoch = mosaic_cfg.get("disable_after_epoch", float("inf"))

        # Disable mosaic after specified epoch
        if self.current_epoch >= disable_after_epoch:
            mosaic_prob = 0.0

        mixup_cfg = self.config.get("mixup", {})
        mixup_prob = mixup_cfg.get("probability", 0.0)
        mixup_alpha = mixup_cfg.get("alpha", 0.2)

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

            # Apply mosaic or mixup (mutually exclusive, before per-image transforms)
            aug_roll = random.random()
            if aug_roll < mosaic_prob and len(self.sample_cache) >= 3:
                # Apply mosaic
                target_size = self.sample_scale()
                image_np, clean_boxes, clean_labels = self.apply_mosaic(
                    image_np, clean_boxes, clean_labels, target_size
                )
            elif aug_roll < (mosaic_prob + mixup_prob) and len(self.sample_cache) >= 1:
                # Apply mixup
                image_np, clean_boxes, clean_labels = self.apply_mixup(
                    image_np, clean_boxes, clean_labels, mixup_alpha
                )

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
