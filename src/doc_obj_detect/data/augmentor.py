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
from augraphy import (
    AugraphyPipeline,
    BindingsAndFasteners,
    Brightness,
    BrightnessTexturize,
    ColorPaper,
    ColorShift,
    DirtyRollers,
    Dithering,
    DotMatrix,
    DoubleExposure,
    Faxify,
    Folding,
    Gamma,
    GlitchEffect,
    InkBleed,
    InkColorSwap,
    InkMottling,
    Jpeg,
    LinesDegradation,
    LowInkPeriodicLines,
    LowInkRandomLines,
    NoiseTexturize,
    NoisyLines,
    OneOf,
    PatternGenerator,
    ShadowCast,
    Squish,
    SubtleNoise,
    WaterMark,
)


class SobelEdgeExtraction(A.ImageOnlyTransform):
    """Sobel edge detection to enhance document boundaries (DocLayout-YOLO approach).

    This augmentation applies Sobel filtering to extract edges and blends them with
    the original image. It helps the model learn document structure boundaries such as
    table borders, column edges, and text block boundaries.

    Args:
        blend_alpha: Weight of edge image in the blend (0.0-1.0). Higher values create
                    stronger edge emphasis. Default: 0.2 (20% edge, 80% original).
        always_apply: Whether to always apply this transform. Default: False.
        p: Probability of applying the transform. Default: 0.2.
    """

    def __init__(self, blend_alpha: float = 0.2, always_apply: bool = False, p: float = 0.2):
        super().__init__(always_apply, p)
        self.blend_alpha = blend_alpha

    def apply(self, img: np.ndarray, **params) -> np.ndarray:
        """Apply Sobel edge extraction and blend with original image."""
        # Convert to grayscale for edge detection
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Apply Sobel operators in X and Y directions
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

        # Compute gradient magnitude
        magnitude = np.sqrt(sobelx**2 + sobely**2)

        # Normalize to 0-255 range
        magnitude = np.clip(magnitude / magnitude.max() * 255, 0, 255).astype(np.uint8)

        # Convert edge map back to RGB
        edge_rgb = cv2.cvtColor(magnitude, cv2.COLOR_GRAY2BGR)

        # Blend edge map with original image
        blended = cv2.addWeighted(img, 1.0 - self.blend_alpha, edge_rgb, self.blend_alpha, 0)

        return blended

    def get_transform_init_args_names(self) -> tuple[str, ...]:
        """Return parameter names for serialization."""
        return ("blend_alpha",)


class AlbumentationsAugmentor:
    """Augmentation pipeline using albumentations for per-image transforms and mosaic."""

    def __init__(self, config: dict[str, Any] | None):
        self.config = config or {}
        self.current_epoch = 0
        # Cache for batch-level augmentations (mosaic)
        self.sample_cache: list[tuple[np.ndarray, list, list]] = []
        self.cache_size = 20  # Reduced from 100 to lower RAM usage with multiple workers
        self.mosaic_metadata_key = "mosaic_metadata"

        # Initialize Augraphy pipeline if enabled
        self.augraphy_pipeline = None
        augraphy_cfg = self.config.get("augraphy", {})
        if augraphy_cfg.get("enabled", False):
            self.augraphy_pipeline = self._build_augraphy_pipeline(augraphy_cfg)

    def set_epoch(self, epoch: int) -> None:
        """Set current epoch for epoch-dependent augmentations."""
        self.current_epoch = epoch

    def _build_augraphy_pipeline(self, augraphy_cfg: dict[str, Any]) -> AugraphyPipeline:
        """Build Augraphy pipeline for document-specific augmentations.

        Augraphy provides realistic document degradation effects like:
        - Paper texture and quality variations
        - Ink bleeding and fading
        - Scanner artifacts
        - Printing defects
        - Document aging effects
        """
        from augraphy import InkMottling as InkMottlingPost

        # Per-phase probabilities (allow fine-grained control)
        # Use per-phase probabilities if specified, otherwise fall back to global probability
        default_prob = augraphy_cfg.get("probability", 0.5)
        prob_ink = augraphy_cfg.get("ink_probability", default_prob)
        prob_paper = augraphy_cfg.get("paper_probability", default_prob)
        prob_post = augraphy_cfg.get("post_probability", default_prob)

        # Mirror default pipeline structure but keep fast (>=0.5 img/sec) and avoid crop/rotate
        ink_phase = [
            InkColorSwap(
                ink_swap_color="random",
                ink_swap_sequence_number_range=(5, 10),
                ink_swap_min_width_range=(2, 3),
                ink_swap_max_width_range=(100, 120),
                ink_swap_min_height_range=(2, 3),
                ink_swap_max_height_range=(100, 120),
                ink_swap_min_area_range=(10, 20),
                ink_swap_max_area_range=(400, 500),
                p=prob_ink * 0.2,
            ),
            LinesDegradation(
                line_roi=(0.0, 0.0, 1.0, 1.0),
                line_gradient_range=(32, 255),
                line_gradient_direction=(0, 2),
                line_split_probability=(0.2, 0.4),
                line_replacement_value=(250, 255),
                line_min_length=(30, 40),
                line_long_to_short_ratio=(5, 7),
                line_replacement_probability=(0.4, 0.5),
                line_replacement_thickness=(1, 3),
                p=prob_ink * 0.2,
            ),
            OneOf(
                [
                    Dithering(
                        dither=random.choice(["ordered", "floyd-steinberg"]),
                        order=(3, 5),
                    ),
                    InkBleed(
                        intensity_range=(0.1, 0.2),
                        kernel_size=random.choice([(7, 7), (5, 5), (3, 3)]),
                        severity=(0.4, 0.6),
                    ),
                ],
                p=prob_ink * 0.2,
            ),
            InkMottling(
                ink_mottling_alpha_range=(0.2, 0.3),
                ink_mottling_noise_scale_range=(2, 2),
                ink_mottling_gaussian_kernel_range=(3, 5),
                p=prob_ink * 0.2,
            ),
            OneOf(
                [
                    LowInkRandomLines(
                        count_range=(5, 10),
                        use_consistent_lines=random.choice([True, False]),
                        noise_probability=0.1,
                    ),
                    LowInkPeriodicLines(
                        count_range=(2, 5),
                        period_range=(16, 32),
                        use_consistent_lines=random.choice([True, False]),
                        noise_probability=0.1,
                    ),
                ],
                p=prob_ink * 0.2,
            ),
        ]

        paper_phase = [
            ColorPaper(
                hue_range=(0, 255),
                saturation_range=(10, 40),
                p=prob_paper * 0.2,
            ),
            OneOf(
                [
                    PatternGenerator(
                        imgx=random.randint(256, 512),
                        imgy=random.randint(256, 512),
                        n_rotation_range=(10, 15),
                        color="random",
                        alpha_range=(0.25, 0.5),
                    ),
                    SubtleNoise(
                        subtle_range=random.randint(5, 10),
                    ),
                    DirtyRollers(
                        line_width_range=(2, 32),
                        scanline_type=0,
                    ),
                    DoubleExposure(),
                ],
                p=prob_paper * 0.2,
            ),
            WaterMark(
                watermark_word="random",
                watermark_font_size=(10, 15),
                watermark_font_thickness=(20, 25),
                watermark_rotation=(0, 360),
                watermark_location="random",
                watermark_color="random",
                watermark_method="darken",
                p=prob_paper * 0.05,
            ),
            OneOf(
                [
                    NoiseTexturize(
                        sigma_range=(3, 10),
                        turbulence_range=(2, 5),
                        texture_width_range=(300, 500),
                        texture_height_range=(300, 500),
                    ),
                    BrightnessTexturize(
                        texturize_range=(0.9, 0.99),
                        deviation=0.03,
                    ),
                ],
                p=prob_paper * 0.2,
            ),
        ]

        post_phase = [
            OneOf(
                [
                    GlitchEffect(
                        glitch_direction="random",
                        glitch_number_range=(8, 16),
                        glitch_size_range=(5, 50),
                        glitch_offset_range=(10, 50),
                    ),
                    ColorShift(
                        color_shift_offset_x_range=(3, 5),
                        color_shift_offset_y_range=(3, 5),
                        color_shift_iterations=(2, 3),
                        color_shift_brightness_range=(0.9, 1.1),
                        color_shift_gaussian_kernel_range=(3, 3),
                    ),
                ],
                p=prob_post * 0.2,
            ),
            OneOf(
                [
                    DirtyRollers(
                        line_width_range=(2, 32),
                        scanline_type=0,
                    ),
                ],
                p=prob_post * 0.2,
            ),
            OneOf(
                [
                    Brightness(
                        brightness_range=(0.9, 1.1),
                        min_brightness=0,
                        min_brightness_value=(120, 150),
                    ),
                    Gamma(
                        gamma_range=(0.9, 1.1),
                    ),
                ],
                p=prob_post * 0.2,
            ),
            OneOf(
                [
                    SubtleNoise(
                        subtle_range=random.randint(5, 10),
                    ),
                    Jpeg(
                        quality_range=(25, 95),
                    ),
                ],
                p=prob_post * 0.2,
            ),
            # Markup/Scribbles disabled due to upstream ink generator float std bug
            # OneOf block removed to avoid random.randint numpy.float64 errors
            OneOf(
                [
                    ShadowCast(
                        shadow_side="random",
                        shadow_vertices_range=(1, 20),
                        shadow_width_range=(0.3, 0.8),
                        shadow_height_range=(0.3, 0.8),
                        shadow_color=(0, 0, 0),
                        shadow_opacity_range=(0.2, 0.9),
                        shadow_iterations_range=(1, 2),
                        shadow_blur_kernel_range=(101, 301),
                    ),
                ],
                p=prob_post * 0.2,
            ),
            OneOf(
                [
                    NoisyLines(
                        noisy_lines_direction="random",
                        noisy_lines_location="random",
                        noisy_lines_number_range=(5, 20),
                        noisy_lines_color=(0, 0, 0),
                        noisy_lines_thickness_range=(1, 2),
                        noisy_lines_random_noise_intensity_range=(0.01, 0.1),
                        noisy_lines_length_interval_range=(0, 100),
                        noisy_lines_gaussian_kernel_value_range=(3, 5),
                        noisy_lines_overlay_method="ink_to_paper",
                    ),
                    BindingsAndFasteners(
                        overlay_types="darken",
                        foreground=None,
                        effect_type="random",
                        width_range="random",
                        height_range="random",
                        angle_range=(-30, 30),
                        ntimes=(2, 6),
                        nscales=(0.9, 1.0),
                        edge="random",
                        edge_offset=(10, 50),
                        use_figshare_library=0,
                    ),
                ],
                p=prob_post * 0.2,
            ),
            OneOf(
                [
                    Squish(
                        squish_direction="random",
                        squish_location="random",
                        squish_number_range=(5, 10),
                        squish_distance_range=(5, 7),
                        squish_line="random",
                        squish_line_thickness_range=(1, 1),
                    ),
                ],
                p=prob_post * 0.2,
            ),
            OneOf(
                [
                    DotMatrix(
                        dot_matrix_shape="random",
                        dot_matrix_dot_width_range=(3, 3),
                        dot_matrix_dot_height_range=(3, 3),
                        dot_matrix_min_width_range=(1, 2),
                        dot_matrix_max_width_range=(150, 200),
                        dot_matrix_min_height_range=(1, 2),
                        dot_matrix_max_height_range=(150, 200),
                        dot_matrix_min_area_range=(10, 20),
                        dot_matrix_max_area_range=(2000, 5000),
                        dot_matrix_median_kernel_value_range=(128, 255),
                        dot_matrix_gaussian_kernel_value_range=(1, 3),
                        dot_matrix_rotate_value_range=(0, 360),
                    ),
                    Faxify(
                        scale_range=(0.3, 0.6),
                        monochrome=random.choice([0, 1]),
                        monochrome_method="random",
                        monochrome_arguments={},
                        halftone=random.choice([0, 1]),
                        invert=1,
                        half_kernel_size=random.choice([(1, 1), (2, 2)]),
                        angle=(0, 360),
                        sigma=(1, 3),
                    ),
                ],
                p=prob_post * 0.2,
            ),
            OneOf(
                [
                    InkMottlingPost(
                        ink_mottling_alpha_range=(0.2, 0.3),
                        ink_mottling_noise_scale_range=(2, 2),
                        ink_mottling_gaussian_kernel_range=(3, 5),
                    ),
                ],
                p=prob_post * 0.2,
            ),
            OneOf(
                [
                    Folding(
                        fold_x=None,
                        fold_deviation=(0, 0),
                        fold_count=random.randint(2, 8),
                        fold_noise=0.01,
                        fold_angle_range=(0, 0),  # avoid rotation
                        gradient_width=(0.1, 0.2),
                        gradient_height=(0.01, 0.02),
                        backdrop_color=(0, 0, 0),
                    ),
                ],
                p=prob_post * 0.2,
            ),
        ]
        # Keep ink opaque enough to avoid washed-out results; configurable via YAML
        # Default to "darken" to avoid brightening overlays when paper is white
        overlay_type = augraphy_cfg.get("overlay_type", "darken")
        overlay_alpha = float(augraphy_cfg.get("overlay_alpha", 1.0))
        overlay_alpha = max(0.0, min(1.0, overlay_alpha))

        return AugraphyPipeline(
            ink_phase=ink_phase,
            paper_phase=paper_phase,
            post_phase=post_phase,
            overlay_type=overlay_type,
            overlay_alpha=overlay_alpha,
        )

    def sample_scale(self) -> int:
        """Sample a scale from multi_scale_sizes config."""
        sizes = self.config.get("multi_scale_sizes", [512])
        if not sizes:
            return 512
        if len(sizes) == 1:
            return sizes[0]
        return random.choice(sizes)

    def build_geometric_transform(self) -> A.Compose:
        """Build geometric transform pipeline (always applied before photometric/Augraphy).

        Includes: resize, perspective, flips, rotation, random crop.
        These transforms are essential for layout detection and don't conflict with degradation.
        """
        cfg = self.config

        transforms: list[A.BasicTransform] = []

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

        return A.Compose(
            transforms,
            bbox_params=A.BboxParams(
                format="coco",
                label_fields=["category_ids"],
                min_visibility=0.3,
            ),
        )

    def build_photometric_transform(self) -> A.Compose:
        """Build photometric transform pipeline (mutually exclusive with Augraphy).

        Includes: brightness/contrast, blur, compression, noise, elastic, sobel edge.
        All effects are always applied when photometric pipeline is chosen.
        Individual probabilities removed for clarity - use choice_probability to control.
        """
        cfg = self.config
        transforms: list[A.BasicTransform] = []

        # Brightness/contrast (always applied in photometric mode)
        brightness_cfg = cfg.get("brightness_contrast", {})
        limit = brightness_cfg.get("limit", 0.2)
        transforms.append(
            A.RandomBrightnessContrast(
                brightness_limit=limit,
                contrast_limit=limit,
                p=0.5,  # 50% chance within photometric pipeline
            )
        )

        # Blur (motion or Gaussian)
        blur_cfg = cfg.get("blur", {})
        blur_limit = blur_cfg.get("blur_limit", 3)
        transforms.append(
            A.OneOf(
                [
                    A.MotionBlur(blur_limit=blur_limit, p=1.0),
                    A.GaussianBlur(blur_limit=blur_limit, p=1.0),
                ],
                p=0.3,  # 30% chance within photometric pipeline
            )
        )

        # JPEG compression
        compression_cfg = cfg.get("compression", {})
        transforms.append(
            A.ImageCompression(
                quality_range=(
                    compression_cfg.get("quality_min", 75),
                    compression_cfg.get("quality_max", 100),
                ),
                p=0.3,  # 30% chance within photometric pipeline
            )
        )

        # Gaussian noise
        noise_cfg = cfg.get("noise", {})
        transforms.append(
            A.GaussNoise(
                std_range=(
                    noise_cfg.get("std_min", 0.0),
                    noise_cfg.get("std_max", 0.01),
                ),
                p=0.3,  # 30% chance within photometric pipeline
            )
        )

        # Elastic transform (distortion effect)
        elastic_cfg = cfg.get("elastic", {})
        if elastic_cfg.get("probability", 0) > 0:
            transforms.append(
                A.ElasticTransform(
                    alpha=elastic_cfg.get("alpha", 30),
                    sigma=elastic_cfg.get("sigma", 5),
                    p=elastic_cfg["probability"],
                )
            )

        # Sobel edge extraction (optional)
        sobel_cfg = cfg.get("sobel_edge", {})
        if sobel_cfg.get("enabled", False):
            blend_alpha = sobel_cfg.get("blend_alpha", 0.2)
            transforms.append(SobelEdgeExtraction(blend_alpha=blend_alpha, p=0.2))

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

        # Build transform pipelines
        photometric_transform = self.build_photometric_transform()

        # Determine if Augraphy is enabled
        augraphy_enabled = self.augraphy_pipeline is not None

        images_out: list[np.ndarray] = []
        annotations_out: list[dict] = []

        for image, anns in zip(examples["image"], examples["annotations"], strict=False):
            # Handle both PIL images and numpy arrays
            if isinstance(image, np.ndarray):
                image_np = image
            else:
                image_np = np.array(image.convert("RGB"))

            # Normalize dtype
            if image_np.dtype != np.uint8:
                image_np = (
                    (image_np * 255).astype(np.uint8)
                    if image_np.max() <= 1.0
                    else image_np.astype(np.uint8)
                )

            # Standardize to OpenCV channel order for augmentations (BGR/BGRA)
            if len(image_np.shape) == 2:
                image_np = cv2.cvtColor(image_np, cv2.COLOR_GRAY2BGR)
            elif len(image_np.shape) == 3 and image_np.shape[2] == 3:
                image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
            elif len(image_np.shape) == 3 and image_np.shape[2] == 4:
                image_np = cv2.cvtColor(image_np, cv2.COLOR_RGBA2BGRA)

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

            # Compute scale to target short side while capping long side
            target_short = self.sample_scale()
            max_long_side = self.config.get("max_long_side")
            if max_long_side:
                target_short = min(target_short, max_long_side)
            h, w = image_np.shape[:2]
            scale = target_short / float(min(h, w))
            if max_long_side:
                scale = min(scale, max_long_side / float(max(h, w)))

            new_h = max(1, int(round(h * scale)))
            new_w = max(1, int(round(w * scale)))

            # Resize image and boxes
            image_np = cv2.resize(image_np, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            scaled_boxes = []
            for bbox in clean_boxes:
                x, y, bw, bh = bbox
                scaled_boxes.append([x * scale, y * scale, bw * scale, bh * scale])
            clean_boxes = scaled_boxes

            # Update cache (before per-image transforms)
            if len(self.sample_cache) >= self.cache_size:
                self.sample_cache.pop(0)
            self.sample_cache.append((image_np.copy(), clean_boxes.copy(), clean_labels.copy()))

            # Step 1: Apply geometric transforms (always applied)
            random_crop_cfg = self.config.get("random_crop", {})
            apply_crop = (
                random_crop_cfg.get("probability", 0) > 0
                and random.random() < random_crop_cfg["probability"]
            )

            if apply_crop:
                area_min = random_crop_cfg.get("area_min", 0.6)
                area_max = random_crop_cfg.get("area_max", 0.95)
                target_h, target_w = image_np.shape[:2]
                crop_transform = A.Compose(
                    [
                        A.RandomResizedCrop(
                            size=(target_h, target_w),
                            scale=(area_min, area_max),
                            ratio=(0.75, 1.33),
                            p=1.0,
                        )
                    ],
                    bbox_params=A.BboxParams(
                        format="coco",
                        label_fields=["category_ids"],
                        min_visibility=0.3,
                    ),
                )
                # Build geometric transform without crop
                geometric_transform = self.build_geometric_transform()
            else:
                crop_transform = None
                geometric_transform = self.build_geometric_transform()
            try:
                # Apply crop first if enabled
                if crop_transform:
                    cropped = crop_transform(
                        image=image_np,
                        bboxes=clean_boxes,
                        category_ids=clean_labels,
                    )
                    image_np = cropped["image"]
                    clean_boxes = cropped["bboxes"]
                    clean_labels = cropped["category_ids"]

                augmented = geometric_transform(
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
                    geometric_transform = self.build_geometric_transform()
                    try:
                        augmented = geometric_transform(
                            image=image_np,
                            bboxes=clamped,
                            category_ids=clean_labels,
                        )
                    except ValueError:
                        augmented["bboxes"] = clamped

            # Step 2: Choose EITHER photometric Albumentations OR Augraphy (mutually exclusive)
            # This prevents compound effects from both degradation pipelines
            if augraphy_enabled:
                # Configurable choice: photometric augmentations OR Augraphy
                # Get augraphy_choice_probability from config (default 0.5 = 50/50)
                augraphy_choice_prob = self.config.get("augraphy", {}).get(
                    "choice_probability", 0.5
                )
                use_augraphy = random.random() < augraphy_choice_prob

                # IMPORTANT: Skip Augraphy if all phase probabilities are 0.0
                # Augraphy applies ink-to-paper merging even with empty phases, causing unwanted brightness/color shifts
                if use_augraphy:
                    augraphy_cfg = self.config.get("augraphy", {})
                    prob_ink = augraphy_cfg.get("ink_probability", 0.5)
                    prob_paper = augraphy_cfg.get("paper_probability", 0.5)
                    prob_post = augraphy_cfg.get("post_probability", 0.5)

                    # If all phase probabilities are 0, skip Augraphy (fallback to photometric)
                    if prob_ink == 0.0 and prob_paper == 0.0 and prob_post == 0.0:
                        use_augraphy = False

                if use_augraphy:
                    # Apply Augraphy (document-specific degradations)
                    augmented["image"] = self.augraphy_pipeline(augmented["image"])
                else:
                    # Apply photometric Albumentations (brightness, blur, noise, etc.)
                    try:
                        photometric_result = photometric_transform(
                            image=augmented["image"],
                            bboxes=augmented["bboxes"],
                            category_ids=augmented["category_ids"],
                        )
                        augmented["image"] = photometric_result["image"]
                    except ValueError:
                        # If photometric fails, keep geometric-only result
                        pass
            else:
                # If Augraphy not enabled, always apply photometric Albumentations
                try:
                    photometric_result = photometric_transform(
                        image=augmented["image"],
                        bboxes=augmented["bboxes"],
                        category_ids=augmented["category_ids"],
                    )
                    augmented["image"] = photometric_result["image"]
                except ValueError:
                    # If photometric fails, keep geometric-only result
                    pass

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

        # Convert outputs back to RGB/RGBA for downstream consumers and visualization
        rgb_images_out: list[np.ndarray] = []
        for img in images_out:
            if len(img.shape) == 2:
                rgb_images_out.append(cv2.cvtColor(img, cv2.COLOR_GRAY2RGB))
            elif len(img.shape) == 3 and img.shape[2] == 3:
                rgb_images_out.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            elif len(img.shape) == 3 and img.shape[2] == 4:
                rgb_images_out.append(cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA))
            else:
                rgb_images_out.append(img)

        images_out = rgb_images_out
        examples["image"] = images_out
        examples["annotations"] = annotations_out
        return examples
