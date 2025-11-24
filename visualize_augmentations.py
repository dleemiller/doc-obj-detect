#!/usr/bin/env python3
"""
Visualize all augmentations used in the project.
Creates a grid showing original image + each augmentation type.
"""

from pathlib import Path

import albumentations as A
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw


def create_sample_document(width=640, height=800):
    """Create a synthetic document-like image with bboxes for visualization."""
    # Create white background
    img = Image.new("RGB", (width, height), color="white")
    draw = ImageDraw.Draw(img)

    # Draw document-like content
    # Title
    draw.rectangle([50, 50, 590, 100], fill="lightblue", outline="blue", width=2)

    # Two-column text blocks
    draw.rectangle([50, 120, 290, 350], fill="lightgray", outline="black", width=1)
    draw.rectangle([310, 120, 590, 350], fill="lightgray", outline="black", width=1)

    # Table
    draw.rectangle([50, 380, 590, 550], fill="lightyellow", outline="orange", width=2)
    for i in range(4):
        y = 380 + i * 42
        draw.line([50, y, 590, y], fill="orange", width=1)
    for i in range(4):
        x = 50 + i * 180
        draw.line([x, 380, x, 550], fill="orange", width=1)

    # Figure
    draw.rectangle([50, 580, 290, 730], fill="lightgreen", outline="green", width=2)
    draw.text((100, 650), "Figure", fill="green")

    # Caption
    draw.rectangle([310, 680, 590, 730], fill="pink", outline="red", width=1)

    # Convert to numpy
    img_array = np.array(img)

    # Bboxes in [x_min, y_min, x_max, y_max] format
    bboxes = [
        [50, 50, 590, 100, "title"],  # Title
        [50, 120, 290, 350, "text"],  # Left column
        [310, 120, 590, 350, "text"],  # Right column
        [50, 380, 590, 550, "table"],  # Table
        [50, 580, 290, 730, "figure"],  # Figure
        [310, 680, 590, 730, "caption"],  # Caption
    ]

    return img_array, bboxes


def visualize_with_boxes(ax, image, bboxes, title):
    """Visualize image with bounding boxes."""
    ax.imshow(image)
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.axis("off")

    # Color map for classes
    colors = {
        "title": "blue",
        "text": "gray",
        "table": "orange",
        "figure": "green",
        "caption": "red",
    }

    for bbox in bboxes:
        x_min, y_min, x_max, y_max, label = bbox
        width = x_max - x_min
        height = y_max - y_min
        rect = patches.Rectangle(
            (x_min, y_min),
            width,
            height,
            linewidth=2,
            edgecolor=colors.get(label, "black"),
            facecolor="none",
            linestyle="--",
            alpha=0.8,
        )
        ax.add_patch(rect)


def apply_augmentation(image, bboxes, transform, name):
    """Apply augmentation and return transformed image and boxes."""
    # Convert boxes to albumentations format
    bbox_list = [[b[0], b[1], b[2], b[3]] for b in bboxes]
    labels = [b[4] for b in bboxes]

    try:
        transformed = transform(image=image, bboxes=bbox_list, class_labels=labels)

        # Reconstruct bboxes with labels
        new_bboxes = []
        for bbox, label in zip(transformed["bboxes"], transformed["class_labels"], strict=False):
            new_bboxes.append([bbox[0], bbox[1], bbox[2], bbox[3], label])

        return transformed["image"], new_bboxes
    except Exception as e:
        print(f"Warning: {name} failed - {e}")
        return image, bboxes


def main():
    # Create output directory
    output_dir = Path("outputs/augmentation_visualizations")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create sample document
    image, bboxes = create_sample_document()

    # Define augmentations to visualize
    augmentations = {
        "Original": A.Compose(
            [A.NoOp()], bbox_params=A.BboxParams(format="pascal_voc", label_fields=["class_labels"])
        ),
        "Horizontal Flip (p=0.5)": A.Compose(
            [A.HorizontalFlip(p=1.0)],
            bbox_params=A.BboxParams(format="pascal_voc", label_fields=["class_labels"]),
        ),
        "Vertical Flip (p=0.5)": A.Compose(
            [A.VerticalFlip(p=1.0)],
            bbox_params=A.BboxParams(format="pascal_voc", label_fields=["class_labels"]),
        ),
        "Random Crop (p=0.7, area=0.5-0.9)": A.Compose(
            [A.RandomResizedCrop(size=(800, 640), scale=(0.7, 0.9), p=1.0)],
            bbox_params=A.BboxParams(
                format="pascal_voc", label_fields=["class_labels"], min_visibility=0.3
            ),
        ),
        "Rotation (±5°, p=0.5)": A.Compose(
            [A.Rotate(limit=5, border_mode=0, p=1.0)],
            bbox_params=A.BboxParams(format="pascal_voc", label_fields=["class_labels"]),
        ),
        "Brightness/Contrast (p=0.5)": A.Compose(
            [A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=1.0)],
            bbox_params=A.BboxParams(format="pascal_voc", label_fields=["class_labels"]),
        ),
        "Blur (p=0.3)": A.Compose(
            [A.GaussianBlur(blur_limit=3, p=1.0)],
            bbox_params=A.BboxParams(format="pascal_voc", label_fields=["class_labels"]),
        ),
        "JPEG Compression (p=0.3)": A.Compose(
            [A.ImageCompression(quality_lower=75, quality_upper=80, p=1.0)],
            bbox_params=A.BboxParams(format="pascal_voc", label_fields=["class_labels"]),
        ),
        "Gaussian Noise (p=0.3)": A.Compose(
            [A.GaussNoise(var_limit=(0.005, 0.01), p=1.0)],
            bbox_params=A.BboxParams(format="pascal_voc", label_fields=["class_labels"]),
        ),
        "Perspective (p=0.3, fine-tune)": A.Compose(
            [A.Perspective(scale=(0.02, 0.05), p=1.0)],
            bbox_params=A.BboxParams(format="pascal_voc", label_fields=["class_labels"]),
        ),
        "Elastic Transform (p=0.2, fine-tune)": A.Compose(
            [A.ElasticTransform(alpha=30, sigma=5, p=1.0, border_mode=0)],
            bbox_params=A.BboxParams(format="pascal_voc", label_fields=["class_labels"]),
        ),
    }

    # Create visualization grid
    n_augs = len(augmentations)
    n_cols = 3
    n_rows = (n_augs + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
    axes = axes.flatten()

    for idx, (name, transform) in enumerate(augmentations.items()):
        aug_image, aug_bboxes = apply_augmentation(image.copy(), bboxes, transform, name)
        visualize_with_boxes(axes[idx], aug_image, aug_bboxes, name)

    # Hide extra subplots
    for idx in range(len(augmentations), len(axes)):
        axes[idx].axis("off")

    plt.tight_layout()
    output_path = output_dir / "all_augmentations.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"✓ Saved comprehensive visualization to {output_path}")

    # Create individual augmentation comparison (original vs augmented)
    fig, axes = plt.subplots(4, 4, figsize=(16, 16))
    axes = axes.flatten()

    comparison_augs = [
        "Horizontal Flip (p=0.5)",
        "Vertical Flip (p=0.5)",
        "Random Crop (p=0.7, area=0.5-0.9)",
        "Rotation (±5°, p=0.5)",
        "Brightness/Contrast (p=0.5)",
        "Blur (p=0.3)",
        "JPEG Compression (p=0.3)",
        "Gaussian Noise (p=0.3)",
    ]

    idx = 0
    for aug_name in comparison_augs:
        # Original
        visualize_with_boxes(axes[idx], image, bboxes, "Original")
        idx += 1

        # Augmented
        transform = augmentations[aug_name]
        aug_image, aug_bboxes = apply_augmentation(image.copy(), bboxes, transform, aug_name)
        visualize_with_boxes(axes[idx], aug_image, aug_bboxes, aug_name)
        idx += 1

    plt.tight_layout()
    output_path = output_dir / "augmentation_comparisons.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"✓ Saved side-by-side comparisons to {output_path}")

    # Create a summary figure with key augmentations
    key_augs = {
        "Original": augmentations["Original"],
        "Horizontal Flip": augmentations["Horizontal Flip (p=0.5)"],
        "Random Crop": augmentations["Random Crop (p=0.7, area=0.5-0.9)"],
        "Rotation ±5°": augmentations["Rotation (±5°, p=0.5)"],
        "Brightness/Contrast": augmentations["Brightness/Contrast (p=0.5)"],
        "Blur": augmentations["Blur (p=0.3)"],
    }

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for idx, (name, transform) in enumerate(key_augs.items()):
        aug_image, aug_bboxes = apply_augmentation(image.copy(), bboxes, transform, name)
        visualize_with_boxes(axes[idx], aug_image, aug_bboxes, name)

    plt.suptitle(
        "Key Document Augmentations (ICDAR 2023 + DocLayout-YOLO)",
        fontsize=14,
        fontweight="bold",
        y=0.995,
    )
    plt.tight_layout()
    output_path = output_dir / "key_augmentations.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"✓ Saved key augmentations summary to {output_path}")

    print(f"\n✓ All visualizations saved to {output_dir}/")
    print("  - all_augmentations.png (complete grid)")
    print("  - augmentation_comparisons.png (side-by-side)")
    print("  - key_augmentations.png (summary)")


if __name__ == "__main__":
    main()
