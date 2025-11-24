#!/usr/bin/env python3
"""
Visualize all augmentations used in the project.
Creates a grid showing original image + each augmentation type.
Uses real PubLayNet documents for realistic visualization.
"""

from pathlib import Path

import albumentations as A
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from datasets import load_dataset
from PIL import Image

# Import our data loader
from doc_obj_detect.data.constants import PUBLAYNET_CLASSES


def load_publaynet_samples(num_samples=5):
    """Load real PubLayNet samples using our project's data loader."""
    print(f"Loading {num_samples} real PubLayNet samples...")

    try:
        # Use our project's PubLayNet dataset
        dataset = load_dataset("shunk031/PubLayNet", split="train", streaming=True)
        print("✓ Connected to PubLayNet dataset")

        # Load samples
        samples = []
        dataset_iter = iter(dataset)

        for i in range(num_samples * 3):  # Try more samples to ensure we get enough good ones
            try:
                sample = next(dataset_iter)

                # Extract image
                image = sample["image"]
                if isinstance(image, Image.Image):
                    image = np.array(image.convert("RGB"))

                # Extract annotations
                annotations = sample["annotations"]
                bboxes = []

                if annotations and "bbox" in annotations:
                    # PubLayNet format: annotations contains bbox list and category_id list
                    bbox_list = annotations["bbox"]
                    category_list = annotations["category_id"]

                    for bbox, category_id in zip(bbox_list, category_list):
                        # COCO format: [x_min, y_min, width, height]
                        x_min, y_min, w, h = bbox
                        x_max = x_min + w
                        y_max = y_min + h

                        # Get label name
                        label = PUBLAYNET_CLASSES.get(category_id, "unknown")
                        bboxes.append([x_min, y_min, x_max, y_max, label])

                # Only add samples with at least 3 annotations for good visualization
                if len(bboxes) >= 3:
                    samples.append((image, bboxes))
                    print(f"  Loaded sample {len(samples)}/{num_samples} with {len(bboxes)} annotations, size: {image.shape}")

                if len(samples) >= num_samples:
                    break

            except Exception as e:
                print(f"  Warning: Skipping sample {i}: {e}")
                continue

        if len(samples) == 0:
            raise ValueError("No valid samples loaded")

        print(f"✓ Successfully loaded {len(samples)} real PubLayNet samples")
        return samples

    except Exception as e:
        print(f"Error loading PubLayNet: {e}")
        print("Using synthetic document samples as fallback...")
        return create_synthetic_samples(num_samples)


def create_synthetic_samples(num_samples=5):
    """Create synthetic document-like images as fallback."""
    from PIL import ImageDraw

    samples = []
    for i in range(num_samples):
        # Vary dimensions slightly
        width = 640 + (i * 50)
        height = 800 + (i * 60)

        img = Image.new("RGB", (width, height), color="white")
        draw = ImageDraw.Draw(img)

        # Draw document elements with varying layouts
        # Title
        draw.rectangle([50, 50, width-50, 100], fill="lightblue", outline="blue", width=2)

        # Text blocks (vary positions)
        if i % 2 == 0:  # Two column
            draw.rectangle([50, 120, width//2-20, 350], fill="lightgray", outline="black", width=1)
            draw.rectangle([width//2+20, 120, width-50, 350], fill="lightgray", outline="black", width=1)
        else:  # Single column
            draw.rectangle([50, 120, width-50, 350], fill="lightgray", outline="black", width=1)

        # Table
        draw.rectangle([50, 380, width-50, 550], fill="lightyellow", outline="orange", width=2)

        # Figure
        draw.rectangle([50, 580, width//2, height-70], fill="lightgreen", outline="green", width=2)

        img_array = np.array(img)

        bboxes = [
            [50, 50, width-50, 100, "title"],
            [50, 120, width//2-20 if i%2==0 else width-50, 350, "text"],
            [50, 380, width-50, 550, "table"],
            [50, 580, width//2, height-70, "figure"],
        ]

        if i % 2 == 0:  # Add second text column
            bboxes.append([width//2+20, 120, width-50, 350, "text"])

        samples.append((img_array, bboxes))

    print(f"✓ Created {num_samples} synthetic samples")
    return samples


def visualize_with_boxes(ax, image, bboxes, title):
    """Visualize image with bounding boxes."""
    ax.imshow(image)
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.axis("off")

    # Color map for classes (PubLayNet)
    colors = {
        "text": "cyan",
        "title": "blue",
        "list": "magenta",
        "table": "orange",
        "figure": "green",
        "caption": "red",
        "unknown": "gray",
    }

    for bbox in bboxes:
        x_min, y_min, x_max, y_max, label = bbox
        width = x_max - x_min
        height = y_max - y_min

        # Draw thicker, more visible bounding box
        rect = patches.Rectangle(
            (x_min, y_min),
            width,
            height,
            linewidth=3,  # Increased from 2
            edgecolor=colors.get(label, "black"),
            facecolor="none",
            linestyle="-",  # Solid line instead of dashed for better visibility
            alpha=1.0,  # Full opacity
        )
        ax.add_patch(rect)

        # Add label text at top-left corner of bbox
        ax.text(
            x_min + 5,
            y_min + 15,
            label,
            fontsize=8,
            color="white",
            weight="bold",
            bbox=dict(
                facecolor=colors.get(label, "black"),
                alpha=0.7,
                edgecolor="none",
                pad=2,
            ),
        )


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


def create_mosaic_visualization(images_list, bboxes_list, target_size=640):
    """Create a mosaic visualization from multiple images."""
    # Use albumentations Mosaic with proper metadata
    transform = A.Compose(
        [A.Mosaic(height=target_size, width=target_size, p=1.0)],
        bbox_params=A.BboxParams(format="pascal_voc", label_fields=["class_labels"]),
    )

    # Prepare the primary image
    bbox_list = [[b[0], b[1], b[2], b[3]] for b in bboxes_list[0]]
    labels = [b[4] for b in bboxes_list[0]]

    try:
        # Apply mosaic - albumentations will pull from metadata
        transformed = transform(image=images_list[0], bboxes=bbox_list, class_labels=labels)

        # Reconstruct bboxes
        new_bboxes = []
        for bbox, label in zip(transformed["bboxes"], transformed["class_labels"], strict=False):
            new_bboxes.append([bbox[0], bbox[1], bbox[2], bbox[3], label])

        return transformed["image"], new_bboxes
    except Exception as e:
        print(f"Warning: Mosaic failed - {e}")
        return images_list[0], bboxes_list[0]


def create_mixup_visualization(img1, bboxes1, img2, bboxes2, alpha=0.5):
    """Create a mixup visualization from two images."""
    # Resize images to same size
    h, w = img1.shape[:2]
    if img2.shape[:2] != (h, w):
        from PIL import Image
        img2_pil = Image.fromarray(img2)
        img2_pil = img2_pil.resize((w, h))
        img2 = np.array(img2_pil)

    # Mix images
    mixed_image = (alpha * img1 + (1 - alpha) * img2).astype(np.uint8)

    # Combine bboxes from both images
    combined_bboxes = bboxes1 + bboxes2

    return mixed_image, combined_bboxes


def main():
    # Create output directory
    output_dir = Path("outputs/augmentation_visualizations")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load multiple samples for mosaic/mixup
    samples = load_publaynet_samples(num_samples=5)

    # Use first sample as primary
    image, bboxes = samples[0]

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
    # Now with 9 augmentations (7 regular + mosaic + mixup) = 18 slots (9 rows x 2 cols)
    fig, axes = plt.subplots(5, 4, figsize=(16, 20))
    axes = axes.flatten()

    comparison_augs = [
        "Horizontal Flip (p=0.5)",
        "Vertical Flip (p=0.5)",
        "Random Crop (p=0.7, area=0.5-0.9)",
        "Rotation (±5°, p=0.5)",
        "Brightness/Contrast (p=0.5)",
        "Blur (p=0.3)",
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

    # Add Mosaic visualization
    visualize_with_boxes(axes[idx], image, bboxes, "Original")
    idx += 1
    mosaic_img, mosaic_bboxes = create_mosaic_visualization(
        [s[0] for s in samples[:4]],
        [s[1] for s in samples[:4]]
    )
    visualize_with_boxes(axes[idx], mosaic_img, mosaic_bboxes, "Mosaic (p=0.5)")
    idx += 1

    # Add MixUp visualization
    visualize_with_boxes(axes[idx], image, bboxes, "Original")
    idx += 1
    mixup_img, mixup_bboxes = create_mixup_visualization(
        samples[0][0], samples[0][1],
        samples[1][0], samples[1][1],
        alpha=0.5
    )
    visualize_with_boxes(axes[idx], mixup_img, mixup_bboxes, "MixUp (p=0.15, alpha=0.2)")
    idx += 1

    # Hide unused axes
    for i in range(idx, len(axes)):
        axes[i].axis("off")

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
