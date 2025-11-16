"""Visualization tools for data augmentation and model outputs."""

import argparse
import random
from pathlib import Path

import matplotlib.patches as patches
import matplotlib.pyplot as plt
from datasets import load_dataset

from doc_obj_detect.data import apply_augmentations


def visualize_augmentations(num_samples: int = 4, output_dir: str | None = None) -> None:
    """Visualize augmentation effects on sample images.

    Args:
        num_samples: Number of samples to visualize
        output_dir: Directory to save visualizations (if None, displays interactively)
    """
    print("Loading PubLayNet dataset...")
    dataset = load_dataset("shunk031/PubLayNet", split="train", streaming=True)

    # Get random samples
    samples = []
    for i, example in enumerate(dataset):
        if i >= num_samples * 10:  # Get more to choose from
            break
        samples.append(example)

    # Randomly select num_samples
    selected = random.sample(samples, min(num_samples, len(samples)))

    # Class labels
    class_labels = {0: "text", 1: "title", 2: "list", 3: "table", 4: "figure"}

    # Augmentation config (document-appropriate)
    aug_config = {
        "horizontal_flip": 0.5,
        "rotate_limit": 5,
        "brightness_contrast": 0.2,
        "noise_std": 0.01,
    }

    for idx, example in enumerate(selected):
        # Create figure with 2 subplots (original and augmented)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

        image = example["image"]
        annotations = example["objects"]

        # Original image
        ax1.imshow(image)
        ax1.set_title("Original", fontsize=14, fontweight="bold")
        ax1.axis("off")

        # Draw bounding boxes on original
        for bbox, cat_id in zip(annotations["bbox"], annotations["category_id"], strict=False):
            x, y, w, h = bbox
            # Remap category IDs (1-indexed to 0-indexed)
            cat_id_mapped = cat_id - 1 if cat_id > 0 else cat_id
            label = class_labels.get(cat_id_mapped, f"class_{cat_id_mapped}")

            rect = patches.Rectangle(
                (x, y),
                w,
                h,
                linewidth=2,
                edgecolor="red",
                facecolor="none",
            )
            ax1.add_patch(rect)
            ax1.text(
                x,
                y - 5,
                label,
                fontsize=8,
                color="red",
                bbox={"facecolor": "white", "alpha": 0.7, "edgecolor": "none", "pad": 1},
            )

        # Apply augmentations
        augmented = apply_augmentations(
            image=image,
            annotations=annotations,
            horizontal_flip=aug_config["horizontal_flip"],
            rotate_limit=aug_config["rotate_limit"],
            brightness_contrast=aug_config["brightness_contrast"],
            noise_std=aug_config["noise_std"],
        )

        # Augmented image
        ax2.imshow(augmented["image"])
        ax2.set_title("Augmented", fontsize=14, fontweight="bold")
        ax2.axis("off")

        # Draw bounding boxes on augmented
        aug_annotations = augmented["objects"]
        for bbox, cat_id in zip(
            aug_annotations["bbox"], aug_annotations["category_id"], strict=False
        ):
            x, y, w, h = bbox
            cat_id_mapped = cat_id - 1 if cat_id > 0 else cat_id
            label = class_labels.get(cat_id_mapped, f"class_{cat_id_mapped}")

            rect = patches.Rectangle(
                (x, y),
                w,
                h,
                linewidth=2,
                edgecolor="lime",
                facecolor="none",
            )
            ax2.add_patch(rect)
            ax2.text(
                x,
                y - 5,
                label,
                fontsize=8,
                color="lime",
                bbox={"facecolor": "white", "alpha": 0.7, "edgecolor": "none", "pad": 1},
            )

        plt.tight_layout()

        if output_dir:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            save_path = output_path / f"augmentation_example_{idx + 1}.png"
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"Saved: {save_path}")
            plt.close()
        else:
            plt.show()


def main() -> None:
    """CLI entry point for augmentation visualization."""
    parser = argparse.ArgumentParser(
        description="Visualize data augmentation effects on document images"
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=4,
        help="Number of samples to visualize (default: 4)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to save visualizations (default: display interactively)",
    )

    args = parser.parse_args()

    visualize_augmentations(num_samples=args.num_samples, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
