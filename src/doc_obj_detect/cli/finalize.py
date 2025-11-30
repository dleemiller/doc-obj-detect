"""Finalize checkpoint for HuggingFace Hub upload.

This module provides functionality to prepare training checkpoints for sharing:
- Extracts EMA weights as main model weights
- Removes training artifacts (optimizer, scheduler, etc.)
- Generates minimal README template
- Reduces checkpoint size by ~75%
"""

import json
import logging
import shutil
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

logger = logging.getLogger(__name__)


def finalize_checkpoint(
    checkpoint_path: str | Path,
    output_path: str | Path,
    use_ema: bool = True,
    create_readme: bool = True,
) -> None:
    """Prepare checkpoint for HuggingFace upload.

    This function:
    1. Loads model weights (EMA or regular)
    2. Saves as main model.safetensors
    3. Copies config files
    4. Optionally generates minimal README template
    5. Reports size reduction

    Args:
        checkpoint_path: Path to training checkpoint directory
        output_path: Path to output directory for finalized model
        use_ema: Use EMA weights if available (default: True)
        create_readme: Generate minimal README.md template (default: True)

    Example:
        >>> finalize_checkpoint(
        ...     "outputs/pretrain_publaynet/checkpoint-best",
        ...     "models/publaynet-dfine-large",
        ... )
    """
    checkpoint_path = Path(checkpoint_path)
    output_path = Path(output_path)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    output_path.mkdir(parents=True, exist_ok=True)
    logger.info("Finalizing checkpoint for HuggingFace upload")
    logger.info("  Input: %s", checkpoint_path)
    logger.info("  Output: %s", output_path)

    # Step 1: Load weights (EMA or regular)
    state_dict = _load_weights(checkpoint_path, use_ema)

    # Step 2: Save as main model
    output_model = output_path / "model.safetensors"
    save_file(state_dict, str(output_model))
    logger.info("✅ Saved model weights to %s", output_model)

    # Step 3: Copy config files
    _copy_configs(checkpoint_path, output_path)

    # Step 4: Generate minimal README
    if create_readme:
        readme = output_path / "README.md"
        readme.write_text(_generate_minimal_readme(checkpoint_path, use_ema))
        logger.info("✅ Generated minimal README: %s", readme)
        logger.info("   → Edit this file to add model-specific details")

    # Step 5: Report size reduction
    _report_size_reduction(checkpoint_path, output_path)

    logger.info("✨ Finalized model ready at: %s", output_path)
    logger.info("   Load with: AutoModelForObjectDetection.from_pretrained('%s')", output_path)


def _load_weights(checkpoint_path: Path, use_ema: bool) -> dict:
    """Load model weights (EMA or regular) from checkpoint."""
    if use_ema:
        ema_path = checkpoint_path / "ema_state.pt"
        if ema_path.exists():
            logger.info("Loading EMA weights from %s", ema_path)
            state_dict = torch.load(ema_path, map_location="cpu", weights_only=True)

            # EMA state might have 'module' or 'shadow' wrapper
            if isinstance(state_dict, dict):
                if "module" in state_dict:
                    state_dict = state_dict["module"]
                elif "shadow" in state_dict:
                    state_dict = state_dict["shadow"]

            # Remove duplicate keys (safetensors can't handle shared tensors)
            # If both "model.decoder.X" and "X" exist, keep only "X"
            state_dict = _deduplicate_keys(state_dict)

            return state_dict

        logger.warning("EMA weights not found at %s, falling back to regular weights", ema_path)
        use_ema = False

    # Load regular weights
    model_path = checkpoint_path / "model.safetensors"
    if model_path.exists():
        logger.info("Loading regular weights from %s", model_path)
        return load_file(str(model_path))

    # Fallback to pytorch_model.bin
    model_bin = checkpoint_path / "pytorch_model.bin"
    if model_bin.exists():
        logger.info("Loading regular weights from %s", model_bin)
        return torch.load(model_bin, map_location="cpu", weights_only=True)

    raise FileNotFoundError(f"No model weights found in {checkpoint_path}")


def _deduplicate_keys(state_dict: dict) -> dict:
    """Remove duplicate keys that share memory.

    D-FINE EMA sometimes has both 'model.decoder.X' and 'X' keys pointing to same tensor.
    Keep only the shorter key name (without 'model.decoder.' prefix), cloning tensors to break sharing.
    """
    # Create new dict with deduplicated keys
    new_state_dict = {}
    keys_with_prefix = []
    keys_without_prefix = []

    # Separate keys into prefixed and non-prefixed
    for key in state_dict.keys():
        if key.startswith("model."):
            keys_with_prefix.append(key)
        else:
            keys_without_prefix.append(key)

    # First, add all non-prefixed keys (these are what we want)
    for key in keys_without_prefix:
        new_state_dict[key] = state_dict[key].clone()

    # Then, add prefixed keys only if non-prefixed version doesn't exist
    for key in keys_with_prefix:
        # Try to match against model.decoder.X pattern
        short_key = (
            key.replace("model.decoder.", "", 1)
            if key.startswith("model.decoder.")
            else key.replace("model.", "", 1)
        )
        if short_key not in new_state_dict:
            # No unprefixed version exists, keep the prefixed one
            new_state_dict[key] = state_dict[key].clone()

    duplicates_removed = len(state_dict) - len(new_state_dict)
    if duplicates_removed > 0:
        logger.info(
            "Removed %d duplicate keys (shared memory), cloned %d tensors",
            duplicates_removed,
            len(new_state_dict),
        )

    return new_state_dict


def _copy_configs(checkpoint_path: Path, output_path: Path) -> None:
    """Copy configuration files to output directory."""
    config_files = [
        "config.json",
        "preprocessor_config.json",
        "generation_config.json",
    ]

    for config_file in config_files:
        src = checkpoint_path / config_file
        if src.exists():
            shutil.copy(src, output_path / config_file)
            logger.info("✅ Copied %s", config_file)


def _generate_minimal_readme(checkpoint_path: Path, used_ema: bool) -> str:
    """Generate minimal README.md template with TODOs."""
    # Extract minimal info we can know
    config_path = checkpoint_path / "config.json"
    num_classes = "Unknown"
    class_names = "Unknown"

    if config_path.exists():
        try:
            with open(config_path) as f:
                config = json.load(f)
                num_classes = config.get("num_labels", "Unknown")
                id2label = config.get("id2label", {})
                if id2label:
                    class_names = ", ".join(id2label.values())
        except (json.JSONDecodeError, KeyError):
            pass

    weights_note = "EMA (Exponential Moving Average)" if used_ema else "Final training weights"

    return f"""---
tags:
- object-detection
- document-layout-analysis
library_name: transformers
---

# Document Layout Analysis Model

*TODO: Add model name and description*

## Model Details

- **Task**: Object Detection
- **Domain**: Document Layout Analysis
- **Classes ({num_classes})**: {class_names}
- **Weights**: {weights_note}

*TODO: Add architecture details (backbone, head, parameters)*

## Usage

```python
from transformers import AutoModelForObjectDetection, AutoImageProcessor
from PIL import Image
import torch

# Load model
model = AutoModelForObjectDetection.from_pretrained("path/to/model")
processor = AutoImageProcessor.from_pretrained("path/to/model")

# Run inference
image = Image.open("document.png")
inputs = processor(images=image, return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs)

# Post-process results
target_sizes = torch.tensor([image.size[::-1]])
results = processor.post_process_object_detection(
    outputs,
    target_sizes=target_sizes,
    threshold=0.5
)[0]

# Print detections
for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
    label_name = model.config.id2label[label.item()]
    box_coords = [round(coord, 2) for coord in box.tolist()]
    print(f"{{label_name}}: {{score:.3f}} at {{box_coords}}")
```

## Training

*TODO: Add training details*
- Dataset:
- Epochs:
- Batch size:
- Learning rate:
- Augmentations:

## Performance

*TODO: Add benchmark results*

| Dataset | mAP | mAP@50 | mAP@75 |
|---------|-----|--------|--------|
|         |     |        |        |

## Citation

*TODO: Add relevant citations*

```bibtex
@article{{TODO,
  title={{TODO}},
  author={{TODO}},
  year={{TODO}}
}}
```

## License

*TODO: Add license*
"""


def _report_size_reduction(checkpoint_path: Path, output_path: Path) -> None:
    """Report size reduction from original checkpoint to finalized model."""
    original_size = sum(f.stat().st_size for f in checkpoint_path.glob("*") if f.is_file())
    final_size = sum(f.stat().st_size for f in output_path.glob("*") if f.is_file())

    original_gb = original_size / 1e9
    final_gb = final_size / 1e9
    reduction_pct = (1 - final_size / original_size) * 100

    logger.info("")
    logger.info("📦 Size Reduction:")
    logger.info("   Original: %.2f GB", original_gb)
    logger.info("   Final: %.2f GB", final_gb)
    logger.info("   Saved: %.2f GB (%.1f%% reduction)", original_gb - final_gb, reduction_pct)
