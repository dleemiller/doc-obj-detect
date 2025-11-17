"""Tests for data loading and preprocessing."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
from PIL import Image

from doc_obj_detect.data import (
    DOCLAYNET_CLASSES,
    PUBLAYNET_CLASSES,
    apply_augmentations,
    collate_fn,
    format_annotations_for_detr,
    get_augmentation_transform,
    prepare_dataset_for_training,
)
from doc_obj_detect.visualize import visualize_augmentations


def test_class_labels():
    """Test class labels are defined correctly."""
    assert len(PUBLAYNET_CLASSES) == 5
    assert len(DOCLAYNET_CLASSES) == 11


def test_format_annotations_for_detr():
    """Test COCO annotation formatting."""
    annotations = [{"bbox": [10, 20, 100, 50], "category_id": 0, "area": 5000}]
    result = format_annotations_for_detr(123, annotations)

    assert result["image_id"] == 123
    assert result["annotations"] == annotations


def test_apply_augmentations():
    """Test augmentation application updates bbox and area."""
    dummy_image = Image.fromarray(np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8))
    dummy_annotations = [
        {"bbox": [100, 100, 200, 150], "category_id": 0, "area": 30000, "iscrowd": 0}
    ]

    examples = {"image": [dummy_image], "annotations": [dummy_annotations]}
    transform = get_augmentation_transform({"rotate_limit": 5})
    result = apply_augmentations(examples, transform)

    # Check structure maintained and area recalculated
    assert len(result["image"]) == 1
    assert len(result["annotations"]) == 1
    result_bbox = result["annotations"][0][0]["bbox"]
    result_area = result["annotations"][0][0]["area"]
    assert result_area == result_bbox[2] * result_bbox[3]


def test_collate_fn():
    """Test batch collation."""
    batch = [
        {
            "pixel_values": torch.randn(3, 512, 512),
            "pixel_mask": torch.ones(512, 512),
            "labels": {"image_id": 1},
        },
        {
            "pixel_values": torch.randn(3, 512, 512),
            "pixel_mask": torch.ones(512, 512),
            "labels": {"image_id": 2},
        },
    ]

    result = collate_fn(batch)

    assert result["pixel_values"].shape == (2, 3, 512, 512)
    assert result["pixel_mask"].shape == (2, 512, 512)
    assert len(result["labels"]) == 2


@patch("doc_obj_detect.data.load_publaynet")
def test_prepare_dataset_for_training(mock_load):
    """Test dataset preparation."""
    mock_dataset = MagicMock()
    mock_dataset.with_transform.return_value = mock_dataset
    mock_load.return_value = (mock_dataset, PUBLAYNET_CLASSES)

    dataset, labels = prepare_dataset_for_training("publaynet", "train", MagicMock())

    assert labels == PUBLAYNET_CLASSES
    assert mock_dataset.with_transform.called


def test_prepare_dataset_invalid_name():
    """Test invalid dataset name raises error."""
    with pytest.raises(ValueError, match="Unknown dataset"):
        prepare_dataset_for_training("invalid", "train", MagicMock())


@patch("doc_obj_detect.visualize.load_publaynet")
def test_visualize_augmentations(mock_load, tmp_path):
    """Test augmentation visualization."""
    # Create mock dataset
    mock_sample = {
        "image": Image.fromarray(np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)),
        "annotations": [{"bbox": [100, 100, 200, 150], "category_id": 0, "area": 30000}],
    }
    mock_dataset = MagicMock()
    mock_dataset.__getitem__ = MagicMock(return_value=mock_sample)
    mock_load.return_value = (mock_dataset, {"0": "text"})

    # Run visualization
    output_dir = str(tmp_path / "test_output")
    visualize_augmentations("publaynet", num_samples=1, output_dir=output_dir)

    # Verify output directory was created
    assert (tmp_path / "test_output").exists()
