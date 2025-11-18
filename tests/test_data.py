"""Tests for data loading and preprocessing."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
from PIL import Image

from doc_obj_detect.data.augmentor import AlbumentationsAugmentor
from doc_obj_detect.data.constants import DOCLAYNET_CLASSES, PUBLAYNET_CLASSES
from doc_obj_detect.data.pipeline import DatasetFactory, collate_fn
from doc_obj_detect.visualize import visualize_augmentations


def test_class_labels():
    """Test class labels are defined correctly."""
    assert len(PUBLAYNET_CLASSES) == 5
    assert len(DOCLAYNET_CLASSES) == 11


def test_apply_augmentations():
    """Test augmentation application updates bbox and area."""
    dummy_image = Image.fromarray(np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8))
    dummy_annotations = [
        {"bbox": [100, 100, 200, 150], "category_id": 0, "area": 30000, "iscrowd": 0}
    ]

    examples = {"image": [dummy_image], "annotations": [dummy_annotations]}
    augmentor = AlbumentationsAugmentor({"rotate_limit": 5})
    result = augmentor.augment(examples)

    # Check structure maintained and area recalculated
    assert len(result["image"]) == 1
    assert len(result["annotations"]) == 1
    result_ann = result["annotations"][0]
    result_bbox = result_ann["bbox"][0]
    result_area = result_ann["area"][0]
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


@patch("doc_obj_detect.data.datasets.DatasetLoader.load_publaynet")
def test_dataset_factory_build(mock_load):
    """Test dataset preparation via factory."""
    mock_dataset = MagicMock()
    mock_dataset.with_transform.return_value = mock_dataset
    mock_load.return_value = (mock_dataset, PUBLAYNET_CLASSES)

    factory = DatasetFactory(
        dataset_name="publaynet",
        image_processor=MagicMock(),
        pad_stride=32,
        cache_dir=None,
        augmentation_config=None,
    )
    dataset, labels = factory.build(split="train", apply_augmentation=False)

    assert labels == PUBLAYNET_CLASSES
    assert mock_dataset.with_transform.called


def test_dataset_factory_invalid_name():
    """Test invalid dataset name raises error."""
    with pytest.raises(ValueError, match="Unknown dataset"):
        DatasetFactory(
            dataset_name="invalid",
            image_processor=MagicMock(),
            pad_stride=32,
        ).build(split="train")


@patch("doc_obj_detect.visualize.DatasetLoader.load_publaynet")
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
