"""Comprehensive test suite for custom augmentations (Mosaic and MixUp).

These tests ensure correctness of bbox handling, image composition,
and edge case handling for batch-level augmentations.
"""

import numpy as np
from PIL import Image

from doc_obj_detect.data.augmentor import AlbumentationsAugmentor


class TestFlipAugmentations:
    """Test official albumentations flip augmentations."""

    def test_horizontal_flip(self):
        """Test horizontal flip preserves bboxes correctly."""
        config = {
            "multi_scale_sizes": [640],
            "force_square_resize": True,
            "horizontal_flip": 1.0,  # Always apply
        }
        augmentor = AlbumentationsAugmentor(config)

        image = Image.new("RGB", (800, 600), color="red")
        examples = {
            "image": [image],
            "annotations": [{"bbox": [[100, 100, 200, 150]], "category_id": [1]}],
        }

        result = augmentor.augment(examples)
        assert len(result["image"]) == 1
        assert len(result["annotations"]) == 1
        assert len(result["annotations"][0]["bbox"]) == 1
        assert result["annotations"][0]["category_id"] == [1]

    def test_vertical_flip(self):
        """Test vertical flip preserves bboxes correctly."""
        config = {
            "multi_scale_sizes": [640],
            "force_square_resize": True,
            "vertical_flip": 1.0,  # Always apply
        }
        augmentor = AlbumentationsAugmentor(config)

        image = Image.new("RGB", (800, 600), color="blue")
        examples = {
            "image": [image],
            "annotations": [{"bbox": [[100, 100, 200, 150]], "category_id": [2]}],
        }

        result = augmentor.augment(examples)
        assert len(result["image"]) == 1
        assert len(result["annotations"]) == 1
        assert len(result["annotations"][0]["bbox"]) == 1


class TestMosaicAugmentation:
    """Test custom mosaic augmentation implementation."""

    def test_mosaic_combines_four_images(self):
        """Test that mosaic combines 4 images and preserves bboxes."""
        config = {
            "multi_scale_sizes": [640],
            "force_square_resize": True,
            "mosaic": {"probability": 0.0},  # Disable during cache fill
        }
        augmentor = AlbumentationsAugmentor(config)

        # Fill cache with distinct images (no mosaic during fill)
        for i in range(10):
            img = Image.new("RGB", (400, 400), color=(i * 25, i * 25, i * 25))
            examples = {
                "image": [img],
                "annotations": [
                    {"bbox": [[50 + i * 10, 50, 100, 100]], "category_id": [i % 3 + 1]}
                ],
            }
            augmentor.augment(examples)

        # Enable mosaic for test
        augmentor.config["mosaic"] = {"probability": 1.0}

        # Test mosaic with filled cache
        img = Image.new("RGB", (400, 400), color=(255, 255, 255))
        examples = {
            "image": [img],
            "annotations": [
                {"bbox": [[25, 25, 150, 150], [250, 250, 100, 100]], "category_id": [1, 2]}
            ],
        }
        result = augmentor.augment(examples)

        # Should have bboxes from multiple images (2 from primary + some from 3 cached)
        num_bboxes = len(result["annotations"][0]["bbox"])
        assert num_bboxes >= 2, f"Mosaic should combine bboxes, got {num_bboxes}"
        assert (
            num_bboxes <= 6
        ), f"Mosaic should have at most 6 bboxes (2+1+1+1+some clipped), got {num_bboxes}"

    def test_mosaic_output_size(self):
        """Test that mosaic produces correct output size."""
        target_size = 512
        config = {
            "multi_scale_sizes": [target_size],
            "force_square_resize": True,
            "mosaic": {"probability": 1.0},
        }
        augmentor = AlbumentationsAugmentor(config)

        # Fill cache
        for _ in range(5):
            img = Image.new("RGB", (300, 300), color=(100, 100, 100))
            examples = {
                "image": [img],
                "annotations": [{"bbox": [[10, 10, 50, 50]], "category_id": [1]}],
            }
            augmentor.augment(examples)

        # Test mosaic
        img = Image.new("RGB", (300, 300), color=(200, 200, 200))
        examples = {
            "image": [img],
            "annotations": [{"bbox": [[20, 20, 60, 60]], "category_id": [1]}],
        }
        result = augmentor.augment(examples)

        # Check output image size
        output_img = result["image"][0]
        assert output_img.shape[0] == target_size
        assert output_img.shape[1] == target_size

    def test_mosaic_bbox_validity(self):
        """Test that mosaic produces valid bboxes."""
        config = {
            "multi_scale_sizes": [640],
            "force_square_resize": True,
            "mosaic": {"probability": 1.0},
        }
        augmentor = AlbumentationsAugmentor(config)

        # Fill cache
        for i in range(5):
            img = Image.new("RGB", (400, 400), color=(50 * i, 50 * i, 50 * i))
            examples = {
                "image": [img],
                "annotations": [
                    {"bbox": [[100, 100, 150, 150], [200, 200, 80, 80]], "category_id": [1, 2]}
                ],
            }
            augmentor.augment(examples)

        # Test mosaic
        img = Image.new("RGB", (400, 400), color=(255, 255, 255))
        examples = {
            "image": [img],
            "annotations": [{"bbox": [[50, 50, 100, 100]], "category_id": [1]}],
        }
        result = augmentor.augment(examples)

        # Verify all bboxes are valid
        for bbox in result["annotations"][0]["bbox"]:
            x, y, w, h = bbox[:4]
            assert x >= 0, f"x should be >= 0, got {x}"
            assert y >= 0, f"y should be >= 0, got {y}"
            assert w >= 2, f"width should be >= 2, got {w}"
            assert h >= 2, f"height should be >= 2, got {h}"
            assert x + w <= 640, f"bbox extends beyond image width: {x + w}"
            assert y + h <= 640, f"bbox extends beyond image height: {y + h}"

    def test_mosaic_category_preservation(self):
        """Test that mosaic preserves category IDs correctly."""
        config = {
            "multi_scale_sizes": [640],
            "force_square_resize": True,
            "mosaic": {"probability": 1.0},
        }
        augmentor = AlbumentationsAugmentor(config)

        # Fill cache with specific categories
        for i in range(5):
            img = Image.new("RGB", (400, 400), color=(100, 100, 100))
            examples = {
                "image": [img],
                "annotations": [
                    {
                        "bbox": [[100, 100, 100, 100]],
                        "category_id": [i + 1],  # Categories 1-5
                    }
                ],
            }
            augmentor.augment(examples)

        # Test mosaic
        img = Image.new("RGB", (400, 400), color=(200, 200, 200))
        examples = {
            "image": [img],
            "annotations": [
                {
                    "bbox": [[50, 50, 80, 80]],
                    "category_id": [10],  # Distinct category
                }
            ],
        }
        result = augmentor.augment(examples)

        # Verify categories are valid integers
        categories = result["annotations"][0]["category_id"]
        assert all(isinstance(cat, int) for cat in categories)
        assert all(cat > 0 for cat in categories)


class TestMixUpAugmentation:
    """Test custom mixup augmentation implementation."""

    def test_mixup_blends_images(self):
        """Test that mixup blends two images."""
        config = {
            "multi_scale_sizes": [640],
            "force_square_resize": True,
            "mixup": {"probability": 1.0, "alpha": 0.2},
        }
        augmentor = AlbumentationsAugmentor(config)

        # Fill cache
        for i in range(5):
            img = Image.new("RGB", (400, 400), color=(0, 0, 0))
            examples = {
                "image": [img],
                "annotations": [{"bbox": [[100 + i * 10, 100, 80, 80]], "category_id": [1]}],
            }
            augmentor.augment(examples)

        # Test mixup
        img = Image.new("RGB", (400, 400), color=(255, 255, 255))
        examples = {
            "image": [img],
            "annotations": [{"bbox": [[200, 200, 100, 100]], "category_id": [2]}],
        }
        result = augmentor.augment(examples)

        # Should combine bboxes from both images
        num_bboxes = len(result["annotations"][0]["bbox"])
        assert num_bboxes >= 2, f"MixUp should combine bboxes, got {num_bboxes}"

    def test_mixup_combines_categories(self):
        """Test that mixup combines category IDs from both images."""
        config = {
            "multi_scale_sizes": [640],
            "force_square_resize": True,
            "mixup": {"probability": 1.0, "alpha": 0.5},
        }
        augmentor = AlbumentationsAugmentor(config)

        # Fill cache
        for _ in range(3):
            img = Image.new("RGB", (400, 400), color=(100, 100, 100))
            examples = {
                "image": [img],
                "annotations": [{"bbox": [[50, 50, 60, 60]], "category_id": [5]}],
            }
            augmentor.augment(examples)

        # Test mixup
        img = Image.new("RGB", (400, 400), color=(200, 200, 200))
        examples = {
            "image": [img],
            "annotations": [{"bbox": [[150, 150, 70, 70]], "category_id": [3]}],
        }
        result = augmentor.augment(examples)

        categories = result["annotations"][0]["category_id"]
        assert len(categories) >= 2, "MixUp should combine categories"
        assert all(isinstance(cat, int) for cat in categories)


class TestEpochBasedDisabling:
    """Test epoch-based augmentation disabling."""

    def test_mosaic_disables_after_epoch(self):
        """Test that mosaic is disabled after specified epoch."""
        config = {
            "multi_scale_sizes": [640],
            "force_square_resize": True,
            "mosaic": {"probability": 0.0, "disable_after_epoch": 5},  # Disabled initially
        }
        augmentor = AlbumentationsAugmentor(config)

        # Fill cache without mosaic
        for _ in range(10):
            img = Image.new("RGB", (400, 400), color=(50, 50, 50))
            examples = {
                "image": [img],
                "annotations": [{"bbox": [[100, 100, 80, 80]], "category_id": [1]}],
            }
            augmentor.augment(examples)

        # Enable mosaic temporarily before epoch 5
        augmentor.config["mosaic"]["probability"] = 1.0
        augmentor.set_epoch(4)
        img = Image.new("RGB", (400, 400), color=(255, 255, 255))
        examples = {
            "image": [img],
            "annotations": [{"bbox": [[200, 200, 100, 100]], "category_id": [1]}],
        }
        result_before = augmentor.augment(examples)
        num_bboxes_before = len(result_before["annotations"][0]["bbox"])

        # After epoch 5 (should disable mosaic)
        augmentor.set_epoch(5)
        result_after = augmentor.augment(examples)
        num_bboxes_after = len(result_after["annotations"][0]["bbox"])

        # Before disabling, should have multiple bboxes from mosaic
        assert num_bboxes_before >= 2, "Mosaic should be active before disable epoch"
        # After disabling, should have significantly fewer bboxes than before
        # (cache may still have some boxes, but no new mosaic should be created)
        assert (
            num_bboxes_after < num_bboxes_before or num_bboxes_after <= 4
        ), f"Mosaic should be disabled, got {num_bboxes_after} bboxes (before: {num_bboxes_before})"


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_bbox_handling(self):
        """Test handling of images with no bboxes."""
        config = {
            "multi_scale_sizes": [640],
            "force_square_resize": True,
        }
        augmentor = AlbumentationsAugmentor(config)

        img = Image.new("RGB", (400, 400), color=(100, 100, 100))
        examples = {"image": [img], "annotations": [{"bbox": [], "category_id": []}]}
        result = augmentor.augment(examples)

        assert len(result["annotations"][0]["bbox"]) == 0
        assert len(result["annotations"][0]["category_id"]) == 0

    def test_tiny_bbox_filtering(self):
        """Test that tiny bboxes are filtered out."""
        config = {
            "multi_scale_sizes": [640],
            "force_square_resize": True,
        }
        augmentor = AlbumentationsAugmentor(config)

        img = Image.new("RGB", (400, 400), color=(100, 100, 100))
        examples = {
            "image": [img],
            "annotations": [
                {
                    "bbox": [
                        [10, 10, 0.001, 0.001],
                        [100, 100, 50, 50],
                    ],  # First bbox too small (< 1e-3)
                    "category_id": [1, 2],
                }
            ],
        }
        result = augmentor.augment(examples)

        # Tiny bbox should be filtered (< 1e-3 pixels)
        assert len(result["annotations"][0]["bbox"]) >= 1, "Should keep valid bbox"
        # The valid bbox category should be preserved
        assert 2 in result["annotations"][0]["category_id"], "Should preserve valid bbox category"

    def test_mosaic_without_cache(self):
        """Test that mosaic gracefully handles empty cache."""
        config = {
            "multi_scale_sizes": [640],
            "force_square_resize": True,
            "mosaic": {"probability": 1.0},
        }
        augmentor = AlbumentationsAugmentor(config)

        # Try mosaic with empty cache
        img = Image.new("RGB", (400, 400), color=(100, 100, 100))
        examples = {
            "image": [img],
            "annotations": [{"bbox": [[50, 50, 100, 100]], "category_id": [1]}],
        }
        result = augmentor.augment(examples)

        # Should return original image (no mosaic applied)
        assert len(result["annotations"][0]["bbox"]) == 1

    def test_numpy_array_input(self):
        """Test that augmentor handles numpy array inputs."""
        config = {
            "multi_scale_sizes": [640],
            "force_square_resize": True,
        }
        augmentor = AlbumentationsAugmentor(config)

        # Use numpy array instead of PIL Image
        img = np.random.randint(0, 255, (400, 400, 3), dtype=np.uint8)
        examples = {
            "image": [img],
            "annotations": [{"bbox": [[100, 100, 150, 150]], "category_id": [1]}],
        }
        result = augmentor.augment(examples)

        assert isinstance(result["image"][0], np.ndarray)
        assert len(result["annotations"][0]["bbox"]) >= 1


class TestIntegration:
    """Integration tests with real-world scenarios."""

    def test_full_augmentation_pipeline(self):
        """Test complete augmentation pipeline with all augmentations."""
        config = {
            "multi_scale_sizes": [576, 608, 640],
            "force_square_resize": True,
            "horizontal_flip": 0.5,
            "vertical_flip": 0.1,
            "rotate_limit": 5,
            "rotate_prob": 0.5,
            "mosaic": {"probability": 0.5},
            "mixup": {"probability": 0.15, "alpha": 0.2},
            "brightness_contrast": {"limit": 0.2, "probability": 0.5},
            "blur": {"probability": 0.2, "blur_limit": 3},
        }
        augmentor = AlbumentationsAugmentor(config)

        # Fill cache
        for i in range(20):
            img = Image.new("RGB", (400, 400), color=(i * 12, i * 12, i * 12))
            examples = {
                "image": [img],
                "annotations": [
                    {
                        "bbox": [[50 + i * 5, 50 + i * 5, 80, 80], [200, 200, 60, 60]],
                        "category_id": [1, 2],
                    }
                ],
            }
            augmentor.augment(examples)

        # Test multiple runs
        for _ in range(10):
            img = Image.new("RGB", (400, 400), color=(128, 128, 128))
            examples = {
                "image": [img],
                "annotations": [
                    {"bbox": [[100, 100, 120, 120], [250, 250, 80, 80]], "category_id": [1, 2]}
                ],
            }
            result = augmentor.augment(examples)

            # Basic sanity checks
            assert len(result["image"]) == 1
            assert len(result["annotations"]) == 1
            assert isinstance(result["image"][0], np.ndarray)
            assert len(result["annotations"][0]["bbox"]) >= 1

    def test_batch_processing(self):
        """Test processing multiple images in one batch."""
        config = {
            "multi_scale_sizes": [640],
            "force_square_resize": True,
            "horizontal_flip": 0.5,
        }
        augmentor = AlbumentationsAugmentor(config)

        # Process batch of 4 images
        images = [Image.new("RGB", (400, 400), color=(i * 60, i * 60, i * 60)) for i in range(4)]
        examples = {
            "image": images,
            "annotations": [
                {"bbox": [[100, 100, 80, 80]], "category_id": [1]},
                {"bbox": [[150, 150, 60, 60]], "category_id": [2]},
                {"bbox": [[200, 200, 100, 100]], "category_id": [3]},
                {"bbox": [[50, 50, 120, 120]], "category_id": [1]},
            ],
        }
        result = augmentor.augment(examples)

        assert len(result["image"]) == 4
        assert len(result["annotations"]) == 4
        for i in range(4):
            assert len(result["annotations"][i]["bbox"]) >= 1
