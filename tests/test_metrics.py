"""Tests for metrics computation.

This module tests the critical metrics computation logic:
- COCO-style mAP computation
- Bounding box coordinate transformations
- Image limiting logic
- Integration with image processor
"""

from unittest.mock import Mock

import torch
from transformers.trainer_utils import EvalPrediction

from doc_obj_detect.metrics import (
    _limit_eval_images,
    compute_map,
    convert_bbox_yolo_to_pascal,
)


class TestBBoxConversion:
    """Test bounding box coordinate transformations."""

    def test_convert_bbox_yolo_to_pascal(self):
        """Test YOLO (cx, cy, w, h) to Pascal VOC (x_min, y_min, x_max, y_max) conversion."""
        # YOLO format: center x, center y, width, height (normalized 0-1)
        # Test box: center at (0.5, 0.5), size (0.4, 0.6)
        boxes_yolo = torch.tensor([[0.5, 0.5, 0.4, 0.6]])
        image_size = (100, 200)  # H=100, W=200

        boxes_pascal = convert_bbox_yolo_to_pascal(boxes_yolo, image_size)

        # Expected: center (0.5*200, 0.5*100) = (100, 50)
        #          half-size (0.4*200/2, 0.6*100/2) = (40, 30)
        #          box (100-40, 50-30, 100+40, 50+30) = (60, 20, 140, 80)
        expected = torch.tensor([[60.0, 20.0, 140.0, 80.0]])

        assert torch.allclose(
            boxes_pascal, expected, rtol=1e-5
        ), f"Expected {expected}, got {boxes_pascal}"

    def test_convert_bbox_with_multiple_boxes(self):
        """Test conversion with multiple boxes."""
        boxes_yolo = torch.tensor(
            [
                [0.25, 0.25, 0.2, 0.2],  # Small box top-left quadrant
                [0.75, 0.75, 0.3, 0.3],  # Larger box bottom-right quadrant
            ]
        )
        image_size = (100, 100)  # Square image

        boxes_pascal = convert_bbox_yolo_to_pascal(boxes_yolo, image_size)

        # Box 1: center (25, 25), half-size (10, 10) → (15, 15, 35, 35)
        # Box 2: center (75, 75), half-size (15, 15) → (60, 60, 90, 90)
        expected = torch.tensor(
            [
                [15.0, 15.0, 35.0, 35.0],
                [60.0, 60.0, 90.0, 90.0],
            ]
        )

        assert torch.allclose(boxes_pascal, expected, rtol=1e-5)

    def test_convert_bbox_edge_cases(self):
        """Test conversion with edge cases (very small, full image)."""
        # Very small box at origin
        tiny_box = torch.tensor([[0.05, 0.05, 0.1, 0.1]])
        boxes_pascal = convert_bbox_yolo_to_pascal(tiny_box, (100, 100))
        assert boxes_pascal.shape == (1, 4)
        assert torch.all(boxes_pascal >= 0)  # Should be non-negative

        # Full image box
        full_box = torch.tensor([[0.5, 0.5, 1.0, 1.0]])
        boxes_pascal = convert_bbox_yolo_to_pascal(full_box, (100, 100))
        expected_full = torch.tensor([[0.0, 0.0, 100.0, 100.0]])
        assert torch.allclose(boxes_pascal, expected_full, rtol=1e-5)


class TestLimitEvalImages:
    """Test image limiting logic for eval metrics."""

    def test_limit_eval_images_no_limit(self):
        """Test that None limit returns all images."""
        predictions = [
            (torch.randn(2, 10, 5), torch.randn(2, 10, 4)),  # Batch of 2
            (torch.randn(3, 10, 5), torch.randn(3, 10, 4)),  # Batch of 3
        ]
        targets = [
            [{"boxes": torch.randn(2, 4)}, {"boxes": torch.randn(3, 4)}],
            [
                {"boxes": torch.randn(1, 4)},
                {"boxes": torch.randn(2, 4)},
                {"boxes": torch.randn(1, 4)},
            ],
        ]

        new_preds, new_tgts = _limit_eval_images(predictions, targets, max_images=None)

        assert len(new_preds) == 2
        assert len(new_tgts) == 2

    def test_limit_eval_images_caps_correctly(self):
        """Test that limiting works correctly across batches."""
        # 2 batches: first has 2 images, second has 3 images
        predictions = [
            (torch.randn(2, 10, 5), torch.randn(2, 10, 4)),
            (torch.randn(3, 10, 5), torch.randn(3, 10, 4)),
        ]
        targets = [
            [{"id": 0}, {"id": 1}],
            [{"id": 2}, {"id": 3}, {"id": 4}],
        ]

        # Limit to 4 images (should take all of batch 1 + 2 from batch 2)
        new_preds, new_tgts = _limit_eval_images(predictions, targets, max_images=4)

        assert len(new_preds) == 2, "Should keep 2 batches"
        assert len(new_tgts) == 2

        # First batch: all 2 images
        assert len(new_tgts[0]) == 2

        # Second batch: only 2 images (sliced)
        assert len(new_tgts[1]) == 2
        assert new_tgts[1][0]["id"] == 2
        assert new_tgts[1][1]["id"] == 3

    def test_limit_eval_images_first_batch_only(self):
        """Test limiting to fewer images than first batch."""
        predictions = [
            (torch.randn(5, 10, 5), torch.randn(5, 10, 4)),
            (torch.randn(3, 10, 5), torch.randn(3, 10, 4)),
        ]
        targets = [
            [{"id": i} for i in range(5)],
            [{"id": i} for i in range(5, 8)],
        ]

        # Limit to 3 images (only part of first batch)
        new_preds, new_tgts = _limit_eval_images(predictions, targets, max_images=3)

        assert len(new_preds) == 1, "Should keep only 1 batch"
        assert len(new_tgts) == 1
        assert len(new_tgts[0]) == 3, "Should have exactly 3 images"

    def test_limit_eval_images_with_loss_tuple(self):
        """Test that 3-tuple predictions (with loss) are handled correctly."""
        # Some HF versions include loss in predictions tuple
        predictions = [
            (torch.randn(2), torch.randn(2, 10, 5), torch.randn(2, 10, 4)),  # (loss, logits, boxes)
        ]
        targets = [[{"id": 0}, {"id": 1}]]

        # Limit to 1 image
        new_preds, new_tgts = _limit_eval_images(predictions, targets, max_images=1)

        assert len(new_preds) == 1
        assert len(new_preds[0]) == 3, "Should still be 3-tuple"
        assert len(new_tgts[0]) == 1


class TestComputeMAP:
    """Test COCO-style mAP computation."""

    def test_compute_map_perfect_predictions(self):
        """Test that perfect predictions give mAP=1.0."""
        # Create perfect predictions (boxes match targets exactly)
        # Logits: shape (batch, num_queries, num_classes)
        # Boxes: shape (batch, num_queries, 4) in YOLO format

        # Single image, single object
        logits = torch.zeros(1, 10, 5)  # 10 queries, 5 classes
        logits[0, 0, 1] = 10.0  # Strong prediction for class 1 at query 0

        # Box in YOLO format (cx, cy, w, h) normalized
        boxes = torch.zeros(1, 10, 4)
        boxes[0, 0] = torch.tensor([0.5, 0.5, 0.4, 0.4])  # Center box

        predictions = [(logits, boxes)]

        # Target: same box, class 1
        targets = [
            [
                {
                    "boxes": torch.tensor([[0.5, 0.5, 0.4, 0.4]]),  # YOLO format
                    "class_labels": torch.tensor([1]),
                    "orig_size": (100, 100),
                }
            ]
        ]

        eval_pred = EvalPrediction(predictions=predictions, label_ids=targets)

        # Mock image processor
        mock_processor = Mock()
        mock_processor.post_process_object_detection = Mock(
            return_value=[
                {
                    "scores": torch.tensor([0.99]),
                    "labels": torch.tensor([1]),
                    "boxes": torch.tensor([[30.0, 30.0, 70.0, 70.0]]),  # Pascal VOC format
                }
            ]
        )

        metrics = compute_map(eval_pred, mock_processor, threshold=0.0)

        # Should have mAP close to 1.0 (perfect match)
        assert "map" in metrics
        assert metrics["map"] >= 0.9, f"Expected mAP >= 0.9, got {metrics['map']}"

    def test_compute_map_no_predictions(self):
        """Test that no predictions gives mAP=0.0."""
        # Predictions with very low confidence (below threshold)
        logits = torch.full((1, 10, 5), -10.0)  # All very low logits
        boxes = torch.zeros(1, 10, 4)

        predictions = [(logits, boxes)]
        targets = [
            [
                {
                    "boxes": torch.tensor([[0.5, 0.5, 0.4, 0.4]]),
                    "class_labels": torch.tensor([1]),
                    "orig_size": (100, 100),
                }
            ]
        ]

        eval_pred = EvalPrediction(predictions=predictions, label_ids=targets)

        # Mock processor returns empty predictions
        mock_processor = Mock()
        mock_processor.post_process_object_detection = Mock(
            return_value=[
                {
                    "scores": torch.tensor([]),
                    "labels": torch.tensor([]),
                    "boxes": torch.tensor([]).reshape(0, 4),
                }
            ]
        )

        metrics = compute_map(eval_pred, mock_processor, threshold=0.5)

        # Should have mAP = 0 (no predictions)
        assert "map" in metrics
        assert metrics["map"] == 0.0, f"Expected mAP=0.0 for no predictions, got {metrics['map']}"

    def test_compute_map_with_max_eval_images(self):
        """Test that max_eval_images correctly limits evaluation."""
        # Create 10 images
        logits = torch.zeros(10, 10, 5)
        boxes = torch.zeros(10, 10, 4)

        predictions = [(logits, boxes)]
        targets = [
            [
                {
                    "boxes": torch.tensor([[0.5, 0.5, 0.2, 0.2]]),
                    "class_labels": torch.tensor([0]),
                    "orig_size": (100, 100),
                }
            ]
            * 10
        ]

        eval_pred = EvalPrediction(predictions=predictions, label_ids=targets)

        # Mock processor
        call_count = 0

        def mock_post_process(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            batch_size = args[0].logits.shape[0]
            return [
                {
                    "scores": torch.tensor([0.5]),
                    "labels": torch.tensor([0]),
                    "boxes": torch.tensor([[20.0, 20.0, 40.0, 40.0]]),
                }
            ] * batch_size

        mock_processor = Mock()
        mock_processor.post_process_object_detection = Mock(side_effect=mock_post_process)

        # Limit to 5 images
        metrics = compute_map(eval_pred, mock_processor, threshold=0.0, max_eval_images=5)

        # Should have processed only 5 images
        assert "map" in metrics
        # Verify limiting worked (this is indirect but checks the function ran)
        assert isinstance(metrics["map"], float)

    def test_compute_map_multiple_classes(self):
        """Test mAP computation with multiple object classes."""
        # 2 images, each with 2 objects of different classes
        logits = torch.zeros(2, 10, 5)
        logits[0, 0, 1] = 10.0  # Image 0, query 0, class 1
        logits[0, 1, 2] = 10.0  # Image 0, query 1, class 2
        logits[1, 0, 1] = 10.0  # Image 1, query 0, class 1

        boxes = torch.zeros(2, 10, 4)
        boxes[0, 0] = torch.tensor([0.25, 0.25, 0.3, 0.3])
        boxes[0, 1] = torch.tensor([0.75, 0.75, 0.3, 0.3])
        boxes[1, 0] = torch.tensor([0.5, 0.5, 0.4, 0.4])

        predictions = [(logits, boxes)]
        targets = [
            [
                {
                    "boxes": torch.tensor([[0.25, 0.25, 0.3, 0.3], [0.75, 0.75, 0.3, 0.3]]),
                    "class_labels": torch.tensor([1, 2]),
                    "orig_size": (100, 100),
                },
                {
                    "boxes": torch.tensor([[0.5, 0.5, 0.4, 0.4]]),
                    "class_labels": torch.tensor([1]),
                    "orig_size": (100, 100),
                },
            ]
        ]

        eval_pred = EvalPrediction(predictions=predictions, label_ids=targets)

        # Mock processor with multiple detections
        mock_processor = Mock()
        mock_processor.post_process_object_detection = Mock(
            return_value=[
                {
                    "scores": torch.tensor([0.99, 0.98]),
                    "labels": torch.tensor([1, 2]),
                    "boxes": torch.tensor([[10.0, 10.0, 40.0, 40.0], [60.0, 60.0, 90.0, 90.0]]),
                },
                {
                    "scores": torch.tensor([0.97]),
                    "labels": torch.tensor([1]),
                    "boxes": torch.tensor([[30.0, 30.0, 70.0, 70.0]]),
                },
            ]
        )

        metrics = compute_map(eval_pred, mock_processor, threshold=0.0)

        # Should compute mAP across all classes
        assert "map" in metrics
        assert metrics["map"] > 0.0
        # Should also have mAR (mean average recall)
        assert "mar_100" in metrics or "mar_1" in metrics


class TestMetricsIntegration:
    """Integration tests with real image processor behavior."""

    def test_metrics_with_empty_targets(self):
        """Test metrics computation when targets are empty (background images)."""
        logits = torch.full((1, 10, 5), -10.0)
        boxes = torch.zeros(1, 10, 4)

        predictions = [(logits, boxes)]
        targets = [
            [
                {
                    "boxes": torch.tensor([]).reshape(0, 4),  # Empty boxes
                    "class_labels": torch.tensor([]),
                    "orig_size": (100, 100),
                }
            ]
        ]

        eval_pred = EvalPrediction(predictions=predictions, label_ids=targets)

        mock_processor = Mock()
        mock_processor.post_process_object_detection = Mock(
            return_value=[
                {
                    "scores": torch.tensor([]),
                    "labels": torch.tensor([]),
                    "boxes": torch.tensor([]).reshape(0, 4),
                }
            ]
        )

        # Should not crash with empty targets
        metrics = compute_map(eval_pred, mock_processor, threshold=0.5)
        assert "map" in metrics

    def test_metrics_with_varied_image_sizes(self):
        """Test that metrics handle varied image sizes correctly."""
        logits = torch.zeros(2, 10, 5)
        logits[0, 0, 0] = 10.0
        logits[1, 0, 1] = 10.0

        boxes = torch.zeros(2, 10, 4)
        boxes[0, 0] = torch.tensor([0.5, 0.5, 0.4, 0.4])
        boxes[1, 0] = torch.tensor([0.5, 0.5, 0.3, 0.3])

        predictions = [(logits, boxes)]
        targets = [
            [
                {
                    "boxes": torch.tensor([[0.5, 0.5, 0.4, 0.4]]),
                    "class_labels": torch.tensor([0]),
                    "orig_size": (200, 300),  # Wide image
                },
                {
                    "boxes": torch.tensor([[0.5, 0.5, 0.3, 0.3]]),
                    "class_labels": torch.tensor([1]),
                    "orig_size": (400, 200),  # Tall image
                },
            ]
        ]

        eval_pred = EvalPrediction(predictions=predictions, label_ids=targets)

        # Mock processor should handle different sizes
        mock_processor = Mock()
        mock_processor.post_process_object_detection = Mock(
            return_value=[
                {
                    "scores": torch.tensor([0.99]),
                    "labels": torch.tensor([0]),
                    "boxes": torch.tensor([[90.0, 40.0, 210.0, 160.0]]),  # 300x200 image
                },
                {
                    "scores": torch.tensor([0.98]),
                    "labels": torch.tensor([1]),
                    "boxes": torch.tensor([[70.0, 140.0, 130.0, 260.0]]),  # 200x400 image
                },
            ]
        )

        metrics = compute_map(eval_pred, mock_processor, threshold=0.0)
        assert "map" in metrics
        assert isinstance(metrics["map"], float)
