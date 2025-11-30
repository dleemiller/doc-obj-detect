"""Dataset preparation pipeline."""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor

from doc_obj_detect.data.augmentor import AlbumentationsAugmentor
from doc_obj_detect.data.constants import (
    DOCSYNTH_DEFAULT_CLASS,
    DOCSYNTH_TO_PUBLAYNET_MAPPING,
    PUBLAYNET_ID_MAPPING,
)
from doc_obj_detect.data.datasets import DatasetLoader


def collate_fn(batch: list[dict]) -> dict:
    pixel_values = torch.stack([item["pixel_values"] for item in batch])
    pixel_mask = torch.stack([item["pixel_mask"] for item in batch])
    labels = [item["labels"] for item in batch]
    return {
        "pixel_values": pixel_values,
        "pixel_mask": pixel_mask,
        "labels": labels,
    }


class DatasetFactory:
    """Orchestrates dataset loading, augmentation, and preprocessing."""

    def __init__(
        self,
        dataset_name: str,
        image_processor: AutoImageProcessor,
        pad_stride: int,
        cache_dir: str | None = None,
        augmentation_config: dict[str, Any] | None = None,
    ):
        self.dataset_name = dataset_name.lower()
        self.image_processor = image_processor
        self.pad_stride = max(1, pad_stride)
        self.cache_dir = cache_dir
        self.augmentor = (
            AlbumentationsAugmentor(augmentation_config) if augmentation_config else None
        )

    def _load_split(self, split: str):
        if self.dataset_name == "publaynet":
            return DatasetLoader.load_publaynet(split, self.cache_dir), "publaynet"
        if self.dataset_name == "doclaynet":
            return DatasetLoader.load_doclaynet(split, self.cache_dir), "doclaynet"
        if self.dataset_name == "docsynth":
            return DatasetLoader.load_docsynth(split, self.cache_dir), "docsynth"
        raise ValueError(f"Unknown dataset: {self.dataset_name}")

    def build(
        self,
        split: str,
        max_samples: int | None = None,
        apply_augmentation: bool = False,
    ) -> tuple[object, dict[int, str]]:
        (dataset, class_labels), dataset_type = self._load_split(split)
        if max_samples is not None:
            max_idx = min(max_samples, len(dataset))
            dataset = dataset.select(range(max_idx))

        augment = self.augmentor.augment if (apply_augmentation and self.augmentor) else None
        preprocess = self._build_preprocessor(dataset_type)

        def transform_fn(examples: dict) -> dict:
            batch = augment(examples) if augment else examples
            return preprocess(batch)

        dataset = dataset.with_transform(transform_fn)
        return dataset, class_labels

    def _build_preprocessor(self, dataset_type: str) -> Callable[[dict], dict]:
        processor = self.image_processor

        def infer_resized_shape(height: int, width: int, size_dict: dict) -> tuple[int, int]:
            height = max(1, int(height))
            width = max(1, int(width))
            if not isinstance(size_dict, dict):
                return height, width
            if "height" in size_dict and "width" in size_dict:
                return max(1, int(size_dict["height"])), max(1, int(size_dict["width"]))
            if "shortest_edge" in size_dict:
                target_short = size_dict["shortest_edge"]
                short, long = sorted((height, width))
                scale = float(target_short) / max(1.0, float(short))
                new_height = max(1, int(round(height * scale)))
                new_width = max(1, int(round(width * scale)))
                max_long = size_dict.get("longest_edge")
                if max_long is not None:
                    current_long = max(new_height, new_width)
                    if current_long > max_long:
                        long_scale = float(max_long) / float(current_long)
                        new_height = max(1, int(round(new_height * long_scale)))
                        new_width = max(1, int(round(new_width * long_scale)))
                return new_height, new_width
            if "max_height" in size_dict and "max_width" in size_dict:
                return (
                    min(max(1, int(size_dict["max_height"])), height),
                    min(max(1, int(size_dict["max_width"])), width),
                )
            return height, width

        def obb_to_bbox(corners: list[float], img_width: int, img_height: int) -> list[float]:
            """Convert oriented bounding box (4 corners) to axis-aligned bbox (x, y, w, h)."""
            # corners format: [x1, y1, x2, y2, x3, y3, x4, y4] (normalized 0-1)
            # Extract all x and y coordinates
            x_coords = [corners[i] * img_width for i in range(0, 8, 2)]
            y_coords = [corners[i] * img_height for i in range(1, 8, 2)]

            # Find bounding box
            x_min, x_max = min(x_coords), max(x_coords)
            y_min, y_max = min(y_coords), max(y_coords)

            # Convert to COCO format (x, y, width, height)
            return [x_min, y_min, x_max - x_min, y_max - y_min]

        def preprocess_batch(examples: dict) -> dict:
            # Handle DocSynth format (image_data bytes instead of image)
            if dataset_type == "docsynth":
                import io

                images = []
                for img_data in examples["image_data"]:
                    img = Image.open(io.BytesIO(img_data))
                    images.append(img.convert("RGB"))
            else:
                images = []
                for img in examples["image"]:
                    if isinstance(img, np.ndarray):
                        if img.dtype != np.uint8:
                            img = np.clip(img, 0, 255).astype(np.uint8)
                        img = Image.fromarray(img)
                    images.append(img.convert("RGB"))

            annotations = []
            # Handle DocSynth format (oriented bounding boxes)
            if dataset_type == "docsynth":
                for idx, (img, anno_list) in enumerate(
                    zip(images, examples["anno_string"], strict=False)
                ):
                    img_width, img_height = img.size
                    anns_list = []

                    for anno_str in anno_list:
                        parts = anno_str.split()
                        if len(parts) != 9:  # class_id + 8 coordinates
                            continue

                        # Parse class ID and map to PubLayNet
                        docsynth_class = int(parts[0])
                        publaynet_class = DOCSYNTH_TO_PUBLAYNET_MAPPING.get(
                            docsynth_class, DOCSYNTH_DEFAULT_CLASS
                        )

                        # Convert OBB corners to axis-aligned bbox
                        corners = [float(p) for p in parts[1:]]
                        bbox = obb_to_bbox(corners, img_width, img_height)
                        area = bbox[2] * bbox[3]

                        anns_list.append(
                            {
                                "bbox": bbox,
                                "category_id": publaynet_class,
                                "area": area,
                                "iscrowd": 0,
                            }
                        )

                    annotations.append({"image_id": idx, "annotations": anns_list})

            # Handle both COCO format (publaynet) and flat format (doclaynet-v1.1)
            elif "annotations" in examples:
                # COCO format: examples["annotations"] is a list of dicts
                for img_id, anns_dict in zip(
                    examples["image_id"], examples["annotations"], strict=False
                ):
                    anns_list = []
                    for bbox, cat_id, area, crowd in zip(
                        anns_dict["bbox"],
                        anns_dict["category_id"],
                        anns_dict["area"],
                        anns_dict["iscrowd"],
                        strict=False,
                    ):
                        model_cat_id = (
                            PUBLAYNET_ID_MAPPING.get(cat_id, cat_id - 1)
                            if dataset_type == "publaynet"
                            else cat_id
                        )
                        anns_list.append(
                            {
                                "bbox": bbox,
                                "category_id": model_cat_id,
                                "area": area,
                                "iscrowd": crowd,
                            }
                        )
                    annotations.append({"image_id": img_id, "annotations": anns_list})
            else:
                # Flat format (DocLayNet-v1.1): bboxes and category_id are parallel lists
                img_ids = examples.get("metadata", [])
                if not img_ids:
                    img_ids = list(range(len(examples["bboxes"])))
                else:
                    img_ids = [m["image_id"] for m in img_ids]

                for img_id, bboxes_list, cat_ids_list in zip(
                    img_ids, examples["bboxes"], examples["category_id"], strict=False
                ):
                    anns_list = []
                    for bbox, cat_id in zip(bboxes_list, cat_ids_list, strict=False):
                        # Compute area from bbox (x, y, w, h)
                        area = bbox[2] * bbox[3] if len(bbox) >= 4 else 0
                        anns_list.append(
                            {
                                "bbox": bbox,
                                "category_id": cat_id,
                                "area": area,
                                "iscrowd": 0,  # DocLayNet doesn't have crowd annotations
                            }
                        )
                    annotations.append({"image_id": img_id, "annotations": anns_list})

            processor_size = getattr(processor, "size", None)
            do_resize = getattr(processor, "do_resize", False)
            candidate_sizes = []
            for img in images:
                h = getattr(img, "height", 1)
                w = getattr(img, "width", 1)
                if do_resize and isinstance(processor_size, dict):
                    candidate_sizes.append(infer_resized_shape(h, w, processor_size))
                else:
                    candidate_sizes.append((max(1, h), max(1, w)))

            target_height = max((h for h, _ in candidate_sizes), default=1)
            target_width = max((w for _, w in candidate_sizes), default=1)

            pad_height = max(1, math.ceil(target_height / self.pad_stride) * self.pad_stride)
            pad_width = max(1, math.ceil(target_width / self.pad_stride) * self.pad_stride)

            encoding = processor(
                images=images,
                annotations=annotations,
                return_tensors="pt",
                pad_size={"height": pad_height, "width": pad_width},
            )
            labels = list(encoding["labels"])

            # For DocLayNet: add original dimensions from metadata to labels
            if "metadata" in examples and examples["metadata"]:
                for i, metadata in enumerate(examples["metadata"]):
                    if (
                        i < len(labels)
                        and "original_height" in metadata
                        and "original_width" in metadata
                    ):
                        labels[i]["original_height"] = metadata["original_height"]
                        labels[i]["original_width"] = metadata["original_width"]

            return {
                "pixel_values": encoding["pixel_values"],
                "pixel_mask": encoding["pixel_mask"],
                "labels": labels,
            }

        return preprocess_batch
