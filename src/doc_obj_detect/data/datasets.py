"""HF dataset loaders."""

from __future__ import annotations

from datasets import load_dataset

from doc_obj_detect.data.constants import DOCLAYNET_CLASSES, PUBLAYNET_CLASSES


class DatasetLoader:
    """Factory for huggingface datasets."""

    @staticmethod
    def load_publaynet(split: str, cache_dir: str | None) -> tuple[object, dict[int, str]]:
        dataset = load_dataset(
            "shunk031/PubLayNet",
            split=split,
            cache_dir=cache_dir,
        )
        return dataset, PUBLAYNET_CLASSES

    @staticmethod
    def load_doclaynet(split: str, cache_dir: str | None) -> tuple[object, dict[int, str]]:
        # Use DocLayNet-v1.1 which has been converted from COCO format
        # and doesn't require trust_remote_code
        dataset = load_dataset(
            "docling-project/DocLayNet-v1.1",
            split=split,
            cache_dir=cache_dir,
        )
        return dataset, DOCLAYNET_CLASSES
