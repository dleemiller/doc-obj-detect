"""HF dataset loaders."""

from __future__ import annotations

from datasets import load_dataset

from doc_obj_detect.data.constants import (
    DOCLAYNET_CLASSES,
    DOCSYNTH_DEFAULT_CLASS,
    DOCSYNTH_TO_PUBLAYNET_MAPPING,
    PUBLAYNET_CLASSES,
)


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

    @staticmethod
    def load_docsynth(split: str, cache_dir: str | None) -> tuple[object, dict[int, str]]:
        """Load DocSynth-300K dataset with class mapping to PubLayNet.

        DocSynth-300K uses M6Doc's 74 element types, which we map to PubLayNet's 5 classes.
        The dataset only has a 'train' split, so we use it for all splits.

        Args:
            split: Dataset split ('train' only available)
            cache_dir: Optional cache directory

        Returns:
            Tuple of (dataset, class_mapping) where class_mapping uses PubLayNet classes
        """
        # DocSynth-300K only has 'train' split
        if split not in ["train", "validation", "val"]:
            raise ValueError(f"DocSynth-300K only supports 'train' split, got '{split}'")

        # Load the dataset (use 'train' for all splits since only train is available)
        dataset = load_dataset(
            "juliozhao/DocSynth300K",
            split="train",
            cache_dir=cache_dir,
        )

        # Return dataset with PubLayNet classes since we'll map DocSynth → PubLayNet
        return dataset, PUBLAYNET_CLASSES
