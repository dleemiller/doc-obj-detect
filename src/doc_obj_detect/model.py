"""Backward-compatible shims for the new models package."""

from doc_obj_detect.models.builder import ModelFactory, create_model
from doc_obj_detect.models.info import get_trainable_parameters

__all__ = ["ModelFactory", "create_model", "get_trainable_parameters"]
