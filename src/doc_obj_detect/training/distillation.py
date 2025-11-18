"""Trainer extensions for knowledge distillation."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from doc_obj_detect.config import DistillationConfig
from doc_obj_detect.training.trainer_core import SplitLRTrainer


class DistillationTrainer(SplitLRTrainer):
    """SplitLRTrainer with an auxiliary distillation loss."""

    def __init__(self, *args, teacher_model=None, distill_config: DistillationConfig, **kwargs):
        if teacher_model is None:
            raise ValueError("teacher_model must be provided for distillation")
        self.teacher_model = teacher_model
        self.distill_config = distill_config
        self._teacher_device = None
        super().__init__(*args, **kwargs)

        self.teacher_model.eval()
        for param in self.teacher_model.parameters():
            param.requires_grad = False

    def _ensure_teacher_device(self):
        if self._teacher_device is None:
            self._teacher_device = next(self.teacher_model.parameters()).device
        model_device = next(self.model.parameters()).device
        if self._teacher_device != model_device:
            self.teacher_model.to(model_device)
            self._teacher_device = model_device

    def compute_loss(self, model, inputs, return_outputs=False):  # type: ignore[override]
        self._ensure_teacher_device()
        student_outputs = model(**inputs)
        student_loss = student_outputs.loss

        with torch.no_grad():
            teacher_inputs = {
                "pixel_values": inputs["pixel_values"],
                "pixel_mask": inputs.get("pixel_mask"),
            }
            teacher_outputs = self.teacher_model(**teacher_inputs)

        kd_loss = self._compute_distillation_loss(student_outputs, teacher_outputs)
        total_loss = self.distill_config.alpha * kd_loss + self.distill_config.beta * student_loss

        if return_outputs:
            return total_loss, student_outputs
        return total_loss

    def _compute_distillation_loss(self, student_outputs, teacher_outputs):
        loss_type = self.distill_config.loss_type
        if loss_type == "kl":
            temperature = self.distill_config.temperature
            student_logits = student_outputs.logits / temperature
            teacher_logits = teacher_outputs.logits / temperature
            kd_loss = F.kl_div(
                F.log_softmax(student_logits, dim=-1),
                F.softmax(teacher_logits, dim=-1),
                reduction="batchmean",
            ) * (temperature**2)
        else:  # mse
            kd_loss = F.mse_loss(student_outputs.logits, teacher_outputs.logits)

        if self.distill_config.distill_features:
            kd_loss = kd_loss + F.mse_loss(student_outputs.pred_boxes, teacher_outputs.pred_boxes)

        return kd_loss
