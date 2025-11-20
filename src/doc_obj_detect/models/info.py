"""Model statistics helpers."""

from transformers import DFineForObjectDetection


def get_trainable_parameters(model: DFineForObjectDetection) -> dict[str, float]:
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params

    return {
        "total": total_params,
        "trainable": trainable_params,
        "frozen": frozen_params,
        "trainable_percent": 100 * trainable_params / total_params if total_params > 0 else 0.0,
    }
