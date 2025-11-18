# Document Layout Analysis with D-FINE

Modern document understanding needs both page-level accuracy and production-ready throughput. This project trains a ConvNeXt-DINOv3 backbone paired with the D-FINE detector, then fine-tunes on DocLayNet and distills into lighter RT-DETR variants. All tooling is driven through `uv` so dependencies stay reproducible and CUDA wheels are easy to swap.

## Highlights
- **Architecture:** ConvNeXt Large DINOv3 backbone feeding D-FINE (multi-scale encoder, hybrid matching) with 300 object queries.
- **Aspect-Preserving Multiscale:** Short edge randomly sampled from `[480, 512, 544, 576, 608, 640]` while keeping native aspect ratio and capping the long side at 928 px.
- **Datasets:** PubLayNet for pretraining (5 classes) → DocLayNet for fine-tuning (11 classes). Hugging Face Datasets handles download/caching. Config schemas now validate image sizes, batch sizes, and augmentations so invalid settings fail fast (e.g., multiscale sizes must be 64–4096 px, `max_long_side` ≥ max short edge).
- **Pipelines:** `configs/pretrain_publaynet.yaml`, `configs/finetune_doclaynet.yaml`, plus small-sample configs for debugging and evaluation.

## Quick Start
```bash
git clone <repo-url>
cd doc-obj-detect
uv sync --extra dev          # installs runtime + ruff/pre-commit
pre-commit install           # optional but recommended
```

### Train / Evaluate / Distill (Unified CLI)
```bash
uv run doc-obj-detect train --config configs/pretrain_publaynet.yaml
uv run doc-obj-detect evaluate --config configs/pretrain_publaynet.yaml \
    --checkpoint outputs/pretrain_publaynet_dfine/checkpoint-XXXX
uv run doc-obj-detect distill --config configs/distill.yaml
```

### Monitoring & Utilities
- TensorBoard: `tensorboard --logdir outputs/<run>/logs`
- Augmentation preview: `uv run doc-obj-detect visualize --dataset publaynet`
- Dataset sanity check: `uv run doc-obj-detect dataset-info --dataset doclaynet`

## Project Layout
```
src/doc_obj_detect/
├─ cli/                 # central CLI entry point + subcommand handlers
├─ config/              # Pydantic schemas + YAML loaders
├─ data/                # HF dataset wrappers, Albumentations augmentor, collate_fn
├─ models/              # ModelFactory + parameter utilities
├─ training/
│   ├─ base_runner.py   # shared dataset/processor/run-dir helpers
│   ├─ runner.py        # TrainerRunner (teacher training)
│   ├─ distill_runner.py# DistillRunner (student KD training)
│   ├─ distillation.py  # DistillationTrainer (KL/MSE KD losses)
│   ├─ evaluator.py     # EvaluatorRunner (checkpoint eval loop)
│   ├─ trainer_core.py  # SplitLRTrainer (custom HF Trainer subclass)
│   └─ callbacks.py     # Backbone unfreeze, etc.
├─ metrics.py           # COCO-style mAP via torchmetrics
└─ visualize.py         # Augmentation visualization utility
configs/                # YAML configs for pretrain, finetune, distill, testing
tests/                  # Pytest suites covering configs/data/runners/CLI
```

## Key Settings
- **Optimizer:** AdamW, `learning_rate=1e-4`, gradient clipping at 1.0, BF16 mixed precision.
- **Batch size:** 16 images/device for long runs (use multi-GPU data parallel if needed).
- **Scheduler:** Cosine with `min_lr=1e-5` (mirrors SOTA doc-det pipelines); step LR drop optional.
- **Evaluation:** Short edge fixed at the max multiscale size (640) with the same 928 px long-edge cap to match training augmentations. Validators guarantee `max_long_side` is never smaller than the largest training scale.

## Notes
- All scripts run through the CLI (`uv run doc-obj-detect ...`) so we don’t accumulate separate entry points.
- Distillation currently exposes logit KL/MSE loss with optional predicted-box MSE; GO-LSD-style distribution distillation is scoped for a future milestone (see RESEARCH.md §4.1).
- PubLayNet/DocLayNet downloads rely on Hugging Face Datasets; set `data.cache_dir` if `$HF_HOME` is not writable.

See [RESEARCH.md](RESEARCH.md) for architecture experiments and ablations, and `CLAUDE.md` / `AGENTS.md` for contributor conventions.
