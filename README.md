# Document Layout Analysis with D-FINE

Modern document understanding needs both page-level accuracy and production-ready throughput. This project trains a ConvNeXt‑DINOv3 backbone paired with the D‑FINE detector, fine‑tunes on DocLayNet, and then distills into smaller D‑FINE variants (ConvNeXt Base / Small / Tiny). All tooling flows through `uv run doc-obj-detect …`, so dependencies stay reproducible and CUDA wheels are easy to swap.

## Highlights
- **Teacher architecture:** ConvNeXt Base DINOv3 backbone feeding D-FINE (keeps VRAM low enough for a 4-level stride pyramid) with 300 queries.
- **Aspect-preserving multiscale:** Short side sampled from `[576, 608, 640]`, long side capped at 928 px; evaluation always resizes to a fixed 640 px short side with no augmentation.
- **Students:** Smaller ConvNeXt backbones (Base / Small / Tiny) paired with D-FINE; distillation loss supports KL (with temperature) or MSE on logits plus optional box-level MSE.
- **Datasets:** PubLayNet pretraining (5 classes) followed by DocLayNet finetuning (11 classes). Config schemas enforce sane ranges for image sizes, batch sizes, learning rates, multiscale settings, etc., so misconfigured runs fail fast.
- **Configs:** `configs/pretrain_publaynet.yaml`, `configs/finetune_doclaynet.yaml`, and `configs/distill.yaml` describe the entire pipeline; modify YAML rather than editing code.

## Quick Start
```bash
git clone git@github.com:dleemiller/doc-obj-detect.git
cd doc-obj-detect
```

### Train / Evaluate / Distill (Unified CLI)
```bash
# Pretraining / finetuning
uv run doc-obj-detect train --config configs/pretrain_publaynet.yaml
uv run doc-obj-detect train --config configs/finetune_doclaynet.yaml

# Checkpoint evaluation (short side fixed at 640 px, no augmentation)
uv run doc-obj-detect evaluate \
  --config configs/pretrain_publaynet.yaml \
  --checkpoint outputs/pretrain_publaynet_dfine/checkpoint-1234

# Distil teacher -> smaller D-FINE student
uv run doc-obj-detect distill --config configs/distill.yaml
```

### Monitoring & Utilities
- TensorBoard: `tensorboard --logdir outputs/<run>/logs`
- Augmentation preview: `uv run doc-obj-detect visualize --dataset publaynet`
- Dataset sanity check: `uv run doc-obj-detect dataset-info --dataset doclaynet`

## Project Layout
```
src/doc_obj_detect/
├─ cli/                 # central CLI entry point + subcommand handlers (train/eval/distill/etc.)
├─ config/              # Pydantic schemas, YAML loaders, validator logic
├─ data/                # HF dataset wrappers, Albumentations augmentor, collate_fn
├─ models/              # ModelFactory + parameter statistics
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
- **Evaluation:** Short edge fixed at 640 px with a 928 px long-edge cap (no augmentations). Validators guarantee `max_long_side` is never smaller than the largest training scale.

## Notes
- All scripts run through the CLI (`uv run doc-obj-detect ...`) so we don’t accumulate separate entry points.
- Distillation currently exposes logit KL/MSE loss with optional predicted-box MSE; GO-LSD-style distribution distillation is planned (see RESEARCH.md §4.1).
- PubLayNet/DocLayNet downloads rely on Hugging Face Datasets; set `data.cache_dir` if `$HF_HOME` is not writable.
- `RESEARCH.md` captures architecture decisions and training rationale (it is not an ablation log).

See [RESEARCH.md](./docs/RESEARCH.md) for the design rationale plus future work, and `CLAUDE.md` / `AGENTS.md` for contributor conventions.
