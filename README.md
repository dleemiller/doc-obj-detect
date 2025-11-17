# Document Layout Analysis with D-FINE

Modern document understanding needs both page-level accuracy and production-ready throughput. This project trains a ConvNeXt-DINOv3 backbone paired with the D-FINE detector, then fine-tunes on DocLayNet and distills into lighter RT-DETR variants. All tooling is driven through `uv` so dependencies stay reproducible and CUDA wheels are easy to swap.

## Highlights
- **Architecture:** ConvNeXt Large DINOv3 backbone feeding D-FINE (multi-scale encoder, hybrid matching) with 300 object queries.
- **Aspect-Preserving Multiscale:** Short edge randomly sampled from `[480, 512, 544, 576, 608, 640]` while keeping native aspect ratio and capping the long side at 928 px.
- **Datasets:** PubLayNet for pretraining (5 classes) → DocLayNet for fine-tuning (11 classes). Hugging Face Datasets handles download/caching.
- **Pipelines:** `configs/pretrain_publaynet.yaml`, `configs/finetune_doclaynet.yaml`, plus small-sample configs for debugging and evaluation.

## Quick Start
```bash
git clone <repo-url>
cd doc-obj-detect
uv sync --extra dev          # installs runtime + ruff/pre-commit
pre-commit install           # optional but recommended
```

### Train & Evaluate
```bash
uv run train --config configs/pretrain_publaynet.yaml
uv run python src/doc_obj_detect/evaluate.py \
    --checkpoint outputs/pretrain_publaynet_dfine/checkpoint-XXXX
```

### Monitoring & Utilities
- TensorBoard: `tensorboard --logdir outputs/pretrain_publaynet_dfine/logs`
- VRAM sweep: `uv run python scripts/benchmark_vram.py`
- Augmentation gallery: `uv run visualize-augmentations publaynet`

## Project Layout
```
src/doc_obj_detect/
├─ config.py          # Pydantic schemas (model/data/augmentation/training)
├─ data.py            # HF dataset wrappers + aspect-preserving multiscale aug
├─ model.py           # D-FINE creation + processor loading
├─ train.py / trainer.py
├─ evaluate.py        # Standalone evaluation script
└─ visualize.py       # Augmentation visualizations
configs/              # YAML configs for pretrain, finetune, distill, tests
scripts/              # Eval helpers, VRAM benchmarks
tests/                # Pytest smoke tests for data/aug logic
```

## Key Settings
- **Optimizer:** AdamW, `learning_rate=1e-4`, gradient clipping at 1.0, BF16 mixed precision.
- **Batch size:** 16 images/device for long runs (use multi-GPU data parallel if needed).
- **Scheduler:** Cosine with `min_lr=1e-5` (mirrors SOTA doc-det pipelines); step LR drop optional.
- **Evaluation:** Short edge fixed at the max multiscale size (640) with the same 928 px long-edge cap to match training augmentations.

## Notes
- Always run tooling via `uv run …`; plain `python` or `pre-commit` outside uv won’t see the locked dependencies.
- PubLayNet requires local HF caches (`datasets` warns that `trust_remote_code` is deprecated). If cache permissions block you, copy the dataset directory into a workspace-owned path and set `data.cache_dir`.
- For DocLayNet training use 24 epochs as outlined in the paper and consider staged multiscale (start at `[512]`, widen after a few epochs) if gradients are noisy.

See [RESEARCH.md](RESEARCH.md) for architecture experiments and ablations, and `CLAUDE.md` / `AGENTS.md` for contributor conventions.
