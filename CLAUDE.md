# CLAUDE.md

This file provides guidance to Claude Code when working with this repository.

## Project Overview

Document object detection using ConvNeXt-DINOv3 backbones + D-FINE detection heads. Pipeline: pretrain on PubLayNet → fine-tune on DocLayNet → optional distillation to smaller models.

**Why ConvNeXt-DINOv3**: Self-supervised pretraining on 1.7B images provides strong dense prediction features for document understanding.

**Why D-FINE**: Fine-grained Distribution Refinement handles small objects better than standard DETR, critical for dense document layouts.

## Development Environment

**Python**: 3.12 (managed with uv)
**GPU**: Single NVIDIA GPU, CUDA 12.8
**Key tools**: HuggingFace Transformers, Albumentations, Augraphy, TensorBoard

## Essential Commands

```bash
# Training
uv run doc-obj-detect train --config configs/pretrain_publaynet.yaml
uv run doc-obj-detect train --config configs/finetune_doclaynet.yaml --load path/to/checkpoint

# Evaluation
uv run doc-obj-detect evaluate --checkpoint path/to/checkpoint --config configs/config.yaml

# Development
uv run pytest                    # Run tests
ruff check . && ruff format .   # Lint and format

# Visualization
uv run doc-obj-detect visualize --dataset publaynet --mode comparison
```

**See `agent_docs/training_commands.md` for complete CLI reference**

## Architecture

- **Data**: `src/doc_obj_detect/data/` - Datasets, augmentation pipeline
- **Training**: `src/doc_obj_detect/training/` - Trainers, EMA, callbacks
- **Models**: HuggingFace Transformers AutoModel (ConvNeXt + D-FINE from timm/transformers)
- **CLI**: `src/doc_obj_detect/cli/main.py` - All subcommands

## Key Workflows

### Training Pipeline
1. Load dataset via `DatasetLoader` (`src/doc_obj_detect/data/datasets.py`)
2. Apply augmentations via `AlbumentationsAugmentor` (`src/doc_obj_detect/data/augmentor.py`)
3. Train with EMA callback (`src/doc_obj_detect/training/callbacks.py:189`)
4. Evaluate with COCO metrics (`src/doc_obj_detect/training/evaluator.py`)

**See `agent_docs/augmentation_pipeline.md` for augmentation details**

### Checkpoint Management
- EMA weights automatically saved alongside regular weights (`ema_state.pt`)
- Use `--load` for transfer learning (fresh optimizer)
- Use `--resume-from-checkpoint` to continue interrupted training
- Use `finalize` command before HuggingFace upload (extracts EMA, removes training artifacts)

## Augmentation System

**Three-stage pipeline**: Geometric (always) → Photometric OR Augraphy (mutually exclusive) → Mosaic (optional)

**Control knob**: `augraphy.choice_probability` in config (0.0=photometric only, 1.0=augraphy only, 0.5=50/50 mix)

**Why two degradation paths**: Photometric uses classic CV transforms (fast). Augraphy simulates realistic document aging/scanning (slower but more realistic).

**See `agent_docs/augmentation_pipeline.md` for implementation details**

## Configuration

**Location**: `configs/`

- `pretrain_publaynet.yaml` - Production pretraining config
- `finetune_doclaynet.yaml` - Fine-tuning config
- `architectures/dfine_xlarge.yaml` - Model architecture parameters

**Format**: YAML with sections: model, dfine, data, augmentation, training, output

## Validation

```bash
# After code changes
uv run pytest                                    # Run tests
nvidia-smi                                       # Check GPU memory
uv run doc-obj-detect evaluate --checkpoint ... # Validate checkpoint loads

# After config changes
uv run doc-obj-detect visualize --mode comparison --config path/to/config.yaml
```

## Common Issues

**Import errors**: Always use `uv run` prefix (dependencies in uv-managed venv)
**GPU OOM**: Reduce `batch_size` in config or use gradient accumulation
**EMA not loading**: Check for `ema_state.pt` in checkpoint dir, use `--no-ema` if missing
**Augmentation bugs**: Visualize first with `visualize --mode comparison`
