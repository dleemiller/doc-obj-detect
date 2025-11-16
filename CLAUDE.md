# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This project implements document object detection using vision transformers and DETR-based models. The goal is to achieve SOTA performance with a large perception encoder-based model, then distill to smaller, faster models suitable for production deployment.

**Target Pipeline:**
1. Pretrain on PubLayNet dataset
2. Fine-tune on DocLayNet dataset
3. Distill to smaller models (e.g., ConvNeXtV2 + RT-DETR)

**Model Architecture:**
- Backbone: Vision Transformer with Perception Encoder (ViT-PE Spatial Gigantic - 1.8B params)
  - Model: timm/vit_pe_spatial_gigantic_patch14_448.fb
  - Key insight: Uses intermediate layer features for optimal detection performance
- Detection Head: Deformable DETR with multi-scale deformable attention (num_feature_levels=4)
- Distillation targets: Smaller backbone/detection combinations for deployment (e.g., ConvNeXtV2 + RT-DETR)

## Development Environment

**Python:** 3.12 (managed with uv)
**Package Manager:** uv
**Linting/Formatting:** ruff
**Testing:** pytest (target 70% coverage of critical paths)
**Pre-commit:** Used for code quality checks
**Training Hardware:** Single NVIDIA Blackwell Pro 6000 (96GB VRAM), CUDA 12.8

## Common Commands

**Project Scripts:**
The project uses uv scripts for processing and training tasks:
```bash
# Data preprocessing
uv run preprocess-data publaynet
uv run preprocess-data doclaynet

# Training (implemented and tested)
uv run train --config configs/pretrain_publaynet.yaml
uv run train --config configs/test_quick.yaml  # Quick 30-step test

# Visualization
uv run visualize-augmentations --num-samples 4 --output-dir outputs/viz

# Future: Distillation
uv run distill --teacher-checkpoint path/to/checkpoint --config configs/distill.yaml
```

**Development:**
```bash
# Install dependencies
uv sync

# Run linting/formatting
ruff check .
ruff format .

# Run tests
uv run pytest
uv run pytest tests/test_data.py  # Single test file
uv run pytest tests/test_data.py::test_apply_augmentations  # Single test

# Pre-commit hooks
pre-commit run --all-files
```

## Architecture Guidelines

**Required Stack:**
- HuggingFace Transformers: All models and trainers
- HuggingFace Datasets: Data loading and processing
- Albumentations: Data augmentation (choose augmentations appropriate for document images)
- TensorBoard: Training monitoring and visualization

**Data Augmentation Considerations:**
Document object detection requires different augmentations than natural images:
- Preserve text readability (use small rotation limits like ±5°)
- Maintain document structure and layout
- Use geometric transformations that simulate real scanning/camera variations
- Avoid heavy color augmentations that break document appearance
- **CRITICAL:** Avoid RandomShadow or similar occlusion augmentations that cover large portions of content
- Focus on realistic scanning artifacts: brightness, contrast, blur, noise, JPEG compression

**Training Approach:**
- Use HuggingFace Trainers for all training loops
- Implement custom metrics for object detection (mAP, etc.)
- Monitor training with TensorBoard

**Distillation Strategy:**
- Carefully balance distillation loss (MSE vs KL divergence) with ground truth label losses (e.g., cross-entropy)
- Experiment with loss weight ratios to optimize student model performance
- The large model serves as teacher; smaller models (ConvNeXtV2 + RT-DETR) as students

**Code Quality:**
- Follow DRY principles rigorously
- Target 70% test coverage for critical components (data loading, model inference, training logic, distillation)
- Use type hints throughout
- Maintain clean separation between data processing, model definition, training, and evaluation

## Project Structure (Expected)

The codebase should organize around these key components:
- **Data processing pipeline:** Scripts for downloading, preprocessing PubLayNet/DocLayNet
- **Model definitions:** Vision backbone + detection head implementations using Transformers
- **Training modules:** Pretraining, fine-tuning, and distillation trainers
- **Evaluation:** Object detection metrics and visualization
- **Augmentation:** Document-specific augmentation pipelines with Albumentations
- **Configuration:** YAML configs for different training stages and model variants
