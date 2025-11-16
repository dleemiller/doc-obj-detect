# Document Object Detection

State-of-the-art document layout analysis using Vision Transformers and Deformable DETR. This project implements a three-stage pipeline: pretraining on PubLayNet, fine-tuning on DocLayNet, and knowledge distillation to efficient models for production deployment.

## Overview

**Goal:** Achieve SOTA document object detection quality with a large teacher model, then distill to smaller, faster models suitable for production.

**Architecture:**
- **Backbone:** Vision Transformer with Perception Encoder (ViT-PE Spatial)
- **Detection Head:** Deformable DETR with multi-scale deformable attention
- **Framework:** HuggingFace Transformers + Datasets

**Training Pipeline:**
1. **Pretrain** on PubLayNet (335k images, 5 classes)
2. **Fine-tune** on DocLayNet (~80k images, 11 classes)
3. **Distill** to efficient models (e.g., ConvNeXtV2 + RT-DETR)

See [RESEARCH.md](RESEARCH.md) for detailed architecture rationale and empirical testing results.

## Quick Start

### Installation

```bash
# Clone repository
git clone <repository-url>
cd doc-obj-detect

# Install dependencies with uv
uv sync
```

### Training

```bash
# Pretrain on PubLayNet
uv run train --config configs/pretrain_publaynet.yaml

# Monitor training with TensorBoard
tensorboard --logdir outputs/pretrain_publaynet/logs
```

### Data Preprocessing

```bash
# Preprocess datasets (automatically downloads from HuggingFace)
uv run preprocess-data publaynet
uv run preprocess-data doclaynet
```

### Visualization

```bash
# Visualize data augmentations
uv run visualize-augmentations
```

## Model Architecture

### Teacher Model (Current)

```yaml
Backbone: ViT-PE-Spatial-Gigantic (1.8B parameters)
  - Input: 448×448 images
  - Features: Multi-scale intermediate layers
  - Pretrained: Yes (Facebook Research PE weights)

Detection Head: Deformable DETR
  - Queries: 300
  - Encoder/Decoder: 6 layers each
  - Feature levels: 4 (multi-scale)
  - Attention heads: 8
```

**Performance Target:** 92-96 mAP on PubLayNet validation set

### Student Models (Planned)

```yaml
Option 1: ConvNeXtV2-Base + RT-DETR
  - Parameters: ~100M
  - Speed: 10-50x faster inference
  - Target: 90-95% of teacher quality

Option 2: ViT-PE-Spatial-Small + Deformable DETR
  - Parameters: ~80M
  - Maintains transformer architecture
  - Target: 92-97% of teacher quality
```

## Training Configuration

### Current Production Config

**Hardware:** Single NVIDIA Blackwell Pro 6000 (96GB VRAM)

**Key Settings:**
```yaml
Batch size: 64              # Empirically tested (~65-70GB peak VRAM)
Training epochs: 12         # Based on document detection research
Learning rate: 1e-4         # AdamW optimizer
Warmup steps: 1000
Mixed precision: bfloat16
Early stopping: 5 eval cycles (2500 steps)
```

**Total Training:**
- Steps: ~62,808 (335,703 samples ÷ 64 batch size × 12 epochs)
- Duration: ~TBD hours on Blackwell Pro 6000
- Checkpoints: Saved every 500 steps

See [configs/pretrain_publaynet.yaml](configs/pretrain_publaynet.yaml) for complete configuration.

## Project Structure

```
doc-obj-detect/
├── configs/                      # Training configurations
│   ├── pretrain_publaynet.yaml   # PubLayNet pretraining config
│   └── test_quick.yaml           # Quick test config (30 steps)
├── scripts/                      # Utility scripts
│   └── benchmark_vram.py         # VRAM benchmarking tool
├── src/doc_obj_detect/          # Main package
│   ├── config.py                 # Pydantic config schemas
│   ├── data.py                   # Dataset loading and preprocessing
│   ├── model.py                  # Model creation (ViT-PE + DETR)
│   ├── train.py                  # Training loop
│   ├── evaluate.py               # Evaluation metrics
│   └── visualize.py              # Data visualization
├── tests/                        # Test suite (pytest)
├── RESEARCH.md                   # Architecture research and rationale
├── CLAUDE.md                     # Development guidelines
└── pyproject.toml               # Dependencies and project config
```

## Datasets

### PubLayNet (Pretraining)

- **Size:** 335,703 training images, 11,245 validation images
- **Classes:** 5 (text, title, list, table, figure)
- **Source:** Automatically parsed from PubMed Central articles
- **License:** Community Data License Agreement – Permissive

### DocLayNet (Fine-tuning)

- **Size:** ~80,000 document images
- **Classes:** 11 (more fine-grained categories)
- **Source:** Real-world business documents
- **Diversity:** Multiple document types and layouts

Both datasets are loaded automatically via HuggingFace Datasets.

## Data Augmentation

Document-specific augmentations designed to preserve readability:

```yaml
horizontal_flip: 0.5          # 50% probability
rotate_limit: 5°              # Small rotations preserve text
brightness_contrast: 0.2      # Subtle lighting variations
noise_std: 0.01              # Minimal noise for scanning artifacts
```

**What we avoid:**
- Large rotations (breaks text readability)
- Heavy color jittering (documents are mostly grayscale)
- Aggressive crops (breaks layout structure)

See [src/doc_obj_detect/data.py](src/doc_obj_detect/data.py) for implementation.

## Development

### Environment

```bash
# Python 3.12 managed with uv
uv sync

# Code formatting and linting
ruff check .
ruff format .

# Run tests (target: 70% coverage of critical paths)
pytest
pytest tests/test_data.py::test_specific_function

# Pre-commit hooks (ruff, yaml, trailing whitespace, etc.)
pre-commit run --all-files
```

### Utilities

**VRAM Benchmarking:**
```bash
python scripts/benchmark_vram.py
```
Tests different model sizes and batch sizes to determine optimal configuration.

**TensorBoard Monitoring:**
```bash
tensorboard --logdir outputs/pretrain_publaynet/logs
```
Monitor loss curves, learning rate schedule, and evaluation metrics.

## Key Design Decisions

### Why ViT-PE Spatial?

The Perception Encoder (PE) research shows that **optimal features for object detection exist in intermediate layers**, not at the output. PE Spatial achieves 66.0 box mAP on COCO (SOTA) and is specifically designed for dense prediction tasks.

### Why Deformable DETR?

- Multi-scale feature extraction from intermediate ViT layers (no FPN needed)
- Efficient deformable attention for dense document layouts
- Native support for timm backbones in HuggingFace Transformers
- Proven architecture with strong baselines

### Why 12 Epochs?

Literature review shows document detection with pretrained backbones typically uses 6-12 epochs. The 1.8B parameter ViT-PE backbone is already pretrained on strong visual representations, so fewer epochs are needed compared to training from scratch (50+ epochs).

See [RESEARCH.md](RESEARCH.md) for complete rationale with references.

## Results

### PubLayNet Pretraining (In Progress)

| Metric | Target | Actual |
|--------|--------|--------|
| mAP@0.5:0.95 | 92-96 | TBD |
| mAP@0.5 | 95-98 | TBD |
| Training time | ~TBD hrs | TBD |
| Peak VRAM | ~70 GB | TBD |

Results will be updated after training completes.

## Knowledge Distillation (Planned)

**Strategy:** Train high-quality teacher model first, then distill to multiple student architectures.

**Distillation Loss:**
```python
total_loss = α * distillation_loss + β * ground_truth_loss

# Carefully tune α and β to balance:
# - distillation_loss: MSE or KL divergence from teacher
# - ground_truth_loss: Cross-entropy with dataset labels
```

**Student Candidates:**
1. ConvNeXtV2 + RT-DETR (target for deployment)
2. ViT-PE-Small + Deformable DETR (smaller ViT option)

## Hardware Requirements

**Training:**
- GPU: 96GB VRAM recommended (tested on NVIDIA Blackwell Pro 6000)
- CPU: 8+ cores for data loading
- RAM: 32GB+
- Storage: ~100GB for datasets and checkpoints

**Inference (after distillation):**
- Student models designed for deployment on GPUs with 8-16GB VRAM
- Target: Real-time inference on document images

## Citation

If you use this work, please cite the key papers:

```bibtex
@article{perception_encoder_2025,
  title={Perception Encoder: The best visual embeddings are not at the output of the network},
  author={Facebook Research},
  journal={arXiv preprint arXiv:2504.13181},
  year={2025}
}

@article{deformable_detr,
  title={Deformable DETR: Deformable Transformers for End-to-End Object Detection},
  author={Zhu, Xizhou and Su, Weijie and Lu, Lewei and Li, Bin and Wang, Xiaogang and Dai, Jifeng},
  journal={arXiv preprint arXiv:2010.04159},
  year={2020}
}

@inproceedings{publaynet,
  title={PubLayNet: largest dataset ever for document layout analysis},
  author={Zhong, Xu and Tang, Jianbin and Yepes, Antonio Jimeno},
  booktitle={2019 International Conference on Document Analysis and Recognition (ICDAR)},
  pages={1015--1022},
  year={2019},
  organization={IEEE}
}
```

## License

[Specify your license here]

## Acknowledgments

- **Facebook Research** for the Perception Encoder models and research
- **HuggingFace** for Transformers and Datasets libraries
- **IBM Research** for the PubLayNet dataset
- **DS4SD** for the DocLayNet dataset
