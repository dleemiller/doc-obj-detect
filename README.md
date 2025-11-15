We will follow this: https://huggingface.co/docs/transformers/main/en/tasks/training_vision_backbone

Backbone: perception encoder https://huggingface.co/timm/vit_pe_spatial_base_patch16_512.fb
Detection - something good for object detection that is in huggingface transformers (deformable detr? detr? best possible for application)

Datasets - pretrain on PubLayNet
Fine tune on DocLayNet

Distillation to small size models by composing new backbone/detection:
- eg convnextv2 and rt-detr

Requirements:
use transformers trainers, and models
use appropriate augumentations with albumentations (think about most appropriate for doc obj detection)
cuda 12.8 - training hardware is single blackwell pro 6000 96GB
uv/ruff/pre-commit/pytest dev environment
DRY principles
aim for 70% test coverage of most critical parts
use project scripts to do processing and training - eg uv run preprocess-data publaynet
aim for SOTA with big/slow perception encoder based model, distill to smaller
carefully choose between MSE/KL div for distillation loss and balance with ground truth label losses (eg cross entropy)
use tensorboard for monitoring training
use huggingface datasets for data
