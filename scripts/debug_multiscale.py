"""Debug script to verify multi-scale feature extraction from ConvNeXt-DINOv3 backbone."""

import torch

from doc_obj_detect.config import load_train_config
from doc_obj_detect.model import create_model


def debug_multiscale_features(config_path: str = "configs/pretrain_publaynet.yaml"):
    """Check if Deformable DETR extracts multi-scale features from ViT backbone."""

    print("=" * 80)
    print("Multi-Scale Feature Extraction Diagnostic")
    print("=" * 80)

    # Load config
    config = load_train_config(config_path)

    print("\nConfiguration:")
    print(f"  Backbone: {config.model.backbone}")
    print(f"  Num feature levels: {config.dfine.num_feature_levels}")
    print(f"  Image size: {config.data.image_size}")

    # Create model
    print("\nInitializing model...")
    model, image_processor = create_model(
        backbone=config.model.backbone,
        num_classes=config.model.num_classes,
        use_pretrained_backbone=config.model.use_pretrained_backbone,
        freeze_backbone=False,
        image_size=config.data.image_size,
        **config.dfine.model_dump(),
    )

    model.eval()

    # Print model config details
    print("\nModel Config:")
    print(f"  use_timm_backbone: {model.config.use_timm_backbone}")
    print(f"  backbone: {model.config.backbone}")
    print(f"  num_feature_levels: {model.config.num_feature_levels}")
    print(f"  backbone_config: {model.config.backbone_config}")

    # Inspect model structure
    print("\nModel Structure:")
    print(f"  Model type: {type(model.model)}")
    print(f"  Backbone type: {type(model.model.backbone)}")

    # Check if backbone has a timm model inside
    if hasattr(model.model.backbone, "model"):
        print(f"  Backbone.model type: {type(model.model.backbone.model)}")

        # Check if timm model has feature_info
        if hasattr(model.model.backbone.model, "feature_info"):
            feature_info = model.model.backbone.model.feature_info
            print("\n  Feature Info from timm model:")
            for i, info in enumerate(feature_info):
                print(f"    Level {i}: {info}")

    # Check encoder
    print("\nEncoder:")
    print(f"  Type: {type(model.model.encoder)}")
    if hasattr(model.model.encoder, "config"):
        print(f"  Num feature levels: {model.model.encoder.config.num_feature_levels}")

    # Create dummy input for full model
    batch_size = 2
    image_size = config.data.image_size
    pixel_values = torch.randn(batch_size, 3, image_size, image_size)
    pixel_mask = torch.ones(batch_size, image_size, image_size)

    # Create dummy labels
    labels = []
    for _ in range(batch_size):
        labels.append(
            {
                "class_labels": torch.tensor([0, 1]),
                "boxes": torch.tensor([[0.5, 0.5, 0.3, 0.3], [0.2, 0.2, 0.1, 0.1]]),
            }
        )

    print("\nRunning forward pass...")

    # Track intermediate outputs
    encoder_outputs = []

    def encoder_hook(module, input, output):
        """Capture encoder outputs."""
        encoder_outputs.append(output)
        if hasattr(output, "last_hidden_state"):
            print(f"  Encoder output shape: {output.last_hidden_state.shape}")
        elif isinstance(output, tuple):
            print(f"  Encoder output tuple length: {len(output)}")
            for i, o in enumerate(output):
                if isinstance(o, torch.Tensor):
                    print(f"    Output {i} shape: {o.shape}")

    # Register hook on encoder
    encoder_hook_handle = model.model.encoder.register_forward_hook(encoder_hook)

    # Also hook the backbone's forward to see what it produces
    backbone_outputs = []

    def backbone_hook(module, input, output):
        """Capture backbone outputs."""
        backbone_outputs.append(output)
        if isinstance(output, dict):
            print(f"  Backbone output (dict) keys: {output.keys()}")
            if "feature_maps" in output:
                fmaps = output["feature_maps"]
                print(f"  Feature maps: {len(fmaps)} levels")
                for i, fmap in enumerate(fmaps):
                    print(f"    Level {i}: {fmap.shape}")
        elif isinstance(output, list | tuple):
            print(f"  Backbone output ({type(output).__name__}): {len(output)} items")
            for i, item in enumerate(output):
                if isinstance(item, torch.Tensor):
                    print(f"    Item {i} (tensor) shape: {item.shape}")
                elif isinstance(item, list | tuple):
                    print(f"    Item {i} ({type(item).__name__}): {len(item)} sub-items")
                    for j, subitem in enumerate(item):
                        if isinstance(subitem, torch.Tensor):
                            print(f"      Sub-item {j} shape: {subitem.shape}")
                else:
                    print(f"    Item {i} type: {type(item)}")
        elif isinstance(output, torch.Tensor):
            print(f"  Backbone output (single tensor): {output.shape}")
        else:
            print(f"  Backbone output type: {type(output)}")

    backbone_hook_handle = model.model.backbone.register_forward_hook(backbone_hook)

    with torch.no_grad():
        try:
            outputs = model(
                pixel_values=pixel_values,
                pixel_mask=pixel_mask,
                labels=labels,
            )
            print("\n✅ Forward successful!")
            print(f"  Loss: {outputs.loss.item():.4f}")
            print(f"  Logits shape: {outputs.logits.shape}")
            print(f"  Pred boxes shape: {outputs.pred_boxes.shape}")
        except Exception as e:
            print(f"\n❌ Forward failed: {e}")
            import traceback

            traceback.print_exc()

    encoder_hook_handle.remove()
    backbone_hook_handle.remove()

    print(f"\nBackbone outputs captured: {len(backbone_outputs)}")
    print(f"Encoder outputs captured: {len(encoder_outputs)}")

    # Analyze what we got
    if len(backbone_outputs) > 0:
        output = backbone_outputs[0]
        print(f"\n{'='*80}")
        print("ANALYSIS:")
        print(f"{'='*80}")

        if isinstance(output, dict) and "feature_maps" in output:
            num_levels = len(output["feature_maps"])
            print(f"✅ Backbone produces {num_levels} feature levels")
            if num_levels >= 4:
                print("✅ Multi-scale feature extraction is WORKING!")
            elif num_levels == 1:
                print("❌ CRITICAL: Only single-scale features!")
                print("   This explains training plateau.")
            else:
                print(f"⚠️  Partial multi-scale: {num_levels} levels (expected 4)")
        elif isinstance(output, tuple) and len(output) == 2:
            # HuggingFace format: (last_hidden_state, hidden_states_tuple)
            last_hidden, hidden_states = output
            if isinstance(hidden_states, list | tuple):
                print("Backbone returns tuple: (last_hidden_state, hidden_states)")
                print(
                    f"  Last hidden state shape: {last_hidden.shape if isinstance(last_hidden, torch.Tensor) else 'N/A'}"
                )
                print(f"  Hidden states: {len(hidden_states)} layers")

                # Check if these are multi-scale features
                if len(hidden_states) >= 4:
                    print(f"\n✅ MULTI-SCALE DETECTED: {len(hidden_states)} intermediate features!")
                    print("   Shapes of intermediate features:")
                    for i, h in enumerate(hidden_states):
                        if isinstance(h, torch.Tensor):
                            print(f"     Layer {i}: {h.shape}")
                    print("\n✅ Multi-scale feature extraction IS WORKING!")
                elif len(hidden_states) == 1:
                    print("\n❌ CRITICAL: Only 1 hidden state - single-scale only!")
                else:
                    print(f"\n⚠️  Partial: {len(hidden_states)} hidden states")
            else:
                print(f"Second item in tuple is not list/tuple: {type(hidden_states)}")
        else:
            print("❌ Backbone output format unexpected")
            print(f"   Expected dict with 'feature_maps' or tuple, got: {type(output)}")

    print("\n" + "=" * 80)
    print("DEBUGGING RECOMMENDATIONS:")
    print("=" * 80)

    # Check the actual backbone code
    print("\nBackbone class hierarchy:")
    for cls in type(model.model.backbone).__mro__[:5]:
        print(f"  - {cls}")

    # Print forward method signature
    import inspect

    try:
        sig = inspect.signature(model.model.backbone.forward)
        print(f"\nBackbone.forward signature: {sig}")
    except Exception as e:
        print(f"  Could not inspect: {e}")


if __name__ == "__main__":
    debug_multiscale_features()
