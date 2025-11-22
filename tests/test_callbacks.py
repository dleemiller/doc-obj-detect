"""Tests for custom trainer callbacks."""

import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import torch.nn as nn

from doc_obj_detect.training.callbacks import EMACallback, ModelEMA, UnfreezeBackboneCallback

# =============================================================================
# Fixtures and Helpers
# =============================================================================


class SimpleModel(nn.Module):
    """Simple model for testing."""

    def __init__(self, input_dim=10, hidden_dim=20, output_dim=2):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)


class ModelWithBackbone(nn.Module):
    """Model with nested backbone structure for testing UnfreezeBackboneCallback."""

    def __init__(self):
        super().__init__()
        self.model = SimpleNamespace(
            backbone=nn.Sequential(
                nn.Linear(10, 20),
                nn.ReLU(),
            )
        )
        self.head = nn.Linear(20, 2)

    def forward(self, x):
        x = self.model.backbone(x)
        return self.head(x)


@pytest.fixture
def simple_model():
    """Fixture for simple model."""
    return SimpleModel()


@pytest.fixture
def model_with_backbone():
    """Fixture for model with backbone."""
    return ModelWithBackbone()


def create_mock_trainer_state(global_step=0, epoch=0, max_steps=1000):
    """Create mock trainer state."""
    return SimpleNamespace(
        global_step=global_step,
        epoch=epoch,
        max_steps=max_steps,
    )


def create_mock_training_args(
    output_dir="/tmp/test",
    per_device_train_batch_size=8,
    gradient_accumulation_steps=1,
    num_train_epochs=3,
):
    """Create mock training arguments."""
    return SimpleNamespace(
        output_dir=output_dir,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        num_train_epochs=num_train_epochs,
    )


# =============================================================================
# ModelEMA Tests
# =============================================================================


def test_model_ema_initialization(simple_model):
    """Test ModelEMA initialization."""
    ema = ModelEMA(simple_model, decay=0.999, warmup_steps=100)

    assert ema.decay == 0.999
    assert ema.warmup_steps == 100
    assert ema.updates == 0

    # Check that EMA model is a separate copy
    assert ema.module is not simple_model
    assert isinstance(ema.module, SimpleModel)

    # Check that EMA parameters don't require gradients
    for param in ema.module.parameters():
        assert not param.requires_grad


def test_model_ema_update(simple_model):
    """Test EMA weight updates."""
    ema = ModelEMA(simple_model, decay=0.9, warmup_steps=0)

    # Get initial weights
    original_weight = simple_model.fc1.weight.data.clone()
    ema_weight_initial = ema.module.fc1.weight.data.clone()

    # Verify initial weights are the same
    assert torch.allclose(original_weight, ema_weight_initial)

    # Modify model weights
    with torch.no_grad():
        simple_model.fc1.weight.data += 1.0

    # Update EMA
    ema.update(simple_model)

    # EMA weight should be between original and new weight
    # Formula: ema = 0.9 * ema + 0.1 * new
    ema_weight_after = ema.module.fc1.weight.data
    expected_weight = 0.9 * original_weight + 0.1 * simple_model.fc1.weight.data

    assert torch.allclose(ema_weight_after, expected_weight, atol=1e-6)
    assert ema.updates == 1


def test_model_ema_warmup():
    """Test EMA warmup schedule."""
    import math

    model = SimpleModel()
    ema = ModelEMA(model, decay=0.999, warmup_steps=100)

    # At step 0, effective decay should be ~0
    assert ema.updates == 0
    decay_0 = ema._get_decay()
    assert decay_0 < 0.01  # Very small

    # Simulate 50 updates
    for _ in range(50):
        ema.update(model)

    # At step 50, decay should be intermediate
    # Formula: 0.999 * (1.0 - exp(-0.5)) ≈ 0.999 * 0.393 ≈ 0.393
    decay_50 = ema._get_decay()
    expected_50 = 0.999 * (1.0 - math.exp(-0.5))
    assert abs(decay_50 - expected_50) < 0.001

    # Simulate 50 more updates (total 100)
    for _ in range(50):
        ema.update(model)

    # At step 100, decay should be at ~63.2% of target
    # Formula: 0.999 * (1.0 - exp(-1)) ≈ 0.999 * 0.632 ≈ 0.631
    decay_100 = ema._get_decay()
    expected_100 = 0.999 * (1.0 - math.exp(-1))
    assert abs(decay_100 - expected_100) < 0.001

    # After many more updates, should asymptotically approach target decay
    # At 5x warmup steps, should be very close
    for _ in range(400):
        ema.update(model)
    decay_final = ema._get_decay()
    # Formula: 0.999 * (1.0 - exp(-5)) ≈ 0.999 * 0.993 ≈ 0.992
    expected_final = 0.999 * (1.0 - math.exp(-5))
    assert abs(decay_final - expected_final) < 0.001
    # Should be at least 99% of target decay
    assert decay_final > 0.99


def test_model_ema_state_dict(simple_model):
    """Test EMA state dict save/load."""
    ema = ModelEMA(simple_model, decay=0.999, warmup_steps=100)

    # Perform some updates
    for _ in range(10):
        ema.update(simple_model)

    # Save state
    state = ema.state_dict()
    assert "module" in state
    assert "updates" in state
    assert "decay" in state
    assert "warmup_steps" in state
    assert state["updates"] == 10

    # Create new EMA and load state
    ema2 = ModelEMA(simple_model, decay=0.5, warmup_steps=50)
    ema2.load_state_dict(state)

    # Check restored values
    assert ema2.updates == 10
    assert ema2.decay == 0.999
    assert ema2.warmup_steps == 100

    # Check weights match
    for p1, p2 in zip(ema.module.parameters(), ema2.module.parameters(), strict=False):
        assert torch.allclose(p1, p2)


def test_model_ema_de_parallel(simple_model):
    """Test de-parallel model unwrapping."""
    # Create wrapped model (simulating DDP)
    wrapped_model = SimpleNamespace(module=simple_model)

    ema = ModelEMA(wrapped_model, decay=0.999)

    # Should have unwrapped the model
    assert isinstance(ema.module, SimpleModel)


def test_model_ema_device_placement():
    """Test EMA model device placement."""
    model = SimpleModel()

    # Test CPU placement
    ema_cpu = ModelEMA(model, device="cpu")
    assert next(ema_cpu.module.parameters()).device.type == "cpu"

    # Test CUDA placement (if available)
    if torch.cuda.is_available():
        ema_cuda = ModelEMA(model, device="cuda")
        assert next(ema_cuda.module.parameters()).device.type == "cuda"


# =============================================================================
# EMACallback Tests
# =============================================================================


def test_ema_callback_initialization():
    """Test EMACallback initialization."""
    callback = EMACallback(decay=0.9999, warmup_steps=1000, use_ema_for_eval=True)

    assert callback.decay == 0.9999
    assert callback.warmup_steps == 1000
    assert callback.use_ema_for_eval is True
    assert callback.ema is None
    assert callback._original_model is None
    assert callback._is_swapped is False


def test_ema_callback_train_begin(simple_model):
    """Test EMA callback on_train_begin."""
    callback = EMACallback(decay=0.999, warmup_steps=100)
    args = create_mock_training_args()
    state = create_mock_trainer_state()
    control = SimpleNamespace()

    # Call on_train_begin
    result = callback.on_train_begin(args, state, control, model=simple_model)

    # Check EMA was initialized
    assert callback.ema is not None
    assert isinstance(callback.ema, ModelEMA)
    assert callback.ema.decay == 0.999
    assert callback.ema.warmup_steps == 100
    assert result is control


def test_ema_callback_train_begin_no_model():
    """Test EMA callback on_train_begin without model."""
    callback = EMACallback()
    args = create_mock_training_args()
    state = create_mock_trainer_state()
    control = SimpleNamespace()

    # Call without model - should handle gracefully
    with patch("doc_obj_detect.training.callbacks.logger") as mock_logger:
        callback.on_train_begin(args, state, control, model=None)
        mock_logger.warning.assert_called_once()

    assert callback.ema is None


def test_ema_callback_step_end(simple_model):
    """Test EMA callback on_step_end."""
    callback = EMACallback(decay=0.999, warmup_steps=100)
    args = create_mock_training_args()
    state = create_mock_trainer_state()
    control = SimpleNamespace()

    # Initialize EMA
    callback.on_train_begin(args, state, control, model=simple_model)

    # Get initial update count
    initial_updates = callback.ema.updates

    # Call on_step_end
    callback.on_step_end(args, state, control, model=simple_model)

    # Check EMA was updated
    assert callback.ema.updates == initial_updates + 1


def test_ema_callback_evaluate_swap(simple_model):
    """Test EMA callback swaps weights during evaluation."""
    callback = EMACallback(decay=0.999, warmup_steps=0, use_ema_for_eval=True)
    args = create_mock_training_args()
    state = create_mock_trainer_state()
    control = SimpleNamespace()

    # Initialize EMA
    callback.on_train_begin(args, state, control, model=simple_model)

    # Modify model weights so EMA and model differ
    for _ in range(10):
        with torch.no_grad():
            simple_model.fc1.weight.data += 0.1
        callback.ema.update(simple_model)

    # Create mock trainer
    trainer = SimpleNamespace(model=simple_model)

    # Call on_evaluate - should swap to EMA
    callback.on_evaluate(args, state, control, model=simple_model, trainer=trainer)

    # Check swap occurred
    assert callback._is_swapped is True
    assert callback._original_model is simple_model
    assert trainer.model is callback.ema.module
    assert trainer.model is not simple_model


def test_ema_callback_evaluate_no_swap_when_disabled(simple_model):
    """Test EMA callback doesn't swap when use_ema_for_eval=False."""
    callback = EMACallback(decay=0.999, warmup_steps=0, use_ema_for_eval=False)
    args = create_mock_training_args()
    state = create_mock_trainer_state()
    control = SimpleNamespace()

    # Initialize EMA
    callback.on_train_begin(args, state, control, model=simple_model)

    # Create mock trainer
    trainer = SimpleNamespace(model=simple_model)

    # Call on_evaluate - should NOT swap
    callback.on_evaluate(args, state, control, model=simple_model, trainer=trainer)

    # Check no swap occurred
    assert callback._is_swapped is False
    assert trainer.model is simple_model


def test_ema_callback_prediction_step_restore(simple_model):
    """Test EMA callback restores weights after evaluation."""
    callback = EMACallback(decay=0.999, warmup_steps=0, use_ema_for_eval=True)
    args = create_mock_training_args()
    state = create_mock_trainer_state()
    control = SimpleNamespace()

    # Initialize and swap
    callback.on_train_begin(args, state, control, model=simple_model)
    trainer = SimpleNamespace(model=simple_model)
    callback.on_evaluate(args, state, control, model=simple_model, trainer=trainer)

    assert callback._is_swapped is True

    # Call on_prediction_step - should restore
    callback.on_prediction_step(args, state, control, trainer=trainer)

    # Check restoration
    assert callback._is_swapped is False
    assert trainer.model is simple_model
    assert callback._original_model is None


def test_ema_callback_save_checkpoint(simple_model):
    """Test EMA callback saves checkpoint."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)
        args = create_mock_training_args(output_dir=str(output_dir))
        state = create_mock_trainer_state(global_step=100)
        control = SimpleNamespace()

        callback = EMACallback(decay=0.999, warmup_steps=0)
        callback.on_train_begin(args, state, control, model=simple_model)

        # Perform some updates
        for _ in range(10):
            callback.ema.update(simple_model)

        # Create checkpoint directory
        checkpoint_dir = output_dir / "checkpoint-100"
        checkpoint_dir.mkdir(parents=True)

        # Call on_save
        callback.on_save(args, state, control)

        # Check EMA state was saved
        ema_path = checkpoint_dir / "ema_state.pt"
        assert ema_path.exists()

        # Load and verify
        ema_state = torch.load(ema_path)
        assert "module" in ema_state
        assert ema_state["updates"] == 10


def test_ema_callback_train_end(simple_model):
    """Test EMA callback saves final state at train end."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)
        args = create_mock_training_args(output_dir=str(output_dir))
        state = create_mock_trainer_state()
        control = SimpleNamespace()

        callback = EMACallback(decay=0.999, warmup_steps=0)
        callback.on_train_begin(args, state, control, model=simple_model)

        # Perform some updates
        for _ in range(5):
            callback.ema.update(simple_model)

        # Call on_train_end
        callback.on_train_end(args, state, control)

        # Check final EMA state was saved
        ema_path = output_dir / "ema_state.pt"
        assert ema_path.exists()

        # Load and verify
        ema_state = torch.load(ema_path)
        assert ema_state["updates"] == 5


# =============================================================================
# UnfreezeBackboneCallback Tests
# =============================================================================


def test_unfreeze_backbone_callback_init_validation():
    """Test UnfreezeBackboneCallback initialization validation."""
    # Should raise if neither step nor epoch specified
    with pytest.raises(ValueError, match="Must specify either"):
        UnfreezeBackboneCallback()


def test_unfreeze_backbone_callback_unfreeze_at_step(model_with_backbone):
    """Test unfreezing at specific step."""
    callback = UnfreezeBackboneCallback(unfreeze_at_step=50)
    args = create_mock_training_args()
    state = create_mock_trainer_state(global_step=0)
    control = SimpleNamespace()

    # Freeze backbone initially
    for param in model_with_backbone.model.backbone.parameters():
        param.requires_grad = False

    # On train begin
    callback.on_train_begin(args, state, control)
    assert callback._target_step == 50

    # Before target step - should remain frozen
    state.global_step = 49
    callback.on_step_end(args, state, control, model=model_with_backbone)
    assert not all(p.requires_grad for p in model_with_backbone.model.backbone.parameters())

    # At target step - should unfreeze
    state.global_step = 50
    callback.on_step_end(args, state, control, model=model_with_backbone)
    assert all(p.requires_grad for p in model_with_backbone.model.backbone.parameters())

    # After unfreezing, should not try to unfreeze again
    assert callback._done is True


def test_unfreeze_backbone_callback_unfreeze_at_epoch(model_with_backbone):
    """Test unfreezing at specific epoch."""
    callback = UnfreezeBackboneCallback(unfreeze_at_epoch=2)
    args = create_mock_training_args(
        per_device_train_batch_size=8,
        gradient_accumulation_steps=1,
        num_train_epochs=5,
    )
    # With max_steps=1000 and 5 epochs, steps_per_epoch = 200
    state = create_mock_trainer_state(global_step=0, max_steps=1000)
    control = SimpleNamespace()

    # Freeze backbone initially
    for param in model_with_backbone.model.backbone.parameters():
        param.requires_grad = False

    # On train begin - should calculate target step
    callback.on_train_begin(args, state, control)

    # Target should be ~400 (2 epochs * 200 steps/epoch)
    assert callback._target_step == 400

    # Before target - should remain frozen
    state.global_step = 399
    callback.on_step_end(args, state, control, model=model_with_backbone)
    assert not all(p.requires_grad for p in model_with_backbone.model.backbone.parameters())

    # At target - should unfreeze
    state.global_step = 400
    callback.on_step_end(args, state, control, model=model_with_backbone)
    assert all(p.requires_grad for p in model_with_backbone.model.backbone.parameters())


def test_unfreeze_backbone_callback_handles_missing_backbone(simple_model):
    """Test callback handles model without backbone gracefully."""
    callback = UnfreezeBackboneCallback(unfreeze_at_step=10)
    args = create_mock_training_args()
    state = create_mock_trainer_state(global_step=10)
    control = SimpleNamespace()

    callback.on_train_begin(args, state, control)

    # Should not crash when model has no backbone
    result = callback.on_step_end(args, state, control, model=simple_model)
    assert result is control


def test_unfreeze_backbone_callback_handles_missing_model():
    """Test callback handles missing model gracefully."""
    callback = UnfreezeBackboneCallback(unfreeze_at_step=10)
    args = create_mock_training_args()
    state = create_mock_trainer_state(global_step=10)
    control = SimpleNamespace()

    callback.on_train_begin(args, state, control)

    # Should not crash when model is None
    result = callback.on_step_end(args, state, control, model=None)
    assert result is control


# =============================================================================
# Integration Tests
# =============================================================================


def test_ema_callback_full_training_lifecycle(simple_model):
    """Test complete EMA callback lifecycle through training."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)
        args = create_mock_training_args(output_dir=str(output_dir))
        control = SimpleNamespace()

        callback = EMACallback(decay=0.999, warmup_steps=10, use_ema_for_eval=True)

        # 1. Train begin
        state = create_mock_trainer_state(global_step=0)
        callback.on_train_begin(args, state, control, model=simple_model)
        assert callback.ema is not None

        # 2. Training steps
        for step in range(20):
            state.global_step = step
            with torch.no_grad():
                simple_model.fc1.weight.data += 0.01
            callback.on_step_end(args, state, control, model=simple_model)

        assert callback.ema.updates == 20

        # 3. Evaluation
        trainer = SimpleNamespace(model=simple_model)
        callback.on_evaluate(args, state, control, model=simple_model, trainer=trainer)
        assert trainer.model is callback.ema.module

        # 4. Restore after eval
        callback.on_prediction_step(args, state, control, trainer=trainer)
        assert trainer.model is simple_model

        # 5. Save checkpoint
        state.global_step = 100
        checkpoint_dir = output_dir / "checkpoint-100"
        checkpoint_dir.mkdir(parents=True)
        callback.on_save(args, state, control)
        assert (checkpoint_dir / "ema_state.pt").exists()

        # 6. Train end
        callback.on_train_end(args, state, control)
        assert (output_dir / "ema_state.pt").exists()
