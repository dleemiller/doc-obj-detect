from types import SimpleNamespace

import pytest

from doc_obj_detect.config import (
    DataConfig,
    DFineConfig,
    ModelConfig,
    OutputConfig,
    TrainConfig,
    TrainingConfig,
)
from doc_obj_detect.models import ModelArtifacts
from doc_obj_detect.training.runner import TrainerRunner


class DummyTrainer:
    init_kwargs = None
    train_called = False
    saved_path = None

    def __init__(self, *_, **kwargs):
        DummyTrainer.init_kwargs = kwargs

    def train(self):
        DummyTrainer.train_called = True

    def save_model(self, path):
        DummyTrainer.saved_path = path


class DummyProcessor:
    saved_path = None

    def __init__(self):
        self.do_resize = False
        self.do_pad = False
        self.size = None

    def save_pretrained(self, path):
        DummyProcessor.saved_path = path


class DummyDatasetFactory:
    def __init__(self, dataset_name, image_processor, **kwargs):
        self.dataset_name = dataset_name
        self.image_processor = image_processor
        self.kwargs = kwargs

    def build(self, split, **kwargs):
        dataset = [f"{self.dataset_name}-{split}"]
        labels = {0: "cls"}
        return dataset, labels


class DummyModelFactory:
    def __init__(self, *_, **__):
        pass

    @classmethod
    def from_config(cls, *_, **__):
        return cls()

    def build(self):
        dummy_backbone = SimpleNamespace(parameters=lambda: [])
        dummy_model = SimpleNamespace(model=SimpleNamespace(backbone=dummy_backbone))
        processor = DummyProcessor()
        return ModelArtifacts(model=dummy_model, processor=processor)


def build_train_config(tmp_path) -> TrainConfig:
    model_cfg = ModelConfig(
        backbone="dummy",
        num_classes=2,
        use_pretrained_backbone=False,
        freeze_backbone=False,
    )
    dfine_cfg = DFineConfig()
    data_cfg = DataConfig(
        dataset="publaynet",
        train_split="train",
        val_split="validation",
        image_size=128,
        batch_size=2,
        num_workers=0,
        cache_dir=None,
        max_eval_samples=5,
    )
    training_cfg = TrainingConfig(
        num_train_epochs=1,
        learning_rate=1e-4,
        weight_decay=0.0,
        warmup_ratio=0.0,
        gradient_accumulation_steps=1,
        bf16=False,
        fp16=False,
        save_steps=10,
        eval_steps=10,
        logging_steps=5,
        eval_strategy="steps",
        save_strategy="steps",
        save_total_limit=1,
        load_best_model_at_end=False,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        push_to_hub=False,
    )
    output_cfg = OutputConfig(
        output_dir=str(tmp_path / "out"),
        checkpoint_dir=None,
        log_dir=None,
        run_name="unit",
    )
    return TrainConfig(
        model=model_cfg,
        dfine=dfine_cfg,
        data=data_cfg,
        augmentation=None,
        training=training_cfg,
        output=output_cfg,
    )


@pytest.fixture(autouse=True)
def reset_dummy_trainer():
    DummyTrainer.init_kwargs = None
    DummyTrainer.train_called = False
    DummyTrainer.saved_path = None
    DummyProcessor.saved_path = None
    yield


def test_trainer_runner_wires_dependencies(monkeypatch, tmp_path):
    config = build_train_config(tmp_path)

    monkeypatch.setattr("doc_obj_detect.training.runner.ModelFactory", DummyModelFactory)
    monkeypatch.setattr("doc_obj_detect.training.runner.DatasetFactory", DummyDatasetFactory)
    monkeypatch.setattr("doc_obj_detect.training.runner.SplitLRTrainer", DummyTrainer)
    monkeypatch.setattr(
        "doc_obj_detect.training.runner.get_trainable_parameters",
        lambda *_, **__: {"total": 0, "trainable": 0, "frozen": 0, "trainable_percent": 0.0},
    )

    runner = TrainerRunner(config)
    runner.run()

    assert DummyTrainer.train_called is True
    assert DummyTrainer.init_kwargs is not None
    assert DummyTrainer.init_kwargs["train_dataset"] == ["publaynet-train"]
    assert DummyTrainer.init_kwargs["eval_dataset"] == ["publaynet-validation"]
    assert DummyProcessor.saved_path.endswith("final_model")
    assert DummyTrainer.saved_path.endswith("final_model")
