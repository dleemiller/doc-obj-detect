from types import SimpleNamespace

from doc_obj_detect.config import (
    DataConfig,
    DFineConfig,
    DistillationConfig,
    DistillConfig,
    ModelConfig,
    OutputConfig,
    TeacherConfig,
    TrainingConfig,
)


def build_distill_config(tmp_path):
    teacher = TeacherConfig(
        checkpoint=str(tmp_path / "teacher"), backbone="large", detector="dfine"
    )
    model_cfg = ModelConfig(backbone="base", architecture="dfine_small", num_classes=2)
    dfine_cfg = DFineConfig()
    distill_cfg = DistillationConfig(loss_type="mse", temperature=1.0, alpha=0.5, beta=0.5)
    data_cfg = DataConfig(
        dataset="publaynet",
        train_split="train",
        val_split="validation",
        image_size=128,
        batch_size=2,
        num_workers=0,
        cache_dir=None,
        max_eval_samples=2,
    )
    training_cfg = TrainingConfig(
        num_train_epochs=1,
        learning_rate=1e-4,
        weight_decay=0.0,
        warmup_ratio=0.0,
        gradient_accumulation_steps=1,
        bf16=False,
        fp16=False,
        save_steps=5,
        eval_steps=5,
        logging_steps=1,
        eval_strategy="steps",
        save_strategy="steps",
        save_total_limit=1,
        load_best_model_at_end=False,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        push_to_hub=False,
    )
    output_cfg = OutputConfig(
        output_dir=str(tmp_path / "out"), log_dir=str(tmp_path / "logs"), run_name="distill"
    )
    return DistillConfig(
        teacher=teacher,
        model=model_cfg,
        dfine=dfine_cfg,
        distillation=distill_cfg,
        data=data_cfg,
        augmentation=None,
        training=training_cfg,
        output=output_cfg,
    )


def test_distill_runner_wires_trainer(monkeypatch, tmp_path):
    from doc_obj_detect.training.distill_runner import DistillRunner

    class DummyModel:
        def __init__(self):
            self.model = SimpleNamespace(backbone=SimpleNamespace(parameters=lambda: []))

    class DummyArtifacts:
        def __init__(self):
            self.model = DummyModel()
            self.processor = SimpleNamespace(
                do_resize=False, do_pad=False, size=None, save_pretrained=lambda path: None
            )

    class DummyFactory:
        def __init__(self, *_, **__):
            pass

        @classmethod
        def from_config(cls, *_, **__):
            return cls()

        def build(self):
            return DummyArtifacts()

    class DummyDatasetFactory:
        def __init__(self, *_, **__):
            pass

        def build(self, split, **__):
            return [split], {0: "cls"}

    class DummyTrainer:
        init_kwargs = None
        train_called = False
        saved = None

        def __init__(self, *_, **kwargs):
            DummyTrainer.init_kwargs = kwargs

        def train(self):
            DummyTrainer.train_called = True

        def save_model(self, path):
            DummyTrainer.saved = path

    class DummyTeacher:
        def eval(self):
            return self

        def parameters(self):
            return [SimpleNamespace(device="cpu", requires_grad=False)]

        def to(self, *_):
            return self

    monkeypatch.setattr("doc_obj_detect.training.distill_runner.ModelFactory", DummyFactory)
    monkeypatch.setattr("doc_obj_detect.training.base_runner.DatasetFactory", DummyDatasetFactory)
    monkeypatch.setattr("doc_obj_detect.training.distill_runner.DistillationTrainer", DummyTrainer)
    monkeypatch.setattr(
        "doc_obj_detect.training.distill_runner.DFineForObjectDetection",
        SimpleNamespace(from_pretrained=lambda *_: DummyTeacher()),
    )

    config = build_distill_config(tmp_path)
    runner = DistillRunner(config)
    runner.run()

    assert DummyTrainer.train_called is True
    assert DummyTrainer.saved.endswith("final_model")
