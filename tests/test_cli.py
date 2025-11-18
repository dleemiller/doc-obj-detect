import importlib

cli_main = importlib.import_module("doc_obj_detect.cli.main")


def test_cli_train_dispatch(monkeypatch):
    calls = {}

    class DummyRunner:
        @classmethod
        def from_config(cls, config_path):
            calls["config"] = config_path
            return cls()

        def run(self):
            calls["run"] = True

    monkeypatch.setattr(cli_main, "TrainerRunner", DummyRunner)
    cli_main.main(["train", "--config", "foo.yaml"])

    assert calls["config"] == "foo.yaml"
    assert calls["run"] is True


def test_cli_evaluate_dispatch(monkeypatch):
    calls = {}

    class DummyEvalRunner:
        @classmethod
        def from_config(cls, config_path):
            calls["config"] = config_path
            return cls()

        def run(self, **kwargs):
            calls["kwargs"] = kwargs

    monkeypatch.setattr(cli_main, "EvaluatorRunner", DummyEvalRunner)
    cli_main.main(
        [
            "evaluate",
            "--checkpoint",
            "ckpt",
            "--config",
            "cfg.yaml",
            "--batch-size",
            "2",
            "--num-workers",
            "1",
        ]
    )

    assert calls["config"] == "cfg.yaml"
    assert calls["kwargs"]["checkpoint_path"] == "ckpt"
    assert calls["kwargs"]["batch_size"] == 2
    assert calls["kwargs"]["num_workers"] == 1


def test_cli_distill_dispatch(monkeypatch):
    calls = {}

    class DummyDistillRunner:
        @classmethod
        def from_config(cls, config_path):
            calls["config"] = config_path
            return cls()

        def run(self):
            calls["run"] = True

    monkeypatch.setattr(cli_main, "DistillRunner", DummyDistillRunner)
    cli_main.main(["distill", "--config", "distill.yaml"])

    assert calls["config"] == "distill.yaml"
    assert calls["run"] is True
