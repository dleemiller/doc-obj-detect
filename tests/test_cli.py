import importlib

import numpy as np

cli_main = importlib.import_module("doc_obj_detect.cli.main")


def test_cli_train_dispatch(monkeypatch):
    calls = {}

    class DummyRunner:
        @classmethod
        def from_config(cls, config_path):
            calls["config"] = config_path
            return cls()

        def run(self, resume_from_checkpoint=None):
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


def test_cli_bbox_hist_uses_data_in(monkeypatch):
    calls = {}

    def fake_load(path):
        calls["data_in"] = path
        return np.array([1.0]), np.array([2.0]), [8]

    def fake_stride(strides):
        return {8: (16, 64)}

    def fake_plot(**kwargs):
        calls["plot"] = kwargs
        return kwargs["output_path"]

    monkeypatch.setattr(cli_main, "load_bbox_data", fake_load)
    monkeypatch.setattr(cli_main, "compute_stride_ranges", fake_stride)
    monkeypatch.setattr(cli_main, "plot_bbox_histograms", fake_plot)

    cli_main.main(["bbox-hist", "--data-in", "stats.npz", "--output", "hist.png"])

    assert calls["data_in"] == "stats.npz"
    assert calls["plot"]["output_path"] == "hist.png"


def test_cli_bbox_hist_collects_and_saves(monkeypatch):
    calls = {}

    def fake_collect(config_path, split, max_samples, target_short_side):
        calls["collect"] = {
            "config": config_path,
            "split": split,
            "short": target_short_side,
            "max": max_samples,
        }
        return np.array([1.0, 2.0]), np.array([3.0]), [4, 8]

    def fake_stride(strides):
        return {stride: (stride * 2, stride * 4) for stride in strides}

    def fake_save(path, widths, heights, strides):
        calls["saved"] = path
        return path

    def fake_plot(**kwargs):
        calls["plot"] = kwargs
        return kwargs["output_path"]

    monkeypatch.setattr(cli_main, "collect_bbox_samples", fake_collect)
    monkeypatch.setattr(cli_main, "compute_stride_ranges", fake_stride)
    monkeypatch.setattr(cli_main, "save_bbox_data", fake_save)
    monkeypatch.setattr(cli_main, "plot_bbox_histograms", fake_plot)

    cli_main.main(
        [
            "bbox-hist",
            "--config",
            "cfg.yaml",
            "--split",
            "validation",
            "--short-side",
            "700",
            "--data-out",
            "dump.npz",
            "--output",
            "hist.png",
        ]
    )

    assert calls["collect"]["config"] == "cfg.yaml"
    assert calls["collect"]["split"] == "validation"
    assert calls["collect"]["short"] == 700
    assert calls["saved"] == "dump.npz"
    assert calls["plot"]["output_path"] == "hist.png"
