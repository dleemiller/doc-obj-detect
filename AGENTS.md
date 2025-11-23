# Repository Guidelines

## Project Structure & Module Organization
`src/doc_obj_detect` houses the package with modular organization: `cli/` (unified command interface), `config/` (Pydantic schemas and loaders), `data/` (dataset wrappers and augmentation), `models/` (model factory and utilities), `training/` (runners for train/eval/distill), and `utils/` (logging and path helpers). Experiments live in `configs/` (e.g., `pretrain_publaynet.yaml`). Tests reside in `tests/`, and generated checkpoints, logs, and TensorBoard runs should stay under `outputs/` to keep the tree reproducible.

## Build, Test & Development Commands
Install everything through uv: `uv sync` (optionally `uv sync --extra cu128` when targeting CUDA 12.8 wheels). Core workflows run via the unified CLI: `uv run doc-obj-detect train --config configs/pretrain_publaynet.yaml`, `uv run doc-obj-detect evaluate --checkpoint <path> --config <config>`, `uv run doc-obj-detect distill --config configs/distill.yaml`, and `uv run doc-obj-detect visualize --dataset publaynet`. Use `tensorboard --logdir outputs/pretrain_publaynet_dfine/logs` for monitoring. Testing: `uv run pytest` or `uv run pytest tests/test_data.py::test_loader_shapes`.

## Coding Style & Naming Conventions
Python 3.12 is required, with Ruff enforcing linting (`ruff check .`) and formatting (`ruff format .`) at 100-character lines, double quotes, and space indentation. Keep modules snake_case, classes in PascalCase, and functions/variables in snake_case. Configuration filenames follow `verb_dataset.yaml` to distinguish pipelines. Prefer type hints everywhere and fail fast via `pydantic.ValidationError`.

## Testing Guidelines
Pytest drives validation (`pytest` or `pytest tests/test_data.py::test_loader_shapes`). The default `pytest.ini` targets files named `test_*.py`, classes `Test*`, and functions `test_*`. Maintain ≥70% coverage on critical paths, regenerate reports with `pytest --cov=src --cov-report=html` (output in `htmlcov/`). When adding datasets or configs, provide synthetic fixtures under `tests/fixtures/` to avoid large downloads.

## Commit & Pull Request Guidelines
Follow the existing short, imperative commit style (`Add multi-scale training`, `Update config`). Keep one logical change per commit, reference issue IDs when available, and include before/after metrics for training tweaks. Pull requests should summarize intent, list the configuration touched, note any new artifacts under `outputs/`, and attach TensorBoard screenshots or mAP tables when model quality changes. Always mention how to reproduce (exact `uv run ...` command) and confirm lint + tests in the description.

## Environment Notes
- Always invoke tooling via `uv run …`; the project dependencies live inside the uv-managed virtual environment, so bare `python` won’t import installed packages.
