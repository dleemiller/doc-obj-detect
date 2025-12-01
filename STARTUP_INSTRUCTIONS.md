# Python Project Startup Instructions for LLM Coding Assistants

This document provides comprehensive instructions for setting up a new Python project following modern best practices. Use this template when creating any new Python project.

## Project Setup Overview

This template uses:
- **Python 3.12+** for modern language features
- **uv** for fast, reliable dependency management
- **ruff** for linting and formatting
- **pytest** for testing with coverage reporting
- **pre-commit** for automated code quality checks

## Step 1: Initialize Project Structure

Create the following directory structure:

```
project-name/
├── .gitignore
├── .pre-commit-config.yaml
├── .python-version
├── pyproject.toml
├── pytest.ini
├── ruff.toml
├── README.md
├── CLAUDE.md
├── LICENSE
├── src/
│   └── project_name/
│       ├── __init__.py
│       └── cli/
│           ├── __init__.py
│           └── main.py
├── tests/
│   ├── __init__.py
│   └── test_basic.py
└── configs/
    └── .gitkeep
```

**Note:** Add additional modules under `src/project_name/` based on your project needs.

## Step 2: Core Configuration Files

### .python-version
```
3.12
```

### pyproject.toml
```toml
[project]
name = "project-name"
version = "0.1.0"
description = "Brief description of your project"
readme = "README.md"
requires-python = ">=3.12"
dependencies = [
    # Add your core dependencies here
]

[project.optional-dependencies]
dev = [
    "pytest>=8.3.0",
    "pytest-cov>=6.0.0",
    "ruff>=0.8.0",
    "pre-commit>=4.0.0",
]

[project.scripts]
project-name = "project_name.cli.main:main"

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/project_name"]

[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = ["test_*.py"]
python_classes = ["Test*"]
python_functions = ["test_*"]
addopts = "--cov=src --cov-report=term-missing --cov-report=html"

[tool.coverage.run]
source = ["src/project_name"]
omit = ["*/tests/*", "*/__pycache__/*"]

[tool.coverage.report]
exclude_lines = [
    "pragma: no cover",
    "def __repr__",
    "raise AssertionError",
    "raise NotImplementedError",
    "if __name__ == .__main__.:",
    "if TYPE_CHECKING:",
]
```

### pytest.ini
```ini
[pytest]
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
addopts = -v
```

### ruff.toml
```toml
# Ruff configuration
target-version = "py312"
line-length = 100

[lint]
select = [
    "E",   # pycodestyle errors
    "W",   # pycodestyle warnings
    "F",   # pyflakes
    "I",   # isort
    "B",   # flake8-bugbear
    "C4",  # flake8-comprehensions
    "UP",  # pyupgrade
]
ignore = [
    "E501",  # line too long (handled by formatter)
    "B008",  # do not perform function calls in argument defaults
    "C901",  # too complex
]

[lint.per-file-ignores]
"__init__.py" = ["F401"]  # unused imports

[format]
quote-style = "double"
indent-style = "space"
skip-magic-trailing-comma = false
line-ending = "auto"
```

### .pre-commit-config.yaml
```yaml
repos:
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.8.4
    hooks:
      - id: ruff
        args: [--fix]
      - id: ruff-format

  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v5.0.0
    hooks:
      - id: trailing-whitespace
      - id: end-of-file-fixer
      - id: check-yaml
      - id: check-added-large-files
        args: ['--maxkb=1000']
      - id: check-merge-conflict
      - id: check-toml
```

### .gitignore
```gitignore
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Virtual environments
venv/
ENV/
env/
.venv

# IDEs
.vscode/
.idea/
*.swp
*.swo
*~

# Testing
.pytest_cache/
.coverage
htmlcov/
.tox/

# Project-specific (adjust as needed)
*.log
.DS_Store
```

## Step 3: Documentation Files

### README.md Template
```markdown
# Project Name

Brief description of what this project does.

## Features

- Feature 1
- Feature 2
- Feature 3

## Quick Start

\`\`\`bash
# Clone and setup
git clone <repo-url>
cd project-name
uv sync

# Run tests
uv run pytest

# Run the CLI
uv run project-name --help
\`\`\`

## Installation

\`\`\`bash
# Install dependencies
uv sync

# For development (includes testing tools)
uv sync --extra dev
\`\`\`

## Usage

\`\`\`bash
# Basic usage
uv run project-name <command>

# With options
uv run project-name <command> --option value
\`\`\`

## Development

\`\`\`bash
# Run linting
ruff check .

# Auto-fix linting issues
ruff check . --fix

# Format code
ruff format .

# Run tests
uv run pytest

# Run tests with coverage
uv run pytest --cov=src --cov-report=term-missing

# Generate HTML coverage report
uv run pytest --cov=src --cov-report=html
# Open htmlcov/index.html in browser

# Install pre-commit hooks
uv run pre-commit install

# Run pre-commit on all files
uv run pre-commit run --all-files
\`\`\`

## Testing

Run tests with:

\`\`\`bash
uv run pytest                              # All tests
uv run pytest tests/test_file.py           # Single file
uv run pytest tests/test_file.py::test_fn  # Single test
uv run pytest -v                           # Verbose output
\`\`\`

## License

[Specify license here]
```

### CLAUDE.md Template
```markdown
# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

[Provide a clear, concise overview of what this project does and its key architecture]

**Project Purpose:**
[Describe what problem this solves]

**Key Components:**
[List and describe the main components/modules of your project]

## Development Environment

**Python:** 3.12 (managed with uv)
**Package Manager:** uv
**Linting/Formatting:** ruff
**Testing:** pytest (target 70% coverage)
**Pre-commit:** Used for code quality checks

## Common Commands

**Development:**
\`\`\`bash
# Install dependencies
uv sync

# Run linting/formatting
ruff check .
ruff format .

# Run tests
uv run pytest
uv run pytest tests/test_file.py              # Single file
uv run pytest tests/test_file.py::test_fn     # Single test

# Pre-commit hooks
uv run pre-commit install
uv run pre-commit run --all-files
\`\`\`

**Project Commands:**
\`\`\`bash
uv run project-name --help
# Add your specific commands here
\`\`\`

## Architecture Guidelines

**Code Quality:**
- Follow DRY principles rigorously
- Target 70% test coverage for critical components
- Use type hints throughout
- Maintain clean separation between modules
- Keep functions focused and small

**Dependencies:**
[Explain key dependencies and why they're used]

## Project Structure

[Describe your module organization and what each component does]

## Testing Strategy

- Target 70%+ coverage on critical paths
- Use pytest fixtures for reusable test data
- Mock external dependencies (APIs, databases, etc.)
- Test both happy paths and error cases
- Keep tests fast and independent

## Common Pitfalls to Avoid

- Always use `uv run` for executing Python commands (dependencies are in uv-managed venv)
- Don't commit large files or sensitive data
- Don't skip pre-commit hooks
- Follow existing code style and patterns
- Update tests when changing functionality
```

## Step 4: Initial Code Files

### src/project_name/__init__.py
```python
"""Project Name - Brief description."""

__version__ = "0.1.0"
```

### src/project_name/cli/__init__.py
```python
"""CLI module."""
```

### src/project_name/cli/main.py
```python
"""CLI entry point."""

import sys


def main() -> int:
    """Main CLI entry point.

    Returns:
        Exit code (0 for success, non-zero for error)
    """
    print("Hello from project-name!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

### tests/__init__.py
```python
"""Test suite."""
```

### tests/test_basic.py
```python
"""Basic tests to verify setup."""


def test_imports():
    """Test that package can be imported."""
    import project_name

    assert project_name.__version__ == "0.1.0"
```

## Step 5: Initialize Project

Run these commands to initialize:

```bash
# Create project directory
mkdir project-name
cd project-name

# Initialize git
git init

# Create all the configuration files above

# Create directory structure
mkdir -p src/project_name/cli
mkdir -p tests
mkdir -p configs

# Create __init__.py files
touch src/project_name/__init__.py
touch src/project_name/cli/__init__.py
touch tests/__init__.py

# Initialize uv and install dependencies
uv sync
uv sync --extra dev

# Install pre-commit hooks
uv run pre-commit install

# Run initial tests
uv run pytest

# Check code quality
uv run ruff check .
uv run ruff format .

# Run pre-commit to verify everything works
uv run pre-commit run --all-files
```

## Step 6: Verification Checklist

Ensure everything is set up correctly:

- [ ] `uv sync` completes successfully
- [ ] `uv sync --extra dev` completes successfully
- [ ] `uv run pytest` passes all tests
- [ ] `uv run ruff check .` shows no errors
- [ ] `uv run ruff format .` completes without issues
- [ ] `uv run pre-commit run --all-files` passes
- [ ] `uv run project-name` runs successfully
- [ ] Git repository initialized with `.gitignore`
- [ ] All configuration files present and valid

## Step 7: Customize for Your Project

Now adapt the template to your specific needs:

1. **Update project metadata:**
   - Replace `project-name` and `project_name` throughout
   - Update description in `pyproject.toml`
   - Add actual dependencies for your use case
   - Update README.md with real information

2. **Add domain-specific modules as needed**

3. **Update documentation:**
   - Customize CLAUDE.md with your architecture
   - Add usage examples to README.md
   - Document any special setup requirements

4. **Configure .gitignore:**
   - Add project-specific patterns
   - Ignore generated files, secrets, data files

## Best Practices Summary

1. **Use uv** for all dependency management and command execution
2. **Type hints** everywhere for better IDE support and error detection
3. **Test coverage ≥70%** on critical code paths
4. **DRY principles** - avoid code duplication
5. **Modular architecture** - clear separation of concerns
6. **Pre-commit hooks** - catch issues before they're committed
7. **Clear documentation** - README, docstrings, and CLAUDE.md
8. **Small commits** - one logical change per commit
9. **Descriptive names** - for variables, functions, and modules
10. **Error handling** - fail fast with clear error messages

## Troubleshooting

**uv sync fails:**
- Verify Python 3.12+ installed: `python --version`
- Update uv: `pip install --upgrade uv`
- Check pyproject.toml syntax

**Tests fail with import errors:**
- Always use `uv run pytest`, not bare `pytest`
- Ensure package is installed: `uv sync`

**Pre-commit hooks fail:**
- Fix linting: `uv run ruff check . --fix`
- Format code: `uv run ruff format .`
- Check individual hook output for specific issues

**CLI command not found:**
- Ensure package installed: `uv sync`
- Use full command: `uv run project-name`
- Check [project.scripts] in pyproject.toml

---

## Writing an Effective CLAUDE.md

*Based on best practices from HumanLayer's guide to effective LLM onboarding*

### Core Principles

**Purpose**: CLAUDE.md onboards Claude to your codebase by providing WHAT (tech stack/structure), WHY (project purpose), and HOW (workflows/commands).

**Critical Insight**: LLMs are stateless—they relearn everything each session. CLAUDE.md serves as the primary memory mechanism for your project.

### What to Include

✅ **Project mapping**: Architecture overview, directory structure, component purposes
✅ **Essential workflows**: Build processes, test commands, deployment steps
✅ **Technical context**: Stack details, tool choices, framework decisions
✅ **Verification methods**: How Claude can validate its changes work

### What to AVOID

❌ **Style guidelines**: Never add code style rules—use ruff/formatters instead. This wastes the instruction budget.

❌ **Task-specific details**: Don't include database schemas, API specs, or other narrow details. Keep it universal.

❌ **Auto-generation**: Don't use `/init` to generate CLAUDE.md. Manually craft it—this high-leverage document shapes every interaction.

❌ **Code snippets**: Prefer file:line references over embedded code. Snippets become obsolete as code evolves.

### Critical Constraints

**Instruction Budget**: Frontier LLMs reliably follow ~150-200 instructions. Claude Code's system prompt uses ~50, leaving ~100-150 for your CLAUDE.md.

**Length Target**: Keep under 300 lines. HumanLayer's production CLAUDE.md is under 60 lines.

**Universal Applicability**: Include only information relevant to ALL tasks. Claude has a system reminder that filters out non-universally-applicable content.

### Progressive Disclosure Pattern

For detailed information, use separate markdown files:
- Store complex guidelines in `docs/` or `agent_docs/`
- Reference them briefly in CLAUDE.md
- Use file:line references for specific code locations
- Example: "See `docs/api_design.md` for REST endpoint patterns"

### Structure Template

```markdown
# CLAUDE.md

## Project Overview
[WHAT: 2-3 sentence description of project purpose and architecture]

## Development Environment
**Python**: 3.12 (managed with uv)
**Key Tools**: [List essential tools only]

## Common Commands
[HOW: Only the commands used daily]

## Architecture
[WHAT: High-level component organization]

## Workflows
[HOW: Build, test, deploy processes]
```

### Quality Checklist

Before finalizing CLAUDE.md, verify:

- [ ] Under 300 lines (prefer under 100)
- [ ] No style guidelines (delegated to linters)
- [ ] No code snippets (use file:line references)
- [ ] Every section applies to ALL tasks
- [ ] Commands are actionable and tested
- [ ] Technical decisions explained (WHY)
- [ ] Complex details moved to separate docs
- [ ] No redundant information from system prompts

### Example: Minimal but Effective

```markdown
# CLAUDE.md

## Project Overview
Document object detection using ConvNeXt+D-FINE. Pretrain on PubLayNet, fine-tune on DocLayNet.

## Environment
Python 3.12 (uv), CUDA 12.8, Single GPU training

## Essential Commands
- Train: `uv run doc-obj-detect train --config configs/pretrain.yaml`
- Test: `uv run pytest`
- Lint: `ruff check . && ruff format .`

## Architecture
- Data: `src/doc_obj_detect/data/` - Datasets, augmentation
- Training: `src/doc_obj_detect/training/` - Trainers, callbacks
- Models: HuggingFace Transformers AutoModel

## Validation
Run tests before commits. Check GPU memory with `nvidia-smi`.
```

This 15-line example is often more effective than a 500-line comprehensive guide.

---

## Quick Reference: Essential Commands

```bash
# Setup
uv sync                              # Install dependencies
uv sync --extra dev                  # Include dev dependencies
uv run pre-commit install            # Install git hooks

# Development
uv run pytest                        # Run tests
uv run pytest --cov=src             # With coverage
uv run ruff check .                 # Lint
uv run ruff format .                # Format
uv run pre-commit run --all-files   # Run all hooks

# Running
uv run project-name                 # Run CLI
```

This template provides a solid, general-purpose foundation for any Python project.
