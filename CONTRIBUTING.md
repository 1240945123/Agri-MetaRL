# Contributing to Agri-MetaRL

Thanks for your interest in contributing. This document explains how to report issues, set up a development environment, run the checks, and submit changes.

## Reporting issues

Before opening an issue, please:

1. Check whether the same issue already exists.
2. Include a minimal, reproducible description: the command you ran, the Python version, and the full traceback.
3. For environment-related problems, include the greenhouse location and weather year you were using.

## Development setup

Requirements:

- Python 3.11 or newer
- A virtual environment (do not install into the system Python)

```bash
# Clone and install in editable mode
git clone https://github.com/1240945123/Agri-MetaRL.git
cd Agri-MetaRL
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
python -m pip install -U pip
python -m pip install -e .
```

The `timezone` extra is only needed for locations that require timezone lookups:

```bash
python -m pip install -e ".[timezone]"
```

## Running the checks

The lightweight automated checks are the unit tests:

```bash
python -m pytest tests/agri_metarl tests/environments tests/tasks -q
```

To run the full suite (may require trained artifacts or local datasets):

```bash
python -m pytest -q
```

## Project structure

| Path | Purpose |
|---|---|
| `src/gl_gym/` | Environment, RL algorithms, and reusable package code |
| `configs/` | Agent, environment, and sweep configurations |
| `experiments/scripts/` | Training, evaluation, trajectory, and reporting entry points |
| `tests/` | Automated checks |
| `paper/` | Manuscript source and figures |

## Code style

- Follow [PEP 8](https://peps.python.org/pep-0008/).
- Keep new code Python 3.11 compatible.
- Add or update tests for new behavior.
- Prefer `pathlib` for filesystem paths and explicit `encoding="utf-8"` when reading text files.

## Pull request process

1. Fork the repository and create a feature branch from `main`.
2. Make focused changes with clear, descriptive commit messages.
3. Run the unit tests locally and ensure they pass.
4. Open a pull request with a summary of the change and why it is needed.
5. Reference any related issue number.

All contributions are licensed under the project license (AGPL-3.0). By submitting a pull request, you agree to license your contribution under the same terms.
