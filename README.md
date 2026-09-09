# Agri-MetaRL

Agri-MetaRL is a meta-reinforcement learning method for greenhouse climate control. It extends Recurrent PPO with a `MetaAdvantageHead` that performs task-adaptive advantage correction across weather years, start dates, and scenario conditions.

[![License: AGPL v3](https://img.shields.io/badge/License-AGPL_v3-blue.svg)](LICENSE)
[![Python 3.11](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![CI](https://github.com/1240945123/Agri-MetaRL/actions/workflows/ci.yml/badge.svg)](https://github.com/1240945123/Agri-MetaRL/actions/workflows/ci.yml)

## Repository layout

| Path | Purpose |
|---|---|
| `src/gl_gym/` | Environment, RL algorithms, and reusable Python package code |
| `configs/` | Agent, environment, and sweep configurations |
| `experiments/scripts/` | Training, evaluation, trajectory, and reporting entry points |
| `datasets/` | Dataset documentation; local weather CSVs are ignored |
| `artifacts/` | Local models, results, generated figures, and tracking data; ignored |
| `paper/` | Canonical manuscript source, public figures, and latest standalone PDF |
| `archive/` | Local private and legacy materials; ignored |
| `tests/` | Automated checks |

## Installation

Requirements:

- Python 3.11 or newer
- A local environment capable of installing the dependencies declared in `pyproject.toml`
- Weather CSV files placed as described in [`datasets/README.md`](datasets/README.md)

From the repository root:

```bash
python -m venv .venv
python -m pip install -e .
```

Use a virtual environment and package cache on the workspace drive rather than the C: drive when working on this machine.

## Usage

Train PPO, Recurrent PPO, and Agri-MetaRL:

```bash
python experiments/scripts/train_paper_experiments.py --device cpu
```

Run the paper evaluation pipeline using existing trained models:

```bash
python experiments/scripts/run_paper_pipeline_after_train.py --skip-train
```

Record a 60-day Agri-MetaRL trajectory:

```bash
python experiments/scripts/record_trajectory_60d.py --algorithm agri_metarl
```

Run lightweight automated checks:

```bash
set PYTHONPATH=src
python -m pytest -q
```

PowerShell users can set the import path with:

```powershell
$env:PYTHONPATH = (Join-Path (Resolve-Path '.') 'src')
python -m pytest -q
```

Training outputs are written beneath `artifacts/models/`; evaluation CSVs are written beneath `artifacts/results/`.

## Paper

- [Canonical manuscript source](paper/manuscript/main.tex)
- [Figure 1](paper/figures/Figure_1.pdf)
- [Figure 2](paper/figures/Figure_2.pdf)
- [Figure 3](paper/figures/Figure_3.pdf)
- [Figure 4](paper/figures/Figure_4.pdf)
- [Figure 5](paper/figures/Figure_5.pdf)
- [Latest available standalone PDF](paper/published/Agri-MetaRL-2026-03-19.pdf)

The canonical source and figures are dated 2026-03-23. The standalone PDF is dated 2026-03-19 and may not exactly match the later source revision. See [`paper/README.md`](paper/README.md).

## Citation

Citation metadata is provided in [`CITATION.cff`](CITATION.cff). The preferred article citation is:

```bibtex
@article{xie2026agrimetarl,
  title   = {Agri-MetaRL: An Agricultural Meta-Reinforcement Learning Algorithm for Greenhouse Climate Control},
  author  = {Xie, Tianchen and Huang, Qiang and Yu, Chengkai and Chen, Qi and Ma, Zhaoxiong and Wang, Mantao},
  journal = {Computers and Electronics in Agriculture},
  year    = {2026},
  note    = {Submitted}
}
```

## Contributing

Contributions are welcome. See [`CONTRIBUTING.md`](CONTRIBUTING.md) for the development setup, coding conventions, and pull request process.

## License

This project is licensed under the GNU Affero General Public License v3.0. See [`LICENSE`](LICENSE).
