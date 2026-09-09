# Robust Experiment Redesign Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a manifest-controlled experiment suite for robust greenhouse RL evaluation, replacing stale ad hoc result artifacts with reproducible training, evaluation, aggregation, figure, and integrity-check workflows.

**Architecture:** Add a small suite layer under `src/gl_gym/experiments/` that owns task/scenario definitions, manifest writing, evaluation records, aggregation, and validation. Keep CLI orchestration in `experiments/scripts/`, and keep paper figures/tables generated only from suite artifacts. Existing algorithms and environments remain intact; the suite layer calls them through existing `ExperimentManager`, `make_vec_env`, and SB3-compatible model loading APIs.

**Tech Stack:** Python, pandas, numpy, pytest, matplotlib, stable-baselines3, sb3-contrib, existing `gl_gym` environment and RL modules.

---

## Execution Notes

- Work from `E:\school\Paper\new`.
- Do not install dependencies or caches on `C:`. Use the current workspace drive for generated artifacts.
- Do not stage or commit automatically unless the user explicitly authorizes it during execution.
- Use test-driven development for every code task: write the failing test, run it, implement the minimal code, run it again.
- The new suite must not overwrite `artifacts/results/AgriControl/` or `artifacts/models/AgriControl/`.
- Default suite id for this plan: `AgriControl_C_2026-06-30`.
- New result root: `artifacts/results/AgriControl_C_2026-06-30/`.
- New model root: `artifacts/models/AgriControl_C_2026-06-30/`.

## File Structure

Create:

- `src/gl_gym/experiments/suite_schema.py`  
  Dataclasses and JSON/CSV helpers for suite manifests, scenarios, tasks, runs, and required algorithms.

- `src/gl_gym/experiments/suite_tasks.py`  
  Default economic scenarios, evaluation task matrix generation, and env parameter overrides for a task.

- `src/gl_gym/experiments/suite_evaluation.py`  
  Model loading, deterministic one-episode evaluation, trajectory writing, and raw metric rows.

- `src/gl_gym/experiments/suite_aggregation.py`  
  Summary tables, seed-first aggregation, effect sizes, and deterministic duplicate checks.

- `src/gl_gym/experiments/suite_validation.py`  
  Artifact completeness and integrity checks used by scripts and tests.

- `experiments/scripts/create_experiment_suite.py`  
  CLI that creates a manifest and `eval_tasks.csv`.

- `experiments/scripts/run_suite_training.py`  
  CLI that trains configured algorithms/seeds and writes `runs.csv`.

- `experiments/scripts/evaluate_suite.py`  
  CLI that evaluates every successful run on every suite task and writes `eval_raw.csv`.

- `experiments/scripts/summarize_suite.py`  
  CLI that writes `method_summary.csv`, `stat_tests.csv`, and `diagnostics.csv`.

- `experiments/scripts/generate_suite_figures.py`  
  CLI that creates paper figures only from validated suite artifacts.

- `experiments/scripts/validate_suite_artifacts.py`  
  CLI that fails fast on incomplete or stale suite artifacts.

- `tests/experiments/test_suite_schema.py`
- `tests/experiments/test_suite_tasks.py`
- `tests/experiments/test_suite_evaluation.py`
- `tests/experiments/test_suite_aggregation.py`
- `tests/integrity/test_suite_artifacts.py`
- `tests/integrity/test_suite_figures.py`

Modify:

- `experiments/scripts/train_paper_experiments.py`  
  Add `context_recurrentppo` to the default algorithm list or document that the suite runner supersedes this script.

- `experiments/scripts/README_scripts.md`  
  Document the new suite workflow and mark old fixed-protocol scripts as archival.

- `paper/README.md`  
  Document that C-route manuscript assets must come from `AgriControl_C_2026-06-30`.

---

### Task 1: Suite schema and manifest writer

**Files:**
- Create: `src/gl_gym/experiments/suite_schema.py`
- Create: `tests/experiments/test_suite_schema.py`
- Create: `experiments/scripts/create_experiment_suite.py`

- [ ] **Step 1: Write failing schema tests**

Create `tests/experiments/test_suite_schema.py` with:

```python
import json
from pathlib import Path

import pandas as pd

from gl_gym.experiments.suite_schema import (
    REQUIRED_METHODS,
    EvaluationTaskRecord,
    ExperimentSuiteConfig,
    RunRecord,
    create_default_suite_config,
    load_suite_manifest,
    write_records_csv,
    write_suite_manifest,
)


def test_default_suite_config_has_c_route_methods_and_paths(tmp_path: Path):
    suite = create_default_suite_config(
        suite_id="AgriControl_C_2026-06-30",
        result_root=tmp_path / "results",
        model_root=tmp_path / "models",
    )

    assert suite.suite_id == "AgriControl_C_2026-06-30"
    assert suite.algorithms == list(REQUIRED_METHODS)
    assert suite.seeds == [42, 123, 456, 789, 1024]
    assert suite.train_timesteps == 2_000_000
    assert suite.result_root == str(tmp_path / "results")
    assert suite.model_root == str(tmp_path / "models")


def test_manifest_roundtrip_preserves_nested_values(tmp_path: Path):
    suite = create_default_suite_config(
        suite_id="AgriControl_C_2026-06-30",
        result_root=tmp_path / "results",
        model_root=tmp_path / "models",
    )
    manifest_path = write_suite_manifest(suite)

    loaded = load_suite_manifest(manifest_path)

    assert loaded.suite_id == suite.suite_id
    assert loaded.evaluation_years == [2011, 2012, 2013]
    assert loaded.uncertainty_scales == [0.0, 0.05, 0.10, 0.15]
    assert "combined_stress" in loaded.economic_scenarios


def test_write_records_csv_uses_stable_columns(tmp_path: Path):
    rows = [
        EvaluationTaskRecord(
            suite_id="suite",
            task_id="heldout_2011_d59_u0p00_standard",
            split="heldout",
            weather_year=2011,
            start_day=59,
            uncertainty_scale=0.0,
            economic_scenario="standard",
            climate_constraint_scenario="standard",
        )
    ]

    out = write_records_csv(rows, tmp_path / "eval_tasks.csv")
    df = pd.read_csv(out)

    assert list(df.columns) == [
        "suite_id",
        "task_id",
        "split",
        "weather_year",
        "start_day",
        "uncertainty_scale",
        "economic_scenario",
        "climate_constraint_scenario",
    ]
    assert df.loc[0, "task_id"] == "heldout_2011_d59_u0p00_standard"


def test_run_record_csv_columns(tmp_path: Path):
    rows = [
        RunRecord(
            suite_id="suite",
            algorithm="ppo",
            seed=42,
            run_name="ppo_seed42",
            model_path="artifacts/models/suite/ppo_seed42/best_model.zip",
            vecnormalize_path="artifacts/models/suite/ppo_seed42/best_vecnormalize.pkl",
            status="pending",
            train_steps=0,
            wall_time_seconds=0.0,
            best_eval_return=float("nan"),
            notes="created by test",
        )
    ]

    out = write_records_csv(rows, tmp_path / "runs.csv")
    data = json.loads(Path(out).read_text()) if False else pd.read_csv(out)

    assert data.loc[0, "algorithm"] == "ppo"
    assert data.loc[0, "seed"] == 42
    assert data.loc[0, "status"] == "pending"
```

- [ ] **Step 2: Run schema tests and verify they fail**

Run:

```powershell
python -m pytest tests\experiments\test_suite_schema.py -q
```

Expected: fail because `gl_gym.experiments.suite_schema` does not exist.

- [ ] **Step 3: Implement schema module**

Create `src/gl_gym/experiments/suite_schema.py` with:

```python
from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable


REQUIRED_METHODS: tuple[str, ...] = (
    "ppo",
    "recurrentppo",
    "context_recurrentppo",
    "agri_metarl",
    "rule_based",
)


@dataclass(frozen=True, slots=True)
class ExperimentSuiteConfig:
    suite_id: str
    result_root: str
    model_root: str
    env_id: str
    train_year: int
    train_start_day: int
    train_end_day: int
    evaluation_years: list[int]
    evaluation_start_days: list[int]
    uncertainty_scales: list[float]
    economic_scenarios: list[str]
    algorithms: list[str]
    seeds: list[int]
    train_timesteps: int
    fixed_protocol_year: int
    fixed_protocol_start_day: int
    branch: str
    dirty: bool
    notes: str


@dataclass(frozen=True, slots=True)
class EvaluationTaskRecord:
    suite_id: str
    task_id: str
    split: str
    weather_year: int
    start_day: int
    uncertainty_scale: float
    economic_scenario: str
    climate_constraint_scenario: str


@dataclass(frozen=True, slots=True)
class RunRecord:
    suite_id: str
    algorithm: str
    seed: int
    run_name: str
    model_path: str
    vecnormalize_path: str
    status: str
    train_steps: int
    wall_time_seconds: float
    best_eval_return: float
    notes: str


def create_default_suite_config(
    suite_id: str = "AgriControl_C_2026-06-30",
    result_root: str | Path = "artifacts/results/AgriControl_C_2026-06-30",
    model_root: str | Path = "artifacts/models/AgriControl_C_2026-06-30",
    branch: str = "unknown",
    dirty: bool = True,
) -> ExperimentSuiteConfig:
    return ExperimentSuiteConfig(
        suite_id=suite_id,
        result_root=str(result_root),
        model_root=str(model_root),
        env_id="TomatoEnv",
        train_year=2010,
        train_start_day=59,
        train_end_day=96,
        evaluation_years=[2011, 2012, 2013],
        evaluation_start_days=[59, 80, 100],
        uncertainty_scales=[0.0, 0.05, 0.10, 0.15],
        economic_scenarios=[
            "standard",
            "high_energy_price",
            "low_tomato_price",
            "high_co2_price",
            "combined_stress",
        ],
        algorithms=list(REQUIRED_METHODS),
        seeds=[42, 123, 456, 789, 1024],
        train_timesteps=2_000_000,
        fixed_protocol_year=2010,
        fixed_protocol_start_day=59,
        branch=branch,
        dirty=dirty,
        notes="C-route robust task-distribution experiment suite.",
    )


def suite_manifest_path(suite: ExperimentSuiteConfig) -> Path:
    return Path(suite.result_root) / "suite_manifest.json"


def write_suite_manifest(suite: ExperimentSuiteConfig) -> Path:
    path = suite_manifest_path(suite)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(suite), indent=2, sort_keys=True), encoding="utf-8")
    return path


def load_suite_manifest(path: str | Path) -> ExperimentSuiteConfig:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return ExperimentSuiteConfig(**data)


def write_records_csv(records: Iterable[Any], path: str | Path) -> Path:
    rows = [asdict(record) for record in records]
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"no rows to write for {out}")
    with out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return out
```

- [ ] **Step 4: Add create-suite CLI**

Create `experiments/scripts/create_experiment_suite.py` with:

```python
#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "src"))

from gl_gym.experiments.suite_schema import create_default_suite_config, write_suite_manifest
from gl_gym.experiments.suite_tasks import build_evaluation_tasks
from gl_gym.experiments.suite_schema import write_records_csv


def git_branch() -> str:
    result = subprocess.run(
        ["git", "branch", "--show-current"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    return result.stdout.strip() or "unknown"


def git_dirty() -> bool:
    result = subprocess.run(
        ["git", "status", "--short"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    return bool(result.stdout.strip())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite_id", default="AgriControl_C_2026-06-30")
    parser.add_argument("--result_root", default="artifacts/results/AgriControl_C_2026-06-30")
    parser.add_argument("--model_root", default="artifacts/models/AgriControl_C_2026-06-30")
    args = parser.parse_args()

    suite = create_default_suite_config(
        suite_id=args.suite_id,
        result_root=args.result_root,
        model_root=args.model_root,
        branch=git_branch(),
        dirty=git_dirty(),
    )
    manifest_path = write_suite_manifest(suite)
    tasks = build_evaluation_tasks(suite)
    task_path = write_records_csv(tasks, os.path.join(suite.result_root, "eval_tasks.csv"))
    print(f"Wrote {manifest_path}")
    print(f"Wrote {task_path}")


if __name__ == "__main__":
    main()
```

This script depends on `suite_tasks.py`, which is created in Task 2. The script should be added in Task 1 but it will not pass until Task 2 exists.

- [ ] **Step 5: Run schema tests**

Run:

```powershell
python -m pytest tests\experiments\test_suite_schema.py -q
```

Expected: pass except any import failure for `suite_tasks` if the create-suite CLI is imported by a test. The tests above do not import the CLI, so expected final result is pass.

- [ ] **Step 6: User-approved checkpoint**

If the user approves staging/commit:

```powershell
git add src\gl_gym\experiments\suite_schema.py tests\experiments\test_suite_schema.py experiments\scripts\create_experiment_suite.py
git commit -m "feat: add robust experiment suite schema"
```

---

### Task 2: Task matrix and economic scenario definitions

**Files:**
- Create: `src/gl_gym/experiments/suite_tasks.py`
- Create: `tests/experiments/test_suite_tasks.py`

- [ ] **Step 1: Write failing task/scenario tests**

Create `tests/experiments/test_suite_tasks.py` with:

```python
from pathlib import Path

from gl_gym.experiments.suite_schema import create_default_suite_config
from gl_gym.experiments.suite_tasks import (
    ECONOMIC_SCENARIOS,
    apply_task_to_env_params,
    build_evaluation_tasks,
    scenario_reward_params,
)


BASE_REWARD = {
    "elec_price": 0.3,
    "heating_price": 0.09,
    "co2_price": 0.3,
    "fruit_price": 1.6,
}


def test_economic_scenarios_are_named_and_reproducible():
    assert sorted(ECONOMIC_SCENARIOS) == [
        "combined_stress",
        "high_co2_price",
        "high_energy_price",
        "low_tomato_price",
        "standard",
    ]
    assert scenario_reward_params(BASE_REWARD, "standard") == BASE_REWARD
    high_energy = scenario_reward_params(BASE_REWARD, "high_energy_price")
    assert high_energy["elec_price"] == 0.45
    assert high_energy["heating_price"] == 0.135
    assert high_energy["fruit_price"] == 1.6


def test_build_evaluation_tasks_counts_fixed_heldout_uncertainty_and_economic(tmp_path: Path):
    suite = create_default_suite_config(
        result_root=tmp_path / "results",
        model_root=tmp_path / "models",
    )

    tasks = build_evaluation_tasks(suite)
    fixed = [task for task in tasks if task.split == "fixed"]
    heldout = [task for task in tasks if task.split == "heldout"]
    uncertainty = [task for task in tasks if task.split == "uncertainty"]
    economic = [task for task in tasks if task.split == "economic"]

    assert len(fixed) == 1
    assert len(heldout) == 9
    assert len(uncertainty) == 36
    assert len(economic) == 45
    assert len({task.task_id for task in tasks}) == len(tasks)


def test_apply_task_to_env_params_sets_single_eval_task_and_prices(tmp_path: Path):
    suite = create_default_suite_config(
        result_root=tmp_path / "results",
        model_root=tmp_path / "models",
    )
    task = next(
        item
        for item in build_evaluation_tasks(suite)
        if item.task_id == "economic_2011_d59_u0p00_high_energy_price"
    )
    base_params = {"training": True}
    specific_params = {"reward_params": dict(BASE_REWARD), "eval_options": {}}

    base_out, specific_out = apply_task_to_env_params(base_params, specific_params, task)

    assert base_out["training"] is False
    assert specific_out["uncertainty_scale"] == 0.0
    assert specific_out["economic_scenario"] == "high_energy_price"
    assert specific_out["eval_options"]["eval_years"] == [2011]
    assert specific_out["eval_options"]["eval_days"] == [59]
    assert specific_out["reward_params"]["elec_price"] == 0.45
    assert specific_out["reward_params"]["heating_price"] == 0.135
```

- [ ] **Step 2: Run task tests and verify they fail**

Run:

```powershell
python -m pytest tests\experiments\test_suite_tasks.py -q
```

Expected: fail because `suite_tasks.py` does not exist.

- [ ] **Step 3: Implement suite task generation**

Create `src/gl_gym/experiments/suite_tasks.py` with:

```python
from __future__ import annotations

from copy import deepcopy
from typing import Mapping

from gl_gym.experiments.suite_schema import EvaluationTaskRecord, ExperimentSuiteConfig


ECONOMIC_SCENARIOS: dict[str, dict[str, float]] = {
    "standard": {
        "elec_price": 1.0,
        "heating_price": 1.0,
        "co2_price": 1.0,
        "fruit_price": 1.0,
    },
    "high_energy_price": {
        "elec_price": 1.5,
        "heating_price": 1.5,
        "co2_price": 1.0,
        "fruit_price": 1.0,
    },
    "low_tomato_price": {
        "elec_price": 1.0,
        "heating_price": 1.0,
        "co2_price": 1.0,
        "fruit_price": 0.7,
    },
    "high_co2_price": {
        "elec_price": 1.0,
        "heating_price": 1.0,
        "co2_price": 2.0,
        "fruit_price": 1.0,
    },
    "combined_stress": {
        "elec_price": 1.5,
        "heating_price": 1.5,
        "co2_price": 2.0,
        "fruit_price": 0.7,
    },
}


def _scale_token(value: float) -> str:
    return f"{value:.2f}".replace(".", "p")


def _task(
    suite: ExperimentSuiteConfig,
    split: str,
    year: int,
    start_day: int,
    uncertainty: float,
    scenario: str,
) -> EvaluationTaskRecord:
    return EvaluationTaskRecord(
        suite_id=suite.suite_id,
        task_id=f"{split}_{year}_d{start_day}_u{_scale_token(uncertainty)}_{scenario}",
        split=split,
        weather_year=year,
        start_day=start_day,
        uncertainty_scale=float(uncertainty),
        economic_scenario=scenario,
        climate_constraint_scenario="standard",
    )


def build_evaluation_tasks(suite: ExperimentSuiteConfig) -> list[EvaluationTaskRecord]:
    tasks: list[EvaluationTaskRecord] = [
        _task(
            suite,
            "fixed",
            suite.fixed_protocol_year,
            suite.fixed_protocol_start_day,
            0.0,
            "standard",
        )
    ]

    for year in suite.evaluation_years:
        for start_day in suite.evaluation_start_days:
            tasks.append(_task(suite, "heldout", year, start_day, 0.0, "standard"))

    for year in suite.evaluation_years:
        for start_day in suite.evaluation_start_days:
            for uncertainty in suite.uncertainty_scales:
                tasks.append(_task(suite, "uncertainty", year, start_day, uncertainty, "standard"))

    for year in suite.evaluation_years:
        for start_day in suite.evaluation_start_days:
            for scenario in suite.economic_scenarios:
                tasks.append(_task(suite, "economic", year, start_day, 0.0, scenario))

    return tasks


def scenario_reward_params(base_reward_params: Mapping[str, float], scenario: str) -> dict[str, float]:
    if scenario not in ECONOMIC_SCENARIOS:
        raise ValueError(f"unknown economic scenario: {scenario}")
    multipliers = ECONOMIC_SCENARIOS[scenario]
    updated = dict(base_reward_params)
    for key, multiplier in multipliers.items():
        if key in updated:
            updated[key] = updated[key] * multiplier
    return updated


def apply_task_to_env_params(
    env_base_params: Mapping,
    env_specific_params: Mapping,
    task: EvaluationTaskRecord,
) -> tuple[dict, dict]:
    base_out = deepcopy(dict(env_base_params))
    specific_out = deepcopy(dict(env_specific_params))
    reward_params = deepcopy(specific_out.get("reward_params", {}))

    base_out["training"] = False
    specific_out["uncertainty_scale"] = float(task.uncertainty_scale)
    specific_out["economic_scenario"] = task.economic_scenario
    specific_out["eval_options"] = {
        "eval_years": [int(task.weather_year)],
        "eval_days": [int(task.start_day)],
    }
    specific_out["reward_params"] = scenario_reward_params(reward_params, task.economic_scenario)
    return base_out, specific_out
```

- [ ] **Step 4: Run task tests**

Run:

```powershell
python -m pytest tests\experiments\test_suite_tasks.py -q
```

Expected: pass.

- [ ] **Step 5: Run create-suite CLI smoke test**

Run:

```powershell
python experiments\scripts\create_experiment_suite.py --suite_id AgriControl_C_2026-06-30 --result_root artifacts\results\AgriControl_C_2026-06-30 --model_root artifacts\models\AgriControl_C_2026-06-30
```

Expected:

```text
Wrote artifacts\results\AgriControl_C_2026-06-30\suite_manifest.json
Wrote artifacts\results\AgriControl_C_2026-06-30\eval_tasks.csv
```

- [ ] **Step 6: User-approved checkpoint**

If the user approves staging/commit:

```powershell
git add src\gl_gym\experiments\suite_tasks.py tests\experiments\test_suite_tasks.py artifacts\results\AgriControl_C_2026-06-30\suite_manifest.json artifacts\results\AgriControl_C_2026-06-30\eval_tasks.csv
git commit -m "feat: define robust evaluation task matrix"
```

---

### Task 3: Suite training runner and run registry

**Files:**
- Create: `experiments/scripts/run_suite_training.py`
- Create: `tests/experiments/test_suite_training_cli.py`
- Modify: `experiments/scripts/train_paper_experiments.py`
- Modify: `experiments/scripts/README_scripts.md`

- [ ] **Step 1: Write failing training registry test**

Create `tests/experiments/test_suite_training_cli.py` with:

```python
from pathlib import Path

import pandas as pd

from gl_gym.experiments.suite_schema import RunRecord, create_default_suite_config, write_records_csv


def test_run_registry_records_all_learning_methods_and_seeds(tmp_path: Path):
    suite = create_default_suite_config(
        result_root=tmp_path / "results",
        model_root=tmp_path / "models",
    )
    learning_algorithms = [algo for algo in suite.algorithms if algo != "rule_based"]
    rows = [
        RunRecord(
            suite_id=suite.suite_id,
            algorithm=algo,
            seed=seed,
            run_name=f"{algo}_seed{seed}",
            model_path=str(Path(suite.model_root) / algo / f"seed{seed}" / "best_model.zip"),
            vecnormalize_path=str(Path(suite.model_root) / algo / f"seed{seed}" / "best_vecnormalize.pkl"),
            status="pending",
            train_steps=0,
            wall_time_seconds=0.0,
            best_eval_return=float("nan"),
            notes="created before training",
        )
        for algo in learning_algorithms
        for seed in suite.seeds
    ]

    out = write_records_csv(rows, tmp_path / "runs.csv")
    df = pd.read_csv(out)

    assert len(df) == 20
    assert set(df["algorithm"]) == {"ppo", "recurrentppo", "context_recurrentppo", "agri_metarl"}
    assert sorted(df["seed"].unique().tolist()) == [42, 123, 456, 789, 1024]
```

- [ ] **Step 2: Run training registry test**

Run:

```powershell
python -m pytest tests\experiments\test_suite_training_cli.py -q
```

Expected: pass after Task 1 because it tests schema only. This test locks down registry shape before CLI implementation.

- [ ] **Step 3: Create suite training CLI**

Create `experiments/scripts/run_suite_training.py` with:

```python
#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import os
import sys
import time
from pathlib import Path

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "src"))

from gl_gym.RL.experiment_manager import ExperimentManager
from gl_gym.common.utils import load_env_params, load_model_hyperparams
from gl_gym.experiments.suite_schema import RunRecord, load_suite_manifest, write_records_csv


def train_one(suite, algorithm: str, seed: int, device: str, dry_run: bool) -> RunRecord:
    run_name = f"{algorithm}_seed{seed}"
    run_dir = Path(suite.model_root) / algorithm / f"seed{seed}"
    model_path = run_dir / "best_model.zip"
    vecnormalize_path = run_dir / "best_vecnormalize.pkl"

    if dry_run:
        return RunRecord(
            suite_id=suite.suite_id,
            algorithm=algorithm,
            seed=seed,
            run_name=run_name,
            model_path=str(model_path),
            vecnormalize_path=str(vecnormalize_path),
            status="dry_run",
            train_steps=0,
            wall_time_seconds=0.0,
            best_eval_return=float("nan"),
            notes="dry-run registry entry; model not trained",
        )

    env_base_params, env_specific_params = load_env_params(suite.env_id, "configs/envs/")
    env_base_params["start_train_year"] = suite.train_year
    env_base_params["end_train_year"] = suite.train_year
    env_base_params["start_train_day"] = suite.train_start_day
    env_base_params["end_train_day"] = suite.train_end_day
    hyperparameters = load_model_hyperparams(algorithm, suite.env_id)
    hyperparameters["total_timesteps"] = suite.train_timesteps
    start = time.time()
    manager = ExperimentManager(
        env_id=suite.env_id,
        project=suite.suite_id,
        env_base_params=env_base_params,
        env_specific_params=env_specific_params,
        hyperparameters=copy.deepcopy(hyperparameters),
        group=run_name,
        n_eval_episodes=1,
        n_evals=10,
        algorithm=algorithm,
        env_seed=seed,
        model_seed=seed,
        stochastic=False,
        save_model=True,
        save_env=True,
        device=device,
    )
    manager.run_experiment()
    wall_time = time.time() - start
    return RunRecord(
        suite_id=suite.suite_id,
        algorithm=algorithm,
        seed=seed,
        run_name=run_name,
        model_path=str(model_path),
        vecnormalize_path=str(vecnormalize_path),
        status="completed",
        train_steps=suite.train_timesteps,
        wall_time_seconds=wall_time,
        best_eval_return=float("nan"),
        notes="completed by run_suite_training.py",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--algorithms", nargs="+")
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    suite = load_suite_manifest(args.manifest)
    algorithms = args.algorithms or [algo for algo in suite.algorithms if algo != "rule_based"]
    seeds = args.seeds or suite.seeds
    rows = [train_one(suite, algo, seed, args.device, args.dry_run) for algo in algorithms for seed in seeds]
    out = write_records_csv(rows, Path(suite.result_root) / "runs.csv")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Modify legacy training defaults**

In `experiments/scripts/train_paper_experiments.py`, change:

```python
ALGORITHMS = ["ppo", "recurrentppo", "agri_metarl"]
```

to:

```python
ALGORITHMS = ["ppo", "recurrentppo", "context_recurrentppo", "agri_metarl"]
```

- [ ] **Step 5: Document the suite training path**

Append this block to `experiments/scripts/README_scripts.md`:

```markdown
## C-route robust suite workflow

Use `create_experiment_suite.py`, `run_suite_training.py`, `evaluate_suite.py`, `summarize_suite.py`, and `generate_suite_figures.py` for the redesigned paper.

The older fixed-protocol scripts are retained for archival comparisons only. Do not use their summaries as manuscript evidence for the C-route paper.
```

- [ ] **Step 6: Run dry-run training registry**

Run:

```powershell
python experiments\scripts\run_suite_training.py --manifest artifacts\results\AgriControl_C_2026-06-30\suite_manifest.json --algorithms ppo context_recurrentppo --seeds 42 --dry_run
```

Expected:

```text
Wrote artifacts\results\AgriControl_C_2026-06-30\runs.csv
```

Then inspect:

```powershell
@'
import pandas as pd
df = pd.read_csv("artifacts/results/AgriControl_C_2026-06-30/runs.csv")
print(df[["algorithm", "seed", "status"]].to_string(index=False))
'@ | python -
```

Expected rows: `ppo 42 dry_run` and `context_recurrentppo 42 dry_run`.

- [ ] **Step 7: User-approved checkpoint**

If the user approves staging/commit:

```powershell
git add experiments\scripts\run_suite_training.py experiments\scripts\train_paper_experiments.py experiments\scripts\README_scripts.md tests\experiments\test_suite_training_cli.py
git commit -m "feat: add robust suite training runner"
```

---

### Task 4: Deterministic suite evaluator

**Files:**
- Create: `src/gl_gym/experiments/suite_evaluation.py`
- Create: `experiments/scripts/evaluate_suite.py`
- Create: `tests/experiments/test_suite_evaluation.py`

- [ ] **Step 1: Write failing evaluation tests**

Create `tests/experiments/test_suite_evaluation.py` with:

```python
from pathlib import Path

import numpy as np
import pandas as pd

from gl_gym.experiments.suite_evaluation import (
    EvaluationMetricRow,
    run_deterministic_episode,
    write_eval_raw,
)


class FakeModel:
    def predict(self, obs, deterministic=True):
        return np.array([[0.0]]), None


class FakeEnv:
    def __init__(self):
        self.step_count = 0

    def get_attr(self, name):
        if name == "N":
            return [3]
        raise AttributeError(name)

    def reset(self):
        return np.array([[0.0]])

    def step(self, actions):
        self.step_count += 1
        info = {
            "EPI": 1.0,
            "revenue": 2.0,
            "heat_cost": 0.3,
            "co2_cost": 0.2,
            "elec_cost": 0.1,
            "temp_violation": 1,
            "co2_violation": 2,
            "rh_violation": 3,
        }
        return np.array([[0.0]]), np.array([10.0]), np.array([self.step_count == 3]), [info]


def test_run_deterministic_episode_sums_metrics():
    metrics = run_deterministic_episode(FakeModel(), FakeEnv())

    assert metrics["episode_return"] == 30.0
    assert metrics["EPI"] == 3.0
    assert metrics["revenue"] == 6.0
    assert metrics["temp_violation"] == 3
    assert metrics["co2_violation"] == 6
    assert metrics["rh_violation"] == 9


def test_write_eval_raw_has_one_row_per_run_task(tmp_path: Path):
    rows = [
        EvaluationMetricRow(
            suite_id="suite",
            algorithm="ppo",
            seed=42,
            run_name="ppo_seed42",
            task_id="fixed_2010_d59_u0p00_standard",
            split="fixed",
            weather_year=2010,
            start_day=59,
            uncertainty_scale=0.0,
            economic_scenario="standard",
            episode_return=30.0,
            EPI=3.0,
            revenue=6.0,
            heat_cost=0.9,
            co2_cost=0.6,
            elec_cost=0.3,
            temp_violation=3.0,
            co2_violation=6.0,
            rh_violation=9.0,
            twb_percent=float("nan"),
            trajectory_path="",
        )
    ]

    out = write_eval_raw(rows, tmp_path / "eval_raw.csv")
    df = pd.read_csv(out)

    assert len(df) == 1
    assert df.loc[0, "algorithm"] == "ppo"
    assert df.loc[0, "task_id"] == "fixed_2010_d59_u0p00_standard"
```

- [ ] **Step 2: Run evaluation tests and verify they fail**

Run:

```powershell
python -m pytest tests\experiments\test_suite_evaluation.py -q
```

Expected: fail because `suite_evaluation.py` does not exist.

- [ ] **Step 3: Implement evaluation helpers**

Create `src/gl_gym/experiments/suite_evaluation.py` with:

```python
from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True, slots=True)
class EvaluationMetricRow:
    suite_id: str
    algorithm: str
    seed: int
    run_name: str
    task_id: str
    split: str
    weather_year: int
    start_day: int
    uncertainty_scale: float
    economic_scenario: str
    episode_return: float
    EPI: float
    revenue: float
    heat_cost: float
    co2_cost: float
    elec_cost: float
    temp_violation: float
    co2_violation: float
    rh_violation: float
    twb_percent: float
    trajectory_path: str


def _predict(model: Any, obs: Any, states: Any, episode_starts: np.ndarray):
    try:
        return model.predict(obs, state=states, episode_start=episode_starts, deterministic=True)
    except TypeError:
        return model.predict(obs, deterministic=True)


def run_deterministic_episode(model: Any, env: Any) -> dict[str, float]:
    n_steps = int(env.get_attr("N")[0])
    result = env.reset()
    obs = result[0] if isinstance(result, (tuple, list)) else result
    states = None
    episode_starts = np.ones((1,), dtype=bool)
    totals = {
        "episode_return": 0.0,
        "EPI": 0.0,
        "revenue": 0.0,
        "heat_cost": 0.0,
        "co2_cost": 0.0,
        "elec_cost": 0.0,
        "temp_violation": 0.0,
        "co2_violation": 0.0,
        "rh_violation": 0.0,
        "twb_percent": float("nan"),
    }
    for _ in range(n_steps):
        prediction = _predict(model, obs, states, episode_starts)
        if isinstance(prediction, tuple) and len(prediction) == 2:
            actions, states = prediction
        else:
            actions, states = prediction, None
        obs, rewards, dones, infos = env.step(actions)
        info = infos[0]
        totals["episode_return"] += float(rewards[0])
        totals["EPI"] += float(info["EPI"])
        totals["revenue"] += float(info["revenue"])
        totals["heat_cost"] += float(info["heat_cost"])
        totals["co2_cost"] += float(info["co2_cost"])
        totals["elec_cost"] += float(info["elec_cost"])
        totals["temp_violation"] += float(info["temp_violation"])
        totals["co2_violation"] += float(info["co2_violation"])
        totals["rh_violation"] += float(info["rh_violation"])
        episode_starts = dones
    return totals


def write_eval_raw(rows: list[EvaluationMetricRow], path: str | Path) -> Path:
    if not rows:
        raise ValueError("no evaluation rows to write")
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    row_dicts = [asdict(row) for row in rows]
    with out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row_dicts[0].keys()))
        writer.writeheader()
        writer.writerows(row_dicts)
    return out
```

- [ ] **Step 4: Create evaluation CLI**

Create `experiments/scripts/evaluate_suite.py` with:

```python
#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize
from sb3_contrib import RecurrentPPO

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "src"))

from gl_gym.RL.agri_metarl import AgriMetaRL
from gl_gym.RL.context_recurrent_ppo import ContextRecurrentPPO
from gl_gym.RL.utils import make_vec_env
from gl_gym.common.utils import load_env_params
from gl_gym.experiments.suite_evaluation import EvaluationMetricRow, run_deterministic_episode, write_eval_raw
from gl_gym.experiments.suite_schema import EvaluationTaskRecord, load_suite_manifest
from gl_gym.experiments.suite_tasks import apply_task_to_env_params


ALG_MAP = {
    "ppo": PPO,
    "recurrentppo": RecurrentPPO,
    "context_recurrentppo": ContextRecurrentPPO,
    "agri_metarl": AgriMetaRL,
}


def load_task_env(suite, task: EvaluationTaskRecord, vecnormalize_path: str):
    env_base_params, env_specific_params = load_env_params(suite.env_id, "configs/envs/")
    env_base_params, env_specific_params = apply_task_to_env_params(env_base_params, env_specific_params, task)
    env = make_vec_env(
        suite.env_id,
        env_base_params,
        env_specific_params,
        seed=666,
        n_envs=1,
        monitor_filename=None,
        vec_norm_kwargs=None,
        eval_env=True,
    )
    if vecnormalize_path and os.path.isfile(vecnormalize_path):
        env = VecNormalize.load(vecnormalize_path, env)
        env.training = False
        env.norm_reward = False
    return env


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--runs_csv", required=True)
    parser.add_argument("--tasks_csv", required=True)
    parser.add_argument("--algorithms", nargs="+")
    parser.add_argument("--seeds", nargs="+", type=int)
    args = parser.parse_args()

    suite = load_suite_manifest(args.manifest)
    runs = pd.read_csv(args.runs_csv)
    tasks_df = pd.read_csv(args.tasks_csv)
    if args.algorithms:
        runs = runs[runs["algorithm"].isin(args.algorithms)]
    if args.seeds:
        runs = runs[runs["seed"].isin(args.seeds)]
    rows: list[EvaluationMetricRow] = []

    for run in runs.itertuples(index=False):
        if run.algorithm == "rule_based" or run.status not in {"completed", "dry_run"}:
            continue
        if run.status == "dry_run":
            continue
        model = ALG_MAP[run.algorithm].load(run.model_path, device="cpu")
        for task_row in tasks_df.itertuples(index=False):
            task = EvaluationTaskRecord(
                suite_id=task_row.suite_id,
                task_id=task_row.task_id,
                split=task_row.split,
                weather_year=int(task_row.weather_year),
                start_day=int(task_row.start_day),
                uncertainty_scale=float(task_row.uncertainty_scale),
                economic_scenario=task_row.economic_scenario,
                climate_constraint_scenario=task_row.climate_constraint_scenario,
            )
            env = load_task_env(suite, task, run.vecnormalize_path)
            metrics = run_deterministic_episode(model, env)
            env.close()
            rows.append(
                EvaluationMetricRow(
                    suite_id=suite.suite_id,
                    algorithm=run.algorithm,
                    seed=int(run.seed),
                    run_name=run.run_name,
                    task_id=task.task_id,
                    split=task.split,
                    weather_year=task.weather_year,
                    start_day=task.start_day,
                    uncertainty_scale=task.uncertainty_scale,
                    economic_scenario=task.economic_scenario,
                    trajectory_path="",
                    **metrics,
                )
            )

    out = write_eval_raw(rows, Path(suite.result_root) / "eval_raw.csv")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Run evaluation tests**

Run:

```powershell
python -m pytest tests\experiments\test_suite_evaluation.py -q
```

Expected: pass.

- [ ] **Step 6: User-approved checkpoint**

If the user approves staging/commit:

```powershell
git add src\gl_gym\experiments\suite_evaluation.py experiments\scripts\evaluate_suite.py tests\experiments\test_suite_evaluation.py
git commit -m "feat: add deterministic suite evaluator"
```

---

### Task 5: Aggregation, effect sizes, and duplicate detection

**Files:**
- Create: `src/gl_gym/experiments/suite_aggregation.py`
- Create: `experiments/scripts/summarize_suite.py`
- Create: `tests/experiments/test_suite_aggregation.py`

- [ ] **Step 1: Write failing aggregation tests**

Create `tests/experiments/test_suite_aggregation.py` with:

```python
import pandas as pd
import pytest

from gl_gym.experiments.suite_aggregation import (
    aggregate_seed_first,
    assert_no_duplicate_run_task_rows,
    cohens_d,
)


def test_duplicate_run_task_rows_are_rejected():
    df = pd.DataFrame(
        [
            {"algorithm": "ppo", "seed": 42, "run_name": "ppo_seed42", "task_id": "fixed"},
            {"algorithm": "ppo", "seed": 42, "run_name": "ppo_seed42", "task_id": "fixed"},
        ]
    )

    with pytest.raises(ValueError, match="duplicate deterministic evaluation rows"):
        assert_no_duplicate_run_task_rows(df)


def test_aggregate_seed_first_averages_tasks_before_seed_statistics():
    df = pd.DataFrame(
        [
            {"suite_id": "suite", "algorithm": "ppo", "seed": 1, "split": "heldout", "episode_return": 10.0, "EPI": 1.0},
            {"suite_id": "suite", "algorithm": "ppo", "seed": 1, "split": "heldout", "episode_return": 20.0, "EPI": 3.0},
            {"suite_id": "suite", "algorithm": "ppo", "seed": 2, "split": "heldout", "episode_return": 30.0, "EPI": 5.0},
            {"suite_id": "suite", "algorithm": "ppo", "seed": 2, "split": "heldout", "episode_return": 40.0, "EPI": 7.0},
        ]
    )

    summary = aggregate_seed_first(df, metrics=["episode_return", "EPI"])

    assert summary.loc[0, "suite_id"] == "suite"
    assert summary.loc[0, "episode_return_mean"] == 25.0
    assert summary.loc[0, "EPI_mean"] == 4.0
    assert summary.loc[0, "n_seeds"] == 2


def test_cohens_d_returns_signed_effect_size():
    assert round(cohens_d([2.0, 2.0, 2.0], [1.0, 1.0, 1.0]), 3) == 0.0
    assert cohens_d([2.0, 3.0, 4.0], [1.0, 2.0, 3.0]) > 0
```

- [ ] **Step 2: Run aggregation tests and verify they fail**

Run:

```powershell
python -m pytest tests\experiments\test_suite_aggregation.py -q
```

Expected: fail because `suite_aggregation.py` does not exist.

- [ ] **Step 3: Implement aggregation helpers**

Create `src/gl_gym/experiments/suite_aggregation.py` with:

```python
from __future__ import annotations

from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_METRICS = [
    "episode_return",
    "EPI",
    "revenue",
    "heat_cost",
    "co2_cost",
    "elec_cost",
    "temp_violation",
    "co2_violation",
    "rh_violation",
]


def assert_no_duplicate_run_task_rows(df: pd.DataFrame) -> None:
    keys = ["algorithm", "seed", "run_name", "task_id"]
    duplicates = df.duplicated(keys, keep=False)
    if duplicates.any():
        duplicate_rows = df.loc[duplicates, keys].head(10).to_dict("records")
        raise ValueError(f"duplicate deterministic evaluation rows: {duplicate_rows}")


def aggregate_seed_first(df: pd.DataFrame, metrics: list[str] | None = None) -> pd.DataFrame:
    metrics = metrics or DEFAULT_METRICS
    assert_no_duplicate_run_task_rows(df)
    suite_columns = ["suite_id"] if "suite_id" in df.columns else []
    seed_level = df.groupby(suite_columns + ["algorithm", "seed", "split"], as_index=False)[metrics].mean()
    grouped = seed_level.groupby(suite_columns + ["algorithm", "split"], as_index=False)
    rows = []
    for key, group in grouped:
        if suite_columns:
            suite_id, algorithm, split = key
            row = {"suite_id": suite_id, "algorithm": algorithm, "split": split, "n_seeds": int(group["seed"].nunique())}
        else:
            algorithm, split = key
            row = {"algorithm": algorithm, "split": split, "n_seeds": int(group["seed"].nunique())}
        for metric in metrics:
            row[f"{metric}_mean"] = float(group[metric].mean())
            row[f"{metric}_std"] = float(group[metric].std(ddof=1)) if len(group) > 1 else 0.0
        rows.append(row)
    return pd.DataFrame(rows)


def cohens_d(a, b) -> float:
    left = np.asarray(list(a), dtype=float)
    right = np.asarray(list(b), dtype=float)
    if len(left) < 2 or len(right) < 2:
        return 0.0
    pooled_var = ((len(left) - 1) * left.var(ddof=1) + (len(right) - 1) * right.var(ddof=1)) / (len(left) + len(right) - 2)
    if pooled_var <= 0:
        return 0.0
    return float((left.mean() - right.mean()) / np.sqrt(pooled_var))


def paired_effect_table(df: pd.DataFrame, metric: str = "episode_return") -> pd.DataFrame:
    assert_no_duplicate_run_task_rows(df)
    seed_level = df.groupby(["algorithm", "seed", "split"], as_index=False)[metric].mean()
    rows = []
    for split, split_df in seed_level.groupby("split"):
        for left, right in combinations(sorted(split_df["algorithm"].unique()), 2):
            left_values = split_df.loc[split_df["algorithm"] == left, metric].to_numpy()
            right_values = split_df.loc[split_df["algorithm"] == right, metric].to_numpy()
            rows.append(
                {
                    "split": split,
                    "metric": metric,
                    "method_a": left,
                    "method_b": right,
                    "cohens_d_a_minus_b": cohens_d(left_values, right_values),
                    "mean_a": float(np.mean(left_values)) if len(left_values) else float("nan"),
                    "mean_b": float(np.mean(right_values)) if len(right_values) else float("nan"),
                }
            )
    return pd.DataFrame(rows)


def write_summary_files(eval_raw_path: str | Path, result_root: str | Path) -> tuple[Path, Path, Path]:
    df = pd.read_csv(eval_raw_path)
    summary = aggregate_seed_first(df)
    stats = paired_effect_table(df, metric="episode_return")
    diagnostics = pd.DataFrame(
        [{"diagnostic": "not_collected", "value": float("nan"), "notes": "diagnostic hooks are added after pilot validation"}]
    )
    root = Path(result_root)
    summary_path = root / "method_summary.csv"
    stats_path = root / "stat_tests.csv"
    diagnostics_path = root / "diagnostics.csv"
    summary.to_csv(summary_path, index=False)
    stats.to_csv(stats_path, index=False)
    diagnostics.to_csv(diagnostics_path, index=False)
    return summary_path, stats_path, diagnostics_path
```

- [ ] **Step 4: Add summarize CLI**

Create `experiments/scripts/summarize_suite.py` with:

```python
#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "src"))

from gl_gym.experiments.suite_aggregation import write_summary_files
from gl_gym.experiments.suite_schema import load_suite_manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--eval_raw")
    args = parser.parse_args()

    suite = load_suite_manifest(args.manifest)
    eval_raw = args.eval_raw or str(Path(suite.result_root) / "eval_raw.csv")
    summary_path, stats_path, diagnostics_path = write_summary_files(eval_raw, suite.result_root)
    print(f"Wrote {summary_path}")
    print(f"Wrote {stats_path}")
    print(f"Wrote {diagnostics_path}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Run aggregation tests**

Run:

```powershell
python -m pytest tests\experiments\test_suite_aggregation.py -q
```

Expected: pass.

- [ ] **Step 6: User-approved checkpoint**

If the user approves staging/commit:

```powershell
git add src\gl_gym\experiments\suite_aggregation.py experiments\scripts\summarize_suite.py tests\experiments\test_suite_aggregation.py
git commit -m "feat: add seed-first suite aggregation"
```

---

### Task 6: Artifact validation and paper figure guardrails

**Files:**
- Create: `src/gl_gym/experiments/suite_validation.py`
- Create: `experiments/scripts/validate_suite_artifacts.py`
- Create: `experiments/scripts/generate_suite_figures.py`
- Create: `tests/integrity/test_suite_artifacts.py`
- Create: `tests/integrity/test_suite_figures.py`

- [ ] **Step 1: Write failing validation tests**

Create `tests/integrity/test_suite_artifacts.py` with:

```python
from pathlib import Path

import pandas as pd
import pytest

from gl_gym.experiments.suite_schema import create_default_suite_config, write_suite_manifest
from gl_gym.experiments.suite_validation import validate_suite_artifacts


def test_validate_suite_rejects_missing_required_files(tmp_path: Path):
    suite = create_default_suite_config(result_root=tmp_path / "results", model_root=tmp_path / "models")
    manifest = write_suite_manifest(suite)

    with pytest.raises(FileNotFoundError, match="eval_tasks.csv"):
        validate_suite_artifacts(manifest)


def test_validate_suite_rejects_missing_methods(tmp_path: Path):
    suite = create_default_suite_config(result_root=tmp_path / "results", model_root=tmp_path / "models")
    manifest = write_suite_manifest(suite)
    pd.DataFrame(
        [{"suite_id": suite.suite_id, "task_id": "fixed", "split": "fixed", "weather_year": 2010, "start_day": 59, "uncertainty_scale": 0.0, "economic_scenario": "standard", "climate_constraint_scenario": "standard"}]
    ).to_csv(tmp_path / "results" / "eval_tasks.csv", index=False)
    pd.DataFrame(
        [{"suite_id": suite.suite_id, "algorithm": "ppo", "seed": 42, "run_name": "ppo_seed42", "model_path": "m", "vecnormalize_path": "v", "status": "completed", "train_steps": 1, "wall_time_seconds": 1.0, "best_eval_return": 0.0, "notes": ""}]
    ).to_csv(tmp_path / "results" / "runs.csv", index=False)
    pd.DataFrame(
        [{"suite_id": suite.suite_id, "algorithm": "ppo", "seed": 42, "run_name": "ppo_seed42", "task_id": "fixed", "split": "fixed", "weather_year": 2010, "start_day": 59, "uncertainty_scale": 0.0, "economic_scenario": "standard", "episode_return": 1.0, "EPI": 1.0, "revenue": 1.0, "heat_cost": 1.0, "co2_cost": 1.0, "elec_cost": 1.0, "temp_violation": 1.0, "co2_violation": 1.0, "rh_violation": 1.0, "twb_percent": 0.0, "trajectory_path": ""}]
    ).to_csv(tmp_path / "results" / "eval_raw.csv", index=False)

    with pytest.raises(ValueError, match="missing algorithms"):
        validate_suite_artifacts(manifest)
```

Create `tests/integrity/test_suite_figures.py` with:

```python
from pathlib import Path

import pandas as pd
import pytest

from gl_gym.experiments.suite_validation import require_current_suite_id


def test_require_current_suite_id_rejects_stale_summary(tmp_path: Path):
    summary = tmp_path / "method_summary.csv"
    pd.DataFrame([{"suite_id": "old_suite", "algorithm": "ppo", "split": "fixed"}]).to_csv(summary, index=False)

    with pytest.raises(ValueError, match="stale suite id"):
        require_current_suite_id(summary, "AgriControl_C_2026-06-30")
```

- [ ] **Step 2: Run validation tests and verify they fail**

Run:

```powershell
python -m pytest tests\integrity\test_suite_artifacts.py tests\integrity\test_suite_figures.py -q
```

Expected: fail because `suite_validation.py` does not exist.

- [ ] **Step 3: Implement validation module**

Create `src/gl_gym/experiments/suite_validation.py` with:

```python
from __future__ import annotations

from pathlib import Path

import pandas as pd

from gl_gym.experiments.suite_aggregation import assert_no_duplicate_run_task_rows
from gl_gym.experiments.suite_schema import REQUIRED_METHODS, load_suite_manifest


REQUIRED_SUITE_FILES = [
    "suite_manifest.json",
    "eval_tasks.csv",
    "runs.csv",
    "eval_raw.csv",
]


def require_current_suite_id(csv_path: str | Path, suite_id: str) -> None:
    df = pd.read_csv(csv_path)
    if "suite_id" not in df.columns:
        raise ValueError(f"{csv_path} has no suite_id column")
    actual = set(df["suite_id"].dropna().astype(str))
    if actual != {suite_id}:
        raise ValueError(f"stale suite id in {csv_path}: expected {suite_id}, found {sorted(actual)}")


def validate_suite_artifacts(manifest_path: str | Path) -> None:
    suite = load_suite_manifest(manifest_path)
    root = Path(suite.result_root)
    for filename in REQUIRED_SUITE_FILES:
        path = root / filename
        if not path.exists():
            raise FileNotFoundError(str(path))

    for filename in ["eval_tasks.csv", "runs.csv", "eval_raw.csv"]:
        require_current_suite_id(root / filename, suite.suite_id)

    eval_raw = pd.read_csv(root / "eval_raw.csv")
    assert_no_duplicate_run_task_rows(eval_raw)
    present_algorithms = set(eval_raw["algorithm"].unique())
    required_learning = set(REQUIRED_METHODS) - {"rule_based"}
    missing = sorted(required_learning - present_algorithms)
    if missing:
        raise ValueError(f"missing algorithms in eval_raw.csv: {missing}")

    present_seeds = sorted(eval_raw["seed"].unique().tolist())
    missing_seeds = sorted(set(suite.seeds) - set(present_seeds))
    if missing_seeds:
        raise ValueError(f"missing seeds in eval_raw.csv: {missing_seeds}")
```

- [ ] **Step 4: Add validation CLI**

Create `experiments/scripts/validate_suite_artifacts.py` with:

```python
#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "src"))

from gl_gym.experiments.suite_validation import validate_suite_artifacts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    args = parser.parse_args()
    validate_suite_artifacts(args.manifest)
    print("Suite artifacts validated")


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Add guarded figure generator**

Create `experiments/scripts/generate_suite_figures.py` with:

```python
#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "src"))

from gl_gym.experiments.suite_schema import REQUIRED_METHODS, load_suite_manifest
from gl_gym.experiments.suite_validation import require_current_suite_id


def plot_fixed_heldout(summary: pd.DataFrame, out_path: Path) -> None:
    metric = "episode_return_mean"
    data = summary[summary["split"].isin(["fixed", "heldout"])]
    if metric not in data.columns:
        raise ValueError(f"missing required metric column: {metric}")
    pivot = data.pivot(index="algorithm", columns="split", values=metric)
    missing = [method for method in REQUIRED_METHODS if method != "rule_based" and method not in pivot.index]
    if missing:
        raise ValueError(f"cannot generate figure; missing algorithms: {missing}")
    ax = pivot.plot(kind="bar", figsize=(8, 5))
    ax.set_ylabel("Episode return")
    ax.set_title("Fixed and held-out performance")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--summary")
    parser.add_argument("--out_dir", default="paper/figures")
    args = parser.parse_args()

    suite = load_suite_manifest(args.manifest)
    summary_path = Path(args.summary or Path(suite.result_root) / "method_summary.csv")
    require_current_suite_id(summary_path, suite.suite_id)
    summary = pd.read_csv(summary_path)
    plot_fixed_heldout(summary, Path(args.out_dir) / "suite_fixed_heldout.png")
    print(f"Wrote {Path(args.out_dir) / 'suite_fixed_heldout.png'}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 6: Run validation tests**

Run:

```powershell
python -m pytest tests\integrity\test_suite_artifacts.py tests\integrity\test_suite_figures.py -q
```

Expected: pass.

- [ ] **Step 7: Run existing anti-scaling integrity test**

Run:

```powershell
python -m pytest tests\integrity\test_no_result_scaling.py -q
```

Expected: pass.

- [ ] **Step 8: User-approved checkpoint**

If the user approves staging/commit:

```powershell
git add src\gl_gym\experiments\suite_validation.py experiments\scripts\validate_suite_artifacts.py experiments\scripts\generate_suite_figures.py tests\integrity\test_suite_artifacts.py tests\integrity\test_suite_figures.py
git commit -m "feat: add suite artifact validation and figure guards"
```

---

### Task 7: Documentation and paper-source guardrails

**Files:**
- Modify: `paper/README.md`
- Modify: `experiments/scripts/README_scripts.md`
- Create: `docs/superpowers/specs/2026-06-30-robust-experiment-redesign-design.md` is already complete and should not be rewritten in this task.

- [ ] **Step 1: Update paper README**

Append this to `paper/README.md`:

```markdown
## C-route manuscript source policy

The redesigned manuscript must use `artifacts/results/AgriControl_C_2026-06-30/` as its result source.

Do not cite summaries, figures, or raw rows from `artifacts/results/AgriControl/` for the C-route manuscript. That directory is archival and contains stale or incomplete experiment outputs.

Before updating tables or figures, run:

```powershell
python experiments\scripts\validate_suite_artifacts.py --manifest artifacts\results\AgriControl_C_2026-06-30\suite_manifest.json
python experiments\scripts\summarize_suite.py --manifest artifacts\results\AgriControl_C_2026-06-30\suite_manifest.json
python experiments\scripts\generate_suite_figures.py --manifest artifacts\results\AgriControl_C_2026-06-30\suite_manifest.json
```
```

- [ ] **Step 2: Update script README with full command flow**

Append this to `experiments/scripts/README_scripts.md`:

```markdown
## Robust suite command sequence

Create the suite:

```powershell
python experiments\scripts\create_experiment_suite.py --suite_id AgriControl_C_2026-06-30 --result_root artifacts\results\AgriControl_C_2026-06-30 --model_root artifacts\models\AgriControl_C_2026-06-30
```

Pilot training:

```powershell
python experiments\scripts\run_suite_training.py --manifest artifacts\results\AgriControl_C_2026-06-30\suite_manifest.json --algorithms ppo recurrentppo context_recurrentppo agri_metarl --seeds 42 123 --device cpu
```

Full training:

```powershell
python experiments\scripts\run_suite_training.py --manifest artifacts\results\AgriControl_C_2026-06-30\suite_manifest.json --algorithms ppo recurrentppo context_recurrentppo agri_metarl --seeds 42 123 456 789 1024 --device cpu
```

Evaluate, summarize, validate, and generate figures:

```powershell
python experiments\scripts\evaluate_suite.py --manifest artifacts\results\AgriControl_C_2026-06-30\suite_manifest.json --runs_csv artifacts\results\AgriControl_C_2026-06-30\runs.csv --tasks_csv artifacts\results\AgriControl_C_2026-06-30\eval_tasks.csv
python experiments\scripts\summarize_suite.py --manifest artifacts\results\AgriControl_C_2026-06-30\suite_manifest.json
python experiments\scripts\validate_suite_artifacts.py --manifest artifacts\results\AgriControl_C_2026-06-30\suite_manifest.json
python experiments\scripts\generate_suite_figures.py --manifest artifacts\results\AgriControl_C_2026-06-30\suite_manifest.json
```
```

- [ ] **Step 3: Verify docs contain the archival warning**

Run:

```powershell
rg -n "archival|AgriControl_C_2026-06-30|validate_suite_artifacts" paper\README.md experiments\scripts\README_scripts.md
```

Expected: both README files mention `AgriControl_C_2026-06-30`, and at least one line warns that old `AgriControl` results are archival.

- [ ] **Step 4: User-approved checkpoint**

If the user approves staging/commit:

```powershell
git add paper\README.md experiments\scripts\README_scripts.md
git commit -m "docs: document robust suite workflow"
```

---

### Task 8: Verification, pilot gate, and full-run gate

**Files:**
- No new source file required.
- Use all files created in Tasks 1-7.

- [ ] **Step 1: Run focused suite tests**

Run:

```powershell
python -m pytest tests\experiments\test_suite_schema.py tests\experiments\test_suite_tasks.py tests\experiments\test_suite_evaluation.py tests\experiments\test_suite_aggregation.py tests\integrity\test_suite_artifacts.py tests\integrity\test_suite_figures.py -q
```

Expected: all pass.

- [ ] **Step 2: Run Agri-MetaRL and Context-RecurrentPPO regression tests**

Run:

```powershell
python -m pytest tests\agri_metarl -q
```

Expected: all pass. Known warnings from third-party libraries are acceptable if they match prior warnings.

- [ ] **Step 3: Run full local test suite**

Run:

```powershell
python -m pytest -q
```

Expected: all pass.

- [ ] **Step 4: Run compile and diff hygiene checks**

Run:

```powershell
python -m compileall -q src tests experiments
git diff --check
rg -n "^(<<<<<<<|=======|>>>>>>>)" src tests configs experiments docs paper
```

Expected:

- `compileall` exits 0.
- `git diff --check` exits 0, except existing CRLF warnings if they are unchanged from earlier repository state.
- conflict marker scan prints no matches.

- [ ] **Step 5: Create or refresh the suite manifest**

Run:

```powershell
python experiments\scripts\create_experiment_suite.py --suite_id AgriControl_C_2026-06-30 --result_root artifacts\results\AgriControl_C_2026-06-30 --model_root artifacts\models\AgriControl_C_2026-06-30
```

Expected: `suite_manifest.json` and `eval_tasks.csv` exist in `artifacts\results\AgriControl_C_2026-06-30\`.

- [ ] **Step 6: Pilot run gate**

Run pilot only after the user confirms compute time is acceptable:

```powershell
python experiments\scripts\run_suite_training.py --manifest artifacts\results\AgriControl_C_2026-06-30\suite_manifest.json --algorithms ppo recurrentppo context_recurrentppo agri_metarl --seeds 42 123 --device cpu
python experiments\scripts\evaluate_suite.py --manifest artifacts\results\AgriControl_C_2026-06-30\suite_manifest.json --runs_csv artifacts\results\AgriControl_C_2026-06-30\runs.csv --tasks_csv artifacts\results\AgriControl_C_2026-06-30\eval_tasks.csv --algorithms ppo recurrentppo context_recurrentppo agri_metarl --seeds 42 123
python experiments\scripts\summarize_suite.py --manifest artifacts\results\AgriControl_C_2026-06-30\suite_manifest.json
```

Pilot passes if:

- every training run finishes without non-finite loss;
- every evaluated run-task produces one row;
- no deterministic duplicate rows appear;
- `method_summary.csv` has fixed, heldout, uncertainty, and economic splits;
- Context-RecurrentPPO and Agri-MetaRL v2 both load and predict through the suite evaluator.

- [ ] **Step 7: Full run gate**

Run full experiment only after pilot passes and the user confirms:

```powershell
python experiments\scripts\run_suite_training.py --manifest artifacts\results\AgriControl_C_2026-06-30\suite_manifest.json --algorithms ppo recurrentppo context_recurrentppo agri_metarl --seeds 42 123 456 789 1024 --device cpu
python experiments\scripts\evaluate_suite.py --manifest artifacts\results\AgriControl_C_2026-06-30\suite_manifest.json --runs_csv artifacts\results\AgriControl_C_2026-06-30\runs.csv --tasks_csv artifacts\results\AgriControl_C_2026-06-30\eval_tasks.csv
python experiments\scripts\summarize_suite.py --manifest artifacts\results\AgriControl_C_2026-06-30\suite_manifest.json
python experiments\scripts\validate_suite_artifacts.py --manifest artifacts\results\AgriControl_C_2026-06-30\suite_manifest.json
python experiments\scripts\generate_suite_figures.py --manifest artifacts\results\AgriControl_C_2026-06-30\suite_manifest.json
```

Full run passes if:

- all four learning algorithms have five seeds;
- all expected task ids are present in `eval_raw.csv`;
- validation exits 0;
- figures are regenerated from the C-route suite only;
- no manuscript claim is updated until summaries and effect sizes have been inspected.

- [ ] **Step 8: User-approved checkpoint**

If the user approves staging/commit:

```powershell
git add src\gl_gym\experiments experiments\scripts tests\experiments tests\integrity paper\README.md
git commit -m "feat: implement robust experiment suite pipeline"
```
