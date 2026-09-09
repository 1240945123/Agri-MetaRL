"""Schema helpers for robust experiment suites."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass, fields, is_dataclass
from pathlib import Path
from typing import Iterable


REQUIRED_METHODS = (
    "ppo",
    "recurrentppo",
    "context_recurrentppo",
    "agri_metarl",
    "rule_based",
)


@dataclass
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


@dataclass
class EvaluationTaskRecord:
    suite_id: str
    task_id: str
    split: str
    weather_year: int
    start_day: int
    uncertainty_scale: float
    economic_scenario: str
    climate_constraint_scenario: str


@dataclass
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
    result_root: str | Path | None = None,
    model_root: str | Path | None = None,
    branch: str = "unknown",
    dirty: bool = True,
    notes: str = "C-route robust task-distribution experiment suite.",
) -> ExperimentSuiteConfig:
    """Create the default C-route robust experiment suite configuration."""

    result_root = result_root or Path("artifacts") / "results" / suite_id
    model_root = model_root or Path("artifacts") / "models" / suite_id

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
        notes=notes,
    )


def suite_manifest_path(suite: ExperimentSuiteConfig) -> Path:
    """Return the canonical manifest path for an experiment suite."""

    return Path(suite.result_root) / "suite_manifest.json"


def write_suite_manifest(suite: ExperimentSuiteConfig, path: str | Path | None = None) -> Path:
    """Write an experiment suite manifest as stable, indented JSON."""

    manifest_path = Path(path) if path is not None else suite_manifest_path(suite)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(asdict(suite), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest_path


def load_suite_manifest(path: str | Path) -> ExperimentSuiteConfig:
    """Load an experiment suite manifest from JSON."""

    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return ExperimentSuiteConfig(**data)


def write_records_csv(records: Iterable[object], path: str | Path) -> Path:
    """Write dataclass records to CSV with columns in dataclass field order."""

    rows = list(records)
    if not rows:
        raise ValueError("write_records_csv requires at least one record")

    first = rows[0]
    if not is_dataclass(first):
        raise TypeError("write_records_csv records must be dataclass instances")

    columns = [field.name for field in fields(first)]
    csv_path = Path(path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))

    return csv_path
