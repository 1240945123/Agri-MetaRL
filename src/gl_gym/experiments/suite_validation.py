"""Validation guardrails for robust experiment suite artifacts."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from gl_gym.experiments.suite_aggregation import assert_no_duplicate_run_task_rows
from gl_gym.experiments.suite_schema import REQUIRED_METHODS, load_suite_manifest


REQUIRED_SUITE_FILES = (
    "suite_manifest.json",
    "eval_tasks.csv",
    "runs.csv",
    "eval_raw.csv",
)


def require_current_suite_id(csv_path: str | Path, suite_id: str) -> pd.DataFrame:
    """Read a CSV and reject artifacts from another experiment suite."""

    path = Path(csv_path)
    data = pd.read_csv(path)
    if "suite_id" not in data.columns:
        raise ValueError(f"stale suite id in {path}: missing suite_id column")

    suite_ids = data["suite_id"]
    invalid_mask = suite_ids.isna() | suite_ids.astype(str).str.strip().eq("")
    if invalid_mask.any():
        raise ValueError(f"invalid suite_id in {path}: null or blank suite_id rows present")

    actual = set(suite_ids.astype(str).str.strip())
    if actual != {str(suite_id)}:
        raise ValueError(f"stale suite id in {path}: expected {suite_id}, found {sorted(actual)}")
    return data


def validate_suite_artifacts(manifest_path: str | Path) -> None:
    """Validate that a suite result directory is complete and internally current."""

    suite = load_suite_manifest(manifest_path)
    result_root = Path(suite.result_root)

    for filename in REQUIRED_SUITE_FILES:
        path = result_root / filename
        if not path.is_file():
            raise FileNotFoundError(f"missing required suite artifact: {path}")

    require_current_suite_id(result_root / "eval_tasks.csv", suite.suite_id)
    require_current_suite_id(result_root / "runs.csv", suite.suite_id)
    eval_raw = require_current_suite_id(result_root / "eval_raw.csv", suite.suite_id)

    assert_no_duplicate_run_task_rows(eval_raw)

    expected_algorithms = set(REQUIRED_METHODS) - {"rule_based"}
    present_algorithms = set(eval_raw["algorithm"].dropna().astype(str))
    missing_algorithms = sorted(expected_algorithms - present_algorithms)
    if missing_algorithms:
        raise ValueError(f"missing algorithms in eval_raw.csv: {missing_algorithms}")

    expected_seeds = set(suite.seeds)
    missing_seeds_by_algorithm: dict[str, list[int]] = {}
    for algorithm in sorted(expected_algorithms):
        algorithm_rows = eval_raw[eval_raw["algorithm"].astype(str) == algorithm]
        present_seeds = set(algorithm_rows["seed"].dropna().astype(int))
        missing_seeds = sorted(expected_seeds - present_seeds)
        if missing_seeds:
            missing_seeds_by_algorithm[algorithm] = missing_seeds

    if missing_seeds_by_algorithm:
        raise ValueError(f"missing seeds in eval_raw.csv: {missing_seeds_by_algorithm}")
