from pathlib import Path

import pandas as pd
import pytest

from gl_gym.experiments.suite_schema import create_default_suite_config, write_suite_manifest
from gl_gym.experiments.suite_validation import validate_suite_artifacts


def _write_current_suite_inputs(result_root: Path, suite_id: str) -> None:
    pd.DataFrame(
        [
            {
                "suite_id": suite_id,
                "task_id": "fixed_2010_d59_u0p00_standard",
                "split": "fixed",
                "weather_year": 2010,
                "start_day": 59,
                "uncertainty_scale": 0.0,
                "economic_scenario": "standard",
                "climate_constraint_scenario": "standard",
            }
        ]
    ).to_csv(result_root / "eval_tasks.csv", index=False)
    pd.DataFrame(
        [
            {
                "suite_id": suite_id,
                "algorithm": "ppo",
                "seed": 42,
                "run_name": "ppo_seed42",
                "model_path": "model.zip",
                "vecnormalize_path": "vec.pkl",
                "status": "completed",
                "train_steps": 1,
                "wall_time_seconds": 1.0,
                "best_eval_return": 1.0,
                "notes": "",
            }
        ]
    ).to_csv(result_root / "runs.csv", index=False)


def _eval_row(
    suite_id: str = "suite",
    algorithm: str = "ppo",
    seed: int = 42,
    run_name: str | None = None,
    task_id: str = "fixed_2010_d59_u0p00_standard",
) -> dict[str, object]:
    return {
        "suite_id": suite_id,
        "algorithm": algorithm,
        "seed": seed,
        "run_name": run_name or f"{algorithm}_seed{seed}",
        "task_id": task_id,
        "split": "fixed",
        "weather_year": 2010,
        "start_day": 59,
        "uncertainty_scale": 0.0,
        "economic_scenario": "standard",
        "climate_constraint_scenario": "standard",
        "episode_return": 1.0,
        "EPI": 1.0,
        "revenue": 1.0,
        "heat_cost": 0.0,
        "co2_cost": 0.0,
        "elec_cost": 0.0,
        "temp_violation": 0.0,
        "co2_violation": 0.0,
        "rh_violation": 0.0,
    }


def test_validate_suite_artifacts_reports_missing_required_file(tmp_path: Path):
    suite = create_default_suite_config(
        suite_id="suite",
        result_root=tmp_path / "results",
        model_root=tmp_path / "models",
    )
    manifest = write_suite_manifest(suite)

    with pytest.raises(FileNotFoundError, match="eval_tasks.csv"):
        validate_suite_artifacts(manifest)


def test_validate_suite_artifacts_rejects_missing_algorithms(tmp_path: Path):
    suite = create_default_suite_config(
        suite_id="suite",
        result_root=tmp_path / "results",
        model_root=tmp_path / "models",
    )
    manifest = write_suite_manifest(suite)
    result_root = Path(suite.result_root)
    _write_current_suite_inputs(result_root, suite.suite_id)
    rows = [_eval_row(suite_id=suite.suite_id, algorithm="ppo", seed=seed) for seed in suite.seeds]
    pd.DataFrame(rows).to_csv(result_root / "eval_raw.csv", index=False)

    with pytest.raises(ValueError, match="missing algorithms"):
        validate_suite_artifacts(manifest)


def test_validate_suite_artifacts_rejects_duplicate_deterministic_eval_rows(tmp_path: Path):
    suite = create_default_suite_config(
        suite_id="suite",
        result_root=tmp_path / "results",
        model_root=tmp_path / "models",
    )
    manifest = write_suite_manifest(suite)
    result_root = Path(suite.result_root)
    _write_current_suite_inputs(result_root, suite.suite_id)
    duplicate = _eval_row(suite_id=suite.suite_id)
    rows = [
        *(
            _eval_row(suite_id=suite.suite_id, algorithm=algorithm, seed=seed)
            for algorithm in ["ppo", "recurrentppo", "context_recurrentppo", "agri_metarl"]
            for seed in suite.seeds
        ),
        duplicate,
    ]
    pd.DataFrame(rows).to_csv(result_root / "eval_raw.csv", index=False)

    with pytest.raises(ValueError, match="duplicate deterministic evaluation rows"):
        validate_suite_artifacts(manifest)


def test_validate_suite_artifacts_rejects_missing_seed_per_algorithm(tmp_path: Path):
    suite = create_default_suite_config(
        suite_id="suite",
        result_root=tmp_path / "results",
        model_root=tmp_path / "models",
    )
    manifest = write_suite_manifest(suite)
    result_root = Path(suite.result_root)
    _write_current_suite_inputs(result_root, suite.suite_id)
    rows = [
        _eval_row(suite_id=suite.suite_id, algorithm=algorithm, seed=seed)
        for algorithm in ["ppo", "recurrentppo", "context_recurrentppo", "agri_metarl"]
        for seed in suite.seeds
        if not (algorithm == "agri_metarl" and seed == suite.seeds[-1])
    ]
    pd.DataFrame(rows).to_csv(result_root / "eval_raw.csv", index=False)

    with pytest.raises(ValueError, match="missing seeds"):
        validate_suite_artifacts(manifest)
