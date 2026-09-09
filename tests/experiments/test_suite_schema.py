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
