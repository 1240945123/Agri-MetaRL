import subprocess
import sys

import pandas as pd
import pytest

from gl_gym.experiments.suite_aggregation import (
    aggregate_seed_first,
    assert_no_duplicate_run_task_rows,
    cohens_d,
    paired_effect_table,
    write_summary_files,
)
from gl_gym.experiments.suite_schema import create_default_suite_config, write_suite_manifest


def test_duplicate_run_task_rows_are_rejected():
    df = pd.DataFrame(
        [
            {"algorithm": "ppo", "seed": 42, "run_name": "ppo_seed42", "task_id": "fixed"},
            {"algorithm": "ppo", "seed": 42, "run_name": "ppo_seed42", "task_id": "fixed"},
        ]
    )

    with pytest.raises(ValueError, match="duplicate deterministic evaluation rows"):
        assert_no_duplicate_run_task_rows(df)


def test_missing_duplicate_check_columns_are_rejected():
    df = pd.DataFrame([{"algorithm": "ppo", "seed": 42}])

    with pytest.raises(ValueError, match="missing duplicate-check columns"):
        assert_no_duplicate_run_task_rows(df)


def test_aggregate_seed_first_averages_tasks_before_seed_statistics():
    df = pd.DataFrame(
        [
            {"suite_id": "suite", "algorithm": "ppo", "seed": 1, "run_name": "ppo_1", "task_id": "a", "split": "heldout", "episode_return": 10.0, "EPI": 1.0},
            {"suite_id": "suite", "algorithm": "ppo", "seed": 1, "run_name": "ppo_1", "task_id": "b", "split": "heldout", "episode_return": 20.0, "EPI": 3.0},
            {"suite_id": "suite", "algorithm": "ppo", "seed": 2, "run_name": "ppo_2", "task_id": "a", "split": "heldout", "episode_return": 30.0, "EPI": 5.0},
            {"suite_id": "suite", "algorithm": "ppo", "seed": 2, "run_name": "ppo_2", "task_id": "b", "split": "heldout", "episode_return": 40.0, "EPI": 7.0},
        ]
    )

    summary = aggregate_seed_first(df, metrics=["episode_return", "EPI"])

    assert summary.loc[0, "suite_id"] == "suite"
    assert summary.loc[0, "episode_return_mean"] == 25.0
    assert summary.loc[0, "EPI_mean"] == 4.0
    assert summary.loc[0, "episode_return_std"] == pytest.approx(14.1421356237)
    assert summary.loc[0, "n_seeds"] == 2


def test_aggregate_seed_first_single_seed_std_is_nan():
    df = pd.DataFrame(
        [
            {"algorithm": "ppo", "seed": 1, "run_name": "ppo_1", "task_id": "a", "split": "heldout", "episode_return": 10.0},
            {"algorithm": "ppo", "seed": 1, "run_name": "ppo_1", "task_id": "b", "split": "heldout", "episode_return": 20.0},
        ]
    )

    summary = aggregate_seed_first(df, metrics=["episode_return"])

    assert summary.loc[0, "episode_return_mean"] == 15.0
    assert pd.isna(summary.loc[0, "episode_return_std"])
    assert summary.loc[0, "n_seeds"] == 1


def test_cohens_d_returns_signed_effect_size():
    assert round(cohens_d([2.0, 2.0, 2.0], [1.0, 1.0, 1.0]), 3) == 0.0
    assert cohens_d([2.0, 3.0, 4.0], [1.0, 2.0, 3.0]) > 0


def test_paired_effect_table_uses_only_common_seeds():
    df = pd.DataFrame(
        [
            {"suite_id": "suite", "algorithm": "ppo", "seed": 1, "split": "heldout", "episode_return": 10.0},
            {"suite_id": "suite", "algorithm": "ppo", "seed": 2, "split": "heldout", "episode_return": 20.0},
            {"suite_id": "suite", "algorithm": "ppo", "seed": 3, "split": "heldout", "episode_return": 300.0},
            {"suite_id": "suite", "algorithm": "agri_metarl", "seed": 1, "split": "heldout", "episode_return": 1.0},
            {"suite_id": "suite", "algorithm": "agri_metarl", "seed": 2, "split": "heldout", "episode_return": 2.0},
        ]
    )

    effects = paired_effect_table(df)

    assert effects.loc[0, "suite_id"] == "suite"
    assert effects.loc[0, "n_pairs"] == 2
    means = {
        effects.loc[0, "method_a"]: effects.loc[0, "mean_a"],
        effects.loc[0, "method_b"]: effects.loc[0, "mean_b"],
    }
    assert means == {"agri_metarl": 1.5, "ppo": 15.0}


def test_paired_effect_table_keeps_suite_and_split_groups_separate():
    df = pd.DataFrame(
        [
            {"suite_id": "suite_a", "algorithm": "ppo", "seed": 1, "split": "heldout", "episode_return": 10.0},
            {"suite_id": "suite_a", "algorithm": "ppo", "seed": 2, "split": "heldout", "episode_return": 20.0},
            {"suite_id": "suite_a", "algorithm": "agri_metarl", "seed": 1, "split": "heldout", "episode_return": 1.0},
            {"suite_id": "suite_a", "algorithm": "agri_metarl", "seed": 2, "split": "heldout", "episode_return": 2.0},
            {"suite_id": "suite_b", "algorithm": "ppo", "seed": 1, "split": "fixed", "episode_return": 100.0},
            {"suite_id": "suite_b", "algorithm": "ppo", "seed": 2, "split": "fixed", "episode_return": 200.0},
            {"suite_id": "suite_b", "algorithm": "agri_metarl", "seed": 1, "split": "fixed", "episode_return": 10.0},
            {"suite_id": "suite_b", "algorithm": "agri_metarl", "seed": 2, "split": "fixed", "episode_return": 20.0},
        ]
    )

    effects = paired_effect_table(df)

    assert len(effects) == 2
    assert set(zip(effects["suite_id"], effects["split"], strict=True)) == {
        ("suite_a", "heldout"),
        ("suite_b", "fixed"),
    }


def _eval_raw_rows() -> list[dict[str, object]]:
    return [
        {
            "suite_id": "suite",
            "algorithm": algorithm,
            "seed": seed,
            "run_name": f"{algorithm}_{seed}",
            "task_id": "task_1",
            "split": "heldout",
            "climate_constraint_scenario": "standard",
            "episode_return": float(seed + offset),
            "EPI": float(seed),
            "revenue": 1.0,
            "heat_cost": 2.0,
            "co2_cost": 3.0,
            "elec_cost": 4.0,
            "temp_violation": 0.0,
            "co2_violation": 0.0,
            "rh_violation": 0.0,
        }
        for algorithm, offset in (("ppo", 0), ("agri_metarl", 10))
        for seed in (1, 2)
    ]


def test_write_summary_files_writes_suite_outputs(tmp_path):
    eval_raw_path = tmp_path / "eval_raw.csv"
    pd.DataFrame(_eval_raw_rows()).to_csv(eval_raw_path, index=False)

    paths = write_summary_files(eval_raw_path, tmp_path)

    assert paths["method_summary"] == tmp_path / "method_summary.csv"
    assert paths["stat_tests"] == tmp_path / "stat_tests.csv"
    assert paths["diagnostics"] == tmp_path / "diagnostics.csv"
    for path in paths.values():
        assert path.is_file()

    method_summary = pd.read_csv(paths["method_summary"])
    stat_tests = pd.read_csv(paths["stat_tests"])
    diagnostics = pd.read_csv(paths["diagnostics"])
    assert "suite_id" in method_summary.columns
    assert "suite_id" in stat_tests.columns
    assert diagnostics.loc[0, "diagnostic"] == "not_collected"
    diagnostic_text = diagnostics.to_csv(index=False)
    assert "TODO" not in diagnostic_text
    assert "TBD" not in diagnostic_text


def test_summarize_suite_cli_writes_outputs_and_prints_paths(tmp_path):
    result_root = tmp_path / "results"
    suite = create_default_suite_config(suite_id="suite", result_root=result_root)
    manifest_path = write_suite_manifest(suite)
    eval_raw_path = tmp_path / "eval_raw.csv"
    pd.DataFrame(_eval_raw_rows()).to_csv(eval_raw_path, index=False)

    completed = subprocess.run(
        [
            sys.executable,
            "experiments/scripts/summarize_suite.py",
            "--manifest",
            str(manifest_path),
            "--eval_raw",
            str(eval_raw_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "Wrote" in completed.stdout
    assert (result_root / "method_summary.csv").is_file()
    assert (result_root / "stat_tests.csv").is_file()
    assert (result_root / "diagnostics.csv").is_file()
