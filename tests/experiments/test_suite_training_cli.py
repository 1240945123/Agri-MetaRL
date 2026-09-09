from pathlib import Path
import importlib.util
import sys

import pandas as pd
import pytest

from gl_gym.experiments.suite_schema import (
    RunRecord,
    create_default_suite_config,
    write_records_csv,
    write_suite_manifest,
)


def load_run_suite_training_module():
    script_path = Path(__file__).resolve().parents[2] / "experiments" / "scripts" / "run_suite_training.py"
    spec = importlib.util.spec_from_file_location("run_suite_training", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


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


def test_train_one_dry_run_records_experiment_manager_layout(tmp_path: Path):
    module = load_run_suite_training_module()
    suite = create_default_suite_config(
        result_root=tmp_path / "results",
    )

    record = module.train_one(suite, "ppo", 42, "cpu", dry_run=True)

    assert record.status == "dry_run"
    assert record.train_steps == 0
    assert "ppo/deterministic/models/ppo_seed42/best_model.zip" in record.model_path.replace("\\", "/")
    assert (
        "ppo/deterministic/envs/ppo_seed42/best_vecnormalize.pkl"
        in record.vecnormalize_path.replace("\\", "/")
    )


def test_train_one_rejects_custom_model_root(tmp_path: Path):
    module = load_run_suite_training_module()
    suite = create_default_suite_config(
        result_root=tmp_path / "results",
        model_root=tmp_path / "custom_models",
    )

    with pytest.raises(ValueError, match="custom model_root.*not supported"):
        module.train_one(suite, "ppo", 42, "cpu", dry_run=True)


def test_train_one_uses_train_timesteps_override_for_real_training(tmp_path: Path, monkeypatch):
    module = load_run_suite_training_module()
    suite = create_default_suite_config(
        result_root=tmp_path / "results",
    )
    captured = {}

    def fake_load_env_params(env_id, config_path):
        assert env_id == "TomatoEnv"
        return {}, {}

    def fake_load_model_hyperparams(algorithm, env_id):
        assert algorithm == "ppo"
        assert env_id == "TomatoEnv"
        return {"total_timesteps": 123}

    class FakeExperimentManager:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        def run_experiment(self):
            captured["ran"] = True

    monkeypatch.setattr(module, "load_env_params", fake_load_env_params)
    monkeypatch.setattr(module, "load_model_hyperparams", fake_load_model_hyperparams)
    monkeypatch.setattr(module, "ExperimentManager", FakeExperimentManager)

    record = module.train_one(
        suite,
        "ppo",
        42,
        "cpu",
        dry_run=False,
        train_timesteps=300_000,
    )

    assert captured["hyperparameters"]["total_timesteps"] == 300_000
    assert captured["run_name"] == "ppo_seed42"
    assert captured["ran"] is True
    assert record.status == "completed"
    assert record.train_steps == 300_000


def test_train_one_applies_smoke_training_overrides(tmp_path: Path, monkeypatch):
    module = load_run_suite_training_module()
    suite = create_default_suite_config(
        result_root=tmp_path / "results",
    )
    captured = {}

    def fake_load_env_params(env_id, config_path):
        return {}, {}

    def fake_load_model_hyperparams(algorithm, env_id):
        return {
            "total_timesteps": 2_000_000,
            "n_envs": 8,
            "n_steps": 2048,
            "batch_size": 512,
            "n_epochs": 8,
        }

    class FakeExperimentManager:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        def run_experiment(self):
            captured["ran"] = True

    monkeypatch.setattr(module, "load_env_params", fake_load_env_params)
    monkeypatch.setattr(module, "load_model_hyperparams", fake_load_model_hyperparams)
    monkeypatch.setattr(module, "ExperimentManager", FakeExperimentManager)

    record = module.train_one(
        suite,
        "recurrentppo",
        42,
        "cpu",
        dry_run=False,
        train_timesteps=64,
        n_envs=1,
        n_steps=64,
        batch_size=64,
        n_epochs=1,
        n_evals=1,
    )

    assert captured["hyperparameters"]["total_timesteps"] == 64
    assert captured["hyperparameters"]["n_envs"] == 1
    assert captured["hyperparameters"]["n_steps"] == 64
    assert captured["hyperparameters"]["batch_size"] == 64
    assert captured["hyperparameters"]["n_epochs"] == 1
    assert captured["n_evals"] == 1
    assert captured["ran"] is True
    assert record.status == "completed"


def test_experiment_manager_artifact_run_name_prefers_explicit_run_name():
    from gl_gym.RL.experiment_manager import ExperimentManager

    manager = ExperimentManager.__new__(ExperimentManager)
    manager.run_name = "ppo_seed42"
    manager.run = type("Run", (), {"name": "dummy-disabled-name"})()

    assert manager.artifact_run_name() == "ppo_seed42"


def test_run_training_suite_writes_registry_after_each_run(tmp_path: Path, monkeypatch):
    module = load_run_suite_training_module()
    suite = create_default_suite_config(result_root=tmp_path / "results")
    writes = []

    def fake_write_records_csv(records, path):
        writes.append([(record.algorithm, record.seed, record.status) for record in records])
        return Path(path)

    def fake_train_one(suite, algorithm, seed, **kwargs):
        return RunRecord(
            suite_id=suite.suite_id,
            algorithm=algorithm,
            seed=seed,
            run_name=f"{algorithm}_seed{seed}",
            model_path="model.zip",
            vecnormalize_path="vec.pkl",
            status="completed",
            train_steps=1,
            wall_time_seconds=0.1,
            best_eval_return=float("nan"),
            notes="ok",
        )

    monkeypatch.setattr(module, "write_records_csv", fake_write_records_csv)
    monkeypatch.setattr(module, "train_one", fake_train_one)

    records = module.run_training_suite(
        suite,
        algorithms=["ppo", "recurrentppo"],
        seeds=[42],
        device="cpu",
        dry_run=False,
        registry_path=tmp_path / "results" / "runs.csv",
    )

    assert len(records) == 2
    assert writes == [
        [("ppo", 42, "completed")],
        [("ppo", 42, "completed"), ("recurrentppo", 42, "completed")],
    ]


def test_run_training_suite_records_failed_run_and_continues(tmp_path: Path, monkeypatch):
    module = load_run_suite_training_module()
    suite = create_default_suite_config(result_root=tmp_path / "results")

    def fake_train_one(suite, algorithm, seed, **kwargs):
        if algorithm == "ppo":
            raise RuntimeError("boom")
        return RunRecord(
            suite_id=suite.suite_id,
            algorithm=algorithm,
            seed=seed,
            run_name=f"{algorithm}_seed{seed}",
            model_path="model.zip",
            vecnormalize_path="vec.pkl",
            status="completed",
            train_steps=1,
            wall_time_seconds=0.1,
            best_eval_return=float("nan"),
            notes="ok",
        )

    monkeypatch.setattr(module, "train_one", fake_train_one)

    records = module.run_training_suite(
        suite,
        algorithms=["ppo", "recurrentppo"],
        seeds=[42],
        device="cpu",
        dry_run=False,
        registry_path=tmp_path / "results" / "runs.csv",
    )

    assert [(record.algorithm, record.status) for record in records] == [
        ("ppo", "failed"),
        ("recurrentppo", "completed"),
    ]
    assert "boom" in records[0].notes
    assert (tmp_path / "results" / "runs.csv").exists()


def test_run_training_suite_skips_completed_runs_with_artifacts(tmp_path: Path, monkeypatch):
    module = load_run_suite_training_module()
    suite = create_default_suite_config(result_root=tmp_path / "results")
    model_path = tmp_path / "best_model.zip"
    vec_path = tmp_path / "best_vecnormalize.pkl"
    model_path.write_text("model", encoding="utf-8")
    vec_path.write_text("vec", encoding="utf-8")
    existing = RunRecord(
        suite_id=suite.suite_id,
        algorithm="ppo",
        seed=42,
        run_name="ppo_seed42",
        model_path=str(model_path),
        vecnormalize_path=str(vec_path),
        status="completed",
        train_steps=suite.train_timesteps,
        wall_time_seconds=1.0,
        best_eval_return=float("nan"),
        notes="previous run",
    )
    write_records_csv([existing], tmp_path / "results" / "runs.csv")

    def fail_if_called(*args, **kwargs):
        raise AssertionError("completed run should have been skipped")

    monkeypatch.setattr(module, "train_one", fail_if_called)

    records = module.run_training_suite(
        suite,
        algorithms=["ppo"],
        seeds=[42],
        device="cpu",
        dry_run=False,
        registry_path=tmp_path / "results" / "runs.csv",
        skip_completed=True,
    )

    assert len(records) == 1
    assert records[0].status == "completed"
    assert "previous run" in records[0].notes


def test_run_training_suite_retrains_short_smoke_run_when_target_steps_are_higher(
    tmp_path: Path, monkeypatch
):
    module = load_run_suite_training_module()
    suite = create_default_suite_config(result_root=tmp_path / "results")
    model_path = tmp_path / "best_model.zip"
    vec_path = tmp_path / "best_vecnormalize.pkl"
    model_path.write_text("model", encoding="utf-8")
    vec_path.write_text("vec", encoding="utf-8")
    existing = RunRecord(
        suite_id=suite.suite_id,
        algorithm="ppo",
        seed=42,
        run_name="ppo_seed42",
        model_path=str(model_path),
        vecnormalize_path=str(vec_path),
        status="completed",
        train_steps=64,
        wall_time_seconds=1.0,
        best_eval_return=float("nan"),
        notes="short smoke run",
    )
    write_records_csv([existing], tmp_path / "results" / "runs.csv")

    def fake_train_one(suite, algorithm, seed, **kwargs):
        return RunRecord(
            suite_id=suite.suite_id,
            algorithm=algorithm,
            seed=seed,
            run_name=f"{algorithm}_seed{seed}",
            model_path=str(model_path),
            vecnormalize_path=str(vec_path),
            status="completed",
            train_steps=2_000_000,
            wall_time_seconds=10.0,
            best_eval_return=float("nan"),
            notes="full run",
        )

    monkeypatch.setattr(module, "train_one", fake_train_one)

    records = module.run_training_suite(
        suite,
        algorithms=["ppo"],
        seeds=[42],
        device="cpu",
        dry_run=False,
        registry_path=tmp_path / "results" / "runs.csv",
        skip_completed=True,
    )

    assert records[0].train_steps == 2_000_000
    assert records[0].notes == "full run"


def test_train_one_resumes_from_last_checkpoint_when_requested(tmp_path: Path, monkeypatch):
    module = load_run_suite_training_module()
    suite = create_default_suite_config(result_root=tmp_path / "results")
    captured = {}
    last_model = tmp_path / "last_model.zip"
    last_vec = tmp_path / "last_vecnormalize.pkl"
    last_model.write_text("model", encoding="utf-8")
    last_vec.write_text("vec", encoding="utf-8")
    missing_best_model = tmp_path / "missing_best_model.zip"
    missing_best_vec = tmp_path / "missing_best_vecnormalize.pkl"

    monkeypatch.setattr(
        module,
        "run_artifact_paths",
        lambda suite, algorithm, run_name: (missing_best_model, missing_best_vec),
    )
    monkeypatch.setattr(
        module,
        "run_last_artifact_paths",
        lambda suite, algorithm, run_name: (last_model, last_vec),
    )
    monkeypatch.setattr(module, "checkpoint_num_timesteps", lambda path: 64)

    def fake_load_env_params(env_id, config_path):
        return {}, {}

    def fake_load_model_hyperparams(algorithm, env_id):
        return {"total_timesteps": 2_000_000, "n_envs": 8}

    class FakeExperimentManager:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        def run_experiment(self):
            captured["ran"] = True

    monkeypatch.setattr(module, "load_env_params", fake_load_env_params)
    monkeypatch.setattr(module, "load_model_hyperparams", fake_load_model_hyperparams)
    monkeypatch.setattr(module, "ExperimentManager", FakeExperimentManager)

    record = module.train_one(
        suite,
        "context_recurrentppo",
        42,
        "cpu",
        dry_run=False,
        train_timesteps=2_000_000,
        resume_partial=True,
    )

    assert captured["continue_model_path"] == str(last_model)
    assert captured["continue_vecnormalize_path"] == str(last_vec)
    assert captured["ran"] is True
    assert record.status == "completed"


def test_train_one_resumes_from_highest_step_checkpoint_and_trains_remaining(
    tmp_path: Path, monkeypatch
):
    module = load_run_suite_training_module()
    suite = create_default_suite_config(result_root=tmp_path / "results")
    captured = {}
    best_model = tmp_path / "best_model.zip"
    best_vec = tmp_path / "best_vecnormalize.pkl"
    last_model = tmp_path / "last_model.zip"
    last_vec = tmp_path / "last_vecnormalize.pkl"
    for path in [best_model, best_vec, last_model, last_vec]:
        path.write_text(path.name, encoding="utf-8")

    monkeypatch.setattr(
        module,
        "run_artifact_paths",
        lambda suite, algorithm, run_name: (best_model, best_vec),
    )
    monkeypatch.setattr(
        module,
        "run_last_artifact_paths",
        lambda suite, algorithm, run_name: (last_model, last_vec),
    )
    monkeypatch.setattr(
        module,
        "checkpoint_num_timesteps",
        lambda path: 800_000 if Path(path).name == "best_model.zip" else 64,
    )

    def fake_load_env_params(env_id, config_path):
        return {}, {}

    def fake_load_model_hyperparams(algorithm, env_id):
        return {"total_timesteps": 2_000_000, "n_envs": 8}

    class FakeExperimentManager:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        def run_experiment(self):
            captured["ran"] = True

    monkeypatch.setattr(module, "load_env_params", fake_load_env_params)
    monkeypatch.setattr(module, "load_model_hyperparams", fake_load_model_hyperparams)
    monkeypatch.setattr(module, "ExperimentManager", FakeExperimentManager)

    record = module.train_one(
        suite,
        "context_recurrentppo",
        42,
        "cpu",
        dry_run=False,
        train_timesteps=2_000_000,
        resume_partial=True,
    )

    assert captured["continue_model_path"] == str(best_model)
    assert captured["continue_vecnormalize_path"] == str(best_vec)
    assert captured["hyperparameters"]["total_timesteps"] == 1_200_000
    assert record.train_steps == 2_000_000


def test_default_learning_algorithms_exclude_rule_based(tmp_path: Path):
    module = load_run_suite_training_module()
    suite = create_default_suite_config(
        result_root=tmp_path / "results",
        model_root=tmp_path / "models",
    )

    assert module.select_learning_algorithms(suite) == [
        "ppo",
        "recurrentppo",
        "context_recurrentppo",
        "agri_metarl",
    ]


def test_cli_rejects_rule_based_without_writing_registry(tmp_path: Path, monkeypatch):
    module = load_run_suite_training_module()
    suite = create_default_suite_config(
        result_root=tmp_path / "results",
        model_root=tmp_path / "models",
    )
    manifest = write_suite_manifest(suite, tmp_path / "suite_manifest.json")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_suite_training.py",
            "--manifest",
            str(manifest),
            "--algorithms",
            "rule_based",
            "--dry_run",
        ],
    )

    with pytest.raises(SystemExit):
        module.main()

    assert not (Path(suite.result_root) / "runs.csv").exists()
