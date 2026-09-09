#!/usr/bin/env python3
"""Run training jobs for a robust experiment suite and write a run registry."""

from __future__ import annotations

import argparse
import copy
import os
import sys
import time
from pathlib import Path

import pandas as pd
from stable_baselines3.common.save_util import load_from_zip_file


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from gl_gym.RL.experiment_manager import ExperimentManager
from gl_gym.common.utils import load_env_params, load_model_hyperparams
from gl_gym.experiments.suite_schema import RunRecord, load_suite_manifest, write_records_csv
from gl_gym.paths import MODEL_DIR


def expected_experiment_manager_model_root(suite) -> Path:
    return MODEL_DIR / suite.suite_id


def validate_experiment_manager_model_root(suite) -> Path:
    expected_root = expected_experiment_manager_model_root(suite)
    configured_root = Path(suite.model_root)
    if configured_root.resolve(strict=False) != expected_root.resolve(strict=False):
        raise ValueError(
            "custom model_root is not supported by ExperimentManager-backed training; "
            f"expected {expected_root}, got {configured_root}"
        )
    return expected_root


def run_artifact_paths(suite, algorithm: str, run_name: str) -> tuple[Path, Path]:
    run_root = validate_experiment_manager_model_root(suite) / algorithm / "deterministic"
    return (
        run_root / "models" / run_name / "best_model.zip",
        run_root / "envs" / run_name / "best_vecnormalize.pkl",
    )


def run_last_artifact_paths(suite, algorithm: str, run_name: str) -> tuple[Path, Path]:
    run_root = validate_experiment_manager_model_root(suite) / algorithm / "deterministic"
    return (
        run_root / "models" / run_name / "last_model.zip",
        run_root / "envs" / run_name / "last_vecnormalize.pkl",
    )


def checkpoint_num_timesteps(model_path: str | Path) -> int:
    data, _, _ = load_from_zip_file(Path(model_path), device="cpu")
    return int(data.get("num_timesteps") or 0)


def select_resume_checkpoint(
    best_model_path: Path,
    best_vecnormalize_path: Path,
    last_model_path: Path,
    last_vecnormalize_path: Path,
) -> tuple[Path, Path, int] | None:
    candidates: list[tuple[Path, Path, int]] = []
    for model_path, vecnormalize_path in [
        (best_model_path, best_vecnormalize_path),
        (last_model_path, last_vecnormalize_path),
    ]:
        if model_path.is_file() and vecnormalize_path.is_file():
            try:
                candidates.append((model_path, vecnormalize_path, checkpoint_num_timesteps(model_path)))
            except Exception:
                continue
    if not candidates:
        return None
    return max(candidates, key=lambda item: item[2])


def select_learning_algorithms(suite, requested: list[str] | None = None) -> list[str]:
    algorithms = requested or [algo for algo in suite.algorithms if algo != "rule_based"]
    if "rule_based" in algorithms:
        raise ValueError("rule_based is not trainable; omit it from --algorithms")
    return algorithms


def run_key(algorithm: str, seed: int) -> tuple[str, int]:
    return algorithm, int(seed)


def read_run_registry(path: str | Path) -> list[RunRecord]:
    registry_path = Path(path)
    if not registry_path.is_file():
        return []
    data = pd.read_csv(registry_path)
    return [
        RunRecord(
            suite_id=str(row.suite_id),
            algorithm=str(row.algorithm),
            seed=int(row.seed),
            run_name=str(row.run_name),
            model_path=str(row.model_path),
            vecnormalize_path=str(row.vecnormalize_path),
            status=str(row.status),
            train_steps=int(row.train_steps),
            wall_time_seconds=float(row.wall_time_seconds),
            best_eval_return=float(row.best_eval_return),
            notes="" if pd.isna(row.notes) else str(row.notes),
        )
        for row in data.itertuples(index=False)
    ]


def completed_artifacts_exist(record: RunRecord, min_train_steps: int) -> bool:
    return (
        record.status == "completed"
        and record.train_steps >= min_train_steps
        and Path(record.model_path).is_file()
        and Path(record.vecnormalize_path).is_file()
    )


def failed_record(suite, algorithm: str, seed: int, exc: BaseException) -> RunRecord:
    run_name = f"{algorithm}_seed{seed}"
    model_path, vecnormalize_path = run_artifact_paths(suite, algorithm, run_name)
    message = f"{type(exc).__name__}: {exc}"
    return RunRecord(
        suite_id=suite.suite_id,
        algorithm=algorithm,
        seed=seed,
        run_name=run_name,
        model_path=str(model_path),
        vecnormalize_path=str(vecnormalize_path),
        status="failed",
        train_steps=0,
        wall_time_seconds=0.0,
        best_eval_return=float("nan"),
        notes=message[:500],
    )


def train_one(
    suite,
    algorithm: str,
    seed: int,
    device: str,
    dry_run: bool,
    train_timesteps: int | None = None,
    n_envs: int | None = None,
    n_steps: int | None = None,
    batch_size: int | None = None,
    n_epochs: int | None = None,
    n_evals: int | None = None,
    resume_partial: bool = False,
) -> RunRecord:
    run_name = f"{algorithm}_seed{seed}"
    model_path, vecnormalize_path = run_artifact_paths(suite, algorithm, run_name)
    last_model_path, last_vecnormalize_path = run_last_artifact_paths(suite, algorithm, run_name)
    planned_train_timesteps = train_timesteps if train_timesteps is not None else suite.train_timesteps

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

    start_time = time.time()
    env_base_params, env_specific_params = load_env_params(suite.env_id, os.path.join("configs", "envs"))
    env_base_params["start_train_year"] = suite.train_year
    env_base_params["end_train_year"] = suite.train_year
    env_base_params["start_train_day"] = suite.train_start_day
    env_base_params["end_train_day"] = suite.train_end_day

    hyperparameters = load_model_hyperparams(algorithm, suite.env_id)
    hyperparameters["total_timesteps"] = planned_train_timesteps
    if n_envs is not None:
        hyperparameters["n_envs"] = n_envs
    if n_steps is not None:
        hyperparameters["n_steps"] = n_steps
    if batch_size is not None:
        hyperparameters["batch_size"] = batch_size
    if n_epochs is not None:
        hyperparameters["n_epochs"] = n_epochs
    continue_model_path = None
    continue_vecnormalize_path = None
    if resume_partial:
        checkpoint = select_resume_checkpoint(
            model_path,
            vecnormalize_path,
            last_model_path,
            last_vecnormalize_path,
        )
        if checkpoint is not None:
            checkpoint_model_path, checkpoint_vecnormalize_path, checkpoint_steps = checkpoint
            if 0 < checkpoint_steps < planned_train_timesteps:
                continue_model_path = str(checkpoint_model_path)
                continue_vecnormalize_path = str(checkpoint_vecnormalize_path)
                hyperparameters["total_timesteps"] = planned_train_timesteps - checkpoint_steps

    manager = ExperimentManager(
        env_id=suite.env_id,
        project=suite.suite_id,
        env_base_params=env_base_params,
        env_specific_params=env_specific_params,
        hyperparameters=copy.deepcopy(hyperparameters),
        group=run_name,
        n_eval_episodes=1,
        n_evals=n_evals if n_evals is not None else 10,
        algorithm=algorithm,
        env_seed=seed,
        model_seed=seed,
        stochastic=False,
        save_model=True,
        save_env=True,
        device=device,
        run_name=run_name,
        continue_model_path=continue_model_path,
        continue_vecnormalize_path=continue_vecnormalize_path,
    )
    manager.run_experiment()

    return RunRecord(
        suite_id=suite.suite_id,
        algorithm=algorithm,
        seed=seed,
        run_name=run_name,
        model_path=str(model_path),
        vecnormalize_path=str(vecnormalize_path),
        status="completed",
        train_steps=planned_train_timesteps,
        wall_time_seconds=time.time() - start_time,
        best_eval_return=float("nan"),
        notes="training completed",
    )


def run_training_suite(
    suite,
    algorithms: list[str],
    seeds: list[int],
    device: str,
    dry_run: bool,
    registry_path: str | Path,
    train_timesteps: int | None = None,
    n_envs: int | None = None,
    n_steps: int | None = None,
    batch_size: int | None = None,
    n_epochs: int | None = None,
    n_evals: int | None = None,
    skip_completed: bool = False,
    resume_partial: bool = False,
) -> list[RunRecord]:
    planned_train_timesteps = train_timesteps if train_timesteps is not None else suite.train_timesteps
    records_by_key = {
        run_key(record.algorithm, record.seed): record
        for record in read_run_registry(registry_path)
    }
    ordered_keys = list(records_by_key)

    for algorithm in algorithms:
        for seed in seeds:
            key = run_key(algorithm, seed)
            existing = records_by_key.get(key)
            if (
                skip_completed
                and existing is not None
                and completed_artifacts_exist(existing, planned_train_timesteps)
            ):
                if key not in ordered_keys:
                    ordered_keys.append(key)
                continue

            if key not in ordered_keys:
                ordered_keys.append(key)

            try:
                record = train_one(
                    suite,
                    algorithm=algorithm,
                    seed=seed,
                    device=device,
                    dry_run=dry_run,
                    train_timesteps=train_timesteps,
                    n_envs=n_envs,
                    n_steps=n_steps,
                    batch_size=batch_size,
                    n_epochs=n_epochs,
                    n_evals=n_evals,
                    resume_partial=resume_partial,
                )
            except Exception as exc:  # noqa: BLE001 - persist failure and continue long suites
                record = failed_record(suite, algorithm, seed, exc)

            records_by_key[key] = record
            write_records_csv([records_by_key[item] for item in ordered_keys], registry_path)

    return [records_by_key[key] for key in ordered_keys]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--algorithms", nargs="+")
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--train_timesteps", type=int)
    parser.add_argument("--n_envs", type=int)
    parser.add_argument("--n_steps", type=int)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--n_epochs", type=int)
    parser.add_argument("--n_evals", type=int)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--skip_completed", action="store_true")
    parser.add_argument("--resume_partial", action="store_true")
    args = parser.parse_args()

    suite = load_suite_manifest(args.manifest)
    try:
        algorithms = select_learning_algorithms(suite, args.algorithms)
        validate_experiment_manager_model_root(suite)
    except ValueError as exc:
        parser.error(str(exc))
    seeds = args.seeds or suite.seeds

    out = Path(suite.result_root) / "runs.csv"
    run_training_suite(
        suite,
        algorithms=algorithms,
        seeds=seeds,
        device=args.device,
        dry_run=args.dry_run,
        registry_path=out,
        train_timesteps=args.train_timesteps,
        n_envs=args.n_envs,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        n_evals=args.n_evals,
        skip_completed=args.skip_completed,
        resume_partial=args.resume_partial,
    )
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
