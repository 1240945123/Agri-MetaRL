#!/usr/bin/env python3
"""Evaluate completed robust experiment suite runs on deterministic tasks."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd
from sb3_contrib import RecurrentPPO
from stable_baselines3 import PPO

from gl_gym.RL.agri_metarl import AgriMetaRL
from gl_gym.RL.context_recurrent_ppo import ContextRecurrentPPO
from gl_gym.experiments.suite_evaluation import (
    EvaluationMetricRow,
    append_eval_raw,
    completed_eval_keys,
    evaluation_key,
    load_task_env,
    run_deterministic_episode,
    task_from_row,
    validate_completed_run_paths,
)
from gl_gym.experiments.suite_schema import load_suite_manifest


ALG_MAP = {
    "ppo": PPO,
    "recurrentppo": RecurrentPPO,
    "context_recurrentppo": ContextRecurrentPPO,
    "agri_metarl": AgriMetaRL,
}


def filter_tasks(
    tasks: pd.DataFrame,
    splits: list[str] | None = None,
    task_ids: list[str] | None = None,
    limit_tasks: int | None = None,
) -> pd.DataFrame:
    """Return a deterministic subset of evaluation tasks for smoke/pilot runs."""

    selected = tasks
    if splits:
        selected = selected[selected["split"].isin(splits)]
    if task_ids:
        selected = selected[selected["task_id"].isin(task_ids)]
    if limit_tasks is not None:
        if limit_tasks < 1:
            raise ValueError("--limit_tasks must be positive")
        selected = selected.head(limit_tasks)
    return selected.reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--runs_csv", required=True)
    parser.add_argument("--tasks_csv", required=True)
    parser.add_argument("--algorithms", nargs="+")
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--splits", nargs="+")
    parser.add_argument("--task_ids", nargs="+")
    parser.add_argument("--limit_tasks", type=int)
    parser.add_argument(
        "--resume_eval",
        action="store_true",
        help="Skip algorithm/seed/task rows already present in eval_raw.csv.",
    )
    args = parser.parse_args()

    suite = load_suite_manifest(args.manifest)
    runs = pd.read_csv(args.runs_csv)
    tasks = pd.read_csv(args.tasks_csv)

    if args.algorithms:
        runs = runs[runs["algorithm"].isin(args.algorithms)]
    if args.seeds:
        runs = runs[runs["seed"].isin(args.seeds)]
    tasks = filter_tasks(
        tasks,
        splits=args.splits,
        task_ids=args.task_ids,
        limit_tasks=args.limit_tasks,
    )

    out_path = Path(suite.result_root) / "eval_raw.csv"
    completed_keys = completed_eval_keys(out_path) if args.resume_eval else set()
    if out_path.exists() and not args.resume_eval:
        out_path.unlink()

    rows_written = 0
    task_records = [task_from_row(row) for row in tasks.itertuples(index=False)]

    for run in runs.itertuples(index=False):
        if run.algorithm == "rule_based":
            continue
        if run.status == "dry_run":
            continue
        if run.status != "completed":
            continue

        validate_completed_run_paths(run)
        model = ALG_MAP[run.algorithm].load(run.model_path, device="cpu")
        for task in task_records:
            key = evaluation_key(run.algorithm, int(run.seed), task.task_id)
            if key in completed_keys:
                print(
                    f"Skipping completed eval: {run.algorithm} seed={run.seed} task={task.task_id}",
                    flush=True,
                )
                continue
            env = load_task_env(suite, task, run.vecnormalize_path)
            try:
                metrics = run_deterministic_episode(model, env)
            finally:
                env.close()
            row = EvaluationMetricRow(
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
                climate_constraint_scenario=task.climate_constraint_scenario,
                trajectory_path="",
                **metrics,
            )
            append_eval_raw(row, out_path)
            completed_keys.add(key)
            rows_written += 1
            print(
                f"Evaluated {run.algorithm} seed={run.seed} task={task.task_id}",
                flush=True,
            )

    if rows_written == 0:
        print("No completed runs to evaluate; eval_raw.csv was not written.")
        return

    print(f"Wrote {rows_written} rows to {out_path}")


if __name__ == "__main__":
    main()
