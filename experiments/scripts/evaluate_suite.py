#!/usr/bin/env python3
"""Evaluate completed robust experiment suite runs on deterministic tasks."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
import sys
from pathlib import Path
from typing import Any, Callable, Mapping


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
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
from gl_gym.experiments.shield_evaluation import aggregate_episode_interventions


SHIELD_METHOD = "minimal_feasibility_shield_v1"
SHIELD_SUFFIX = "__minimal_feasibility_shield_v1"
STAGE2_CONDITIONS = (
    "zero_ode_failures",
    "intervention_rate_within_0p5pct",
    "paired_return_loss_within_2pct",
    "paired_violation_burden_within_5pct",
)


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


def build_parser() -> argparse.ArgumentParser:
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
    parser.add_argument("--action_shield", action="store_true")
    parser.add_argument("--stage2_decision")
    parser.add_argument("--result_root")
    parser.add_argument("--interventions_out")
    return parser


def validate_cli_mode(args: argparse.Namespace) -> None:
    shield_only = (args.stage2_decision, args.result_root, args.interventions_out)
    if args.action_shield:
        if args.stage2_decision is None or args.result_root is None:
            raise ValueError("--action_shield requires --stage2_decision and --result_root")
    elif any(value is not None for value in shield_only):
        raise ValueError("shield-only arguments require --action_shield")


def _sha(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _strict_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"), parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))
    if not isinstance(value, dict):
        raise ValueError(f"strict JSON mapping required: {path}")
    return value


def _overlap(a: Path, b: Path) -> bool:
    return a == b or a in b.parents or b in a.parents


def _validate_shield_prerequisite(
    args: argparse.Namespace, suite: Any, runs: pd.DataFrame
) -> tuple[Path, dict[str, Any], dict[str, Any]]:
    """Validate immutable Stage-2 approval and output topology before execution."""
    decision_path = Path(args.stage2_decision).resolve()
    if not decision_path.is_file() or decision_path.is_symlink():
        raise FileNotFoundError("Stage-2 decision must be a regular file")
    decision = _strict_json(decision_path)
    if set(decision) != {"outcome", "stage", "conditions", "evidence", "reasons"}:
        raise ValueError("Stage-2 decision has invalid exact schema")
    if decision["stage"] != "stage2_shielded_context_ab" or decision["outcome"] != "continue_to_full_suite":
        raise ValueError("Stage-2 did not approve continuation to the full suite")
    conditions = decision["conditions"]
    if not isinstance(conditions, dict) or set(conditions) != set(STAGE2_CONDITIONS) or any(v is not True for v in conditions.values()):
        raise ValueError("Stage-2 requires the four exact passing conditions")
    if not isinstance(decision["evidence"], dict) or decision["reasons"] != []:
        raise ValueError("Stage-2 decision evidence/reasons are inconsistent")
    stage2_root = decision_path.parent
    manifest_path = stage2_root / "shield_manifest.json"
    if not manifest_path.is_file() or manifest_path.is_symlink():
        raise FileNotFoundError("Stage-2 shield_manifest.json is required")
    manifest = _strict_json(manifest_path)
    if manifest.get("method") != SHIELD_METHOD:
        raise ValueError("Stage-2 shield method is stale")
    # Reuse the Stage-2 implementation's canonical, source-sensitive values.
    from experiments.scripts import run_shielded_context_ab as stage2
    if manifest.get("fixed_lambdas") != list(stage2.DEFAULT_LAMBDAS):
        raise ValueError("Stage-2 fixed lambda grid is stale")
    if manifest.get("formal_solver_options") != dict(stage2.FORMAL_CVODES_OPTIONS):
        raise ValueError("Stage-2 formal solver configuration is stale")
    _, rule_sha = stage2._load_rule_params()
    checks = {
        "source_manifest_sha256": _sha(args.manifest),
        "source_tasks_sha256": _sha(args.tasks_csv),
        "rule_config_sha256": rule_sha,
        "env_config_sha256": _sha(stage2.ENV_CONFIG_PATH),
        "runtime_source_tree_sha256": stage2._runtime_source_tree_sha256(),
        **stage2._behavior_source_hashes(),
    }
    for name, expected in checks.items():
        if manifest.get(name) != expected:
            raise ValueError(f"Stage-2 provenance is stale: {name}")
    checkpoints = manifest.get("checkpoints")
    if not isinstance(checkpoints, list):
        raise ValueError("Stage-2 checkpoint provenance is missing")
    approved = {(int(item["seed"]), item["model_sha256"], item["vecnormalize_sha256"]) for item in checkpoints}
    selected = runs[(runs["algorithm"] == "agri_metarl") & (runs["status"] == "completed")]
    current = {(int(row.seed), _sha(row.model_path), _sha(row.vecnormalize_path)) for row in selected.itertuples(index=False)}
    if current != approved:
        raise ValueError("Stage-2 checkpoint provenance does not match selected full-suite runs")
    output = Path(args.result_root).resolve()
    protected = [Path(suite.result_root).resolve(), stage2_root]
    for key in ("unshielded_result_root", "stage1_root", "failure_root"):
        if manifest.get(key):
            protected.append(Path(manifest[key]).resolve())
    work = output.parent / f".{output.name}.work"
    for candidate in (output, work):
        if any(stage2._collides(candidate, root) for root in protected):
            raise ValueError("shield output/work roots must be disjoint from prerequisite roots")
    if output.exists() and not output.is_dir():
        raise ValueError("shield result_root exists as a file")
    return output, manifest, decision


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(value, handle, allow_nan=False, sort_keys=True, indent=2)
            handle.write("\n")
        os.replace(temp, path)
    except BaseException:
        try: os.unlink(temp)
        except OSError: pass
        raise


def close_environment(env: Any, primary_error: BaseException | None) -> None:
    """Close an environment without masking an episode/model failure."""
    try:
        env.close()
    except BaseException as close_error:
        if primary_error is None:
            raise
        primary_error.add_note(
            f"environment close also failed: {type(close_error).__name__}: {close_error}"
        )


def _replace_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    os.close(handle)
    try:
        frame.to_csv(temporary, index=False)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def prepare_shield_resume(
    eval_path: Path, interventions_path: Path, *, stage2_manifest_sha256: str
) -> set[tuple[str, int, str]]:
    """Retain only rows whose full shield evidence and provenance still validate."""
    if not eval_path.is_file() or not interventions_path.is_file():
        return set()
    import numpy as np
    raw = pd.read_csv(eval_path)
    evidence = pd.read_csv(interventions_path)
    required = {
        "algorithm", "method", "seed", "task_id", "completed", "model_sha256",
        "vecnormalize_sha256", "stage2_manifest_sha256", "executed_action_trace_path",
        "requested_action_trace_path", "intervention_records_path",
    }
    valid: set[tuple[str, int, str]] = set()
    valid_evidence_indices: list[int] = []
    if required.issubset(evidence.columns):
        for index, row in evidence.iterrows():
            key = (str(row.algorithm), int(row.seed), str(row.task_id))
            try:
                if (
                    row.method != SHIELD_METHOD or row.algorithm != "agri_metarl" + SHIELD_SUFFIX
                    or row.completed not in (True, 1) or row.stage2_manifest_sha256 != stage2_manifest_sha256
                ):
                    raise ValueError("stale evidence provenance")
                executed = np.load(Path(row.executed_action_trace_path), allow_pickle=False)
                requested = np.load(Path(row.requested_action_trace_path), allow_pickle=False)
                records = json.loads(Path(row.intervention_records_path).read_text(encoding="utf-8"))
                if executed.ndim != 2 or requested.shape != executed.shape or len(records) != len(executed):
                    raise ValueError("trace shape mismatch")
                summary = aggregate_episode_interventions(records, executed.shape[1])
                if int(row.total_steps) != summary["total_steps"] or int(row.intervention_count) != summary["intervention_count"]:
                    raise ValueError("summary mismatch")
                raw_matches = raw[
                    (raw["algorithm"] == key[0]) & (raw["seed"] == key[1]) & (raw["task_id"] == key[2])
                ]
                if len(raw_matches) != 1:
                    raise ValueError("raw evidence must be one-to-one")
            except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
                continue
            valid.add(key); valid_evidence_indices.append(index)
    cleaned_raw = raw[
        [
            (str(row.algorithm), int(row.seed), str(row.task_id)) in valid
            for row in raw.itertuples(index=False)
        ]
    ]
    cleaned_evidence = evidence.loc[valid_evidence_indices]
    if len(cleaned_raw) != len(raw): _replace_csv(cleaned_raw, eval_path)
    if len(cleaned_evidence) != len(evidence): _replace_csv(cleaned_evidence, interventions_path)
    return valid


def run(
    args: argparse.Namespace,
    *,
    model_map: Mapping[str, Any] = ALG_MAP,
    env_loader: Callable[..., Any] = load_task_env,
    episode_runner: Callable[..., Any] = run_deterministic_episode,
) -> int:
    validate_cli_mode(args)

    suite = load_suite_manifest(args.manifest)
    runs = pd.read_csv(args.runs_csv)
    tasks = pd.read_csv(args.tasks_csv)

    if args.algorithms:
        runs = runs[runs["algorithm"].isin(args.algorithms)]
    if args.seeds:
        runs = runs[runs["seed"].isin(args.seeds)]
    shield_root: Path | None = None
    if args.action_shield:
        if set(runs["algorithm"]) - {"agri_metarl"}:
            raise ValueError("the approved action-shield experiment supports only agri_metarl")
        shield_root, _, _ = _validate_shield_prerequisite(args, suite, runs)
    tasks = filter_tasks(
        tasks,
        splits=args.splits,
        task_ids=args.task_ids,
        limit_tasks=args.limit_tasks,
    )

    out_path = (shield_root if shield_root is not None else Path(suite.result_root)) / "eval_raw.csv"
    interventions_path = (Path(args.interventions_out).resolve() if args.interventions_out else (shield_root / "interventions.csv" if shield_root else None))
    if interventions_path is not None and (_overlap(interventions_path, Path(suite.result_root).resolve()) or _overlap(interventions_path, Path(args.stage2_decision).resolve().parent)):
        raise ValueError("interventions output must be disjoint from protected roots")
    if args.resume_eval and args.action_shield:
        completed_keys = prepare_shield_resume(
            out_path, interventions_path,
            stage2_manifest_sha256=_sha(Path(args.stage2_decision).parent / "shield_manifest.json"),
        )
    else:
        completed_keys = completed_eval_keys(out_path) if args.resume_eval else set()
    if out_path.exists() and not args.resume_eval:
        out_path.unlink()
    if args.action_shield and interventions_path.exists() and not args.resume_eval:
        interventions_path.unlink()

    rows_written = 0
    task_records = [task_from_row(row) for row in tasks.itertuples(index=False)]
    formal_complete = (
        args.splits is None and args.task_ids is None and args.limit_tasks is None
        and len(task_records) == 91
    )
    shield_params = None
    if args.action_shield:
        from experiments.scripts.run_shielded_context_ab import _load_rule_params
        shield_params, _ = _load_rule_params()

    for run in runs.itertuples(index=False):
        if run.algorithm == "rule_based":
            continue
        if run.status == "dry_run":
            continue
        if run.status != "completed":
            continue

        validate_completed_run_paths(run)
        model = model_map[run.algorithm].load(run.model_path, device="cpu")
        for task in task_records:
            output_algorithm = run.algorithm + SHIELD_SUFFIX if args.action_shield else run.algorithm
            key = evaluation_key(output_algorithm, int(run.seed), task.task_id)
            if key in completed_keys:
                print(
                    f"Skipping completed eval: {run.algorithm} seed={run.seed} task={task.task_id}",
                    flush=True,
                )
                continue
            env = env_loader(suite, task, run.vecnormalize_path, **({"shield_params": shield_params} if args.action_shield else {}))
            primary_error: BaseException | None = None
            try:
                result = episode_runner(model, env, return_diagnostics=True) if args.action_shield else episode_runner(model, env)
            except BaseException as error:
                primary_error = error
                raise
            finally:
                close_environment(env, primary_error)
            if args.action_shield:
                metrics, diagnostics = result
                executed = diagnostics.get("action_trace")
                requested = diagnostics.get("requested_action_trace")
                records = diagnostics.get("action_shield_records")
                if not isinstance(executed, __import__("numpy").ndarray) or not isinstance(requested, __import__("numpy").ndarray) or not isinstance(records, (tuple, list)) or executed.ndim != 2 or requested.shape != executed.shape or len(records) != len(executed):
                    raise ValueError("incomplete action-shield diagnostics")
                indexed = [dict(record, step_index=i) for i, record in enumerate(records)]
                summary = aggregate_episode_interventions(indexed, executed.shape[1])
                work = shield_root.parent / f".{shield_root.name}.work"
                token = f"{int(run.seed)}__{task.task_id}"
                trace_dir = work / "traces"
                record_dir = work / "intervention_records"
                trace_dir.mkdir(parents=True, exist_ok=True); record_dir.mkdir(parents=True, exist_ok=True)
                import numpy as np
                executed_path = trace_dir / f"{token}__executed.npy"
                requested_path = trace_dir / f"{token}__requested.npy"
                records_path = record_dir / f"{token}.json"
                np.save(executed_path, executed, allow_pickle=False); np.save(requested_path, requested, allow_pickle=False)
                _atomic_json(records_path, indexed)
            row = EvaluationMetricRow(
                suite_id=suite.suite_id,
                algorithm=output_algorithm,
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
            if args.action_shield:
                evidence_row = {
                    "suite_id": suite.suite_id, "algorithm": output_algorithm, "method": SHIELD_METHOD,
                    "seed": int(run.seed), "task_id": task.task_id, "completed": True,
                    "split": task.split, "weather_year": task.weather_year,
                    "start_day": task.start_day, "uncertainty_scale": task.uncertainty_scale,
                    "economic_scenario": task.economic_scenario,
                    "climate_constraint_scenario": task.climate_constraint_scenario,
                    "executed_action_trace_path": str(executed_path.resolve()),
                    "requested_action_trace_path": str(requested_path.resolve()),
                    "intervention_records_path": str(records_path.resolve()), **summary,
                    "model_sha256": _sha(run.model_path), "vecnormalize_sha256": _sha(run.vecnormalize_path),
                    "stage2_manifest_sha256": _sha(Path(args.stage2_decision).parent / "shield_manifest.json"),
                    "formal_complete": formal_complete,
                }
                pd.DataFrame([evidence_row]).to_csv(interventions_path, mode="a", header=not interventions_path.exists(), index=False)
            completed_keys.add(key)
            rows_written += 1
            print(
                f"Evaluated {run.algorithm} seed={run.seed} task={task.task_id}",
                flush=True,
            )

    if rows_written == 0:
        print("No completed runs to evaluate; eval_raw.csv was not written.")
        return 0

    if args.action_shield and formal_complete:
        approved_seeds = {
            int(item["seed"])
            for item in _strict_json(Path(args.stage2_decision).parent / "shield_manifest.json")["checkpoints"]
        }
        expected = {
            ("agri_metarl" + SHIELD_SUFFIX, seed, task.task_id)
            for seed in approved_seeds for task in task_records
        }
        actual = completed_eval_keys(out_path)
        if actual != expected:
            raise RuntimeError(
                f"shield full-suite evaluation is incomplete: expected {len(expected)} exact task keys, got {len(actual)}"
            )

    print(f"Wrote {rows_written} rows to {out_path}")
    return rows_written


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
