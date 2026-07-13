#!/usr/bin/env python3
"""Run the preregistered AgriMetaRL online-context A/B diagnostic."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import platform
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any, Callable


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd

from gl_gym.RL.agri_metarl import AgriMetaRL
from gl_gym.experiments.context_ab import (
    DIAGNOSTIC_TASK_IDS,
    MODES,
    PAIR_METRICS,
    select_diagnostic_tasks,
    write_context_ab_artifacts,
)
from gl_gym.experiments.suite_evaluation import (
    load_task_env,
    run_deterministic_episode,
    task_from_row,
)
from gl_gym.experiments.suite_schema import load_suite_manifest


APPROVED_SEEDS = (42, 123)
DEFAULT_RESULT_ROOT = Path(
    "artifacts/results/AgriControl_C_2026-07-10-v3-context-ab"
)


def build_diagnostic_runs(
    model_root: str | Path,
    seeds: list[int] | tuple[int, ...],
) -> list[dict[str, Any]]:
    """Resolve and validate the approved last checkpoints for both seeds."""

    if tuple(seeds) != APPROVED_SEEDS:
        raise ValueError("approved diagnostic requires seeds exactly 42 123")
    root = Path(model_root)
    runs: list[dict[str, Any]] = []
    for seed in APPROVED_SEEDS:
        model_path = (
            root
            / "agri_metarl"
            / "deterministic"
            / "models"
            / f"agri_metarl_seed{seed}"
            / "last_model.zip"
        )
        vecnormalize_path = (
            root
            / "agri_metarl"
            / "deterministic"
            / "envs"
            / f"agri_metarl_seed{seed}"
            / "last_vecnormalize.pkl"
        )
        if not model_path.is_file():
            raise FileNotFoundError(f"model checkpoint does not exist: {model_path}")
        if not vecnormalize_path.is_file():
            raise FileNotFoundError(
                f"vecnormalize checkpoint does not exist: {vecnormalize_path}"
            )
        runs.append(
            {
                "seed": seed,
                "model_path": model_path.resolve(),
                "vecnormalize_path": vecnormalize_path.resolve(),
            }
        )
    return runs


def validate_result_root(
    result_root: str | Path,
    source_suite_result_root: str | Path,
) -> Path:
    """Reject any diagnostic output that could overwrite formal suite results."""

    resolved = Path(result_root).resolve()
    source = Path(source_suite_result_root).resolve()
    if resolved == source:
        raise ValueError(
            "diagnostic result root must differ from the formal source suite result root"
        )
    return resolved


def resume_row_is_complete(
    row: dict[str, Any] | pd.Series,
    *,
    checkpoint_steps: int | None = None,
) -> bool:
    """Return whether a progress row has usable evidence and a valid action trace."""

    required = {
        "seed",
        "task_id",
        "inference_mode",
        "checkpoint_steps",
        "action_trace_path",
        *PAIR_METRICS,
    }
    if not required.issubset(row.keys()):
        return False
    try:
        row_steps = float(row["checkpoint_steps"])
        if not np.isfinite(row_steps) or not row_steps.is_integer() or row_steps < 0:
            return False
        if checkpoint_steps is not None and int(row_steps) != checkpoint_steps:
            return False
        metrics = np.asarray([float(row[name]) for name in PAIR_METRICS], dtype=float)
        if not np.isfinite(metrics).all():
            return False
        trace = np.load(Path(str(row["action_trace_path"])), allow_pickle=False)
        return bool(
            trace.ndim == 2
            and trace.shape[0] > 0
            and trace.shape[1] > 0
            and np.isfinite(trace).all()
        )
    except (KeyError, OSError, OverflowError, TypeError, ValueError):
        return False


def _normalize_diagnostic_scalar(name: str, value: Any) -> Any:
    """Return a stable CSV-safe scalar without hiding missing diagnostics."""

    if value is None:
        return float("nan")
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, (str, bool, int, float)):
        return value
    raise TypeError(f"diagnostic {name!r} must be a CSV-safe scalar")


def _validated_diagnostics(
    diagnostics: dict[str, Any],
    *,
    metric_names: set[str],
) -> dict[str, Any]:
    required = {"support_ready_step", "context_norm_mean", "context_norm_max"}
    missing = required.difference(diagnostics)
    if missing:
        raise KeyError(
            f"episode diagnostics are missing required keys: {sorted(missing)}"
        )
    reserved = {
        "seed",
        "task_id",
        "split",
        "inference_mode",
        "checkpoint_steps",
        "action_trace_path",
    }
    collisions = set(diagnostics).intersection(reserved | metric_names)
    if collisions:
        raise ValueError(
            "episode diagnostics collide with raw row fields: "
            f"{sorted(collisions)}"
        )
    return {
        name: _normalize_diagnostic_scalar(name, value)
        for name, value in diagnostics.items()
    }


def _provenance() -> dict[str, Any]:
    def git(*arguments: str) -> str:
        result = subprocess.run(
            ["git", *arguments],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=True,
        )
        return result.stdout.strip()

    return {
        "git_commit": git("rev-parse", "HEAD"),
        "dirty": bool(git("status", "--porcelain")),
    }


def _progress_path(result_root: Path) -> Path:
    return result_root.parent / f".{result_root.name}.work" / "progress.csv"


def _load_progress(path: Path, resume: bool) -> list[dict[str, Any]]:
    if not resume:
        if path.parent.exists():
            shutil.rmtree(path.parent)
        return []
    if not path.is_file():
        return []
    return pd.read_csv(path).to_dict(orient="records")


def _write_progress(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    pd.DataFrame(rows).to_csv(temporary, index=False)
    temporary.replace(path)


def _key(row: dict[str, Any]) -> tuple[int, str, str]:
    return (int(row["seed"]), str(row["task_id"]), str(row["inference_mode"]))


def _trace_path(root: Path, seed: int, task_id: str, mode: str) -> Path:
    return (root / "traces" / f"seed{seed}__{task_id}__{mode}.npy").resolve()


def run_diagnostic(
    *,
    suite: Any,
    tasks: pd.DataFrame,
    runs: list[dict[str, Any]],
    result_root: str | Path,
    source_manifest: str | Path,
    source_tasks_csv: str | Path,
    device: str,
    resume: bool,
    model_loader: Callable[[Path, str], Any] | None = None,
    env_loader: Callable[[Any, Any, Path], Any] = load_task_env,
    episode_runner: Callable[..., Any] = run_deterministic_episode,
    provenance_loader: Callable[[], dict[str, Any]] = _provenance,
) -> pd.DataFrame:
    """Execute all 32 episodes with injectable model, environment, and runner hooks."""

    root = validate_result_root(result_root, suite.result_root)
    selected = select_diagnostic_tasks(tasks)
    task_records = [task_from_row(row) for row in selected.itertuples(index=False)]
    progress_path = _progress_path(root)
    progress_rows = _load_progress(progress_path, resume)
    if not resume and root.exists():
        if not root.is_dir():
            raise ValueError(f"diagnostic result root must be a directory: {root}")
        shutil.rmtree(root / "traces", ignore_errors=True)
    target_keys = {
        (int(run["seed"]), task_id, mode)
        for run in runs
        for task_id in DIAGNOSTIC_TASK_IDS
        for mode in MODES
    }
    completed: dict[tuple[int, str, str], dict[str, Any]] = {}
    for row in progress_rows:
        try:
            row_key = _key(row)
        except (KeyError, TypeError, ValueError):
            continue
        if row_key in target_keys and resume_row_is_complete(row):
            completed[row_key] = row

    load_model = model_loader or (
        lambda path, selected_device: AgriMetaRL.load(
            str(path), device=selected_device
        )
    )
    root.joinpath("traces").mkdir(parents=True, exist_ok=True)
    checkpoint_records: list[dict[str, Any]] = []
    for run in runs:
        model = load_model(Path(run["model_path"]), device)
        checkpoint_steps = int(model.num_timesteps)
        checkpoint_records.append(
            {
                "seed": int(run["seed"]),
                "model_path": str(Path(run["model_path"]).resolve()),
                "vecnormalize_path": str(Path(run["vecnormalize_path"]).resolve()),
                "checkpoint_steps": checkpoint_steps,
            }
        )
        for task in task_records:
            for mode in MODES:
                key = (int(run["seed"]), task.task_id, mode)
                if key in completed and resume_row_is_complete(
                    completed[key], checkpoint_steps=checkpoint_steps
                ):
                    continue
                env = env_loader(suite, task, Path(run["vecnormalize_path"]))
                try:
                    metrics, diagnostics = episode_runner(
                        model,
                        env,
                        inference_mode=mode,
                        return_diagnostics=True,
                    )
                finally:
                    env.close()
                diagnostics = dict(diagnostics)
                action_trace = np.asarray(
                    diagnostics.pop("action_trace"), dtype=np.float32
                )
                normalized_diagnostics = _validated_diagnostics(
                    diagnostics,
                    metric_names=set(metrics),
                )
                trace_path = _trace_path(root, int(run["seed"]), task.task_id, mode)
                np.save(trace_path, action_trace, allow_pickle=False)
                row = {
                    **metrics,
                    **normalized_diagnostics,
                    "seed": int(run["seed"]),
                    "task_id": task.task_id,
                    "split": task.split,
                    "inference_mode": mode,
                    "checkpoint_steps": checkpoint_steps,
                    "action_trace_path": str(trace_path),
                }
                completed[key] = row
                ordered_rows = [completed[key] for key in sorted(completed)]
                _write_progress(ordered_rows, progress_path)
                print(
                    f"Evaluated seed={run['seed']} task={task.task_id} mode={mode}",
                    flush=True,
                )

    rows = [completed[key] for key in sorted(target_keys) if key in completed]
    if len(rows) != 32:
        raise RuntimeError(f"diagnostic completed {len(rows)} of 32 required episodes")
    raw = pd.DataFrame(rows)
    provenance = provenance_loader()
    manifest = {
        "source_manifest": str(Path(source_manifest).resolve()),
        "source_tasks_csv": str(Path(source_tasks_csv).resolve()),
        "source_suite_id": str(suite.suite_id),
        "source_suite_result_root": str(Path(suite.result_root).resolve()),
        "result_root": str(root),
        "checkpoints": checkpoint_records,
        "selected_task_ids": list(DIAGNOSTIC_TASK_IDS),
        "inference_modes": list(MODES),
        "seeds": list(APPROVED_SEEDS),
        "device": device,
        "git_commit": str(provenance["git_commit"]),
        "dirty": bool(provenance["dirty"]),
        "python_version": platform.python_version(),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    write_context_ab_artifacts(raw, root, manifest)
    return raw


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source_manifest", required=True)
    parser.add_argument("--source_tasks_csv", required=True)
    parser.add_argument("--model_root", required=True)
    parser.add_argument("--result_root", default=str(DEFAULT_RESULT_ROOT))
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--resume", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    source_manifest = Path(args.source_manifest)
    source_tasks_csv = Path(args.source_tasks_csv)
    if not source_manifest.is_file():
        raise FileNotFoundError(f"source manifest does not exist: {source_manifest}")
    if not source_tasks_csv.is_file():
        raise FileNotFoundError(f"source task CSV does not exist: {source_tasks_csv}")
    suite = load_suite_manifest(source_manifest)
    validate_result_root(args.result_root, suite.result_root)
    runs = build_diagnostic_runs(args.model_root, args.seeds)
    run_diagnostic(
        suite=suite,
        tasks=pd.read_csv(source_tasks_csv),
        runs=runs,
        result_root=args.result_root,
        source_manifest=source_manifest,
        source_tasks_csv=source_tasks_csv,
        device=args.device,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()
