"""Pair and gate the approved zero-context versus online-context diagnostic."""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Callable
from uuid import uuid4

import numpy as np
import pandas as pd


DIAGNOSTIC_TASK_IDS = (
    "fixed_2010_d59_u0p00_standard",
    "heldout_2011_d59_u0p00_standard",
    "heldout_2012_d59_u0p00_standard",
    "heldout_2013_d59_u0p00_standard",
    "uncertainty_2012_d80_u0p05_standard",
    "uncertainty_2013_d100_u0p15_standard",
    "economic_2011_d59_u0p00_high_energy_price",
    "economic_2013_d100_u0p00_combined_stress",
)
MODES = ("zero_context", "online_context")
PAIR_METRICS = (
    "episode_return",
    "EPI",
    "temp_violation",
    "co2_violation",
    "rh_violation",
)
VIOLATION_METRICS = ("temp_violation", "co2_violation", "rh_violation")
EPSILON = 1e-9
EXPECTED_SEEDS = frozenset({42, 123})


def select_diagnostic_tasks(tasks: pd.DataFrame) -> pd.DataFrame:
    """Select all approved tasks in their preregistered order."""

    if "task_id" not in tasks.columns:
        raise ValueError("task table is missing task_id column")
    duplicate_ids = tasks.loc[tasks["task_id"].duplicated(keep=False), "task_id"].unique()
    if len(duplicate_ids):
        raise ValueError(f"duplicate task IDs: {duplicate_ids.tolist()}")
    indexed = tasks.set_index("task_id", drop=False)
    missing = [task_id for task_id in DIAGNOSTIC_TASK_IDS if task_id not in indexed.index]
    if missing:
        raise ValueError(f"missing diagnostic task IDs: {missing}")
    return indexed.loc[list(DIAGNOSTIC_TASK_IDS)].reset_index(drop=True)


def _validate_raw_pairs(raw: pd.DataFrame) -> None:
    required = {
        "seed",
        "task_id",
        "split",
        "inference_mode",
        "action_trace_path",
        "support_ready_step",
        *PAIR_METRICS,
    }
    missing = sorted(required.difference(raw.columns))
    if missing:
        raise ValueError(f"raw diagnostic table is missing columns: {missing}")

    key = ["seed", "task_id", "inference_mode"]
    duplicates = raw.duplicated(key, keep=False)
    if duplicates.any():
        sample = raw.loc[duplicates, key].head(10).to_dict(orient="records")
        raise ValueError(f"duplicate seed/task/mode rows: {sample}")

    invalid_groups: list[tuple[Any, Any]] = []
    inconsistent_splits: list[tuple[Any, Any]] = []
    for group_key, group in raw.groupby(["seed", "task_id"], dropna=False, sort=False):
        if len(group) != 2 or set(group["inference_mode"]) != set(MODES):
            invalid_groups.append(group_key)
        if group["split"].nunique(dropna=False) != 1:
            inconsistent_splits.append(group_key)
    if invalid_groups:
        raise ValueError(
            "each seed/task pair must contain both inference modes exactly once: "
            f"{invalid_groups[:10]}"
        )
    if inconsistent_splits:
        raise ValueError(f"split differs between inference modes: {inconsistent_splits[:10]}")

    numeric = raw.loc[:, PAIR_METRICS].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(numeric).all():
        raise ValueError("paired diagnostic metrics must contain only finite values")


def _validated_actions(actions: Any, path: Any) -> np.ndarray:
    try:
        array = np.asarray(actions, dtype=float)
    except (TypeError, ValueError) as error:
        raise ValueError(f"action trace {path!r} must be numeric") from error
    if array.ndim != 2:
        raise ValueError(f"action trace {path!r} must be a 2D (steps, action_dim) array")
    if array.shape[0] == 0:
        raise ValueError(f"action trace {path!r} must be nonempty")
    if array.shape[1] == 0:
        raise ValueError(f"action trace {path!r} must have positive dimensions")
    if not np.isfinite(array).all():
        raise ValueError(f"action trace {path!r} must contain only finite values")
    return array


def build_paired_deltas(
    raw: pd.DataFrame,
    load_actions: Callable[[Any], Any] | None = None,
) -> pd.DataFrame:
    """Build one online-minus-zero record for every seed/task pair."""

    _validate_raw_pairs(raw)
    keys = ["seed", "task_id", "split"]
    value_columns = [*PAIR_METRICS, "action_trace_path"]
    zero = raw.loc[raw["inference_mode"] == MODES[0], keys + value_columns].copy()
    online = raw.loc[
        raw["inference_mode"] == MODES[1],
        keys + value_columns + ["support_ready_step"],
    ].copy()
    paired = zero.merge(online, on=keys, how="inner", validate="one_to_one", suffixes=("_zero", "_online"))
    if len(paired) * 2 != len(raw):
        raise ValueError("both inference modes must share the same seed, task ID, and split")

    for metric in PAIR_METRICS:
        zero_column = f"{metric}_zero"
        online_column = f"{metric}_online"
        paired[f"{metric}_delta"] = paired[online_column] - paired[zero_column]
    for metric in VIOLATION_METRICS:
        paired[f"{metric}_online_to_zero_ratio"] = _normalized_violation_ratio(
            paired[f"{metric}_zero"].to_numpy(dtype=float),
            paired[f"{metric}_online"].to_numpy(dtype=float),
        )

    action_loader = load_actions or (lambda path: np.load(path, allow_pickle=False))
    action_deltas: list[float] = []
    for row in paired.itertuples(index=False):
        zero_actions = _validated_actions(action_loader(row.action_trace_path_zero), row.action_trace_path_zero)
        online_actions = _validated_actions(action_loader(row.action_trace_path_online), row.action_trace_path_online)
        if zero_actions.shape != online_actions.shape:
            raise ValueError(
                "paired action traces must have the same shape: "
                f"{zero_actions.shape} != {online_actions.shape}"
            )
        try:
            readiness = float(row.support_ready_step)
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"support_ready_step must be numeric, got {row.support_ready_step!r}"
            ) from error
        if (
            not np.isfinite(readiness)
            or not readiness.is_integer()
            or readiness < 1
            or readiness >= zero_actions.shape[0]
        ):
            raise ValueError(
                "support_ready_step must be a finite integer from 1 through "
                f"{zero_actions.shape[0] - 1}, got {row.support_ready_step!r}"
            )
        start = int(readiness)
        action_deltas.append(
            float(np.mean(np.abs(online_actions[start:] - zero_actions[start:])))
        )
    paired["mean_abs_action_delta"] = action_deltas
    return paired


def _normalized_violation_ratio(zero: np.ndarray, online: np.ndarray) -> np.ndarray:
    """Return neutral ratios for zero/zero cells without hiding new violations."""

    zero = np.asarray(zero, dtype=float)
    online = np.asarray(online, dtype=float)
    ratios = online / (np.abs(zero) + EPSILON)
    both_zero = (zero == 0.0) & (online == 0.0)
    ratios[both_zero] = 1.0
    return ratios


def _failure_decision(reason: str) -> dict[str, Any]:
    conditions = {
        "actions_change_both_seeds": False,
        "positive_nonfixed_return": False,
        "no_seed_large_return_loss": False,
        "violation_burden_within_5pct": False,
        "fixed_return_within_2pct": False,
    }
    return {
        "outcome": "redesign_before_training",
        "conditions": conditions,
        "evidence": {
            "action_change_max_by_seed": {},
            "mean_nonfixed_return_delta": None,
            "nonfixed_relative_return_delta_by_seed": {},
            "mean_normalized_violation_burden": None,
            "fixed_relative_return_delta_by_seed": {},
        },
        "reasons": [reason],
    }


def _seed_scalars(series: pd.Series) -> dict[str, float]:
    return {str(seed): float(value) for seed, value in series.items()}


def _expected_split(task_id: str) -> str:
    return task_id.split("_", 1)[0]


def _paired_structure_error(paired: pd.DataFrame) -> str | None:
    required = {"seed", "task_id", "split"}
    missing = sorted(required.difference(paired.columns))
    if missing:
        return f"missing columns {missing}"
    if len(paired) != 16:
        return f"expected 16 paired rows, got {len(paired)}"
    if paired.duplicated(["seed", "task_id"]).any():
        return "duplicate seed/task paired rows"
    if set(paired["seed"]) != EXPECTED_SEEDS:
        return f"expected seeds {sorted(EXPECTED_SEEDS)}, got {sorted(set(paired['seed']))}"
    approved = set(DIAGNOSTIC_TASK_IDS)
    for seed in sorted(EXPECTED_SEEDS):
        seed_rows = paired.loc[paired["seed"] == seed]
        if set(seed_rows["task_id"]) != approved:
            missing_tasks = sorted(approved.difference(seed_rows["task_id"]))
            extra_tasks = sorted(set(seed_rows["task_id"]).difference(approved))
            return f"seed {seed} task IDs differ; missing={missing_tasks}, extra={extra_tasks}"
        expected_splits = seed_rows["task_id"].map(_expected_split)
        if not seed_rows["split"].reset_index(drop=True).equals(expected_splits.reset_index(drop=True)):
            return f"seed {seed} has task/split mismatch"
        if int((seed_rows["split"] == "fixed").sum()) != 1 or int(
            (seed_rows["split"] != "fixed").sum()
        ) != 7:
            return f"seed {seed} must have one fixed and seven non-fixed rows"
    return None


def _validate_raw_experiment_structure(raw: pd.DataFrame) -> None:
    if set(raw["seed"]) != EXPECTED_SEEDS:
        raise ValueError(
            f"approved diagnostic seeds must be {sorted(EXPECTED_SEEDS)}, "
            f"got {sorted(set(raw['seed']))}"
        )
    approved = set(DIAGNOSTIC_TASK_IDS)
    for seed in sorted(EXPECTED_SEEDS):
        seed_rows = raw.loc[raw["seed"] == seed]
        task_ids = set(seed_rows["task_id"])
        if task_ids != approved:
            missing = sorted(approved.difference(task_ids))
            extra = sorted(task_ids.difference(approved))
            raise ValueError(
                f"approved diagnostic task IDs differ for seed {seed}: "
                f"missing={missing}, extra={extra}"
            )
        expected_splits = seed_rows["task_id"].map(_expected_split)
        if not seed_rows["split"].reset_index(drop=True).equals(expected_splits.reset_index(drop=True)):
            raise ValueError(f"approved diagnostic task IDs have incorrect splits for seed {seed}")
        fixed_count = int((seed_rows["split"] == "fixed").sum())
        nonfixed_count = int((seed_rows["split"] != "fixed").sum())
        if fixed_count != 2 or nonfixed_count != 14:
            raise ValueError(
                f"seed {seed} must have two mode rows for one fixed and seven non-fixed tasks"
            )


def evaluate_context_gate(paired: pd.DataFrame) -> dict[str, Any]:
    """Evaluate the five preregistered conditions for continuing to 500k steps."""

    structure_error = _paired_structure_error(paired)
    if structure_error is not None:
        return _failure_decision(f"invalid experiment structure: {structure_error}")

    required = {
        "seed",
        "task_id",
        "split",
        "mean_abs_action_delta",
        "episode_return_zero",
        "episode_return_delta",
        *(f"{metric}_zero" for metric in VIOLATION_METRICS),
        *(f"{metric}_online" for metric in VIOLATION_METRICS),
    }
    missing = sorted(required.difference(paired.columns))
    if missing:
        return _failure_decision(f"missing gate evidence columns: {missing}")
    if paired.empty:
        return _failure_decision("gate evidence is empty")

    numeric_columns = sorted(required.difference({"seed", "task_id", "split"}))
    numeric = paired[numeric_columns].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(numeric).all():
        return _failure_decision("gate evidence contains non-finite values")

    nonfixed = paired.loc[paired["split"] != "fixed"]
    fixed = paired.loc[paired["split"] == "fixed"]
    if nonfixed.empty or fixed.empty:
        return _failure_decision("gate evidence must contain fixed and non-fixed rows")

    action_max = paired.groupby("seed", sort=True)["mean_abs_action_delta"].max()
    nonfixed_groups = nonfixed.groupby("seed", sort=True)
    nonfixed_relative = nonfixed_groups["episode_return_delta"].mean() / (
        nonfixed_groups["episode_return_zero"].apply(lambda values: values.abs().mean()) + EPSILON
    )
    fixed_groups = fixed.groupby("seed", sort=True)
    fixed_relative = fixed_groups["episode_return_delta"].mean() / (
        fixed_groups["episode_return_zero"].apply(lambda values: values.abs().mean()) + EPSILON
    )
    violation_ratios = np.column_stack(
        [
            _normalized_violation_ratio(
                paired[f"{metric}_zero"].to_numpy(dtype=float),
                paired[f"{metric}_online"].to_numpy(dtype=float),
            )
            for metric in VIOLATION_METRICS
        ]
    )
    mean_normalized_burden = float(violation_ratios.mean())
    mean_nonfixed_return_delta = float(nonfixed["episode_return_delta"].mean())
    exactly_two_seeds = len(action_max) == 2

    conditions = {
        "actions_change_both_seeds": bool(exactly_two_seeds and (action_max > EPSILON).all()),
        "positive_nonfixed_return": bool(mean_nonfixed_return_delta > 0.0),
        "no_seed_large_return_loss": bool(exactly_two_seeds and (nonfixed_relative >= -0.02).all()),
        "violation_burden_within_5pct": bool(mean_normalized_burden <= 1.05),
        "fixed_return_within_2pct": bool(exactly_two_seeds and (fixed_relative >= -0.02).all()),
    }
    evidence = {
        "action_change_max_by_seed": _seed_scalars(action_max),
        "mean_nonfixed_return_delta": mean_nonfixed_return_delta,
        "nonfixed_relative_return_delta_by_seed": _seed_scalars(nonfixed_relative),
        "mean_normalized_violation_burden": mean_normalized_burden,
        "fixed_relative_return_delta_by_seed": _seed_scalars(fixed_relative),
    }
    reasons = [name for name, passed in conditions.items() if not passed]
    return {
        "outcome": "continue_to_500k" if all(conditions.values()) else "redesign_before_training",
        "conditions": conditions,
        "evidence": evidence,
        "reasons": reasons,
    }


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _write_strict_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(_json_safe(value), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def write_context_ab_artifacts(
    raw: pd.DataFrame,
    result_root: str | Path,
    manifest: dict[str, Any],
) -> dict[str, Path]:
    """Validate and write the five context A/B diagnostic artifacts."""

    if len(raw) != 32:
        raise ValueError(f"context A/B diagnostic requires exactly 32 rows, got {len(raw)}")
    _validate_raw_pairs(raw)
    if set(raw["inference_mode"]) != set(MODES):
        raise ValueError(f"diagnostic inference modes must be exactly {list(MODES)}")
    _validate_raw_experiment_structure(raw)
    missing_traces = [str(path) for path in raw["action_trace_path"] if not Path(path).is_file()]
    if missing_traces:
        raise ValueError(f"action trace files do not exist: {missing_traces[:10]}")

    paired = build_paired_deltas(raw)
    summary = (
        raw.groupby(["inference_mode", "split"], sort=True, dropna=False)[list(PAIR_METRICS)]
        .mean()
        .reset_index()
    )
    decision = evaluate_context_gate(paired)
    root = Path(result_root)
    root.parent.mkdir(parents=True, exist_ok=True)
    paths = {
        "eval_raw": root / "eval_raw.csv",
        "paired_deltas": root / "paired_deltas.csv",
        "split_summary": root / "split_summary.csv",
        "diagnostic_manifest": root / "diagnostic_manifest.json",
        "decision": root / "decision.json",
    }
    staging = Path(
        tempfile.mkdtemp(prefix=f".{root.name}.staging-", dir=str(root.parent))
    )
    backup = root.parent / f".{root.name}.backup-{uuid4().hex}"
    root_was_backed_up = False
    try:
        if root.exists():
            if not root.is_dir():
                raise ValueError(f"diagnostic result root must be a directory: {root}")
            shutil.copytree(root, staging, dirs_exist_ok=True)

        staging_traces = staging / "traces"
        shutil.rmtree(staging_traces, ignore_errors=True)
        staging_traces.mkdir(parents=True)
        published_raw = raw.copy()
        final_trace_paths: dict[tuple[int, str, str], Path] = {}
        for index, row in raw.iterrows():
            filename = (
                f"seed{int(row['seed'])}__{row['task_id']}__"
                f"{row['inference_mode']}.npy"
            )
            source_trace = Path(str(row["action_trace_path"]))
            shutil.copy2(source_trace, staging_traces / filename)
            final_trace = (root / "traces" / filename).resolve()
            published_raw.at[index, "action_trace_path"] = str(final_trace)
            final_trace_paths[
                (int(row["seed"]), str(row["task_id"]), str(row["inference_mode"]))
            ] = final_trace
        for mode in MODES:
            column = f"action_trace_path_{'zero' if mode == MODES[0] else 'online'}"
            paired[column] = [
                str(final_trace_paths[(int(row.seed), str(row.task_id), mode)])
                for row in paired.itertuples(index=False)
            ]

        staging_paths = {name: staging / path.name for name, path in paths.items()}
        published_raw.to_csv(staging_paths["eval_raw"], index=False)
        paired.to_csv(staging_paths["paired_deltas"], index=False)
        summary.to_csv(staging_paths["split_summary"], index=False)
        _write_strict_json(staging_paths["diagnostic_manifest"], manifest)
        _write_strict_json(staging_paths["decision"], decision)

        if root.exists():
            os.replace(root, backup)
            root_was_backed_up = True
        try:
            os.replace(staging, root)
        except BaseException:
            if root_was_backed_up:
                if root.exists():
                    failed_publication = root.parent / f".{root.name}.failed-{uuid4().hex}"
                    os.replace(root, failed_publication)
                    shutil.rmtree(failed_publication, ignore_errors=True)
                os.replace(backup, root)
                root_was_backed_up = False
            raise
        if root_was_backed_up:
            root_was_backed_up = False
            shutil.rmtree(backup, ignore_errors=True)
    finally:
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)
        if root_was_backed_up and backup.exists() and not root.exists():
            os.replace(backup, root)
            root_was_backed_up = False
        if backup.exists():
            shutil.rmtree(backup, ignore_errors=True)
    return paths
