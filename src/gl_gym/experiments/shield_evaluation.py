"""Strict evidence aggregation and publication for action-shield evaluations."""

from __future__ import annotations

import json
import math
import os
import shutil
import tempfile
import uuid
from collections.abc import Mapping, Sequence
from datetime import date, datetime, time, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from gl_gym.environments.action_shield import DEFAULT_LAMBDAS


SCHEMA_VERSION = "minimal-feasibility-action-shield-v1"
EPSILON = 1e-9
RETURN_LOSS_THRESHOLD = 0.02
VIOLATION_RATIO_THRESHOLD = 1.05
INTERVENTION_RATE_THRESHOLD = 0.005
REQUIRED_METRICS = (
    "episode_return",
    "temp_violation",
    "co2_violation",
    "rh_violation",
)
VIOLATION_METRICS = REQUIRED_METRICS[1:]


def _strict_int(value: Any, *, name: str, minimum: int | None = None) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if minimum is not None and result < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return result


def _finite_float(value: Any, *, name: str, minimum: float | None = None) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.number)):
        raise TypeError(f"{name} must be numeric")
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite")
    if minimum is not None and result < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return result


def _stable_finite_mean(values: Any, *, name: str) -> float:
    finite_values = [_finite_float(value, name=name) for value in values]
    if not finite_values:
        raise ValueError(f"{name} requires at least one value")
    count = len(finite_values)
    try:
        result = math.fsum(value / count for value in finite_values)
    except OverflowError as exc:
        raise ValueError(f"{name} mean must remain finite") from exc
    return _finite_float(result, name=f"{name} mean")


def _vector(value: Any, *, name: str, action_dim: int) -> np.ndarray:
    raw = np.asarray(value)
    if raw.shape != (action_dim,):
        raise ValueError(f"{name} must have exact shape ({action_dim},)")
    if not np.issubdtype(raw.dtype, np.number) or np.issubdtype(raw.dtype, np.bool_) or np.issubdtype(raw.dtype, np.complexfloating):
        raise ValueError(f"{name} must be a finite numeric vector")
    result = np.array(raw, dtype=float, copy=True)
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain only finite values")
    if np.any((result < -1.0) | (result > 1.0)):
        raise ValueError(f"{name} must be within [-1, 1]")
    return result


def _required(record: Mapping[str, Any], name: str) -> Any:
    if name not in record:
        raise ValueError(f"shield record is missing {name}")
    return record[name]


def _validate_attempts(
    attempts: Any,
    *,
    selected_lambda: float,
    intervened: bool,
    action_dim: int,
    requested: np.ndarray,
    reference: np.ndarray | None,
) -> list[Mapping[str, Any]]:
    if not isinstance(attempts, list):
        raise TypeError("candidate_attempts must be a list")
    if not intervened:
        if attempts:
            raise ValueError("candidate_attempts must be empty without intervention")
        return attempts
    expected_count = DEFAULT_LAMBDAS.index(selected_lambda) + 1
    if len(attempts) != expected_count:
        raise ValueError("candidate_attempts must be the fixed prefix ending at selected_lambda")
    for index, attempt in enumerate(attempts):
        if not isinstance(attempt, Mapping):
            raise TypeError("each candidate attempt must be a mapping")
        lam = _finite_float(_required(attempt, "lambda"), name="candidate attempt lambda")
        if lam != DEFAULT_LAMBDAS[index]:
            raise ValueError("candidate_attempts must be the fixed ordered prefix")
        action = _vector(_required(attempt, "action"), name="candidate attempt action", action_dim=action_dim)
        expected_action = (1.0 - lam) * requested + lam * reference
        if not np.array_equal(action, expected_action):
            raise ValueError("candidate attempt action does not match its fixed-grid candidate")
        success = _required(attempt, "success")
        if type(success) is not bool:
            raise TypeError("candidate attempt success must be a strict bool")
        should_succeed = index == len(attempts) - 1
        if success is not should_succeed:
            raise ValueError("only the selected candidate attempt may succeed")
        _finite_float(
            _required(attempt, "elapsed_seconds"),
            name="candidate attempt elapsed_seconds",
            minimum=0.0,
        )
        exception_type = _required(attempt, "exception_type")
        exception_message = _required(attempt, "exception_message")
        if should_succeed:
            if exception_type is not None or exception_message is not None:
                raise ValueError("successful candidate attempt cannot contain an exception")
        elif not isinstance(exception_type, str) or not isinstance(exception_message, str):
            raise ValueError("failed candidate attempt must contain exception strings")
    return attempts


def aggregate_episode_interventions(
    records: Sequence[Mapping[str, Any]], action_dim: int
) -> dict[str, Any]:
    """Validate ordered per-step shield records and return episode evidence."""

    action_dim = _strict_int(action_dim, name="action_dim", minimum=1)
    if isinstance(records, (str, bytes)) or not isinstance(records, Sequence) or not records:
        raise ValueError("records must be a nonempty ordered sequence")

    lambdas: list[float] = []
    norms: dict[str, list[float]] = {name: [] for name in ("l1", "l2", "linf")}
    channel_counts = np.zeros(action_dim, dtype=np.int64)
    extra_attempts = 0
    elapsed_total = 0.0
    first_step: int | None = None

    for expected_step, record in enumerate(records):
        if not isinstance(record, Mapping):
            raise TypeError("each shield record must be a mapping")
        step = _strict_int(_required(record, "step_index"), name="step_index", minimum=0)
        if step != expected_step:
            raise ValueError("step_index values must be exactly consecutive and ordered from 0")
        if _required(record, "schema_version") != SCHEMA_VERSION:
            raise ValueError(f"schema_version must be {SCHEMA_VERSION!r}")
        intervened = _required(record, "intervened")
        if type(intervened) is not bool:
            raise TypeError("intervened must be a strict bool")

        requested = _vector(_required(record, "requested_action"), name="requested_action", action_dim=action_dim)
        executed = _vector(_required(record, "executed_action"), name="executed_action", action_dim=action_dim)
        reference = _required(record, "reference_action")
        if intervened:
            if reference is None:
                raise ValueError("reference_action is required for intervention")
            reference_vector = _vector(reference, name="reference_action", action_dim=action_dim)
        elif reference is not None:
            raise ValueError("reference_action must be None without intervention")
        else:
            reference_vector = None

        selected_lambda = _finite_float(_required(record, "selected_lambda"), name="selected_lambda")
        if intervened:
            if selected_lambda not in DEFAULT_LAMBDAS:
                raise ValueError("selected_lambda must be one of the fixed DEFAULT_LAMBDAS")
        elif selected_lambda != 0.0:
            raise ValueError("selected_lambda must be exactly zero without intervention")

        attempts = _validate_attempts(
            _required(record, "candidate_attempts"),
            selected_lambda=selected_lambda,
            intervened=intervened,
            action_dim=action_dim,
            requested=requested,
            reference=reference_vector,
        )
        if intervened:
            selected_action = _vector(
                _required(attempts[-1], "action"),
                name="selected candidate action",
                action_dim=action_dim,
            )
            if not np.array_equal(executed, selected_action):
                raise ValueError("executed_action must equal the selected candidate action")
        attempt_count = _strict_int(
            _required(record, "extra_solver_attempts"),
            name="extra_solver_attempts",
            minimum=0,
        )
        if len(attempts) != attempt_count:
            raise ValueError("candidate_attempts length must equal extra_solver_attempts")

        original_failure = _required(record, "original_failure")
        if intervened:
            if not isinstance(original_failure, Mapping):
                raise ValueError("original_failure must be a mapping for intervention")
            if set(original_failure) != {"exception_type", "exception_message"}:
                raise ValueError(
                    "original_failure must contain exactly exception_type and exception_message"
                )
            exception_type = original_failure["exception_type"]
            exception_message = original_failure["exception_message"]
            if not isinstance(exception_type, str) or not exception_type:
                raise ValueError("original_failure exception_type must be a nonempty string")
            if not isinstance(exception_message, str):
                raise ValueError("original_failure exception_message must be a string")
        elif original_failure is not None:
            raise ValueError("original_failure must be None without intervention")

        difference = requested - executed
        if not intervened and np.any(difference != 0.0):
            raise ValueError("requested_action and executed_action must match without intervention")
        expected_norms = {
            "l1": float(np.linalg.norm(difference, 1)),
            "l2": float(np.linalg.norm(difference, 2)),
            "linf": float(np.linalg.norm(difference, np.inf)),
        }
        observed_norms: dict[str, float] = {}
        for norm_name, recomputed in expected_norms.items():
            field = f"intervention_{norm_name}"
            observed = _finite_float(_required(record, field), name=field, minimum=0.0)
            if not np.isclose(observed, recomputed, rtol=1e-12, atol=1e-12):
                raise ValueError(f"{field} does not match requested-executed difference")
            observed_norms[norm_name] = observed
        changed = _required(record, "per_channel_changed")
        if not isinstance(changed, (list, tuple)) or len(changed) != action_dim or any(type(item) is not bool for item in changed):
            raise ValueError("per_channel_changed must be a strict bool vector")
        expected_changed = (difference != 0.0).tolist()
        if list(changed) != expected_changed:
            raise ValueError("per_channel_changed does not match exact action differences")
        elapsed = _finite_float(_required(record, "elapsed_seconds"), name="elapsed_seconds", minimum=0.0)

        extra_attempts += attempt_count
        elapsed_total += elapsed
        if not np.isfinite(elapsed_total):
            raise ValueError("shield elapsed time aggregate must remain finite")
        if intervened:
            if first_step is None:
                first_step = step
            lambdas.append(selected_lambda)
            for name, value in observed_norms.items():
                norms[name].append(value)
            channel_counts += np.asarray(changed, dtype=np.int64)

    intervention_count = len(lambdas)

    def summary(values: list[float], operation: str) -> float:
        if not values:
            return 0.0
        if operation == "mean":
            return _stable_finite_mean(values, name="episode intervention evidence")
        return _finite_float(max(values), name="episode intervention evidence maximum")

    result: dict[str, Any] = {
        "total_steps": len(records),
        "intervention_count": intervention_count,
        "intervention_rate": float(intervention_count / len(records)),
        "first_intervention_step": first_step,
        "selected_lambda_mean": summary(lambdas, "mean"),
        "selected_lambda_max": summary(lambdas, "max"),
    }
    for name in ("l1", "l2", "linf"):
        result[f"intervention_{name}_mean"] = summary(norms[name], "mean")
        result[f"intervention_{name}_max"] = summary(norms[name], "max")
    result.update(
        {
            "per_channel_intervention_counts": [int(value) for value in channel_counts],
            "extra_solver_attempts": int(extra_attempts),
            "shield_elapsed_seconds": float(elapsed_total),
            "ode_failure_count": 0,
        }
    )
    return result


def _canonical_key_scalar(value: Any, *, column: str, source: str) -> int | str:
    if column == "seed":
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
            raise ValueError(f"{source} {column} must be an integral non-boolean scalar")
        return int(value)
    if not isinstance(value, (str, np.str_)):
        raise ValueError(f"{source} {column} must be a nonempty string scalar")
    canonical = str(value)
    if not canonical:
        raise ValueError(f"{source} {column} must be a nonempty string scalar")
    return canonical


def _expected_key_set(expected_keys: Any, key_columns: tuple[str, ...]) -> set[tuple[int | str, ...]]:
    try:
        raw_keys = list(expected_keys)
    except TypeError as exc:
        raise TypeError("expected_keys must be iterable") from exc
    result: set[tuple[int | str, ...]] = set()
    for item in raw_keys:
        if not isinstance(item, tuple) or len(item) != len(key_columns):
            raise ValueError("each expected key tuple must have exact key_columns length")
        key = tuple(
            _canonical_key_scalar(value, column=column, source="expected key")
            for column, value in zip(key_columns, item, strict=True)
        )
        if key in result:
            raise ValueError(f"duplicate expected key after canonicalization: {key!r}")
        result.add(key)
    if not result:
        raise ValueError("expected_keys must be nonempty")
    return result


def _validate_gate_table(
    table: pd.DataFrame,
    *,
    label: str,
    expected: set[tuple[int | str, ...]],
    key_columns: tuple[str, ...],
) -> pd.DataFrame:
    if not isinstance(table, pd.DataFrame):
        raise TypeError(f"{label} must be a pandas DataFrame")
    required = list(key_columns) + ["completed", "ode_failure_count", *REQUIRED_METRICS]
    if label == "shielded":
        required += ["total_steps", "intervention_count"]
    missing = [column for column in required if column not in table.columns]
    if missing:
        raise ValueError(f"{label} is missing required columns: {missing}")
    validated = table.copy(deep=True)
    for column in key_columns:
        canonical_values = [
            _canonical_key_scalar(value, column=column, source=label)
            for value in table[column].array
        ]
        validated[column] = pd.Series(canonical_values, index=validated.index, dtype=object)
    if validated.duplicated(subset=list(key_columns), keep=False).any():
        raise ValueError(f"{label} contains duplicate keys after canonicalization")
    actual = set(validated.loc[:, list(key_columns)].itertuples(index=False, name=None))
    if actual != expected:
        raise ValueError(f"{label} keys do not exactly match expected_keys")

    completed_values: list[bool] = []
    for row_index, row in validated.iterrows():
        completed = row["completed"]
        if not isinstance(completed, (bool, np.bool_)):
            raise TypeError(f"{label} completed must contain only booleans")
        completed_values.append(bool(completed))
        _strict_int(row["ode_failure_count"], name=f"{label} ode_failure_count", minimum=0)
        for metric in REQUIRED_METRICS:
            value = row[metric]
            if completed:
                _finite_float(value, name=f"{label} {metric}")
            if metric in VIOLATION_METRICS and not pd.isna(value):
                if _finite_float(value, name=f"{label} {metric}") < 0.0:
                    raise ValueError(f"{label} {metric} cannot be negative")
        if label == "shielded":
            total_steps = _strict_int(row["total_steps"], name="shielded total_steps", minimum=1)
            count = _strict_int(row["intervention_count"], name="shielded intervention_count", minimum=0)
            if count > total_steps:
                raise ValueError("shielded intervention_count cannot exceed total_steps")
    validated["completed"] = completed_values
    return validated


def _paired_validated(
    shielded: pd.DataFrame,
    unshielded: pd.DataFrame,
    expected_keys: Any,
    key_columns: tuple[str, ...],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if not isinstance(key_columns, tuple) or not key_columns or any(type(c) is not str or not c for c in key_columns):
        raise ValueError("key_columns must be a nonempty tuple of column names")
    if len(set(key_columns)) != len(key_columns):
        raise ValueError("key_columns must be unique")
    expected = _expected_key_set(expected_keys, key_columns)
    shield = _validate_gate_table(shielded, label="shielded", expected=expected, key_columns=key_columns)
    unshield = _validate_gate_table(unshielded, label="unshielded", expected=expected, key_columns=key_columns)
    merged = shield.merge(unshield, on=list(key_columns), how="inner", suffixes=("_shielded", "_unshielded"), validate="one_to_one")
    paired = merged.loc[merged["completed_shielded"] & merged["completed_unshielded"]].copy()
    if paired.empty:
        raise ValueError("zero rows were completed by both tables for paired evaluation")
    return shield, unshield, paired


def build_paired_shield_deltas(
    shielded: pd.DataFrame,
    unshielded: pd.DataFrame,
    expected_keys: Any,
    key_columns: tuple[str, ...] = ("seed", "task_id", "inference_mode"),
) -> pd.DataFrame:
    """Return completed-pair deltas and violation ratios after strict validation."""

    _, _, paired = _paired_validated(shielded, unshielded, expected_keys, key_columns)
    output = paired.loc[:, list(key_columns)].copy()
    for metric in REQUIRED_METRICS:
        shield_values = paired[f"{metric}_shielded"].astype(float)
        base_values = paired[f"{metric}_unshielded"].astype(float)
        output[f"{metric}_shielded"] = shield_values.to_numpy(copy=True)
        output[f"{metric}_unshielded"] = base_values.to_numpy(copy=True)
        with np.errstate(over="ignore", invalid="ignore"):
            delta = shield_values - base_values
        if not np.all(np.isfinite(delta)):
            raise ValueError(f"derived {metric} deltas must remain finite")
        output[f"{metric}_delta"] = delta.to_numpy(copy=True)
        if metric in VIOLATION_METRICS:
            with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                ratios = shield_values / (base_values.abs() + EPSILON)
            ratios = ratios.mask((shield_values == 0.0) & (base_values == 0.0), 1.0)
            if not np.all(np.isfinite(ratios)):
                raise ValueError(f"derived {metric} ratios must remain finite")
            output[f"{metric}_ratio"] = ratios.to_numpy(copy=True)
    return output.reset_index(drop=True)


def evaluate_shield_gate(
    shielded: pd.DataFrame,
    unshielded: pd.DataFrame,
    expected_keys: Any,
    key_columns: tuple[str, ...] = ("seed", "task_id", "inference_mode"),
) -> dict[str, Any]:
    """Evaluate the predeclared action-shield acceptance conditions."""

    shield, unshield, paired = _paired_validated(shielded, unshielded, expected_keys, key_columns)
    shield_failures = int(sum(int(value) for value in shield["ode_failure_count"]))
    unshield_failures = int(sum(int(value) for value in unshield["ode_failure_count"]))
    shield_completion_count = int(shield["completed"].sum())
    unshield_completion_count = int(unshield["completed"].sum())
    total_steps = int(sum(int(value) for value in shield["total_steps"]))
    intervention_count = int(sum(int(value) for value in shield["intervention_count"]))
    intervention_rate = _finite_float(intervention_count / total_steps, name="intervention_rate")

    with np.errstate(over="ignore", invalid="ignore"):
        delta = paired["episode_return_shielded"].astype(float) - paired["episode_return_unshielded"].astype(float)
    if not np.all(np.isfinite(delta)):
        raise ValueError("derived episode return deltas must remain finite")
    denominator = _stable_finite_mean(
        paired["episode_return_unshielded"].astype(float).abs(),
        name="absolute unshielded return",
    ) + EPSILON
    denominator = _finite_float(denominator, name="return loss denominator", minimum=EPSILON)
    mean_return_delta = _stable_finite_mean(delta, name="paired return delta")
    relative_return_loss = _finite_float(
        max(0.0, -mean_return_delta / denominator),
        name="relative_return_loss",
        minimum=0.0,
    )
    ratio_means: dict[str, float] = {}
    for metric in VIOLATION_METRICS:
        shield_values = paired[f"{metric}_shielded"].astype(float)
        base_values = paired[f"{metric}_unshielded"].astype(float)
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            ratio = shield_values / (base_values.abs() + EPSILON)
        ratio = ratio.mask((shield_values == 0.0) & (base_values == 0.0), 1.0)
        if not np.all(np.isfinite(ratio)):
            raise ValueError(f"derived {metric} ratios must remain finite")
        ratio_means[metric] = _stable_finite_mean(ratio, name=f"{metric} ratio")
    paired_violation_ratio_mean = _stable_finite_mean(
        ratio_means.values(), name="paired violation ratio"
    )

    conditions = {
        "zero_ode_failures": shield_failures == 0 and shield_completion_count == len(shield),
        "intervention_rate_within_0p5pct": intervention_rate <= INTERVENTION_RATE_THRESHOLD,
        "paired_return_loss_within_2pct": relative_return_loss <= RETURN_LOSS_THRESHOLD,
        "paired_violation_burden_within_5pct": paired_violation_ratio_mean <= VIOLATION_RATIO_THRESHOLD,
    }
    reasons = [name for name, passed in conditions.items() if not passed]
    evidence: dict[str, Any] = {
        "shielded_row_count": int(len(shield)),
        "unshielded_row_count": int(len(unshield)),
        "shielded_completion_count": shield_completion_count,
        "unshielded_completion_count": unshield_completion_count,
        "shielded_incomplete_count": int(len(shield) - shield_completion_count),
        "unshielded_incomplete_count": int(len(unshield) - unshield_completion_count),
        "shielded_ode_failure_count": shield_failures,
        "unshielded_ode_failure_count": unshield_failures,
        "shielded_failure_count": shield_failures,
        "unshielded_failure_count": unshield_failures,
        "paired_count": int(len(paired)),
        "shielded_total_steps": total_steps,
        "shielded_intervention_count": intervention_count,
        "intervention_rate": intervention_rate,
        "mean_paired_return_delta": mean_return_delta,
        "relative_return_loss": relative_return_loss,
        "paired_violation_ratio_mean": paired_violation_ratio_mean,
        "intervention_rate_threshold": INTERVENTION_RATE_THRESHOLD,
        "relative_return_loss_threshold": RETURN_LOSS_THRESHOLD,
        "violation_ratio_mean_threshold": VIOLATION_RATIO_THRESHOLD,
        "thresholds": {
            "intervention_rate": INTERVENTION_RATE_THRESHOLD,
            "relative_return_loss": RETURN_LOSS_THRESHOLD,
            "violation_ratio_mean": VIOLATION_RATIO_THRESHOLD,
        },
    }
    evidence.update({f"{metric}_ratio_mean": value for metric, value in ratio_means.items()})
    return {
        "outcome": "pass" if not reasons else "fail",
        "conditions": {name: bool(value) for name, value in conditions.items()},
        "evidence": evidence,
        "reasons": reasons,
    }


def _is_supported_csv_scalar(value: Any) -> bool:
    return isinstance(
        value,
        (
            str,
            bool,
            int,
            float,
            np.bool_,
            np.integer,
            np.floating,
            date,
            datetime,
            time,
            timedelta,
            pd.Timestamp,
            pd.Timedelta,
            np.datetime64,
            np.timedelta64,
        ),
    )


def _scalar_is_missing(value: Any) -> bool:
    if value is None or value is pd.NA or value is pd.NaT:
        return True
    missing = pd.isna(value)
    if not isinstance(missing, (bool, np.bool_)):
        raise TypeError("cell missingness must be scalar")
    return bool(missing)


def _check_frame(frame: pd.DataFrame, *, name: str, duplicate_check: bool) -> pd.DataFrame:
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        raise ValueError(f"{name} must be a nonempty DataFrame")
    nullable = {column for column in frame.columns if column == "first_intervention_step" or column.endswith("_first_intervention_step")}
    validated = frame.copy(deep=True)
    for column in frame.columns:
        normalized: list[Any] = []
        for value in frame[column].array:
            qualified_name = f"{name}.{column}"
            try:
                missing = _scalar_is_missing(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{qualified_name} must contain only scalar cells") from exc
            if missing:
                if column in nullable:
                    normalized.append(None)
                    continue
                raise ValueError(f"{qualified_name} must contain finite nonmissing values")
            if not _is_supported_csv_scalar(value):
                raise ValueError(f"{qualified_name} must contain only supported scalar cells")
            if isinstance(value, (float, np.floating, int, np.integer)) and not isinstance(
                value, (bool, np.bool_)
            ):
                if not np.isfinite(value):
                    raise ValueError(f"{qualified_name} must contain finite values")
            elif isinstance(value, (complex, np.complexfloating)):
                raise ValueError(f"{qualified_name} must contain finite real values")
            normalized.append(value.item() if isinstance(value, np.generic) else value)
        if column in nullable:
            validated[column] = pd.Series(normalized, index=validated.index, dtype=object)
    if duplicate_check:
        available = [column for column in ("seed", "task_id", "inference_mode") if column in validated.columns]
        if available and validated.duplicated(subset=available, keep=False).any():
            raise ValueError(f"{name} contains duplicate keys")
    return validated


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return [_json_safe(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError("JSON mapping keys must be strings")
            result[key] = _json_safe(item)
        return result
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not np.isfinite(value):
            raise ValueError("JSON values must be finite")
        return value
    raise TypeError(f"value of type {type(value).__name__} is not JSON-safe")


def _directory_trees_equal(source: Path, restored: Path) -> bool:
    source_entries = {path.relative_to(source): path for path in source.rglob("*")}
    restored_entries = {path.relative_to(restored): path for path in restored.rglob("*")}
    if set(source_entries) != set(restored_entries):
        return False
    for relative_path, source_path in source_entries.items():
        restored_path = restored_entries[relative_path]
        if source_path.is_symlink() != restored_path.is_symlink():
            return False
        if source_path.is_symlink():
            if os.readlink(source_path) != os.readlink(restored_path):
                return False
        elif source_path.is_dir() != restored_path.is_dir():
            return False
        elif source_path.is_file():
            if not restored_path.is_file():
                return False
            if source_path.stat().st_size != restored_path.stat().st_size:
                return False
            with source_path.open("rb") as source_handle, restored_path.open("rb") as restored_handle:
                while True:
                    source_chunk = source_handle.read(1024 * 1024)
                    restored_chunk = restored_handle.read(1024 * 1024)
                    if source_chunk != restored_chunk:
                        return False
                    if not source_chunk:
                        break
        else:
            return False
    return True


def _restore_prior_root(
    backup: Path,
    root: Path,
    publication_error: BaseException,
) -> bool:
    try:
        os.replace(backup, root)
    except Exception as restoration_error:
        publication_error.add_note(
            "atomic backup rename restoration failed: "
            f"{type(restoration_error).__name__}: {restoration_error}"
        )
    else:
        return True

    try:
        if root.exists():
            raise RuntimeError("result_root unexpectedly exists before fallback copy")
        shutil.copytree(backup, root, copy_function=shutil.copy2, symlinks=True)
        if not _directory_trees_equal(backup, root):
            raise OSError("fallback copy verification did not match the backup tree")
    except Exception as fallback_error:
        publication_error.add_note(
            "fallback copy restoration failed; sole backup preserved: "
            f"{type(fallback_error).__name__}: {fallback_error}"
        )
        if root.exists():
            try:
                shutil.rmtree(root)
            except Exception as cleanup_error:
                publication_error.add_note(
                    "partial fallback root cleanup failed: "
                    f"{type(cleanup_error).__name__}: {cleanup_error}"
                )
        return False

    try:
        shutil.rmtree(backup)
    except Exception as cleanup_error:
        publication_error.add_note(
            "verified fallback restored result_root but stale backup cleanup failed: "
            f"{type(cleanup_error).__name__}: {cleanup_error}"
        )
    return True


def write_shield_artifacts_atomic(
    raw: pd.DataFrame,
    paired: pd.DataFrame,
    interventions: pd.DataFrame,
    manifest: Mapping[str, Any],
    decision: Mapping[str, Any],
    result_root: str | Path,
) -> dict[str, Path]:
    """Publish the five shield artifacts as an atomic directory replacement."""

    raw = _check_frame(raw, name="raw", duplicate_check=True)
    paired = _check_frame(paired, name="paired", duplicate_check=False)
    interventions = _check_frame(interventions, name="interventions", duplicate_check=True)
    if not isinstance(manifest, Mapping) or not isinstance(decision, Mapping):
        raise TypeError("manifest and decision must be mappings")
    safe_manifest = _json_safe(manifest)
    safe_decision = _json_safe(decision)

    root = Path(result_root).resolve()
    if root.exists() and not root.is_dir():
        raise ValueError("result_root exists as a file")
    root.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{root.name}.stage-", dir=root.parent))
    backup = root.parent / f".{root.name}.backup-{uuid.uuid4().hex}"
    old_moved = False
    published = False
    try:
        raw.to_csv(stage / "eval_raw.csv", index=False)
        paired.to_csv(stage / "paired_deltas.csv", index=False)
        interventions.to_csv(stage / "interventions.csv", index=False)
        for filename, payload in (("shield_manifest.json", safe_manifest), ("decision.json", safe_decision)):
            with (stage / filename).open("w", encoding="utf-8", newline="\n") as handle:
                json.dump(payload, handle, ensure_ascii=False, allow_nan=False, indent=2, sort_keys=True)
                handle.write("\n")
        if root.exists():
            os.replace(root, backup)
            old_moved = True
        try:
            os.replace(stage, root)
            published = True
        except BaseException as publication_error:
            if old_moved:
                _restore_prior_root(backup, root, publication_error)
                old_moved = False
            raise
        if old_moved:
            try:
                shutil.rmtree(backup)
            except Exception:
                # Publication is already committed. A stale hidden backup is
                # preferable to reporting failure while retaining the new root.
                pass
            else:
                old_moved = False
    finally:
        if stage.exists():
            try:
                shutil.rmtree(stage, ignore_errors=True)
            except Exception:
                pass
        if backup.exists() and not published and old_moved and not root.exists():
            try:
                os.replace(backup, root)
            except Exception:
                # Never remove the only remaining copy of the prior root.
                pass

    return {
        "eval_raw": root / "eval_raw.csv",
        "paired_deltas": root / "paired_deltas.csv",
        "interventions": root / "interventions.csv",
        "shield_manifest": root / "shield_manifest.json",
        "decision": root / "decision.json",
    }
