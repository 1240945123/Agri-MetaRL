"""Deterministic evaluation helpers for robust experiment suites."""

from __future__ import annotations

import csv
import inspect
import os
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

import numpy as np

from gl_gym.environments.action_shield import ActionShieldConfig, DEFAULT_LAMBDAS
from gl_gym.experiments.suite_schema import EvaluationTaskRecord


_ACTION_SHIELD_SCHEMA_VERSION = ActionShieldConfig().schema_version


@dataclass(frozen=True, slots=True)
class EvaluationMetricRow:
    suite_id: str
    algorithm: str
    seed: int
    run_name: str
    task_id: str
    split: str
    weather_year: int
    start_day: int
    uncertainty_scale: float
    economic_scenario: str
    climate_constraint_scenario: str
    episode_return: float
    EPI: float
    revenue: float
    heat_cost: float
    co2_cost: float
    elec_cost: float
    temp_violation: float
    co2_violation: float
    rh_violation: float
    twb_percent: float
    trajectory_path: str


def task_from_row(row: Any) -> EvaluationTaskRecord:
    """Convert a pandas row/itertuples record to the canonical task record."""

    return EvaluationTaskRecord(
        suite_id=str(row.suite_id),
        task_id=str(row.task_id),
        split=str(row.split),
        weather_year=int(row.weather_year),
        start_day=int(row.start_day),
        uncertainty_scale=float(row.uncertainty_scale),
        economic_scenario=str(row.economic_scenario),
        climate_constraint_scenario=str(row.climate_constraint_scenario),
    )


def load_task_env(
    suite: Any,
    task: EvaluationTaskRecord,
    vecnormalize_path: str | Path,
    *,
    shield_params: Mapping[str, Any] | None = None,
):
    """Build one fresh normalized evaluation environment for a suite task."""

    if shield_params is not None and not isinstance(shield_params, Mapping):
        raise TypeError("shield_params must be a mapping")

    from stable_baselines3.common.vec_env import VecNormalize

    from gl_gym.RL.utils import make_vec_env
    from gl_gym.common.utils import load_env_params
    from gl_gym.experiments.suite_tasks import apply_task_to_env_params

    env_base_params, env_specific_params = load_env_params(
        suite.env_id,
        os.path.join("configs", "envs"),
    )
    env_base_params, env_specific_params = apply_task_to_env_params(
        env_base_params,
        env_specific_params,
        task,
    )
    env_id = suite.env_id
    if shield_params is not None:
        env_id = "ShieldedTomatoEnv"
        env_specific_params = deepcopy(env_specific_params)
        env_specific_params["action_shield_params"] = deepcopy(dict(shield_params))
    env = make_vec_env(
        env_id,
        env_base_params,
        env_specific_params,
        seed=666,
        n_envs=1,
        monitor_filename=None,
        vec_norm_kwargs=None,
        eval_env=True,
    )
    vec_path = Path(vecnormalize_path)
    if not vec_path.is_file():
        env.close()
        raise FileNotFoundError(f"VecNormalize statistics do not exist: {vec_path}")
    try:
        env = VecNormalize.load(str(vec_path), env)
    except BaseException:
        env.close()
        raise
    env.training = False
    env.norm_reward = False
    return env


def _validated_action_vector(
    value: Any,
    *,
    name: str,
    expected_shape: tuple[int, ...],
    bounded: bool,
) -> np.ndarray:
    raw = np.asarray(value)
    if not np.issubdtype(raw.dtype, np.number) or np.issubdtype(
        raw.dtype, np.complexfloating
    ):
        raise ValueError(f"action_shield {name} must be a numeric vector")
    vector = np.array(raw, dtype=np.float64, copy=True)
    if vector.ndim != 1 or vector.shape != expected_shape:
        raise ValueError(
            f"action_shield {name} must have exact shape {expected_shape}"
        )
    if not np.all(np.isfinite(vector)):
        raise ValueError(f"action_shield {name} must contain only finite values")
    if bounded and (np.any(vector < -1.0) or np.any(vector > 1.0)):
        raise ValueError(f"action_shield {name} must be within [-1, 1]")
    return vector.astype(np.float32)


def _shielded_executed_action(
    record: Any, requested_action: np.ndarray
) -> tuple[np.ndarray, dict[str, Any]]:
    if not isinstance(record, Mapping):
        raise ValueError("info['action_shield'] must be a mapping")
    if "executed_action" not in record:
        raise ValueError("info['action_shield'] must contain executed_action")

    executed_action = _validated_action_vector(
        record["executed_action"],
        name="executed_action",
        expected_shape=requested_action.shape,
        bounded=True,
    )
    if "requested_action" in record:
        recorded_requested = _validated_action_vector(
            record["requested_action"],
            name="requested_action",
            expected_shape=requested_action.shape,
            bounded=False,
        )
        if not np.array_equal(recorded_requested, requested_action):
            raise ValueError(
                "action_shield requested_action does not match the policy action"
            )

    if record.get("schema_version") != _ACTION_SHIELD_SCHEMA_VERSION:
        raise ValueError(
            "action_shield schema_version must match the canonical v2 schema"
        )
    attempts = record.get("candidate_attempts")
    if not isinstance(attempts, (list, tuple)):
        raise ValueError("action_shield candidate_attempts must be a sequence")
    if len(attempts) > len(DEFAULT_LAMBDAS):
        raise ValueError("action_shield candidate_attempts lambdas must be a prefix")

    attempt_lambdas: list[float] = []
    attempt_successes: list[bool] = []
    for attempt in attempts:
        if not isinstance(attempt, Mapping):
            raise ValueError("action_shield candidate_attempts entries must be mappings")
        lambda_value = attempt.get("lambda")
        if (
            isinstance(lambda_value, (bool, np.bool_))
            or not np.isscalar(lambda_value)
            or not np.isfinite(lambda_value)
        ):
            raise ValueError("action_shield candidate_attempt lambda must be finite")
        success = attempt.get("success")
        if not isinstance(success, (bool, np.bool_)):
            raise ValueError("action_shield candidate_attempt success must be boolean")
        attempt_lambdas.append(float(lambda_value))
        attempt_successes.append(bool(success))

    if tuple(attempt_lambdas) != DEFAULT_LAMBDAS[: len(attempt_lambdas)]:
        raise ValueError("action_shield candidate_attempts lambdas must be a prefix")

    selected_lambda = record.get("selected_lambda")
    if (
        isinstance(selected_lambda, (bool, np.bool_))
        or not np.isscalar(selected_lambda)
        or not np.isfinite(selected_lambda)
    ):
        raise ValueError("action_shield selected_lambda must be finite")
    if attempts:
        if attempt_successes != [False] * (len(attempts) - 1) + [True]:
            raise ValueError(
                "action_shield only the last candidate attempt may succeed"
            )
        if float(selected_lambda) != attempt_lambdas[-1]:
            raise ValueError(
                "action_shield final candidate lambda must equal selected_lambda"
            )
    elif float(selected_lambda) != 0.0:
        raise ValueError(
            "action_shield selected_lambda must be zero without candidate attempts"
        )
    return executed_action, deepcopy(dict(record))


def _predict(model: Any, obs: Any, states: Any, episode_starts: np.ndarray) -> tuple[Any, Any]:
    parameters = inspect.signature(model.predict).parameters.values()
    supports_keyword_arguments = any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD for parameter in parameters
    )
    parameter_names = {parameter.name for parameter in parameters}
    supports_recurrent_state = supports_keyword_arguments or {
        "state",
        "episode_start",
    }.issubset(parameter_names)

    if supports_recurrent_state:
        return model.predict(
            obs,
            state=states,
            episode_start=episode_starts,
            deterministic=True,
        )

    actions, _ = model.predict(obs, deterministic=True)
    return actions, states


def run_deterministic_episode(
    model: Any,
    env: Any,
    inference_mode: str | None = None,
    return_diagnostics: bool = False,
    failure_recorder: Any | None = None,
) -> dict[str, float] | tuple[dict[str, float], dict[str, Any]]:
    required_hooks = (
        "begin_inference_episode",
        "observe_inference_transition",
        "inference_episode_diagnostics",
        "end_inference_episode",
    )
    use_inference_hooks = inference_mode is not None
    if use_inference_hooks:
        missing = [
            name for name in required_hooks if not callable(getattr(model, name, None))
        ]
        if missing:
            raise TypeError(f"inference mode requires model hooks: {missing}")

    n_steps = env.get_attr("N")[0]
    totals = {
        "episode_return": 0.0,
        "EPI": 0.0,
        "revenue": 0.0,
        "heat_cost": 0.0,
        "co2_cost": 0.0,
        "elec_cost": 0.0,
        "temp_violation": 0.0,
        "co2_violation": 0.0,
        "rh_violation": 0.0,
        "twb_percent": float("nan"),
    }
    action_trace: list[np.ndarray] = []
    requested_action_trace: list[np.ndarray] = []
    action_shield_records: list[dict[str, Any]] = []
    shield_presence: bool | None = None
    model_diagnostics: dict[str, Any] = {}

    primary_error: BaseException | None = None
    diagnostics_enabled = False
    try:
        if use_inference_hooks:
            model.begin_inference_episode(inference_mode)

        if failure_recorder is not None:
            env.env_method("set_ode_diagnostics_enabled", True)
            diagnostics_enabled = True

        reset_result = env.reset()
        obs = reset_result[0] if isinstance(reset_result, tuple) else reset_result
        states = None
        episode_starts = np.ones((1,), dtype=bool)

        for step_index in range(n_steps):
            previous_obs = obs
            actions, states = _predict(model, obs, states, episode_starts)
            obs, rewards, dones, infos = env.step(actions)
            requested_action = np.asarray(actions[0], dtype=np.float32).copy()
            totals["episode_return"] += float(rewards[0])
            info = infos[0]
            done = bool(dones[0])
            capture_error: BaseException | None = None
            if failure_recorder is not None:
                try:
                    failure_recorder.record_step(
                        step_index=step_index,
                        policy_observation=np.asarray(previous_obs[0]).copy(),
                        reward=float(rewards[0]),
                        done=done,
                        info=info,
                    )
                except BaseException as error:
                    capture_error = error
            for key in (
                "EPI",
                "revenue",
                "heat_cost",
                "co2_cost",
                "elec_cost",
                "temp_violation",
                "co2_violation",
                "rh_violation",
            ):
                totals[key] += float(info.get(key, 0.0))

            if (
                capture_error is not None
                and done
                and step_index + 1 < n_steps
            ):
                early_done_error = RuntimeError(
                    "evaluation episode terminated before configured horizon: "
                    f"step {step_index + 1} of {n_steps}"
                )
                early_done_error.add_note(
                    "failure capsule capture also failed: "
                    f"{type(capture_error).__name__}: {capture_error}"
                )
                raise early_done_error
            if capture_error is not None:
                raise capture_error

            next_observation = None
            if use_inference_hooks and done:
                next_observation = info.get("terminal_observation")
                if next_observation is None:
                    raise ValueError(
                        "done transition requires a non-None "
                        "info['terminal_observation']"
                    )
            pending_early_done_error = None
            if done and step_index + 1 < n_steps:
                pending_early_done_error = RuntimeError(
                    "evaluation episode terminated before configured horizon: "
                    f"step {step_index + 1} of {n_steps}"
                )

            try:
                has_action_shield = "action_shield" in info
                if shield_presence is None:
                    shield_presence = has_action_shield
                elif shield_presence != has_action_shield:
                    raise ValueError(
                        "mixed action_shield presence within one evaluation episode"
                    )

                executed_action = requested_action
                if has_action_shield:
                    executed_action, detached_record = _shielded_executed_action(
                        info["action_shield"], requested_action
                    )
                    requested_action_trace.append(requested_action)
                    action_shield_records.append(detached_record)
            except ValueError as shield_error:
                if pending_early_done_error is None:
                    raise
                pending_early_done_error.add_note(
                    "action_shield evidence validation also failed: "
                    f"{type(shield_error).__name__}: {shield_error}"
                )
                raise pending_early_done_error from shield_error
            action_trace.append(executed_action)

            if use_inference_hooks:
                if not done:
                    next_observation = obs[0]
                model.observe_inference_transition(
                    previous_obs[0],
                    executed_action,
                    rewards[0],
                    next_observation,
                    done,
                    info,
                )
            if pending_early_done_error is not None:
                raise pending_early_done_error
            episode_starts = dones

        if use_inference_hooks:
            model_diagnostics = dict(model.inference_episode_diagnostics())
    except BaseException as error:
        primary_error = error
        raise
    finally:
        cleanup_errors: list[tuple[str, BaseException]] = []
        if diagnostics_enabled:
            try:
                env.env_method("set_ode_diagnostics_enabled", False)
            except BaseException as error:
                cleanup_errors.append(("ODE diagnostics disable", error))
        if use_inference_hooks:
            try:
                model.end_inference_episode()
            except BaseException as error:
                cleanup_errors.append(("inference episode cleanup", error))

        if primary_error is not None:
            for label, error in cleanup_errors:
                primary_error.add_note(
                    f"{label} also failed: {type(error).__name__}: {error}"
                )
        elif cleanup_errors:
            _, cleanup_primary = cleanup_errors[0]
            for label, error in cleanup_errors[1:]:
                cleanup_primary.add_note(
                    f"{label} also failed: {type(error).__name__}: {error}"
                )
            raise cleanup_primary

    if not return_diagnostics:
        return totals

    stacked_actions = (
        np.stack(action_trace).astype(np.float32, copy=False)
        if action_trace
        else np.empty((0,), dtype=np.float32)
    )
    diagnostics = {**model_diagnostics, "action_trace": stacked_actions}
    if action_shield_records:
        diagnostics.update(
            requested_action_trace=np.stack(requested_action_trace).astype(
                np.float32, copy=False
            ),
            action_shield_records=tuple(action_shield_records),
        )
    return totals, diagnostics


def validate_completed_run_paths(run: Any) -> None:
    if getattr(run, "status") != "completed":
        return

    model_path = Path(getattr(run, "model_path"))
    if not model_path.is_file():
        raise FileNotFoundError(f"completed run model_path does not exist: {model_path}")

    vecnormalize_path = Path(getattr(run, "vecnormalize_path"))
    if not vecnormalize_path.is_file():
        raise FileNotFoundError(
            f"completed run vecnormalize_path does not exist: {vecnormalize_path}"
        )


def write_eval_raw(rows: list[EvaluationMetricRow], path: str | Path) -> Path:
    if not rows:
        raise ValueError("write_eval_raw requires at least one row")

    csv_path = Path(path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    columns = [field.name for field in fields(EvaluationMetricRow)]

    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))

    return csv_path


def evaluation_key(algorithm: str, seed: int, task_id: str) -> tuple[str, int, str]:
    return (str(algorithm), int(seed), str(task_id))


def completed_eval_keys(path: str | Path) -> set[tuple[str, int, str]]:
    csv_path = Path(path)
    if not csv_path.is_file():
        return set()

    keys: set[tuple[str, int, str]] = set()
    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            keys.add(evaluation_key(row["algorithm"], int(row["seed"]), row["task_id"]))
    return keys


def append_eval_raw(row: EvaluationMetricRow, path: str | Path) -> Path:
    csv_path = Path(path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    columns = [field.name for field in fields(EvaluationMetricRow)]
    needs_header = not csv_path.exists() or csv_path.stat().st_size == 0

    with csv_path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        if needs_header:
            writer.writeheader()
        writer.writerow(asdict(row))

    return csv_path
