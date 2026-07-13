"""Deterministic evaluation helpers for robust experiment suites."""

from __future__ import annotations

import csv
import inspect
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

import numpy as np


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
    model_diagnostics: dict[str, Any] = {}

    primary_error: BaseException | None = None
    try:
        if use_inference_hooks:
            model.begin_inference_episode(inference_mode)

        reset_result = env.reset()
        obs = reset_result[0] if isinstance(reset_result, tuple) else reset_result
        states = None
        episode_starts = np.ones((1,), dtype=bool)

        for _ in range(n_steps):
            previous_obs = obs
            actions, states = _predict(model, obs, states, episode_starts)
            obs, rewards, dones, infos = env.step(actions)
            action_trace.append(np.asarray(actions[0], dtype=np.float32).copy())
            totals["episode_return"] += float(rewards[0])
            info = infos[0]
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

            if use_inference_hooks:
                done = bool(dones[0])
                if done:
                    next_observation = info.get("terminal_observation")
                    if next_observation is None:
                        raise ValueError(
                            "done transition requires a non-None "
                            "info['terminal_observation']"
                        )
                else:
                    next_observation = obs[0]
                model.observe_inference_transition(
                    previous_obs[0],
                    actions[0],
                    rewards[0],
                    next_observation,
                    done,
                    info,
                )
            episode_starts = dones

        if use_inference_hooks:
            model_diagnostics = dict(model.inference_episode_diagnostics())
    except BaseException as error:
        primary_error = error
        raise
    finally:
        if use_inference_hooks:
            try:
                model.end_inference_episode()
            except Exception:
                if primary_error is None:
                    raise

    if not return_diagnostics:
        return totals

    stacked_actions = (
        np.stack(action_trace).astype(np.float32, copy=False)
        if action_trace
        else np.empty((0,), dtype=np.float32)
    )
    return totals, {**model_diagnostics, "action_trace": stacked_actions}


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
