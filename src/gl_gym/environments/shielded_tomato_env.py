"""Evaluation-only Tomato environment with transactional action recovery."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from time import perf_counter
from typing import Any
import traceback

import casadi as ca
import numpy as np

from gl_gym.environments.action_shield import (
    ActionShieldConfig,
    CandidateAttempt,
    control_to_reference_action,
    project_first_feasible,
)
from gl_gym.environments.baseline import RuleBasedController
from gl_gym.environments.models.utils import FORMAL_CVODES_OPTIONS, define_model
from gl_gym.environments.noise import parametric_crop_uncertainty
from gl_gym.environments.tomato_env import TomatoEnv


class _ConstructionFailure(BaseException):
    """Escape projection's solver-only ``Exception`` handling."""

    def __init__(self, error: Exception) -> None:
        self.error = error
        self.error_traceback = error.__traceback__


class _PostCallFailure(BaseException):
    """Carry a post-integrator error through the projection exception boundary."""

    def __init__(self, error: Exception) -> None:
        self.error = error
        self.error_traceback = error.__traceback__


class ShieldedTomatoEnv(TomatoEnv):
    """A distinct evaluation environment that shields solver-failing actions."""

    def __init__(
        self,
        *,
        action_shield_params: Mapping[str, Any],
        **kwargs: Any,
    ) -> None:
        if not isinstance(action_shield_params, Mapping):
            raise TypeError("action_shield_params must be a mapping")
        params = deepcopy(dict(action_shield_params))
        self.action_shield_params = params
        self.action_shield_controller = RuleBasedController(**params)
        self.action_shield_config = ActionShieldConfig()
        super().__init__(**kwargs)

    @staticmethod
    def _vector(value: Any, *, name: str, size: int, bounded: bool = False) -> np.ndarray:
        raw = np.asarray(value)
        if not np.issubdtype(raw.dtype, np.number) or np.issubdtype(
            raw.dtype, np.complexfloating
        ):
            raise ValueError(f"{name} must be a numeric vector")
        vector = np.array(raw, dtype=float, copy=True)
        if vector.shape != (size,):
            raise ValueError(f"{name} must have exact shape ({size},)")
        if not np.all(np.isfinite(vector)):
            raise ValueError(f"{name} must contain only finite values")
        if bounded and (np.any(vector < -1.0) or np.any(vector > 1.0)):
            raise ValueError(f"{name} must lie within [-1, 1]")
        return vector

    def _snapshot(self) -> dict[str, Any]:
        fields = (
            "x",
            "u",
            "x_prev",
            "obs",
            "timestep",
            "day_of_year",
            "hour_of_day",
            "terminated",
        )
        reward_state = {
            key: (value if key == "env" else deepcopy(value))
            for key, value in self.reward.__dict__.items()
        }
        return {
            "fields": {name: deepcopy(getattr(self, name)) for name in fields},
            "reward": reward_state,
            "rng": deepcopy(self._np_random.bit_generator.state),
            "F": self.F,
        }

    def _restore(self, snapshot: dict[str, Any], *, restore_rng: bool = True) -> None:
        for name, value in snapshot["fields"].items():
            setattr(self, name, deepcopy(value))
        self.reward.__dict__.clear()
        self.reward.__dict__.update(
            {
                key: (value if key == "env" else deepcopy(value))
                for key, value in snapshot["reward"].items()
            }
        )
        if restore_rng:
            self._np_random.bit_generator.state = deepcopy(snapshot["rng"])
        self.F = snapshot["F"]

    def _restore_rng(self, state: dict[str, Any]) -> None:
        self._np_random.bit_generator.state = deepcopy(state)

    @classmethod
    def _detach_payload(cls, value: Any) -> Any:
        if isinstance(value, np.ndarray):
            return np.array(value, copy=True)
        if isinstance(value, Mapping):
            return {key: cls._detach_payload(item) for key, item in value.items()}
        if isinstance(value, list):
            return [cls._detach_payload(item) for item in value]
        if isinstance(value, tuple):
            return tuple(cls._detach_payload(item) for item in value)
        if isinstance(value, set):
            return {cls._detach_payload(item) for item in value}
        if value is None or isinstance(value, (str, bytes, bool, int, float)):
            return value
        return deepcopy(value)

    def _state_from_result(self, result: Any) -> np.ndarray:
        raw_state = result["xf"]
        if hasattr(raw_state, "full"):
            raw_state = raw_state.full()
        state = np.asarray(raw_state).reshape(-1)
        return self._vector(state, name="final state", size=self.nx)

    def _fresh_integrator(self):
        return define_model(
            nx=self.nx,
            nu=self.nu,
            nd=self.nd,
            n_params=self.num_params,
            dt=self.dt,
            integrator_options=dict(FORMAL_CVODES_OPTIONS),
        )

    @staticmethod
    def _attempt_record(attempt: CandidateAttempt) -> dict[str, Any]:
        return {
            "lambda": float(attempt.lambda_value),
            "action": attempt.action.tolist(),
            "success": bool(attempt.success),
            "elapsed_seconds": float(attempt.elapsed_seconds),
            "exception_type": attempt.exception_type,
            "exception_message": attempt.exception_message,
        }

    def _diagnostic_failure(
        self,
        *,
        error: Exception,
        formatted_traceback: str,
        x0: np.ndarray,
        previous_control: np.ndarray,
        requested_action: np.ndarray,
        attempted_control: np.ndarray,
        weather: np.ndarray,
        sampled_parameters: np.ndarray,
        p_dyn: Any,
        timestep: int,
        day_of_year: float,
        hour_of_day: float,
    ) -> dict[str, Any]:
        return {
            "x0": x0.copy(),
            "u": attempted_control.copy(),
            "previous_control": previous_control.copy(),
            "requested_action": requested_action.copy(),
            "weather": weather.copy(),
            "sampled_parameters": sampled_parameters.copy(),
            "p_dyn": np.asarray(p_dyn, dtype=float).reshape(-1).copy(),
            "timestep": timestep,
            "day_of_year": day_of_year,
            "hour_of_day": hour_of_day,
            "dt": float(self.dt),
            "nx": int(self.nx),
            "nu": int(self.nu),
            "nd": int(self.nd),
            "n_params": int(self.num_params),
            "solver_options": dict(FORMAL_CVODES_OPTIONS),
            "exception_type": type(error).__name__,
            "exception_message": str(error),
            "traceback": formatted_traceback,
        }

    def step(self, action: np.ndarray):
        started = perf_counter()
        requested_action = self._vector(
            action, name="requested action", size=self.nu, bounded=True
        )
        snapshot = self._snapshot()
        raw_observation = np.array(self.obs, copy=True)
        x0 = np.array(self.x, dtype=float, copy=True)
        previous_control = np.array(self.u, dtype=float, copy=True)
        weather = np.array(self.weather_data[self.timestep], dtype=float, copy=True)
        pre_timestep = self.timestep
        pre_day = self.day_of_year
        pre_hour = self.hour_of_day

        try:
            requested_control = self._vector(
                self.action_to_control(requested_action),
                name="requested control",
                size=self.nu,
            )
            raw_sampled_parameters = parametric_crop_uncertainty(
                self.p, self.uncertainty_scale, self._np_random
            )
            post_sample_rng = deepcopy(self._np_random.bit_generator.state)
            sampled_parameters = self._vector(
                raw_sampled_parameters,
                name="sampled parameters",
                size=self.num_params,
            )
            p_dyn = ca.vertcat(ca.DM(weather), sampled_parameters)
            x0_input = ca.DM(x0)
            requested_control_input = ca.DM(requested_control)

            original_error: Exception | None = None
            original_traceback: str | None = None
            self._restore_rng(post_sample_rng)
            try:
                result = self.F(
                    x0=x0_input, u=requested_control_input, p=p_dyn
                )
            except Exception as error:
                original_error = error
                original_traceback = traceback.format_exc()
                # A failed foreign integrator must not contaminate recovery inputs.
                self._restore(snapshot, restore_rng=False)
                self._restore_rng(post_sample_rng)
            else:
                # Malformed solver output is a construction/programming error,
                # not evidence that the requested action is infeasible.
                final_state = self._state_from_result(result)

            if original_error is None:
                executed_action = requested_action.copy()
                executed_control = requested_control
                selected_integrator = snapshot["F"]
                reference_action = None
                selected_lambda = 0.0
                attempts: tuple[CandidateAttempt, ...] = ()
            else:
                self._restore_rng(post_sample_rng)
                target = self._vector(
                    self.action_shield_controller.predict(
                        x0.copy(), weather.copy(), self
                    ),
                    name="reference control target",
                    size=self.nu,
                )
                self._restore(snapshot, restore_rng=False)
                self._restore_rng(post_sample_rng)
                reference_action = control_to_reference_action(
                    target, previous_control, self.delta_u_max
                )
                candidate_controls: list[np.ndarray] = []
                candidate_integrators: list[Any] = []

                def integrate(candidate_action: np.ndarray) -> np.ndarray:
                    try:
                        self._restore(snapshot, restore_rng=False)
                        self._restore_rng(post_sample_rng)
                        candidate_control = self._vector(
                            self.action_to_control(candidate_action),
                            name="candidate control",
                            size=self.nu,
                        )
                        candidate_controls.append(candidate_control.copy())
                        candidate_control_input = ca.DM(candidate_control)
                        integrator = self._fresh_integrator()
                        candidate_integrators.append(integrator)
                    except Exception as construction_error:
                        raise _ConstructionFailure(construction_error) from None
                    self._restore_rng(post_sample_rng)
                    result = integrator(
                        x0=x0_input, u=candidate_control_input, p=p_dyn
                    )
                    try:
                        return self._state_from_result(result)
                    except Exception as post_call_error:
                        raise _PostCallFailure(post_call_error) from None

                try:
                    projection = project_first_feasible(
                        requested_action,
                        reference_action,
                        integrate,
                        self.action_shield_config,
                    )
                except _ConstructionFailure as failure:
                    raise failure.error.with_traceback(failure.error_traceback) from None
                except _PostCallFailure as failure:
                    raise failure.error.with_traceback(failure.error_traceback) from None
                attempts = projection.attempts
                if projection.selected is None or projection.final_state is None:
                    exhausted = RuntimeError(
                        "action shield exhausted all legal candidates"
                    )
                    for attempt in attempts:
                        exhausted.add_note(
                            f"lambda={attempt.lambda_value}: "
                            f"{attempt.exception_type}: {attempt.exception_message}"
                        )
                    raise exhausted from original_error
                executed_action = np.array(projection.selected.action, copy=True)
                executed_control = candidate_controls[len(attempts) - 1]
                selected_integrator = candidate_integrators[len(attempts) - 1]
                selected_lambda = float(projection.selected.lambda_value)
                final_state = np.array(projection.final_state, copy=True)

            self._restore(snapshot, restore_rng=False)
            self._restore_rng(post_sample_rng)
            self.x = final_state
            self.u = executed_control
            self.F = selected_integrator
            self.day_of_year += (self.dt / self.c) % 365
            self.hour_of_day = (self.hour_of_day + self.dt / 3600) % 24
            self.obs = self._get_obs()
            if self._terminalState():
                self.terminated = True
            reward = self._get_reward()
            info = self._get_info()

            difference = requested_action - executed_action
            shield_record = {
                "schema_version": self.action_shield_config.schema_version,
                "intervened": original_error is not None,
                "requested_action": requested_action.tolist(),
                "reference_action": (
                    None if reference_action is None else reference_action.tolist()
                ),
                "executed_action": executed_action.tolist(),
                "executed_control": executed_control.tolist(),
                "selected_lambda": selected_lambda,
                "candidate_attempts": [
                    self._attempt_record(attempt) for attempt in attempts
                ],
                "intervention_l1": float(np.linalg.norm(difference, ord=1)),
                "intervention_l2": float(np.linalg.norm(difference, ord=2)),
                "intervention_linf": float(np.linalg.norm(difference, ord=np.inf)),
                "per_channel_changed": (difference != 0.0).tolist(),
                "extra_solver_attempts": len(attempts),
                "elapsed_seconds": float(perf_counter() - started),
                "original_failure": (
                    None
                    if original_error is None
                    else {
                        "exception_type": type(original_error).__name__,
                        "exception_message": str(original_error),
                    }
                ),
            }
            info["action_shield"] = shield_record

            if self._ode_diagnostics_enabled:
                info["diagnostic_transition"] = {
                    "raw_observation": raw_observation.copy(),
                    "requested_action": requested_action.copy(),
                    "previous_control": previous_control.copy(),
                    "executed_control": executed_control.copy(),
                    "raw_next_observation": np.array(self.obs, copy=True),
                    "raw_next_observation_available": True,
                }
                if original_error is not None:
                    info["integration_failure"] = self._diagnostic_failure(
                        error=original_error,
                        formatted_traceback=original_traceback or "",
                        x0=x0,
                        previous_control=previous_control,
                        requested_action=requested_action,
                        attempted_control=requested_control,
                        weather=weather,
                        sampled_parameters=sampled_parameters,
                        p_dyn=p_dyn,
                        timestep=pre_timestep,
                        day_of_year=pre_day,
                        hour_of_day=pre_hour,
                    )

            info = self._detach_payload(info)
            self.timestep += 1
            self.x_prev = np.copy(self.x)
            return self.obs, reward, self.terminated, False, info
        except BaseException as error:
            try:
                self._restore(snapshot)
            except BaseException as restoration_error:
                error.add_note(
                    "transaction restoration also failed: "
                    f"{type(restoration_error).__name__}: {restoration_error}"
                )
            raise
