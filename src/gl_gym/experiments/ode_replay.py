"""Counterfactual replay and classification of captured ODE failures."""

from __future__ import annotations

from dataclasses import dataclass
import math
from time import perf_counter
from types import SimpleNamespace
from typing import Any, Callable, Iterable, Mapping
import warnings as python_warnings

import numpy as np

from gl_gym.common.utils import load_model_hyperparams
from gl_gym.environments.baseline import RuleBasedController
from gl_gym.environments.models.utils import define_model
from gl_gym.experiments.ode_failure import LoadedFailureCapsule


VARIANT_NAMES = (
    "original",
    "previous_control",
    "rule_based_control",
    "original_2x_substeps",
    "original_4x_substeps",
    "original_strict_tolerance",
)
_CONTROL_VARIANTS = ("previous_control", "rule_based_control")
_SOLVER_VARIANTS = (
    "original_2x_substeps",
    "original_4x_substeps",
    "original_strict_tolerance",
)


@dataclass(frozen=True, slots=True)
class ReplayOutcome:
    variant: str
    available: bool
    success: bool
    elapsed_seconds: float
    final_state: np.ndarray | None
    exception_type: str | None = None
    exception_message: str | None = None
    warnings: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        elapsed = float(self.elapsed_seconds)
        if not math.isfinite(elapsed) or elapsed < 0:
            raise ValueError("elapsed_seconds must be finite and non-negative")
        object.__setattr__(self, "elapsed_seconds", elapsed)
        object.__setattr__(self, "warnings", tuple(str(item) for item in self.warnings))
        if self.final_state is not None:
            final_state = np.asarray(self.final_state)
            if final_state.dtype.kind not in "biuf" or not np.isfinite(final_state).all():
                raise ValueError("final_state must be numeric and finite")
            owned_final_state = np.array(final_state, copy=True)
            owned_final_state.setflags(write=False)
            object.__setattr__(self, "final_state", owned_final_state)


@dataclass(frozen=True, slots=True)
class ReplayReport:
    failure_id: str
    classification: str
    outcomes: tuple[ReplayOutcome, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "outcomes", tuple(self.outcomes))


def build_rule_based_controller() -> RuleBasedController:
    """Construct the configured formal rule-based controller."""
    parameters = load_model_hyperparams("rule_based", "TomatoEnv")
    return RuleBasedController(**parameters)


def classify_replay_outcomes(outcomes: Iterable[ReplayOutcome]) -> str:
    """Conservatively classify a complete set of replay counterfactuals."""
    ordered = tuple(outcomes)
    names = tuple(outcome.variant for outcome in ordered)
    if len(ordered) != len(VARIANT_NAMES) or set(names) != set(VARIANT_NAMES):
        raise ValueError("outcomes must contain exactly one of every replay variant")
    by_name = {outcome.variant: outcome for outcome in ordered}
    original = by_name["original"]
    if not original.available:
        raise ValueError("original replay outcome must be available")
    if original.success:
        return "non_reproduced"

    controls = tuple(by_name[name] for name in _CONTROL_VARIANTS)
    solvers = tuple(by_name[name] for name in _SOLVER_VARIANTS)
    control_success = any(outcome.available and outcome.success for outcome in controls)
    solver_success = any(outcome.available and outcome.success for outcome in solvers)
    if control_success and solver_success:
        return "mixed_control_and_solver_sensitivity"
    if control_success:
        return "policy_induced_control_instability"
    if solver_success and all(
        outcome.available and not outcome.success for outcome in controls
    ):
        return "solver_step_sensitivity"
    if all(outcome.available and not outcome.success for outcome in ordered):
        return "state_or_model_domain_failure"
    return "insufficient_counterfactual_evidence"


def _factory_arguments(inputs: Mapping[str, np.ndarray], dt: float) -> dict[str, Any]:
    return {
        "nx": int(inputs["nx"]),
        "nu": int(inputs["nu"]),
        "nd": int(inputs["nd"]),
        "n_params": int(inputs["n_params"]),
        "dt": dt,
    }


def _numeric_scalar(inputs: Mapping[str, np.ndarray], name: str) -> float:
    value = np.asarray(inputs[name])
    if value.shape != () or value.dtype.kind not in "iuf":
        raise ValueError(f"{name} must be a non-boolean numeric scalar")
    scalar = float(value)
    if not math.isfinite(scalar):
        raise ValueError(f"{name} must be finite")
    return scalar


def _integral_scalar(
    inputs: Mapping[str, np.ndarray], name: str, *, positive: bool
) -> int:
    value = np.asarray(inputs[name])
    if value.shape != () or value.dtype.kind not in "iu":
        raise ValueError(f"{name} must be a non-boolean integral scalar")
    scalar = int(value)
    if (positive and scalar <= 0) or (not positive and scalar < 0):
        qualifier = "positive" if positive else "non-negative"
        raise ValueError(f"{name} must be {qualifier}")
    return scalar


def _finite_vector(
    inputs: Mapping[str, np.ndarray], name: str, expected_size: int
) -> np.ndarray:
    value = np.asarray(inputs[name])
    if value.shape != (expected_size,):
        raise ValueError(f"{name} shape must be ({expected_size},)")
    if value.dtype.kind not in "biuf" or not np.isfinite(value).all():
        raise ValueError(f"{name} must be numeric and finite")
    return value


def _snapshot_and_validate_inputs(
    source: Mapping[str, np.ndarray],
) -> dict[str, np.ndarray]:
    required = {
        "x0",
        "u",
        "previous_control",
        "requested_action",
        "weather",
        "sampled_parameters",
        "p_dyn",
        "timestep",
        "day_of_year",
        "hour_of_day",
        "dt",
        "nx",
        "nu",
        "nd",
        "n_params",
    }
    missing = required.difference(source)
    if missing:
        raise ValueError(f"missing replay inputs: {', '.join(sorted(missing))}")
    inputs = {name: np.array(value, copy=True) for name, value in source.items()}
    nx = _integral_scalar(inputs, "nx", positive=True)
    nu = _integral_scalar(inputs, "nu", positive=True)
    nd = _integral_scalar(inputs, "nd", positive=True)
    n_params = _integral_scalar(inputs, "n_params", positive=True)
    _integral_scalar(inputs, "timestep", positive=False)
    dt = _numeric_scalar(inputs, "dt")
    if dt <= 0:
        raise ValueError("dt must be positive")
    _numeric_scalar(inputs, "day_of_year")
    _numeric_scalar(inputs, "hour_of_day")
    _finite_vector(inputs, "x0", nx)
    for name in ("u", "previous_control", "requested_action"):
        _finite_vector(inputs, name, nu)
    weather = _finite_vector(inputs, "weather", nd)
    sampled = _finite_vector(inputs, "sampled_parameters", n_params)
    p_dyn = _finite_vector(inputs, "p_dyn", nd + n_params)
    if not np.array_equal(p_dyn, np.concatenate((weather, sampled))):
        raise ValueError("p_dyn must equal concat(weather, sampled_parameters)")
    return inputs


def _final_array(result: Any, nx: int) -> np.ndarray:
    if not isinstance(result, Mapping) or "xf" not in result:
        raise ValueError("integrator result must contain 'xf'")
    value = result["xf"]
    if hasattr(value, "full"):
        value = value.full()
    final = np.asarray(value)
    if final.shape == (nx, 1):
        final = final[:, 0]
    if final.shape != (nx,):
        raise ValueError(f"final state shape must be ({nx},), got {final.shape}")
    if final.dtype.kind not in "biuf" or not np.isfinite(final).all():
        raise ValueError("final state must be numeric and finite")
    return np.array(final, copy=True)


def _rule_control(value: Any, nu: int) -> np.ndarray:
    control = np.asarray(value)
    if control.shape != (nu,):
        raise ValueError(f"rule_based_control shape must be ({nu},)")
    if control.dtype.kind not in "biuf" or not np.isfinite(control).all():
        raise ValueError("rule_based_control must be numeric and finite")
    return np.array(control, copy=True)


def _run_variant(
    variant: str,
    inputs: Mapping[str, np.ndarray],
    integrator_factory: Callable[..., Any],
    controller_factory: Callable[[], Any],
) -> ReplayOutcome:
    nx = int(inputs["nx"])
    dt = float(inputs["dt"])
    substeps = 1
    control: Any = inputs["u"]
    factory_args = _factory_arguments(inputs, dt)
    if variant == "previous_control":
        control = inputs["previous_control"]
    elif variant == "original_2x_substeps":
        substeps = 2
        factory_args["dt"] = dt / 2.0
    elif variant == "original_4x_substeps":
        substeps = 4
        factory_args["dt"] = dt / 4.0
    elif variant == "original_strict_tolerance":
        factory_args["integrator_options"] = {"abstol": 1e-6, "reltol": 1e-6}

    started = perf_counter()
    available = True
    final_state = None
    exception_type = None
    exception_message = None
    with python_warnings.catch_warnings(record=True) as caught:
        python_warnings.simplefilter("always")
        try:
            if variant == "rule_based_control":
                try:
                    controller = controller_factory()
                    environment = SimpleNamespace(
                        nu=int(inputs["nu"]),
                        day_of_year=float(inputs["day_of_year"]),
                        hour_of_day=float(inputs["hour_of_day"]),
                    )
                    control = _rule_control(
                        controller.predict(
                            np.array(inputs["x0"], copy=True),
                            np.array(inputs["weather"], copy=True),
                            environment,
                        ),
                        int(inputs["nu"]),
                    )
                except Exception:
                    available = False
                    raise
            integrator = integrator_factory(**factory_args)
            state: Any = np.array(inputs["x0"], copy=True)
            for _ in range(substeps):
                result = integrator(
                    x0=np.array(state, copy=True),
                    u=np.array(control, copy=True),
                    p=np.array(inputs["p_dyn"], copy=True),
                )
                state = _final_array(result, nx)
            final_state = np.array(state, copy=True)
        except Exception as error:
            exception_type = type(error).__name__
            exception_message = str(error)
    elapsed = perf_counter() - started
    warning_messages = tuple(str(item.message) for item in caught)
    success = exception_type is None
    return ReplayOutcome(
        variant=variant,
        available=available,
        success=success,
        elapsed_seconds=elapsed,
        final_state=final_state if success else None,
        exception_type=exception_type,
        exception_message=exception_message,
        warnings=warning_messages,
    )


def replay_failure_capsule(
    capsule: LoadedFailureCapsule,
    integrator_factory: Callable[..., Any] = define_model,
    controller_factory: Callable[[], Any] = build_rule_based_controller,
) -> ReplayReport:
    """Replay all fixed counterfactual variants for a loaded failure capsule."""
    inputs = _snapshot_and_validate_inputs(capsule.failure_inputs)
    outcomes = tuple(
        _run_variant(
            variant,
            inputs,
            integrator_factory,
            controller_factory,
        )
        for variant in VARIANT_NAMES
    )
    classification = classify_replay_outcomes(outcomes)
    return ReplayReport(
        failure_id=str(capsule.manifest["failure_id"]),
        classification=classification,
        outcomes=outcomes,
    )
