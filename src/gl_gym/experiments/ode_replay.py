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
            object.__setattr__(
                self, "final_state", np.array(final_state, copy=True)
            )


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
        not outcome.success for outcome in controls if outcome.available
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


def _validate_replay_inputs(inputs: Mapping[str, np.ndarray]) -> None:
    for name in ("x0", "u", "p_dyn"):
        value = np.asarray(inputs[name])
        if value.dtype.kind not in "biuf" or not np.isfinite(value).all():
            raise ValueError(f"{name} must be numeric and finite")
    dt = np.asarray(inputs["dt"])
    if dt.size != 1 or dt.dtype.kind not in "biuf" or not np.isfinite(dt).all():
        raise ValueError("dt must be a finite scalar")


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
            integrator = integrator_factory(**factory_args)
            if variant == "rule_based_control":
                try:
                    controller = controller_factory()
                except Exception as error:
                    available = False
                    raise error
                environment = SimpleNamespace(
                    nu=int(inputs["nu"]),
                    day_of_year=float(inputs["day_of_year"]),
                    hour_of_day=float(inputs["hour_of_day"]),
                )
                control = controller.predict(
                    inputs["x0"], inputs["weather"], environment
                )
            state: Any = inputs["x0"]
            for _ in range(substeps):
                result = integrator(x0=state, u=control, p=inputs["p_dyn"])
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
    _validate_replay_inputs(capsule.failure_inputs)
    outcomes = tuple(
        _run_variant(
            variant,
            capsule.failure_inputs,
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
