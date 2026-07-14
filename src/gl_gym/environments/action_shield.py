"""Pure minimal-feasibility action projection.

This module deliberately knows nothing about a particular simulator.  It builds a
fixed grid of convex action candidates and records immutable evidence from calls
to a caller-supplied integration function.
"""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Callable, Iterable

import numpy as np
from numpy.typing import ArrayLike, NDArray


DEFAULT_LAMBDAS = (1.0 / 16.0, 1.0 / 8.0, 1.0 / 4.0, 1.0 / 2.0, 1.0)
_SCHEMA_VERSION = "minimal-feasibility-action-shield-v1"


def _immutable_vector(value: ArrayLike, *, name: str) -> NDArray[np.float64]:
    raw = np.asarray(value)
    if not np.issubdtype(raw.dtype, np.number) or np.issubdtype(
        raw.dtype, np.complexfloating
    ):
        raise ValueError(f"{name} must be a numeric vector")
    vector = np.array(raw, dtype=np.float64, copy=True)
    if vector.ndim != 1 or vector.size == 0:
        raise ValueError(f"{name} must be a nonempty one-dimensional vector")
    if not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must contain only finite values")
    vector.setflags(write=False)
    return vector


def _validated_lambdas(lambdas: Iterable[float]) -> tuple[float, ...]:
    try:
        values = tuple(lambdas)
    except TypeError as exc:
        raise ValueError("lambdas must be the fixed default grid") from exc
    if values != DEFAULT_LAMBDAS:
        raise ValueError("lambdas must use the fixed default grid")
    return DEFAULT_LAMBDAS


@dataclass(frozen=True, slots=True)
class ActionShieldConfig:
    lambdas: tuple[float, ...] = DEFAULT_LAMBDAS
    schema_version: str = _SCHEMA_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "lambdas", _validated_lambdas(self.lambdas))
        if self.schema_version != _SCHEMA_VERSION:
            raise ValueError(f"schema_version must be {_SCHEMA_VERSION!r}")


@dataclass(frozen=True, slots=True)
class ActionCandidate:
    lambda_value: float
    action: NDArray[np.float64]

    def __post_init__(self) -> None:
        if not np.isscalar(self.lambda_value) or not np.isfinite(self.lambda_value):
            raise ValueError("lambda_value must be finite")
        object.__setattr__(self, "lambda_value", float(self.lambda_value))
        object.__setattr__(self, "action", _immutable_vector(self.action, name="action"))


@dataclass(frozen=True, slots=True)
class CandidateAttempt:
    lambda_value: float
    action: NDArray[np.float64]
    success: bool
    elapsed_seconds: float
    exception_type: str | None
    exception_message: str | None

    def __post_init__(self) -> None:
        if not np.isscalar(self.lambda_value) or not np.isfinite(self.lambda_value):
            raise ValueError("lambda_value must be finite")
        elapsed = float(self.elapsed_seconds)
        if not np.isfinite(elapsed) or elapsed < 0.0:
            raise ValueError("elapsed_seconds must be finite and nonnegative")
        object.__setattr__(self, "lambda_value", float(self.lambda_value))
        object.__setattr__(self, "action", _immutable_vector(self.action, name="action"))
        object.__setattr__(self, "success", bool(self.success))
        object.__setattr__(self, "elapsed_seconds", elapsed)


@dataclass(frozen=True, slots=True)
class ProjectionResult:
    selected: ActionCandidate | None
    final_state: NDArray[np.float64] | None
    attempts: tuple[CandidateAttempt, ...]

    def __post_init__(self) -> None:
        if self.selected is not None and not isinstance(self.selected, ActionCandidate):
            raise TypeError("selected must be an ActionCandidate or None")
        if self.final_state is not None:
            object.__setattr__(
                self,
                "final_state",
                _immutable_vector(self.final_state, name="final_state"),
            )
        attempts = tuple(self.attempts)
        if not all(isinstance(attempt, CandidateAttempt) for attempt in attempts):
            raise TypeError("attempts must contain only CandidateAttempt values")
        object.__setattr__(self, "attempts", attempts)


def control_to_reference_action(
    target: ArrayLike,
    previous: ArrayLike,
    delta_u_max: float,
) -> NDArray[np.float64]:
    """Convert a physical control target to a normalized, rate-limited action."""

    target_vector = _immutable_vector(target, name="target")
    previous_vector = _immutable_vector(previous, name="previous")
    if target_vector.shape != previous_vector.shape:
        raise ValueError("target and previous must have exactly matching shapes")
    if not np.isscalar(delta_u_max):
        raise ValueError("delta_u_max must be a strictly positive finite scalar")
    delta = float(delta_u_max)
    if not np.isfinite(delta) or delta <= 0.0:
        raise ValueError("delta_u_max must be strictly positive and finite")
    return _immutable_vector(
        np.clip((target_vector - previous_vector) / delta, -1.0, 1.0),
        name="reference_action",
    )


def build_candidates(
    policy_action: ArrayLike,
    reference_action: ArrayLike,
    lambdas: Iterable[float] = DEFAULT_LAMBDAS,
) -> tuple[ActionCandidate, ...]:
    """Build the fixed, ordered convex candidate grid."""

    grid = _validated_lambdas(lambdas)
    policy = _immutable_vector(policy_action, name="policy_action")
    reference = _immutable_vector(reference_action, name="reference_action")
    if policy.shape != reference.shape:
        raise ValueError("policy_action and reference_action must have exactly matching shapes")
    if np.any(policy < -1.0) or np.any(policy > 1.0):
        raise ValueError("policy_action must lie within [-1, 1]")
    if np.any(reference < -1.0) or np.any(reference > 1.0):
        raise ValueError("reference_action must lie within [-1, 1]")
    return tuple(
        ActionCandidate(
            lambda_value,
            lambda_value * policy + (1.0 - lambda_value) * reference,
        )
        for lambda_value in grid
    )


def project_first_feasible(
    policy_action: ArrayLike,
    reference_action: ArrayLike,
    integrate: Callable[[NDArray[np.float64]], ArrayLike],
    config: ActionShieldConfig,
) -> ProjectionResult:
    """Return the first candidate whose integration produces a valid state."""

    if not isinstance(config, ActionShieldConfig):
        raise TypeError("config must be an ActionShieldConfig")
    if not callable(integrate):
        raise TypeError("integrate must be callable")
    candidates = build_candidates(policy_action, reference_action, config.lambdas)
    attempts: list[CandidateAttempt] = []

    for candidate in candidates:
        integration_action = _immutable_vector(candidate.action, name="candidate action")
        started = perf_counter()
        try:
            state = _immutable_vector(integrate(integration_action), name="integration output")
        except Exception as exc:
            attempts.append(
                CandidateAttempt(
                    candidate.lambda_value,
                    candidate.action,
                    False,
                    perf_counter() - started,
                    type(exc).__name__,
                    str(exc),
                )
            )
            continue

        attempts.append(
            CandidateAttempt(
                candidate.lambda_value,
                candidate.action,
                True,
                perf_counter() - started,
                None,
                None,
            )
        )
        return ProjectionResult(candidate, state, tuple(attempts))

    return ProjectionResult(None, None, tuple(attempts))
