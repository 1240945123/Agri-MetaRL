from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from gl_gym.environments.action_shield import (
    DEFAULT_LAMBDAS,
    ActionCandidate,
    ActionShieldConfig,
    CandidateAttempt,
    ProjectionResult,
    build_candidates,
    control_to_reference_action,
    project_first_feasible,
)


def test_control_to_reference_action_rate_limits_each_element() -> None:
    target = np.array([13.0, -1.0, 2.5])
    previous = np.array([10.0, 1.0, 2.0])

    reference = control_to_reference_action(
        target, previous, delta_u_max=np.array([2.0, 2.0, 2.0])
    )

    np.testing.assert_allclose(reference, [1.0, -1.0, 0.25])
    assert not reference.flags.writeable


@pytest.mark.parametrize(
    "delta",
    [
        np.array([0.0]),
        np.array([-1.0]),
        np.array([np.nan]),
        np.array([np.inf]),
    ],
)
def test_control_to_reference_action_rejects_invalid_delta(delta: np.ndarray) -> None:
    with pytest.raises(ValueError):
        control_to_reference_action(np.array([1.0]), np.array([0.0]), delta)


def test_control_to_reference_action_rejects_mismatched_shapes() -> None:
    with pytest.raises(ValueError, match="shape"):
        control_to_reference_action(np.ones(2), np.ones(3), np.ones(2))
    with pytest.raises(ValueError, match="shape"):
        control_to_reference_action(np.ones(2), np.ones(2), np.ones(3))


def test_control_to_reference_action_rejects_scalar_delta() -> None:
    with pytest.raises(ValueError, match="shape"):
        control_to_reference_action(np.ones(2), np.zeros(2), 0.1)


def test_build_candidates_uses_exact_order_and_convex_formula() -> None:
    policy = np.array([1.0, -1.0])
    reference = np.array([-1.0, 0.5])

    candidates = build_candidates(policy, reference)

    assert tuple(candidate.lambda_value for candidate in candidates) == DEFAULT_LAMBDAS
    for candidate, lambda_value in zip(candidates, DEFAULT_LAMBDAS, strict=True):
        expected = (1.0 - lambda_value) * policy + lambda_value * reference
        np.testing.assert_allclose(candidate.action, expected)
        assert not candidate.action.flags.writeable

    np.testing.assert_allclose(candidates[0].action, [0.875, -0.90625])


def test_config_is_frozen_slotted_and_rejects_alternate_grid() -> None:
    config = ActionShieldConfig()

    assert config.lambdas == DEFAULT_LAMBDAS
    assert config.schema_version == "minimal-feasibility-action-shield-v1"
    assert not hasattr(config, "__dict__")
    with pytest.raises(FrozenInstanceError):
        config.schema_version = "changed"  # type: ignore[misc]
    with pytest.raises(ValueError, match="fixed"):
        ActionShieldConfig(lambdas=(1.0,))
    with pytest.raises(ValueError, match="fixed"):
        build_candidates(np.zeros(1), np.zeros(1), lambdas=(1.0,))


def test_evidence_dataclasses_are_frozen_and_slotted() -> None:
    candidate = ActionCandidate(0.25, np.array([0.5]))
    attempt = CandidateAttempt(0.25, np.array([0.5]), True, 0.01, None, None)
    result = ProjectionResult(candidate, np.array([2.0]), (attempt,))

    for value in (candidate, attempt, result):
        assert not hasattr(value, "__dict__")
    with pytest.raises(FrozenInstanceError):
        candidate.lambda_value = 0.5  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        attempt.success = False  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        result.selected = None  # type: ignore[misc]


def test_direct_evidence_dataclasses_detach_and_freeze_arrays() -> None:
    candidate_source = np.array([0.5])
    attempt_source = np.array([-0.5])
    state_source = np.array([2.0])
    candidate = ActionCandidate(0.25, candidate_source)
    attempt = CandidateAttempt(0.25, attempt_source, True, 0.01, None, None)
    result = ProjectionResult(candidate, state_source, (attempt,))

    candidate_source[:] = 1.0
    attempt_source[:] = 1.0
    state_source[:] = 1.0

    np.testing.assert_allclose(candidate.action, [0.5])
    np.testing.assert_allclose(attempt.action, [-0.5])
    np.testing.assert_allclose(result.final_state, [2.0])
    assert not candidate.action.flags.writeable
    assert not attempt.action.flags.writeable
    assert result.final_state is not None and not result.final_state.flags.writeable


@pytest.mark.parametrize(
    ("policy", "reference"),
    [
        (np.zeros(2), np.zeros(3)),
        (np.array([1.01]), np.zeros(1)),
        (np.zeros(1), np.array([-1.01])),
        (np.array([np.nan]), np.zeros(1)),
        (np.zeros(1), np.array([np.inf])),
        (np.array([]), np.array([])),
        (np.zeros((1, 1)), np.zeros((1, 1))),
    ],
)
def test_build_candidates_rejects_invalid_actions(policy: np.ndarray, reference: np.ndarray) -> None:
    with pytest.raises(ValueError):
        build_candidates(policy, reference)


def test_project_stops_at_first_success_and_records_three_attempts() -> None:
    calls: list[np.ndarray] = []

    def integrate(action: np.ndarray) -> np.ndarray:
        calls.append(action)
        if len(calls) < 3:
            raise RuntimeError(f"rejected-{len(calls)}")
        return np.array([4.0, 5.0])

    result = project_first_feasible(
        np.array([1.0, -1.0]),
        np.array([-1.0, 1.0]),
        integrate,
        ActionShieldConfig(),
    )

    assert len(calls) == 3
    assert result.selected is not None
    assert result.selected.lambda_value == 0.25
    assert len(result.attempts) == 3
    assert [attempt.success for attempt in result.attempts] == [False, False, True]
    np.testing.assert_allclose(result.final_state, [4.0, 5.0])
    for attempt in result.attempts:
        assert attempt.elapsed_seconds >= 0.0


@pytest.mark.parametrize(
    "bad_state",
    [np.array([]), np.array([np.nan]), np.array([np.inf]), np.zeros((1, 1))],
)
def test_invalid_integration_outputs_exhaust_the_grid(bad_state: np.ndarray) -> None:
    calls = 0

    def integrate(_action: np.ndarray) -> np.ndarray:
        nonlocal calls
        calls += 1
        return bad_state

    result = project_first_feasible(np.zeros(1), np.zeros(1), integrate, ActionShieldConfig())

    assert calls == len(DEFAULT_LAMBDAS)
    assert result.selected is None
    assert result.final_state is None
    assert len(result.attempts) == len(DEFAULT_LAMBDAS)
    assert all(not attempt.success for attempt in result.attempts)
    assert all(attempt.exception_type == "ValueError" for attempt in result.attempts)


def test_result_evidence_is_detached_and_read_only() -> None:
    policy = np.array([1.0, 0.0])
    reference = np.array([0.0, -1.0])
    external_state = np.array([2.0, 3.0])

    result = project_first_feasible(
        policy, reference, lambda _action: external_state, ActionShieldConfig()
    )
    policy[:] = -1.0
    reference[:] = 1.0
    external_state[:] = 99.0

    assert result.selected is not None
    np.testing.assert_allclose(result.selected.action, [15.0 / 16.0, -1.0 / 16.0])
    np.testing.assert_allclose(result.attempts[0].action, result.selected.action)
    np.testing.assert_allclose(result.final_state, [2.0, 3.0])
    assert not result.selected.action.flags.writeable
    assert not result.attempts[0].action.flags.writeable
    assert result.final_state is not None and not result.final_state.flags.writeable
    with pytest.raises(ValueError):
        result.selected.action[0] = 0.0


def test_integration_cannot_mutate_stored_candidate_evidence() -> None:
    def integrate(action: np.ndarray) -> np.ndarray:
        action.setflags(write=True)
        action[:] = 0.75
        return np.array([1.0])

    result = project_first_feasible(
        np.array([1.0]), np.array([-1.0]), integrate, ActionShieldConfig()
    )

    assert result.selected is not None
    np.testing.assert_allclose(result.selected.action, [7.0 / 8.0])
    np.testing.assert_allclose(result.attempts[0].action, [7.0 / 8.0])


def test_action_candidate_detaches_and_freezes_its_array() -> None:
    source = np.array([0.5])
    candidate = ActionCandidate(0.25, source)
    source[0] = 0.0

    np.testing.assert_allclose(candidate.action, [0.5])
    assert not candidate.action.flags.writeable


def test_integration_exception_is_captured_without_stopping_grid() -> None:
    calls = 0

    def integrate(_action: np.ndarray) -> np.ndarray:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise LookupError("solver rejected candidate")
        return np.array([1.0])

    result = project_first_feasible(np.zeros(1), np.zeros(1), integrate, ActionShieldConfig())

    assert calls == 2
    first = result.attempts[0]
    assert first.exception_type == "LookupError"
    assert first.exception_message == "solver rejected candidate"
    assert not first.success


@pytest.mark.parametrize("fatal", [KeyboardInterrupt(), SystemExit(2)])
def test_base_exceptions_propagate(fatal: BaseException) -> None:
    def integrate(_action: np.ndarray) -> np.ndarray:
        raise fatal

    with pytest.raises(type(fatal)):
        project_first_feasible(np.zeros(1), np.zeros(1), integrate, ActionShieldConfig())


@pytest.mark.parametrize(
    ("policy", "reference", "config"),
    [
        (np.array([2.0]), np.zeros(1), ActionShieldConfig()),
        (np.zeros(2), np.zeros(1), ActionShieldConfig()),
        (np.array([np.nan]), np.zeros(1), ActionShieldConfig()),
        (np.zeros(1), np.zeros(1), "bad-config"),
    ],
)
def test_invalid_projection_inputs_make_no_integration_calls(
    policy: np.ndarray, reference: np.ndarray, config: object
) -> None:
    calls = 0

    def integrate(_action: np.ndarray) -> np.ndarray:
        nonlocal calls
        calls += 1
        return np.zeros(1)

    with pytest.raises((TypeError, ValueError)):
        project_first_feasible(policy, reference, integrate, config)  # type: ignore[arg-type]
    assert calls == 0
