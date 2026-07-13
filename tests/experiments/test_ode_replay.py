from dataclasses import FrozenInstanceError
import math
import warnings

import numpy as np
import pytest

from gl_gym.experiments.ode_failure import LoadedFailureCapsule
from gl_gym.experiments.ode_replay import (
    ReplayOutcome,
    ReplayReport,
    build_rule_based_controller,
    classify_replay_outcomes,
    replay_failure_capsule,
)


VARIANTS = (
    "original",
    "previous_control",
    "rule_based_control",
    "original_2x_substeps",
    "original_4x_substeps",
    "original_strict_tolerance",
)


def _capsule():
    inputs = {
        "x0": np.array([1.25, -2.5], dtype=np.float32),
        "u": np.array([0.1, 0.2], dtype=np.float64),
        "previous_control": np.array([0.3, 0.4], dtype=np.float32),
        "requested_action": np.zeros(2),
        "weather": np.array([5.0, 6.0, 7.0], dtype=np.float32),
        "sampled_parameters": np.array([8.0], dtype=np.float64),
        "p_dyn": np.array([5.0, 6.0, 7.0, 8.0], dtype=np.float64),
        "timestep": np.array(9),
        "day_of_year": np.array(151.5),
        "hour_of_day": np.array(13.25),
        "dt": np.array(12.0),
        "nx": np.array(2),
        "nu": np.array(2),
        "nd": np.array(3),
        "n_params": np.array(1),
    }
    return LoadedFailureCapsule(
        path=None,
        manifest={"failure_id": "failure-123", "solver": {"options": {}}},
        failure_inputs=inputs,
        history_arrays={},
        history_rows=(),
        traceback_text="",
    )


class _DM:
    def __init__(self, value):
        self.value = value

    def full(self):
        return np.asarray(self.value).reshape(-1, 1)


class RecordingFactory:
    def __init__(self, behavior=None):
        self.builds = []
        self.calls = []
        self.behavior = behavior

    def __call__(self, **kwargs):
        build_index = len(self.builds)
        self.builds.append(kwargs)

        def integrate(**call):
            copied = {key: np.array(value, copy=True) for key, value in call.items()}
            self.calls.append((build_index, copied))
            if self.behavior is not None:
                return self.behavior(build_index, copied)
            return {"xf": _DM(copied["x0"] + 1.0)}

        return integrate


class RecordingController:
    def __init__(self, control=(0.8, 0.9)):
        self.control = np.array(control)
        self.calls = []

    def predict(self, x0, weather, env):
        self.calls.append((np.array(x0, copy=True), np.array(weather, copy=True), env))
        return self.control


def _outcome(variant, *, success=False, available=True):
    return ReplayOutcome(variant, available, success, 0.1, np.zeros(2) if success else None)


def _all_outcomes(**statuses):
    return tuple(_outcome(name, **statuses.get(name, {})) for name in VARIANTS)


def _assert_byte_equivalent(actual, expected):
    actual_array = np.asarray(actual)
    expected_array = np.asarray(expected)
    assert actual_array.dtype == expected_array.dtype
    assert actual_array.shape == expected_array.shape
    assert actual_array.tobytes() == expected_array.tobytes()


def test_replay_types_are_frozen_and_outcome_detaches_final_state():
    source = np.array([1.0, 2.0])
    outcome = ReplayOutcome("original", True, True, 0.25, source, warnings=["notice"])
    source[:] = 99
    np.testing.assert_array_equal(outcome.final_state, [1.0, 2.0])
    assert outcome.warnings == ("notice",)
    with pytest.raises(FrozenInstanceError):
        outcome.success = False
    report = ReplayReport("id", "non_reproduced", (outcome,))
    with pytest.raises(FrozenInstanceError):
        report.failure_id = "other"


@pytest.mark.parametrize("elapsed", [math.nan, math.inf, -math.inf])
def test_outcome_rejects_nonfinite_elapsed_time(elapsed):
    with pytest.raises(ValueError, match="elapsed"):
        ReplayOutcome("original", True, False, elapsed, None)


def test_outcome_rejects_nonfinite_final_state():
    with pytest.raises(ValueError, match="final_state"):
        ReplayOutcome("original", True, True, 0.1, np.array([1.0, np.inf]))


def test_replay_runs_six_fresh_variants_in_order_with_exact_inputs_and_options():
    capsule = _capsule()
    factory = RecordingFactory()
    controller = RecordingController()

    report = replay_failure_capsule(
        capsule,
        integrator_factory=factory,
        controller_factory=lambda: controller,
    )

    assert report.failure_id == "failure-123"
    assert report.classification == "non_reproduced"
    assert tuple(item.variant for item in report.outcomes) == VARIANTS
    assert len(factory.builds) == 6
    expected_dts = [12.0, 12.0, 12.0, 6.0, 3.0, 12.0]
    for build, dt in zip(factory.builds, expected_dts):
        assert build["nx"] == 2 and build["nu"] == 2
        assert build["nd"] == 3 and build["n_params"] == 1
        assert build["dt"] == dt
    assert all("integrator_options" not in build for build in factory.builds[:5])
    assert factory.builds[5]["integrator_options"] == {
        "abstol": 1e-6,
        "reltol": 1e-6,
    }

    calls_by_build = [[call for index, call in factory.calls if index == build] for build in range(6)]
    assert [len(calls) for calls in calls_by_build] == [1, 1, 1, 2, 4, 1]
    inputs = capsule.failure_inputs
    for build_calls in calls_by_build:
        for call in build_calls:
            _assert_byte_equivalent(call["p"], inputs["p_dyn"])
    for build_calls in calls_by_build:
        _assert_byte_equivalent(build_calls[0]["x0"], inputs["x0"])
    for build_index in (0, 3, 4, 5):
        for call in calls_by_build[build_index]:
            _assert_byte_equivalent(call["u"], inputs["u"])
    _assert_byte_equivalent(calls_by_build[1][0]["u"], inputs["previous_control"])
    _assert_byte_equivalent(calls_by_build[2][0]["u"], controller.control)
    for substeps in (calls_by_build[3], calls_by_build[4]):
        for previous, current in zip(substeps, substeps[1:]):
            np.testing.assert_array_equal(current["x0"], previous["x0"] + 1.0)

    x0, weather, env = controller.calls[0]
    assert x0.tobytes() == inputs["x0"].tobytes()
    assert weather.tobytes() == inputs["weather"].tobytes()
    assert (env.nu, env.day_of_year, env.hour_of_day) == (2, 151.5, 13.25)


@pytest.mark.parametrize(
    ("field", "bad_value"),
    [
        ("x0", np.array([1.0, np.nan], dtype=np.float32)),
        ("u", np.array([0.1, np.inf], dtype=np.float64)),
        ("p_dyn", np.array([5.0, 6.0, -np.inf, 8.0], dtype=np.float64)),
        ("dt", np.array(np.nan)),
    ],
)
def test_replay_rejects_nonfinite_stored_inputs_before_building_integrator(
    field, bad_value
):
    capsule = _capsule()
    capsule.failure_inputs[field] = bad_value
    factory = RecordingFactory()

    with pytest.raises(ValueError, match=field):
        replay_failure_capsule(
            capsule,
            integrator_factory=factory,
            controller_factory=lambda: RecordingController(),
        )

    assert factory.builds == []


def test_replay_captures_warnings_exceptions_and_elapsed_time():
    def behavior(build, call):
        if build == 0:
            warnings.warn("solver warning", RuntimeWarning)
            raise RuntimeError("cvodes exploded")
        return {"xf": call["x0"]}

    report = replay_failure_capsule(
        _capsule(),
        integrator_factory=RecordingFactory(behavior),
        controller_factory=lambda: RecordingController(),
    )
    original = report.outcomes[0]
    assert original.available is True and original.success is False
    assert original.exception_type == "RuntimeError"
    assert original.exception_message == "cvodes exploded"
    assert original.warnings == ("solver warning",)
    assert math.isfinite(original.elapsed_seconds) and original.elapsed_seconds >= 0


@pytest.mark.parametrize(
    ("bad_final", "message"),
    [(np.array([1.0, np.nan]), "finite"), (np.array([1.0, 2.0, 3.0]), "shape")],
)
def test_replay_marks_nonfinite_or_wrong_shape_final_states_failed(bad_final, message):
    def behavior(build, call):
        return {"xf": bad_final if build == 0 else call["x0"]}

    report = replay_failure_capsule(
        _capsule(), RecordingFactory(behavior), lambda: RecordingController()
    )
    original = report.outcomes[0]
    assert not original.success and original.final_state is None
    assert original.exception_type == "ValueError"
    assert message in original.exception_message


def test_rule_based_variant_is_unavailable_when_controller_cannot_be_built():
    def unavailable():
        raise FileNotFoundError("rule config unavailable")

    factory = RecordingFactory()
    report = replay_failure_capsule(_capsule(), factory, unavailable)
    rule = report.outcomes[2]
    assert len(factory.builds) == 6
    assert rule.available is False and rule.success is False
    assert rule.exception_type == "FileNotFoundError"
    assert rule.exception_message == "rule config unavailable"


def test_rule_prediction_exception_is_an_available_variant_failure():
    class BrokenController:
        def predict(self, x0, weather, env):
            raise IndexError("weather prerequisites invalid")

    report = replay_failure_capsule(
        _capsule(), RecordingFactory(), lambda: BrokenController()
    )
    rule = report.outcomes[2]
    assert rule.available is True and rule.success is False
    assert rule.exception_type == "IndexError"


def test_build_rule_based_controller_uses_existing_loader_and_controller(monkeypatch):
    captured = {}

    def load(algorithm, env_id):
        captured["load"] = (algorithm, env_id)
        return {"one": 1, "two": 2}

    class Controller:
        def __init__(self, **kwargs):
            captured["kwargs"] = kwargs

    monkeypatch.setattr("gl_gym.experiments.ode_replay.load_model_hyperparams", load)
    monkeypatch.setattr("gl_gym.experiments.ode_replay.RuleBasedController", Controller)
    controller = build_rule_based_controller()
    assert isinstance(controller, Controller)
    assert captured == {"load": ("rule_based", "TomatoEnv"), "kwargs": {"one": 1, "two": 2}}


@pytest.mark.parametrize(
    ("statuses", "expected"),
    [
        ({"original": {"success": True}}, "non_reproduced"),
        (
            {"previous_control": {"success": True}, "original_2x_substeps": {"success": True}},
            "mixed_control_and_solver_sensitivity",
        ),
        ({"rule_based_control": {"success": True}}, "policy_induced_control_instability"),
        ({"original_4x_substeps": {"success": True}}, "solver_step_sensitivity"),
        ({}, "state_or_model_domain_failure"),
        (
            {"rule_based_control": {"available": False}},
            "insufficient_counterfactual_evidence",
        ),
    ],
)
def test_classification_labels(statuses, expected):
    assert classify_replay_outcomes(_all_outcomes(**statuses)) == expected


def test_solver_success_with_unavailable_control_uses_available_control_evidence():
    outcomes = _all_outcomes(
        rule_based_control={"available": False},
        original_2x_substeps={"success": True},
    )
    assert classify_replay_outcomes(outcomes) == "solver_step_sensitivity"


@pytest.mark.parametrize(
    "outcomes",
    [
        _all_outcomes()[:-1],
        _all_outcomes() + (_outcome("unknown"),),
        _all_outcomes()[:-1] + (_outcome("original"),),
    ],
)
def test_classification_rejects_missing_unknown_or_duplicate_variants(outcomes):
    with pytest.raises(ValueError, match="variant"):
        classify_replay_outcomes(outcomes)


def test_classification_rejects_unavailable_original():
    outcomes = _all_outcomes(original={"available": False})
    with pytest.raises(ValueError, match="original"):
        classify_replay_outcomes(outcomes)
