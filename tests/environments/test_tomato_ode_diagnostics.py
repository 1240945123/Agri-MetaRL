import casadi as ca
import numpy as np
import pytest

from gl_gym.RL.utils import load_env_params
from gl_gym.environments.models.utils import FORMAL_CVODES_OPTIONS
from gl_gym.environments.tomato_env import TomatoEnv
from gl_gym.paths import CONFIG_DIR


class RecordingIntegrator:
    def __init__(self, *, next_state=None, error=None):
        self.next_state = next_state
        self.error = error
        self.calls = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        return {"xf": ca.DM(self.next_state)}


@pytest.fixture
def env():
    base_params, specific_params = load_env_params("TomatoEnv", str(CONFIG_DIR / "envs"))
    specific_params.pop("eval_options_heldout", None)
    base_params["training"] = False
    instance = TomatoEnv(base_env_params=base_params, **specific_params)
    instance.reset(seed=123)
    return instance


def test_diagnostics_are_disabled_by_default(env, monkeypatch):
    monkeypatch.setattr(
        "gl_gym.environments.tomato_env.parametric_crop_uncertainty",
        lambda p, scale, rng: np.asarray(p).copy(),
    )
    env.F = RecordingIntegrator(next_state=env.x.copy())

    _, _, _, _, info = env.step(np.zeros(env.nu))

    assert "diagnostic_transition" not in info
    assert "integration_failure" not in info


def test_enabled_failure_reports_exact_inputs_exception_and_one_draw(env, monkeypatch):
    draws = []
    sampled_parameters = np.arange(env.num_params, dtype=float) + 0.25

    def sample_once(p, scale, rng):
        draws.append((p, scale, rng))
        return sampled_parameters.copy()

    monkeypatch.setattr(
        "gl_gym.environments.tomato_env.parametric_crop_uncertainty", sample_once
    )
    error = RuntimeError("intentional cvodes failure")
    integrator = RecordingIntegrator(error=error)
    env.F = integrator
    env.set_ode_diagnostics_enabled(True)

    raw_observation = env.obs.copy()
    x0 = env.x.copy()
    previous_control = np.linspace(0.0, 0.5, env.nu)
    env.u = previous_control.copy()
    requested_action = np.linspace(-2.0, 2.0, env.nu)
    executed_control = env.action_to_control(requested_action)
    weather_row = env.weather_data[env.timestep].copy()
    timestep = env.timestep
    day_of_year = env.day_of_year
    hour_of_day = env.hour_of_day

    _, _, terminated, _, info = env.step(requested_action)

    assert terminated is True
    assert len(draws) == 1
    assert len(integrator.calls) == 1
    failure = info["integration_failure"]
    np.testing.assert_array_equal(failure["x0"], x0)
    np.testing.assert_array_equal(failure["previous_control"], previous_control)
    np.testing.assert_array_equal(failure["requested_action"], requested_action)
    np.testing.assert_array_equal(failure["u"], executed_control)
    np.testing.assert_array_equal(failure["weather"], weather_row)
    np.testing.assert_array_equal(failure["sampled_parameters"], sampled_parameters)
    np.testing.assert_array_equal(
        failure["p_dyn"], np.concatenate((weather_row, sampled_parameters))
    )
    assert failure["timestep"] == timestep
    assert failure["day_of_year"] == day_of_year
    assert failure["hour_of_day"] == hour_of_day
    assert failure["dt"] == env.dt
    assert failure["nx"] == env.nx
    assert failure["nu"] == env.nu
    assert failure["nd"] == env.nd
    assert failure["n_params"] == env.num_params
    assert failure["solver_options"] == dict(FORMAL_CVODES_OPTIONS)
    assert failure["exception_type"] == "RuntimeError"
    assert failure["exception_message"] == "intentional cvodes failure"
    assert "RuntimeError: intentional cvodes failure" in failure["traceback"]

    transition = info["diagnostic_transition"]
    np.testing.assert_array_equal(transition["raw_observation"], raw_observation)
    np.testing.assert_array_equal(transition["requested_action"], requested_action)
    np.testing.assert_array_equal(transition["previous_control"], previous_control)
    np.testing.assert_array_equal(transition["executed_control"], executed_control)
    assert transition["raw_next_observation"] is None
    assert transition["raw_next_observation_available"] is False


def test_enabled_success_reports_raw_next_observation_without_failure(env, monkeypatch):
    sampled_parameters = np.asarray(env.p).copy()
    monkeypatch.setattr(
        "gl_gym.environments.tomato_env.parametric_crop_uncertainty",
        lambda p, scale, rng: sampled_parameters.copy(),
    )
    next_state = env.x.copy()
    next_state[2] += 0.5
    env.F = RecordingIntegrator(next_state=next_state)
    env.set_ode_diagnostics_enabled(True)
    raw_observation = env.obs.copy()
    action = np.full(env.nu, 0.2)

    returned_observation, _, _, _, info = env.step(action)

    assert "integration_failure" not in info
    transition = info["diagnostic_transition"]
    np.testing.assert_array_equal(transition["raw_observation"], raw_observation)
    np.testing.assert_array_equal(
        transition["raw_next_observation"], returned_observation
    )
    assert transition["raw_next_observation_available"] is True


def test_input_construction_error_propagates_without_advancing_step(env, monkeypatch):
    sentinel = RuntimeError("sentinel p_dyn construction failure")
    original_vertcat = ca.vertcat

    def fail_input_construction(*args):
        raise sentinel

    monkeypatch.setattr(
        "gl_gym.environments.tomato_env.parametric_crop_uncertainty",
        lambda p, scale, rng: np.asarray(p).copy(),
    )
    monkeypatch.setattr(
        "gl_gym.environments.tomato_env.ca.vertcat", fail_input_construction
    )
    integrator = RecordingIntegrator(next_state=env.x.copy())
    env.F = integrator
    env.set_ode_diagnostics_enabled(True)
    timestep = env.timestep
    day_of_year = env.day_of_year
    hour_of_day = env.hour_of_day
    state = env.x.copy()
    terminated = env.terminated

    with pytest.raises(RuntimeError) as raised:
        env.step(np.zeros(env.nu))

    assert raised.value is sentinel
    assert integrator.calls == []
    assert env.timestep == timestep
    assert env.day_of_year == day_of_year
    assert env.hour_of_day == hour_of_day
    np.testing.assert_array_equal(env.x, state)
    assert env.terminated is terminated

    monkeypatch.setattr("gl_gym.environments.tomato_env.ca.vertcat", original_vertcat)
    _, _, _, _, later_info = env.step(np.zeros(env.nu))
    assert "integration_failure" not in later_info
    _, reset_info = env.reset(seed=123)
    assert "integration_failure" not in reset_info


def test_returned_diagnostics_are_isolated_from_later_mutation(env, monkeypatch):
    sampled_parameters = np.arange(env.num_params, dtype=float)
    monkeypatch.setattr(
        "gl_gym.environments.tomato_env.parametric_crop_uncertainty",
        lambda p, scale, rng: sampled_parameters.copy(),
    )
    env.F = RecordingIntegrator(error=ValueError("failure for copy test"))
    env.set_ode_diagnostics_enabled(True)
    action = np.full(env.nu, 0.4)

    _, _, _, _, info = env.step(action)
    frozen = {
        key: value.copy()
        for key, value in info["integration_failure"].items()
        if isinstance(value, np.ndarray)
    }
    transition_frozen = {
        key: value.copy()
        for key, value in info["diagnostic_transition"].items()
        if isinstance(value, np.ndarray)
    }

    env.x[:] = -999
    env.u[:] = -888
    action[:] = -777
    sampled_parameters[:] = -666

    for key, expected in frozen.items():
        np.testing.assert_array_equal(info["integration_failure"][key], expected)
    for key, expected in transition_frozen.items():
        np.testing.assert_array_equal(info["diagnostic_transition"][key], expected)
