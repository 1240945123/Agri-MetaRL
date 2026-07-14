from copy import deepcopy

import casadi as ca
import numpy as np
import pytest

from gl_gym.RL.utils import ENVS, make_env
from gl_gym.environments.action_shield import DEFAULT_LAMBDAS
from gl_gym.environments.models.utils import FORMAL_CVODES_OPTIONS
from gl_gym.environments.tomato_env import TomatoEnv


def test_shielded_environment_has_distinct_registry_entry() -> None:
    from gl_gym.environments.shielded_tomato_env import ShieldedTomatoEnv

    assert ENVS["TomatoEnv"] is TomatoEnv
    assert ENVS["ShieldedTomatoEnv"] is ShieldedTomatoEnv
    assert ShieldedTomatoEnv is not TomatoEnv
    assert issubclass(ShieldedTomatoEnv, TomatoEnv)


class DummyReward:
    def __init__(self, env) -> None:
        self.env = env
        self.calls = 0
        self.profit = self.gains = self.variable_costs = self.fixed_costs = 0.0
        self.co2_costs = self.heat_costs = self.elec_costs = 0.0
        self.temp_violation = self.co2_violation = self.rh_violation = 0.0
        self.lamp_violation = 0.0

    def compute_reward(self) -> float:
        self.calls += 1
        self.profit = float(self.env.x[0])
        return self.profit


class Integrator:
    def __init__(self, outcome) -> None:
        self.outcome = outcome
        self.calls = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        if isinstance(self.outcome, BaseException):
            raise self.outcome
        value = self.outcome(kwargs) if callable(self.outcome) else self.outcome
        return {"xf": ca.DM(value)}


class Controller:
    def __init__(self, target) -> None:
        self.target = target
        self.calls = []

    def predict(self, x, d, env):
        self.calls.append((x, d, env))
        return np.array(self.target, copy=True)


@pytest.fixture
def shield_env():
    from gl_gym.environments.shielded_tomato_env import ShieldedTomatoEnv

    env = ShieldedTomatoEnv.__new__(ShieldedTomatoEnv)
    env.nx, env.nu, env.nd, env.num_params = 3, 2, 2, 4
    env.dt, env.c, env.N = 300.0, 86400.0, 20
    env.x = np.array([1.0, 2.0, 3.0])
    env.u = np.array([0.2, 0.4])
    env.x_prev = np.array([0.5, 1.5, 2.5])
    env.obs = env.x.copy()
    env.timestep = 1
    env.day_of_year = 100.0
    env.hour_of_day = 4.0
    env.terminated = False
    env.u_min = np.zeros(2)
    env.u_max = np.ones(2)
    env.delta_u_max = np.array([0.2, 0.4])
    env.p = np.arange(4, dtype=float) + 10.0
    env.uncertainty_scale = 0.3
    env._np_random = np.random.default_rng(123)
    env.weather_data = np.array([[0.0, 0.0], [7.0, 8.0], [9.0, 10.0]])
    env._ode_diagnostics_enabled = False
    env.action_shield_config = __import__(
        "gl_gym.environments.action_shield", fromlist=["ActionShieldConfig"]
    ).ActionShieldConfig()
    env.action_shield_controller = Controller([0.8, 0.0])
    env.reward = DummyReward(env)
    env._get_obs = lambda: env.x.copy()
    env._get_info = lambda: {"controls": env.u}
    env._terminalState = lambda: env.timestep >= env.N
    return env


def _array(kwargs, name):
    return np.asarray(kwargs[name], dtype=float).reshape(-1)


def test_constructor_requires_and_copies_controller_mapping(monkeypatch) -> None:
    from gl_gym.environments import shielded_tomato_env as module

    captured = {}
    monkeypatch.setattr(module.TomatoEnv, "__init__", lambda self, **kwargs: None)
    monkeypatch.setattr(
        module,
        "RuleBasedController",
        lambda **kwargs: captured.setdefault("params", kwargs),
    )
    source = {"nested": [1]}
    env = module.ShieldedTomatoEnv(action_shield_params=source)
    source["nested"].append(2)

    assert env.action_shield_params == {"nested": [1]}
    assert captured["params"] == {"nested": [1]}
    assert env.action_shield_config.lambdas == DEFAULT_LAMBDAS
    with pytest.raises(TypeError, match="mapping"):
        module.ShieldedTomatoEnv(action_shield_params=None)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "bad_action",
    [np.zeros(3), np.zeros((2, 1)), np.array([np.nan, 0.0]), np.array([1.01, 0.0]), ["x", "y"]],
)
def test_invalid_requested_action_has_no_side_effects(shield_env, monkeypatch, bad_action) -> None:
    from gl_gym.environments import shielded_tomato_env as module

    samples = []
    monkeypatch.setattr(module, "parametric_crop_uncertainty", lambda *args: samples.append(args))
    shield_env.F = Integrator(shield_env.x)
    before = (shield_env.x.copy(), shield_env.u.copy(), shield_env.timestep)

    with pytest.raises(ValueError):
        shield_env.step(bad_action)

    assert samples == []
    assert shield_env.F.calls == []
    np.testing.assert_array_equal(shield_env.x, before[0])
    np.testing.assert_array_equal(shield_env.u, before[1])
    assert shield_env.timestep == before[2]


def test_invalid_requested_control_mapping_is_not_treated_as_solver_failure(
    shield_env, monkeypatch
) -> None:
    from gl_gym.environments import shielded_tomato_env as module

    samples = []
    monkeypatch.setattr(module, "parametric_crop_uncertainty", lambda *args: samples.append(args))
    shield_env.action_to_control = lambda action: np.array([np.nan, 0.0])
    shield_env.F = Integrator([4, 5, 6])

    with pytest.raises(ValueError, match="control"):
        shield_env.step(np.zeros(2))

    assert samples == []
    assert shield_env.F.calls == []


def test_invalid_sampled_parameters_fail_before_integration(shield_env, monkeypatch) -> None:
    from gl_gym.environments import shielded_tomato_env as module

    monkeypatch.setattr(
        module,
        "parametric_crop_uncertainty",
        lambda p, scale, rng: np.array([np.nan] * shield_env.num_params),
    )
    shield_env.F = Integrator([4, 5, 6])

    with pytest.raises(ValueError, match="sampled parameters"):
        shield_env.step(np.zeros(2))

    assert shield_env.F.calls == []


def test_original_success_samples_once_and_never_constructs_recovery(shield_env, monkeypatch) -> None:
    from gl_gym.environments import shielded_tomato_env as module

    samples = []
    sampled = np.array([20.0, 21.0, 22.0, 23.0])
    monkeypatch.setattr(
        module,
        "parametric_crop_uncertainty",
        lambda p, scale, rng: samples.append((p.copy(), scale, rng)) or sampled.copy(),
    )
    monkeypatch.setattr(module, "define_model", lambda **kwargs: pytest.fail("fresh factory called"))
    shield_env.action_shield_controller.predict = lambda *args: pytest.fail("controller called")
    shield_env.F = Integrator([4.0, 5.0, 6.0])
    previous = shield_env.u.copy()
    action = np.array([0.5, -0.5])

    observation, reward, terminated, truncated, info = shield_env.step(action)

    assert len(samples) == 1
    assert len(shield_env.F.calls) == 1
    np.testing.assert_array_equal(_array(shield_env.F.calls[0], "x0"), [1.0, 2.0, 3.0])
    np.testing.assert_array_equal(_array(shield_env.F.calls[0], "p"), [7, 8, 20, 21, 22, 23])
    np.testing.assert_allclose(shield_env.u, previous + action * shield_env.delta_u_max)
    np.testing.assert_array_equal(observation, [4.0, 5.0, 6.0])
    assert reward == 4.0 and not terminated and not truncated
    assert shield_env.reward.calls == 1 and shield_env.timestep == 2
    record = info["action_shield"]
    assert record["intervened"] is False
    assert record["selected_lambda"] == 0.0
    assert record["reference_action"] is None
    assert record["requested_action"] == record["executed_action"] == [0.5, -0.5]
    assert record["candidate_attempts"] == [] and record["extra_solver_attempts"] == 0
    assert record["original_failure"] is None


def test_original_success_accepts_numpy_final_state(shield_env, monkeypatch) -> None:
    from gl_gym.environments import shielded_tomato_env as module

    monkeypatch.setattr(module, "parametric_crop_uncertainty", lambda p, scale, rng: p.copy())

    class NumpyIntegrator:
        def __call__(self, **kwargs):
            return {"xf": np.array([11.0, 12.0, 13.0])}

    shield_env.F = NumpyIntegrator()

    observation, *_ = shield_env.step(np.zeros(2))

    np.testing.assert_array_equal(observation, [11.0, 12.0, 13.0])


def test_retry_uses_fixed_order_fresh_integrators_and_reuses_selected_state(
    shield_env, monkeypatch
) -> None:
    from gl_gym.environments import shielded_tomato_env as module

    sampled = np.array([30.0, 31.0, 32.0, 33.0])
    draws = []
    monkeypatch.setattr(
        module,
        "parametric_crop_uncertainty",
        lambda p, scale, rng: draws.append(1) or sampled.copy(),
    )
    original_error = RuntimeError("original rejected")
    shield_env.F = Integrator(original_error)
    previous = shield_env.u.copy()
    policy = np.array([1.0, -1.0])
    target = previous.copy()
    controller = Controller(target)
    shield_env.action_shield_controller = controller
    fresh = []

    def factory(**kwargs):
        assert kwargs["integrator_options"] == dict(FORMAL_CVODES_OPTIONS)
        outcome = RuntimeError(f"candidate-{len(fresh)}") if len(fresh) < 2 else [40, 41, 42]
        integrator = Integrator(outcome)
        fresh.append(integrator)
        return integrator

    monkeypatch.setattr(module, "define_model", factory)
    observation, _, terminated, _, info = shield_env.step(policy)

    assert draws == [1]
    assert len(controller.calls) == 1
    np.testing.assert_array_equal(controller.calls[0][0], [1, 2, 3])
    np.testing.assert_array_equal(controller.calls[0][1], [7, 8])
    assert controller.calls[0][2] is shield_env
    assert len(fresh) == 3 and len({id(value) for value in fresh}) == 3
    reference = np.clip((target - previous) / shield_env.delta_u_max, -1, 1)
    for index, (integrator, lam) in enumerate(zip(fresh, DEFAULT_LAMBDAS, strict=False)):
        call = integrator.calls[0]
        candidate = (1 - lam) * policy + lam * reference
        expected_control = np.clip(previous + candidate * shield_env.delta_u_max, 0, 1)
        np.testing.assert_array_equal(_array(call, "x0"), [1, 2, 3])
        np.testing.assert_array_equal(_array(call, "u"), expected_control)
        np.testing.assert_array_equal(_array(call, "p"), [7, 8, 30, 31, 32, 33])
    np.testing.assert_array_equal(observation, [40, 41, 42])
    assert not terminated and shield_env.reward.calls == 1
    record = info["action_shield"]
    assert record["selected_lambda"] == DEFAULT_LAMBDAS[2]
    assert [item["lambda"] for item in record["candidate_attempts"]] == list(DEFAULT_LAMBDAS[:3])
    assert [item["success"] for item in record["candidate_attempts"]] == [False, False, True]
    assert record["extra_solver_attempts"] == 3
    assert record["original_failure"] == {
        "exception_type": "RuntimeError",
        "exception_message": "original rejected",
    }
    assert record["per_channel_changed"] == [True, True]
    assert record["intervention_l1"] >= record["intervention_l2"] >= record["intervention_linf"]


def test_recovered_diagnostics_keep_original_failure_and_committed_observation(
    shield_env, monkeypatch
) -> None:
    from gl_gym.environments import shielded_tomato_env as module

    monkeypatch.setattr(module, "parametric_crop_uncertainty", lambda p, scale, rng: p.copy())
    shield_env.F = Integrator(ValueError("raw failure"))
    monkeypatch.setattr(module, "define_model", lambda **kwargs: Integrator([8, 9, 10]))
    shield_env.set_ode_diagnostics_enabled(True)

    observation, _, terminated, _, info = shield_env.step(np.array([0.2, -0.2]))

    assert not terminated
    assert info["integration_failure"]["exception_type"] == "ValueError"
    np.testing.assert_array_equal(info["integration_failure"]["x0"], [1, 2, 3])
    transition = info["diagnostic_transition"]
    assert transition["raw_next_observation_available"] is True
    np.testing.assert_array_equal(transition["raw_next_observation"], observation)


def test_exhaustion_restores_state_reward_and_rng_and_attaches_notes(shield_env, monkeypatch) -> None:
    from gl_gym.environments import shielded_tomato_env as module

    monkeypatch.setattr(
        module,
        "parametric_crop_uncertainty",
        lambda p, scale, rng: p + rng.normal(size=p.shape),
    )
    shield_env.F = Integrator(RuntimeError("original"))
    monkeypatch.setattr(module, "define_model", lambda **kwargs: Integrator(RuntimeError("retry")))
    before = {
        name: deepcopy(getattr(shield_env, name))
        for name in ("x", "u", "x_prev", "obs", "timestep", "day_of_year", "hour_of_day", "terminated")
    }
    rng_before = deepcopy(shield_env._np_random.bit_generator.state)

    with pytest.raises(RuntimeError, match="action shield exhausted all legal candidates") as raised:
        shield_env.step(np.zeros(2))

    assert len(raised.value.__notes__) == len(DEFAULT_LAMBDAS)
    for name, expected in before.items():
        np.testing.assert_equal(getattr(shield_env, name), expected)
    assert shield_env.reward.calls == 0
    assert shield_env._np_random.bit_generator.state == rng_before


def test_reference_factory_and_commit_errors_keep_priority_and_restore(shield_env, monkeypatch) -> None:
    from gl_gym.environments import shielded_tomato_env as module

    monkeypatch.setattr(module, "parametric_crop_uncertainty", lambda p, scale, rng: p.copy())
    shield_env.F = Integrator(RuntimeError("original"))
    before = shield_env.x.copy(), shield_env.u.copy(), shield_env.timestep
    sentinel = TypeError("controller bug")
    shield_env.action_shield_controller.predict = lambda *args: (_ for _ in ()).throw(sentinel)
    with pytest.raises(TypeError) as raised:
        shield_env.step(np.zeros(2))
    assert raised.value is sentinel

    shield_env.action_shield_controller = Controller([0.8, 0.0])
    factory_error = KeyError("factory bug")
    monkeypatch.setattr(module, "define_model", lambda **kwargs: (_ for _ in ()).throw(factory_error))
    with pytest.raises(KeyError) as raised:
        shield_env.step(np.zeros(2))
    assert raised.value is factory_error

    monkeypatch.setattr(module, "define_model", lambda **kwargs: Integrator([5, 6, 7]))
    commit_error = LookupError("observation bug")
    shield_env._get_obs = lambda: (_ for _ in ()).throw(commit_error)
    with pytest.raises(LookupError) as raised:
        shield_env.step(np.zeros(2))
    assert raised.value is commit_error
    np.testing.assert_array_equal(shield_env.x, before[0])
    np.testing.assert_array_equal(shield_env.u, before[1])
    assert shield_env.timestep == before[2] and shield_env.reward.calls == 0


def test_commit_error_keeps_priority_if_best_effort_restore_also_fails(
    shield_env, monkeypatch
) -> None:
    from gl_gym.environments import shielded_tomato_env as module

    monkeypatch.setattr(module, "parametric_crop_uncertainty", lambda p, scale, rng: p.copy())
    shield_env.F = Integrator([5, 6, 7])
    commit_error = LookupError("info commit bug")

    def fail_info():
        shield_env._restore = lambda *args, **kwargs: (_ for _ in ()).throw(
            RuntimeError("restore bug")
        )
        raise commit_error

    shield_env._get_info = fail_info

    with pytest.raises(LookupError) as raised:
        shield_env.step(np.zeros(2))

    assert raised.value is commit_error
    assert any("restore bug" in note for note in raised.value.__notes__)


def test_base_exception_from_original_propagates_without_recovery(shield_env, monkeypatch) -> None:
    from gl_gym.environments import shielded_tomato_env as module

    monkeypatch.setattr(module, "parametric_crop_uncertainty", lambda p, scale, rng: p.copy())
    fatal = KeyboardInterrupt()
    shield_env.F = Integrator(fatal)
    monkeypatch.setattr(module, "define_model", lambda **kwargs: pytest.fail("recovery called"))
    with pytest.raises(KeyboardInterrupt):
        shield_env.step(np.zeros(2))
    assert shield_env.timestep == 1


def test_default_make_env_lookup_remains_tomato() -> None:
    assert ENVS["TomatoEnv"] is TomatoEnv
    closure = make_env("TomatoEnv", 0, 1, {}, {}, False)
    assert callable(closure)
