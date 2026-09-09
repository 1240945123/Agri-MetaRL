import numpy as np

from gl_gym.RL import utils as rl_utils
from gl_gym.RL.utils import load_env_params
from gl_gym.environments.tomato_env import TomatoEnv
from gl_gym.paths import CONFIG_DIR
from gl_gym.tasks import TaskDescriptor, TaskInstance


def test_descriptor_round_trip_and_stable_key():
    task = TaskDescriptor(2010, 59, 0.05, "energy_high", "strict")
    assert TaskDescriptor.from_dict(task.to_dict()) == task
    assert task.stable_key == "2010:59:0.050000:energy_high:strict"


def test_instance_identity_separates_equal_task_episodes():
    task = TaskDescriptor(2010, 59, 0.0, "standard", "standard")
    a = TaskInstance(task=task, environment_index=0, episode_index=3)
    b = TaskInstance(task=task, environment_index=0, episode_index=4)
    assert a.stable_key != b.stable_key


def test_environment_emits_task_identity_until_next_reset():
    base_params, env_params = load_env_params("TomatoEnv", str(CONFIG_DIR / "envs"))
    env_params.pop("eval_options_heldout")
    env_params.update(
        uncertainty_scale=0.05,
        economic_scenario="energy_high",
        climate_constraint_scenario="strict",
        environment_index=2,
    )
    env = TomatoEnv(base_env_params=base_params, **env_params)

    env.reset(seed=42)
    _, _, _, _, first_info = env.step(np.zeros(env.nu))
    _, _, _, _, second_info = env.step(np.zeros(env.nu))

    descriptor = TaskDescriptor.from_dict(first_info["task_descriptor"])
    assert descriptor == env.task_descriptor
    assert descriptor.parameter_uncertainty == 0.05
    assert descriptor.economic_scenario == "energy_high"
    assert descriptor.climate_constraint_scenario == "strict"
    assert first_info["task_instance_key"] == second_info["task_instance_key"]

    previous_key = first_info["task_instance_key"]
    env.reset(seed=42)
    _, _, _, _, next_info = env.step(np.zeros(env.nu))
    assert next_info["task_instance_key"] != previous_key


def test_make_env_passes_vector_rank_as_environment_index(monkeypatch):
    captured = {}

    class FakeActionSpace:
        def seed(self, seed):
            captured["action_seed"] = seed

    class FakeEnv:
        def __init__(self, **kwargs):
            captured.update(kwargs)
            self.action_space = FakeActionSpace()

        def reset(self, seed):
            captured["reset_seed"] = seed

    monkeypatch.setitem(rl_utils.ENVS, "FakeEnv", FakeEnv)
    factory = rl_utils.make_env(
        "FakeEnv",
        rank=3,
        seed=10,
        env_base_params={},
        env_specific_params={},
        eval_env=False,
    )

    factory()

    assert captured["environment_index"] == 3
    assert captured["reset_seed"] == 13
