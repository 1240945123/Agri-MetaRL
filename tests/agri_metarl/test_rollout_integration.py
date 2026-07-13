import numpy as np
import pytest
import torch
import yaml
import gymnasium as gym
from types import SimpleNamespace
from gymnasium import spaces
from sb3_contrib.common.recurrent.type_aliases import RNNStates

from gl_gym.RL.agri_metarl.buffer import AgriMetaRLRolloutBuffer
from gl_gym.RL.agri_metarl.calibration import EpisodeCalibrationMemory
from gl_gym.RL.agri_metarl.calibration import (
    CalibrationSample,
    CompletedCalibrationEpisode,
)
from gl_gym.RL.agri_metarl.diagnostics import MetaDiagnostics
from gl_gym.RL.agri_metarl.agri_metarl import AgriMetaRL
from gl_gym.RL.agri_metarl.legacy_agri_metarl import LegacyAgriMetaRL
from gl_gym.RL.agri_metarl.memory import TaskSupportMemory
from gl_gym.RL.agri_metarl.memory import Transition
from gl_gym.RL.agri_metarl.meta_advantage_head import (
    AdvantageResidualHead,
    TransitionSetEncoder,
)


def make_buffer():
    return AgriMetaRLRolloutBuffer(
        buffer_size=4,
        observation_space=spaces.Box(-1, 1, shape=(2,), dtype=np.float32),
        action_space=spaces.Box(-1, 1, shape=(1,), dtype=np.float32),
        hidden_state_shape=(4, 1, 2, 3),
        n_envs=2,
    )


def states():
    pair = (torch.zeros((1, 2, 3)), torch.zeros((1, 2, 3)))
    return RNNStates(pi=pair, vf=pair)


def add_row(buffer, keys, query_mask, entry_ids=(10, 20)):
    buffer.add(
        obs=np.zeros((2, 2), dtype=np.float32),
        action=np.zeros((2, 1), dtype=np.float32),
        reward=np.zeros(2, dtype=np.float32),
        episode_start=np.zeros(2, dtype=bool),
        value=torch.zeros(2),
        log_prob=torch.zeros(2),
        lstm_states=states(),
        task_instance_keys=keys,
        query_mask=query_mask,
        calibration_entry_ids=entry_ids,
    )


def test_buffer_stores_isolated_task_instance_keys_and_query_masks():
    buffer = make_buffer()
    add_row(buffer, ["task-a", "task-b"], [False, True])
    add_row(buffer, ["task-a", "task-b"], [True, True])

    assert buffer.task_instance_keys[:2].tolist() == [
        ["task-a", "task-b"],
        ["task-a", "task-b"],
    ]
    assert buffer.query_mask[:2].tolist() == [[False, True], [True, True]]


def test_rollout_reset_clears_arrays_and_accepts_same_instance_again():
    buffer = make_buffer()
    add_row(buffer, ["task-a", "task-b"], [False, False])

    buffer.reset()

    assert np.all(buffer.task_instance_keys == None)  # noqa: E711
    assert not buffer.query_mask.any()
    add_row(buffer, ["task-a", "task-c"], [True, False])
    assert buffer.task_instance_keys[0].tolist() == ["task-a", "task-c"]
    assert buffer.query_mask[0].tolist() == [True, False]


def test_buffer_stores_calibration_entry_ids_and_reset_clears_them():
    buffer = make_buffer()
    add_row(buffer, ["task-a", "task-b"], [False, True], [7, 11])
    assert buffer.calibration_entry_ids[0].tolist() == [7, 11]
    buffer.reset()
    assert np.all(buffer.calibration_entry_ids == -1)


def test_buffer_stores_support_snapshots_and_sizes():
    buffer = make_buffer()
    support_a = (Transition(np.array([1.0, 0.0]), np.array([0.0]), 1.0, np.array([2.0, 0.0]), False),)
    support_b = (
        Transition(np.array([9.0, 0.0]), np.array([0.0]), 9.0, np.array([10.0, 0.0]), False),
        Transition(np.array([8.0, 0.0]), np.array([0.0]), 8.0, np.array([9.0, 0.0]), False),
    )

    buffer.add(
        obs=np.zeros((2, 2), dtype=np.float32),
        action=np.zeros((2, 1), dtype=np.float32),
        reward=np.zeros(2, dtype=np.float32),
        episode_start=np.zeros(2, dtype=bool),
        value=torch.zeros(2),
        log_prob=torch.zeros(2),
        lstm_states=states(),
        task_instance_keys=["task-a", "task-b"],
        query_mask=[False, True],
        calibration_entry_ids=[7, 11],
        support_snapshots=(support_a, support_b),
        support_sizes=[1, 2],
    )

    assert buffer.support_sizes[0].tolist() == [1, 2]
    assert buffer.support_snapshots[0, 0] == support_a
    assert buffer.support_snapshots[0, 1] == support_b


def test_buffer_samples_include_support_snapshots():
    buffer = make_buffer()
    support_a = (Transition(np.array([1.0, 0.0]), np.array([0.0]), 1.0, np.array([2.0, 0.0]), False),)
    support_b = (Transition(np.array([9.0, 0.0]), np.array([0.0]), 9.0, np.array([10.0, 0.0]), False),)
    add_row(
        buffer,
        ["task-a", "task-b"],
        [True, True],
    )
    add_row(buffer, ["task-a", "task-b"], [True, True])
    add_row(buffer, ["task-a", "task-b"], [True, True])
    add_row(buffer, ["task-a", "task-b"], [True, True])
    for row in range(buffer.buffer_size):
        buffer.support_snapshots[row, 0] = support_a
        buffer.support_snapshots[row, 1] = support_b
    buffer.compute_returns_and_advantage(
        last_values=torch.zeros(2),
        dones=np.zeros(2, dtype=bool),
    )

    sample = next(buffer.get(batch_size=4))

    assert hasattr(sample, "support_snapshots")
    assert len(sample.support_snapshots) == sample.observations.shape[0]
    observed_rewards = [
        snapshot[0].reward for snapshot in sample.support_snapshots if len(snapshot) == 1
    ]
    assert observed_rewards
    assert set(observed_rewards).issubset({support_a[0].reward, support_b[0].reward})


def test_legacy_algorithm_and_configuration_remain_explicitly_loadable():
    with open("configs/agents/agri_metarl_legacy.yml", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)["TomatoEnv"]
    assert issubclass(LegacyAgriMetaRL, object)
    assert config["meta_support_ratio"] == 0.5
    assert config["meta_advantage_clip"] == 2.0


def test_support_warmup_crosses_rollout_boundaries_without_task_leakage():
    algorithm = AgriMetaRL.__new__(AgriMetaRL)
    algorithm.support_memory = TaskSupportMemory(support_size=2, max_instances=4)
    algorithm.calibration_memory = EpisodeCalibrationMemory(0.9, 0.5, 4, 4)
    observations = np.array([[1.0, 0.0], [9.0, 0.0]], dtype=np.float32)
    actions = np.zeros((2, 1), dtype=np.float32)
    next_observations = observations + 1
    dones = np.zeros(2, dtype=bool)
    infos = [
        {"task_descriptor": {}, "task_instance_key": "task-a"},
        {"task_descriptor": {}, "task_instance_key": "task-b"},
    ]

    for rewards in (np.array([1.0, 9.0]), np.array([2.0, 8.0])):
        _, query_mask, _ = algorithm._observe_transitions(
            observations,
            actions,
            rewards,
            next_observations,
            dones,
            infos,
            np.zeros(2),
        )
        assert query_mask.tolist() == [False, False]

    algorithm.support_memory.begin_rollout()
    keys, query_mask, _ = algorithm._observe_transitions(
        observations,
        actions,
        np.array([3.0, 7.0]),
        next_observations,
        dones,
        infos,
        np.zeros(2),
    )

    assert keys.tolist() == ["task-a", "task-b"]
    assert query_mask.tolist() == [True, True]
    assert [item.reward for item in algorithm.support_memory.support("task-a")] == [
        1.0,
        2.0,
    ]
    assert [item.reward for item in algorithm.support_memory.support("task-b")] == [
        9.0,
        8.0,
    ]


def test_attach_rollout_calibration_finalizes_terminal_episode():
    algorithm = AgriMetaRL.__new__(AgriMetaRL)
    algorithm.support_size = 1
    algorithm.support_memory = TaskSupportMemory(support_size=1, max_instances=2)
    algorithm.calibration_memory = EpisodeCalibrationMemory(0.5, 1.0, 2, 2)
    algorithm.meta_diagnostics = MetaDiagnostics(residual_alpha=1.0)
    observations = np.array([[1.0, 0.0]], dtype=np.float32)
    actions = np.zeros((1, 1), dtype=np.float32)
    infos = [{"task_descriptor": {}, "task_instance_key": "task-a"}]

    _, first_query, first_ids = algorithm._observe_transitions(
        observations,
        actions,
        np.array([1.0]),
        observations + 1,
        np.array([False]),
        infos,
        np.array([0.5]),
    )
    _, second_query, second_ids = algorithm._observe_transitions(
        observations + 1,
        actions,
        np.array([4.0]),
        observations + 2,
        np.array([True]),
        infos,
        np.array([1.0]),
    )
    assert first_query.tolist() == [False]
    assert second_query.tolist() == [True]
    algorithm.rollout_buffer = SimpleNamespace(
        calibration_entry_ids=np.array([[first_ids[0]], [second_ids[0]]]),
        advantages=np.array([[0.25], [0.75]], dtype=np.float32),
    )

    algorithm._attach_rollout_calibration()

    assert algorithm.calibration_memory.completed_episode_count == 1
    episode = algorithm.calibration_memory.pop_completed(1, 10)[0]
    assert episode.task_instance_key == "task-a"
    assert episode.samples[0].raw_advantage == 0.75


def make_calibration_algorithm():
    algorithm = AgriMetaRL.__new__(AgriMetaRL)
    algorithm.device = torch.device("cpu")
    algorithm.context_encoder = TransitionSetEncoder(2, 1, context_dim=4, hidden_dim=8)
    algorithm.residual_head = AdvantageResidualHead(2, context_dim=4, alpha=0.5)
    algorithm.meta_optimizer = torch.optim.Adam(
        list(algorithm.context_encoder.parameters())
        + list(algorithm.residual_head.parameters()),
        lr=0.01,
    )
    algorithm.meta_loss_weight = 1.0
    algorithm.residual_regularization = 0.01
    algorithm.max_grad_norm = 1.0
    algorithm.meta_diagnostics = MetaDiagnostics(residual_alpha=0.5)
    return algorithm


def calibration_episode(target=0.4):
    support = (
        Transition(
            np.array([1.0, 0.0]),
            np.array([0.1]),
            1.0,
            np.array([2.0, 0.0]),
            False,
        ),
    )
    return CompletedCalibrationEpisode(
        task_instance_key="task-a",
        support=support,
        samples=(CalibrationSample(np.array([2.0, 0.0]), 0.2, target),),
        mc_gae_abs_difference_mean=abs(target),
        target_clip_fraction=0.0,
    )


def test_supervised_calibration_update_decreases_loss_and_changes_parameters():
    algorithm = make_calibration_algorithm()
    before = {
        name: parameter.detach().clone()
        for name, parameter in algorithm.residual_head.named_parameters()
    }
    first_loss = algorithm._train_calibration_batch((calibration_episode(),))
    second_loss = algorithm._train_calibration_batch((calibration_episode(),))
    after = dict(algorithm.residual_head.named_parameters())
    assert second_loss < first_loss
    assert any(not torch.equal(before[name], after[name]) for name in before)


def test_nonfinite_calibration_batch_does_not_mutate_parameters():
    algorithm = make_calibration_algorithm()
    bad_sample = SimpleNamespace(
        observation=np.array([2.0, 0.0]), raw_advantage=0.2, target_residual=np.nan
    )
    valid = calibration_episode()
    bad_episode = SimpleNamespace(support=valid.support, samples=(bad_sample,))
    before = [
        parameter.detach().clone()
        for parameter in list(algorithm.context_encoder.parameters())
        + list(algorithm.residual_head.parameters())
    ]
    assert algorithm._train_calibration_batch((bad_episode,)) is None
    after = list(algorithm.context_encoder.parameters()) + list(
        algorithm.residual_head.parameters()
    )
    assert algorithm.meta_diagnostics.nonfinite_meta_batch_count == 1
    assert all(torch.equal(old, new) for old, new in zip(before, after, strict=True))


def test_meta_diagnostics_reports_calibration_metrics_as_finite_values():
    diagnostics = MetaDiagnostics(residual_alpha=0.5)
    diagnostics.mc_gae_abs_differences.append(0.75)
    diagnostics.target_clip_fractions.append(0.25)
    diagnostics.calibration_queue_size = 3
    diagnostics.completed_episode_count = 1
    summary = diagnostics.summarize()
    expected = {
        "train/meta_loss",
        "train/context_norm_mean",
        "train/context_between_task_variance",
        "train/residual_abs_mean",
        "train/residual_saturation_rate",
        "train/query_correction_fraction",
        "train/calibration_queue_size",
        "train/completed_episode_count",
        "train/mc_gae_abs_difference_mean",
        "train/target_residual_clip_fraction",
        "train/nonfinite_meta_batch_count",
    }
    assert set(summary) == expected
    assert all(np.isfinite(value) for value in summary.values())


class TinyTaskEnv(gym.Env):
    def __init__(self):
        self.observation_space = spaces.Box(
            -10, 10, shape=(3,), dtype=np.float32
        )
        self.action_space = spaces.Box(-1, 1, shape=(1,), dtype=np.float32)
        self.episode_index = -1
        self.step_index = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.episode_index += 1
        self.step_index = 0
        return np.zeros(3, dtype=np.float32), {}

    def step(self, action):
        self.step_index += 1
        observation = np.full(3, self.step_index, dtype=np.float32)
        terminated = self.step_index >= 6
        info = {
            "task_descriptor": {
                "weather_year": 2010,
                "start_day": 59,
                "parameter_uncertainty": 0.0,
                "economic_scenario": "standard",
                "climate_constraint_scenario": "standard",
            },
            "task_instance_key": (
                "2010:59:0.000000:standard:standard:"
                f"env0:episode{self.episode_index}"
            ),
        }
        reward = float(1.0 - abs(float(action[0])))
        return observation, reward, terminated, False, info


def make_tiny_agri_metarl(support_size=2):
    return AgriMetaRL(
        "MlpLstmPolicy",
        TinyTaskEnv(),
        n_steps=2,
        batch_size=2,
        n_epochs=1,
        support_size=support_size,
        max_task_instances=4,
        context_dim=4,
        transition_hidden_dim=8,
        policy_kwargs={
            "net_arch": {"pi": [8], "vf": [8]},
            "lstm_hidden_size": 8,
        },
        seed=3,
        verbose=0,
        device="cpu",
    )


def test_inference_episode_uses_fresh_memory_separate_from_training_memory():
    model = make_tiny_agri_metarl(support_size=3)

    model.begin_inference_episode("online_context")

    assert isinstance(model._inference_support_memory, TaskSupportMemory)
    assert model._inference_support_memory is not model.support_memory
    assert model._inference_support_memory.support_size == 3
    assert model._inference_task_key is None
    assert model._inference_mode == "online_context"

    model.end_inference_episode()

    assert model._inference_support_memory is None
    assert model._inference_task_key is None
    assert model._inference_mode is None


def test_inference_episode_rejects_unknown_mode():
    model = make_tiny_agri_metarl()

    with pytest.raises(ValueError, match="inference mode"):
        model.begin_inference_episode("adaptive_magic")


def _inference_info(key="eval-task-1"):
    return {
        "task_descriptor": {
            "weather_year": 2011,
            "start_day": 59,
            "parameter_uncertainty": 0.0,
            "economic_scenario": "standard",
            "climate_constraint_scenario": "standard",
        },
        "task_instance_key": key,
    }


def test_online_inference_accumulates_only_evaluation_transitions():
    model = make_tiny_agri_metarl(support_size=1)
    model.begin_inference_episode("online_context")

    model.observe_inference_transition(
        observation=np.zeros(3, dtype=np.float32),
        action=np.zeros(1, dtype=np.float32),
        reward=1.0,
        next_observation=np.ones(3, dtype=np.float32),
        done=False,
        info=_inference_info(),
    )

    assert len(model._inference_support_memory.support("eval-task-1")) == 1
    assert model.support_memory.support("eval-task-1") == ()
    assert model._inference_support_ready_step == 1


@pytest.mark.parametrize("batched", [False, True])
def test_online_predict_uses_context_only_after_support_is_ready(
    monkeypatch, batched
):
    model = make_tiny_agri_metarl(support_size=1)
    seen = []

    def fake_policy_predict(observation, **kwargs):
        seen.append(np.asarray(observation).copy())
        action = np.zeros((1, 1) if batched else (1,), dtype=np.float32)
        return action, None

    monkeypatch.setattr(model.policy, "predict", fake_policy_predict)
    model.begin_inference_episode("online_context")
    first_observation = np.zeros((1, 3) if batched else (3,), dtype=np.float32)
    model.predict(first_observation, deterministic=True)
    model.observe_inference_transition(
        np.zeros(3, dtype=np.float32),
        np.zeros(1, dtype=np.float32),
        1.0,
        np.ones(3, dtype=np.float32),
        False,
        _inference_info(),
    )
    monkeypatch.setattr(
        model,
        "_context_from_support",
        lambda support: (torch.ones(model.context_dim, device=model.device), True),
    )
    second_observation = np.ones((1, 3) if batched else (3,), dtype=np.float32)
    model.predict(second_observation, deterministic=True)

    np.testing.assert_array_equal(seen[0][..., -model.context_dim :], 0.0)
    np.testing.assert_array_equal(seen[1][..., -model.context_dim :], 1.0)


def test_zero_context_mode_never_reads_inference_support(monkeypatch):
    model = make_tiny_agri_metarl(support_size=1)
    model.begin_inference_episode("zero_context")
    monkeypatch.setattr(
        model,
        "_context_from_support",
        lambda support: pytest.fail("zero-context mode encoded support"),
    )

    action, _ = model.predict(np.zeros(3, dtype=np.float32), deterministic=True)

    assert action.shape == (1,)


@pytest.mark.parametrize("bad_info", [{}, {"task_instance_key": "k"}])
def test_online_transition_requires_complete_task_identity(bad_info):
    model = make_tiny_agri_metarl()
    model.begin_inference_episode("online_context")

    with pytest.raises(KeyError, match="task identity"):
        model.observe_inference_transition(
            np.zeros(3), np.zeros(1), 1.0, np.ones(3), False, bad_info
        )


@pytest.mark.parametrize(
    ("observation", "action", "reward", "next_observation"),
    [
        (np.array([np.nan, 0.0, 0.0]), np.zeros(1), 1.0, np.ones(3)),
        (np.zeros(3), np.array([np.inf]), 1.0, np.ones(3)),
        (np.zeros(3), np.zeros(1), np.nan, np.ones(3)),
        (np.zeros(3), np.zeros(1), 1.0, np.array([0.0, np.inf, 0.0])),
    ],
)
def test_online_transition_rejects_nonfinite_values(
    observation, action, reward, next_observation
):
    model = make_tiny_agri_metarl()
    model.begin_inference_episode("online_context")

    with pytest.raises(ValueError, match="finite"):
        model.observe_inference_transition(
            observation,
            action,
            reward,
            next_observation,
            False,
            _inference_info(),
        )


@pytest.mark.parametrize(
    "huge_field", ["observation", "action", "reward", "next_observation"]
)
def test_online_transition_rejects_float32_overflow_without_mutating_memory(
    huge_field,
):
    model = make_tiny_agri_metarl(support_size=1)
    model.begin_inference_episode("online_context")
    values = {
        "observation": np.zeros(3, dtype=np.float64),
        "action": np.zeros(1, dtype=np.float64),
        "reward": 1.0,
        "next_observation": np.ones(3, dtype=np.float64),
    }
    if huge_field == "reward":
        values[huge_field] = 1e300
    else:
        values[huge_field][0] = 1e300

    with pytest.raises(ValueError, match="finite"):
        model.observe_inference_transition(
            values["observation"],
            values["action"],
            values["reward"],
            values["next_observation"],
            False,
            _inference_info(),
        )

    assert model._inference_task_key is None
    assert model._inference_step == 0
    assert model._inference_support_memory.support("eval-task-1") == ()


def test_online_transition_canonicalizes_singleton_evaluator_batch():
    model = make_tiny_agri_metarl(support_size=1)
    model.begin_inference_episode("online_context")

    model.observe_inference_transition(
        np.zeros((1, 3), dtype=np.float32),
        np.zeros((1, 1), dtype=np.float32),
        1.0,
        np.ones((1, 3), dtype=np.float32),
        False,
        _inference_info(),
    )

    transition = model._inference_support_memory.support("eval-task-1")[0]
    assert transition.observation.shape == (3,)
    assert transition.next_observation.shape == (3,)
    assert transition.action.shape == (1,)
    assert model._inference_support_ready_step == 1
    action, _ = model.predict(
        np.ones((1, 3), dtype=np.float32), deterministic=True
    )
    assert action.shape == (1, 1)


@pytest.mark.parametrize(
    ("observation", "action", "next_observation"),
    [
        (np.zeros((2, 3)), np.zeros(1), np.ones(3)),
        (np.zeros(3), np.zeros(1), np.ones((2, 3))),
        (np.zeros(3), np.zeros(2), np.ones(3)),
        (np.zeros(3), np.zeros((2, 1)), np.ones(3)),
    ],
)
def test_online_transition_rejects_non_singleton_evaluator_shapes(
    observation, action, next_observation
):
    model = make_tiny_agri_metarl()
    model.begin_inference_episode("online_context")

    with pytest.raises(ValueError, match="shape"):
        model.observe_inference_transition(
            observation,
            action,
            1.0,
            next_observation,
            False,
            _inference_info(),
        )


def test_online_predict_rejects_more_than_one_evaluation_environment():
    model = make_tiny_agri_metarl()
    model.begin_inference_episode("online_context")

    with pytest.raises(ValueError, match="single evaluation environment"):
        model.predict(np.zeros((2, 3), dtype=np.float32), deterministic=True)


def test_inference_episode_diagnostics_are_json_friendly():
    model = make_tiny_agri_metarl(support_size=1)
    model.begin_inference_episode("online_context")
    empty = model.inference_episode_diagnostics()
    assert np.isnan(empty["support_ready_step"])
    assert empty["context_norm_mean"] == 0.0
    assert empty["context_norm_max"] == 0.0

    model._inference_support_ready_step = 1
    model._inference_context_norms.extend([2.0, 4.0])
    diagnostics = model.inference_episode_diagnostics()
    assert diagnostics == {
        "support_ready_step": 1.0,
        "context_norm_mean": 3.0,
        "context_norm_max": 4.0,
    }
    assert all(isinstance(value, float) for value in diagnostics.values())


def test_three_rollout_cpu_smoke_trains_calibration_without_nonfinite_values():
    model = AgriMetaRL(
        "MlpLstmPolicy",
        TinyTaskEnv(),
        n_steps=2,
        batch_size=2,
        n_epochs=1,
        support_size=2,
        max_task_instances=4,
        context_dim=4,
        transition_hidden_dim=8,
        residual_alpha=0.25,
        calibration_min_query_samples=1,
        calibration_max_query_samples=16,
        policy_kwargs={
            "net_arch": {"pi": [8], "vf": [8]},
            "lstm_hidden_size": 8,
        },
        seed=3,
        verbose=0,
        device="cpu",
    )

    model.learn(total_timesteps=6)

    assert model.num_timesteps >= 6
    assert model.meta_diagnostics.last_summary[
        "train/nonfinite_meta_batch_count"
    ] == 0
    assert model.meta_diagnostics.last_summary["train/completed_episode_count"] >= 1
    assert model.meta_diagnostics.last_summary["train/meta_loss"] >= 0
    assert all(
        torch.isfinite(parameter).all()
        for parameter in model.context_encoder.parameters()
    )
    assert all(
        torch.isfinite(parameter).all()
        for parameter in model.residual_head.parameters()
    )
    assert model._n_updates > 0


def test_agri_metarl_passes_calibration_queue_sample_bound_to_memory():
    model = AgriMetaRL(
        "MlpLstmPolicy",
        TinyTaskEnv(),
        n_steps=2,
        batch_size=2,
        n_epochs=1,
        support_size=2,
        max_task_instances=4,
        context_dim=4,
        transition_hidden_dim=8,
        calibration_max_queue_samples=5,
        policy_kwargs={
            "net_arch": {"pi": [8], "vf": [8]},
            "lstm_hidden_size": 8,
        },
        seed=3,
        verbose=0,
        device="cpu",
    )

    assert model.calibration_memory.max_completed_query_samples == 5


def test_agri_metarl_augments_policy_observation_space_without_mutating_env_space():
    env = TinyTaskEnv()
    raw_shape = env.observation_space.shape
    model = AgriMetaRL(
        "MlpLstmPolicy",
        env,
        n_steps=2,
        batch_size=2,
        n_epochs=1,
        support_size=2,
        max_task_instances=4,
        context_dim=3,
        transition_hidden_dim=8,
        policy_kwargs={"net_arch": {"pi": [8], "vf": [8]}, "lstm_hidden_size": 8},
        seed=0,
        verbose=0,
        device="cpu",
    )

    assert env.observation_space.shape == raw_shape
    assert model.raw_observation_space.shape == raw_shape
    assert model.observation_space.shape == (raw_shape[0] + 3,)


def test_agri_metarl_predict_accepts_raw_observation():
    model = AgriMetaRL(
        "MlpLstmPolicy",
        TinyTaskEnv(),
        n_steps=2,
        batch_size=2,
        n_epochs=1,
        support_size=2,
        max_task_instances=4,
        context_dim=3,
        transition_hidden_dim=8,
        policy_kwargs={"net_arch": {"pi": [8], "vf": [8]}, "lstm_hidden_size": 8},
        seed=0,
        verbose=0,
        device="cpu",
    )
    raw_obs = np.zeros(model.raw_observation_space.shape, dtype=np.float32)

    action, state = model.predict(raw_obs, deterministic=True)

    assert action.shape == model.action_space.shape
    assert state is None or isinstance(state, tuple)


def test_agri_metarl_policy_receives_augmented_observations_during_learning():
    model = AgriMetaRL(
        "MlpLstmPolicy",
        TinyTaskEnv(),
        n_steps=2,
        batch_size=2,
        n_epochs=1,
        support_size=1,
        max_task_instances=4,
        context_dim=2,
        transition_hidden_dim=8,
        policy_kwargs={"net_arch": {"pi": [8], "vf": [8]}, "lstm_hidden_size": 8},
        seed=0,
        verbose=0,
        device="cpu",
    )

    model.learn(total_timesteps=4)

    assert model.rollout_buffer.observations.shape[-1] == (
        model.raw_observation_space.shape[0] + 2
    )
    assert model.rollout_buffer.query_mask.any()
    assert hasattr(model.rollout_buffer, "support_snapshots")


def test_agri_metarl_policy_loss_updates_context_encoder_without_completed_episode():
    model = AgriMetaRL(
        "MlpLstmPolicy",
        TinyTaskEnv(),
        n_steps=2,
        batch_size=2,
        n_epochs=1,
        support_size=1,
        max_task_instances=4,
        context_dim=2,
        transition_hidden_dim=8,
        policy_kwargs={"net_arch": {"pi": [8], "vf": [8]}, "lstm_hidden_size": 8},
        seed=0,
        verbose=0,
        device="cpu",
    )
    before = {
        name: parameter.detach().clone()
        for name, parameter in model.context_encoder.named_parameters()
    }

    model.learn(total_timesteps=4)

    assert any(
        not torch.equal(before[name], parameter.detach())
        for name, parameter in model.context_encoder.named_parameters()
    )


def test_constraint_penalty_from_info_uses_configured_violation_weights():
    algorithm = AgriMetaRL.__new__(AgriMetaRL)
    algorithm.temp_violation_weight = 0.1
    algorithm.co2_violation_weight = 0.01
    algorithm.rh_violation_weight = 0.001

    penalty = algorithm._constraint_penalty_from_info(
        {"temp_violation": 2.0, "co2_violation": 30.0, "rh_violation": 400.0}
    )

    assert penalty == 0.9
