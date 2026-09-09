import numpy as np
import pytest
import torch
import gymnasium as gym
import yaml
from types import SimpleNamespace
from pathlib import Path
from gymnasium import spaces
from sb3_contrib.common.recurrent.type_aliases import RNNStates

from experiments.scripts.record_trajectory_24h_all import record_trajectory
from gl_gym.experiments import evaluate_rl
from gl_gym.RL.agri_metarl.memory import Transition
from gl_gym.RL.agri_metarl.memory import TaskSupportMemory
from gl_gym.RL.agri_metarl.meta_advantage_head import TransitionSetEncoder
from gl_gym.RL.context_recurrent_ppo import ContextRecurrentPPO
from gl_gym.RL.context_recurrent_ppo_buffer import ContextRecurrentRolloutBuffer
from gl_gym.RL.context_recurrent_ppo_diagnostics import ContextDiagnostics
from gl_gym.RL.experiment_manager import ExperimentManager


REPO_ROOT = Path(__file__).resolve().parents[2]


def _buffer_states() -> RNNStates:
    pair = (torch.zeros((1, 2, 3)), torch.zeros((1, 2, 3)))
    return RNNStates(pi=pair, vf=pair)


def _support_transition(value: float) -> Transition:
    observation = np.array([value, value + 1.0], dtype=np.float32)
    return Transition(
        observation=observation,
        action=np.array([value], dtype=np.float32),
        reward=value,
        next_observation=observation + 1.0,
        done=False,
    )


def _context_buffer() -> ContextRecurrentRolloutBuffer:
    return ContextRecurrentRolloutBuffer(
        buffer_size=4,
        observation_space=spaces.Box(-1, 1, shape=(2,), dtype=np.float32),
        action_space=spaces.Box(-1, 1, shape=(1,), dtype=np.float32),
        hidden_state_shape=(4, 1, 2, 3),
        n_envs=2,
    )


def _add_context_row(
    buffer: ContextRecurrentRolloutBuffer,
    *,
    task_instance_keys=("task-a", "task-b"),
    context_active_mask=(False, True),
    support_snapshots=None,
    support_sizes=(0, 1),
) -> None:
    if support_snapshots is None:
        support_snapshots = ((), (_support_transition(1.0),))
    buffer.add(
        obs=np.zeros((2, 2), dtype=np.float32),
        action=np.zeros((2, 1), dtype=np.float32),
        reward=np.zeros(2, dtype=np.float32),
        episode_start=np.zeros(2, dtype=bool),
        value=torch.zeros(2),
        log_prob=torch.zeros(2),
        lstm_states=_buffer_states(),
        task_instance_keys=task_instance_keys,
        context_active_mask=context_active_mask,
        support_snapshots=support_snapshots,
        support_sizes=support_sizes,
    )


def test_context_recurrentppo_config_matches_expected_keys():
    config_path = REPO_ROOT / "configs" / "agents" / "context_recurrentppo.yml"

    with config_path.open(encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)

    params = config["TomatoEnv"]
    assert set(params) == {
        "n_envs",
        "total_timesteps",
        "policy",
        "n_steps",
        "batch_size",
        "n_epochs",
        "gamma",
        "gae_lambda",
        "clip_range",
        "normalize_advantage",
        "ent_coef",
        "vf_coef",
        "max_grad_norm",
        "use_sde",
        "sde_sample_freq",
        "target_kl",
        "learning_rate",
        "policy_kwargs",
        "support_size",
        "max_task_instances",
        "context_dim",
        "transition_hidden_dim",
    }
    assert not any(
        key.startswith("calibration_") or key in {"residual_alpha", "meta_loss_weight"}
        for key in params
    )
    assert params["n_envs"] == 8
    assert params["total_timesteps"] == 2_000_000
    assert params["policy"] == "MlpLstmPolicy"
    assert params["n_steps"] == 2048
    assert params["batch_size"] == 512
    assert params["n_epochs"] == 8
    assert params["gamma"] == 0.9631
    assert params["gae_lambda"] == 0.9666
    assert params["clip_range"] == 0.2
    assert params["normalize_advantage"] is True
    assert params["ent_coef"] == 0.00006002718320795429
    assert params["vf_coef"] == 0.2599
    assert params["max_grad_norm"] == 0.3
    assert params["use_sde"] is False
    assert params["sde_sample_freq"] == 8
    assert params["target_kl"] is None
    assert params["learning_rate"] == 0.0001161
    assert params["support_size"] == 256
    assert params["max_task_instances"] == 128
    assert params["context_dim"] == 64
    assert params["transition_hidden_dim"] == 128
    assert params["policy_kwargs"] == {
        "net_arch": {"pi": [1024, 1024], "vf": [128, 128]},
        "optimizer_class": "adam",
        "optimizer_kwargs": {"amsgrad": True},
        "activation_fn": "silu",
        "log_std_init": "np.log(1)",
        "lstm_hidden_size": 256,
        "n_lstm_layers": 1,
        "shared_lstm": False,
        "enable_critic_lstm": True,
    }


def test_experiment_manager_registers_context_recurrentppo():
    manager = ExperimentManager(
        env_id="TomatoEnv",
        project="test-project",
        env_base_params={},
        env_specific_params={},
        hyperparameters={"n_envs": 1, "total_timesteps": 1},
        group="test-group",
        n_eval_episodes=1,
        n_evals=1,
        algorithm="context_recurrentppo",
        env_seed=0,
        model_seed=0,
        stochastic=False,
        hp_tuning=True,
    )

    assert manager.models["context_recurrentppo"] is ContextRecurrentPPO
    assert manager.model_class is ContextRecurrentPPO


def test_evaluate_rl_exposes_context_recurrentppo_in_alg_map():
    assert evaluate_rl.ALG_MAP["context_recurrentppo"] is ContextRecurrentPPO
    assert evaluate_rl.ALG is evaluate_rl.ALG_MAP


class _NoModelTrajectoryEnv:
    action_space = SimpleNamespace(shape=(1,))

    def reset(self):
        return np.zeros((1, 11), dtype=np.float32)

    def step(self, actions):
        return np.zeros((1, 11), dtype=np.float32), np.zeros(1), np.array([False]), {}

    def get_attr(self, name):
        return [np.zeros(4, dtype=np.float32)]


def test_record_trajectory_rejects_unknown_algorithm_without_model_file():
    with pytest.raises(KeyError):
        record_trajectory("unknown_algorithm", None, _NoModelTrajectoryEnv(), n_steps=1)


class _TinyTaskEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self):
        self.observation_space = spaces.Box(-10, 10, shape=(2,), dtype=np.float32)
        self.action_space = spaces.Box(-1, 1, shape=(1,), dtype=np.float32)
        self.step_count = 0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.step_count = 0
        return np.array([0.0, 1.0], dtype=np.float32), {}

    def step(self, action):
        self.step_count += 1
        obs = np.array([float(self.step_count), float(self.step_count + 1)], dtype=np.float32)
        terminated = False
        truncated = False
        info = {
            "task_descriptor": "tiny-task",
            "task_instance_key": "tiny-task-0",
        }
        return obs, 1.0, terminated, truncated, info


def _tiny_context_model(env=None) -> ContextRecurrentPPO:
    return ContextRecurrentPPO(
        "MlpLstmPolicy",
        env or _TinyTaskEnv(),
        n_steps=2,
        batch_size=2,
        n_epochs=1,
        support_size=1,
        context_dim=2,
        transition_hidden_dim=4,
        policy_kwargs={"lstm_hidden_size": 8, "net_arch": [8]},
        seed=0,
        device="cpu",
    )


def test_context_recurrent_ppo_predict_accepts_raw_observation():
    env = _TinyTaskEnv()
    model = _tiny_context_model(env)
    raw_obs, _ = env.reset()

    action, state = model.predict(raw_obs, deterministic=True)

    assert env.action_space.contains(np.asarray(action, dtype=env.action_space.dtype))
    assert state is None or isinstance(state, tuple)


def test_context_recurrent_ppo_load_accepts_raw_env_space(tmp_path):
    raw_env = _TinyTaskEnv()
    model = _tiny_context_model(raw_env)
    save_path = tmp_path / "context_recurrent_ppo"

    model.save(save_path)
    loaded = ContextRecurrentPPO.load(save_path, env=_TinyTaskEnv(), device="cpu")

    assert loaded.raw_observation_space.shape == raw_env.observation_space.shape
    assert loaded.observation_space.shape == (
        raw_env.observation_space.shape[0] + loaded.context_dim,
    )


def test_context_recurrent_ppo_save_load_preserves_context_encoder(tmp_path):
    model = _tiny_context_model()
    expected_parameters = {}
    with torch.no_grad():
        for index, (name, parameter) in enumerate(model.context_encoder.named_parameters()):
            values = torch.arange(
                parameter.numel(),
                dtype=parameter.dtype,
                device=parameter.device,
            ).reshape_as(parameter)
            parameter.copy_(values + float(index))
            expected_parameters[name] = parameter.detach().cpu().clone()
    save_path = tmp_path / "context_recurrent_ppo"

    model.save(save_path)
    loaded = ContextRecurrentPPO.load(save_path, device="cpu")

    for name, parameter in loaded.context_encoder.named_parameters():
        torch.testing.assert_close(parameter.detach().cpu(), expected_parameters[name])


def test_context_diagnostics_reports_finite_defaults():
    diagnostics = ContextDiagnostics()

    summary = diagnostics.summarize()

    assert set(summary) == {
        "train/context_active_fraction",
        "train/context_norm_mean",
        "train/context_norm_std",
        "train/no_context_fraction",
        "train/support_size_mean",
        "train/context_between_task_variance",
    }
    assert all(np.isfinite(value) for value in summary.values())
    assert summary["train/context_active_fraction"] == 0.0
    assert summary["train/no_context_fraction"] == 1.0


def test_context_diagnostics_records_active_and_zero_contexts():
    diagnostics = ContextDiagnostics()
    diagnostics.record_contexts(
        contexts=np.array([[0.0, 0.0], [3.0, 4.0]], dtype=np.float32),
        active_mask=np.array([False, True]),
        support_sizes=np.array([0, 4]),
        task_instance_keys=np.array(["task-a", "task-b"], dtype=object),
    )

    summary = diagnostics.summarize()

    assert summary["train/context_active_fraction"] == 0.5
    assert summary["train/no_context_fraction"] == 0.5
    assert summary["train/context_norm_mean"] == 2.5
    assert summary["train/support_size_mean"] == 2.0
    assert summary["train/context_between_task_variance"] >= 0.0


def test_context_diagnostics_summarizes_mixed_none_and_string_keys():
    diagnostics = ContextDiagnostics()
    diagnostics.record_contexts(
        contexts=np.array([[0.0, 0.0], [3.0, 4.0]], dtype=np.float32),
        active_mask=np.array([False, True]),
        support_sizes=np.array([0, 1]),
        task_instance_keys=np.array([None, "task-a"], dtype=object),
    )

    summary = diagnostics.summarize()

    assert all(np.isfinite(value) for value in summary.values())
    assert summary["train/context_between_task_variance"] >= 0.0


def test_context_diagnostics_rejects_non_batched_context_vector():
    diagnostics = ContextDiagnostics()

    with pytest.raises(ValueError, match="contexts must be a 2D array"):
        diagnostics.record_contexts(
            contexts=np.array([1.0, 2.0, 3.0], dtype=np.float64),
            active_mask=np.array([True]),
            support_sizes=np.array([1.0]),
            task_instance_keys=np.array(["task-a"], dtype=object),
        )


def test_context_diagnostics_snapshots_recorded_inputs():
    diagnostics = ContextDiagnostics()
    contexts = np.array([[3.0, 4.0]], dtype=np.float64)
    active_mask = np.array([True])
    support_sizes = np.array([4.0], dtype=np.float64)
    task_instance_keys = np.array(["task-a"], dtype=object)

    diagnostics.record_contexts(
        contexts=contexts,
        active_mask=active_mask,
        support_sizes=support_sizes,
        task_instance_keys=task_instance_keys,
    )
    contexts[0] = [0.0, 0.0]
    active_mask[0] = False
    support_sizes[0] = 0.0
    task_instance_keys[0] = "mutated-task"

    summary = diagnostics.summarize()

    assert summary["train/context_active_fraction"] == 1.0
    assert summary["train/context_norm_mean"] == 5.0
    assert summary["train/support_size_mean"] == 4.0


def test_context_buffer_stores_keys_masks_and_support_snapshots():
    buffer = _context_buffer()
    first_support = [_support_transition(1.0)]
    second_support = (_support_transition(2.0), _support_transition(3.0))

    _add_context_row(
        buffer,
        task_instance_keys=("task-a", "task-b"),
        context_active_mask=(False, True),
        support_snapshots=(first_support, second_support),
        support_sizes=(1, 2),
    )
    first_support.append(_support_transition(99.0))

    assert buffer.task_instance_keys[0].tolist() == ["task-a", "task-b"]
    assert buffer.context_active_mask[0].tolist() == [False, True]
    assert buffer.support_sizes[0].tolist() == [1, 2]
    assert isinstance(buffer.support_snapshots[0, 0], tuple)
    assert isinstance(buffer.support_snapshots[0, 1], tuple)
    assert len(buffer.support_snapshots[0, 0]) == 1
    assert len(buffer.support_snapshots[0, 1]) == 2


def test_context_buffer_reset_clears_context_metadata():
    buffer = _context_buffer()
    _add_context_row(buffer)

    buffer.reset()

    assert np.all(buffer.task_instance_keys == None)  # noqa: E711
    assert not buffer.context_active_mask.any()
    assert not buffer.support_sizes.any()
    assert all(snapshot == () for snapshot in buffer.support_snapshots.reshape(-1))


def test_context_buffer_rejects_wrong_metadata_length():
    buffer = _context_buffer()

    with pytest.raises(ValueError, match="task_instance_keys.*one.*per environment"):
        _add_context_row(buffer, task_instance_keys=("task-a",))


def test_context_buffer_invalid_metadata_length_does_not_advance_position():
    buffer = _context_buffer()
    original_pos = buffer.pos
    original_full = getattr(buffer, "full", None)

    with pytest.raises(ValueError, match="task_instance_keys.*one.*per environment"):
        _add_context_row(buffer, task_instance_keys=("task-a",))

    assert buffer.pos == original_pos
    if original_full is not None:
        assert buffer.full == original_full


def test_context_algorithm_augments_observation_space_without_mutating_env_space():
    raw_space = spaces.Box(-1, 1, shape=(3,), dtype=np.float32)

    augmented = ContextRecurrentPPO._augmented_observation_space(raw_space, 5)

    assert raw_space.shape == (3,)
    assert augmented.shape == (8,)
    assert augmented.dtype == raw_space.dtype
    assert np.isneginf(augmented.low[-5:]).all()
    assert np.isposinf(augmented.high[-5:]).all()


def test_context_algorithm_rejects_non_box_observation_space():
    with pytest.raises(TypeError, match="Box observation space"):
        ContextRecurrentPPO._augmented_observation_space(spaces.Discrete(3), 5)


def test_context_algorithm_rejects_integer_box_observation_space():
    raw_space = spaces.Box(0, 255, shape=(3,), dtype=np.int32)

    with pytest.raises(TypeError, match="float|floating"):
        ContextRecurrentPPO._augmented_observation_space(raw_space, 5)


def test_zero_context_before_support_is_sufficient():
    algorithm = ContextRecurrentPPO.__new__(ContextRecurrentPPO)
    algorithm.context_dim = 4
    algorithm.support_size = 2
    algorithm.device = torch.device("cpu")

    context, active = algorithm._context_from_support(())

    assert active is False
    assert torch.equal(context, torch.zeros(4))


def test_context_from_scalar_action_support_uses_2d_action_tensor_on_cpu():
    algorithm = ContextRecurrentPPO.__new__(ContextRecurrentPPO)
    algorithm.context_dim = 4
    algorithm.support_size = 1
    algorithm.device = torch.device("cpu")
    algorithm.action_dim = 1
    algorithm.context_encoder = TransitionSetEncoder(
        obs_dim=2,
        action_dim=1,
        context_dim=algorithm.context_dim,
        hidden_dim=8,
    )
    support = (
        Transition(
            observation=np.array([0.0, 1.0], dtype=np.float32),
            action=np.asarray(0.0, dtype=np.float32),
            reward=1.0,
            next_observation=np.array([1.0, 2.0], dtype=np.float32),
            done=False,
        ),
    )

    tensorized = algorithm._tensorize_support(support)
    context, active = algorithm._context_from_support(support)

    assert tensorized["actions"].shape == (1, 1)
    assert active is True
    assert context.device.type == "cpu"
    assert context.shape == (algorithm.context_dim,)


def test_observe_transition_records_support_then_query():
    algorithm = ContextRecurrentPPO.__new__(ContextRecurrentPPO)
    algorithm.support_size = 1
    algorithm.support_memory = TaskSupportMemory(support_size=1, max_instances=4)

    first = algorithm._observe_raw_transitions(
        observations=np.array([[1.0, 2.0]], dtype=np.float32),
        actions=np.array([[0.5]], dtype=np.float32),
        rewards=np.array([1.0], dtype=np.float32),
        next_observations=np.array([[2.0, 3.0]], dtype=np.float32),
        dones=np.array([False]),
        infos=[{"task_descriptor": "task-a", "task_instance_key": "task-a"}],
    )
    second = algorithm._observe_raw_transitions(
        observations=np.array([[2.0, 3.0]], dtype=np.float32),
        actions=np.array([[0.25]], dtype=np.float32),
        rewards=np.array([2.0], dtype=np.float32),
        next_observations=np.array([[3.0, 4.0]], dtype=np.float32),
        dones=np.array([False]),
        infos=[{"task_descriptor": "task-a", "task_instance_key": "task-a"}],
    )

    assert first["context_active_mask"].tolist() == [False]
    assert first["support_snapshots"][0] == ()
    assert first["support_sizes"].tolist() == [0]
    assert second["context_active_mask"].tolist() == [True]
    assert len(second["support_snapshots"][0]) == 1
    assert second["support_sizes"].tolist() == [1]


def test_augment_observations_uses_zero_context_for_inactive_support():
    algorithm = ContextRecurrentPPO.__new__(ContextRecurrentPPO)
    algorithm.context_dim = 2
    algorithm.support_size = 2
    algorithm.device = torch.device("cpu")

    augmented, contexts, active = algorithm._augment_raw_observations(
        raw_observations=np.array([[1.0, 2.0]], dtype=np.float32),
        support_snapshots=((),),
    )

    assert augmented.shape == (1, 4)
    np.testing.assert_array_equal(augmented[0, -2:], np.zeros(2, dtype=np.float32))
    np.testing.assert_array_equal(contexts, np.zeros((1, 2), dtype=np.float32))
    assert active.tolist() == [False]


def test_training_context_recompute_keeps_encoder_in_graph():
    algorithm = ContextRecurrentPPO.__new__(ContextRecurrentPPO)
    algorithm.context_dim = 4
    algorithm.support_size = 1
    algorithm.device = torch.device("cpu")
    algorithm.raw_observation_space = spaces.Box(-10, 10, shape=(2,), dtype=np.float32)
    algorithm.context_encoder = TransitionSetEncoder(
        obs_dim=2,
        action_dim=1,
        context_dim=algorithm.context_dim,
        hidden_dim=8,
    )
    algorithm.action_dim = 1
    support = (_support_transition(1.0),)
    raw_obs = torch.tensor([[2.0, 0.0]], dtype=torch.float32)

    augmented = algorithm._augment_training_observations(raw_obs, (support,))
    loss = augmented[:, -4:].sum()
    loss.backward()

    assert any(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in algorithm.context_encoder.parameters()
    )


def test_training_context_recompute_uses_zero_for_inactive_support():
    algorithm = ContextRecurrentPPO.__new__(ContextRecurrentPPO)
    algorithm.context_dim = 3
    algorithm.support_size = 2
    algorithm.device = torch.device("cpu")
    algorithm.raw_observation_space = spaces.Box(-10, 10, shape=(2,), dtype=np.float32)
    raw_obs = torch.tensor([[2.0, 0.0]], dtype=torch.float32)

    augmented = algorithm._augment_training_observations(raw_obs, ((),))

    expected = torch.tensor([[2.0, 0.0, 0.0, 0.0, 0.0]], dtype=torch.float32)
    torch.testing.assert_close(augmented, expected)


def test_context_buffer_samples_include_support_snapshots_aligned_to_mask():
    buffer = _context_buffer()
    first_support = (_support_transition(1.0),)
    second_support = (_support_transition(2.0),)
    _add_context_row(
        buffer,
        support_snapshots=(first_support, second_support),
        support_sizes=(1, 1),
    )
    _add_context_row(
        buffer,
        support_snapshots=((), ()),
        support_sizes=(0, 0),
    )
    _add_context_row(buffer)
    _add_context_row(buffer)

    sample = next(buffer.get(batch_size=2))
    assert hasattr(sample, "support_snapshots")
    assert len(sample.support_snapshots) == sample.observations.shape[0]
    assert all(isinstance(snapshot, tuple) for snapshot in sample.support_snapshots)


def test_learn_collects_rollout_with_context_metadata_smoke():
    model = ContextRecurrentPPO(
        "MlpLstmPolicy",
        _TinyTaskEnv(),
        n_steps=2,
        batch_size=2,
        n_epochs=1,
        support_size=1,
        context_dim=2,
        transition_hidden_dim=4,
        policy_kwargs={"lstm_hidden_size": 8, "net_arch": [8]},
        seed=0,
    )

    model.learn(total_timesteps=2)

    assert model._last_obs.shape == (1, 2)
    assert model.rollout_buffer.observations.shape[-1] == 4
    assert model.rollout_buffer.task_instance_keys[0, 0] == "tiny-task-0"
    assert model.rollout_buffer.context_active_mask[0, 0] == False  # noqa: E712
    assert model.rollout_buffer.support_sizes[1, 0] == 1


def test_learn_context_diagnostics_summarizes_first_step_none_keys():
    model = ContextRecurrentPPO(
        "MlpLstmPolicy",
        _TinyTaskEnv(),
        n_steps=2,
        batch_size=2,
        n_epochs=1,
        support_size=1,
        context_dim=2,
        transition_hidden_dim=4,
        policy_kwargs={"lstm_hidden_size": 8, "net_arch": [8]},
        seed=0,
    )

    model.learn(total_timesteps=2)
    summary = model.context_diagnostics.summarize()

    assert all(np.isfinite(value) for value in summary.values())
    assert summary["train/context_between_task_variance"] >= 0.0


def test_collect_rollout_stores_policy_support_metadata_not_post_step_support():
    model = ContextRecurrentPPO(
        "MlpLstmPolicy",
        _TinyTaskEnv(),
        n_steps=1,
        batch_size=1,
        n_epochs=1,
        support_size=1,
        context_dim=2,
        transition_hidden_dim=4,
        policy_kwargs={"lstm_hidden_size": 8, "net_arch": [8]},
        seed=0,
    )
    model.support_memory.observe("tiny-task-0", _support_transition(42.0))
    model._last_task_instance_keys = np.array([None], dtype=object)

    model.learn(total_timesteps=1)

    assert model.rollout_buffer.task_instance_keys[0, 0] == "tiny-task-0"
    assert model.rollout_buffer.support_sizes[0, 0] == 0
    assert model.rollout_buffer.context_active_mask[0, 0] == False  # noqa: E712
    assert model.rollout_buffer.support_snapshots[0, 0] == ()


def test_single_sample_advantage_normalization_keeps_parameters_finite():
    model = ContextRecurrentPPO(
        "MlpLstmPolicy",
        _TinyTaskEnv(),
        n_steps=1,
        batch_size=1,
        n_epochs=1,
        normalize_advantage=True,
        support_size=1,
        context_dim=2,
        transition_hidden_dim=4,
        policy_kwargs={"lstm_hidden_size": 8, "net_arch": [8]},
        seed=0,
    )

    model.learn(total_timesteps=1)

    parameters = list(model.policy.parameters()) + list(model.context_encoder.parameters())
    assert parameters
    assert all(torch.isfinite(parameter).all().item() for parameter in parameters)


def test_context_recurrent_ppo_smoke_updates_encoder_parameters():
    model = ContextRecurrentPPO(
        "MlpLstmPolicy",
        _TinyTaskEnv(),
        n_steps=2,
        batch_size=2,
        n_epochs=1,
        support_size=1,
        max_task_instances=4,
        context_dim=4,
        transition_hidden_dim=8,
        policy_kwargs={"lstm_hidden_size": 8, "net_arch": [8]},
        seed=0,
        device="cpu",
    )
    before_parameters = {
        name: parameter.detach().clone()
        for name, parameter in model.context_encoder.named_parameters()
    }

    model.learn(total_timesteps=8)

    assert model.num_timesteps >= 8
    assert model._n_updates > 0
    assert any(
        not torch.equal(before_parameters[name], parameter.detach())
        for name, parameter in model.context_encoder.named_parameters()
    )
    assert all(
        torch.isfinite(parameter).all().item()
        for parameter in model.context_encoder.parameters()
    )
    assert model.context_diagnostics.last_summary["train/context_active_fraction"] >= 0.0


def test_context_recurrent_ppo_has_no_advantage_residual_or_calibration():
    model = ContextRecurrentPPO(
        "MlpLstmPolicy",
        _TinyTaskEnv(),
        n_steps=2,
        batch_size=2,
        n_epochs=1,
        support_size=1,
        max_task_instances=4,
        context_dim=4,
        transition_hidden_dim=8,
        policy_kwargs={"lstm_hidden_size": 8, "net_arch": [8]},
        seed=0,
        device="cpu",
    )

    assert not hasattr(model, "residual_head")
    assert not hasattr(model, "calibration_memory")
