"""
Agri-MetaRL: Recurrent PPO with support/query MetaAdvantageHead for task-adaptive advantage correction.
"""
from copy import deepcopy
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch as th
from gymnasium import spaces
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.save_util import load_from_zip_file
from stable_baselines3.common.type_aliases import GymEnv, Schedule
from stable_baselines3.common.utils import explained_variance, obs_as_tensor
from stable_baselines3.common.vec_env import VecEnv

from sb3_contrib.common.recurrent.buffers import RecurrentDictRolloutBuffer, RecurrentRolloutBuffer
from sb3_contrib.common.recurrent.policies import RecurrentActorCriticPolicy
from sb3_contrib.ppo_recurrent import RecurrentPPO

from gl_gym.RL.agri_metarl.buffer import AgriMetaRLRolloutBuffer
from gl_gym.RL.agri_metarl.calibration import EpisodeCalibrationMemory
from gl_gym.RL.agri_metarl.diagnostics import MetaDiagnostics
from gl_gym.RL.agri_metarl.memory import TaskSupportMemory, Transition
from gl_gym.RL.agri_metarl.meta_advantage_head import (
    AdvantageResidualHead,
    TransitionSetEncoder,
)


class AgriMetaRL(RecurrentPPO):
    """
    Recurrent PPO with MetaAdvantageHead: support set encodes task context,
    query set uses corrected advantages A_tilde = A + delta(obs, A, c_tau).
    """

    INFERENCE_MODES = frozenset({"online_context", "zero_context"})

    def __init__(
        self,
        policy: str | type[RecurrentActorCriticPolicy],
        env: GymEnv | str,
        support_size: int = 256,
        max_task_instances: int = 128,
        context_dim: int = 64,
        transition_hidden_dim: int = 128,
        residual_alpha: float = 0.5,
        meta_loss_weight: float = 1.0,
        residual_regularization: float = 0.05,
        max_pending_episodes: int = 32,
        max_completed_episodes: int = 128,
        calibration_min_query_samples: int = 32,
        calibration_max_query_samples: int = 1024,
        calibration_max_queue_samples: int = 16_384,
        constraint_penalty_weight: float = 0.0,
        temp_violation_weight: float = 0.0,
        co2_violation_weight: float = 0.0,
        rh_violation_weight: float = 0.0,
        **kwargs,
    ):
        self.support_size = support_size
        self.max_task_instances = max_task_instances
        self.context_dim = context_dim
        self.transition_hidden_dim = transition_hidden_dim
        self.residual_alpha = residual_alpha
        self.meta_loss_weight = meta_loss_weight
        self.residual_regularization = residual_regularization
        self.max_pending_episodes = max_pending_episodes
        self.max_completed_episodes = max_completed_episodes
        self.calibration_min_query_samples = calibration_min_query_samples
        self.calibration_max_query_samples = calibration_max_query_samples
        self.calibration_max_queue_samples = calibration_max_queue_samples
        self.constraint_penalty_weight = constraint_penalty_weight
        self.temp_violation_weight = temp_violation_weight
        self.co2_violation_weight = co2_violation_weight
        self.rh_violation_weight = rh_violation_weight
        super().__init__(policy, env, **kwargs)

    @classmethod
    def load(
        cls,
        path: str | Path,
        env: GymEnv | None = None,
        device: th.device | str = "auto",
        custom_objects: dict[str, Any] | None = None,
        print_system_info: bool = False,
        force_reset: bool = True,
        **kwargs,
    ):
        if env is not None and (
            custom_objects is None or "observation_space" not in custom_objects
        ):
            data, _, _ = load_from_zip_file(
                path,
                device=device,
                custom_objects=custom_objects,
                print_system_info=False,
            )
            if data is not None and "raw_observation_space" in data:
                custom_objects = {
                    **(custom_objects or {}),
                    "observation_space": data["raw_observation_space"],
                }

        return super().load(
            path,
            env=env,
            device=device,
            custom_objects=custom_objects,
            print_system_info=print_system_info,
            force_reset=force_reset,
            **kwargs,
        )

    @staticmethod
    def _augmented_observation_space(
        observation_space: spaces.Space,
        context_dim: int,
    ) -> spaces.Box:
        if not isinstance(observation_space, spaces.Box) or len(observation_space.shape) != 1:
            raise TypeError("AgriMetaRL requires a flat Box observation space")
        if not np.issubdtype(observation_space.dtype, np.floating):
            raise TypeError("AgriMetaRL requires a floating-point Box observation space")

        context_low = np.full(context_dim, -np.inf, dtype=observation_space.dtype)
        context_high = np.full(context_dim, np.inf, dtype=observation_space.dtype)
        low = np.concatenate(
            [np.array(observation_space.low, copy=True).reshape(-1), context_low]
        ).astype(observation_space.dtype, copy=False)
        high = np.concatenate(
            [np.array(observation_space.high, copy=True).reshape(-1), context_high]
        ).astype(observation_space.dtype, copy=False)
        return spaces.Box(low=low, high=high, dtype=observation_space.dtype)

    def _setup_model(self) -> None:
        self._setup_lr_schedule()
        self.set_random_seed(self.seed)
        self.raw_observation_space = getattr(
            self,
            "raw_observation_space",
            self.observation_space,
        )
        self.observation_space = self._augmented_observation_space(
            self.raw_observation_space,
            self.context_dim,
        )

        buffer_cls = (
            AgriMetaRLRolloutBuffer
            if not isinstance(self.observation_space, spaces.Dict)
            else None  # Dict not extended here for brevity; use RecurrentDictRolloutBuffer + task_ids in subclass if needed
        )
        if buffer_cls is None:
            from sb3_contrib.common.recurrent.buffers import RecurrentDictRolloutBuffer
            buffer_cls = RecurrentDictRolloutBuffer

        from sb3_contrib.common.recurrent.type_aliases import RNNStates
        from sb3_contrib.common.recurrent.policies import RecurrentActorCriticPolicy

        self.policy = self.policy_class(
            self.observation_space,
            self.action_space,
            self.lr_schedule,
            use_sde=self.use_sde,
            **self.policy_kwargs,
        )
        self.policy = self.policy.to(self.device)

        if not isinstance(self.policy, RecurrentActorCriticPolicy):
            raise ValueError("Policy must subclass RecurrentActorCriticPolicy")

        lstm = self.policy.lstm_actor
        single_hidden_state_shape = (lstm.num_layers, self.n_envs, lstm.hidden_size)
        self._last_lstm_states = RNNStates(
            (
                th.zeros(single_hidden_state_shape, device=self.device),
                th.zeros(single_hidden_state_shape, device=self.device),
            ),
            (
                th.zeros(single_hidden_state_shape, device=self.device),
                th.zeros(single_hidden_state_shape, device=self.device),
            ),
        )

        hidden_state_buffer_shape = (self.n_steps, lstm.num_layers, self.n_envs, lstm.hidden_size)

        if isinstance(self.observation_space, spaces.Dict):
            from sb3_contrib.common.recurrent.buffers import RecurrentDictRolloutBuffer
            self.rollout_buffer = RecurrentDictRolloutBuffer(
                self.n_steps,
                self.observation_space,
                self.action_space,
                hidden_state_buffer_shape,
                self.device,
                gamma=self.gamma,
                gae_lambda=self.gae_lambda,
                n_envs=self.n_envs,
            )
        else:
            self.rollout_buffer = AgriMetaRLRolloutBuffer(
                self.n_steps,
                self.observation_space,
                self.action_space,
                hidden_state_buffer_shape,
                self.device,
                gamma=self.gamma,
                gae_lambda=self.gae_lambda,
                n_envs=self.n_envs,
            )

        obs_dim = self.raw_observation_space.shape[0]
        action_dim = int(np.prod(self.action_space.shape))
        self.support_memory = TaskSupportMemory(
            support_size=self.support_size,
            max_instances=self.max_task_instances,
        )
        self.calibration_memory = EpisodeCalibrationMemory(
            gamma=self.gamma,
            residual_alpha=self.residual_alpha,
            max_pending_episodes=self.max_pending_episodes,
            max_completed_episodes=self.max_completed_episodes,
            max_completed_query_samples=self.calibration_max_queue_samples,
            constraint_penalty_weight=self.constraint_penalty_weight,
        )
        self.context_encoder = TransitionSetEncoder(
            obs_dim=obs_dim,
            action_dim=action_dim,
            context_dim=self.context_dim,
            hidden_dim=self.transition_hidden_dim,
        ).to(self.device)
        self.residual_head = AdvantageResidualHead(
            obs_dim=obs_dim,
            context_dim=self.context_dim,
            alpha=self.residual_alpha,
        ).to(self.device)
        self.meta_optimizer = th.optim.Adam(
            list(self.context_encoder.parameters()) + list(self.residual_head.parameters()),
            lr=self.learning_rate,
        )
        self.policy.optimizer.add_param_group(
            {
                "params": list(self.context_encoder.parameters()),
                "lr": self.lr_schedule(1),
            }
        )
        self.meta_diagnostics = MetaDiagnostics(residual_alpha=self.residual_alpha)
        self._last_task_instance_keys = np.full(self.n_envs, None, dtype=object)
        self._inference_mode = None
        self._inference_support_memory = None
        self._inference_task_key = None
        self._inference_context_norms = None
        self._inference_support_ready_step = None
        self._inference_step = None

        from stable_baselines3.common.utils import get_schedule_fn
        self.clip_range = get_schedule_fn(self.clip_range)
        if self.clip_range_vf is not None:
            self.clip_range_vf = get_schedule_fn(self.clip_range_vf)

    def _get_torch_save_params(self):
        state_dicts, torch_variables = super()._get_torch_save_params()
        return [*state_dicts, "context_encoder", "residual_head"], torch_variables

    def begin_inference_episode(self, mode: str) -> None:
        if mode not in self.INFERENCE_MODES:
            raise ValueError(
                f"unknown inference mode {mode!r}; expected one of "
                f"{sorted(self.INFERENCE_MODES)}"
            )
        self._inference_mode = mode
        self._inference_support_memory = TaskSupportMemory(
            support_size=self.support_size,
            max_instances=1,
        )
        self._inference_task_key = None
        self._inference_context_norms = []
        self._inference_support_ready_step = None
        self._inference_step = 0

    def end_inference_episode(self) -> None:
        self._inference_mode = None
        self._inference_support_memory = None
        self._inference_task_key = None
        self._inference_context_norms = None
        self._inference_support_ready_step = None
        self._inference_step = None

    def observe_inference_transition(
        self,
        observation,
        action,
        reward: float,
        next_observation,
        done: bool,
        info: dict[str, Any],
    ) -> None:
        if self._inference_mode is None or self._inference_support_memory is None:
            raise RuntimeError("begin_inference_episode() must be called first")

        observation_array = np.asarray(observation)
        action_array = np.asarray(action)
        next_observation_array = np.asarray(next_observation)
        if (
            not np.isfinite(observation_array).all()
            or not np.isfinite(action_array).all()
            or not np.isfinite(next_observation_array).all()
            or not np.isfinite(reward)
        ):
            raise ValueError("inference transition must contain only finite values")
        if (
            "task_descriptor" not in info
            or info["task_descriptor"] is None
            or "task_instance_key" not in info
            or info["task_instance_key"] is None
        ):
            raise KeyError("inference info must contain complete task identity")

        task_key = str(info["task_instance_key"])
        if self._inference_task_key not in (None, task_key):
            raise ValueError("task identity changed inside one inference episode")
        self._inference_task_key = task_key
        transition = Transition(
            observation=np.asarray(observation_array, dtype=np.float32),
            action=np.asarray(action_array, dtype=np.float32).reshape(-1),
            reward=float(reward),
            next_observation=np.asarray(next_observation_array, dtype=np.float32),
            done=bool(done),
        )
        self._inference_support_memory.observe(task_key, transition)
        self._inference_step += 1
        if (
            self._inference_support_ready_step is None
            and len(self._inference_support_memory.support(task_key))
            >= self.support_size
        ):
            self._inference_support_ready_step = self._inference_step

    def _inference_context(self) -> np.ndarray:
        context = np.zeros(self.context_dim, dtype=np.float32)
        if self._inference_mode != "online_context" or self._inference_task_key is None:
            return context
        if self._inference_support_memory is None:
            raise RuntimeError("online inference support memory is not initialized")

        support = self._inference_support_memory.support(self._inference_task_key)
        encoded, ready = self._context_from_support(support)
        if not ready:
            return context
        context = encoded.detach().cpu().numpy().astype(np.float32, copy=False)
        if not np.isfinite(context).all():
            raise ValueError("inference context contains non-finite values")
        self._inference_context_norms.append(float(np.linalg.norm(context)))
        return context

    def inference_episode_diagnostics(self) -> dict[str, float]:
        norms = np.asarray(self._inference_context_norms or (), dtype=float)
        support_ready_step = (
            np.nan
            if self._inference_support_ready_step is None
            else self._inference_support_ready_step
        )
        return {
            "support_ready_step": float(support_ready_step),
            "context_norm_mean": float(norms.mean()) if norms.size else 0.0,
            "context_norm_max": float(norms.max()) if norms.size else 0.0,
        }

    def predict(
        self,
        observation,
        state=None,
        episode_start=None,
        deterministic: bool = False,
    ):
        observation_array = np.asarray(observation)
        raw_dim = int(self.raw_observation_space.shape[0])
        augmented_dim = raw_dim + self.context_dim

        if observation_array.shape == (raw_dim,):
            context = (
                np.zeros(self.context_dim, dtype=observation_array.dtype)
                if self._inference_mode is None
                else self._inference_context()
            )
            observation = np.concatenate([observation_array, context], axis=0)
        elif observation_array.shape == (augmented_dim,):
            observation = observation_array
        elif observation_array.ndim >= 2 and observation_array.shape[-1] == raw_dim:
            if (
                self._inference_mode == "online_context"
                and observation_array.shape[0] != 1
            ):
                raise ValueError(
                    "online inference requires a single evaluation environment"
                )
            if (
                self._inference_mode is not None
                and observation_array.shape == (1, raw_dim)
            ):
                context = self._inference_context().reshape(1, -1)
            else:
                context_shape = (*observation_array.shape[:-1], self.context_dim)
                context = np.zeros(context_shape, dtype=observation_array.dtype)
            observation = np.concatenate([observation_array, context], axis=-1)

        return self.policy.predict(
            observation,
            state=state,
            episode_start=episode_start,
            deterministic=deterministic,
        )

    def _observe_transitions(
        self,
        observations: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        next_observations: np.ndarray,
        dones: np.ndarray,
        infos: list[dict[str, Any]],
        values: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        task_instance_keys = np.empty(len(infos), dtype=object)
        query_mask = np.zeros(len(infos), dtype=bool)
        calibration_entry_ids = np.full(len(infos), -1, dtype=np.int64)
        for env_index, info in enumerate(infos):
            if "task_descriptor" not in info or "task_instance_key" not in info:
                raise KeyError("environment info must contain complete task identity")
            task_instance_keys[env_index] = info["task_instance_key"]
            next_observation = (
                info.get("terminal_observation", next_observations[env_index])
                if dones[env_index]
                else next_observations[env_index]
            )
            transition = Transition(
                observation=observations[env_index],
                action=np.asarray(actions[env_index]).reshape(-1),
                reward=rewards[env_index],
                next_observation=next_observation,
                done=dones[env_index],
            )
            query_mask[env_index] = self.support_memory.observe(
                task_instance_keys[env_index], transition
            )
            value_array = np.asarray(values[env_index]).reshape(-1)
            if value_array.size != 1:
                raise ValueError("each environment must provide exactly one value")
            calibration_entry_ids[env_index] = self.calibration_memory.observe(
                task_instance_key=task_instance_keys[env_index],
                transition=transition,
                value=float(value_array.item()),
                is_query=bool(query_mask[env_index]),
                constraint_penalty=self._constraint_penalty_from_info(info),
            )
        return task_instance_keys, query_mask, calibration_entry_ids

    def _constraint_penalty_from_info(self, info: dict[str, Any]) -> float:
        return float(
            getattr(self, "temp_violation_weight", 0.0)
            * float(info.get("temp_violation", 0.0))
            + getattr(self, "co2_violation_weight", 0.0)
            * float(info.get("co2_violation", 0.0))
            + getattr(self, "rh_violation_weight", 0.0)
            * float(info.get("rh_violation", 0.0))
        )

    def collect_rollouts(
        self,
        env: VecEnv,
        callback: BaseCallback,
        rollout_buffer,
        n_rollout_steps: int,
    ) -> bool:
        from stable_baselines3.common.buffers import RolloutBuffer
        assert isinstance(
            rollout_buffer, (AgriMetaRLRolloutBuffer, RecurrentRolloutBuffer)
        ), f"{rollout_buffer} doesn't support task_id"

        assert self._last_obs is not None
        self.policy.set_training_mode(False)

        n_steps = 0
        rollout_buffer.reset()
        self.support_memory.begin_rollout()
        if self.use_sde:
            self.policy.reset_noise(env.num_envs)

        callback.on_rollout_start()
        lstm_states = deepcopy(self._last_lstm_states)
        current_task_keys = np.asarray(
            getattr(
                self,
                "_last_task_instance_keys",
                np.full(env.num_envs, None, dtype=object),
            ),
            dtype=object,
        )

        while n_steps < n_rollout_steps:
            if self.use_sde and self.sde_sample_freq > 0 and n_steps % self.sde_sample_freq == 0:
                self.policy.reset_noise(env.num_envs)

            raw_observations = np.asarray(self._last_obs, dtype=np.float32)
            support_snapshots = self._support_snapshots_for_task_keys(current_task_keys)
            support_sizes = np.asarray(
                [len(snapshot) for snapshot in support_snapshots],
                dtype=np.int64,
            )
            augmented_obs = self._augment_raw_observations(
                raw_observations,
                support_snapshots,
            )
            with th.no_grad():
                obs_tensor = obs_as_tensor(augmented_obs, self.device)
                episode_starts = th.tensor(self._last_episode_starts, dtype=th.float32, device=self.device)
                actions, values, log_probs, lstm_states = self.policy(obs_tensor, lstm_states, episode_starts)

            actions = actions.cpu().numpy()
            clipped_actions = np.clip(actions, self.action_space.low, self.action_space.high) if isinstance(self.action_space, spaces.Box) else actions

            new_obs, rewards, dones, infos = env.step(clipped_actions)
            self.num_timesteps += env.num_envs
            callback.update_locals(locals())
            if not callback.on_step():
                return False

            self._update_info_buffer(infos, dones)
            n_steps += 1

            if isinstance(self.action_space, spaces.Discrete):
                actions = actions.reshape(-1, 1)

            for idx, done_ in enumerate(dones):
                if done_ and infos[idx].get("terminal_observation") is not None and infos[idx].get("TimeLimit.truncated", False):
                    terminal_augmented_obs = self._augment_raw_observations(
                        np.asarray([infos[idx]["terminal_observation"]], dtype=np.float32),
                        (support_snapshots[idx],),
                    )
                    terminal_obs = obs_as_tensor(terminal_augmented_obs, self.device)
                    with th.no_grad():
                        terminal_lstm_state = (
                            lstm_states.vf[0][:, idx : idx + 1, :].contiguous(),
                            lstm_states.vf[1][:, idx : idx + 1, :].contiguous(),
                        )
                        episode_starts_t = th.tensor([False], dtype=th.float32, device=self.device)
                        terminal_value = self.policy.predict_values(terminal_obs, terminal_lstm_state, episode_starts_t)[0]
                    rewards[idx] += self.gamma * terminal_value

            task_instance_keys, query_mask, calibration_entry_ids = (
                self._observe_transitions(
                    observations=raw_observations,
                    actions=clipped_actions,
                    rewards=rewards,
                    next_observations=new_obs,
                    dones=dones,
                    infos=infos,
                    values=values.detach().cpu().numpy(),
                )
            )

            if isinstance(rollout_buffer, AgriMetaRLRolloutBuffer):
                rollout_buffer.add(
                    augmented_obs,
                    actions,
                    rewards,
                    self._last_episode_starts,
                    values,
                    log_probs,
                    lstm_states=deepcopy(self._last_lstm_states),
                    task_instance_keys=task_instance_keys,
                    query_mask=query_mask,
                    calibration_entry_ids=calibration_entry_ids,
                    support_snapshots=support_snapshots,
                    support_sizes=support_sizes,
                )
            else:
                rollout_buffer.add(
                    augmented_obs,
                    actions,
                    rewards,
                    self._last_episode_starts,
                    values,
                    log_probs,
                    lstm_states=deepcopy(self._last_lstm_states),
                )

            next_task_keys = task_instance_keys.copy()
            next_task_keys[np.asarray(dones, dtype=bool)] = None
            current_task_keys = next_task_keys
            self._last_obs = new_obs
            self._last_episode_starts = dones
            self._last_lstm_states = lstm_states

        with th.no_grad():
            final_support_snapshots = self._support_snapshots_for_task_keys(
                current_task_keys
            )
            final_augmented_obs = self._augment_raw_observations(
                np.asarray(self._last_obs, dtype=np.float32),
                final_support_snapshots,
            )
            episode_starts_t = th.tensor(self._last_episode_starts, dtype=th.float32, device=self.device)
            last_values = self.policy.predict_values(
                obs_as_tensor(final_augmented_obs, self.device),
                self._last_lstm_states.vf,
                episode_starts_t,
            )
        self._last_task_instance_keys = current_task_keys
        rollout_buffer.compute_returns_and_advantage(last_values=last_values, dones=self._last_episode_starts)
        if isinstance(rollout_buffer, AgriMetaRLRolloutBuffer):
            self._attach_rollout_calibration()

        callback.on_rollout_end()
        return True

    def _attach_rollout_calibration(self) -> None:
        buffer = self.rollout_buffer
        self.calibration_memory.attach_rollout(
            buffer.calibration_entry_ids.reshape(-1),
            buffer.advantages.reshape(-1),
        )
        for task_key in self.calibration_memory.ready_task_keys():
            support = self.support_memory.support(task_key)
            if len(support) != self.support_size:
                raise RuntimeError(
                    f"missing frozen support for completed task: {task_key}"
                )
            episode = self.calibration_memory.finalize(task_key, support)
            self.meta_diagnostics.completed_episode_count += 1
            self.meta_diagnostics.mc_gae_abs_differences.append(
                episode.mc_gae_abs_difference_mean
            )
            self.meta_diagnostics.target_clip_fractions.append(
                episode.target_clip_fraction
            )

    def _tensorize_support(
        self, support: tuple[Transition, ...]
    ) -> dict[str, th.Tensor]:
        return {
            "observations": th.as_tensor(
                np.stack([item.observation for item in support]),
                device=self.device,
                dtype=th.float32,
            ),
            "actions": th.as_tensor(
                np.stack([item.action for item in support]),
                device=self.device,
                dtype=th.float32,
            ),
            "rewards": th.as_tensor(
                [item.reward for item in support], device=self.device, dtype=th.float32
            ),
            "next_observations": th.as_tensor(
                np.stack([item.next_observation for item in support]),
                device=self.device,
                dtype=th.float32,
            ),
            "dones": th.as_tensor(
                [item.done for item in support], device=self.device, dtype=th.bool
            ),
        }

    def _context_from_support(
        self, support: tuple[Transition, ...]
    ) -> tuple[th.Tensor, bool]:
        if len(support) < self.support_size:
            return th.zeros(self.context_dim, device=self.device), False
        context = self.context_encoder(**self._tensorize_support(support))
        return context, True

    def _support_snapshots_for_task_keys(self, task_instance_keys) -> tuple[tuple, ...]:
        snapshots = []
        for key in task_instance_keys:
            if key is None:
                snapshots.append(())
            else:
                snapshots.append(self.support_memory.support(str(key)))
        return tuple(snapshots)

    def _augment_raw_observations(
        self,
        raw_observations: np.ndarray,
        support_snapshots,
    ) -> np.ndarray:
        raw_observations = np.asarray(raw_observations, dtype=np.float32)
        contexts = []
        with th.no_grad():
            for support in support_snapshots:
                context, _ = self._context_from_support(tuple(support))
                contexts.append(context.detach().cpu().numpy().astype(np.float32))
        context_array = np.asarray(contexts, dtype=np.float32).reshape(
            raw_observations.shape[0],
            self.context_dim,
        )
        return np.concatenate([raw_observations, context_array], axis=1).astype(
            np.float32,
            copy=False,
        )

    def _augment_training_observations(
        self,
        augmented_or_raw_observations: th.Tensor,
        support_snapshots,
    ) -> th.Tensor:
        observations = augmented_or_raw_observations.to(self.device)
        raw_dim = int(self.raw_observation_space.shape[0])
        raw_observations = observations[..., :raw_dim].float()
        contexts = []
        for support in support_snapshots:
            support = tuple(support)
            if len(support) < self.support_size:
                contexts.append(
                    th.zeros(
                        self.context_dim,
                        device=self.device,
                        dtype=raw_observations.dtype,
                    )
                )
            else:
                contexts.append(
                    self.context_encoder(**self._tensorize_support(support)).to(
                        dtype=raw_observations.dtype
                    )
                )
        context_tensor = th.stack(contexts, dim=0).reshape(
            raw_observations.shape[0],
            self.context_dim,
        )
        return th.cat([raw_observations, context_tensor], dim=-1)

    def _train_calibration_batch(self, episodes) -> float | None:
        if not episodes:
            return None
        prediction_groups = []
        target_groups = []
        for episode in episodes:
            if not episode.samples:
                continue
            context = self.context_encoder(**self._tensorize_support(episode.support))
            observations = th.as_tensor(
                np.stack([sample.observation for sample in episode.samples]),
                device=self.device,
                dtype=th.float32,
            )
            raw_advantages = th.as_tensor(
                [sample.raw_advantage for sample in episode.samples],
                device=self.device,
                dtype=th.float32,
            )
            targets = th.as_tensor(
                [sample.target_residual for sample in episode.samples],
                device=self.device,
                dtype=th.float32,
            )
            _, predictions = self.residual_head(
                observations, raw_advantages, context
            )
            prediction_groups.append(predictions)
            target_groups.append(targets)
        if not prediction_groups:
            return None
        predictions = th.cat(prediction_groups)
        targets = th.cat(target_groups)
        if not th.isfinite(predictions).all() or not th.isfinite(targets).all():
            self.meta_diagnostics.nonfinite_meta_batch_count += 1
            return None
        loss = self.meta_loss_weight * (
            th.nn.functional.smooth_l1_loss(predictions, targets)
            + self.residual_regularization * predictions.square().mean()
        )
        if not th.isfinite(loss):
            self.meta_diagnostics.nonfinite_meta_batch_count += 1
            return None
        self.meta_optimizer.zero_grad()
        loss.backward()
        th.nn.utils.clip_grad_norm_(
            list(self.context_encoder.parameters())
            + list(self.residual_head.parameters()),
            self.max_grad_norm,
        )
        self.meta_optimizer.step()
        return float(loss.detach().cpu())

    def _apply_meta_advantage_correction(self) -> None:
        if not isinstance(self.rollout_buffer, AgriMetaRLRolloutBuffer):
            return
        buffer = self.rollout_buffer
        self.meta_diagnostics.transition_count += buffer.buffer_size * buffer.n_envs
        keys = {
            key
            for key in buffer.task_instance_keys[buffer.query_mask]
            if key is not None
        }
        for key in keys:
            support = self.support_memory.support(key)
            if len(support) < self.support_size:
                continue
            rows, envs = np.where(
                buffer.query_mask & (buffer.task_instance_keys == key)
            )
            if rows.size == 0:
                continue
            context = self.context_encoder(**self._tensorize_support(support))
            raw_dim = int(self.raw_observation_space.shape[0])
            observations = th.as_tensor(
                buffer.observations[rows, envs][..., :raw_dim],
                device=self.device,
                dtype=th.float32,
            )
            raw_advantages = th.as_tensor(
                buffer.advantages[rows, envs], device=self.device, dtype=th.float32
            )
            corrected, residual = self.residual_head(
                observations, raw_advantages, context
            )
            buffer.advantages[rows, envs] = corrected.detach().cpu().numpy()

            self.meta_diagnostics.record_group(
                context.detach().cpu().numpy(), residual.detach().cpu().numpy()
            )

    def _train_completed_calibration(self) -> float | None:
        episodes = self.calibration_memory.pop_completed(
            self.calibration_min_query_samples,
            self.calibration_max_query_samples,
        )
        if not episodes:
            return None
        loss = self._train_calibration_batch(episodes)
        if loss is not None:
            self.meta_diagnostics.meta_losses.append(loss)
        return loss

    def train(self) -> None:
        self.policy.set_training_mode(True)
        self._update_learning_rate(self.policy.optimizer)
        self._apply_meta_advantage_correction()
        self._train_completed_calibration()
        self.meta_diagnostics.calibration_queue_size = (
            self.calibration_memory.completed_query_sample_count
        )
        summary = self.meta_diagnostics.summarize()
        for name, value in summary.items():
            self.logger.record(name, value)
        self.meta_diagnostics.last_summary = dict(summary)
        self.meta_diagnostics.reset()

        clip_range = self.clip_range(self._current_progress_remaining)
        if self.clip_range_vf is not None:
            clip_range_vf = self.clip_range_vf(self._current_progress_remaining)

        entropy_losses = []
        pg_losses, value_losses = [], []
        clip_fractions = []

        continue_training = True

        for epoch in range(self.n_epochs):
            approx_kl_divs = []
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                actions = rollout_data.actions
                if isinstance(self.action_space, spaces.Discrete):
                    actions = rollout_data.actions.long().flatten()

                mask = rollout_data.mask > 1e-8

                if self.use_sde:
                    self.policy.reset_noise(self.batch_size)

                observations = rollout_data.observations
                if hasattr(rollout_data, "support_snapshots"):
                    observations = self._augment_training_observations(
                        rollout_data.observations,
                        rollout_data.support_snapshots,
                    )

                values, log_prob, entropy = self.policy.evaluate_actions(
                    observations,
                    actions,
                    rollout_data.lstm_states,
                    rollout_data.episode_starts,
                )

                values = values.flatten()
                advantages = rollout_data.advantages
                if self.normalize_advantage:
                    valid_advantages = advantages[mask]
                    if valid_advantages.numel() > 1:
                        advantage_mean = valid_advantages.mean()
                        advantage_std = valid_advantages.std(unbiased=False)
                        if th.isfinite(advantage_mean) and th.isfinite(advantage_std):
                            advantages = (advantages - advantage_mean) / (
                                advantage_std + 1e-8
                            )

                ratio = th.exp(log_prob - rollout_data.old_log_prob)

                policy_loss_1 = advantages * ratio
                policy_loss_2 = advantages * th.clamp(
                    ratio,
                    1 - clip_range,
                    1 + clip_range,
                )
                policy_loss = -th.mean(th.min(policy_loss_1, policy_loss_2)[mask])

                pg_losses.append(policy_loss.item())
                clip_fraction = th.mean(
                    (th.abs(ratio - 1) > clip_range).float()[mask]
                ).item()
                clip_fractions.append(clip_fraction)

                if self.clip_range_vf is None:
                    values_pred = values
                else:
                    values_pred = rollout_data.old_values + th.clamp(
                        values - rollout_data.old_values,
                        -clip_range_vf,
                        clip_range_vf,
                    )
                value_loss = th.mean(((rollout_data.returns - values_pred) ** 2)[mask])
                value_losses.append(value_loss.item())

                if entropy is None:
                    entropy_loss = -th.mean(-log_prob[mask])
                else:
                    entropy_loss = -th.mean(entropy[mask])
                entropy_losses.append(entropy_loss.item())

                loss = (
                    policy_loss
                    + self.ent_coef * entropy_loss
                    + self.vf_coef * value_loss
                )

                with th.no_grad():
                    log_ratio = log_prob - rollout_data.old_log_prob
                    approx_kl_div = th.mean(
                        ((th.exp(log_ratio) - 1) - log_ratio)[mask]
                    ).cpu().numpy()
                    approx_kl_divs.append(approx_kl_div)

                if self.target_kl is not None and approx_kl_div > 1.5 * self.target_kl:
                    continue_training = False
                    if self.verbose >= 1:
                        print(
                            f"Early stopping at step {epoch} due to reaching max kl: "
                            f"{approx_kl_div:.2f}"
                        )
                    break

                self.policy.optimizer.zero_grad()
                loss.backward()
                th.nn.utils.clip_grad_norm_(
                    list(self.policy.parameters())
                    + list(self.context_encoder.parameters()),
                    self.max_grad_norm,
                )
                self.policy.optimizer.step()

            if not continue_training:
                break

        self._n_updates += self.n_epochs
        explained_var = explained_variance(
            self.rollout_buffer.values.flatten(),
            self.rollout_buffer.returns.flatten(),
        )

        self.logger.record("train/entropy_loss", np.mean(entropy_losses))
        self.logger.record("train/policy_gradient_loss", np.mean(pg_losses))
        self.logger.record("train/value_loss", np.mean(value_losses))
        self.logger.record("train/approx_kl", np.mean(approx_kl_divs))
        self.logger.record("train/clip_fraction", np.mean(clip_fractions))
        self.logger.record("train/loss", loss.item())
        self.logger.record("train/explained_variance", explained_var)
        if hasattr(self.policy, "log_std"):
            self.logger.record("train/std", th.exp(self.policy.log_std).mean().item())

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/clip_range", clip_range)
        if self.clip_range_vf is not None:
            self.logger.record("train/clip_range_vf", clip_range_vf)
