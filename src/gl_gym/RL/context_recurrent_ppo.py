"""Context-augmented recurrent PPO setup helpers."""

from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import torch as th
from gymnasium import spaces
from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.save_util import load_from_zip_file
from stable_baselines3.common.type_aliases import GymEnv
from stable_baselines3.common.utils import (
    explained_variance,
    get_schedule_fn,
    obs_as_tensor,
)
from stable_baselines3.common.vec_env import VecEnv
from sb3_contrib.common.recurrent.buffers import (
    RecurrentDictRolloutBuffer,
    RecurrentRolloutBuffer,
)
from sb3_contrib.common.recurrent.policies import RecurrentActorCriticPolicy
from sb3_contrib.common.recurrent.type_aliases import RNNStates
from sb3_contrib.ppo_recurrent import RecurrentPPO

from gl_gym.RL.agri_metarl.memory import TaskSupportMemory, Transition
from gl_gym.RL.agri_metarl.meta_advantage_head import TransitionSetEncoder
from gl_gym.RL.context_recurrent_ppo_buffer import ContextRecurrentRolloutBuffer
from gl_gym.RL.context_recurrent_ppo_diagnostics import ContextDiagnostics


class ContextRecurrentPPO(RecurrentPPO):
    """Recurrent PPO whose policy input is augmented with encoded task context."""

    def __init__(
        self,
        policy: str | type[RecurrentActorCriticPolicy],
        env: GymEnv | str,
        support_size: int = 256,
        max_task_instances: int = 128,
        context_dim: int = 64,
        transition_hidden_dim: int = 128,
        **kwargs,
    ) -> None:
        self.support_size = support_size
        self.max_task_instances = max_task_instances
        self.context_dim = context_dim
        self.transition_hidden_dim = transition_hidden_dim
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
        """Return a flat Box space with context coordinates appended."""

        if not isinstance(observation_space, spaces.Box) or len(observation_space.shape) != 1:
            raise TypeError("ContextRecurrentPPO requires a flat Box observation space")
        if not np.issubdtype(observation_space.dtype, np.floating):
            raise TypeError(
                "ContextRecurrentPPO requires a floating-point Box observation space"
            )

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

        hidden_state_buffer_shape = (
            self.n_steps,
            lstm.num_layers,
            self.n_envs,
            lstm.hidden_size,
        )
        self.rollout_buffer = ContextRecurrentRolloutBuffer(
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
        action_shape = getattr(self.action_space, "shape", ())
        self.action_dim = int(np.prod(action_shape)) if action_shape else 1
        self.support_memory = TaskSupportMemory(
            support_size=self.support_size,
            max_instances=self.max_task_instances,
        )
        self.context_encoder = TransitionSetEncoder(
            obs_dim=obs_dim,
            action_dim=self.action_dim,
            context_dim=self.context_dim,
            hidden_dim=self.transition_hidden_dim,
        ).to(self.device)
        self.context_diagnostics = ContextDiagnostics()
        self._last_task_instance_keys = np.full(self.n_envs, None, dtype=object)

        self.policy.optimizer.add_param_group(
            {
                "params": list(self.context_encoder.parameters()),
                "lr": self.lr_schedule(1),
            }
        )

        self.clip_range = get_schedule_fn(self.clip_range)
        if self.clip_range_vf is not None:
            if isinstance(self.clip_range_vf, (float, int)):
                assert self.clip_range_vf > 0, (
                    "`clip_range_vf` must be positive, pass `None` to deactivate vf clipping"
                )
            self.clip_range_vf = get_schedule_fn(self.clip_range_vf)

    def _get_torch_save_params(self):
        state_dicts, torch_variables = super()._get_torch_save_params()
        return [*state_dicts, "context_encoder"], torch_variables

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
            context = np.zeros(self.context_dim, dtype=observation_array.dtype)
            observation = np.concatenate([observation_array, context], axis=0)
        elif observation_array.shape == (augmented_dim,):
            observation = observation_array
        elif observation_array.ndim >= 2 and observation_array.shape[-1] == raw_dim:
            context_shape = (*observation_array.shape[:-1], self.context_dim)
            context = np.zeros(context_shape, dtype=observation_array.dtype)
            observation = np.concatenate([observation_array, context], axis=-1)

        return self.policy.predict(
            observation,
            state=state,
            episode_start=episode_start,
            deterministic=deterministic,
        )

    def _tensorize_support(
        self, support: tuple[Transition, ...]
    ) -> dict[str, th.Tensor]:
        action_dim = getattr(self, "action_dim", None)
        if action_dim is None:
            action_dim = int(np.asarray(support[0].action).reshape(-1).shape[0])
        actions = np.stack(
            [np.asarray(item.action).reshape(-1) for item in support]
        ).reshape(len(support), int(action_dim))
        return {
            "observations": th.as_tensor(
                np.stack([item.observation for item in support]),
                device=self.device,
                dtype=th.float32,
            ),
            "actions": th.as_tensor(
                actions,
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

    def _augment_raw_observations(
        self,
        raw_observations: np.ndarray,
        support_snapshots,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        raw_observations = np.asarray(raw_observations, dtype=np.float32)
        contexts: list[np.ndarray] = []
        active_mask: list[bool] = []

        with th.no_grad():
            for support in support_snapshots:
                context, active = self._context_from_support(tuple(support))
                contexts.append(context.detach().cpu().numpy().astype(np.float32))
                active_mask.append(active)

        context_array = np.asarray(contexts, dtype=np.float32).reshape(
            raw_observations.shape[0], self.context_dim
        )
        augmented = np.concatenate([raw_observations, context_array], axis=1).astype(
            np.float32,
            copy=False,
        )
        return augmented, context_array, np.asarray(active_mask, dtype=bool)

    def _raw_observation_dim(self, observations) -> int:
        raw_observation_space = getattr(self, "raw_observation_space", None)
        if raw_observation_space is not None:
            return int(raw_observation_space.shape[0])
        return int(observations.shape[-1] - self.context_dim)

    def _augment_training_observations(
        self,
        augmented_or_raw_observations: th.Tensor,
        support_snapshots,
    ) -> th.Tensor:
        observations = augmented_or_raw_observations.to(self.device)
        raw_dim = self._raw_observation_dim(observations)
        raw_observations = observations[..., :raw_dim].float()

        contexts: list[th.Tensor] = []
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

    def _support_snapshots_for_task_keys(self, task_instance_keys) -> tuple[tuple, ...]:
        snapshots = []
        for key in task_instance_keys:
            if key is None:
                snapshots.append(())
            else:
                snapshots.append(self.support_memory.support(str(key)))
        return tuple(snapshots)

    def _observe_raw_transitions(
        self,
        observations: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        next_observations: np.ndarray,
        dones: np.ndarray,
        infos: list[dict],
    ) -> dict[str, np.ndarray | tuple[tuple, ...]]:
        task_instance_keys: list[str] = []
        support_snapshots: list[tuple[Transition, ...]] = []
        context_active_mask: list[bool] = []
        support_sizes: list[int] = []

        for env_index, info in enumerate(infos):
            if "task_descriptor" not in info or "task_instance_key" not in info:
                raise KeyError(
                    "rollout info must include task_descriptor and task_instance_key"
                )
            task_key = str(info["task_instance_key"])
            snapshot = self.support_memory.support(task_key)
            support_snapshots.append(snapshot)
            support_sizes.append(len(snapshot))
            context_active_mask.append(len(snapshot) >= self.support_size)
            task_instance_keys.append(task_key)

            transition_next_observation = next_observations[env_index]
            if dones[env_index] and info.get("terminal_observation") is not None:
                transition_next_observation = info["terminal_observation"]
            self.support_memory.observe(
                task_key,
                Transition(
                    observation=observations[env_index],
                    action=actions[env_index],
                    reward=float(rewards[env_index]),
                    next_observation=transition_next_observation,
                    done=bool(dones[env_index]),
                ),
            )

        return {
            "task_instance_keys": np.asarray(task_instance_keys, dtype=object),
            "context_active_mask": np.asarray(context_active_mask, dtype=bool),
            "support_snapshots": tuple(support_snapshots),
            "support_sizes": np.asarray(support_sizes, dtype=np.int64),
        }

    def collect_rollouts(
        self,
        env: VecEnv,
        callback: BaseCallback,
        rollout_buffer: RolloutBuffer,
        n_rollout_steps: int,
    ) -> bool:
        assert isinstance(
            rollout_buffer, (RecurrentRolloutBuffer, RecurrentDictRolloutBuffer)
        ), f"{rollout_buffer} doesn't support recurrent policy"
        assert self._last_obs is not None, "No previous observation was provided"

        self.policy.set_training_mode(False)

        n_steps = 0
        rollout_buffer.reset()
        self.support_memory.begin_rollout()
        current_task_keys = np.asarray(
            getattr(
                self,
                "_last_task_instance_keys",
                np.full(env.num_envs, None, dtype=object),
            ),
            dtype=object,
        )

        if self.use_sde:
            self.policy.reset_noise(env.num_envs)

        callback.on_rollout_start()

        lstm_states = deepcopy(self._last_lstm_states)

        while n_steps < n_rollout_steps:
            if (
                self.use_sde
                and self.sde_sample_freq > 0
                and n_steps % self.sde_sample_freq == 0
            ):
                self.policy.reset_noise(env.num_envs)

            support_snapshots = self._support_snapshots_for_task_keys(
                current_task_keys
            )
            support_sizes = np.asarray(
                [len(snapshot) for snapshot in support_snapshots], dtype=np.int64
            )
            augmented_obs, contexts, active_mask = self._augment_raw_observations(
                self._last_obs,
                support_snapshots,
            )
            self.context_diagnostics.record_contexts(
                contexts=contexts,
                active_mask=active_mask,
                support_sizes=support_sizes,
                task_instance_keys=current_task_keys,
            )

            with th.no_grad():
                obs_tensor = obs_as_tensor(augmented_obs, self.device)
                episode_starts = th.tensor(
                    self._last_episode_starts, dtype=th.float32, device=self.device
                )
                actions, values, log_probs, lstm_states = self.policy.forward(
                    obs_tensor, lstm_states, episode_starts
                )

            actions = actions.cpu().numpy()
            clipped_actions = actions
            if isinstance(self.action_space, spaces.Box):
                clipped_actions = np.clip(
                    actions, self.action_space.low, self.action_space.high
                )

            raw_observations = np.asarray(self._last_obs, dtype=np.float32)
            new_obs, rewards, dones, infos = env.step(clipped_actions)

            self.num_timesteps += env.num_envs

            callback.update_locals(locals())
            if callback.on_step() is False:
                return False

            self._update_info_buffer(infos)
            n_steps += 1

            transition_metadata = self._observe_raw_transitions(
                observations=raw_observations,
                actions=clipped_actions,
                rewards=rewards,
                next_observations=new_obs,
                dones=dones,
                infos=infos,
            )

            if isinstance(self.action_space, spaces.Discrete):
                actions = actions.reshape(-1, 1)

            for idx, done_ in enumerate(dones):
                if (
                    done_
                    and infos[idx].get("terminal_observation") is not None
                    and infos[idx].get("TimeLimit.truncated", False)
                ):
                    terminal_key = transition_metadata["task_instance_keys"][idx]
                    terminal_support = self.support_memory.support(str(terminal_key))
                    terminal_augmented_obs, _, _ = self._augment_raw_observations(
                        np.asarray(
                            [infos[idx]["terminal_observation"]], dtype=np.float32
                        ),
                        (terminal_support,),
                    )
                    terminal_obs = obs_as_tensor(terminal_augmented_obs, self.device)
                    with th.no_grad():
                        terminal_lstm_state = (
                            lstm_states.vf[0][:, idx : idx + 1, :].contiguous(),
                            lstm_states.vf[1][:, idx : idx + 1, :].contiguous(),
                        )
                        episode_starts = th.tensor(
                            [False], dtype=th.float32, device=self.device
                        )
                        terminal_value = self.policy.predict_values(
                            terminal_obs, terminal_lstm_state, episode_starts
                        )[0]
                    rewards[idx] += self.gamma * terminal_value

            rollout_buffer.add(
                augmented_obs,
                actions,
                rewards,
                self._last_episode_starts,
                values,
                log_probs,
                lstm_states=self._last_lstm_states,
                task_instance_keys=transition_metadata["task_instance_keys"],
                context_active_mask=active_mask,
                support_snapshots=support_snapshots,
                support_sizes=support_sizes,
            )

            next_task_keys = transition_metadata["task_instance_keys"].copy()
            next_task_keys[np.asarray(dones, dtype=bool)] = None
            current_task_keys = next_task_keys
            self._last_obs = new_obs
            self._last_episode_starts = dones
            self._last_lstm_states = lstm_states

        with th.no_grad():
            final_support_snapshots = self._support_snapshots_for_task_keys(
                current_task_keys
            )
            final_augmented_obs, _, _ = self._augment_raw_observations(
                new_obs,
                final_support_snapshots,
            )
            episode_starts = th.tensor(dones, dtype=th.float32, device=self.device)
            values = self.policy.predict_values(
                obs_as_tensor(final_augmented_obs, self.device),
                lstm_states.vf,
                episode_starts,
            )

        self._last_task_instance_keys = current_task_keys
        rollout_buffer.compute_returns_and_advantage(last_values=values, dones=dones)

        callback.on_rollout_end()

        return True

    def train(self) -> None:
        """
        Update policy using rollout data and recomputed train-time context.
        """
        self.policy.set_training_mode(True)
        self._update_learning_rate(self.policy.optimizer)
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
                    ratio, 1 - clip_range, 1 + clip_range
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

        if hasattr(self, "context_diagnostics"):
            for key, value in self.context_diagnostics.summarize().items():
                self.logger.record(key, value)
            self.context_diagnostics.reset()
