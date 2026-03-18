"""
Agri-MetaRL: Recurrent PPO with support/query MetaAdvantageHead for task-adaptive advantage correction.
"""
from copy import deepcopy
from typing import Any, Optional

import numpy as np
import torch as th
from gymnasium import spaces
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.type_aliases import GymEnv, Schedule
from stable_baselines3.common.utils import obs_as_tensor
from stable_baselines3.common.vec_env import VecEnv

from sb3_contrib.common.recurrent.buffers import RecurrentDictRolloutBuffer, RecurrentRolloutBuffer
from sb3_contrib.common.recurrent.policies import RecurrentActorCriticPolicy
from sb3_contrib.ppo_recurrent import RecurrentPPO

from gl_gym.RL.agri_metarl.buffer import AgriMetaRLRolloutBuffer, encode_task_id
from gl_gym.RL.agri_metarl.meta_advantage_head import MetaAdvantageHead


class AgriMetaRL(RecurrentPPO):
    """
    Recurrent PPO with MetaAdvantageHead: support set encodes task context,
    query set uses corrected advantages A_tilde = A + delta(obs, A, c_tau).
    """

    def __init__(
        self,
        policy: str | type[RecurrentActorCriticPolicy],
        env: GymEnv | str,
        meta_support_ratio: float = 0.5,
        meta_min_support_steps: int = 256,
        meta_beta: float = 0.05,
        meta_advantage_clip: float = 2.0,
        meta_context_dim: int = 64,
        meta_use_batch_norm: bool = True,
        meta_use_output_clip: bool = True,
        meta_use_obs_in_correction: bool = True,
        meta_use_advantage_in_correction: bool = True,
        **kwargs,
    ):
        self.meta_support_ratio = meta_support_ratio
        self.meta_min_support_steps = meta_min_support_steps
        self.meta_beta = meta_beta
        self.meta_advantage_clip = meta_advantage_clip
        self.meta_context_dim = meta_context_dim
        self.meta_use_batch_norm = meta_use_batch_norm
        self.meta_use_output_clip = meta_use_output_clip
        self.meta_use_obs_in_correction = meta_use_obs_in_correction
        self.meta_use_advantage_in_correction = meta_use_advantage_in_correction
        super().__init__(policy, env, **kwargs)

    def _setup_model(self) -> None:
        self._setup_lr_schedule()
        self.set_random_seed(self.seed)

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

        obs_dim = self.observation_space.shape[0]
        self.meta_head = MetaAdvantageHead(
            obs_dim=obs_dim,
            context_dim=self.meta_context_dim,
            use_batch_norm=self.meta_use_batch_norm,
            use_output_clip=self.meta_use_output_clip,
            use_obs_in_correction=self.meta_use_obs_in_correction,
            use_advantage_in_correction=self.meta_use_advantage_in_correction,
        ).to(self.device)
        self.meta_optimizer = th.optim.Adam(self.meta_head.parameters(), lr=self.learning_rate)

        from stable_baselines3.common.utils import get_schedule_fn
        self.clip_range = get_schedule_fn(self.clip_range)
        if self.clip_range_vf is not None:
            self.clip_range_vf = get_schedule_fn(self.clip_range_vf)

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
        if self.use_sde:
            self.policy.reset_noise(env.num_envs)

        callback.on_rollout_start()
        lstm_states = deepcopy(self._last_lstm_states)

        while n_steps < n_rollout_steps:
            if self.use_sde and self.sde_sample_freq > 0 and n_steps % self.sde_sample_freq == 0:
                self.policy.reset_noise(env.num_envs)

            with th.no_grad():
                obs_tensor = obs_as_tensor(self._last_obs, self.device)
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
                    terminal_obs = self.policy.obs_to_tensor(infos[idx]["terminal_observation"])[0]
                    with th.no_grad():
                        terminal_lstm_state = (
                            lstm_states.vf[0][:, idx : idx + 1, :].contiguous(),
                            lstm_states.vf[1][:, idx : idx + 1, :].contiguous(),
                        )
                        episode_starts_t = th.tensor([False], dtype=th.float32, device=self.device)
                        terminal_value = self.policy.predict_values(terminal_obs, terminal_lstm_state, episode_starts_t)[0]
                    rewards[idx] += self.gamma * terminal_value

            task_ids = np.array([infos[i].get("task_id", (0, 0)) for i in range(env.num_envs)])
            if isinstance(rollout_buffer, AgriMetaRLRolloutBuffer):
                rollout_buffer.add(
                    self._last_obs,
                    actions,
                    rewards,
                    self._last_episode_starts,
                    values,
                    log_probs,
                    lstm_states=deepcopy(self._last_lstm_states),
                    task_ids=task_ids,
                )
            else:
                rollout_buffer.add(
                    self._last_obs,
                    actions,
                    rewards,
                    self._last_episode_starts,
                    values,
                    log_probs,
                    lstm_states=deepcopy(self._last_lstm_states),
                )

            self._last_obs = new_obs
            self._last_episode_starts = dones
            self._last_lstm_states = lstm_states

        with th.no_grad():
            episode_starts_t = th.tensor(self._last_episode_starts, dtype=th.float32, device=self.device)
            last_values = self.policy.predict_values(
                obs_as_tensor(self._last_obs, self.device),
                self._last_lstm_states.vf,
                episode_starts_t,
            )
        rollout_buffer.compute_returns_and_advantage(last_values=last_values, dones=self._last_episode_starts)

        callback.on_rollout_end()
        return True

    def _apply_meta_advantage_correction(self) -> None:
        """Compute support/query split per task, encode context, correct advantages on query set, update buffer and meta head."""
        if not isinstance(self.rollout_buffer, AgriMetaRLRolloutBuffer):
            return
        buf = self.rollout_buffer
        n_steps, n_envs = buf.buffer_size, buf.n_envs
        obs = buf.observations  # (n_steps, n_envs, obs_dim)
        rewards = buf.rewards
        returns = buf.returns
        advantages = buf.advantages
        episode_starts = buf.episode_starts
        task_ids = buf.task_ids
        state_dim = self.meta_head.state_dim_for_context

        meta_losses = []
        for j in range(n_envs):
            starts = np.where(episode_starts[:, j] > 0.5)[0]
            starts = list(starts) + [n_steps]
            for idx in range(len(starts) - 1):
                i0, i1 = starts[idx], starts[idx + 1]
                ep_len = i1 - i0
                if ep_len < self.meta_min_support_steps:
                    continue
                n_sup = max(self.meta_min_support_steps, int(ep_len * self.meta_support_ratio))
                n_sup = min(n_sup, ep_len - 1)
                sup_idx = np.arange(i0, i0 + n_sup)
                qry_idx = np.arange(i0 + n_sup, i1)
                if len(qry_idx) == 0:
                    continue

                obs_sup = obs[sup_idx, j, :]
                obs_qry = obs[qry_idx, j, :]
                adv_sup = advantages[sup_idx, j]
                adv_qry = advantages[qry_idx, j]
                rew_sup = rewards[sup_idx, j]
                ret_sup = returns[sup_idx, j]
                ret_qry = returns[qry_idx, j]
                val_qry = buf.values[qry_idx, j]

                state_mean_sup = np.mean(obs_sup[:, :state_dim], axis=0).astype(np.float32)
                state_std_sup = np.std(obs_sup[:, :state_dim], axis=0).astype(np.float32)
                if state_std_sup.max() < 1e-8:
                    state_std_sup = np.ones_like(state_std_sup, dtype=np.float32) * 1e-8
                adv_mean_sup = float(np.mean(adv_sup))
                adv_std_sup = float(np.std(adv_sup)) or 1e-8
                rew_mean_sup = float(np.mean(rew_sup))
                rew_std_sup = float(np.std(rew_sup)) or 1e-8
                ret_mean_sup = float(np.mean(ret_sup))
                ret_std_sup = float(np.std(ret_sup)) or 1e-8

                t_state_mean = th.tensor(state_mean_sup, device=self.device, dtype=th.float32).unsqueeze(0)
                t_state_std = th.tensor(state_std_sup, device=self.device, dtype=th.float32).unsqueeze(0)
                t_adv_m = th.tensor([adv_mean_sup], device=self.device)
                t_adv_s = th.tensor([adv_std_sup], device=self.device)
                t_rew_m = th.tensor([rew_mean_sup], device=self.device)
                t_rew_s = th.tensor([rew_std_sup], device=self.device)
                t_ret_m = th.tensor([ret_mean_sup], device=self.device)
                t_ret_s = th.tensor([ret_std_sup], device=self.device)

                context = self.meta_head.compute_context_from_stats(
                    t_state_mean, t_state_std, t_adv_m, t_adv_s, t_rew_m, t_rew_s, t_ret_m, t_ret_s,
                )
                context = context.expand(len(qry_idx), -1)
                obs_q = th.tensor(obs_qry, device=self.device, dtype=th.float32)
                adv_q = th.tensor(adv_qry, device=self.device, dtype=th.float32)
                delta = self.meta_head(obs_q, adv_q, context)
                a_tilde = adv_q + delta
                a_tilde = self.meta_head.normalize_and_clip(a_tilde, clip_range=self.meta_advantage_clip)
                buf.advantages[qry_idx, j] = a_tilde.detach().cpu().numpy()

                a_hat = (ret_qry - val_qry).astype(np.float32)
                a_hat = (a_hat - a_hat.mean()) / (a_hat.std() + 1e-8)
                a_hat_t = th.tensor(a_hat, device=self.device, dtype=th.float32)
                meta_loss = th.mean((a_tilde - a_hat_t) ** 2) + self.meta_beta * th.mean((a_tilde - adv_q) ** 2)
                meta_losses.append(meta_loss)

        if meta_losses:
            self.meta_optimizer.zero_grad()
            total_meta = th.stack(meta_losses).mean()
            total_meta.backward()
            th.nn.utils.clip_grad_norm_(self.meta_head.parameters(), self.max_grad_norm)
            self.meta_optimizer.step()
            if self.verbose >= 2:
                self.logger.record("train/meta_loss", total_meta.item())

    def train(self) -> None:
        self.policy.set_training_mode(True)
        self._update_learning_rate(self.policy.optimizer)
        self._apply_meta_advantage_correction()
        super().train()
