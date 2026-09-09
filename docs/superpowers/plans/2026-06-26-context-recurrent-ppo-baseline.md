# Context-RecurrentPPO Baseline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a fair Context-RecurrentPPO baseline that learns task context through PPO policy/value gradients while leaving rewards, GAE, and Agri-MetaRL advantage calibration untouched.

**Architecture:** Add a standalone `ContextRecurrentPPO` algorithm with its own rollout buffer, diagnostics, config, and registration points. It reuses `TaskSupportMemory`, `Transition`, and `TransitionSetEncoder`, but augments policy observations with context instead of applying residual advantage correction.

**Tech Stack:** Python, PyTorch, Gymnasium, Stable-Baselines3, sb3-contrib RecurrentPPO, pytest, YAML configs.

---

## Scope and File Map

Create:

- `src/gl_gym/RL/context_recurrent_ppo.py` — standalone baseline algorithm.
- `src/gl_gym/RL/context_recurrent_ppo_buffer.py` — recurrent rollout buffer that stores task keys, query masks, and immutable support snapshots.
- `src/gl_gym/RL/context_recurrent_ppo_diagnostics.py` — small diagnostics collector for context activity and norms.
- `configs/agents/context_recurrentppo.yml` — config matched to `recurrentppo.yml` and Agri-MetaRL support settings.
- `tests/agri_metarl/test_context_recurrent_ppo.py` — focused unit/smoke tests.

Modify:

- `src/gl_gym/RL/experiment_manager.py` — register `context_recurrentppo`.
- `src/gl_gym/experiments/evaluate_rl.py` — register algorithm for ad hoc evaluation.
- `experiments/scripts/evaluate_fixed_protocol.py` — register algorithm for fixed protocol.
- `experiments/scripts/evaluate_train_test_generalization.py` — register algorithm.
- `experiments/scripts/evaluate_few_update.py` — register algorithm.
- `experiments/scripts/record_trajectory_60d.py` — register algorithm.
- `experiments/scripts/record_trajectory_24h_all.py` — register algorithm.

Do not modify:

- Agri-MetaRL v2 calibration behavior.
- `AdvantageResidualHead`.
- reward calculation.
- GAE calculation.
- archived result-scaling scripts.

---

### Task 1: Add context diagnostics

**Files:**

- Create: `src/gl_gym/RL/context_recurrent_ppo_diagnostics.py`
- Test: `tests/agri_metarl/test_context_recurrent_ppo.py`

- [ ] **Step 1: Write failing diagnostics tests**

Add these imports and tests to `tests/agri_metarl/test_context_recurrent_ppo.py`:

```python
import numpy as np

from gl_gym.RL.context_recurrent_ppo_diagnostics import ContextDiagnostics


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
```

- [ ] **Step 2: Run the failing diagnostics tests**

Run:

```powershell
python -m pytest tests\agri_metarl\test_context_recurrent_ppo.py::test_context_diagnostics_reports_finite_defaults tests\agri_metarl\test_context_recurrent_ppo.py::test_context_diagnostics_records_active_and_zero_contexts -q
```

Expected: FAIL with `ModuleNotFoundError: No module named 'gl_gym.RL.context_recurrent_ppo_diagnostics'`.

- [ ] **Step 3: Implement diagnostics**

Create `src/gl_gym/RL/context_recurrent_ppo_diagnostics.py`:

```python
"""Diagnostics for Context-RecurrentPPO context usage."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np


@dataclass
class ContextDiagnostics:
    context_norms: list[float] = field(default_factory=list)
    active_flags: list[bool] = field(default_factory=list)
    support_sizes: list[int] = field(default_factory=list)
    task_contexts: dict[str, list[np.ndarray]] = field(
        default_factory=lambda: defaultdict(list)
    )
    last_summary: dict[str, float] = field(default_factory=dict)

    def record_contexts(
        self,
        contexts: np.ndarray,
        active_mask: np.ndarray,
        support_sizes: np.ndarray,
        task_instance_keys: np.ndarray,
    ) -> None:
        contexts = np.asarray(contexts, dtype=np.float32)
        active_mask = np.asarray(active_mask, dtype=bool).reshape(-1)
        support_sizes = np.asarray(support_sizes, dtype=np.int64).reshape(-1)
        task_instance_keys = np.asarray(task_instance_keys, dtype=object).reshape(-1)
        if contexts.shape[0] != active_mask.size:
            raise ValueError("contexts and active_mask must have matching rows")
        if contexts.shape[0] != support_sizes.size:
            raise ValueError("contexts and support_sizes must have matching rows")
        if contexts.shape[0] != task_instance_keys.size:
            raise ValueError("contexts and task_instance_keys must have matching rows")

        norms = np.linalg.norm(contexts, axis=1)
        self.context_norms.extend(float(value) for value in norms)
        self.active_flags.extend(bool(value) for value in active_mask)
        self.support_sizes.extend(int(value) for value in support_sizes)
        for key, context in zip(task_instance_keys, contexts, strict=True):
            if key is not None:
                self.task_contexts[str(key)].append(np.array(context, copy=True))

    def summarize(self) -> dict[str, float]:
        if self.active_flags:
            active = np.asarray(self.active_flags, dtype=np.float32)
            active_fraction = float(active.mean())
        else:
            active_fraction = 0.0

        if self.context_norms:
            norms = np.asarray(self.context_norms, dtype=np.float32)
            norm_mean = float(norms.mean())
            norm_std = float(norms.std())
        else:
            norm_mean = 0.0
            norm_std = 0.0

        support_size_mean = (
            float(np.asarray(self.support_sizes, dtype=np.float32).mean())
            if self.support_sizes
            else 0.0
        )

        task_means = [
            np.stack(contexts).mean(axis=0)
            for contexts in self.task_contexts.values()
            if contexts
        ]
        if len(task_means) >= 2:
            between_task_variance = float(np.stack(task_means).var(axis=0).mean())
        else:
            between_task_variance = 0.0

        summary = {
            "train/context_active_fraction": active_fraction,
            "train/context_norm_mean": norm_mean,
            "train/context_norm_std": norm_std,
            "train/no_context_fraction": 1.0 - active_fraction,
            "train/support_size_mean": support_size_mean,
            "train/context_between_task_variance": between_task_variance,
        }
        self.last_summary = dict(summary)
        return summary

    def reset(self) -> None:
        self.context_norms.clear()
        self.active_flags.clear()
        self.support_sizes.clear()
        self.task_contexts.clear()
```

- [ ] **Step 4: Run diagnostics tests**

Run:

```powershell
python -m pytest tests\agri_metarl\test_context_recurrent_ppo.py::test_context_diagnostics_reports_finite_defaults tests\agri_metarl\test_context_recurrent_ppo.py::test_context_diagnostics_records_active_and_zero_contexts -q
```

Expected: PASS.

- [ ] **Step 5: Checkpoint without commit**

Run:

```powershell
git status --short src\gl_gym\RL\context_recurrent_ppo_diagnostics.py tests\agri_metarl\test_context_recurrent_ppo.py
```

Expected: files are modified/untracked. Do not stage or commit unless the user explicitly asks.

---

### Task 2: Add a context-aware recurrent rollout buffer

**Files:**

- Create: `src/gl_gym/RL/context_recurrent_ppo_buffer.py`
- Modify: `tests/agri_metarl/test_context_recurrent_ppo.py`

- [ ] **Step 1: Write failing buffer tests**

Append to `tests/agri_metarl/test_context_recurrent_ppo.py`:

```python
import torch
from gymnasium import spaces
from sb3_contrib.common.recurrent.type_aliases import RNNStates

from gl_gym.RL.agri_metarl.memory import Transition
from gl_gym.RL.context_recurrent_ppo_buffer import ContextRecurrentRolloutBuffer


def _buffer_states():
    pair = (torch.zeros((1, 2, 3)), torch.zeros((1, 2, 3)))
    return RNNStates(pi=pair, vf=pair)


def _support_transition(value: float):
    observation = np.array([value, 0.0], dtype=np.float32)
    return Transition(
        observation=observation,
        action=np.array([0.1], dtype=np.float32),
        reward=value,
        next_observation=observation + 1.0,
        done=False,
    )


def test_context_buffer_stores_keys_masks_and_support_snapshots():
    buffer = ContextRecurrentRolloutBuffer(
        buffer_size=2,
        observation_space=spaces.Box(-1, 1, shape=(2,), dtype=np.float32),
        action_space=spaces.Box(-1, 1, shape=(1,), dtype=np.float32),
        hidden_state_shape=(2, 1, 2, 3),
        n_envs=2,
    )
    support_a = (_support_transition(1.0), _support_transition(2.0))
    support_b = (_support_transition(9.0),)

    buffer.add(
        obs=np.zeros((2, 2), dtype=np.float32),
        action=np.zeros((2, 1), dtype=np.float32),
        reward=np.ones(2, dtype=np.float32),
        episode_start=np.zeros(2, dtype=bool),
        value=torch.zeros(2),
        log_prob=torch.zeros(2),
        lstm_states=_buffer_states(),
        task_instance_keys=np.array(["task-a", "task-b"], dtype=object),
        context_active_mask=np.array([True, False]),
        support_snapshots=(support_a, support_b),
        support_sizes=np.array([2, 1]),
    )

    assert buffer.task_instance_keys[0].tolist() == ["task-a", "task-b"]
    assert buffer.context_active_mask[0].tolist() == [True, False]
    assert buffer.support_sizes[0].tolist() == [2, 1]
    assert buffer.support_snapshots[0][0] == support_a
    assert buffer.support_snapshots[0][1] == support_b


def test_context_buffer_reset_clears_context_metadata():
    buffer = ContextRecurrentRolloutBuffer(
        buffer_size=1,
        observation_space=spaces.Box(-1, 1, shape=(2,), dtype=np.float32),
        action_space=spaces.Box(-1, 1, shape=(1,), dtype=np.float32),
        hidden_state_shape=(1, 1, 1, 3),
        n_envs=1,
    )
    buffer.add(
        obs=np.zeros((1, 2), dtype=np.float32),
        action=np.zeros((1, 1), dtype=np.float32),
        reward=np.ones(1, dtype=np.float32),
        episode_start=np.zeros(1, dtype=bool),
        value=torch.zeros(1),
        log_prob=torch.zeros(1),
        lstm_states=RNNStates(
            pi=(torch.zeros((1, 1, 3)), torch.zeros((1, 1, 3))),
            vf=(torch.zeros((1, 1, 3)), torch.zeros((1, 1, 3))),
        ),
        task_instance_keys=np.array(["task-a"], dtype=object),
        context_active_mask=np.array([True]),
        support_snapshots=((_support_transition(1.0),),),
        support_sizes=np.array([1]),
    )

    buffer.reset()

    assert np.all(buffer.task_instance_keys == None)  # noqa: E711
    assert not buffer.context_active_mask.any()
    assert np.all(buffer.support_sizes == 0)
    assert buffer.support_snapshots[0][0] == ()
```

- [ ] **Step 2: Run failing buffer tests**

Run:

```powershell
python -m pytest tests\agri_metarl\test_context_recurrent_ppo.py::test_context_buffer_stores_keys_masks_and_support_snapshots tests\agri_metarl\test_context_recurrent_ppo.py::test_context_buffer_reset_clears_context_metadata -q
```

Expected: FAIL with `ModuleNotFoundError: No module named 'gl_gym.RL.context_recurrent_ppo_buffer'`.

- [ ] **Step 3: Implement buffer**

Create `src/gl_gym/RL/context_recurrent_ppo_buffer.py`:

```python
"""Rollout storage for Context-RecurrentPPO."""

from __future__ import annotations

import numpy as np
from sb3_contrib.common.recurrent.buffers import RecurrentRolloutBuffer
from sb3_contrib.common.recurrent.type_aliases import RNNStates


class ContextRecurrentRolloutBuffer(RecurrentRolloutBuffer):
    """Store per-sample task identity and support snapshots for context replay."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._reset_context_metadata()

    def _reset_context_metadata(self) -> None:
        self.task_instance_keys = np.full(
            (self.buffer_size, self.n_envs), None, dtype=object
        )
        self.context_active_mask = np.zeros(
            (self.buffer_size, self.n_envs), dtype=bool
        )
        self.support_sizes = np.zeros((self.buffer_size, self.n_envs), dtype=np.int64)
        self.support_snapshots = np.empty(
            (self.buffer_size, self.n_envs), dtype=object
        )
        for row in range(self.buffer_size):
            for env_index in range(self.n_envs):
                self.support_snapshots[row, env_index] = ()

    def reset(self) -> None:
        super().reset()
        self._reset_context_metadata()

    def add(
        self,
        *args,
        lstm_states: RNNStates,
        task_instance_keys,
        context_active_mask,
        support_snapshots,
        support_sizes,
        **kwargs,
    ) -> None:
        row_index = self.pos
        super().add(*args, lstm_states=lstm_states, **kwargs)

        keys = np.asarray(task_instance_keys, dtype=object).reshape(-1)
        active = np.asarray(context_active_mask, dtype=bool).reshape(-1)
        sizes = np.asarray(support_sizes, dtype=np.int64).reshape(-1)
        snapshots = np.asarray(support_snapshots, dtype=object).reshape(-1)

        if keys.size != self.n_envs:
            raise ValueError("task_instance_keys must contain one key per environment")
        if active.size != self.n_envs:
            raise ValueError("context_active_mask must contain one flag per environment")
        if sizes.size != self.n_envs:
            raise ValueError("support_sizes must contain one size per environment")
        if snapshots.size != self.n_envs:
            raise ValueError("support_snapshots must contain one snapshot per environment")

        self.task_instance_keys[row_index, :] = keys
        self.context_active_mask[row_index, :] = active
        self.support_sizes[row_index, :] = sizes
        for env_index, snapshot in enumerate(snapshots):
            self.support_snapshots[row_index, env_index] = tuple(snapshot)
```

- [ ] **Step 4: Run buffer tests**

Run:

```powershell
python -m pytest tests\agri_metarl\test_context_recurrent_ppo.py::test_context_buffer_stores_keys_masks_and_support_snapshots tests\agri_metarl\test_context_recurrent_ppo.py::test_context_buffer_reset_clears_context_metadata -q
```

Expected: PASS.

- [ ] **Step 5: Checkpoint without commit**

Run:

```powershell
git status --short src\gl_gym\RL\context_recurrent_ppo_buffer.py tests\agri_metarl\test_context_recurrent_ppo.py
```

Expected: files are modified/untracked. Do not stage or commit unless the user explicitly asks.

---

### Task 3: Add ContextRecurrentPPO setup and helper methods

**Files:**

- Create: `src/gl_gym/RL/context_recurrent_ppo.py`
- Modify: `tests/agri_metarl/test_context_recurrent_ppo.py`

- [ ] **Step 1: Write failing helper tests**

Append to `tests/agri_metarl/test_context_recurrent_ppo.py`:

```python
from gl_gym.RL.context_recurrent_ppo import ContextRecurrentPPO


def test_context_algorithm_augments_observation_space_without_mutating_env_space():
    raw_space = spaces.Box(-1, 1, shape=(3,), dtype=np.float32)
    augmented = ContextRecurrentPPO._augmented_observation_space(raw_space, 5)

    assert raw_space.shape == (3,)
    assert augmented.shape == (8,)
    assert augmented.dtype == raw_space.dtype


def test_context_algorithm_rejects_non_box_observation_space():
    with pytest.raises(TypeError, match="Box observation space"):
        ContextRecurrentPPO._augmented_observation_space(
            spaces.Discrete(3), context_dim=4
        )


def test_zero_context_before_support_is_sufficient():
    algorithm = ContextRecurrentPPO.__new__(ContextRecurrentPPO)
    algorithm.context_dim = 4
    algorithm.support_size = 2
    algorithm.device = torch.device("cpu")

    context, active = algorithm._context_from_support(())

    assert active is False
    assert torch.equal(context, torch.zeros(4))
```

Also add `import pytest` near the top of the file if not already present.

- [ ] **Step 2: Run failing helper tests**

Run:

```powershell
python -m pytest tests\agri_metarl\test_context_recurrent_ppo.py::test_context_algorithm_augments_observation_space_without_mutating_env_space tests\agri_metarl\test_context_recurrent_ppo.py::test_context_algorithm_rejects_non_box_observation_space tests\agri_metarl\test_context_recurrent_ppo.py::test_zero_context_before_support_is_sufficient -q
```

Expected: FAIL until `ContextRecurrentPPO` exists.

- [ ] **Step 3: Implement class setup and helpers**

Create `src/gl_gym/RL/context_recurrent_ppo.py`:

```python
"""Context-RecurrentPPO baseline.

This baseline uses task support memory to infer context and concatenates that
context to policy/value observations. It intentionally does not use Agri-MetaRL
advantage residual correction or Monte Carlo calibration.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import numpy as np
import torch as th
from gymnasium import spaces
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.type_aliases import GymEnv
from stable_baselines3.common.utils import obs_as_tensor
from stable_baselines3.common.vec_env import VecEnv
from sb3_contrib.common.recurrent.buffers import RecurrentRolloutBuffer
from sb3_contrib.common.recurrent.policies import RecurrentActorCriticPolicy
from sb3_contrib.common.recurrent.type_aliases import RNNStates
from sb3_contrib.ppo_recurrent import RecurrentPPO

from gl_gym.RL.agri_metarl.memory import TaskSupportMemory, Transition
from gl_gym.RL.agri_metarl.meta_advantage_head import TransitionSetEncoder
from gl_gym.RL.context_recurrent_ppo_buffer import ContextRecurrentRolloutBuffer
from gl_gym.RL.context_recurrent_ppo_diagnostics import ContextDiagnostics


class ContextRecurrentPPO(RecurrentPPO):
    """RecurrentPPO with task context concatenated to policy/value inputs."""

    def __init__(
        self,
        policy: str | type[RecurrentActorCriticPolicy],
        env: GymEnv | str,
        support_size: int = 256,
        max_task_instances: int = 128,
        context_dim: int = 64,
        transition_hidden_dim: int = 128,
        **kwargs,
    ):
        self.support_size = support_size
        self.max_task_instances = max_task_instances
        self.context_dim = context_dim
        self.transition_hidden_dim = transition_hidden_dim
        super().__init__(policy, env, **kwargs)

    @staticmethod
    def _augmented_observation_space(
        observation_space: spaces.Space, context_dim: int
    ) -> spaces.Box:
        if not isinstance(observation_space, spaces.Box) or observation_space.shape is None:
            raise TypeError("ContextRecurrentPPO requires a Box observation space")
        if len(observation_space.shape) != 1:
            raise TypeError("ContextRecurrentPPO requires a flat Box observation space")
        low = np.concatenate(
            [
                np.asarray(observation_space.low, dtype=np.float32).reshape(-1),
                np.full(context_dim, -np.inf, dtype=np.float32),
            ]
        )
        high = np.concatenate(
            [
                np.asarray(observation_space.high, dtype=np.float32).reshape(-1),
                np.full(context_dim, np.inf, dtype=np.float32),
            ]
        )
        return spaces.Box(low=low, high=high, dtype=observation_space.dtype)

    def _setup_model(self) -> None:
        self.raw_observation_space = self.observation_space
        self.observation_space = self._augmented_observation_space(
            self.raw_observation_space, self.context_dim
        )
        self._setup_lr_schedule()
        self.set_random_seed(self.seed)

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

        raw_obs_dim = int(np.prod(self.raw_observation_space.shape))
        action_dim = int(np.prod(self.action_space.shape))
        self.support_memory = TaskSupportMemory(
            support_size=self.support_size,
            max_instances=self.max_task_instances,
        )
        self.context_encoder = TransitionSetEncoder(
            obs_dim=raw_obs_dim,
            action_dim=action_dim,
            context_dim=self.context_dim,
            hidden_dim=self.transition_hidden_dim,
        ).to(self.device)
        self.context_diagnostics = ContextDiagnostics()
        self.policy.optimizer.add_param_group(
            {"params": list(self.context_encoder.parameters())}
        )

        from stable_baselines3.common.utils import get_schedule_fn

        self.clip_range = get_schedule_fn(self.clip_range)
        if self.clip_range_vf is not None:
            self.clip_range_vf = get_schedule_fn(self.clip_range_vf)

    def _tensorize_support(self, support: tuple[Transition, ...]) -> dict[str, th.Tensor]:
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
        return context.reshape(-1), True
```

- [ ] **Step 4: Run helper tests**

Run:

```powershell
python -m pytest tests\agri_metarl\test_context_recurrent_ppo.py::test_context_algorithm_augments_observation_space_without_mutating_env_space tests\agri_metarl\test_context_recurrent_ppo.py::test_context_algorithm_rejects_non_box_observation_space tests\agri_metarl\test_context_recurrent_ppo.py::test_zero_context_before_support_is_sufficient -q
```

Expected: PASS.

- [ ] **Step 5: Checkpoint without commit**

Run:

```powershell
git status --short src\gl_gym\RL\context_recurrent_ppo.py tests\agri_metarl\test_context_recurrent_ppo.py
```

Expected: files are modified/untracked. Do not stage or commit unless the user explicitly asks.

---

### Task 4: Implement rollout-time context collection

**Files:**

- Modify: `src/gl_gym/RL/context_recurrent_ppo.py`
- Modify: `tests/agri_metarl/test_context_recurrent_ppo.py`

- [ ] **Step 1: Write failing rollout helper tests**

Append to `tests/agri_metarl/test_context_recurrent_ppo.py`:

```python
def test_observe_transition_records_support_then_query():
    algorithm = ContextRecurrentPPO.__new__(ContextRecurrentPPO)
    algorithm.support_size = 1
    algorithm.support_memory = TaskSupportMemory(support_size=1, max_instances=4)

    observations = np.array([[1.0, 0.0]], dtype=np.float32)
    actions = np.array([[0.1]], dtype=np.float32)
    rewards = np.array([1.0], dtype=np.float32)
    next_observations = np.array([[2.0, 0.0]], dtype=np.float32)
    dones = np.array([False])
    infos = [{"task_descriptor": {}, "task_instance_key": "task-a"}]

    keys, active, snapshots, sizes = algorithm._observe_raw_transitions(
        observations, actions, rewards, next_observations, dones, infos
    )
    assert keys.tolist() == ["task-a"]
    assert active.tolist() == [False]
    assert snapshots[0] == ()
    assert sizes.tolist() == [0]

    keys, active, snapshots, sizes = algorithm._observe_raw_transitions(
        observations, actions, rewards, next_observations, dones, infos
    )
    assert active.tolist() == [True]
    assert len(snapshots[0]) == 1
    assert sizes.tolist() == [1]


def test_augment_observations_uses_zero_context_for_inactive_support():
    algorithm = ContextRecurrentPPO.__new__(ContextRecurrentPPO)
    algorithm.context_dim = 2
    algorithm.support_size = 2
    algorithm.device = torch.device("cpu")

    raw = np.array([[1.0, 2.0]], dtype=np.float32)
    augmented, contexts, active = algorithm._augment_raw_observations(raw, ((),))

    assert augmented.shape == (1, 4)
    assert augmented.tolist() == [[1.0, 2.0, 0.0, 0.0]]
    assert contexts.tolist() == [[0.0, 0.0]]
    assert active.tolist() == [False]
```

- [ ] **Step 2: Run failing rollout helper tests**

Run:

```powershell
python -m pytest tests\agri_metarl\test_context_recurrent_ppo.py::test_observe_transition_records_support_then_query tests\agri_metarl\test_context_recurrent_ppo.py::test_augment_observations_uses_zero_context_for_inactive_support -q
```

Expected: FAIL because helper methods are missing.

- [ ] **Step 3: Add rollout helper methods**

Add these methods to `ContextRecurrentPPO`:

```python
    def _augment_raw_observations(
        self,
        raw_observations: np.ndarray,
        support_snapshots: tuple[tuple[Transition, ...], ...],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        contexts: list[np.ndarray] = []
        active_flags: list[bool] = []
        with th.no_grad():
            for support in support_snapshots:
                context, active = self._context_from_support(support)
                contexts.append(context.detach().cpu().numpy())
                active_flags.append(active)
        context_array = np.asarray(contexts, dtype=np.float32)
        raw_array = np.asarray(raw_observations, dtype=np.float32)
        augmented = np.concatenate([raw_array, context_array], axis=-1)
        return augmented, context_array, np.asarray(active_flags, dtype=bool)

    def _observe_raw_transitions(
        self,
        observations: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        next_observations: np.ndarray,
        dones: np.ndarray,
        infos: list[dict[str, Any]],
    ) -> tuple[np.ndarray, np.ndarray, tuple[tuple[Transition, ...], ...], np.ndarray]:
        task_instance_keys = np.empty(len(infos), dtype=object)
        active = np.zeros(len(infos), dtype=bool)
        support_snapshots: list[tuple[Transition, ...]] = []
        support_sizes = np.zeros(len(infos), dtype=np.int64)

        for env_index, info in enumerate(infos):
            if "task_descriptor" not in info or "task_instance_key" not in info:
                raise KeyError("environment info must contain complete task identity")
            key = info["task_instance_key"]
            task_instance_keys[env_index] = key
            support = self.support_memory.support(key)
            support_snapshots.append(support)
            support_sizes[env_index] = len(support)
            active[env_index] = len(support) >= self.support_size

            next_observation = (
                info.get("terminal_observation", next_observations[env_index])
                if dones[env_index]
                else next_observations[env_index]
            )
            transition = Transition(
                observation=observations[env_index],
                action=np.asarray(actions[env_index]).reshape(-1),
                reward=float(rewards[env_index]),
                next_observation=next_observation,
                done=bool(dones[env_index]),
            )
            self.support_memory.observe(key, transition)

        return task_instance_keys, active, tuple(support_snapshots), support_sizes
```

- [ ] **Step 4: Run rollout helper tests**

Run:

```powershell
python -m pytest tests\agri_metarl\test_context_recurrent_ppo.py::test_observe_transition_records_support_then_query tests\agri_metarl\test_context_recurrent_ppo.py::test_augment_observations_uses_zero_context_for_inactive_support -q
```

Expected: PASS.

- [ ] **Step 5: Implement collect_rollouts**

Add a `collect_rollouts()` method to `ContextRecurrentPPO` based on `AgriMetaRL.collect_rollouts`, with these required differences:

```python
    def collect_rollouts(
        self,
        env: VecEnv,
        callback: BaseCallback,
        rollout_buffer: ContextRecurrentRolloutBuffer,
        n_rollout_steps: int,
    ) -> bool:
        assert isinstance(rollout_buffer, (ContextRecurrentRolloutBuffer, RecurrentRolloutBuffer))
        assert self._last_obs is not None
        self.policy.set_training_mode(False)

        n_steps = 0
        rollout_buffer.reset()
        self.support_memory.begin_rollout()
        if self.use_sde:
            self.policy.reset_noise(env.num_envs)

        callback.on_rollout_start()
        lstm_states = deepcopy(self._last_lstm_states)

        while n_steps < n_rollout_steps:
            if self.use_sde and self.sde_sample_freq > 0 and n_steps % self.sde_sample_freq == 0:
                self.policy.reset_noise(env.num_envs)

            initial_empty_support_snapshots = tuple(() for _ in range(env.num_envs))
            if n_steps == 0:
                support_snapshots = initial_empty_support_snapshots
            else:
                support_snapshots = tuple(
                    self.support_memory.support(str(key))
                    if key is not None else ()
                    for key in getattr(self, "_last_task_instance_keys", np.full(env.num_envs, None, dtype=object))
                )
            augmented_obs, contexts, active_mask = self._augment_raw_observations(
                self._last_obs, support_snapshots
            )
            self.context_diagnostics.record_contexts(
                contexts,
                active_mask,
                np.asarray([len(snapshot) for snapshot in support_snapshots]),
                getattr(self, "_last_task_instance_keys", np.full(env.num_envs, None, dtype=object)),
            )

            with th.no_grad():
                obs_tensor = obs_as_tensor(augmented_obs, self.device)
                episode_starts = th.tensor(
                    self._last_episode_starts, dtype=th.float32, device=self.device
                )
                actions, values, log_probs, lstm_states = self.policy(
                    obs_tensor, lstm_states, episode_starts
                )

            actions = actions.cpu().numpy()
            clipped_actions = (
                np.clip(actions, self.action_space.low, self.action_space.high)
                if isinstance(self.action_space, spaces.Box)
                else actions
            )

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
                    terminal_raw = np.asarray(infos[idx]["terminal_observation"], dtype=np.float32).reshape(1, -1)
                    terminal_augmented, _, _ = self._augment_raw_observations(
                        terminal_raw, (support_snapshots[idx],)
                    )
                    terminal_obs = self.policy.obs_to_tensor(terminal_augmented)[0]
                    with th.no_grad():
                        terminal_lstm_state = (
                            lstm_states.vf[0][:, idx : idx + 1, :].contiguous(),
                            lstm_states.vf[1][:, idx : idx + 1, :].contiguous(),
                        )
                        episode_starts_t = th.tensor([False], dtype=th.float32, device=self.device)
                        terminal_value = self.policy.predict_values(
                            terminal_obs, terminal_lstm_state, episode_starts_t
                        )[0]
                    rewards[idx] += self.gamma * terminal_value

            task_instance_keys, context_active, support_snapshots, support_sizes = (
                self._observe_raw_transitions(
                    observations=self._last_obs,
                    actions=clipped_actions,
                    rewards=rewards,
                    next_observations=new_obs,
                    dones=dones,
                    infos=infos,
                )
            )

            buffer_augmented_obs, _, _ = self._augment_raw_observations(
                self._last_obs, support_snapshots
            )
            rollout_buffer.add(
                buffer_augmented_obs,
                actions,
                rewards,
                self._last_episode_starts,
                values,
                log_probs,
                lstm_states=deepcopy(self._last_lstm_states),
                task_instance_keys=task_instance_keys,
                context_active_mask=context_active,
                support_snapshots=support_snapshots,
                support_sizes=support_sizes,
            )

            self._last_task_instance_keys = task_instance_keys
            self._last_obs = new_obs
            self._last_episode_starts = dones
            self._last_lstm_states = lstm_states

        final_support = tuple(
            self.support_memory.support(str(key))
            if key is not None else ()
            for key in getattr(self, "_last_task_instance_keys", np.full(env.num_envs, None, dtype=object))
        )
        final_augmented_obs, _, _ = self._augment_raw_observations(
            self._last_obs, final_support
        )
        with th.no_grad():
            episode_starts_t = th.tensor(
                self._last_episode_starts, dtype=th.float32, device=self.device
            )
            last_values = self.policy.predict_values(
                obs_as_tensor(final_augmented_obs, self.device),
                self._last_lstm_states.vf,
                episode_starts_t,
            )
        rollout_buffer.compute_returns_and_advantage(
            last_values=last_values, dones=self._last_episode_starts
        )
        callback.on_rollout_end()
        return True
```

If this method becomes brittle during execution, keep the behavior identical to `AgriMetaRL.collect_rollouts` and only swap policy observations from raw to augmented.

- [ ] **Step 6: Run focused tests**

Run:

```powershell
python -m pytest tests\agri_metarl\test_context_recurrent_ppo.py -q
```

Expected: PASS for all tests currently in that file.

- [ ] **Step 7: Checkpoint without commit**

Run:

```powershell
git status --short src\gl_gym\RL\context_recurrent_ppo.py tests\agri_metarl\test_context_recurrent_ppo.py
```

Expected: modified files. Do not stage or commit unless the user explicitly asks.

---

### Task 5: Override training so context is recomputed with gradients

**Files:**

- Modify: `src/gl_gym/RL/context_recurrent_ppo.py`
- Modify: `tests/agri_metarl/test_context_recurrent_ppo.py`

- [ ] **Step 1: Write failing train-context tests**

Append to `tests/agri_metarl/test_context_recurrent_ppo.py`:

```python
def test_training_context_recompute_keeps_encoder_in_graph():
    algorithm = ContextRecurrentPPO.__new__(ContextRecurrentPPO)
    algorithm.context_dim = 4
    algorithm.support_size = 1
    algorithm.device = torch.device("cpu")
    algorithm.context_encoder = TransitionSetEncoder(2, 1, context_dim=4, hidden_dim=8)
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
    raw_obs = torch.tensor([[2.0, 0.0]], dtype=torch.float32)

    augmented = algorithm._augment_training_observations(raw_obs, ((),))

    assert torch.equal(augmented, torch.tensor([[2.0, 0.0, 0.0, 0.0, 0.0]]))
```

- [ ] **Step 2: Run failing train-context tests**

Run:

```powershell
python -m pytest tests\agri_metarl\test_context_recurrent_ppo.py::test_training_context_recompute_keeps_encoder_in_graph tests\agri_metarl\test_context_recurrent_ppo.py::test_training_context_recompute_uses_zero_for_inactive_support -q
```

Expected: FAIL because `_augment_training_observations` does not exist.

- [ ] **Step 3: Implement train-time augmentation**

Add to `ContextRecurrentPPO`:

```python
    def _augment_training_observations(
        self,
        augmented_or_raw_observations: th.Tensor,
        support_snapshots: tuple[tuple[Transition, ...], ...],
    ) -> th.Tensor:
        raw_dim = int(np.prod(self.raw_observation_space.shape)) if hasattr(self, "raw_observation_space") else (
            augmented_or_raw_observations.shape[-1] - self.context_dim
        )
        raw_observations = augmented_or_raw_observations[..., :raw_dim]
        flat_raw = raw_observations.reshape(-1, raw_dim)
        flat_supports = tuple(support_snapshots)
        if len(flat_supports) != flat_raw.shape[0]:
            raise ValueError("support_snapshots must match flattened observation rows")

        contexts: list[th.Tensor] = []
        for support in flat_supports:
            if len(support) < self.support_size:
                contexts.append(th.zeros(self.context_dim, device=self.device))
            else:
                contexts.append(
                    self.context_encoder(**self._tensorize_support(support)).reshape(-1)
                )
        context_tensor = th.stack(contexts, dim=0).to(flat_raw.device)
        flat_augmented = th.cat([flat_raw, context_tensor], dim=-1)
        return flat_augmented.reshape(*raw_observations.shape[:-1], raw_dim + self.context_dim)
```

- [ ] **Step 4: Run train-context tests**

Run:

```powershell
python -m pytest tests\agri_metarl\test_context_recurrent_ppo.py::test_training_context_recompute_keeps_encoder_in_graph tests\agri_metarl\test_context_recurrent_ppo.py::test_training_context_recompute_uses_zero_for_inactive_support -q
```

Expected: PASS.

- [ ] **Step 5: Override `train()`**

Add a `train()` implementation copied from `sb3_contrib.ppo_recurrent.RecurrentPPO.train`, with one behavioral change: before `evaluate_actions()`, replace `rollout_data.observations` with `_augment_training_observations(...)`.

Use this structure:

```python
    def train(self) -> None:
        from stable_baselines3.common.utils import explained_variance

        self.policy.set_training_mode(True)
        self._update_learning_rate(self.policy.optimizer)
        clip_range = self.clip_range(self._current_progress_remaining)
        if self.clip_range_vf is not None:
            clip_range_vf = self.clip_range_vf(self._current_progress_remaining)

        entropy_losses = []
        pg_losses, value_losses = [], []
        clip_fractions = []
        continue_training = True

        flat_support_snapshots = tuple(self.rollout_buffer.support_snapshots.reshape(-1))

        for epoch in range(self.n_epochs):
            approx_kl_divs = []
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                actions = rollout_data.actions
                if isinstance(self.action_space, spaces.Discrete):
                    actions = rollout_data.actions.long().flatten()

                mask = rollout_data.mask > 1e-8

                if self.use_sde:
                    self.policy.reset_noise(self.batch_size)

                batch_supports = tuple(
                    flat_support_snapshots[int(index)]
                    for index in rollout_data.indices
                )
                observations = self._augment_training_observations(
                    rollout_data.observations,
                    batch_supports,
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
                    advantages = (advantages - advantages[mask].mean()) / (
                        advantages[mask].std() + 1e-8
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

                loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss

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
                            f"Early stopping at step {epoch} due to reaching max kl: {approx_kl_div:.2f}"
                        )
                    break

                self.policy.optimizer.zero_grad()
                loss.backward()
                th.nn.utils.clip_grad_norm_(
                    list(self.policy.parameters()) + list(self.context_encoder.parameters()),
                    self.max_grad_norm,
                )
                self.policy.optimizer.step()

            if not continue_training:
                break

        self._n_updates += self.n_epochs
        explained_var = explained_variance(
            self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten()
        )
        context_summary = self.context_diagnostics.summarize()
        for name, value in context_summary.items():
            self.logger.record(name, value)
        self.context_diagnostics.reset()

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
```

During execution, if `rollout_data.indices` is not available in the installed sb3-contrib version, extend `ContextRecurrentRolloutBuffer.get()` to yield a named tuple with `indices`, or add a parallel metadata batching helper. Do not silently fall back to detached rollout contexts.

- [ ] **Step 6: Run context tests**

Run:

```powershell
python -m pytest tests\agri_metarl\test_context_recurrent_ppo.py -q
```

Expected: PASS.

- [ ] **Step 7: Checkpoint without commit**

Run:

```powershell
git status --short src\gl_gym\RL\context_recurrent_ppo.py tests\agri_metarl\test_context_recurrent_ppo.py
```

Expected: modified files. Do not stage or commit unless the user explicitly asks.

---

### Task 6: Add smoke tests proving encoder parameters update

**Files:**

- Modify: `tests/agri_metarl/test_context_recurrent_ppo.py`

- [ ] **Step 1: Add tiny task environment and smoke tests**

Append:

```python
import gymnasium as gym


class TinyContextTaskEnv(gym.Env):
    def __init__(self):
        self.observation_space = spaces.Box(-10, 10, shape=(3,), dtype=np.float32)
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
        reward = float(1.0 - abs(float(action[0])))
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
        return observation, reward, terminated, False, info


def test_context_recurrent_ppo_smoke_updates_encoder_parameters():
    model = ContextRecurrentPPO(
        "MlpLstmPolicy",
        TinyContextTaskEnv(),
        n_steps=2,
        batch_size=2,
        n_epochs=1,
        support_size=1,
        max_task_instances=4,
        context_dim=4,
        transition_hidden_dim=8,
        policy_kwargs={
            "net_arch": {"pi": [8], "vf": [8]},
            "lstm_hidden_size": 8,
        },
        seed=5,
        verbose=0,
        device="cpu",
    )
    before = [
        parameter.detach().clone()
        for parameter in model.context_encoder.parameters()
    ]

    model.learn(total_timesteps=8)

    after = list(model.context_encoder.parameters())
    assert model.num_timesteps >= 8
    assert model._n_updates > 0
    assert any(
        not torch.equal(old, new)
        for old, new in zip(before, after, strict=True)
    )
    assert all(torch.isfinite(parameter).all() for parameter in after)
    assert model.context_diagnostics.last_summary[
        "train/context_active_fraction"
    ] >= 0.0


def test_context_recurrent_ppo_has_no_advantage_residual_or_calibration():
    model = ContextRecurrentPPO(
        "MlpLstmPolicy",
        TinyContextTaskEnv(),
        n_steps=2,
        batch_size=2,
        n_epochs=1,
        support_size=1,
        context_dim=4,
        transition_hidden_dim=8,
        policy_kwargs={
            "net_arch": {"pi": [8], "vf": [8]},
            "lstm_hidden_size": 8,
        },
        seed=6,
        verbose=0,
        device="cpu",
    )

    assert not hasattr(model, "residual_head")
    assert not hasattr(model, "calibration_memory")
```

- [ ] **Step 2: Run smoke tests**

Run:

```powershell
python -m pytest tests\agri_metarl\test_context_recurrent_ppo.py::test_context_recurrent_ppo_smoke_updates_encoder_parameters tests\agri_metarl\test_context_recurrent_ppo.py::test_context_recurrent_ppo_has_no_advantage_residual_or_calibration -q
```

Expected: PASS. If the encoder does not update, stop and fix train-time context recomputation; do not accept detached context.

- [ ] **Step 3: Run all ContextRecurrentPPO tests**

Run:

```powershell
python -m pytest tests\agri_metarl\test_context_recurrent_ppo.py -q
```

Expected: PASS.

- [ ] **Step 4: Checkpoint without commit**

Run:

```powershell
git status --short tests\agri_metarl\test_context_recurrent_ppo.py
```

Expected: modified test file. Do not stage or commit unless the user explicitly asks.

---

### Task 7: Add config and algorithm registration

**Files:**

- Create: `configs/agents/context_recurrentppo.yml`
- Modify: `src/gl_gym/RL/experiment_manager.py`
- Modify: `src/gl_gym/experiments/evaluate_rl.py`
- Modify: `experiments/scripts/evaluate_fixed_protocol.py`
- Modify: `experiments/scripts/evaluate_train_test_generalization.py`
- Modify: `experiments/scripts/evaluate_few_update.py`
- Modify: `experiments/scripts/record_trajectory_60d.py`
- Modify: `experiments/scripts/record_trajectory_24h_all.py`
- Modify: `tests/agri_metarl/test_context_recurrent_ppo.py`

- [ ] **Step 1: Add failing config/registration tests**

Append:

```python
import yaml


def test_context_recurrentppo_config_matches_expected_keys():
    with open("configs/agents/context_recurrentppo.yml", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)["TomatoEnv"]

    assert config["policy"] == "MlpLstmPolicy"
    assert config["support_size"] == 256
    assert config["max_task_instances"] == 128
    assert config["context_dim"] == 64
    assert config["transition_hidden_dim"] == 128
    assert "residual_alpha" not in config
    assert "meta_loss_weight" not in config
    assert "calibration_min_query_samples" not in config


def test_experiment_manager_registers_context_recurrentppo():
    from gl_gym.RL.experiment_manager import ExperimentManager

    manager = ExperimentManager.__new__(ExperimentManager)
    manager.models = {
        "context_recurrentppo": ContextRecurrentPPO,
    }

    assert manager.models["context_recurrentppo"] is ContextRecurrentPPO
```

- [ ] **Step 2: Run failing config test**

Run:

```powershell
python -m pytest tests\agri_metarl\test_context_recurrent_ppo.py::test_context_recurrentppo_config_matches_expected_keys -q
```

Expected: FAIL until config exists.

- [ ] **Step 3: Create config**

Create `configs/agents/context_recurrentppo.yml`:

```yaml
# Context-RecurrentPPO: task context concatenated to recurrent policy/value inputs
TomatoEnv:
  n_envs: 8
  total_timesteps: 2_000_000
  policy: MlpLstmPolicy
  n_steps: 2048
  batch_size: 512
  n_epochs: 8
  gamma: 0.9631
  gae_lambda: 0.9666
  clip_range: 0.2
  normalize_advantage: True
  ent_coef: 0.00006002718320795429
  vf_coef: 0.2599
  max_grad_norm: 0.3
  use_sde: False
  sde_sample_freq: 8
  target_kl: null
  learning_rate: 0.0001161

  policy_kwargs:
    net_arch: { pi: [1024, 1024], vf: [128, 128] }
    optimizer_class: adam
    optimizer_kwargs: { amsgrad: true }
    activation_fn: silu
    log_std_init: np.log(1)
    lstm_hidden_size: 256
    n_lstm_layers: 1
    shared_lstm: false
    enable_critic_lstm: true

  support_size: 256
  max_task_instances: 128
  context_dim: 64
  transition_hidden_dim: 128
```

- [ ] **Step 4: Register in experiment manager**

Modify imports in `src/gl_gym/RL/experiment_manager.py`:

```python
from gl_gym.RL.context_recurrent_ppo import ContextRecurrentPPO
```

Modify the model map:

```python
self.models = {
    "ppo": PPO,
    "sac": SAC,
    "recurrentppo": RecurrentPPO,
    "context_recurrentppo": ContextRecurrentPPO,
    "agri_metarl": AgriMetaRL,
}
```

Also update the docstring line that lists supported algorithms so it includes `context_recurrentppo`.

- [ ] **Step 5: Register in script ALG_MAPs**

For each script with a local `ALG_MAP`, add the import:

```python
from gl_gym.RL.context_recurrent_ppo import ContextRecurrentPPO
```

Then change maps from:

```python
ALG_MAP = {"ppo": PPO, "recurrentppo": RecurrentPPO, "agri_metarl": AgriMetaRL}
```

to:

```python
ALG_MAP = {
    "ppo": PPO,
    "recurrentppo": RecurrentPPO,
    "context_recurrentppo": ContextRecurrentPPO,
    "agri_metarl": AgriMetaRL,
}
```

Apply this to:

- `src/gl_gym/experiments/evaluate_rl.py`
- `experiments/scripts/evaluate_fixed_protocol.py`
- `experiments/scripts/evaluate_train_test_generalization.py`
- `experiments/scripts/evaluate_few_update.py`
- `experiments/scripts/record_trajectory_60d.py`
- `experiments/scripts/record_trajectory_24h_all.py`

- [ ] **Step 6: Run config/registration tests**

Run:

```powershell
python -m pytest tests\agri_metarl\test_context_recurrent_ppo.py::test_context_recurrentppo_config_matches_expected_keys tests\agri_metarl\test_context_recurrent_ppo.py::test_experiment_manager_registers_context_recurrentppo -q
```

Expected: PASS.

- [ ] **Step 7: Checkpoint without commit**

Run:

```powershell
git status --short configs\agents\context_recurrentppo.yml src\gl_gym\RL\experiment_manager.py src\gl_gym\experiments\evaluate_rl.py experiments\scripts\evaluate_fixed_protocol.py experiments\scripts\evaluate_train_test_generalization.py experiments\scripts\evaluate_few_update.py experiments\scripts\record_trajectory_60d.py experiments\scripts\record_trajectory_24h_all.py
```

Expected: modified/untracked files. Do not stage or commit unless the user explicitly asks.

---

### Task 8: Final verification and integration safety checks

**Files:**

- No new files unless failures reveal necessary fixes.

- [ ] **Step 1: Run focused ContextRecurrentPPO tests**

Run:

```powershell
python -m pytest tests\agri_metarl\test_context_recurrent_ppo.py -q
```

Expected: all tests PASS.

- [ ] **Step 2: Run Agri-MetaRL regression tests**

Run:

```powershell
python -m pytest tests\agri_metarl -q
```

Expected: all tests PASS. Existing Windows interpreter shutdown warnings from SB3/pandas/pyarrow may appear if exit code is still 0.

- [ ] **Step 3: Run scaling integrity test**

Run:

```powershell
python -m pytest tests\integrity\test_no_result_scaling.py -q
```

Expected: PASS.

- [ ] **Step 4: Run full test suite**

Run:

```powershell
python -m pytest -q
```

Expected: PASS.

- [ ] **Step 5: Compile source and tests**

Run:

```powershell
python -m compileall -q src tests
```

Expected: exit code 0.

- [ ] **Step 6: Check whitespace and conflict markers**

Run:

```powershell
git diff --check
rg -n "<<<<<<<|=======|>>>>>>>" src tests configs experiments docs
```

Expected: `git diff --check` exits 0; `rg` finds no conflict markers.

- [ ] **Step 7: Check forbidden coupling**

Run:

```powershell
rg -n "AdvantageResidualHead|EpisodeCalibrationMemory|calibration_memory|residual_head|_apply_meta_advantage_correction|normalize_and_clip" src\gl_gym\RL\context_recurrent_ppo.py tests\agri_metarl\test_context_recurrent_ppo.py
```

Expected: no matches except test assertions that check these attributes are absent. If matches appear in algorithm implementation, remove the coupling.

- [ ] **Step 8: Summarize changed files**

Run:

```powershell
git status --short
```

Expected: shows the new Context-RecurrentPPO implementation, tests, config, and registration changes. Do not stage or commit unless the user explicitly asks.

---

## Self-Review Notes

Spec coverage:

- Standalone baseline class: Task 3.
- Shared support memory and encoder: Tasks 3-5.
- Zero context warm-up: Tasks 3-5.
- Training-time context recomputation with gradients: Task 5 and Task 6.
- No residual head or MC calibration: Task 6 and Task 8.
- Diagnostics: Task 1 and Task 5.
- Config and registration: Task 7.
- Verification: Task 8.

Known implementation risk:

- The installed `RecurrentRolloutBuffer.get()` may not expose original flat indices. If `rollout_data.indices` is unavailable, implement explicit metadata batching in `ContextRecurrentRolloutBuffer.get()` rather than using detached rollout-time contexts. The validity requirement is non-negotiable: train-time context must be recomputed as a tensor connected to `TransitionSetEncoder`.

Repository policy:

- This plan intentionally omits automatic `git add` and `git commit` steps because the current repository workflow keeps staging/committing under explicit user control.
