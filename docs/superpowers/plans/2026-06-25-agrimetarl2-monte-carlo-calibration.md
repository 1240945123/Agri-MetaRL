# Agri-MetaRL 2.0 Monte Carlo Calibration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Train the Agri-MetaRL 2.0 transition-set encoder and bounded advantage residual from complete-episode Monte Carlo calibration targets without leaking future rewards into rollout inference.

**Architecture:** A dedicated `EpisodeCalibrationMemory` joins episode fragments across rollout boundaries. Rollout rows carry stable calibration entry IDs; after GAE is computed, the algorithm attaches raw advantages, finalizes terminal episodes with immutable support snapshots, applies the current residual head to active query rows, and trains the meta modules from completed-episode Huber targets.

**Tech Stack:** Python 3.12, NumPy, PyTorch, Gymnasium, Stable-Baselines3, sb3-contrib, pytest, YAML.

**Commit policy:** Do not stage or commit. The user explicitly requested local checkpoints only.

---

## File Map

**Create**

- `src/gl_gym/RL/agri_metarl/calibration.py`: immutable calibration records and bounded cross-rollout episode memory.
- `tests/agri_metarl/test_episode_calibration.py`: lifecycle, numerical, isolation, capacity, and non-finite tests.

**Modify**

- `src/gl_gym/RL/agri_metarl/buffer.py`: store stable calibration entry IDs per rollout row.
- `src/gl_gym/RL/agri_metarl/meta_advantage_head.py`: initialize the final residual layer to zero for safe warm-up.
- `src/gl_gym/RL/agri_metarl/agri_metarl.py`: collect episode records, attach GAE, finalize episodes, separate correction from supervised meta training, and delete the dead legacy correction method.
- `src/gl_gym/RL/agri_metarl/diagnostics.py`: calibration queue, target, completion, and rejected-batch diagnostics.
- `src/gl_gym/RL/agri_metarl/__init__.py`: export calibration record types.
- `configs/agents/agri_metarl.yml`: calibration capacities and optimizer settings.
- `tests/agri_metarl/test_rollout_integration.py`: buffer IDs, algorithm lifecycle, loss descent, rejection, and smoke coverage.
- `docs/superpowers/specs/2026-06-25-agrimetarl2-monte-carlo-calibration-design.md`: implementation-status note only after verification.

### Task 1: Complete-Episode Calibration Memory

**Files:**
- Create: `src/gl_gym/RL/agri_metarl/calibration.py`
- Create: `tests/agri_metarl/test_episode_calibration.py`

- [ ] **Step 1: Write failing cross-rollout and numerical tests**

Create tests using the wished-for API:

```python
import numpy as np
import pytest

from gl_gym.RL.agri_metarl.calibration import EpisodeCalibrationMemory
from gl_gym.RL.agri_metarl.memory import Transition


def transition(reward: float, done: bool = False) -> Transition:
    return Transition(
        observation=np.array([reward, 0.0], dtype=np.float32),
        action=np.array([0.0], dtype=np.float32),
        reward=reward,
        next_observation=np.array([reward + 1.0, 0.0], dtype=np.float32),
        done=done,
    )


def test_three_rollout_fragments_finalize_only_after_terminal_and_gae_attachment():
    memory = EpisodeCalibrationMemory(
        gamma=0.5, residual_alpha=2.0, max_pending_episodes=2,
        max_completed_episodes=2,
    )
    entry_ids = []
    entry_ids.append(memory.observe("task-a", transition(1.0), value=0.5, is_query=False))
    memory.attach_rollout([entry_ids[-1]], [0.25])
    assert memory.ready_task_keys() == ()
    entry_ids.append(memory.observe("task-a", transition(2.0), value=1.0, is_query=True))
    memory.attach_rollout([entry_ids[-1]], [0.75])
    assert memory.ready_task_keys() == ()
    entry_ids.append(memory.observe("task-a", transition(4.0, done=True), value=1.5, is_query=True))
    assert memory.ready_task_keys() == ()
    memory.attach_rollout([entry_ids[-1]], [1.25])
    assert memory.ready_task_keys() == ("task-a",)

    support = (transition(1.0),)
    completed = memory.finalize("task-a", support)

    # Returns are [3, 4, 4]; MC advantages are [2.5, 3, 2.5].
    assert [sample.target_residual for sample in completed.samples] == [2.0, 1.25]
    assert [sample.raw_advantage for sample in completed.samples] == [0.75, 1.25]
    assert completed.support == support
    assert memory.ready_task_keys() == ()


def test_task_instances_never_share_pending_steps():
    memory = EpisodeCalibrationMemory(0.9, 1.0, 4, 4)
    a = memory.observe("task-a", transition(1.0, done=True), 0.0, True)
    b = memory.observe("task-b", transition(9.0, done=True), 0.0, True)
    memory.attach_rollout([a, b], [0.2, 0.8])
    episode_a = memory.finalize("task-a", (transition(1.0),))
    episode_b = memory.finalize("task-b", (transition(9.0),))
    assert episode_a.task_instance_key == "task-a"
    assert episode_b.task_instance_key == "task-b"
    assert episode_a.samples[0].observation[0] == 1.0
    assert episode_b.samples[0].observation[0] == 9.0


def test_pending_capacity_raises_instead_of_dropping_active_episode():
    memory = EpisodeCalibrationMemory(0.9, 1.0, 1, 2)
    memory.observe("task-a", transition(1.0), 0.0, False)
    with pytest.raises(RuntimeError, match="pending episode capacity"):
        memory.observe("task-b", transition(2.0), 0.0, False)
```

- [ ] **Step 2: Verify the tests are red**

Run:

```powershell
python -m pytest tests\agri_metarl\test_episode_calibration.py -v
```

Expected: collection fails because `gl_gym.RL.agri_metarl.calibration` does not exist.

- [ ] **Step 3: Implement immutable records and lifecycle**

Implement these public types and methods in `calibration.py`:

```python
from collections import OrderedDict, deque
from dataclasses import dataclass

import numpy as np

from gl_gym.RL.agri_metarl.memory import Transition


@dataclass(frozen=True, slots=True)
class CalibrationSample:
    observation: np.ndarray
    raw_advantage: float
    target_residual: float


@dataclass(frozen=True, slots=True)
class CompletedCalibrationEpisode:
    task_instance_key: str
    support: tuple[Transition, ...]
    samples: tuple[CalibrationSample, ...]
    mc_gae_abs_difference_mean: float
    target_clip_fraction: float


@dataclass(slots=True)
class _PendingStep:
    entry_id: int
    observation: np.ndarray
    reward: float
    value: float
    done: bool
    is_query: bool
    raw_advantage: float | None = None


class EpisodeCalibrationMemory:
    def __init__(self, gamma, residual_alpha, max_pending_episodes, max_completed_episodes):
        if not 0 <= gamma <= 1:
            raise ValueError("gamma must be in [0, 1]")
        if residual_alpha <= 0:
            raise ValueError("residual_alpha must be positive")
        if max_pending_episodes <= 0 or max_completed_episodes <= 0:
            raise ValueError("capacities must be positive")
        self.gamma = float(gamma)
        self.residual_alpha = float(residual_alpha)
        self.max_pending_episodes = int(max_pending_episodes)
        self.max_completed_episodes = int(max_completed_episodes)
        self._pending = OrderedDict()
        self._entry_index = {}
        self._completed = deque(maxlen=self.max_completed_episodes)
        self._next_entry_id = 0

    def observe(self, task_instance_key, transition, value, is_query):
        if task_instance_key not in self._pending:
            if len(self._pending) >= self.max_pending_episodes:
                raise RuntimeError("pending episode capacity exceeded")
            self._pending[task_instance_key] = []
        numeric = np.r_[transition.observation.reshape(-1), transition.reward, value]
        if not np.isfinite(numeric).all():
            raise ValueError("episode step contains non-finite values")
        entry_id = self._next_entry_id
        self._next_entry_id += 1
        step = _PendingStep(
            entry_id=entry_id,
            observation=np.array(transition.observation, copy=True),
            reward=float(transition.reward), value=float(value),
            done=bool(transition.done), is_query=bool(is_query),
        )
        self._pending[task_instance_key].append(step)
        self._entry_index[entry_id] = (task_instance_key, step)
        return entry_id

    def attach_rollout(self, entry_ids, raw_advantages):
        for entry_id, raw_advantage in zip(entry_ids, raw_advantages, strict=True):
            if int(entry_id) < 0:
                continue
            if int(entry_id) not in self._entry_index:
                raise KeyError(f"unknown calibration entry id: {entry_id}")
            _, step = self._entry_index[int(entry_id)]
            if step.raw_advantage is not None:
                raise ValueError(f"duplicate GAE attachment: {entry_id}")
            if not np.isfinite(raw_advantage):
                raise ValueError("raw advantage must be finite")
            step.raw_advantage = float(raw_advantage)

    def ready_task_keys(self):
        return tuple(
            key for key, steps in self._pending.items()
            if steps and steps[-1].done and all(step.raw_advantage is not None for step in steps)
        )

    def finalize(self, task_instance_key, support):
        if task_instance_key not in self.ready_task_keys():
            raise RuntimeError("episode is not ready for finalization")
        steps = self._pending.pop(task_instance_key)
        returns = np.empty(len(steps), dtype=np.float64)
        running = 0.0
        for index in range(len(steps) - 1, -1, -1):
            running = steps[index].reward + self.gamma * running
            returns[index] = running
        query_differences, samples, clipped = [], [], 0
        for step, episode_return in zip(steps, returns, strict=True):
            difference = float(episode_return - step.value - step.raw_advantage)
            if step.is_query:
                target = float(np.clip(
                    difference, -self.residual_alpha, self.residual_alpha
                ))
                query_differences.append(abs(difference))
                clipped += int(target != difference)
                samples.append(CalibrationSample(step.observation, step.raw_advantage, target))
            self._entry_index.pop(step.entry_id)
        episode = CompletedCalibrationEpisode(
            task_instance_key, tuple(support), tuple(samples),
            float(np.mean(query_differences)) if query_differences else 0.0,
            float(clipped / len(samples)) if samples else 0.0,
        )
        if samples:
            self._completed.append(episode)
        return episode

    def pop_completed(self, minimum_query_samples, maximum_query_samples):
        selected, count = [], 0
        while self._completed and count < maximum_query_samples:
            episode = self._completed[0]
            if selected and count + len(episode.samples) > maximum_query_samples:
                break
            selected.append(self._completed.popleft())
            count += len(episode.samples)
        if count < minimum_query_samples:
            self._completed.extendleft(reversed(selected))
            return ()
        return tuple(selected)

    @property
    def completed_episode_count(self):
        return len(self._completed)

    @property
    def completed_query_sample_count(self):
        return sum(len(episode.samples) for episode in self._completed)
```

`CalibrationSample.__post_init__()` must replace `observation` with a copied array whose `writeable` flag is false, cast both scalar fields to `float`, and reject non-finite values with `ValueError("calibration sample contains non-finite values")`.

- [ ] **Step 4: Run focused tests and add edge cases**

Add these exact edge assertions before running the suite:

```python
def test_duplicate_and_unknown_gae_attachments_are_rejected():
    memory = EpisodeCalibrationMemory(0.9, 1.0, 2, 2)
    entry_id = memory.observe("task-a", transition(1.0), 0.0, False)
    memory.attach_rollout([entry_id], [0.2])
    with pytest.raises(ValueError, match="duplicate GAE attachment"):
        memory.attach_rollout([entry_id], [0.3])
    with pytest.raises(KeyError, match="unknown calibration entry id"):
        memory.attach_rollout([999], [0.1])


def test_nonfinite_step_is_rejected():
    memory = EpisodeCalibrationMemory(0.9, 1.0, 2, 2)
    with pytest.raises(ValueError, match="non-finite"):
        memory.observe("task-a", transition(np.nan), 0.0, True)


def test_completed_queue_evicts_oldest_episode():
    memory = EpisodeCalibrationMemory(0.9, 1.0, 3, 1)
    for key, reward in (("task-a", 1.0), ("task-b", 2.0)):
        entry_id = memory.observe(key, transition(reward, done=True), 0.0, True)
        memory.attach_rollout([entry_id], [0.0])
        memory.finalize(key, (transition(reward),))
    episodes = memory.pop_completed(1, 10)
    assert [episode.task_instance_key for episode in episodes] == ["task-b"]


def test_clip_fraction_counts_query_rows_only():
    memory = EpisodeCalibrationMemory(0.0, 1.0, 1, 2)
    support_id = memory.observe("task-a", transition(100.0), 0.0, False)
    query_id = memory.observe("task-a", transition(2.0, done=True), 0.0, True)
    memory.attach_rollout([support_id, query_id], [0.0, 0.0])
    episode = memory.finalize("task-a", (transition(100.0),))
    assert episode.target_clip_fraction == 1.0
    assert episode.mc_gae_abs_difference_mean == 2.0
```

Then run:

```powershell
python -m pytest tests\agri_metarl\test_episode_calibration.py -v
```

Expected: all calibration-memory tests pass.

- [ ] **Step 5: Diff checkpoint**

Run `git diff --check -- src/gl_gym/RL/agri_metarl/calibration.py tests/agri_metarl/test_episode_calibration.py`. Do not stage or commit.

### Task 2: Rollout Buffer Calibration Entry IDs

**Files:**
- Modify: `src/gl_gym/RL/agri_metarl/buffer.py`
- Modify: `tests/agri_metarl/test_rollout_integration.py`

- [ ] **Step 1: Extend the existing buffer test first**

Change `add_row()` to pass `calibration_entry_ids`, and assert exact storage and reset behavior:

```python
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


def test_buffer_stores_calibration_entry_ids_and_reset_clears_them():
    buffer = make_buffer()
    add_row(buffer, ["task-a", "task-b"], [False, True], [7, 11])
    assert buffer.calibration_entry_ids[0].tolist() == [7, 11]
    buffer.reset()
    assert np.all(buffer.calibration_entry_ids == -1)
```

- [ ] **Step 2: Verify red state**

Run the named test. Expected: FAIL because `add()` rejects `calibration_entry_ids` or the field is absent.

- [ ] **Step 3: Add and validate the field**

In both `__init__()` and `reset()` initialize:

```python
self.calibration_entry_ids = np.full(
    (self.buffer_size, self.n_envs), -1, dtype=np.int64
)
```

Add the keyword-only `calibration_entry_ids=None` argument to `add()`. Convert to a flat `int64` array, require one value per environment, and store at `row_index`. Preserve the current task-key and query-mask behavior.

- [ ] **Step 4: Run buffer and memory suites**

```powershell
python -m pytest tests\agri_metarl\test_rollout_integration.py tests\agri_metarl\test_episode_calibration.py -v
```

Expected: all tests pass.

### Task 3: Attach Rollout GAE and Finalize Ready Episodes

**Files:**
- Modify: `src/gl_gym/RL/agri_metarl/agri_metarl.py`
- Modify: `src/gl_gym/RL/agri_metarl/diagnostics.py`
- Modify: `tests/agri_metarl/test_rollout_integration.py`

- [ ] **Step 1: Write a failing algorithm lifecycle test**

Construct an `AgriMetaRL` instance with `__new__`, real support/calibration memories, and a real minimal rollout buffer. Feed three transitions for one task across simulated rollout resets. After each buffer computes or is assigned advantages, call the wished-for `_attach_rollout_calibration()` method. Assert no completed episode after fragments one and two, and one completed episode after the terminal third fragment.

The test must also assert that the stored `target_residual` changes when a future terminal reward changes, while `_observe_transitions()` returns identical query flags and entry IDs before that future reward occurs. This proves future reward is a label only, not an inference input.

- [ ] **Step 2: Verify red state**

Run the named lifecycle test. Expected: FAIL because `_attach_rollout_calibration()` and algorithm-owned `calibration_memory` do not exist.

- [ ] **Step 3: Initialize calibration memory and config arguments**

Add constructor arguments:

```python
max_pending_episodes: int = 32
max_completed_episodes: int = 128
calibration_min_query_samples: int = 32
calibration_max_query_samples: int = 1024
```

In `_setup_model()` create:

```python
self.calibration_memory = EpisodeCalibrationMemory(
    gamma=self.gamma,
    residual_alpha=self.residual_alpha,
    max_pending_episodes=self.max_pending_episodes,
    max_completed_episodes=self.max_completed_episodes,
)
```

- [ ] **Step 4: Return calibration IDs during transition observation**

Change `_observe_transitions()` to return `(task_instance_keys, query_mask, calibration_entry_ids)`. Pass policy values from `collect_rollouts()` as `values.detach().cpu().numpy()`. After calling `support_memory.observe`, call:

```python
entry_id = self.calibration_memory.observe(
    task_instance_key=task_instance_keys[env_index],
    transition=transition,
    value=float(values[env_index]),
    is_query=bool(query_mask[env_index]),
)
```

Pass rollout-time policy values into `_observe_transitions()` and store returned IDs in `AgriMetaRLRolloutBuffer.add()`.

- [ ] **Step 5: Attach GAE after return computation**

First add list fields `mc_gae_abs_differences` and `target_clip_fractions`, plus integer field `completed_episode_count`, to `MetaDiagnostics`. Extend `reset()` to clear these per-window values. Task 5 adds their summary keys.

Implement:

```python
def _attach_rollout_calibration(self):
    buffer = self.rollout_buffer
    self.calibration_memory.attach_rollout(
        buffer.calibration_entry_ids.reshape(-1),
        buffer.advantages.reshape(-1),
    )
    for task_key in self.calibration_memory.ready_task_keys():
        support = self.support_memory.support(task_key)
        if len(support) != self.support_size:
            raise RuntimeError(f"missing frozen support for completed task: {task_key}")
        episode = self.calibration_memory.finalize(task_key, support)
        self.meta_diagnostics.completed_episode_count += 1
        self.meta_diagnostics.mc_gae_abs_differences.append(
            episode.mc_gae_abs_difference_mean
        )
        self.meta_diagnostics.target_clip_fractions.append(
            episode.target_clip_fraction
        )
```

Call it immediately after `compute_returns_and_advantage()` and before `callback.on_rollout_end()`.

Important: flatten using the same row-major `(step, env)` ordering for IDs and advantages. Add a two-environment test with distinct IDs to lock this invariant.

- [ ] **Step 6: Run lifecycle and complete Agri-MetaRL tests**

```powershell
python -m pytest tests\agri_metarl -q
```

Expected: all tests pass and no existing cross-task isolation assertion changes.

### Task 4: Supervised Meta Update and Safe Warm-Up

**Files:**
- Modify: `src/gl_gym/RL/agri_metarl/meta_advantage_head.py`
- Modify: `src/gl_gym/RL/agri_metarl/agri_metarl.py`
- Modify: `tests/agri_metarl/test_meta_advantage_head_v2.py`
- Modify: `tests/agri_metarl/test_rollout_integration.py`

- [ ] **Step 1: Write the zero-initialization test**

```python
def test_residual_head_starts_as_identity_correction():
    head = AdvantageResidualHead(obs_dim=6, context_dim=8, alpha=0.5)
    raw = torch.randn(4)
    corrected, residual = head(torch.randn(4, 6), raw, torch.randn(8))
    torch.testing.assert_close(residual, torch.zeros_like(residual))
    torch.testing.assert_close(corrected, raw)
```

Run it and confirm RED because the current final layer uses random initialization. Then zero the final `Linear` weight and bias with `nn.init.zeros_` and rerun to GREEN.

- [ ] **Step 2: Write deterministic supervised-update tests**

Build two `CompletedCalibrationEpisode` objects with different support transitions and finite samples. Call a wished-for `_train_calibration_batch(episodes)` twice and assert:

```python
assert second_loss < first_loss
assert any(not torch.equal(before[name], after[name]) for name in before)
```

Create a second test using `types.SimpleNamespace` to supply one sample-like object with `target_residual=np.nan` because the public immutable dataclass rejects non-finite construction. Assert the return value is `None`, `nonfinite_meta_batch_count` increments, and every encoder/head parameter remains bitwise unchanged.

- [ ] **Step 3: Verify supervised tests are red**

Expected: FAIL because `_train_calibration_batch()` does not exist.

- [ ] **Step 4: Implement tensorization and Huber training**

Implement `_train_calibration_batch()` so each episode independently encodes its immutable support snapshot, repeats that context for its query samples, and concatenates all predictions and targets:

```python
prediction_groups, target_groups = [], []
for episode in episodes:
    context = self.context_encoder(**self._tensorize_support(episode.support))
    observations = th.as_tensor(
        np.stack([sample.observation for sample in episode.samples]),
        device=self.device, dtype=th.float32,
    )
    raw_advantages = th.as_tensor(
        [sample.raw_advantage for sample in episode.samples],
        device=self.device, dtype=th.float32,
    )
    targets = th.as_tensor(
        [sample.target_residual for sample in episode.samples],
        device=self.device, dtype=th.float32,
    )
    _, predictions = self.residual_head(observations, raw_advantages, context)
    prediction_groups.append(predictions)
    target_groups.append(targets)

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
    list(self.context_encoder.parameters()) + list(self.residual_head.parameters()),
    self.max_grad_norm,
)
self.meta_optimizer.step()
return float(loss.detach().cpu())
```

The method must return early for an empty episode tuple and must never call `backward()` for a rejected batch.

- [ ] **Step 5: Separate correction and calibration training**

Refactor `_apply_meta_advantage_correction()` to do only:

- group active query rows by task-instance key;
- encode current frozen support;
- compute and write detached corrected advantages;
- record context/residual/query diagnostics.

Remove diagnostic summary logging and `reset()` from this method so calibration metrics produced later in the same `train()` call are not delayed or lost.

Remove its `meta_losses`, optimizer, backward, and optimizer-step logic. Add:

```python
def _train_completed_calibration(self):
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
```

Order `train()` as:

```python
self.policy.set_training_mode(True)
self._update_learning_rate(self.policy.optimizer)
self._apply_meta_advantage_correction()
self._train_completed_calibration()
summary = self.meta_diagnostics.summarize()
for name, value in summary.items():
    self.logger.record(name, value)
self.meta_diagnostics.last_summary = dict(summary)
self.meta_diagnostics.reset()
super().train()
```

- [ ] **Step 6: Delete dead v2 legacy code**

Delete `_legacy_apply_meta_advantage_correction()` entirely from active `agri_metarl.py`, along with unused imports and task-ID/statistic references. Do not modify `legacy_agri_metarl.py` or `agri_metarl_legacy.yml`.

- [ ] **Step 7: Run focused tests**

```powershell
python -m pytest tests\agri_metarl\test_meta_advantage_head_v2.py tests\agri_metarl\test_rollout_integration.py -v
```

Expected: zero initialization, loss descent, parameter mutation, non-finite rejection, and existing correction tests all pass.

### Task 5: Diagnostics, Configuration, and Exports

**Files:**
- Modify: `src/gl_gym/RL/agri_metarl/diagnostics.py`
- Modify: `src/gl_gym/RL/agri_metarl/__init__.py`
- Modify: `configs/agents/agri_metarl.yml`
- Modify: `tests/agri_metarl/test_rollout_integration.py`

- [ ] **Step 1: Write failing diagnostic summary test**

Populate one `MetaDiagnostics` instance and require these exact keys and finite values:

```python
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
assert set(diagnostics.summarize()) == expected
assert all(np.isfinite(value) for value in diagnostics.summarize().values())
```

- [ ] **Step 2: Add finite-safe fields**

The MC/GAE, clipping, and completion fields already exist from Task 3. Add integer fields `calibration_queue_size` and `nonfinite_meta_batch_count`, plus `last_summary: dict[str, float]`. Summaries use zero for empty lists. `reset()` clears per-window values and counters but preserves `last_summary`.

Before summarizing during training, assign:

```python
self.meta_diagnostics.calibration_queue_size = (
    self.calibration_memory.completed_query_sample_count
)
```

- [ ] **Step 3: Update active v2 configuration**

Add under the meta settings:

```yaml
  max_pending_episodes: 32
  max_completed_episodes: 128
  calibration_min_query_samples: 32
  calibration_max_query_samples: 1024
```

Do not change `configs/agents/agri_metarl_legacy.yml`.

- [ ] **Step 4: Export public calibration types**

Export `CalibrationSample`, `CompletedCalibrationEpisode`, and `EpisodeCalibrationMemory` from `gl_gym.RL.agri_metarl.__init__`. Preserve explicit `AgriMetaRL` and `LegacyAgriMetaRL` exports.

- [ ] **Step 5: Run diagnostics/config tests**

Run all `tests/agri_metarl/`. Expected: all pass.

### Task 6: CPU Lifecycle Smoke Test and Repository Verification

**Files:**
- Modify: `tests/agri_metarl/test_rollout_integration.py`
- Modify: `docs/superpowers/specs/2026-06-25-agrimetarl2-monte-carlo-calibration-design.md`

- [ ] **Step 1: Add a real three-rollout CPU smoke test**

Use a tiny Gymnasium environment with episode length six and `n_steps=2`. Configure `support_size=2`, small LSTM/network dimensions, and `calibration_min_query_samples=1`. Train for at least eight timesteps so one complete episode is finalized and a later rollout can consume its calibration episode.

Assertions:

```python
assert model.num_timesteps >= 8
assert model.meta_diagnostics.last_summary["train/nonfinite_meta_batch_count"] == 0
assert all(torch.isfinite(parameter).all() for parameter in model.context_encoder.parameters())
assert all(torch.isfinite(parameter).all() for parameter in model.residual_head.parameters())
assert model._n_updates > 0
```

Use the `last_summary` field added in Task 5. The smoke test also asserts:

```python
assert model.meta_diagnostics.last_summary["train/completed_episode_count"] >= 1
assert model.meta_diagnostics.last_summary["train/meta_loss"] >= 0
```

- [ ] **Step 2: Run the smoke test independently**

```powershell
python -m pytest tests\agri_metarl\test_rollout_integration.py -k "smoke" -v
```

Expected: PASS on CPU with no NaN/Inf.

- [ ] **Step 3: Run the full repository suite**

```powershell
python -m pytest -q
```

Expected: all tests pass. Existing deprecation and `PytestReturnNotNoneWarning` warnings may remain; no new warnings are allowed.

- [ ] **Step 4: Run integrity and diff checks**

```powershell
python -m pytest tests\integrity\test_no_result_scaling.py -v
git diff --check
rg -n "_legacy_apply_meta_advantage_correction|normalize_and_clip" src\gl_gym\RL\agri_metarl\agri_metarl.py
```

Expected: integrity test passes, `git diff --check` exits zero, and `rg` returns no matches in the active v2 implementation.

- [ ] **Step 5: Record implementation status without claiming Pilot evidence**

Append a short status section to the design spec listing test commands and results. State explicitly that mechanism verification is complete but performance claims remain blocked until the three-seed Pilot.

- [ ] **Step 6: Final local checkpoint**

Run:

```powershell
git diff --stat
git status --short
```

Report changed files, exact test counts, warnings, and remaining Pilot work. Do not stage, commit, push, or rewrite the manuscript from unvalidated results.
