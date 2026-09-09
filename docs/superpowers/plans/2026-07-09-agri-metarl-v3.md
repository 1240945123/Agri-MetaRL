# AgriMetaRL-v3 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make AgriMetaRL a context-conditioned recurrent policy with bounded, constraint-aware meta advantage calibration.

**Architecture:** Reuse the proven context-augmentation pattern from `ContextRecurrentPPO`, but keep AgriMetaRL’s calibration memory and residual head as an auxiliary training signal. The raw greenhouse observation is augmented with support-set context for policy/value learning; calibration samples also carry constraint penalties so residual targets discourage high-violation behavior.

**Tech Stack:** Python, PyTorch, Stable-Baselines3, sb3-contrib `RecurrentPPO`, Gymnasium spaces, pytest.

---

## File Structure

- Modify `src/gl_gym/RL/agri_metarl/agri_metarl.py`
  - Add raw/augmented observation-space handling.
  - Add context augmentation helpers.
  - Make rollout collection use augmented observations for policy/value calls.
  - Keep residual calibration training.
  - Add save/load support for raw observation spaces.
- Modify `src/gl_gym/RL/agri_metarl/buffer.py`
  - Store support snapshots and support sizes in `AgriMetaRLRolloutBuffer`.
  - Return support snapshots in rollout samples.
- Modify `src/gl_gym/RL/agri_metarl/calibration.py`
  - Add optional constraint penalty to pending steps and calibration samples.
  - Include penalty in residual targets.
- Modify `configs/agents/agri_metarl.yml`
  - Add conservative constraint residual weights.
- Modify `tests/agri_metarl/test_rollout_integration.py`
  - Add behavior tests for context-conditioned AgriMetaRL.
- Modify `tests/agri_metarl/test_episode_calibration.py`
  - Add constraint-aware residual target tests.
- Modify `tests/agri_metarl/test_context_recurrent_ppo.py` only if shared helpers are extracted.

---

### Task 1: Add AgriMetaRL context-conditioned observation space

**Files:**
- Modify: `src/gl_gym/RL/agri_metarl/agri_metarl.py`
- Test: `tests/agri_metarl/test_rollout_integration.py`

- [x] **Step 1: Write failing tests**

Add tests near the existing AgriMetaRL smoke tests:

```python
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
```

```python
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
    assert state is None
```

- [x] **Step 2: Run tests and verify red**

Run:

```powershell
python -m pytest tests\agri_metarl\test_rollout_integration.py::test_agri_metarl_augments_policy_observation_space_without_mutating_env_space tests\agri_metarl\test_rollout_integration.py::test_agri_metarl_predict_accepts_raw_observation -q
```

Expected: failures because `AgriMetaRL` does not yet expose `raw_observation_space`, does not augment `observation_space`, and `predict()` does not pad context.

- [x] **Step 3: Implement minimal context-space support**

In `src/gl_gym/RL/agri_metarl/agri_metarl.py`, add:

```python
from pathlib import Path
from stable_baselines3.common.save_util import load_from_zip_file
```

Add a `load()` override and helper methods patterned after `ContextRecurrentPPO`:

```python
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
def _augmented_observation_space(observation_space: spaces.Space, context_dim: int) -> spaces.Box:
    if not isinstance(observation_space, spaces.Box) or len(observation_space.shape) != 1:
        raise TypeError("AgriMetaRL requires a flat Box observation space")
    if not np.issubdtype(observation_space.dtype, np.floating):
        raise TypeError("AgriMetaRL requires a floating-point Box observation space")
    context_low = np.full(context_dim, -np.inf, dtype=observation_space.dtype)
    context_high = np.full(context_dim, np.inf, dtype=observation_space.dtype)
    low = np.concatenate([np.array(observation_space.low, copy=True).reshape(-1), context_low])
    high = np.concatenate([np.array(observation_space.high, copy=True).reshape(-1), context_high])
    return spaces.Box(low=low.astype(observation_space.dtype), high=high.astype(observation_space.dtype), dtype=observation_space.dtype)
```

At the start of `_setup_model()`:

```python
self.raw_observation_space = getattr(self, "raw_observation_space", self.observation_space)
self.observation_space = self._augmented_observation_space(
    self.raw_observation_space,
    self.context_dim,
)
```

Use `self.raw_observation_space.shape[0]` for `obs_dim` when constructing `TransitionSetEncoder` and `AdvantageResidualHead`.

Add:

```python
def _get_torch_save_params(self):
    state_dicts, torch_variables = super()._get_torch_save_params()
    return [*state_dicts, "context_encoder", "residual_head"], torch_variables

def predict(self, observation, state=None, episode_start=None, deterministic: bool = False):
    observation_array = np.asarray(observation)
    raw_dim = int(self.raw_observation_space.shape[0])
    augmented_dim = raw_dim + self.context_dim
    if observation_array.shape == (raw_dim,):
        context = np.zeros(self.context_dim, dtype=observation_array.dtype)
        observation = np.concatenate([observation_array, context], axis=0)
    elif observation_array.shape == (augmented_dim,):
        observation = observation_array
    elif observation_array.ndim >= 2 and observation_array.shape[-1] == raw_dim:
        context = np.zeros((*observation_array.shape[:-1], self.context_dim), dtype=observation_array.dtype)
        observation = np.concatenate([observation_array, context], axis=-1)
    return self.policy.predict(
        observation,
        state=state,
        episode_start=episode_start,
        deterministic=deterministic,
    )
```

- [x] **Step 4: Verify green**

Run the same two tests. Expected: PASS.

---

### Task 2: Store support snapshots in AgriMetaRL rollout buffer

**Files:**
- Modify: `src/gl_gym/RL/agri_metarl/buffer.py`
- Modify: `src/gl_gym/RL/agri_metarl/agri_metarl.py`
- Test: `tests/agri_metarl/test_rollout_integration.py`

- [x] **Step 1: Write failing tests**

Add:

```python
def test_agri_metarl_rollout_buffer_stores_support_snapshots():
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

    model.learn(total_timesteps=2)

    assert hasattr(model.rollout_buffer, "support_snapshots")
    assert hasattr(model.rollout_buffer, "support_sizes")
    assert model.rollout_buffer.support_sizes.shape == (2, 1)
    assert isinstance(model.rollout_buffer.support_snapshots[0, 0], tuple)
```

- [x] **Step 2: Run test and verify red**

Run:

```powershell
python -m pytest tests\agri_metarl\test_rollout_integration.py::test_agri_metarl_rollout_buffer_stores_support_snapshots -q
```

Expected: FAIL because `AgriMetaRLRolloutBuffer` does not yet store support snapshots.

- [x] **Step 3: Implement buffer metadata**

In `src/gl_gym/RL/agri_metarl/buffer.py`, extend `AgriMetaRLRolloutBuffer.reset()`:

```python
self.support_snapshots = np.empty((self.buffer_size, self.n_envs), dtype=object)
self.support_snapshots[:, :] = ()
self.support_sizes = np.zeros((self.buffer_size, self.n_envs), dtype=np.int64)
```

Extend `add()` signature:

```python
support_snapshots=None,
support_sizes=None,
```

Inside `add()`, before calling `super().add(...)`, store:

```python
if support_snapshots is not None:
    if len(support_snapshots) != self.n_envs:
        raise ValueError("support_snapshots must contain one snapshot per environment")
    self.support_snapshots[row_index, :] = tuple(tuple(snapshot) for snapshot in support_snapshots)
if support_sizes is not None:
    sizes = np.asarray(support_sizes, dtype=np.int64).reshape(-1)
    if sizes.shape[0] != self.n_envs:
        raise ValueError("support_sizes must contain one size per environment")
    self.support_sizes[row_index, :] = sizes
```

If `get()` uses a custom sample dataclass, include `support_snapshots` in samples. If it inherits unchanged from sb3-contrib and does not expose custom sample attributes, add a small wrapper following the existing context buffer pattern in `src/gl_gym/RL/context_recurrent_ppo_buffer.py`.

- [x] **Step 4: Pass metadata from collect_rollouts**

Before policy action selection in `AgriMetaRL.collect_rollouts()`, compute support snapshots for the current task keys. After `_observe_transitions()`, pass `support_snapshots` and `support_sizes` to `rollout_buffer.add(...)`.

- [x] **Step 5: Verify green**

Run:

```powershell
python -m pytest tests\agri_metarl\test_rollout_integration.py::test_agri_metarl_rollout_buffer_stores_support_snapshots -q
```

Expected: PASS.

---

### Task 3: Use context-augmented observations during AgriMetaRL rollout and training

**Files:**
- Modify: `src/gl_gym/RL/agri_metarl/agri_metarl.py`
- Test: `tests/agri_metarl/test_rollout_integration.py`

- [x] **Step 1: Write failing tests**

Add:

```python
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

    assert model.rollout_buffer.observations.shape[-1] == model.raw_observation_space.shape[0] + 2
    assert model.rollout_buffer.query_mask.any()
```

- [x] **Step 2: Run test and verify red**

Run:

```powershell
python -m pytest tests\agri_metarl\test_rollout_integration.py::test_agri_metarl_policy_receives_augmented_observations_during_learning -q
```

Expected: FAIL because rollout observations are still raw.

- [x] **Step 3: Implement context helpers**

Add to `AgriMetaRL`:

```python
def _context_from_support(self, support: tuple[Transition, ...]) -> tuple[th.Tensor, bool]:
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

def _augment_raw_observations(self, raw_observations: np.ndarray, support_snapshots) -> np.ndarray:
    raw_observations = np.asarray(raw_observations, dtype=np.float32)
    contexts = []
    with th.no_grad():
        for support in support_snapshots:
            context, _ = self._context_from_support(tuple(support))
            contexts.append(context.detach().cpu().numpy().astype(np.float32))
    context_array = np.asarray(contexts, dtype=np.float32).reshape(raw_observations.shape[0], self.context_dim)
    return np.concatenate([raw_observations, context_array], axis=1).astype(np.float32, copy=False)

def _augment_training_observations(self, augmented_or_raw_observations: th.Tensor, support_snapshots) -> th.Tensor:
    observations = augmented_or_raw_observations.to(self.device)
    raw_dim = int(self.raw_observation_space.shape[0])
    raw_observations = observations[..., :raw_dim].float()
    contexts = []
    for support in support_snapshots:
        support = tuple(support)
        if len(support) < self.support_size:
            contexts.append(th.zeros(self.context_dim, device=self.device, dtype=raw_observations.dtype))
        else:
            contexts.append(self.context_encoder(**self._tensorize_support(support)).to(dtype=raw_observations.dtype))
    context_tensor = th.stack(contexts, dim=0).reshape(raw_observations.shape[0], self.context_dim)
    return th.cat([raw_observations, context_tensor], dim=-1)
```

- [x] **Step 4: Use augmented observations**

In `collect_rollouts()`, keep `raw_observations = np.asarray(self._last_obs, dtype=np.float32)` for transition storage, but use:

```python
support_snapshots = self._support_snapshots_for_task_keys(current_task_keys)
augmented_obs = self._augment_raw_observations(raw_observations, support_snapshots)
obs_tensor = obs_as_tensor(augmented_obs, self.device)
```

Store `augmented_obs` in the rollout buffer, while `_observe_transitions()` still receives raw observations and raw next observations.

In `_apply_meta_advantage_correction()`, pass raw observations to residual head:

```python
raw_dim = int(self.raw_observation_space.shape[0])
observations = th.as_tensor(buffer.observations[rows, envs][..., :raw_dim], device=self.device, dtype=th.float32)
```

In `train()`, before `self.policy.evaluate_actions(...)`, if overriding the full RecurrentPPO training loop is not currently present, replace the call path by copying the necessary train loop from `ContextRecurrentPPO.train()` and inserting:

```python
observations = self._augment_training_observations(
    rollout_data.observations,
    rollout_data.support_snapshots,
)
```

Keep `_apply_meta_advantage_correction()` and `_train_completed_calibration()` before PPO loss computation.

- [x] **Step 5: Verify green**

Run:

```powershell
python -m pytest tests\agri_metarl\test_rollout_integration.py::test_agri_metarl_policy_receives_augmented_observations_during_learning -q
```

Expected: PASS.

---

### Task 4: Add constraint-aware calibration targets

**Files:**
- Modify: `src/gl_gym/RL/agri_metarl/calibration.py`
- Modify: `src/gl_gym/RL/agri_metarl/agri_metarl.py`
- Modify: `configs/agents/agri_metarl.yml`
- Test: `tests/agri_metarl/test_episode_calibration.py`

- [x] **Step 1: Write failing calibration test**

Add:

```python
def test_constraint_penalty_reduces_query_residual_target():
    memory = EpisodeCalibrationMemory(
        gamma=0.0,
        residual_alpha=2.0,
        max_pending_episodes=1,
        max_completed_episodes=2,
        constraint_penalty_weight=0.5,
    )
    entry_id = memory.observe(
        "task-a",
        transition(1.0, done=True),
        value=0.0,
        is_query=True,
        constraint_penalty=2.0,
    )
    memory.attach_rollout([entry_id], [0.0])

    episode = memory.finalize("task-a", (transition(1.0),))

    assert episode.samples[0].target_residual == 0.0
```

Explanation: without the penalty, target would be `1.0`; with `0.5 * 2.0`, it becomes `0.0`.

- [x] **Step 2: Run test and verify red**

Run:

```powershell
python -m pytest tests\agri_metarl\test_episode_calibration.py::test_constraint_penalty_reduces_query_residual_target -q
```

Expected: FAIL because `constraint_penalty_weight` and `constraint_penalty` do not exist.

- [x] **Step 3: Implement calibration penalty**

In `_PendingStep`, add:

```python
constraint_penalty: float = 0.0
```

In `EpisodeCalibrationMemory.__init__()`, add:

```python
constraint_penalty_weight: float = 0.0,
```

Validate non-negative:

```python
if constraint_penalty_weight < 0:
    raise ValueError("constraint_penalty_weight must be non-negative")
self.constraint_penalty_weight = float(constraint_penalty_weight)
```

In `observe()`, add:

```python
constraint_penalty: float = 0.0,
```

Validate finite and store it.

In `finalize()`, modify:

```python
difference = float(episode_return - step.value - step.raw_advantage)
penalized_difference = difference - self.constraint_penalty_weight * step.constraint_penalty
target = float(np.clip(penalized_difference, -self.residual_alpha, self.residual_alpha))
```

- [x] **Step 4: Compute penalty from environment info**

In `AgriMetaRL.__init__()`, add:

```python
constraint_penalty_weight: float = 0.0,
temp_violation_weight: float = 0.0,
co2_violation_weight: float = 0.0,
rh_violation_weight: float = 0.0,
```

Store them on `self`.

When constructing `EpisodeCalibrationMemory`, pass `constraint_penalty_weight=self.constraint_penalty_weight`.

Add helper:

```python
def _constraint_penalty_from_info(self, info: dict[str, Any]) -> float:
    return float(
        self.temp_violation_weight * float(info.get("temp_violation", 0.0))
        + self.co2_violation_weight * float(info.get("co2_violation", 0.0))
        + self.rh_violation_weight * float(info.get("rh_violation", 0.0))
    )
```

When calling `calibration_memory.observe(...)`, pass:

```python
constraint_penalty=self._constraint_penalty_from_info(info)
```

- [x] **Step 5: Add conservative config defaults**

In `configs/agents/agri_metarl.yml`, add:

```yaml
  constraint_penalty_weight: 0.05
  temp_violation_weight: 0.001
  co2_violation_weight: 0.0001
  rh_violation_weight: 0.0005
```

- [x] **Step 6: Verify green**

Run:

```powershell
python -m pytest tests\agri_metarl\test_episode_calibration.py -q
```

Expected: PASS.

---

### Task 5: Run focused integration and smoke training

**Files:**
- No production-code changes unless tests expose a defect.

- [x] **Step 1: Run AgriMetaRL tests**

Run:

```powershell
python -m pytest tests\agri_metarl -q
```

Expected: all tests pass.

- [x] **Step 2: Run experiment CLI tests**

Run:

```powershell
python -m pytest tests\experiments\test_suite_training_cli.py tests\experiments\test_suite_evaluation.py -q
```

Expected: all tests pass.

- [x] **Step 3: Run compile check**

Run:

```powershell
python -m compileall -q src tests experiments
```

Expected: exit code 0.

- [x] **Step 4: Run a small AgriMetaRL smoke suite**

Run:

```powershell
$env:WANDB_MODE = "disabled"
python experiments\scripts\run_suite_training.py --manifest artifacts\results\AgriControl_C_2026-06-30\suite_manifest.json --algorithms agri_metarl --seeds 42 --device cpu --train_timesteps 65536 --n_envs 1 --n_steps 64 --batch_size 64 --n_epochs 1 --n_evals 1
```

Expected: run completes and writes a completed `agri_metarl seed=42` record for the smoke configuration. If this would overwrite the formal 2M run registry, create a separate smoke manifest or run only through a temporary result root before executing this exact command.

---

### Task 6: Run pilot experiment for method selection

**Files:**
- Generated artifacts under `artifacts/results/AgriControl_C_2026-06-30/`
- Generated logs under `artifacts/logs/`

- [ ] **Step 1: Train 2-seed AgriMetaRL-v3 pilot**

Run:

```powershell
$env:WANDB_MODE = "disabled"
python experiments\scripts\run_suite_training.py --manifest artifacts\results\AgriControl_C_2026-06-30\suite_manifest.json --algorithms agri_metarl --seeds 42 123 --device cpu
```

Expected: both AgriMetaRL runs complete at 2,000,000 steps.

- [ ] **Step 2: Remove stale AgriMetaRL eval rows before re-evaluation**

Run:

```powershell
$resultRoot = "artifacts\results\AgriControl_C_2026-06-30"
$eval = Join-Path $resultRoot "eval_raw.csv"
$backup = Join-Path $resultRoot ("eval_raw.before_agri_v3_" + (Get-Date -Format "yyyyMMdd_HHmmss") + ".csv")
Copy-Item -LiteralPath $eval -Destination $backup
$rows = Import-Csv $eval | Where-Object { $_.algorithm -ne "agri_metarl" }
$rows | Export-Csv $eval -NoTypeInformation
```

Expected: old AgriMetaRL rows are backed up and removed.

- [ ] **Step 3: Re-evaluate AgriMetaRL only**

Run:

```powershell
python experiments\scripts\evaluate_suite.py --manifest artifacts\results\AgriControl_C_2026-06-30\suite_manifest.json --runs_csv artifacts\results\AgriControl_C_2026-06-30\runs.csv --tasks_csv artifacts\results\AgriControl_C_2026-06-30\eval_tasks.csv --algorithms agri_metarl --seeds 42 123 --resume_eval
```

Expected: 182 AgriMetaRL rows are written.

- [ ] **Step 4: Summarize**

Run:

```powershell
python experiments\scripts\summarize_suite.py --manifest artifacts\results\AgriControl_C_2026-06-30\suite_manifest.json
```

Expected: `method_summary.csv`, `stat_tests.csv`, and `diagnostics.csv` update.

- [ ] **Step 5: Decide whether to expand to 5 seeds**

Use this Python summary:

```powershell
@'
import pandas as pd
root='artifacts/results/AgriControl_C_2026-06-30'
df=pd.read_csv(f'{root}/eval_raw.csv')
metrics=['episode_return','EPI','revenue','temp_violation','co2_violation','rh_violation']
print(df.groupby(['algorithm','split'])[metrics].mean().round(3).to_string())
'@ | python -
```

Expand to 5 seeds only if AgriMetaRL-v3 improves heldout, uncertainty, or economic return versus AgriMetaRL-v2 and reduces CO₂/RH violations.

---

## Self-Review Notes

- Spec coverage: context-conditioned policy, auxiliary calibration, bounded memory, constraint-aware residual, and pilot evaluation are each mapped to tasks.
- Placeholder scan: no `TBD`/`TODO` placeholders remain.
- Type consistency: names used in plan match current code conventions: `AgriMetaRL`, `EpisodeCalibrationMemory`, `AgriMetaRLRolloutBuffer`, `support_snapshots`, `support_sizes`, `calibration_max_queue_samples`.
- Scope check: this plan is focused on AgriMetaRL-v3 only; 5-seed formal training is explicitly gated on pilot results.
