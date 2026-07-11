# Online Context Evaluation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add leakage-safe online context inference for AgriMetaRL-v3 and run a same-checkpoint, two-mode diagnostic that decides whether further training is justified.

**Architecture:** Keep online inference state inside `AgriMetaRL`, separate from all training memories. Extend the generic deterministic evaluator with optional capability-based lifecycle hooks, then add a focused diagnostic module and CLI that evaluate fixed task IDs in `online_context` and `zero_context` modes, save action traces, compute paired deltas, and apply the approved research gate.

**Tech Stack:** Python 3.11, NumPy, pandas, PyTorch, Gymnasium, Stable-Baselines3/sb3-contrib, pytest.

---

## File Structure

- Modify `src/gl_gym/RL/agri_metarl/agri_metarl.py`
  - Own episode-scoped inference memory, mode, task key, and context diagnostics.
  - Augment raw observations with online context only when explicitly enabled.
- Modify `src/gl_gym/experiments/suite_evaluation.py`
  - Invoke optional inference lifecycle hooks around one deterministic episode.
  - Return action/context diagnostics only when requested; preserve the current return contract by default.
- Create `src/gl_gym/experiments/context_ab.py`
  - Define the fixed task IDs, diagnostic row schema, paired aggregation, gate evaluation, and artifact writers.
- Create `experiments/scripts/run_context_ab.py`
  - Load the approved checkpoints and tasks, run both inference modes, save action traces and tabular artifacts.
- Modify `tests/agri_metarl/test_rollout_integration.py`
  - Verify isolation, readiness, online augmentation, zero-context compatibility, validation, and cleanup.
- Modify `tests/experiments/test_suite_evaluation.py`
  - Verify hook order, terminal observations, exception cleanup, and compatibility with ordinary models.
- Create `tests/experiments/test_context_ab.py`
  - Verify task selection, paired deltas, action-trace comparison, gate outcomes, and output schemas.

## Task 1: Add an isolated inference lifecycle to AgriMetaRL

**Files:**
- Modify: `src/gl_gym/RL/agri_metarl/agri_metarl.py:240-285`
- Test: `tests/agri_metarl/test_rollout_integration.py`

- [ ] **Step 1: Write failing lifecycle tests**

Append a `_tiny_agri_model()` factory and these tests:

```python
def _tiny_agri_model(*, support_size=2):
    return AgriMetaRL(
        "MlpLstmPolicy",
        TinyTaskEnv(),
        n_steps=2,
        batch_size=2,
        n_epochs=1,
        support_size=support_size,
        max_task_instances=4,
        context_dim=3,
        transition_hidden_dim=8,
        policy_kwargs={"net_arch": {"pi": [8], "vf": [8]}, "lstm_hidden_size": 8},
        seed=0,
        verbose=0,
        device="cpu",
    )


def test_inference_episode_uses_fresh_memory_separate_from_training_memory():
    model = _tiny_agri_model(support_size=1)
    training_memory = model.support_memory
    model.begin_inference_episode("online_context")

    assert model._inference_support_memory is not training_memory
    assert model._inference_task_key is None
    assert model._inference_mode == "online_context"

    model.end_inference_episode()
    assert model._inference_support_memory is None
    assert model._inference_task_key is None
    assert model._inference_mode is None


def test_inference_episode_rejects_unknown_mode():
    model = _tiny_agri_model()
    with pytest.raises(ValueError, match="inference mode"):
        model.begin_inference_episode("adaptive_magic")
```

- [ ] **Step 2: Run tests and verify red**

Run:

```powershell
python -m pytest tests\agri_metarl\test_rollout_integration.py::test_inference_episode_uses_fresh_memory_separate_from_training_memory tests\agri_metarl\test_rollout_integration.py::test_inference_episode_rejects_unknown_mode -q
```

Expected: both fail because the lifecycle methods do not exist.

- [ ] **Step 3: Implement the minimal lifecycle**

Add near `predict()`:

```python
INFERENCE_MODES = frozenset({"online_context", "zero_context"})

def begin_inference_episode(self, mode: str) -> None:
    if mode not in self.INFERENCE_MODES:
        raise ValueError(f"unsupported inference mode: {mode!r}")
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
    self._inference_context_norms = []
    self._inference_support_ready_step = None
    self._inference_step = 0
```

Define `INFERENCE_MODES` as a class attribute and initialize the same attributes to their inactive values in `_setup_model()` so loaded and newly created models have identical state.

- [ ] **Step 4: Run lifecycle tests**

Run the Step 2 command. Expected: `2 passed`.

- [ ] **Step 5: Commit**

```powershell
git add src/gl_gym/RL/agri_metarl/agri_metarl.py tests/agri_metarl/test_rollout_integration.py
git commit -m "feat: add isolated AgriMetaRL inference lifecycle"
```

## Task 2: Update inference memory and use online context

**Files:**
- Modify: `src/gl_gym/RL/agri_metarl/agri_metarl.py:256-326`
- Test: `tests/agri_metarl/test_rollout_integration.py`

- [ ] **Step 1: Write failing transition and prediction tests**

```python
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
    model = _tiny_agri_model(support_size=1)
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


def test_online_predict_uses_context_only_after_support_is_ready(monkeypatch):
    model = _tiny_agri_model(support_size=1)
    seen = []

    def fake_policy_predict(observation, **kwargs):
        seen.append(np.asarray(observation).copy())
        return np.zeros(1, dtype=np.float32), None

    monkeypatch.setattr(model.policy, "predict", fake_policy_predict)
    model.begin_inference_episode("online_context")
    model.predict(np.zeros(3, dtype=np.float32), deterministic=True)
    model.observe_inference_transition(
        np.zeros(3, dtype=np.float32), np.zeros(1, dtype=np.float32), 1.0,
        np.ones(3, dtype=np.float32), False, _inference_info(),
    )
    monkeypatch.setattr(
        model,
        "_context_from_support",
        lambda support: (torch.ones(3, device=model.device), True),
    )
    model.predict(np.ones(3, dtype=np.float32), deterministic=True)

    np.testing.assert_array_equal(seen[0][-3:], np.zeros(3))
    np.testing.assert_array_equal(seen[1][-3:], np.ones(3))


def test_zero_context_mode_never_reads_inference_support(monkeypatch):
    model = _tiny_agri_model(support_size=1)
    model.begin_inference_episode("zero_context")
    monkeypatch.setattr(
        model,
        "_context_from_support",
        lambda support: pytest.fail("zero-context mode encoded support"),
    )
    action, _ = model.predict(np.zeros(3, dtype=np.float32), deterministic=True)
    assert action.shape == (1,)
```

Add validation tests for incomplete task identity and non-finite inputs:

```python
@pytest.mark.parametrize("bad_info", [{}, {"task_instance_key": "k"}])
def test_online_transition_requires_complete_task_identity(bad_info):
    model = _tiny_agri_model()
    model.begin_inference_episode("online_context")
    with pytest.raises(KeyError, match="task identity"):
        model.observe_inference_transition(
            np.zeros(3), np.zeros(1), 1.0, np.ones(3), False, bad_info
        )
```

- [ ] **Step 2: Run tests and verify red**

Run:

```powershell
python -m pytest tests\agri_metarl\test_rollout_integration.py -k "online_inference or online_predict or zero_context_mode or complete_task_identity" -q
```

Expected: failures because transition observation and online augmentation are absent.

- [ ] **Step 3: Implement transition observation**

```python
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
    arrays = [np.asarray(observation), np.asarray(action), np.asarray(next_observation)]
    if not all(np.isfinite(value).all() for value in arrays) or not np.isfinite(reward):
        raise ValueError("inference transition must contain only finite values")
    if "task_descriptor" not in info or "task_instance_key" not in info:
        raise KeyError("inference info must contain complete task identity")
    key = str(info["task_instance_key"])
    if self._inference_task_key not in (None, key):
        raise ValueError("task identity changed inside one inference episode")
    self._inference_task_key = key
    transition = Transition(
        observation=np.asarray(observation, dtype=np.float32),
        action=np.asarray(action, dtype=np.float32).reshape(-1),
        reward=float(reward),
        next_observation=np.asarray(next_observation, dtype=np.float32),
        done=bool(done),
    )
    self._inference_support_memory.observe(key, transition)
    self._inference_step += 1
    if (
        self._inference_support_ready_step is None
        and len(self._inference_support_memory.support(key)) >= self.support_size
    ):
        self._inference_support_ready_step = self._inference_step
```

- [ ] **Step 4: Modify raw-observation prediction**

Factor context selection into:

```python
def _inference_context(self) -> np.ndarray:
    context = np.zeros(self.context_dim, dtype=np.float32)
    if self._inference_mode != "online_context" or self._inference_task_key is None:
        return context
    support = self._inference_support_memory.support(self._inference_task_key)
    encoded, ready = self._context_from_support(support)
    if not ready:
        return context
    context = encoded.detach().cpu().numpy().astype(np.float32)
    if not np.isfinite(context).all():
        raise ValueError("inference context contains non-finite values")
    self._inference_context_norms.append(float(np.linalg.norm(context)))
    return context
```

In `predict()`, use `_inference_context()` instead of zeros for either an unbatched raw observation with shape `(raw_dim,)` or the evaluator's single-environment batch with shape `(1, raw_dim)` while an inference episode is active. Concatenate the context directly for the unbatched case and as `context.reshape(1, -1)` for the batched case. Reject online inference batches with more than one environment. Keep the current vectorized zero-padding behavior outside this explicit protocol.

Add:

```python
def inference_episode_diagnostics(self) -> dict[str, float]:
    norms = np.asarray(self._inference_context_norms, dtype=float)
    return {
        "support_ready_step": float(self._inference_support_ready_step or np.nan),
        "context_norm_mean": float(norms.mean()) if norms.size else 0.0,
        "context_norm_max": float(norms.max()) if norms.size else 0.0,
    }
```

- [ ] **Step 5: Run focused and full AgriMetaRL tests**

```powershell
python -m pytest tests\agri_metarl\test_rollout_integration.py -q
python -m pytest tests\agri_metarl -q
```

Expected: both commands pass.

- [ ] **Step 6: Commit**

```powershell
git add src/gl_gym/RL/agri_metarl/agri_metarl.py tests/agri_metarl/test_rollout_integration.py
git commit -m "feat: condition AgriMetaRL inference on online support"
```

## Task 3: Add capability-based hooks to deterministic evaluation

**Files:**
- Modify: `src/gl_gym/experiments/suite_evaluation.py:39-96`
- Modify: `tests/experiments/test_suite_evaluation.py`

- [ ] **Step 1: Write a hook-order fake and failing tests**

```python
class HookedFakeModel(FakeModel):
    def __init__(self, fail_predict=False):
        self.events = []
        self.fail_predict = fail_predict

    def begin_inference_episode(self, mode):
        self.events.append(("begin", mode))

    def predict(self, obs, deterministic=True, **kwargs):
        self.events.append(("predict", np.asarray(obs).copy()))
        if self.fail_predict:
            raise RuntimeError("predict failed")
        return np.array([[0.0]]), None

    def observe_inference_transition(self, observation, action, reward, next_observation, done, info):
        self.events.append(("observe", np.asarray(observation).copy(), np.asarray(next_observation).copy(), bool(done)))

    def inference_episode_diagnostics(self):
        return {"support_ready_step": 1.0, "context_norm_mean": 2.0, "context_norm_max": 3.0}

    def end_inference_episode(self):
        self.events.append(("end",))


def test_deterministic_episode_invokes_online_hooks_in_order():
    model = HookedFakeModel()
    metrics, diagnostics = run_deterministic_episode(
        model, FakeEnv(), inference_mode="online_context", return_diagnostics=True
    )
    assert model.events[0] == ("begin", "online_context")
    assert [event[0] for event in model.events].count("observe") == 3
    assert model.events[-1] == ("end",)
    assert metrics["episode_return"] == 30.0
    assert diagnostics["context_norm_max"] == 3.0


def test_deterministic_episode_cleans_up_when_prediction_fails():
    model = HookedFakeModel(fail_predict=True)
    with pytest.raises(RuntimeError, match="predict failed"):
        run_deterministic_episode(model, FakeEnv(), inference_mode="online_context")
    assert model.events[-1] == ("end",)


def test_plain_model_retains_existing_evaluation_contract():
    metrics = run_deterministic_episode(FakeModel(), FakeEnv())
    assert isinstance(metrics, dict)
    assert metrics["episode_return"] == 30.0
```

- [ ] **Step 2: Run tests and verify red**

```powershell
python -m pytest tests\experiments\test_suite_evaluation.py -k "hooks_in_order or cleans_up or existing_evaluation_contract" -q
```

Expected: hook tests fail because the new arguments are unsupported; compatibility test passes.

- [ ] **Step 3: Implement optional hook orchestration**

Change the signature:

```python
def run_deterministic_episode(
    model: Any,
    env: Any,
    inference_mode: str | None = None,
    return_diagnostics: bool = False,
) -> dict[str, float] | tuple[dict[str, float], dict[str, Any]]:
```

Before the loop, require all four hooks when `inference_mode` is not `None`, then call `begin_inference_episode(inference_mode)`. Inside the loop, preserve `previous_obs = obs` before prediction and pass the executed transition after `env.step()`. Use `info.get("terminal_observation", obs[0])` when `done`, otherwise `obs[0]`. Wrap the whole episode in `try/finally`; capture diagnostics immediately before leaving `try`, and always call `end_inference_episode()` in `finally`.

Collect executed actions into `action_trace = []`, append `np.asarray(actions[0], dtype=np.float32).copy()` each step, and include `np.stack(action_trace)` in the diagnostics dictionary when requested.

Raise:

```python
missing = [name for name in required_hooks if not callable(getattr(model, name, None))]
if missing:
    raise TypeError(f"inference mode requires model hooks: {missing}")
```

- [ ] **Step 4: Run evaluator tests**

```powershell
python -m pytest tests\experiments\test_suite_evaluation.py -q
```

Expected: all tests pass.

- [ ] **Step 5: Commit**

```powershell
git add src/gl_gym/experiments/suite_evaluation.py tests/experiments/test_suite_evaluation.py
git commit -m "feat: support online inference hooks in suite evaluation"
```

## Task 4: Implement diagnostic records, pairing, and decision gate

**Files:**
- Create: `src/gl_gym/experiments/context_ab.py`
- Create: `tests/experiments/test_context_ab.py`

- [ ] **Step 1: Write failing task-selection and gate tests**

Define the approved constant in the test and assert exact selection:

```python
from gl_gym.experiments.context_ab import (
    DIAGNOSTIC_TASK_IDS,
    build_paired_deltas,
    evaluate_context_gate,
    select_diagnostic_tasks,
)


def test_select_diagnostic_tasks_requires_all_approved_ids():
    tasks = pd.DataFrame({"task_id": list(DIAGNOSTIC_TASK_IDS), "split": ["fixed"] * 8})
    selected = select_diagnostic_tasks(tasks)
    assert selected["task_id"].tolist() == list(DIAGNOSTIC_TASK_IDS)

    with pytest.raises(ValueError, match="missing diagnostic task IDs"):
        select_diagnostic_tasks(tasks.iloc[:-1])


def test_build_paired_deltas_uses_online_minus_zero():
    raw = pd.DataFrame([
        {"seed": 42, "task_id": "t", "split": "heldout", "inference_mode": "zero_context", "episode_return": 100.0, "EPI": 2.0, "temp_violation": 10.0, "co2_violation": 20.0, "rh_violation": 30.0, "action_trace_path": "zero.npy"},
        {"seed": 42, "task_id": "t", "split": "heldout", "inference_mode": "online_context", "episode_return": 110.0, "EPI": 3.0, "temp_violation": 8.0, "co2_violation": 18.0, "rh_violation": 27.0, "action_trace_path": "online.npy"},
    ])
    paired = build_paired_deltas(raw, load_actions=lambda path: np.zeros((3, 1)) if "zero" in path else np.ones((3, 1)))
    assert paired.loc[0, "episode_return_delta"] == 10.0
    assert paired.loc[0, "mean_abs_action_delta"] == 1.0


def test_gate_passes_only_when_all_five_conditions_hold():
    paired = passing_paired_fixture()
    decision = evaluate_context_gate(paired)
    assert decision["outcome"] == "continue_to_500k"
    assert all(decision["conditions"].values())

    paired.loc[paired["split"] != "fixed", "episode_return_delta"] = -5.0
    decision = evaluate_context_gate(paired)
    assert decision["outcome"] == "redesign_before_training"
    assert not decision["conditions"]["positive_nonfixed_return"]
```

`passing_paired_fixture()` must contain both seeds, one fixed row and seven non-fixed rows per seed, non-zero action deltas, positive non-fixed return deltas, fixed return losses below 2%, and violation ratios below 1.05.

- [ ] **Step 2: Run tests and verify red**

```powershell
python -m pytest tests\experiments\test_context_ab.py -q
```

Expected: collection fails because `context_ab.py` does not exist.

- [ ] **Step 3: Implement fixed task selection and pairing**

Create `context_ab.py` with:

```python
DIAGNOSTIC_TASK_IDS = (
    "fixed_2010_d59_u0p00_standard",
    "heldout_2011_d59_u0p00_standard",
    "heldout_2012_d59_u0p00_standard",
    "heldout_2013_d59_u0p00_standard",
    "uncertainty_2012_d80_u0p05_standard",
    "uncertainty_2013_d100_u0p15_standard",
    "economic_2011_d59_u0p00_high_energy_price",
    "economic_2013_d100_u0p00_combined_stress",
)
MODES = ("zero_context", "online_context")
PAIR_METRICS = ("episode_return", "EPI", "temp_violation", "co2_violation", "rh_violation")

def select_diagnostic_tasks(tasks: pd.DataFrame) -> pd.DataFrame:
    indexed = tasks.set_index("task_id", drop=False)
    missing = [task_id for task_id in DIAGNOSTIC_TASK_IDS if task_id not in indexed.index]
    if missing:
        raise ValueError(f"missing diagnostic task IDs: {missing}")
    return indexed.loc[list(DIAGNOSTIC_TASK_IDS)].reset_index(drop=True)
```

Implement `build_paired_deltas()` by merging the two modes on `seed`, `task_id`, and `split`, subtracting zero columns from online columns, loading both `.npy` action traces, requiring equal shapes, and computing mean absolute action difference.

- [ ] **Step 4: Implement the five-condition gate**

Use `epsilon = 1e-9`. Compute:

```python
conditions = {
    "actions_change_both_seeds": bool((paired.groupby("seed")["mean_abs_action_delta"].max() > epsilon).all()),
    "positive_nonfixed_return": bool(nonfixed["episode_return_delta"].mean() > 0),
    "no_seed_large_return_loss": bool((seed_relative_delta >= -0.02).all()),
    "violation_burden_within_5pct": bool(mean_normalized_burden <= 1.05),
    "fixed_return_within_2pct": bool((fixed_relative_delta >= -0.02).all()),
}
```

For each violation metric, normalize online by `abs(zero) + epsilon`; define the burden as the mean ratio across all paired rows and three violation metrics. Return a JSON-serializable dictionary containing `outcome`, `conditions`, and the scalar values used by each condition.

- [ ] **Step 5: Add artifact writers**

Add `write_context_ab_artifacts(raw, result_root, manifest)` that writes:

- `eval_raw.csv`;
- `paired_deltas.csv`;
- `split_summary.csv` grouped by mode and split;
- `diagnostic_manifest.json`;
- `decision.json`.

Use `json.dumps(..., indent=2, sort_keys=True)` and pandas `to_csv(index=False)`. Refuse to write when raw keys `(seed, task_id, inference_mode)` are duplicated or when the row count is not 32.

- [ ] **Step 6: Run unit tests**

```powershell
python -m pytest tests\experiments\test_context_ab.py -q
```

Expected: all tests pass.

- [ ] **Step 7: Commit**

```powershell
git add src/gl_gym/experiments/context_ab.py tests/experiments/test_context_ab.py
git commit -m "feat: add context A/B diagnostic gate"
```

## Task 5: Add the reproducible diagnostic CLI

**Files:**
- Create: `experiments/scripts/run_context_ab.py`
- Modify: `tests/experiments/test_context_ab.py`

- [ ] **Step 1: Write failing CLI validation tests**

Load the script as a module and test helpers rather than launching greenhouse simulations:

```python
def test_build_diagnostic_runs_uses_last_checkpoints(tmp_path):
    runs = build_diagnostic_runs(
        model_root=tmp_path,
        seeds=[42, 123],
    )
    assert runs[0]["model_path"].endswith("agri_metarl_seed42/last_model.zip")
    assert runs[1]["model_path"].endswith("agri_metarl_seed123/last_model.zip")


def test_cli_result_root_is_not_formal_suite_root():
    with pytest.raises(ValueError, match="diagnostic result root"):
        validate_result_root(
            Path("artifacts/results/AgriControl_C_2026-06-30"),
            Path("artifacts/results/AgriControl_C_2026-06-30"),
        )
```

- [ ] **Step 2: Run tests and verify red**

```powershell
python -m pytest tests\experiments\test_context_ab.py -k "diagnostic_runs or result_root" -q
```

Expected: failures because the CLI helpers do not exist.

- [ ] **Step 3: Implement CLI arguments and validation**

The CLI accepts:

```text
--source_manifest
--source_tasks_csv
--model_root
--result_root
--seeds 42 123
--device cpu
--resume
```

Default `result_root` is `artifacts/results/AgriControl_C_2026-07-10-v3-context-ab`. Require exactly seeds 42 and 123 for this approved diagnostic. Resolve and validate both `last_model.zip` and `last_vecnormalize.pkl` paths before creating output files. Refuse a result root equal to the source suite result root.

- [ ] **Step 4: Implement the 32-episode loop**

Reuse `load_task_env()` and `_task_from_row()` from `evaluate_suite.py` only after moving those helpers into `src/gl_gym/experiments/suite_evaluation.py`; update imports in `evaluate_suite.py` and its tests accordingly. Do not import an executable script as production code.

For each seed, task, and mode:

```python
metrics, diagnostics = run_deterministic_episode(
    model,
    env,
    inference_mode=mode,
    return_diagnostics=True,
)
action_trace = diagnostics.pop("action_trace")
np.save(trace_path, action_trace)
rows.append({
    "seed": seed,
    "task_id": task.task_id,
    "split": task.split,
    "inference_mode": mode,
    "checkpoint_steps": int(model.num_timesteps),
    "action_trace_path": str(trace_path),
    **metrics,
    **diagnostics,
})
```

Reload the model once per mode or call `end_inference_episode()` reliably between tasks. Use a fresh environment per episode. With `--resume`, skip only when both the raw row and action trace exist.

- [ ] **Step 5: Add manifest provenance**

Record exact source manifest, task file, checkpoint and VecNormalize paths, checkpoint steps, selected IDs, modes, seeds, Git commit, dirty flag, Python version, and UTC timestamp. The dirty flag is provenance, not a reason to abort.

- [ ] **Step 6: Run CLI unit tests and a fake smoke**

```powershell
python -m pytest tests\experiments\test_context_ab.py tests\experiments\test_suite_evaluation.py -q
python experiments\scripts\run_context_ab.py --help
```

Expected: tests pass and help exits 0 showing all required arguments.

- [ ] **Step 7: Commit**

```powershell
git add experiments/scripts/run_context_ab.py src/gl_gym/experiments/suite_evaluation.py experiments/scripts/evaluate_suite.py tests/experiments/test_context_ab.py tests/experiments/test_suite_evaluation.py
git commit -m "feat: add reproducible online context diagnostic CLI"
```

## Task 6: Verify implementation and run the approved diagnostic

**Files:**
- Generated: `artifacts/results/AgriControl_C_2026-07-10-v3-context-ab/`
- No source edits unless verification exposes a defect; any defect fix must start with a failing regression test.

- [ ] **Step 1: Run focused tests**

```powershell
python -m pytest tests\agri_metarl tests\experiments\test_suite_evaluation.py tests\experiments\test_context_ab.py -q
```

Expected: all selected tests pass with zero failures.

- [ ] **Step 2: Run all repository tests and compile checks**

```powershell
python -m pytest -q
python -m compileall -q src tests experiments
```

Expected: pytest exits 0 and compileall exits 0.

- [ ] **Step 3: Verify checkpoint metadata before evaluation**

```powershell
@'
import sys
from pathlib import Path
sys.path.insert(0, "src")
from stable_baselines3.common.save_util import load_from_zip_file
root = Path("artifacts/models/AgriControl_C_2026-07-09-v3-pilot3/agri_metarl/deterministic/models")
for seed in (42, 123):
    path = root / f"agri_metarl_seed{seed}" / "last_model.zip"
    data, _, _ = load_from_zip_file(path, device="cpu")
    print(seed, data["num_timesteps"], path.stat().st_size)
'@ | python -
```

Expected: both checkpoints report `196608` steps.

- [ ] **Step 4: Run the 32-episode diagnostic**

```powershell
$env:WANDB_MODE='disabled'
$env:WANDB_DISABLED='true'
python experiments\scripts\run_context_ab.py `
  --source_manifest artifacts\results\AgriControl_C_2026-07-09-v3-pilot3\suite_manifest.json `
  --source_tasks_csv artifacts\results\AgriControl_C_2026-07-09-v3-pilot3\eval_tasks.csv `
  --model_root artifacts\models\AgriControl_C_2026-07-09-v3-pilot3 `
  --result_root artifacts\results\AgriControl_C_2026-07-10-v3-context-ab `
  --seeds 42 123 `
  --device cpu `
  --resume
```

Expected: 32 unique rows and 32 action trace files are produced. The command may be resumed after interruption.

- [ ] **Step 5: Validate artifact integrity**

```powershell
@'
import json
from pathlib import Path
import pandas as pd
root = Path("artifacts/results/AgriControl_C_2026-07-10-v3-context-ab")
raw = pd.read_csv(root / "eval_raw.csv")
assert len(raw) == 32
assert not raw.duplicated(["seed", "task_id", "inference_mode"]).any()
assert set(raw["inference_mode"]) == {"zero_context", "online_context"}
assert raw["action_trace_path"].map(lambda p: Path(p).is_file()).all()
decision = json.loads((root / "decision.json").read_text(encoding="utf-8"))
assert decision["outcome"] in {"continue_to_500k", "redesign_before_training"}
print(raw.groupby(["inference_mode", "split"])[["episode_return", "EPI", "temp_violation", "co2_violation", "rh_violation"]].mean())
print(json.dumps(decision, indent=2))
'@ | python -
```

Expected: assertions pass and the exact gate outcome is printed.

- [ ] **Step 6: Record the research decision**

If `decision.json` says `continue_to_500k`, update the existing v3 plan so the next target is 500,000 steps followed by a full 91-task two-seed evaluation. If it says `redesign_before_training`, do not launch training; start a new brainstorming cycle focused on the failed gate conditions.

- [ ] **Step 7: Commit source changes only if verification required fixes**

Do not commit generated `artifacts/` unless repository policy explicitly tracks them. If a regression fix was required, commit its test and minimal source fix together with a specific message.

## Plan Self-Review

- Spec coverage: lifecycle isolation, normalized transition flow, task identity, two modes, exact eight tasks, 32 episodes, action/context diagnostics, separate artifact root, five-condition gate, cleanup, compatibility, and follow-on decision are covered.
- Placeholder scan: no `TBD`, `TODO`, “implement later”, or unspecified error-handling steps remain.
- Type consistency: lifecycle methods, inference modes, diagnostic keys, file names, and gate outcome names are consistent across tasks.
- Scope: the plan changes evaluation semantics and diagnostic tooling only; training algorithm changes and manuscript updates remain excluded.
