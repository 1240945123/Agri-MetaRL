# Agri-MetaRL 2.0 Integrity Foundation and Pilot Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a trustworthy experiment pipeline and a cross-rollout, task-aware Agri-MetaRL 2.0 implementation, then run a gated three-seed pilot before any confirmatory experiment or manuscript rewrite.

**Architecture:** A complete task descriptor and task-instance identifier flow from environment transitions into a recurrent rollout buffer. A bounded episodic support memory freezes the first support transitions for each task instance across rollout boundaries; query transitions use a learned support context to produce a bounded advantage residual. A separate integrity pipeline validates immutable raw evaluation records before aggregation or figure generation.

**Tech Stack:** Python 3.11+, PyTorch, Gymnasium, Stable-Baselines3, sb3-contrib, NumPy, pandas, SciPy, pytest, YAML, Matplotlib.

**Commit policy:** Do not stage or commit automatically. End every task with a diff and test checkpoint.

---

## Scope Boundary

This plan ends at the Pilot decision gate. Confirmatory five-seed training, full ablations, final statistical analysis, manuscript rewriting, figure redesign, and PDF production receive a second implementation plan only after the Pilot gate passes.

## File Map

**Create**

- `src/gl_gym/tasks.py`: immutable `TaskDescriptor` and `TaskInstance` types.
- `src/gl_gym/RL/agri_metarl/memory.py`: bounded cross-rollout support memory.
- `src/gl_gym/RL/agri_metarl/diagnostics.py`: context/residual/update-coverage diagnostics.
- `src/gl_gym/RL/agri_metarl/legacy_agri_metarl.py`: frozen legacy implementation used only as a baseline.
- `src/gl_gym/RL/context_recurrent_ppo.py`: context-conditioned policy/value baseline.
- `src/gl_gym/evaluation/__init__.py`: evaluation package marker.
- `src/gl_gym/evaluation/records.py`: run manifest and raw-result schemas.
- `src/gl_gym/evaluation/validation.py`: strict integrity validator.
- `src/gl_gym/evaluation/aggregation.py`: seed/task-aware aggregation.
- `experiments/configs/task_distributions.yml`: train, validation, and Pilot task distributions.
- `experiments/configs/pilot.yml`: algorithms, seeds, budgets, and gate thresholds.
- `experiments/scripts/evaluate_task_grid.py`: explicit model-task evaluation.
- `experiments/scripts/run_pilot.py`: resumable Pilot orchestrator.
- `experiments/figures/plot_validated_results.py`: figures from validator-approved aggregates only.
- `tests/tasks/test_task_descriptor.py`.
- `tests/agri_metarl/test_support_memory.py`.
- `tests/agri_metarl/test_meta_advantage_head_v2.py`.
- `tests/agri_metarl/test_rollout_integration.py`.
- `tests/agri_metarl/test_context_recurrent_ppo.py`.
- `tests/evaluation/test_validation.py`.
- `tests/evaluation/test_aggregation.py`.
- `tests/integrity/test_no_result_scaling.py`.

**Modify**

- `src/gl_gym/environments/tomato_env.py`: emit task descriptor and instance ID.
- `src/gl_gym/environments/base_env.py`: scenario/task configuration plumbing.
- `src/gl_gym/RL/utils.py`: pass vector-environment rank into task-instance identity.
- `src/gl_gym/RL/agri_metarl/buffer.py`: store task instance IDs and query masks.
- `src/gl_gym/RL/agri_metarl/meta_advantage_head.py`: transition-set encoder and bounded residual head.
- `src/gl_gym/RL/agri_metarl/agri_metarl.py`: memory-aware rollout collection and correction.
- `configs/agents/agri_metarl.yml`: Agri-MetaRL 2.0 hyperparameters.
- `experiments/scripts/train_paper_experiments.py`: manifest-aware training entry point.
- `experiments/scripts/run_paper_pipeline_after_train.py`: disable legacy unvalidated plotting path.
- `.gitignore`: ignore new Pilot checkpoints/raw outputs while tracking experiment configs and code.

**Relocate without deleting history**

- `artifacts/figures/generators/plot_paper_figures.py` → `archive/previous_versions/unsafe_plot_paper_figures.py` after recording its hash; do not use it for new results.

### Task 0: Restore a Clean Test Baseline

**Files:**
- Modify: `tests/env_test.py`

- [ ] **Step 1: Reproduce the six existing failures**

Run the full test suite with `PYTHONPATH=src`.

Expected: six `TypeError` failures because `eval_options_heldout` is passed to `TomatoEnv.__init__`.

- [ ] **Step 2: Separate test-only held-out metadata**

After `load_env_params()` in test setup, remove the held-out evaluation block before direct construction:

```python
self.heldout_options = self.env_specific_params.pop("eval_options_heldout", None)
self.env = TomatoEnv(
    base_env_params=self.env_base_params,
    **self.env_specific_params,
)
```

Assert `self.heldout_options` is present so the test does not silently hide a malformed configuration.

- [ ] **Step 3: Run the full baseline suite**

Run `python -m pytest -q` with `PYTHONPATH=src`.

Expected: all collected baseline tests pass before Agri-MetaRL 2.0 implementation begins.

- [ ] **Step 4: Review checkpoint**

Confirm the change affects only test construction and does not remove held-out options from production configuration loading.

### Task 1: Quarantine Result-Manipulation Paths

**Files:**
- Create: `tests/integrity/test_no_result_scaling.py`
- Move: `artifacts/figures/generators/plot_paper_figures.py`
- Modify: `experiments/scripts/run_paper_pipeline_after_train.py`

- [ ] **Step 1: Record the legacy plotting script hash**

Run:

```powershell
Get-FileHash artifacts\figures\generators\plot_paper_figures.py -Algorithm SHA256 |
  Format-List Path,Hash
```

Expected: one SHA-256 value recorded in the execution notes before relocation.

- [ ] **Step 2: Write the failing integrity test**

```python
from pathlib import Path


ACTIVE_ROOTS = (Path("src"), Path("experiments"))
FORBIDDEN = (
    "_LEARNING_SCALE",
    "_ECONOMIC_SCALE",
    "scale rewards so Agri-MetaRL",
    "learning_order=True",
)


def test_active_code_contains_no_algorithm_dependent_result_scaling():
    violations = []
    for root in ACTIVE_ROOTS:
        for path in root.rglob("*.py"):
            text = path.read_text(encoding="utf-8")
            for marker in FORBIDDEN:
                if marker in text:
                    violations.append(f"{path}: {marker}")
    assert violations == []
```

- [ ] **Step 3: Run the test against the unsafe script in an active location**

Factor the test around `find_forbidden_scaling(paths)`. For the red step, call it directly with `[Path("artifacts/figures/generators/plot_paper_figures.py")]`; after quarantine, the permanent test calls it with Python files under `ACTIVE_ROOTS`.

Run:

```powershell
$env:PYTHONPATH = (Join-Path (Resolve-Path '.') 'src')
python -m pytest tests/integrity/test_no_result_scaling.py -v
```

Expected: FAIL listing `_LEARNING_SCALE`, `_ECONOMIC_SCALE`, and `learning_order=True`.

- [ ] **Step 4: Quarantine the unsafe script and remove the pipeline call**

Move it to `archive/previous_versions/unsafe_plot_paper_figures.py`. Remove `FIGURE_SCRIPT` and the final figure-generation subprocess from `run_paper_pipeline_after_train.py`. Do not replace it with a permissive plotting call.

- [ ] **Step 5: Finalize the active-root test and run it**

Set `ACTIVE_ROOTS` exactly as shown in Step 2 and rerun the test.

Expected: PASS.

- [ ] **Step 6: Review checkpoint**

Run `git diff -- experiments/scripts/run_paper_pipeline_after_train.py tests/integrity/test_no_result_scaling.py`. Do not commit.

### Task 2: Define Complete Task and Task-Instance Identity

**Files:**
- Create: `src/gl_gym/tasks.py`
- Create: `tests/tasks/test_task_descriptor.py`
- Modify: `src/gl_gym/environments/tomato_env.py`

- [ ] **Step 1: Write failing descriptor tests**

```python
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
```

- [ ] **Step 2: Verify red state**

Run `python -m pytest tests/tasks/test_task_descriptor.py -v` with `PYTHONPATH=src`.

Expected: FAIL because `gl_gym.tasks` does not exist.

- [ ] **Step 3: Implement immutable identities**

```python
from dataclasses import asdict, dataclass


@dataclass(frozen=True, slots=True)
class TaskDescriptor:
    weather_year: int
    start_day: int
    parameter_uncertainty: float
    economic_scenario: str
    climate_constraint_scenario: str

    @property
    def stable_key(self) -> str:
        return (
            f"{self.weather_year}:{self.start_day}:"
            f"{self.parameter_uncertainty:.6f}:"
            f"{self.economic_scenario}:{self.climate_constraint_scenario}"
        )

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict) -> "TaskDescriptor":
        return cls(**value)


@dataclass(frozen=True, slots=True)
class TaskInstance:
    task: TaskDescriptor
    environment_index: int
    episode_index: int

    @property
    def stable_key(self) -> str:
        return f"{self.task.stable_key}:env{self.environment_index}:episode{self.episode_index}"
```

- [ ] **Step 4: Emit identity from every environment transition**

Update `make_vec_env()` in `src/gl_gym/RL/utils.py` so each constructed environment receives its vector rank as `environment_index`. Add explicit environment fields for economic scenario, climate scenario, environment index, and an episode counter incremented in `reset()`. Add both serialized objects to `info`:

```python
info["task_descriptor"] = self.task_descriptor.to_dict()
info["task_instance_key"] = self.task_instance.stable_key
```

- [ ] **Step 5: Add an environment emission test**

Instantiate a short TomatoEnv, reset it, take one action, and assert that the returned `info` reconstructs the configured descriptor and retains the same instance key until the next reset.

- [ ] **Step 6: Run tests**

Run:

```powershell
python -m pytest tests/tasks/test_task_descriptor.py -v
```

Expected: all descriptor and emission tests pass.

- [ ] **Step 7: Review checkpoint**

Run `git diff -- src/gl_gym/tasks.py src/gl_gym/environments/tomato_env.py tests/tasks`. Do not commit.

### Task 3: Build Cross-Rollout Support Memory

**Files:**
- Create: `src/gl_gym/RL/agri_metarl/memory.py`
- Create: `tests/agri_metarl/test_support_memory.py`

- [ ] **Step 1: Write failing memory tests**

Tests must cover:

```python
def test_support_persists_across_rollouts():
    memory = TaskSupportMemory(support_size=3, max_instances=4)
    for step in range(2):
        assert memory.observe("task-a", transition(step)) is False
    memory.begin_rollout()
    assert memory.observe("task-a", transition(2)) is False
    assert memory.observe("task-a", transition(3)) is True
    assert len(memory.support("task-a")) == 3


def test_tasks_never_share_support():
    memory = TaskSupportMemory(support_size=2, max_instances=4)
    memory.observe("task-a", transition(1))
    memory.observe("task-b", transition(2))
    assert memory.support("task-a")[0].reward == 1
    assert memory.support("task-b")[0].reward == 2


def test_support_freezes_before_query():
    memory = TaskSupportMemory(support_size=2, max_instances=4)
    memory.observe("task-a", transition(1))
    memory.observe("task-a", transition(2))
    memory.observe("task-a", transition(99))
    assert [x.reward for x in memory.support("task-a")] == [1, 2]
```

- [ ] **Step 2: Verify red state**

Run `python -m pytest tests/agri_metarl/test_support_memory.py -v`.

Expected: FAIL because `TaskSupportMemory` does not exist.

- [ ] **Step 3: Implement the bounded memory**

Define an immutable `Transition` containing NumPy arrays for observation, action, next observation, scalar reward, and done. Implement `TaskSupportMemory` with an `OrderedDict[str, list[Transition]]`. `observe()` appends only until `support_size`; after the list is full it returns `True` to mark a query transition and never changes frozen support. Evict the least-recently-created task instance when `max_instances` is exceeded.

- [ ] **Step 4: Run memory tests**

Expected: all persistence, isolation, freezing, and eviction tests pass.

- [ ] **Step 5: Review checkpoint**

Inspect `memory.py` for global mutable state and implicit random sampling. Both must be absent.

### Task 4: Store Task Instances and Query Masks in the Rollout Buffer

**Files:**
- Modify: `src/gl_gym/RL/agri_metarl/buffer.py`
- Create: `tests/agri_metarl/test_rollout_integration.py`

- [ ] **Step 1: Write a failing buffer test**

Construct a minimal `AgriMetaRLRolloutBuffer`, add transitions spanning two rollout resets with one stable task-instance key, and assert that each stored row exposes its task key and query flag. Include a second task instance and assert isolation.

- [ ] **Step 2: Add buffer fields**

Store:

```python
self.task_instance_keys = np.empty((self.buffer_size, self.n_envs), dtype=object)
self.query_mask = np.zeros((self.buffer_size, self.n_envs), dtype=bool)
```

Extend `add()` with keyword-only `task_instance_keys` and `query_mask`. Remove the integer year/day encoding path after all callers use the new identities.

- [ ] **Step 3: Run the buffer test**

Expected: stored keys and masks match inputs exactly and reset clears only rollout arrays, not the external support memory.

- [ ] **Step 4: Review checkpoint**

Run the focused buffer and support-memory suites together.

### Task 5: Replace Hand-Written Statistics with a Transition-Set Encoder

**Files:**
- Modify: `src/gl_gym/RL/agri_metarl/meta_advantage_head.py`
- Create: `tests/agri_metarl/test_meta_advantage_head_v2.py`

- [ ] **Step 1: Write failing encoder and residual tests**

Cover exact behavior:

```python
def test_set_encoder_is_permutation_invariant():
    encoder = TransitionSetEncoder(obs_dim=6, action_dim=2, context_dim=8)
    batch = support_batch()
    a = encoder(**batch)
    order = torch.tensor([2, 0, 1])
    b = encoder(**{k: v[order] for k, v in batch.items()})
    torch.testing.assert_close(a, b)


def test_residual_is_bounded_and_has_gradients():
    head = AdvantageResidualHead(obs_dim=6, context_dim=8, alpha=0.5)
    corrected, residual = head(obs(), advantages(), context())
    assert torch.all(residual.abs() <= 0.5 + 1e-6)
    corrected.sum().backward()
    assert all(p.grad is not None for p in head.parameters())
```

- [ ] **Step 2: Verify red state**

Expected: FAIL because the new classes are undefined.

- [ ] **Step 3: Implement the transition encoder**

Concatenate observation, action, reward, next observation, and done per support transition. Map with `Linear -> LayerNorm -> SiLU -> Linear -> SiLU`, then mean-pool over the support dimension. Return a single context vector per task instance.

- [ ] **Step 4: Implement the bounded residual**

Project the query observation, concatenate projected observation, raw advantage, and context, then compute:

```python
residual = self.alpha * torch.tanh(self.residual_mlp(features)).squeeze(-1)
corrected = raw_advantage + residual
```

Do not normalize or clip inside the head.

- [ ] **Step 5: Run tests**

Expected: permutation, shape, bound, finite-value, and gradient tests pass.

### Task 6: Integrate Agri-MetaRL 2.0 and Diagnostics

**Files:**
- Modify: `src/gl_gym/RL/agri_metarl/agri_metarl.py`
- Create: `src/gl_gym/RL/agri_metarl/diagnostics.py`
- Modify: `configs/agents/agri_metarl.yml`
- Extend: `tests/agri_metarl/test_rollout_integration.py`

- [ ] **Step 1: Freeze the legacy baseline before changing the implementation**

Copy the current implementation into `legacy_agri_metarl.py` without behavioral edits, rename its class to `LegacyAgriMetaRL`, and create `configs/agents/agri_metarl_legacy.yml` from the original configuration. Add a loading smoke test that proves the legacy class and configuration remain usable.

- [ ] **Step 2: Write a failing cross-rollout correction test**

Use a tiny vector environment whose episode length exceeds `n_steps`. Assert that query corrections occur in a rollout that does not contain an episode start after support was filled in an earlier rollout. Assert correction coverage is nonzero and task B never receives task A context.

- [ ] **Step 3: Integrate memory during collection**

For each environment transition:

1. reconstruct the task descriptor and instance key from `info`;
2. call `support_memory.observe(instance_key, transition)`;
3. store the instance key and returned query flag in the rollout buffer.

The support memory is owned by the algorithm instance and is not cleared by `rollout_buffer.reset()`.

- [ ] **Step 4: Replace `_apply_meta_advantage_correction()`**

Group query indices by task-instance key, encode frozen support once per group, compute bounded residuals for the group's query observations, write detached corrected advantages into query rows, and calculate the auxiliary loss before the optimizer step. Leave support-row GAE values unchanged.

- [ ] **Step 5: Normalize at PPO minibatch scope**

Use Recurrent PPO's existing advantage normalization on the resulting buffer. Remove the old per-query `normalize_and_clip()` path.

- [ ] **Step 6: Add diagnostics**

Record:

```text
train/meta_loss
train/context_norm_mean
train/context_between_task_variance
train/residual_abs_mean
train/residual_saturation_rate
train/query_correction_fraction
```

Use finite-safe online accumulators; reset diagnostics after logging.

- [ ] **Step 7: Update configuration**

Replace old context-statistic flags with explicit `support_size`, `max_task_instances`, `context_dim`, `transition_hidden_dim`, `residual_alpha`, `meta_loss_weight`, and `residual_regularization` values. Retain the original settings in an `agri_metarl_legacy.yml` config for the legacy baseline.

- [ ] **Step 8: Run focused tests and a CPU smoke train**

Run all `tests/agri_metarl/` tests, then a tiny synthetic environment for two rollouts.

Expected: tests pass; query correction fraction is greater than zero in the second rollout; no NaN/Inf appears. Export both legacy and v2 classes explicitly from `gl_gym.RL.agri_metarl` so evaluation cannot confuse them.

### Task 6A: Implement the Context-Conditioned Recurrent PPO Baseline

**Files:**
- Create: `src/gl_gym/RL/context_recurrent_ppo.py`
- Create: `tests/agri_metarl/test_context_recurrent_ppo.py`
- Create: `configs/agents/context_recurrentppo.yml`

- [ ] **Step 1: Write failing baseline tests**

Use the same `TaskSupportMemory` and `TransitionSetEncoder`. Assert that policy/value observations are augmented with a fixed-size support context, that raw environmental observations are unchanged, and that context from task A is never used for task B.

- [ ] **Step 2: Implement the baseline adapter**

Create a recurrent PPO subclass that builds support context using the same memory rules as Agri-MetaRL 2.0, concatenates the detached context to policy/value features, and uses unmodified GAE advantages. Do not instantiate the advantage residual head.

- [ ] **Step 3: Match the backbone and training budget**

Copy Recurrent PPO optimizer, LSTM, rollout, minibatch, and schedule settings into `context_recurrentppo.yml`; add only context encoder and support-memory settings.

- [ ] **Step 4: Run focused tests and a two-rollout smoke train**

Expected: context augmentation is active after support warm-up, GAE remains unmodified, and no cross-task leakage occurs.

- [ ] **Step 5: Register the baseline**

Add explicit algorithm-map entries for `legacy_agri_metarl`, `agri_metarl_v2`, and `context_recurrentppo` in training and evaluation scripts. Unknown aliases must raise an error rather than fall back to another class.

### Task 7: Define Reproducible Task Distributions

**Files:**
- Create: `experiments/configs/task_distributions.yml`
- Create: `experiments/configs/pilot.yml`
- Modify: `src/gl_gym/environments/base_env.py`
- Modify: `src/gl_gym/environments/tomato_env.py`
- Extend: `tests/tasks/test_task_descriptor.py`

- [ ] **Step 1: Write failing task-sampling tests**

Assert that train samples only years 2001-2012, validation only 2013-2015, and temporal test only 2016-2020. Assert a fixed seed reproduces the same task sequence and no final-test year appears in train/validation.

- [ ] **Step 2: Write explicit task configuration**

Declare named distributions and concrete Pilot scenario coefficients. Use baseline coefficients as the standard scenario; define high/low economic scenarios as documented multipliers in YAML, not in code. Declare standard/strict/relaxed constraint bounds explicitly.

- [ ] **Step 3: Implement seeded task sampling**

Sampling accepts a named distribution and NumPy generator. Evaluation uses an explicit enumerated task list and never random reset selection.

- [ ] **Step 4: Run sampling tests**

Expected: deterministic sequence, disjoint year splits, valid scenario labels, and complete descriptors.

### Task 8: Create Immutable Manifests and Raw Result Records

**Files:**
- Create: `src/gl_gym/evaluation/records.py`
- Create: `tests/evaluation/test_validation.py`
- Modify: `experiments/scripts/train_paper_experiments.py`

- [ ] **Step 1: Write schema tests**

Create a manifest and raw record, serialize each to JSON, deserialize, and assert exact equality. Assert a raw record requires algorithm version, training seed, model path, normalization path, config hash, task descriptor, evaluation mode, and all metrics.

- [ ] **Step 2: Implement records**

Use frozen dataclasses with explicit `to_dict()`/`from_dict()`. Compute configuration hashes from canonical JSON with sorted keys and SHA-256. Represent dirty working trees with `commit_sha` plus a hash of the tracked diff and relevant untracked source files.

- [ ] **Step 3: Make training manifest-aware**

Before training, write one manifest per run beneath `artifacts/experiments/<experiment_id>/manifests/`. Refuse to reuse a run directory whose manifest differs from the requested configuration.

- [ ] **Step 4: Run schema tests**

Expected: round trips pass and mismatched reuse is rejected.

### Task 9: Implement the Integrity Validator

**Files:**
- Create: `src/gl_gym/evaluation/validation.py`
- Extend: `tests/evaluation/test_validation.py`

- [ ] **Step 1: Write rejection tests**

Construct synthetic records and assert separate errors for:

- unequal algorithm seed sets;
- duplicate deterministic model-task pairs;
- missing required tasks;
- incomplete task descriptors;
- manifest/config-hash mismatch;
- non-finite metrics;
- unexpected algorithm names.

Add a regression fixture representing six PPO runs, five Recurrent PPO runs, five Agri-MetaRL runs, and five identical deterministic repetitions. Assert rejection.

- [ ] **Step 2: Implement structured validation errors**

Return a `ValidationReport` with `errors`, `warnings`, method counts, seed counts, task counts, and duplicate keys. The CLI exits nonzero whenever `errors` is nonempty.

- [ ] **Step 3: Run validator tests**

Expected: every invalid fixture fails for the declared reason and a balanced complete fixture passes.

### Task 10: Build Explicit Task-Grid Evaluation and Aggregation

**Files:**
- Create: `experiments/scripts/evaluate_task_grid.py`
- Create: `src/gl_gym/evaluation/aggregation.py`
- Create: `tests/evaluation/test_aggregation.py`

- [ ] **Step 1: Write aggregation tests**

Use balanced synthetic records with known means. Assert aggregation uses one row per model-task pair, preserves seed/task counts, computes mean/SD, and produces a paired method-difference table. Assert duplicate rows cannot reach aggregation without validator approval.

- [ ] **Step 2: Implement explicit task evaluation**

Load tasks from YAML, set the environment to each exact descriptor, evaluate one deterministic episode per model-task pair, and append one JSONL record atomically. Resume by skipping only exact existing keys whose manifest/config hashes match.

- [ ] **Step 3: Implement validated aggregation**

Require a passing `ValidationReport`. Produce tidy per-seed/per-task results, aggregate summaries, worst-task and lower-quantile tables, paired differences, and bootstrap inputs. Do not implement plotting or ranking in this module.

- [ ] **Step 4: Run evaluation and aggregation tests**

Expected: synthetic end-to-end records validate and aggregate to exact expected values.

### Task 11: Create Safe Plotting and LaTeX Asset Generation

**Files:**
- Create: `experiments/figures/plot_validated_results.py`
- Extend: `tests/integrity/test_no_result_scaling.py`
- Extend: `tests/evaluation/test_aggregation.py`

- [ ] **Step 1: Write a failing asset-provenance test**

Require every generated figure/table sidecar to contain the aggregate file hash, validator report hash, generation command, and timestamp. Assert the plotting function preserves the numerical bar/line values supplied by the aggregate table.

- [ ] **Step 2: Implement plotting from aggregates only**

Accept validator-approved aggregate files, use fixed method colors and declared method order, and write figure plus JSON provenance sidecar. No function accepts a scale-by-method mapping.

- [ ] **Step 3: Generate LaTeX tables**

Format aggregate values and confidence intervals into standalone `.tex` fragments. Include a machine-readable CSV-to-cell map for manuscript consistency checks.

- [ ] **Step 4: Run integrity and asset tests**

Expected: numeric preservation and provenance tests pass; forbidden-scaling scan remains clean.

### Task 12: Run the Three-Seed Pilot and Apply the Gate

**Files:**
- Create: `experiments/scripts/run_pilot.py`
- Read: `experiments/configs/pilot.yml`
- Write locally: `artifacts/experiments/agrimetarl2-pilot-*`

- [ ] **Step 1: Verify the environment before training**

Run the full new unit/integration suite, confirm CUDA visibility, report E: free space, and execute one 10,000-step smoke run per learned method. Stop on any non-finite loss, missing manifest, or result-validation error.

- [ ] **Step 2: Run Pilot training resumably**

Train Recurrent PPO, legacy Agri-MetaRL, Agri-MetaRL 2.0, and context-conditioned Recurrent PPO with seeds `42`, `123`, and `456` using the shared reduced Pilot budget declared in `pilot.yml`. Write manifests and checkpoints under the Pilot experiment ID.

- [ ] **Step 3: Evaluate the Pilot task grid**

Evaluate the exact shared ID and OOD Pilot tasks once per model-task pair. Validate raw records before aggregation.

- [ ] **Step 4: Compute gate diagnostics**

The Pilot passes only if all conditions hold:

```text
validation errors = 0
non-finite losses = 0
query correction fraction >= 0.20 after support warm-up
residual saturation rate <= 0.25
context between-task variance > 0
all methods complete the same seed/task matrix
Agri-MetaRL 2.0 training throughput >= 50% of Recurrent PPO throughput
```

Performance superiority is not required for implementation validity, but effect direction and uncertainty are reported before deciding whether to proceed.

- [ ] **Step 5: Pilot review checkpoint**

Present manifests, validator report, diagnostic plots, runtime, raw aggregate values, and any failures. Do not start confirmatory five-seed runs until the user approves the Pilot result and a confirmatory plan is written.
