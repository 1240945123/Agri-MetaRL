# Minimal-Feasibility Action Shield Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an evaluation-only, rate-limit-preserving action shield that transactionally recovers from policy-induced ODE failures and evaluates it through the known-failure, 32-episode context A/B, and 91-task gates.

**Architecture:** Pure action projection and evidence types live outside the environment. A distinct `ShieldedTomatoEnv` reuses the unchanged observation/action contract but owns retry and commit semantics, so the existing `TomatoEnv` remains the unshielded baseline. Evaluation helpers consume the executed action reported by the environment, while shield-specific aggregation and CLIs publish separate, atomic artifacts and enforce stage dependencies.

**Tech Stack:** Python 3.11, NumPy, CasADi/CVODES, Gymnasium, Stable-Baselines3 vector environments, pandas, pytest.

**Design:** `docs/superpowers/specs/2026-07-14-minimal-feasibility-action-shield-design.md`

---

## File map

- Create `src/gl_gym/environments/action_shield.py`: immutable configuration/evidence types, reference-action conversion, candidate construction, and first-feasible projection.
- Create `src/gl_gym/environments/shielded_tomato_env.py`: distinct transactional environment and exact step commit semantics.
- Modify `src/gl_gym/RL/utils.py`: register the distinct environment and allow evaluation loaders to select it without changing training defaults.
- Modify `src/gl_gym/experiments/suite_evaluation.py`: load a shielded task environment and propagate the executed action into traces and inference memory.
- Create `src/gl_gym/experiments/shield_evaluation.py`: episode aggregation, paired shield-versus-unshielded gates, provenance validation, and atomic publication.
- Create `experiments/scripts/run_shield_stage1.py`: replay the known failing episode through the legal action grid and publish the mechanism decision.
- Create `experiments/scripts/run_shielded_context_ab.py`: run the 32 shielded diagnostic episodes, compare against the preserved unshielded evidence, and publish the Stage 2 decision.
- Modify `experiments/scripts/evaluate_suite.py`: opt-in shielded method identifier/output root and prerequisite Stage 2 decision.
- Create `experiments/scripts/evaluate_shield_gate.py`: compare full-suite shielded/unshielded tables and publish the Stage 3 decision.
- Create focused tests under `tests/environments/` and `tests/experiments/`; modify existing evaluation tests only where the executed-action contract is extended.

### Task 1: Pure action projection and immutable evidence

**Files:**
- Create: `src/gl_gym/environments/action_shield.py`
- Create: `tests/environments/test_action_shield.py`

- [ ] **Step 1: Write failing tests for configuration, legal reference conversion, and candidate order**

```python
import numpy as np
import pytest

from gl_gym.environments.action_shield import (
    DEFAULT_LAMBDAS,
    ActionShieldConfig,
    build_candidates,
    control_to_reference_action,
)


def test_reference_action_is_rate_limited_and_candidates_are_ordered():
    previous = np.array([0.5, 0.2])
    target = np.array([1.0, 0.1])
    delta = np.array([0.1, 0.1])
    reference = control_to_reference_action(target, previous, delta)
    np.testing.assert_allclose(reference, [1.0, -1.0])
    policy = np.array([0.0, 0.5])
    candidates = build_candidates(policy, reference, DEFAULT_LAMBDAS)
    assert [candidate.lambda_value for candidate in candidates] == list(DEFAULT_LAMBDAS)
    np.testing.assert_allclose(candidates[0].action, policy * (15 / 16) + reference / 16)


@pytest.mark.parametrize(
    "delta",
    [np.array([0.0, 0.1]), np.array([-0.1, 0.1]), np.array([np.nan, 0.1])],
)
def test_reference_action_rejects_invalid_delta(delta):
    with pytest.raises(ValueError, match="delta_u_max"):
        control_to_reference_action(np.ones(2), np.zeros(2), delta)


def test_default_configuration_is_preregistered_and_immutable():
    config = ActionShieldConfig()
    assert config.lambdas == (1 / 16, 1 / 8, 1 / 4, 1 / 2, 1.0)
    with pytest.raises(Exception):
        config.lambdas = (1.0,)
```

- [ ] **Step 2: Run the focused test and verify RED**

Run: `python -m pytest tests/environments/test_action_shield.py -q`

Expected: collection fails with `ModuleNotFoundError: No module named 'gl_gym.environments.action_shield'`.

- [ ] **Step 3: Implement validated pure types and construction functions**

```python
from dataclasses import dataclass
from typing import Callable, Sequence
import time

import numpy as np

DEFAULT_LAMBDAS = (1 / 16, 1 / 8, 1 / 4, 1 / 2, 1.0)


@dataclass(frozen=True, slots=True)
class ActionShieldConfig:
    lambdas: tuple[float, ...] = DEFAULT_LAMBDAS
    schema_version: str = "minimal-feasibility-action-shield-v1"

    def __post_init__(self) -> None:
        values = np.asarray(self.lambdas, dtype=float)
        if values.shape != (5,) or tuple(values) != DEFAULT_LAMBDAS:
            raise ValueError(f"lambdas must equal {DEFAULT_LAMBDAS}")


@dataclass(frozen=True, slots=True)
class ActionCandidate:
    lambda_value: float
    action: np.ndarray


@dataclass(frozen=True, slots=True)
class CandidateAttempt:
    lambda_value: float
    action: np.ndarray
    success: bool
    elapsed_seconds: float
    exception_type: str | None
    exception_message: str | None


@dataclass(frozen=True, slots=True)
class ProjectionResult:
    selected: ActionCandidate | None
    final_state: np.ndarray | None
    attempts: tuple[CandidateAttempt, ...]


def _vector(name: str, value: np.ndarray) -> np.ndarray:
    result = np.asarray(value, dtype=float).reshape(-1).copy()
    if result.size == 0 or not np.isfinite(result).all():
        raise ValueError(f"{name} must be a finite nonempty vector")
    result.setflags(write=False)
    return result


def control_to_reference_action(target, previous, delta_u_max) -> np.ndarray:
    target = _vector("target_control", target)
    previous = _vector("previous_control", previous)
    delta = _vector("delta_u_max", delta_u_max)
    if target.shape != previous.shape or target.shape != delta.shape:
        raise ValueError("target, previous, and delta_u_max shapes must match")
    if (delta <= 0).any():
        raise ValueError("delta_u_max must be strictly positive")
    return np.clip((target - previous) / delta, -1.0, 1.0)


def build_candidates(policy_action, reference_action, lambdas=DEFAULT_LAMBDAS):
    policy = _vector("policy_action", policy_action)
    reference = _vector("reference_action", reference_action)
    if policy.shape != reference.shape:
        raise ValueError("policy and reference action shapes must match")
    if (np.abs(policy) > 1).any() or (np.abs(reference) > 1).any():
        raise ValueError("actions must lie in [-1, 1]")
    return tuple(
        ActionCandidate(float(weight), (1 - weight) * policy + weight * reference)
        for weight in lambdas
    )
```

- [ ] **Step 4: Add and implement first-feasible selection tests**

```python
def test_projection_selects_first_success_and_stops():
    calls = []
    def integrate(action):
        calls.append(action.copy())
        if len(calls) < 3:
            raise RuntimeError("CV_CONV_FAILURE")
        return np.array([7.0, 8.0])
    result = project_first_feasible(
        np.zeros(2), np.ones(2), integrate, ActionShieldConfig()
    )
    assert result.selected.lambda_value == 1 / 4
    assert len(result.attempts) == 3
    np.testing.assert_array_equal(result.final_state, [7.0, 8.0])


def test_projection_rejects_nonfinite_integrator_output_and_can_exhaust_grid():
    result = project_first_feasible(
        np.zeros(1), np.ones(1), lambda action: np.array([np.nan]), ActionShieldConfig()
    )
    assert result.selected is None
    assert result.final_state is None
    assert len(result.attempts) == 5
```

Add `project_first_feasible(policy_action, reference_action, integrate, config)` to call each candidate once, validate a finite one-dimensional final state, capture expected candidate exceptions without masking input validation errors, stop at the first success, and return detached read-only arrays.

- [ ] **Step 5: Run tests and commit**

Run: `python -m pytest tests/environments/test_action_shield.py -q`

Expected: all tests pass.

```powershell
git add -- src/gl_gym/environments/action_shield.py tests/environments/test_action_shield.py
git commit -m "feat: add minimal action feasibility projection"
```

### Task 2: Transactional shielded environment

**Files:**
- Create: `src/gl_gym/environments/shielded_tomato_env.py`
- Modify: `src/gl_gym/RL/utils.py`
- Create: `tests/environments/test_shielded_tomato_env.py`

- [ ] **Step 1: Write RED tests for distinct registration and reference-controller configuration**

```python
from gl_gym.RL.utils import ENVS
from gl_gym.environments.shielded_tomato_env import ShieldedTomatoEnv


def test_shielded_environment_is_distinct_and_registered():
    assert issubclass(ShieldedTomatoEnv, TomatoEnv)
    assert ENVS["TomatoEnv"] is TomatoEnv
    assert ENVS["ShieldedTomatoEnv"] is ShieldedTomatoEnv
```

Create a lightweight instance with `__new__`, inject deterministic arrays and a fake reference controller, and assert `configure_action_shield()` rejects missing/invalid controller parameters and stores the fixed `ActionShieldConfig` without changing `TomatoEnv`.

- [ ] **Step 2: Run the focused test and verify RED**

Run: `python -m pytest tests/environments/test_shielded_tomato_env.py -q`

Expected: collection fails because `shielded_tomato_env` does not exist.

- [ ] **Step 3: Add the distinct environment and registry entry**

```python
class ShieldedTomatoEnv(TomatoEnv):
    """Evaluation-only TomatoEnv variant with transactional feasibility projection."""

    def __init__(self, *args, action_shield_params: dict, **kwargs):
        super().__init__(*args, **kwargs)
        self._shield_config = ActionShieldConfig()
        self._reference_controller = RuleBasedController(**dict(action_shield_params))

    def _reference_control(self, x0: np.ndarray, weather: np.ndarray) -> np.ndarray:
        value = self._reference_controller.predict(x0.copy(), weather.copy(), self)
        result = np.asarray(value, dtype=float).reshape(-1)
        if result.shape != (self.nu,) or not np.isfinite(result).all():
            raise ValueError("reference controller returned an invalid control")
        return result
```

Register `"ShieldedTomatoEnv": ShieldedTomatoEnv` in `ENVS`. Do not alter the `"TomatoEnv"` entry.

- [ ] **Step 4: Write transactional retry tests before implementing `step`**

Use a fixture that bypasses heavy weather loading, supplies a fake reward/observation path, and injects an integrator factory whose original call and first two retry calls fail. Assert:

```python
obs, reward, terminated, truncated, info = env.step(np.array([0.0]))
assert env.timestep == 1
assert env.integrator_factory_calls == 3
assert info["action_shield"]["intervened"] is True
assert info["action_shield"]["selected_lambda"] == 1 / 4
np.testing.assert_allclose(info["action_shield"]["requested_action"], [0.0])
np.testing.assert_allclose(info["action_shield"]["executed_action"], [0.25])
assert env.reward_calls == 1
assert env.observation_calls == 1
```

Also assert identical `x0`, `weather`, sampled parameters, and `p_dyn` reach every attempt; each retry receives a different integrator object; RNG sampling occurs once; all-failure raises a formal error without advancing time/reward/state; and a successful original action creates `selected_lambda == 0` without building a retry integrator or invoking the reference controller.

- [ ] **Step 5: Implement transactional `step` with a single commit point**

Add focused helpers in `ShieldedTomatoEnv`:

```python
def _new_formal_integrator(self):
    return define_model(
        nx=self.nx, nu=self.nu, nd=self.nd, n_params=self.num_params,
        dt=self.dt, integrator_options=dict(FORMAL_CVODES_OPTIONS),
    )

def _integrate_once(self, integrator, x0, control, p_dyn):
    result = integrator(x0=ca.DM(x0), u=ca.DM(control), p=p_dyn)
    final = np.asarray(result["xf"], dtype=float).reshape(-1)
    if final.shape != (self.nx,) or not np.isfinite(final).all():
        raise RuntimeError("formal integrator returned an invalid final state")
    return final
```

The overridden `step` must snapshot pre-step fields, sample uncertain parameters exactly once, attempt `self.F` once for the policy action, and only then construct fresh formal integrators for projected candidates. After selection, assign `self.x`, `self.u`, time fields, observation, reward, info, timestep, and `x_prev` exactly once. Attach a complete `info["action_shield"]` record and retain the existing ODE diagnostic record for the original failed action. On exhaustion, restore all snapshot fields and raise `RuntimeError("action shield exhausted all legal candidates")` with candidate failures attached as notes.

- [ ] **Step 6: Run environment regression tests and commit**

Run: `python -m pytest tests/environments/test_action_shield.py tests/environments/test_shielded_tomato_env.py tests/environments/test_tomato_ode_diagnostics.py -q`

Expected: all pass; unshielded diagnostics remain unchanged.

```powershell
git add -- src/gl_gym/environments/shielded_tomato_env.py src/gl_gym/RL/utils.py tests/environments/test_shielded_tomato_env.py
git commit -m "feat: add transactional shielded tomato environment"
```

### Task 3: Shielded environment loading and executed-action trajectory semantics

**Files:**
- Modify: `src/gl_gym/experiments/suite_evaluation.py`
- Modify: `tests/experiments/test_suite_evaluation.py`

- [ ] **Step 1: Write RED tests for the shield loader**

```python
def test_load_task_env_can_select_shielded_variant(monkeypatch, tmp_path):
    captured = {}
    monkeypatch.setattr("gl_gym.RL.utils.make_vec_env", lambda env_id, *a, **k: captured.setdefault("env_id", env_id) or FakeVecEnv())
    load_task_env(suite, task, tmp_path / "stats.pkl", shield_params={"one": 1})
    assert captured["env_id"] == "ShieldedTomatoEnv"
```

The real test should mock `VecNormalize.load`, assert `action_shield_params` is added only to a copied environment-specific mapping, and assert the default call still requests `suite.env_id` with no shield parameters.

- [ ] **Step 2: Extend `load_task_env` minimally**

Change its signature to:

```python
def load_task_env(suite, task, vecnormalize_path, *, shield_params=None):
    selected_env_id = "ShieldedTomatoEnv" if shield_params is not None else suite.env_id
    if shield_params is not None:
        env_specific_params = dict(env_specific_params)
        env_specific_params["action_shield_params"] = dict(shield_params)
```

Retain all existing path validation, cleanup, normalization, and evaluation settings.

- [ ] **Step 3: Write RED tests for requested versus executed actions**

```python
def test_episode_uses_executed_action_for_trace_and_context_memory():
    env = ExecutedActionEnv(requested=np.array([0.0]), executed=np.array([0.25]))
    model = HookedFakeModel()
    _, diagnostics = run_deterministic_episode(
        model, env, inference_mode="online_context", return_diagnostics=True
    )
    np.testing.assert_allclose(diagnostics["requested_action_trace"][0], [0.0])
    np.testing.assert_allclose(diagnostics["action_trace"][0], [0.25])
    assert diagnostics["action_shield_records"][0]["selected_lambda"] == 0.25
    observe = next(event for event in model.events if event[0] == "observe")
    np.testing.assert_allclose(observe[2], [0.25])
```

- [ ] **Step 4: Implement the executed-action contract**

After `env.step(actions)`, derive:

```python
requested_action = np.asarray(actions[0], dtype=np.float32).copy()
shield_info = info.get("action_shield")
executed_action = requested_action if shield_info is None else np.asarray(
    shield_info["executed_action"], dtype=np.float32
).copy()
```

Validate shape and finiteness. Store executed actions in the existing `action_trace`, pass `executed_action` to `observe_inference_transition`, and collect detached `action_shield_records`. When at least one step contains shield evidence, return `requested_action_trace` and `action_shield_records` in diagnostics; otherwise retain the exact existing diagnostics key set so the unshielded context runner remains compatible and its traces remain byte-equivalent.

- [ ] **Step 5: Run evaluation tests and commit**

Run: `python -m pytest tests/experiments/test_suite_evaluation.py -q`

Expected: all existing and new tests pass.

```powershell
git add -- src/gl_gym/experiments/suite_evaluation.py tests/experiments/test_suite_evaluation.py
git commit -m "feat: propagate executed actions through evaluation"
```

### Task 4: Shield evidence aggregation and preregistered gates

**Files:**
- Create: `src/gl_gym/experiments/shield_evaluation.py`
- Create: `tests/experiments/test_shield_evaluation.py`

- [ ] **Step 1: Write RED tests for episode aggregation**

```python
def test_episode_aggregation_uses_committed_steps_as_denominator():
    records = [no_intervention(), intervention(lambda_value=0.25, attempts=3)]
    result = aggregate_episode_interventions(records, action_dim=2)
    assert result["total_steps"] == 2
    assert result["intervention_count"] == 1
    assert result["intervention_rate"] == 0.5
    assert result["extra_solver_attempts"] == 3
    assert result["first_intervention_step"] == 1
```

Add rejection tests for duplicate/nonconsecutive steps, nonfinite fields, invalid lambda values, mismatched action dimensions, and candidate probes incorrectly marked as committed steps.

- [ ] **Step 2: Implement strict episode aggregation**

Create `aggregate_episode_interventions(records, action_dim)` returning only JSON-safe scalars/lists: counts, rate, first step, lambda mean/max, norm mean/max, per-channel counts, extra attempts, elapsed overhead, and ODE failure count. Empty episodes and malformed records raise `ValueError`.

- [ ] **Step 3: Write RED tests for paired thresholds and exclusions**

```python
def test_gate_passes_exact_boundaries_and_reports_unpaired_failures():
    decision = evaluate_shield_gate(
        shielded=shield_rows(intervention_rate=0.005, return_scale=0.98, violation_scale=1.05),
        unshielded=unshielded_rows(one_failure=True),
        expected_keys=EXPECTED_KEYS,
    )
    assert decision["conditions"] == {
        "zero_ode_failures": True,
        "intervention_rate_within_0p5pct": True,
        "paired_return_loss_within_2pct": True,
        "paired_violation_burden_within_5pct": True,
    }
    assert decision["evidence"]["unshielded_failure_count"] == 1
```

Add separate tests just over each threshold, zero/zero violation neutrality, a new shield-only violation over a zero baseline, missing expected keys, duplicate keys, and exclusion of incomplete unshielded episodes from paired deltas without excluding them from completion reporting.

- [ ] **Step 4: Implement exact gate formulas and atomic publication**

Use `EPSILON = 1e-9`. Compute relative return loss as `max(0, -mean(shielded - unshielded) / (mean(abs(unshielded)) + EPSILON))`. Compute each violation ratio as shielded divided by `abs(unshielded) + EPSILON`, replacing zero/zero with `1.0`, then average across the three existing metrics and all completed pairs. Implement `write_shield_artifacts_atomic()` to stage and publish `eval_raw.csv`, `paired_deltas.csv`, `interventions.csv`, `shield_manifest.json`, and `decision.json`, preserving any valid prior root when publication fails.

- [ ] **Step 5: Run tests and commit**

Run: `python -m pytest tests/experiments/test_shield_evaluation.py -q`

Expected: all tests pass.

```powershell
git add -- src/gl_gym/experiments/shield_evaluation.py tests/experiments/test_shield_evaluation.py
git commit -m "feat: add action shield evidence gates"
```

### Task 5: Stage 1 known-failure mechanism CLI

**Files:**
- Create: `experiments/scripts/run_shield_stage1.py`
- Create: `tests/experiments/test_shield_stage1_cli.py`

- [ ] **Step 1: Write RED CLI tests**

Test injected capsule loading, exact previous-control/reference conversion, fixed candidate order, fresh integrator factory calls, smallest-success selection, and atomic output. The passing fixture must assert the JSON decision contains:

```python
assert decision == {
    "outcome": "continue_to_context_ab",
    "conditions": {
        "original_reproduced": True,
        "legal_candidate_succeeded": True,
        "smallest_success_selected": True,
        "intervention_recorded": True,
    },
    "selected_lambda": 0.25,
}
```

Also test that a nonreproducing original or exhausted grid produces `outcome == "redesign_action_shield"`, and that output cannot overlap a formal result root or the capsule directory.

- [ ] **Step 2: Implement the CLI around existing capsule/replay validation**

The CLI accepts `--capsule_manifest`, `--output_root`, and `--formal_result_root`. It loads the immutable capsule through the existing loader, builds the rule controller through `build_rule_based_controller`, converts raw rule control to a legal action using stored `previous_control` and environment `delta_u_max` from the supplied environment configuration, and runs the fixed projector with a fresh exact-formal integrator per candidate. It writes `stage1_results.json`, `stage1_states.npz`, and `decision.json` atomically.

- [ ] **Step 3: Run tests and commit**

Run: `python -m pytest tests/experiments/test_shield_stage1_cli.py tests/experiments/test_ode_replay.py -q`

Expected: all pass.

```powershell
git add -- experiments/scripts/run_shield_stage1.py tests/experiments/test_shield_stage1_cli.py
git commit -m "feat: validate shield on captured ODE failure"
```

### Task 6: Stage 2 shielded context A/B runner

**Files:**
- Create: `experiments/scripts/run_shielded_context_ab.py`
- Create: `tests/experiments/test_shielded_context_ab.py`
- Modify: `src/gl_gym/experiments/shield_evaluation.py`

- [ ] **Step 1: Write RED tests for prerequisite and provenance enforcement**

The runner must refuse to start unless Stage 1 `decision.json` is structurally valid, has `outcome == "continue_to_context_ab"`, and matches the checkpoint, source revision, solver options, rule-controller hash, and lambda grid requested for Stage 2. Tests must also reject overlap with the preserved unshielded result root and its `.work`/staging siblings.

- [ ] **Step 2: Write RED tests for all 32 shielded episode records**

Reuse the injectable patterns in `tests/experiments/test_context_ab.py`. Assert exact seeds `(42, 123)`, eight task IDs, both inference modes, distinct `method == "minimal_feasibility_shield_v1"`, requested/executed trace paths, one intervention file per episode, resume checks that include the shield fingerprint, and refusal to publish 31 rows.

- [ ] **Step 3: Implement the isolated runner**

Import selection/model-loading helpers from `run_context_ab.py` instead of copying their logic. Load `configs/agents/rule_based.yml`, pass its `TomatoEnv` mapping to `load_task_env(..., shield_params=...)`, call `run_deterministic_episode(..., return_diagnostics=True, failure_recorder=...)`, persist both traces and the action-shield step records, and use `write_shield_artifacts_atomic()` for final output. Add the fixed shield fields and hashes to the manifest and every resume row.

- [ ] **Step 4: Evaluate the Stage 2 decision against preserved unshielded evidence**

Require an explicit `--unshielded_result_root`. Validate its formal provenance and failure evidence rather than rerunning or overwriting it. Join only jointly completed episode keys for return/violation deltas, report its failure count separately, and emit `outcome == "continue_to_full_suite"` only when all four shield gates pass.

- [ ] **Step 5: Run tests and commit**

Run: `python -m pytest tests/experiments/test_shielded_context_ab.py tests/experiments/test_context_ab.py tests/experiments/test_shield_evaluation.py -q`

Expected: all pass.

```powershell
git add -- experiments/scripts/run_shielded_context_ab.py src/gl_gym/experiments/shield_evaluation.py tests/experiments/test_shielded_context_ab.py
git commit -m "feat: run shielded context diagnostic gate"
```

### Task 7: Stage 3 full-suite integration and gate

**Files:**
- Modify: `experiments/scripts/evaluate_suite.py`
- Create: `experiments/scripts/evaluate_shield_gate.py`
- Create: `tests/experiments/test_suite_evaluation_cli.py`
- Create: `tests/experiments/test_shield_gate_cli.py`

- [ ] **Step 1: Write RED tests for opt-in full-suite shield execution**

Extend the evaluation CLI with `--action_shield`, `--stage2_decision`, and `--result_root`. Tests assert default arguments preserve current behavior, shield mode requires a passing and provenance-matching Stage 2 decision, loads `ShieldedTomatoEnv`, writes `algorithm` as `<base_algorithm>__minimal_feasibility_shield_v1`, and refuses a result root overlapping the unshielded suite.

- [ ] **Step 2: Implement shield mode without changing default suite evaluation**

When `--action_shield` is present, load fixed rule parameters, pass them to `load_task_env`, capture shield diagnostics, and append intervention evidence alongside `eval_raw.csv`. Keep the existing task/seed filters for testing, but the final gate CLI must reject any table that is not the complete expected 91-task protocol for every approved seed/method key.

- [ ] **Step 3: Write and implement the Stage 3 comparison CLI**

`evaluate_shield_gate.py` accepts `--manifest`, `--tasks_csv`, `--unshielded_eval`, `--shielded_eval`, `--interventions`, `--stage2_decision`, and `--output_root`. It validates complete keys and provenance, calls `evaluate_shield_gate`, and atomically writes paired deltas, summary, manifest, and decision. The passing outcome is `paper_evidence_ready`; every other outcome is `redesign_before_claim`.

- [ ] **Step 4: Run tests and commit**

Run: `python -m pytest tests/experiments/test_suite_evaluation_cli.py tests/experiments/test_shield_gate_cli.py tests/experiments/test_suite_evaluation.py -q`

Expected: all pass.

```powershell
git add -- experiments/scripts/evaluate_suite.py experiments/scripts/evaluate_shield_gate.py tests/experiments/test_suite_evaluation_cli.py tests/experiments/test_shield_gate_cli.py
git commit -m "feat: gate full suite action shield evaluation"
```

### Task 8: Full verification, real staged execution, and evidence audit

**Files:**
- Modify only if validation exposes a defect in files owned by Tasks 1-7.
- Produce runtime artifacts only under `artifacts/results/diagnostics/action-shield/` and distinct shielded result roots on `E:`.

- [ ] **Step 1: Run the complete automated test suite**

Run: `python -m pytest -q`

Expected: at least the current baseline of 427 passing tests plus all new tests, with zero failures. Existing skips/warnings may remain only if their identities match the pre-change run.

- [ ] **Step 2: Run compilation and artifact-boundary checks**

Run: `python -m compileall -q src experiments/scripts tests`

Expected: exit code 0.

Run: `git diff --check HEAD~7..HEAD`

Expected: no whitespace errors.

- [ ] **Step 3: Execute Stage 1 with the captured failure**

Run `experiments/scripts/run_shield_stage1.py` against the immutable seed-123 capsule at the recorded diagnostic path, publishing to `artifacts/results/diagnostics/action-shield/stage1-seed123-heldout2011-zero-context`.

Expected: `decision.json` says `continue_to_context_ab`, reproduces the original failure, and selects the smallest successful legal lambda. If it does not, stop; preserve evidence and redesign rather than changing the preregistered grid.

- [ ] **Step 4: Execute the 32-episode Stage 2 diagnostic**

Run `run_shielded_context_ab.py` with the existing two checkpoints, approved source manifest/tasks, the preserved unshielded diagnostic root, and a new shielded root.

Expected: exactly 32 complete rows, zero shielded ODE failures, intervention rate at most 0.005, paired return loss at most 0.02, normalized violation burden at most 1.05, and `outcome == "continue_to_full_suite"`. Otherwise stop and use the recorded intervention distribution to redesign.

- [ ] **Step 5: Execute and gate the complete Stage 3 suite only after Stage 2 passes**

Run the full existing 91-task protocol for the approved seeds with `--action_shield`, then run `evaluate_shield_gate.py` against preserved unshielded results.

Expected: complete key coverage and `outcome == "paper_evidence_ready"`. Do not run Stage 3 if the Stage 2 decision is absent, invalid, or failing.

- [ ] **Step 6: Audit scientific claims and repository state**

Verify machine-readable decisions agree with CSV/NPZ evidence; unshielded failures remain visible; shielded and unshielded roots are disjoint; method/checkpoint/source/config hashes match; no partial output is presented as formal; and no files were installed or cached on `C:`. Record exact commands, elapsed time, artifact paths, and observed gate values in the final handoff.

- [ ] **Step 7: Commit any verification-only documentation correction**

If no correction is required, do not create an empty commit. If a command/path correction to this plan is necessary, stage only the plan and commit:

```powershell
git add -- docs/superpowers/plans/2026-07-14-minimal-feasibility-action-shield.md
git commit -m "docs: correct action shield execution plan"
```
