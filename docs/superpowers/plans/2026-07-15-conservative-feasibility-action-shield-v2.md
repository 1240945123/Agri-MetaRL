# Conservative Feasibility Action Shield v2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace v1's increasing-lambda recovery priority with a separately fingerprinted v2 descending-lambda priority, then rerun the unchanged Stage 1, Stage 2, and Stage 3 evidence gates.

**Architecture:** Keep the policy trigger, reference controller, convex action set, solver, and transactional environment unchanged. Version the pure projector and every downstream evidence contract so v2 records require the exact descending candidate prefix and cannot be confused with v1. Preserve all v1 artifacts and publish v2 to new roots.

**Tech Stack:** Python 3.11, NumPy, pandas, Gymnasium, Stable-Baselines3, CasADi/CVODES, pytest, JSON/CSV/NPY evidence artifacts.

---

## File map

- `src/gl_gym/environments/action_shield.py`: owns the v2 fixed lambda priority and pure first-feasible projection.
- `src/gl_gym/environments/shielded_tomato_env.py`: executes the projector transactionally and emits v2 step records.
- `src/gl_gym/experiments/suite_evaluation.py`: validates per-step shield records during complete episodes.
- `src/gl_gym/experiments/shield_evaluation.py`: aggregates v2 intervention evidence and applies unchanged numerical gates.
- `experiments/scripts/run_shield_stage1.py`: produces the v2 known-failure mechanism decision.
- `experiments/scripts/run_shielded_context_ab.py`: authenticates Stage 1 and produces the v2 32-episode decision.
- `experiments/scripts/evaluate_suite.py`: produces complete v2 full-suite artifacts and fingerprints.
- `experiments/scripts/evaluate_shield_gate.py`: authenticates v2 evidence and applies unchanged Stage 3 thresholds.
- `tests/environments/test_action_shield.py`: pure order and early-stop contract.
- `tests/environments/test_shielded_tomato_env.py`: transactional v2 record contract.
- `tests/experiments/test_suite_evaluation.py`: episode-record validation.
- `tests/experiments/test_shield_evaluation.py`: aggregation and gate regression.
- `tests/experiments/test_shield_stage1.py`: Stage 1 v2 decision and provenance.
- `tests/experiments/test_shielded_context_ab.py`: Stage 2 v2 prerequisite and publication.
- `tests/experiments/test_suite_evaluation_cli.py`: Stage 3 v2 evaluation and resume contract.
- `tests/experiments/test_shield_gate_cli.py`: Stage 3 v2 authentication and unchanged gates.

### Task 1: Version the pure projector and descending priority

**Files:**
- Modify: `src/gl_gym/environments/action_shield.py`
- Modify: `tests/environments/test_action_shield.py`

- [ ] **Step 1: Write failing order and schema tests**

Add assertions equivalent to:

```python
def test_v2_candidates_use_fixed_descending_priority():
    candidates = build_candidates(np.zeros(2), np.ones(2))
    assert [item.lambda_value for item in candidates] == [1.0, 0.5, 0.25, 0.125, 0.0625]
    assert ActionShieldConfig().schema_version == "conservative-feasibility-action-shield-v2"

def test_v2_projection_stops_at_first_feasible_in_descending_priority():
    calls = []
    def integrate(action):
        calls.append(action.copy())
        if len(calls) < 3:
            raise RuntimeError("infeasible")
        return np.array([2.0])
    result = project_first_feasible(np.zeros(1), np.ones(1), integrate, ActionShieldConfig())
    assert [item.lambda_value for item in result.attempts] == [1.0, 0.5, 0.25]
    assert result.selected.lambda_value == 0.25
```

- [ ] **Step 2: Run the focused tests and verify RED**

Run:

```powershell
$env:TEMP=(Resolve-Path 'tmp\codex-pytest'); $env:TMP=$env:TEMP
python -m pytest tests/environments/test_action_shield.py -q
```

Expected: failures show the old ascending order and v1 schema.

- [ ] **Step 3: Implement the minimal v2 constants**

Set the canonical tuple and schema to:

```python
DEFAULT_LAMBDAS = (1.0, 1.0 / 2.0, 1.0 / 4.0, 1.0 / 8.0, 1.0 / 16.0)
_SCHEMA_VERSION = "conservative-feasibility-action-shield-v2"
```

Retain exact-tuple validation and the existing loop in `project_first_feasible`; the loop now inherits the descending priority without additional branches.

- [ ] **Step 4: Run focused tests and verify GREEN**

Run the Task 1 command. Expected: all tests in `test_action_shield.py` pass after updating old order expectations to v2.

- [ ] **Step 5: Commit**

```powershell
git add -- src/gl_gym/environments/action_shield.py tests/environments/test_action_shield.py
git commit -m "feat: prioritize conservative shield candidates"
```

### Task 2: Propagate the v2 step-record contract

**Files:**
- Modify: `src/gl_gym/environments/shielded_tomato_env.py`
- Modify: `src/gl_gym/experiments/suite_evaluation.py`
- Modify: `tests/environments/test_shielded_tomato_env.py`
- Modify: `tests/experiments/test_suite_evaluation.py`

- [ ] **Step 1: Write failing transactional and validator tests**

Update fixtures to require `conservative-feasibility-action-shield-v2`, and add a validator case whose attempts are `[1.0 failed, 0.5 succeeded]`. Assert that `[0.0625 succeeded]` is rejected as a non-prefix v2 record.

```python
record["candidate_attempts"] = [
    {"lambda": 1.0, "success": False, "action": [...], "elapsed_seconds": 0.1,
     "exception_type": "RuntimeError", "exception_message": "fail"},
    {"lambda": 0.5, "success": True, "action": [...], "elapsed_seconds": 0.1,
     "exception_type": None, "exception_message": None},
]
record["selected_lambda"] = 0.5
```

- [ ] **Step 2: Run focused tests and verify RED**

```powershell
python -m pytest tests/environments/test_shielded_tomato_env.py tests/experiments/test_suite_evaluation.py -q
```

Expected: v1 schema/order assertions fail.

- [ ] **Step 3: Update validation to use the v2 canonical prefix**

Import the fixed tuple from `action_shield` instead of duplicating numeric order. Require attempted lambdas to equal `DEFAULT_LAMBDAS[:len(attempts)]`, require only the last attempt to succeed, and require its lambda to equal `selected_lambda`.

- [ ] **Step 4: Run focused tests and verify GREEN**

Run the Task 2 command. Expected: all tests pass.

- [ ] **Step 5: Commit**

```powershell
git add -- src/gl_gym/environments/shielded_tomato_env.py src/gl_gym/experiments/suite_evaluation.py tests/environments/test_shielded_tomato_env.py tests/experiments/test_suite_evaluation.py
git commit -m "fix: validate conservative shield records"
```

### Task 3: Version Stage 1 and Stage 2 evidence

**Files:**
- Modify: `experiments/scripts/run_shield_stage1.py`
- Modify: `experiments/scripts/run_shielded_context_ab.py`
- Modify: `src/gl_gym/experiments/shield_evaluation.py`
- Modify: `tests/experiments/test_shield_stage1.py`
- Modify: `tests/experiments/test_shielded_context_ab.py`
- Modify: `tests/experiments/test_shield_evaluation.py`

- [ ] **Step 1: Write failing v2 provenance tests**

Require Stage 1 and Stage 2 manifests, rows, and decisions to bind the descending `fixed_lambdas`, v2 schema, and a v2 method identifier such as `conservative_feasibility_shield_v2`. Add a negative test showing an otherwise valid v1 Stage 1 decision is rejected.

- [ ] **Step 2: Run focused tests and verify RED**

```powershell
python -m pytest tests/experiments/test_shield_stage1.py tests/experiments/test_shielded_context_ab.py tests/experiments/test_shield_evaluation.py -q
```

Expected: fixtures or loaders still accept/bind v1 identities.

- [ ] **Step 3: Implement v2 identifiers without changing thresholds**

Use the imported v2 schema and descending tuple in source checksums, fixed-grid comparisons, manifests, decisions, and row records. Keep these gates exact:

```python
MAX_INTERVENTION_RATE = 0.005
MAX_RELATIVE_RETURN_LOSS = 0.02
MAX_VIOLATION_RATIO = 1.05
MAX_ODE_FAILURES = 0
```

Stage 1 must require selection of the first successful candidate in the descending fixed order; remove wording or checks that require the numerically smallest lambda.

- [ ] **Step 4: Run focused tests and verify GREEN**

Run the Task 3 command. Expected: all tests pass, including explicit v1 rejection.

- [ ] **Step 5: Commit**

```powershell
git add -- experiments/scripts/run_shield_stage1.py experiments/scripts/run_shielded_context_ab.py src/gl_gym/experiments/shield_evaluation.py tests/experiments/test_shield_stage1.py tests/experiments/test_shielded_context_ab.py tests/experiments/test_shield_evaluation.py
git commit -m "feat: version conservative shield evidence"
```

### Task 4: Version Stage 3 evaluator and gate

**Files:**
- Modify: `experiments/scripts/evaluate_suite.py`
- Modify: `experiments/scripts/evaluate_shield_gate.py`
- Modify: `tests/experiments/test_suite_evaluation_cli.py`
- Modify: `tests/experiments/test_shield_gate_cli.py`

- [ ] **Step 1: Write failing Stage 3 v2 identity tests**

Require shielded rows and manifests to use the v2 algorithm suffix, method, schema, descending fixed-lambda JSON, and method fingerprint. Add a negative test that a v1 manifest/row cannot pass `audit_stage3_artifacts` or the gate loader.

- [ ] **Step 2: Run focused tests and verify RED**

```powershell
python -m pytest tests/experiments/test_suite_evaluation_cli.py tests/experiments/test_shield_gate_cli.py -q
```

Expected: v1 method constants and fixtures fail the new assertions.

- [ ] **Step 3: Implement v2 Stage 3 constants and fingerprints**

Set distinct v2 `SHIELD_SUFFIX` and `SHIELD_METHOD`, import the canonical descending tuple, and ensure `shield_method_fingerprint_components` binds the order exactly. Do not change unshielded method identifiers, pairing keys, task counts, or numerical gate formulas.

- [ ] **Step 4: Run focused tests and verify GREEN**

Run the Task 4 command. Expected: all tests pass.

- [ ] **Step 5: Commit**

```powershell
git add -- experiments/scripts/evaluate_suite.py experiments/scripts/evaluate_shield_gate.py tests/experiments/test_suite_evaluation_cli.py tests/experiments/test_shield_gate_cli.py
git commit -m "feat: gate conservative shield full suite"
```

### Task 5: Full software verification

**Files:**
- Verify only; do not stage unrelated repository reorganization changes.

- [ ] **Step 1: Run integrity and experiment tests**

```powershell
$env:TEMP=(Resolve-Path 'tmp\codex-pytest'); $env:TMP=$env:TEMP
python -m pytest tests/integrity tests/environments tests/experiments -q
```

Expected: all selected tests pass.

- [ ] **Step 2: Run the complete suite**

```powershell
python -m pytest -q
```

Expected: all tests pass with only known third-party warnings/skips.

- [ ] **Step 3: Confirm source state**

```powershell
git status --short --branch
git log -6 --oneline
```

Expected: v2 commits are present; unrelated pre-existing reorganization changes remain unstaged.

### Task 6: Execute the v2 evidence sequence

**Files:**
- Create generated artifacts below `artifacts/results/` using roots containing `v2-conservative`; do not overwrite v1 roots.

- [ ] **Step 1: Run Stage 1 v2**

Replay the existing authenticated seed-123 failure capsule with `run_shield_stage1.py`, writing a new v2 root. Expected decision: all mechanism conditions true and `continue_to_context_ab`.

- [ ] **Step 2: Run Stage 2 v2**

Run `run_shielded_context_ab.py` against the immutable unshielded 32-row comparator and the new Stage 1 decision. Expected: exactly 32 rows, zero shielded ODE failures, and all four unchanged gates true.

- [ ] **Step 3: Regenerate the formal unshielded Stage 3 comparator**

Run `evaluate_suite.py --formal_unshielded_provenance --resume_eval` with the canonical manifest, derived two-seed runs CSV, canonical tasks CSV, and new Stage 2 decision. Expected: 182 rows with explicit completion/failure evidence and atomic publication.

- [ ] **Step 4: Run the formal v2 shielded Stage 3 suite**

Run `evaluate_suite.py --action_shield --resume_eval` with the same protocol and new v2 Stage 2 decision. Expected: 182 completed rows, zero ODE failures, complete intervention traces, and atomic publication.

- [ ] **Step 5: Run and audit the Stage 3 gate**

Run `evaluate_shield_gate.py` with the new unshielded root, new shielded root, new intervention table, and new Stage 2 decision. Then call `audit_stage3_artifacts` on the published gate root. Expected: authenticated paired evidence and a decision determined solely by the four unchanged thresholds.

- [ ] **Step 6: Record the scientific result**

Summarize v1 failure, v2 completion, intervention count/rate, paired return loss, normalized violation burden, per-split results, and remaining limitations in a new Markdown evidence note. Do not edit manuscript claims until the gate result is known.
