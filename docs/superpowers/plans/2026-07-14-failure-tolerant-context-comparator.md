# Failure-Tolerant Context Comparator Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce an immutable exact-32 unshielded context comparator that records genuine ODE failures and continues remaining episodes.

**Architecture:** A new CLI reuses the existing context task/checkpoint/provenance helpers and episode runner. It classifies failures only through attempt-local validated capsules, maintains canonical work-only progress, then atomically publishes a comparator accepted by the existing shielded Stage 2 loader.

**Tech Stack:** Python, NumPy, pandas, Stable-Baselines3, pytest, existing ODE capsule and context evaluation modules.

**Design:** `docs/superpowers/specs/2026-07-14-failure-tolerant-context-comparator-design.md`

---

### Task 1: Comparator runner and failure classifier

**Files:**
- Create: `experiments/scripts/run_unshielded_context_comparator.py`
- Create: `tests/experiments/test_unshielded_context_comparator.py`

- [ ] **Step 1: Write RED tests for exact protocol and ordinary success**

Create injectable fixtures for two seeds, eight canonical tasks, and both modes. Assert exactly 32 runner calls and rows with:

```python
assert set(raw[["seed", "task_id", "inference_mode"]].itertuples(index=False, name=None)) == expected_keys
assert raw["completed"].eq(True).all()
assert raw["status"].eq("completed").all()
assert raw["ode_failure_count"].eq(0).all()
```

Run: `python -m pytest tests/experiments/test_unshielded_context_comparator.py -q`

Expected: RED because the script does not exist.

- [ ] **Step 2: Implement parser, prerequisite validation, and successful rows**

Expose:

```python
def run_unshielded_comparator(
    *, suite, tasks, runs, result_root, failure_root,
    source_manifest, source_tasks_csv, device, resume,
    legacy_progress=None, model_loader=None, env_loader=load_task_env,
    episode_runner=run_deterministic_episode,
    recorder_factory=FailureCapsuleRecorder,
    provenance_loader=_provenance,
) -> pd.DataFrame:
    ...
```

The CLI accepts the same source/checkpoint arguments as `run_context_ab.py`, plus required isolated `--failure_root`, optional `--legacy_progress`, and `--resume`. Reuse approved constants and helpers rather than redefining tasks or seeds.

- [ ] **Step 3: Write RED tests for genuine ODE failure continuation**

Use the real `run_deterministic_episode` wrapper behavior: the environment emits `integration_failure`, the recorder creates one capsule containing the underlying solver error, and the runner raises the early-horizon wrapper. Assert the comparator records one non-scoring row and continues later keys.

Also assert model errors, early termination without a capsule, multiple capsules, recorder errors, `KeyboardInterrupt`, and `SystemExit` stop without publishing.

- [ ] **Step 4: Implement attempt-local capsule classification**

Catch `Exception`, never `BaseException`. Accept only the exact early-horizon wrapper and exactly one new capsule. Reload it with `load_failure_capsule` and validate seed, task, mode, checkpoint, source mapping, formal result root, solver options, failure timestep, and wrapper step. Record failed metrics as NaN. Re-raise every other error with capsule-validation notes.

- [ ] **Step 5: Run focused tests and commit**

Run: `python -m pytest tests/experiments/test_unshielded_context_comparator.py tests/experiments/test_suite_evaluation.py tests/experiments/test_ode_failure.py -q --basetemp E:/t/comparator-1`

Expected: all pass.

```powershell
git add -- experiments/scripts/run_unshielded_context_comparator.py tests/experiments/test_unshielded_context_comparator.py
git commit -m "feat: add failure-tolerant context comparator"
```

### Task 2: Strict resume and legacy progress import

**Files:**
- Modify: `experiments/scripts/run_unshielded_context_comparator.py`
- Modify: `tests/experiments/test_unshielded_context_comparator.py`

- [ ] **Step 1: Write RED tests for canonical resume evidence**

Test valid completed rows skip; changed checkpoint/source/diagnostics/trace recompute; valid failed capsules skip; missing, corrupt, or mismatched capsules recompute. Test paths from another key or result root cannot be reused.

- [ ] **Step 2: Implement shared row validation**

Use canonical per-key trace and capsule locations. Validate strict checkpoint steps, mode-aware readiness, finite completed metrics, failed-row NaNs, trace contents, capsule identity, and a runtime source-tree fingerprint. Parsing failures return stale rather than aborting the whole resume.

- [ ] **Step 3: Write and implement legacy import tests**

Import only completed rows from the existing fail-fast progress CSV after the same provenance/trace/diagnostic validation. Never import a missing or inferred key. Copy accepted traces into comparator-owned work paths and re-sign row identities.

- [ ] **Step 4: Verify and commit**

Run: `python -m pytest tests/experiments/test_unshielded_context_comparator.py tests/experiments/test_context_ab.py -q --basetemp E:/t/comparator-2`

```powershell
git add -- experiments/scripts/run_unshielded_context_comparator.py tests/experiments/test_unshielded_context_comparator.py
git commit -m "feat: resume context comparator evidence"
```

### Task 3: Atomic publication and consumer round trip

**Files:**
- Modify: `experiments/scripts/run_unshielded_context_comparator.py`
- Modify: `tests/experiments/test_unshielded_context_comparator.py`

- [ ] **Step 1: Write RED publication tests**

Cover 31-row refusal, exact final files, final-only paths, old-root preservation, interrupted rename, failed atomic restore with copy fallback, sole-backup preservation, protected source/capsule topology, and no final root after partial execution.

- [ ] **Step 2: Implement atomic finalization**

Stage `eval_raw.csv`, `context_ab_manifest.json`, copied traces, and complete capsules. Rewrite paths to final-root descendants. Hash every file and bind result root, inputs, checkpoints, solver, runtime tree, and row identities in the manifest. Use filesystem-inferred backup recovery and copy fallback.

- [ ] **Step 3: Validate through the Stage 2 consumer**

After publication call `load_unshielded_comparator()` with exact provenance and checkpoints. Reject and restore the prior root if the consumer does not return all 32 keys or any capsule fails validation.

- [ ] **Step 4: Verify, commit, and run real comparator**

Run focused and full tests with short E-drive `--basetemp`, then execute the real two-seed comparator using the existing 18-row legacy progress. Inspect the final 32 rows and capsule counts before starting `run_shielded_context_ab.py`.

```powershell
git add -- experiments/scripts/run_unshielded_context_comparator.py tests/experiments/test_unshielded_context_comparator.py
git commit -m "feat: publish immutable context comparator"
```
