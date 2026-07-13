# ODE Failure Capsule and Offline Replay Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Capture the exact numerical boundary of a greenhouse ODE failure, replay it offline under predefined control and solver variants, and emit a conservative mechanism classification without changing formal evaluation semantics.

**Architecture:** `TomatoEnv` produces copied, path-independent diagnostic payloads only when explicitly enabled. A focused `ode_failure` experiment module owns the 256-step ring buffer, immutable capsule schema, validation, and atomic persistence; `suite_evaluation` only forwards transition data. A separate `ode_replay` module validates exact inputs, runs fresh CasADi integrators and rule-based counterfactuals, while a thin CLI writes reports outside the formal A/B result root.

**Tech Stack:** Python 3.11, NumPy, CasADi/CVODES, Gymnasium, Stable-Baselines3 VecEnv, pytest, JSON/JSONL, NPZ with `allow_pickle=False`, SHA-256, pathlib.

---

## File Map

- Modify `src/gl_gym/environments/models/utils.py`: expose immutable formal CVODES defaults and explicit diagnostic overrides while preserving current defaults.
- Modify `src/gl_gym/environments/tomato_env.py`: enable diagnostics and attach exact per-step and failure payloads to `info`.
- Create `src/gl_gym/experiments/ode_failure.py`: ring buffer, capsule schema, canonical hashes, atomic writer, checksum validator, and loader.
- Modify `src/gl_gym/experiments/suite_evaluation.py`: connect optional recorders to VecEnv episodes without changing the default path.
- Modify `experiments/scripts/run_context_ab.py`: accept a diagnostic failure root and create recorders for selected A/B episodes.
- Create `src/gl_gym/experiments/ode_replay.py`: replay variants, rule-based control reconstruction, outcome records, and conservative classification.
- Create `experiments/scripts/replay_ode_failure.py`: validate one capsule and atomically write JSON, NPZ, and Markdown reports.
- Create `tests/environments/test_model_utils.py`: formal-default and override tests.
- Create `tests/environments/test_tomato_ode_diagnostics.py`: environment evidence and random-draw integrity tests.
- Create `tests/experiments/test_ode_failure.py`: recorder, capsule, checksum, idempotency, and corruption tests.
- Modify `tests/experiments/test_suite_evaluation.py`: evaluator wiring, lifecycle, auto-reset, and unchanged-default tests.
- Modify `tests/experiments/test_context_ab.py`: CLI/run orchestration and non-publication tests.
- Create `tests/experiments/test_ode_replay.py`: replay matrix and classification tests.
- Create `tests/experiments/test_ode_replay_cli.py`: output isolation and CLI validation tests.

### Task 1: Make Solver Construction Explicit Without Changing Formal Defaults

**Files:**
- Modify: `src/gl_gym/environments/models/utils.py:6-38`
- Create: `tests/environments/test_model_utils.py`

- [ ] **Step 1: Write failing tests for formal defaults and diagnostic overrides**

```python
from unittest.mock import Mock

import gl_gym.environments.models.utils as model_utils


def test_define_model_uses_immutable_formal_cvodes_defaults(monkeypatch):
    integrator = Mock(return_value="integrator")
    monkeypatch.setattr(model_utils.ca, "integrator", integrator)
    monkeypatch.setattr(model_utils, "ODE", lambda x, u, d, p: x)

    result = model_utils.define_model(28, 8, 10, 64, 300.0)

    assert result == "integrator"
    assert integrator.call_args.args[1] == "cvodes"
    assert integrator.call_args.args[3:5] == (0.0, 300.0)
    assert integrator.call_args.args[5] == {
        "abstol": 1e-4,
        "reltol": 1e-4,
        "max_num_steps": 70_000,
    }


def test_define_model_merges_explicit_diagnostic_overrides(monkeypatch):
    integrator = Mock(return_value="integrator")
    monkeypatch.setattr(model_utils.ca, "integrator", integrator)
    monkeypatch.setattr(model_utils, "ODE", lambda x, u, d, p: x)

    model_utils.define_model(
        28,
        8,
        10,
        64,
        150.0,
        integrator_options={"abstol": 1e-6, "reltol": 1e-6},
    )

    assert integrator.call_args.args[4] == 150.0
    assert integrator.call_args.args[5] == {
        "abstol": 1e-6,
        "reltol": 1e-6,
        "max_num_steps": 70_000,
    }
    assert dict(model_utils.FORMAL_CVODES_OPTIONS) == {
        "abstol": 1e-4,
        "reltol": 1e-4,
        "max_num_steps": 70_000,
    }
```

- [ ] **Step 2: Run the tests and verify the new API is absent**

Run: `python -m pytest tests/environments/test_model_utils.py -q`

Expected: FAIL because `FORMAL_CVODES_OPTIONS` and `integrator_options` do not exist.

- [ ] **Step 3: Add immutable defaults and an optional override**

```python
from collections.abc import Mapping
from types import MappingProxyType

FORMAL_CVODES_OPTIONS = MappingProxyType(
    {"abstol": 1e-4, "reltol": 1e-4, "max_num_steps": 70_000}
)


def define_model(
    nx: int,
    nu: int,
    nd: int,
    n_params: int,
    dt: float,
    integrator_options: Mapping[str, float | int] | None = None,
):
    x = ca.SX.sym("x", nx)
    u = ca.SX.sym("u", nu)
    d = ca.SX.sym("d", nd)
    p = ca.SX.sym("p", n_params)
    dxdt = ODE(x, u, d, p)
    options = dict(FORMAL_CVODES_OPTIONS)
    if integrator_options is not None:
        options.update(integrator_options)
    return ca.integrator(
        "F",
        "cvodes",
        {"x": x, "u": u, "p": ca.vertcat(d, p), "ode": dxdt},
        0.0,
        dt,
        options,
    )
```

Keep the existing docstring and document that overrides are diagnostic-only; `TomatoEnv` continues calling `define_model` without overrides.

- [ ] **Step 4: Run focused tests**

Run: `python -m pytest tests/environments/test_model_utils.py tests/env_test.py -q`

Expected: all tests PASS; existing environment construction still uses the formal options.

- [ ] **Step 5: Commit the solver-construction boundary**

```powershell
git add src/gl_gym/environments/models/utils.py tests/environments/test_model_utils.py
git commit -m "refactor: expose formal ODE solver settings"
```

### Task 2: Capture Exact Environment Evidence at the Integration Boundary

**Files:**
- Modify: `src/gl_gym/environments/tomato_env.py:32-168`
- Create: `tests/environments/test_tomato_ode_diagnostics.py`

- [ ] **Step 1: Write failing tests using a real reset environment and injected integrators**

Create a fixture using `load_env_params("TomatoEnv", str(CONFIG_DIR / "envs"))`, remove `eval_options_heldout`, set `base_env_params["training"] = False`, construct `TomatoEnv`, and call `reset(seed=123)`.

```python
class RaisingIntegrator:
    def __call__(self, **kwargs):
        raise RuntimeError("cvodes diagnostic failure")


def test_diagnostics_disabled_keeps_info_payload_absent(tomato_env):
    _, _, _, _, info = tomato_env.step(np.zeros(tomato_env.nu))
    assert "diagnostic_transition" not in info
    assert "integration_failure" not in info


def test_failure_payload_preserves_exact_integrator_inputs(tomato_env, monkeypatch):
    tomato_env.set_ode_diagnostics_enabled(True)
    before_x = tomato_env.x.copy()
    before_u = tomato_env.u.copy()
    before_weather = tomato_env.weather_data[tomato_env.timestep].copy()
    action = np.full(tomato_env.nu, 0.25, dtype=np.float32)
    monkeypatch.setattr(tomato_env, "F", RaisingIntegrator())

    _, _, terminated, _, info = tomato_env.step(action)

    transition = info["diagnostic_transition"]
    failure = info["integration_failure"]
    np.testing.assert_array_equal(failure["x0"], before_x)
    np.testing.assert_array_equal(failure["previous_control"], before_u)
    np.testing.assert_array_equal(failure["requested_action"], action)
    np.testing.assert_array_equal(failure["weather"], before_weather)
    np.testing.assert_array_equal(
        failure["p_dyn"], np.concatenate([failure["weather"], failure["sampled_parameters"]])
    )
    np.testing.assert_array_equal(transition["executed_control"], failure["u"])
    assert failure["exception_type"] == "RuntimeError"
    assert failure["exception_message"] == "cvodes diagnostic failure"
    assert "RaisingIntegrator" in failure["traceback"]
    assert transition["raw_next_observation_available"] is False
    assert terminated is True


def test_diagnostic_capture_does_not_draw_uncertainty_twice(tomato_env, monkeypatch):
    calls = 0

    def sampled_once(parameters, scale, generator):
        nonlocal calls
        calls += 1
        return np.asarray(parameters, dtype=float)

    monkeypatch.setattr("gl_gym.environments.tomato_env.parametric_crop_uncertainty", sampled_once)
    tomato_env.set_ode_diagnostics_enabled(True)
    tomato_env.step(np.zeros(tomato_env.nu))
    assert calls == 1
```

Also assert copied arrays cannot be changed by mutating `tomato_env.x`, `tomato_env.u`, or the original action after `step` returns.

- [ ] **Step 2: Run the environment diagnostic tests and verify failure**

Run: `python -m pytest tests/environments/test_tomato_ode_diagnostics.py -q`

Expected: FAIL because the diagnostic toggle and structured payloads are absent.

- [ ] **Step 3: Implement opt-in diagnostic payloads around the single integration call**

Add `self._ode_diagnostics_enabled = False` in `__init__` and:

```python
def set_ode_diagnostics_enabled(self, enabled: bool) -> None:
    self._ode_diagnostics_enabled = bool(enabled)
```

In `step`, copy `raw_observation`, `x0`, `previous_control`, `requested_action`, `weather`, sampled parameters, and executed control before calling `F`. Catch `Exception as error`, preserve `traceback.format_exc()`, set `self.terminated = True`, and attach these dictionaries after `_get_info()` is created:

```python
diagnostic_transition = {
    "raw_observation": raw_observation,
    "requested_action": requested_action,
    "previous_control": previous_control,
    "executed_control": executed_control,
    "raw_next_observation": self.obs.copy() if integration_succeeded else None,
    "raw_next_observation_available": integration_succeeded,
}
integration_failure = {
    "x0": x0,
    "u": executed_control,
    "previous_control": previous_control,
    "requested_action": requested_action,
    "weather": weather,
    "sampled_parameters": np.asarray(params, dtype=float).copy(),
    "p_dyn": np.asarray(p_dyn, dtype=float).reshape(-1).copy(),
    "timestep": pre_step_timestep,
    "day_of_year": pre_step_day,
    "hour_of_day": pre_step_hour,
    "dt": float(self.dt),
    "nx": int(self.nx),
    "nu": int(self.nu),
    "nd": int(self.nd),
    "n_params": int(self.num_params),
    "solver_options": dict(FORMAL_CVODES_OPTIONS),
    "exception_type": type(error).__name__,
    "exception_message": str(error),
    "traceback": traceback.format_exc(),
}
```

Only add `diagnostic_transition` when enabled and only add `integration_failure` when an exception occurred. Do not print the generic error, retry integration, resample uncertainty, or change time/reward behavior in this task.

- [ ] **Step 4: Run environment and legacy tests**

Run: `python -m pytest tests/environments/test_tomato_ode_diagnostics.py tests/env_test.py -q`

Expected: all tests PASS; successful steps omit failure metadata and disabled diagnostics omit both payloads.

- [ ] **Step 5: Commit exact integration evidence**

```powershell
git add src/gl_gym/environments/tomato_env.py tests/environments/test_tomato_ode_diagnostics.py
git commit -m "feat: expose exact ODE failure evidence"
```

### Task 3: Build the Immutable Failure Capsule and Validator

**Files:**
- Create: `src/gl_gym/experiments/ode_failure.py`
- Create: `tests/experiments/test_ode_failure.py`

- [ ] **Step 1: Write failing tests for copying, capacity, atomic output, idempotency, and corruption**

Define test helpers that generate a complete context and a `diagnostic_transition` containing finite float arrays. Cover these public contracts:

```python
from gl_gym.experiments.ode_failure import (
    CapsuleContext,
    FailureCapsuleRecorder,
    load_failure_capsule,
)


def test_recorder_keeps_only_256_copied_transitions(tmp_path):
    recorder = FailureCapsuleRecorder(tmp_path, context_fixture())
    action = np.array([1.0], dtype=np.float32)
    for step in range(300):
        recorder.record_step(step, np.array([step], dtype=np.float32), 1.0, False, info_fixture(action))
    action[0] = 99.0
    assert recorder.history_length == 256
    assert recorder.history_step_indices == tuple(range(44, 300))
    assert recorder.last_requested_action == (1.0,)


def test_failure_writes_valid_idempotent_capsule(tmp_path):
    first_recorder = FailureCapsuleRecorder(tmp_path, context_fixture())
    second_recorder = FailureCapsuleRecorder(tmp_path, context_fixture())
    first = first_recorder.record_step(
        5228, np.array([3.0]), -1.0, True, failure_info_fixture()
    )
    second = second_recorder.record_step(
        5228, np.array([3.0]), -1.0, True, failure_info_fixture()
    )
    assert first == second
    capsule = load_failure_capsule(first)
    assert capsule.manifest["schema_version"] == 1
    np.testing.assert_array_equal(capsule.failure_inputs["x0"], failure_x0())
    assert not list(tmp_path.rglob("*.tmp"))


def test_loader_rejects_checksum_changes(tmp_path):
    path = write_valid_capsule(tmp_path)
    (path / "traceback.txt").write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError, match="checksum"):
        load_failure_capsule(path)


def test_loader_rejects_object_arrays(tmp_path):
    path = write_valid_capsule(tmp_path)
    np.savez(path / "failure_inputs.npz", x0=np.array([object()], dtype=object))
    rewrite_manifest_checksum(path, "failure_inputs.npz")
    with pytest.raises(ValueError, match="object|pickle"):
        load_failure_capsule(path)
```

Also test missing files, JSON NaN, mismatched stable identifier, same-ID/different-content collision, absent raw-next masks, source checksums, and capsule paths sanitized from task identifiers.

- [ ] **Step 2: Run capsule tests and verify import failure**

Run: `python -m pytest tests/experiments/test_ode_failure.py -q`

Expected: FAIL because `gl_gym.experiments.ode_failure` does not exist.

- [ ] **Step 3: Implement focused public types and canonical helpers**

Use these stable interfaces:

```python
@dataclass(frozen=True, slots=True)
class CapsuleContext:
    seed: int
    task_id: str
    inference_mode: str
    task: dict[str, object]
    checkpoint_path: str
    checkpoint_sha256: str
    git_head: str
    dirty: bool
    source_checksums: dict[str, str]
    package_versions: dict[str, str]


@dataclass(frozen=True, slots=True)
class LoadedFailureCapsule:
    path: Path
    manifest: dict[str, object]
    failure_inputs: dict[str, np.ndarray]
    history_arrays: dict[str, np.ndarray]
    history_rows: tuple[dict[str, object], ...]
    traceback_text: str


class FailureCapsuleRecorder:
    def __init__(self, root: str | Path, context: CapsuleContext, capacity: int = 256):
        if capacity != 256:
            raise ValueError("failure capsule capacity must be 256")
        self.root = Path(root)
        self.context = context
        self._history = deque(maxlen=capacity)

    @property
    def history_length(self) -> int:
        return len(self._history)

    def record_step(
        self,
        step_index: int,
        policy_observation: np.ndarray,
        reward: float,
        done: bool,
        info: dict[str, object],
    ) -> Path | None:
        transition = _validated_transition(step_index, policy_observation, reward, done, info)
        self._history.append(transition)
        failure = info.get("integration_failure")
        return None if failure is None else self._write_capsule(failure)
```

Implement `_finite_array`, strict JSON serialization with `allow_nan=False`, SHA-256 for files and canonical array bytes including dtype and shape, filename sanitization, and a stable failure ID from task identity, timestep, `x0`, `u`, and `p_dyn`.

- [ ] **Step 4: Implement atomic directory publication and strict loading**

Write all five required files into `failure_id + ".tmp-" + uuid4().hex`, fsync closed files through normal context managers, validate the temporary directory with `load_failure_capsule`, then rename it to the final directory. If the final directory exists, load and compare its manifest content checksums; return it only when identical, otherwise raise `FileExistsError`.

`load_failure_capsule` must call `np.load(path, allow_pickle=False)`, reject object dtypes, require finite replay inputs, verify `p_dyn == concatenate(weather, sampled_parameters)`, verify every manifest checksum, reject JSON constants using `parse_constant`, and return copied arrays after NPZ handles close.

- [ ] **Step 5: Run capsule tests**

Run: `python -m pytest tests/experiments/test_ode_failure.py -q`

Expected: all tests PASS, including corruption and idempotency cases.

- [ ] **Step 6: Commit capsule persistence**

```powershell
git add src/gl_gym/experiments/ode_failure.py tests/experiments/test_ode_failure.py
git commit -m "feat: add immutable ODE failure capsules"
```

### Task 4: Wire Capsule Capture Into Deterministic A/B Evaluation

**Files:**
- Modify: `src/gl_gym/experiments/suite_evaluation.py:123-235`
- Modify: `experiments/scripts/run_context_ab.py:302-470`
- Modify: `tests/experiments/test_suite_evaluation.py`
- Modify: `tests/experiments/test_context_ab.py`

- [ ] **Step 1: Write failing evaluator lifecycle and auto-reset tests**

Extend fake environments with `env_method` event recording and a failure payload. Assert the recorder sees the failed pre-reset payload once, diagnostics are disabled in `finally`, and the original early-termination exception remains primary:

```python
def test_failure_recorder_uses_info_payload_and_preserves_primary_error():
    recorder = Mock()
    env = EarlyFailureDiagnosticEnv()
    with pytest.raises(RuntimeError, match="step 1 of 3"):
        run_deterministic_episode(
            HookedFakeModel(),
            env,
            inference_mode="zero_context",
            failure_recorder=recorder,
        )
    assert env.diagnostic_events == [True, False]
    recorder.record_step.assert_called_once()
    call = recorder.record_step.call_args.kwargs
    assert call["step_index"] == 0
    assert call["info"]["integration_failure"]["timestep"] == 0
    np.testing.assert_array_equal(call["policy_observation"], np.array([0.0]))


def test_default_episode_does_not_enable_environment_diagnostics():
    env = DiagnosticCapableFakeEnv()
    run_deterministic_episode(FakeModel(), env)
    assert env.diagnostic_events == []
```

Add a test where `record_step` raises during a failed episode; the premature-termination error must remain primary and the capture error must be attached as a note when `BaseException.add_note` is available.

- [ ] **Step 2: Write failing context CLI tests for isolated failure roots**

Assert `build_parser()` accepts `--failure_root`, and a `run_diagnostic` call with the `failure_root` keyword constructs `CapsuleContext` with seed/task/mode/checkpoint hashes, passes a recorder only for executed episodes, and never creates formal `eval_raw.csv` after an injected early failure.

- [ ] **Step 3: Run focused tests and verify the missing arguments**

Run: `python -m pytest tests/experiments/test_suite_evaluation.py tests/experiments/test_context_ab.py -q`

Expected: FAIL because `failure_recorder` and `failure_root` are not supported.

- [ ] **Step 4: Add optional recorder wiring with guaranteed cleanup**

Change the evaluator signature to:

```python
def run_deterministic_episode(
    model: Any,
    env: Any,
    inference_mode: str | None = None,
    return_diagnostics: bool = False,
    failure_recorder: Any | None = None,
) -> dict[str, float] | tuple[dict[str, float], dict[str, Any]]:
```

Before reset, call `env.env_method("set_ode_diagnostics_enabled", True)` only when a recorder exists. After each `env.step`, call `failure_recorder.record_step` before inference hooks inspect terminal observations. Catch a recorder exception into `capture_error`, continue far enough to raise the evaluator's premature-termination error, and attach the capture error as a note; if the environment did not terminate, raise `capture_error` before the next prediction. In the outer `finally`, disable diagnostics and then end inference state. Preserve the first evaluation exception; cleanup exceptions add notes and propagate only when no primary error exists.

- [ ] **Step 5: Add A/B recorder construction without changing default publication**

Add `failure_root: str | Path | None = None` to `run_diagnostic`, plus injectable `recorder_factory=FailureCapsuleRecorder`. Build `CapsuleContext` immediately before each non-resumed episode using `asdict(task)`, run/checkpoint evidence, `_provenance()`, `platform` versions, and checksums for:

- `src/gl_gym/environments/tomato_env.py`;
- `src/gl_gym/environments/models/ode.py`;
- `src/gl_gym/environments/models/utils.py`;
- `configs/envs/TomatoEnv.yml`;
- `configs/agents/rule_based.yml`.

Pass the recorder only when `failure_root` is provided. Add parser argument `--failure_root`; do not change the existing work root, 32-row completion check, or formal artifact writer.

- [ ] **Step 6: Run evaluator and A/B tests**

Run: `python -m pytest tests/experiments/test_suite_evaluation.py tests/experiments/test_context_ab.py tests/experiments/test_ode_failure.py -q`

Expected: all tests PASS; incomplete runs still do not publish formal artifacts.

- [ ] **Step 7: Commit evaluation integration**

```powershell
git add src/gl_gym/experiments/suite_evaluation.py experiments/scripts/run_context_ab.py tests/experiments/test_suite_evaluation.py tests/experiments/test_context_ab.py
git commit -m "feat: capture ODE failures during context evaluation"
```

### Task 5: Implement Exact Offline Replay and Conservative Classification

**Files:**
- Create: `src/gl_gym/experiments/ode_replay.py`
- Create: `tests/experiments/test_ode_replay.py`

- [ ] **Step 1: Write failing classification tests with injected variant outcomes**

```python
@pytest.mark.parametrize(
    ("outcomes", "expected"),
    [
        ({"original": True}, "non_reproduced"),
        (
            {"original": False, "previous_control": True, "rule_based_control": False,
             "original_2x_substeps": False, "original_4x_substeps": False,
             "original_strict_tolerance": False},
            "policy_induced_control_instability",
        ),
        (
            {"original": False, "previous_control": False, "rule_based_control": False,
             "original_2x_substeps": True, "original_4x_substeps": False,
             "original_strict_tolerance": False},
            "solver_step_sensitivity",
        ),
        (
            {"original": False, "previous_control": True, "rule_based_control": False,
             "original_2x_substeps": True, "original_4x_substeps": False,
             "original_strict_tolerance": False},
            "mixed_control_and_solver_sensitivity",
        ),
        (
            {"original": False, "previous_control": False, "rule_based_control": False,
             "original_2x_substeps": False, "original_4x_substeps": False,
             "original_strict_tolerance": False},
            "state_or_model_domain_failure",
        ),
    ],
)
def test_classify_replay_matrix(outcomes, expected):
    assert classify_replay_outcomes(outcome_records(outcomes)) == expected
```

Add cases where rule-based control is unavailable, inputs are non-finite, the original outcome is missing, and a successful integrator returns a non-finite final state.

- [ ] **Step 2: Write failing exact-input, substep-horizon, and rule-based-control tests**

Inject an integrator factory that records `dt`, options, `x0`, `u`, and `p_dyn`. Assert:

- original uses byte-equivalent stored inputs and formal options;
- 2x calls two fresh half-interval transitions in sequence;
- 4x calls four quarter-interval transitions in sequence;
- all substeps use the same stored `p_dyn`;
- strict tolerance uses the full interval and `abstol=reltol=1e-6`;
- `RuleBasedController.predict(x0, weather, namespace)` receives stored day/hour/`nu` and produces the replay control.

- [ ] **Step 3: Run replay tests and verify import failure**

Run: `python -m pytest tests/experiments/test_ode_replay.py -q`

Expected: FAIL because `gl_gym.experiments.ode_replay` does not exist.

- [ ] **Step 4: Implement typed replay records and the six-variant matrix**

```python
@dataclass(frozen=True, slots=True)
class ReplayOutcome:
    variant: str
    available: bool
    success: bool
    elapsed_seconds: float
    final_state: np.ndarray | None
    exception_type: str | None
    exception_message: str | None
    warnings: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ReplayReport:
    failure_id: str
    classification: str
    outcomes: tuple[ReplayOutcome, ...]


def replay_failure_capsule(
    capsule: LoadedFailureCapsule,
    integrator_factory: Callable[..., Any] = define_model,
    controller_factory: Callable[[], RuleBasedController] = build_rule_based_controller,
) -> ReplayReport:
    inputs = capsule.failure_inputs
    outcomes = (
        _run_variant("original", inputs["u"], 1, None, inputs, integrator_factory),
        _run_variant("previous_control", inputs["previous_control"], 1, None, inputs, integrator_factory),
        _run_rule_based_variant(capsule, integrator_factory, controller_factory),
        _run_variant("original_2x_substeps", inputs["u"], 2, None, inputs, integrator_factory),
        _run_variant("original_4x_substeps", inputs["u"], 4, None, inputs, integrator_factory),
        _run_variant(
            "original_strict_tolerance",
            inputs["u"],
            1,
            {"abstol": 1e-6, "reltol": 1e-6},
            inputs,
            integrator_factory,
        ),
    )
    return ReplayReport(
        failure_id=str(capsule.manifest["failure_id"]),
        classification=classify_replay_outcomes(outcomes),
        outcomes=outcomes,
    )
```

Capture Python warnings per variant, time with `perf_counter`, create a fresh integrator per variant, propagate each successful substep state to the next substep, and mark non-finite final states as failures.

- [ ] **Step 5: Implement conservative classification in fixed precedence**

Require an available original outcome. Return `non_reproduced` when it succeeds. Otherwise compute success among available alternative controls and solver variants; return mixed before policy/solver, persistent model-domain only when every required variant is available and fails, and `insufficient_counterfactual_evidence` for missing required evidence.

- [ ] **Step 6: Run replay tests**

Run: `python -m pytest tests/experiments/test_ode_replay.py tests/environments/test_model_utils.py -q`

Expected: all tests PASS with exact horizon and classification assertions.

- [ ] **Step 7: Commit offline replay logic**

```powershell
git add src/gl_gym/experiments/ode_replay.py tests/experiments/test_ode_replay.py
git commit -m "feat: replay and classify ODE failures"
```

### Task 6: Add the Replay CLI and Isolated Reports

**Files:**
- Create: `experiments/scripts/replay_ode_failure.py`
- Create: `tests/experiments/test_ode_replay_cli.py`

- [ ] **Step 1: Write failing CLI tests**

Load the script with `importlib.util`. Test parser requirements, refusal to place output inside the formal result root recorded in capsule metadata, atomic replacement refusal, and successful writing through an injected replay function:

```python
def test_cli_writes_three_isolated_outputs(tmp_path):
    capsule_path = write_valid_capsule(tmp_path / "capsules")
    output = tmp_path / "replay"
    module.run_replay_cli(capsule_path, output, replay_loader=fake_report)
    assert (output / "replay_results.json").is_file()
    assert (output / "replay_states.npz").is_file()
    assert (output / "replay_summary.md").is_file()
    result = json.loads((output / "replay_results.json").read_text(encoding="utf-8"))
    assert result["classification"] == "solver_step_sensitivity"
```

Also load `replay_states.npz` with `allow_pickle=False` and assert only successful final states and finite masks are present.

- [ ] **Step 2: Run CLI tests and verify the script is missing**

Run: `python -m pytest tests/experiments/test_ode_replay_cli.py -q`

Expected: FAIL because `experiments/scripts/replay_ode_failure.py` does not exist.

- [ ] **Step 3: Implement a thin parser and atomic report writer**

Expose:

```python
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capsule", required=True)
    parser.add_argument("--output_root", required=True)
    return parser


def run_replay_cli(
    capsule_path: str | Path,
    output_root: str | Path,
    replay_loader: Callable[[LoadedFailureCapsule], ReplayReport] = replay_failure_capsule,
) -> Path:
    capsule = load_failure_capsule(capsule_path)
    output = validate_isolated_output_root(output_root, capsule.manifest)
    report = replay_loader(capsule)
    return write_replay_report_atomic(report, output)
```

Serialize dataclasses to strict finite JSON, save successful final states as a numeric matrix plus variant names encoded as fixed-width Unicode, and render a Markdown table with variant, availability, success, elapsed time, and exception. Include the classification-specific next action from the approved design. Validate the temporary directory before renaming and never overwrite an existing report directory.

- [ ] **Step 4: Run CLI and replay tests**

Run: `python -m pytest tests/experiments/test_ode_replay_cli.py tests/experiments/test_ode_replay.py tests/experiments/test_ode_failure.py -q`

Expected: all tests PASS and all generated NPZ files load with `allow_pickle=False`.

- [ ] **Step 5: Commit the replay CLI**

```powershell
git add experiments/scripts/replay_ode_failure.py tests/experiments/test_ode_replay_cli.py
git commit -m "feat: add reproducible ODE replay CLI"
```

### Task 7: Verify the Repository, Capture the Known Failure Once, and Replay It

**Files:**
- Generated only under: `artifacts/results/.AgriControl_C_2026-07-10-v3-context-ab.work/failures/`
- Generated only under: `artifacts/results/diagnostics/ode-replay/`
- Do not modify formal A/B files or manuscript artifacts.

- [ ] **Step 1: Run all focused diagnostic tests**

Run:

```powershell
python -m pytest tests/environments/test_model_utils.py tests/environments/test_tomato_ode_diagnostics.py tests/experiments/test_ode_failure.py tests/experiments/test_suite_evaluation.py tests/experiments/test_context_ab.py tests/experiments/test_ode_replay.py tests/experiments/test_ode_replay_cli.py -q
```

Expected: all focused tests PASS with zero failures.

- [ ] **Step 2: Run the full repository suite and compilation check**

Run:

```powershell
python -m pytest -q
python -m compileall -q src experiments/scripts
```

Expected: pytest exits 0 and compileall exits 0.

- [ ] **Step 3: Record formal-root checksums before the real diagnostic**

Run:

```powershell
$formal='artifacts/results/AgriControl_C_2026-07-10-v3-context-ab'
Get-ChildItem $formal -File -ErrorAction SilentlyContinue | Get-FileHash -Algorithm SHA256 | Sort-Object Path | ConvertTo-Json | Set-Content 'artifacts/results/.AgriControl_C_2026-07-10-v3-context-ab.work/formal-before.json'
```

Expected: the snapshot is written only to the hidden work root. The formal root still lacks `eval_raw.csv`, `paired_deltas.csv`, `split_summary.csv`, `diagnostic_manifest.json`, and `decision.json`.

- [ ] **Step 4: Resume once to capture the deterministic seed-123 failure**

Run:

```powershell
python experiments/scripts/run_context_ab.py `
  --source_manifest artifacts/results/AgriControl_C_2026-07-09-v3-pilot3/suite_manifest.json `
  --source_tasks_csv artifacts/results/AgriControl_C_2026-07-09-v3-pilot3/eval_tasks.csv `
  --model_root artifacts/models/AgriControl_C_2026-07-09-v3-pilot3 `
  --result_root artifacts/results/AgriControl_C_2026-07-10-v3-context-ab `
  --failure_root artifacts/results/.AgriControl_C_2026-07-10-v3-context-ab.work/failures `
  --seeds 42 123 --device cpu --resume
```

Expected: non-zero exit at seed 123, `heldout_2011_d59_u0p00_standard`, `zero_context`, with premature termination near step 5,229; exactly one checksum-valid capsule is present. Do not rerun if a valid capsule was written even if the step differs slightly.

- [ ] **Step 5: Validate the capsule and execute offline replay**

Resolve the single capsule directory, then run:

```powershell
$capsule=(Get-ChildItem 'artifacts/results/.AgriControl_C_2026-07-10-v3-context-ab.work/failures' -Recurse -Directory | Where-Object { Test-Path (Join-Path $_.FullName 'manifest.json') } | Select-Object -First 1).FullName
python experiments/scripts/replay_ode_failure.py --capsule $capsule --output_root artifacts/results/diagnostics/ode-replay/seed123-heldout2011-zero-context
```

Expected: exit 0 and three replay outputs. `replay_results.json` contains all six variants and one approved classification label.

- [ ] **Step 6: Prove formal A/B artifacts were not published or mutated**

Run:

```powershell
$formal='artifacts/results/AgriControl_C_2026-07-10-v3-context-ab'
$forbidden=@('eval_raw.csv','paired_deltas.csv','split_summary.csv','diagnostic_manifest.json','decision.json')
$present=@($forbidden | Where-Object { Test-Path (Join-Path $formal $_) })
if($present.Count){ throw "Unexpected formal artifacts: $($present -join ', ')" }
$before=Get-Content 'artifacts/results/.AgriControl_C_2026-07-10-v3-context-ab.work/formal-before.json' -Raw
$after=Get-ChildItem $formal -File -ErrorAction SilentlyContinue | Get-FileHash -Algorithm SHA256 | Sort-Object Path | ConvertTo-Json
if($before.Trim() -ne $after.Trim()){ throw 'Formal result root changed' }
```

Expected: exit 0 with no formal files created or changed.

- [ ] **Step 7: Record the evidence-backed research decision**

Read `replay_summary.md` and report the exact original reproduction outcome, successful counterfactuals, classification, and the matching next design branch. Do not alter manuscript claims or authorize training in this commit.

- [ ] **Step 8: Commit only source/tests; keep generated diagnostic evidence untracked**

Run `git status --short`, verify no generated artifact is staged, and commit any final documentation-only adjustment if the executed command syntax required correction:

```powershell
git add docs/superpowers/plans/2026-07-13-ode-failure-capsule-replay.md
git commit -m "docs: record ODE replay execution details"
```

Skip this final commit when the plan file is unchanged after execution.

## Completion Gate

Implementation is complete only when focused and full tests pass, the exact original replay result is recorded, all available variants have machine-readable outcomes, formal A/B artifacts remain absent and unchanged, and the next research action follows the conservative classification. Any solver/controller modification or resumed training requires a new approved design.
