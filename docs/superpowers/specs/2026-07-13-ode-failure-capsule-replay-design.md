# ODE Failure Capsule and Offline Replay Design

## Status

Approved in conversation on 2026-07-13. This design is the next diagnostic step after the online-context A/B run terminated early on seed 123, task `heldout_2011_d59_u0p00_standard`, in `zero_context` mode. It does not authorize additional training or a change to the formal benchmark dynamics.

## Problem

The context A/B diagnostic stopped at step 5,229 of 5,760 because the greenhouse model's CVODES integration failed near simulation time 114.675. CasADi reported a non-finite Jacobian while the environment caught the exception with a bare `except`, printed a generic message, retained the old state, and returned an early terminal transition. The evaluator now correctly rejects that premature termination, but the available log is insufficient to distinguish among:

- a policy-induced unsafe control transition;
- a state or task outside the model's valid numerical domain; or
- a solver configuration that is too fragile for an otherwise valid transition.

Changing actions, tolerances, or integration steps before collecting the exact failing inputs would confound the diagnosis and could hide a controller robustness defect.

## Objective

Make every ODE integration failure evidence-preserving and deterministically replayable. Capture the exact numerical input and the recent controller trajectory, reproduce the failure without rerunning a full episode, and compare controlled replay variants that classify the failure mechanism.

An integration failure remains a controller/evaluation failure in formal results. Diagnostic retries are evidence only and can never replace, repair, or publish the failed episode's score.

## Scope

This iteration will:

- preserve the original CasADi exception type and message as structured failure metadata;
- retain an in-memory ring buffer for the latest 256 evaluation transitions;
- atomically write one immutable failure capsule when integration fails;
- provide an offline CLI that validates and replays a capsule;
- replay the original transition and predefined control/solver variants;
- emit a machine-readable classification report;
- rerun only the known failing episode to acquire and analyze one capsule.

This iteration will not:

- resume the 32-episode A/B gate or any training;
- change the learned policy, reward, constraints, action bounds, or action increment;
- add an online safety shield or automatic solver fallback;
- change the formal CVODES configuration;
- publish partial A/B metrics as a gate result;
- claim that one diagnosed episode establishes paper-grade robustness.

## Considered Approaches

### 1. Evidence-first failure capsule and offline replay

Capture exact inputs at the failure boundary and test counterfactuals offline. This isolates causes, preserves benchmark semantics, and makes the diagnosis independently auditable. This is the selected approach.

### 2. Add an online action safety layer immediately

Clipping control increments or replacing suspect actions may complete the episode, but it changes the evaluated controller and can conceal whether the learned policy created the instability. This is deferred until replay evidence identifies policy-induced failure.

### 3. Relax tolerances or retry with substeps during evaluation

An automatic retry may mask a solver defect and would silently change the environment used for scoring. Solver variants are allowed only in offline replay during this iteration.

## Architecture

### Environment failure record

Immediately before every call to the CasADi integrator, `TomatoEnv` will materialize the exact integration inputs without regenerating random uncertainty:

- pre-integration state `x0`;
- executed greenhouse control `u` after incremental action conversion and clipping;
- disturbance and uncertain parameter vector `p_dyn` passed to `F`;
- weather row and sampled parameter vector as separate arrays;
- environment timestep, simulation clock, configured `dt`, state/control dimensions, and solver options;
- controller action that produced `u`, plus the previous executed control.

On an integration exception, the environment will store a structured failure record for the returned `info`. It will catch `Exception`, not `BaseException`, and will include the exception class, message, and formatted traceback. It will not advance the state using a retry. The existing early-terminal behavior remains, allowing the evaluator's premature-termination check to fail the episode.

When diagnostic capture is enabled, every step also returns a lightweight `diagnostic_transition` payload containing copied pre-step raw observation, requested action, previous control, executed control, and, on success, the raw next observation. This is the authoritative source for raw values because vector-environment auto-reset may replace observations visible to the evaluator. The payload is absent when capture is disabled, so ordinary training and evaluation avoid the extra copies.

Environment records are data only. The environment does not write files and does not depend on experiment paths.

### Evaluation ring buffer

`run_deterministic_episode()` will optionally accept a failure-capsule recorder. The recorder owns a fixed-capacity deque of 256 transition records. Each record contains:

- zero-based step index and task identity;
- normalized observation presented to the policy;
- raw environment observation before the step;
- requested policy action and executed control;
- reward and selected climate/economic metrics;
- terminal flags;
- raw next observation when a successful transition exists.

The normalized policy-boundary observation comes from the evaluator; raw observations and executed controls come from the environment's `diagnostic_transition` payload. Arrays are copied when recorded so later vector-environment resets cannot mutate the evidence. The buffer is diagnostic state only and is discarded after a successful episode.

When `info` contains an integration failure, the recorder combines the ring buffer, current failed transition, environment failure record, inference mode, seed, task definition, checkpoint identity, and software revision into one capsule. The evaluator then raises the same premature-termination error as before. Failure capture must not suppress or replace the primary evaluation exception.

### Failure capsule format

Each capsule is a directory below a caller-supplied diagnostic work root:

`failures/<seed>/<task_id>/<inference_mode>/<failure_id>/`

The `failure_id` is a stable SHA-256 digest of canonical task identity, step index, and the exact `x0`, `u`, and `p_dyn` byte representations. Reproducing the same failure therefore targets the same directory.

Required files are:

- `manifest.json`: schema version, identifiers, array shapes/dtypes/checksums, checkpoint path and checksum, task configuration, solver configuration, exception metadata, package versions, git HEAD, dirty-worktree status, and checksums of the relevant source/config files;
- `failure_inputs.npz`: exact `x0`, `u`, previous control, policy action, weather, sampled parameters, and `p_dyn`;
- `history.npz`: stacked ring-buffer arrays with explicit masks for optional fields;
- `history.jsonl`: scalar metadata and per-step metric dictionaries aligned with `history.npz`;
- `traceback.txt`: the complete formatted exception traceback.

All numeric arrays use explicit NumPy dtypes and `allow_pickle=False`. JSON contains no NaN or Infinity. Writes go to a sibling temporary directory, are validated, and are atomically renamed. If the stable capsule already exists and all checksums match, capture is idempotent. A mismatch at the same identifier is an integrity error; existing evidence is never overwritten.

The capsule contains only local experimental data and repository paths. It must not serialize model objects, arbitrary Python objects, credentials, or environment variables.

## Offline Replay

### Validation

The replay command accepts one capsule directory and an output directory. Before integration it validates:

- schema version and required files;
- checksums, dtypes, dimensions, and finite numeric inputs;
- compatibility between stored dimensions and the reconstructed model;
- that `p_dyn` exactly equals the stored weather/parameter concatenation;
- task and solver metadata required to reconstruct the integrator.

Invalid or incomplete capsules fail closed. Loading never enables pickle.

### Replay matrix

Every replay uses the stored `x0` and `p_dyn`. No uncertainty is resampled. The primary matrix is:

1. `original`: stored executed control with the formal solver configuration;
2. `previous_control`: previous executed control with the formal solver configuration;
3. `rule_based_control`: control produced by the repository's deterministic rule-based controller after restoring the stored task, state, clock, and disturbance;
4. `original_2x_substeps`: stored control over two equal CVODES substeps spanning the same total `dt`;
5. `original_4x_substeps`: stored control over four equal CVODES substeps spanning the same total `dt`;
6. `original_strict_tolerance`: stored control over the same `dt` with `abstol=reltol=1e-6`, retaining the same maximum-step policy.

The rule-based candidate is rejected as unavailable rather than approximated if its exact prerequisites cannot be reconstructed. Substep variants keep `p_dyn` constant over the original interval, matching the formal one-step disturbance assumption. All variants run in fresh integrator instances and record success, elapsed time, warnings, exception metadata, final state, and finite-value checks.

The original variant must reproduce the stored failure before counterfactual classification is considered valid. If it succeeds, the result is `non_reproduced` and no causal class is assigned.

### Classification rules

Classification is deterministic and conservative:

- `policy_induced_control_instability`: the original replay fails and at least one valid alternative-control replay succeeds under the formal solver;
- `solver_step_sensitivity`: the original and all available alternative-control replays fail under the formal solver, while at least one substep or strict-tolerance replay succeeds;
- `state_or_model_domain_failure`: the original, every available alternative control, and every solver variant fail;
- `mixed_control_and_solver_sensitivity`: an alternative control and a solver variant both succeed, so the single capsule does not isolate one mechanism;
- `insufficient_counterfactual_evidence`: required alternatives are unavailable or outcomes do not satisfy a rule;
- `non_reproduced`: the exact original input succeeds offline.

These labels describe one transition, not an entire seed, policy, or method. The report lists raw outcomes so later analysis is not forced to rely on the label.

### Replay outputs

The CLI writes atomically:

- `replay_results.json`: validated capsule identity, variant configurations, outcomes, exceptions, timings, and classification;
- `replay_states.npz`: successful final states and finite masks;
- `replay_summary.md`: a concise human-readable table and the next recommended research action.

Replay outputs live outside the formal A/B result root and never update `progress.csv`, `eval_raw.csv`, paired deltas, decision records, or manuscript artifacts.

## Error Handling and Scientific Integrity

- ODE failure always marks the evaluated episode invalid; no partial score is published.
- Capsule-write failure is reported alongside the primary ODE failure without replacing its traceback.
- Non-finite inputs are preserved only in the failure record and explicitly flagged; they are never accepted silently by replay validation.
- Vector-environment auto-reset data must not be mistaken for the failed terminal state.
- Diagnostic code must not consume additional random numbers before the integrator call.
- Replays never mutate checkpoints, suite registries, task tables, or source capsules.
- Formal solver or controller changes require a separate approved design after classification.

## Testing

Implementation will be test-driven and cover:

1. Exact pre-integration `x0`, `u`, weather, sampled parameters, and `p_dyn` are captured without another uncertainty draw.
2. Successful environment steps expose no failure payload.
3. An injected integration exception preserves class, message, traceback, and early termination.
4. The ring buffer retains exactly the most recent 256 transitions and copies mutable arrays.
5. A failed final step is included once and is not replaced by vector auto-reset observations.
6. Capsule writes are atomic, checksum-validated, idempotent, and collision-safe.
7. Capsule loading rejects pickle/object arrays, missing fields, invalid JSON numbers, checksum changes, and incompatible dimensions.
8. Exact original replay receives byte-equivalent `x0`, `u`, and `p_dyn`.
9. Substeps span the same total horizon and use no resampled disturbance or parameters.
10. Each classification rule is exercised with a fake integrator, including `non_reproduced` and unavailable rule-based control.
11. Capsule capture does not alter successful deterministic evaluation or inference hook order.
12. The formal A/B publication path remains unchanged and incomplete runs remain unpublished.

Focused tests will precede the full repository suite. The known seed-123 episode is then rerun once to acquire a real capsule, followed by offline replay. Repeated full-episode reruns are not part of this iteration.

## Success Criteria

This diagnostic iteration is complete only when:

- the known failure produces a valid, checksum-verifiable capsule;
- exact original replay either reproduces the failure or records `non_reproduced` with complete evidence;
- all available replay variants finish with machine-readable outcomes;
- a conservative classification and next research decision are recorded;
- no formal A/B result or manuscript claim is changed.

If policy-induced or mixed sensitivity is found, the next design will evaluate a principled action-safety layer and its impact on return and constraint violations. If solver sensitivity is isolated, the next design will compare integration schemes under a fixed controller and establish a new benchmark version. If a state/model-domain failure is found, the next step will audit physical state bounds and the implicated ODE/Jacobian terms before any new training.
