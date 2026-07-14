# Minimal-Feasibility Action Shield Design

**Date:** 2026-07-14

**Status:** Approved by the user's standing authorization to adopt the recommended option

**Scientific role:** Evaluation-only diagnostic controller variant for the existing checkpoint

## 1. Motivation and evidence

The current formal context A/B evaluation cannot produce a valid result because the existing policy deterministically triggers a CVODES integration failure in `heldout_2011_d59_u0p00_standard`, seed 123, `zero_context`, at captured timestep 5228. Exact offline replay reproduces the original failure. Replacing the proposed control with the previous control still fails, while the raw rule-based control succeeds. Increasing integration substeps and tightening tolerances do not resolve the failure. The evidence therefore supports the conservative classification `policy_induced_control_instability` rather than a numerical-resolution defect.

The next scientific question is whether a legal, minimal intervention can eliminate policy-induced ODE failures without materially changing controller performance. This is a diagnostic question. The present design does not retrain the policy, change the learned checkpoint, weaken the formal solver, or reinterpret failed unshielded episodes as successful.

## 2. Objectives and preregistered gates

The primary objective is to evaluate an action-space feasibility shield as a distinct controller variant. It must satisfy all of the following gates before it can support paper claims:

1. **Known-failure gate:** recover a valid transition at the known deterministic failure using a legal projected action and record a nonzero intervention.
2. **Context A/B gate:** complete the preregistered 32-episode context diagnostic with zero ODE failures.
3. **Full-suite gate:** only after the context A/B gate passes, evaluate the shield on the complete 91-task suite using the already approved seed protocol.

For gates 2 and 3, the shield passes only when all four conditions hold:

- ODE failures: exactly zero;
- intervention rate: at most 0.5% of environment steps;
- mean return loss: at most 2% on episodes completed by both shielded and unshielded variants;
- normalized constraint-violation increase: at most 5% on the same paired episodes.

Thresholds are conjunctive. A pass cannot be declared by averaging away a failed condition. Episodes that the unshielded controller cannot complete remain failures in its own result group and are excluded from paired return and violation deltas; completion counts and failure rates are reported separately so that this exclusion cannot create a favorable performance claim.

## 3. Alternatives considered

### 3.1 Recommended: minimal feasibility projection

Try the policy action first. Only after its integration fails, search a fixed sequence of legal actions on the segment between the policy action and a rate-limited rule-based reference action. Select the first feasible candidate. This preserves the policy whenever possible, respects the environment's actuator semantics, and makes intervention magnitude measurable.

### 3.2 Rejected as the primary method: direct rule-control fallback

The raw rule-based control succeeded in replay, but substituting it directly can bypass the environment's incremental action constraint and can create a large, poorly characterized intervention. It remains a diagnostic comparator only.

### 3.3 Rejected as insufficiently supported: static clipping or smoothing

The environment already clips controls and limits changes through `delta_u_max`. The previous-control counterfactual also failed. Additional static clipping or smoothing therefore lacks direct evidence that it targets the observed failure mechanism.

## 4. Architecture and component boundaries

The feature comprises four focused units:

1. **Reference-action adapter** converts a deterministic rule-based target control into a legal action-space reference using the current executed control and the environment's actuator limits.
2. **Feasibility projector** constructs candidates in a fixed order and evaluates them against identical transition inputs using isolated formal integrators.
3. **Shielded step coordinator** owns step-level transaction semantics: it first attempts the requested policy action, invokes projection only after failure, commits exactly one successful transition, or raises a formal failure when no candidate is feasible.
4. **Intervention recorder and aggregator** stores immutable step evidence and derives episode- and experiment-level safety metrics without modifying unshielded outputs.

The shield is opt-in and disabled by default. Existing environment behavior and existing unshielded evaluation paths remain unchanged. The implementation should expose the shield through evaluation configuration or a distinct controller wrapper, not a global monkey patch and not an implicit change to `TomatoEnv.step` semantics.

## 5. Action construction

Let the requested policy action be `a_pi`, the current executed control be `u_prev`, the rule-based target control be `u_rule`, and the environment's per-channel maximum control change be `delta_u_max`.

The legal reference action is

```text
a_ref = clip((u_rule - u_prev) / delta_u_max, -1, 1)
```

Division is elementwise. Configuration validation must reject nonfinite values, shape mismatches, or nonpositive entries in `delta_u_max` before any solver call.

Candidates are defined by

```text
a(lambda) = (1 - lambda) * a_pi + lambda * a_ref
```

using the preregistered ordered grid

```text
lambda = [1/16, 1/8, 1/4, 1/2, 1]
```

`lambda = 0` is the original policy attempt and is executed once before the projector is invoked. Both endpoints are in the environment action box, so every convex candidate is also in the action box. Each candidate is passed through the existing `action_to_control` mapping; no candidate may supply raw control directly to the dynamics.

The first candidate that produces a finite, correctly shaped next state is selected. Searching stops immediately. The fixed grid is part of the method definition and must not be tuned per task, seed, inference mode, or observed outcome.

## 6. Transactional step data flow

For every environment step:

1. Snapshot the pre-step state, current control, auxiliary state required by the environment, weather/disturbance input, dynamic parameters, step index, and any deterministic controller inputs.
2. Obtain `a_pi` from the unchanged policy.
3. Attempt the normal formal transition once with `a_pi`.
4. If it succeeds, commit the transition with `requested_action == executed_action`, `lambda = 0`, and no intervention.
5. If it fails, restore the exact snapshot, obtain the deterministic rule-based target for that same pre-step input, compute `a_ref`, and test candidates in increasing `lambda` order.
6. Run every retry with a newly constructed integrator using exactly `FORMAL_CVODES_OPTIONS` and the same state, disturbance, and parameters. Failed candidates must not contaminate later attempts.
7. Reuse the selected candidate's already computed integration result as the committed next state; do not integrate the selected candidate a second time.
8. Update the environment, reward/constraint accounting, controller recurrent state, and any context memory exactly once using the committed transition and the **executed** action.

Candidate attempts are counterfactual probes, not environment steps. They do not advance time, consume observations, update normalization statistics, mutate controller memory, accumulate reward, or alter episode counters.

## 7. Failure and integrity semantics

- The unshielded controller's original integration failure remains an unshielded failure. Shielded and unshielded records use distinct method identifiers and output roots.
- A failed initial policy attempt is recorded as the trigger for an intervention but is not counted as a shielded episode failure if a later legal candidate succeeds.
- If all candidates fail, the shielded step raises a formal ODE failure. There is no raw-control emergency override and no score for the incomplete episode.
- Nonfinite candidate inputs, invalid output shape, nonfinite next state, reference-controller failure, snapshot mismatch, or restoration failure are explicit shield failures; they cannot silently fall back to unshielded execution.
- Expected CVODES warnings and exceptions from failed candidates are captured with candidate identity. Unrelated input-construction and programming errors retain their original exception priority and are not relabeled as solver infeasibility.
- Formal result publication remains atomic. Interrupted or partially completed runs cannot create a valid final result directory.
- The method/configuration fingerprint includes the lambda grid, reference-controller configuration hash, action mapping parameters, solver options, checkpoint identity, source revision, task protocol, and shield schema version.

## 8. Evidence schema and metrics

Each step record contains at least:

- task, seed, inference mode, episode and step identifiers;
- requested action, reference action, executed action, and executed control;
- intervention flag and selected `lambda`;
- ordered candidate attempts with success/failure, warning/exception summary, and elapsed time;
- action-space intervention norms (`L1`, `L2`, and `L-infinity`);
- per-channel changed flags;
- initial policy-failure evidence reference, when applicable;
- method and provenance fingerprints.

Episode aggregation reports:

- total steps, interventions, and intervention rate;
- first intervention step;
- mean and maximum selected `lambda`;
- mean and maximum intervention norm;
- per-channel intervention counts;
- total extra solver attempts and shield runtime overhead;
- completion status, ODE-failure count, return, and normalized constraint violation.

Experiment aggregation reports both controller groups separately and a paired table for jointly completed episodes. The primary intervention rate denominator is all committed shielded environment steps. Candidate probes are not included in that denominator.

The paired gates reuse the context diagnostic's fixed `EPSILON = 1e-9` convention. Let `delta_return = return_shielded - return_unshielded` over jointly completed episodes. The relative return loss is `max(0, -mean(delta_return) / (mean(abs(return_unshielded)) + EPSILON))` and must be at most `0.02`. For each paired episode and each existing violation metric, the normalized burden is `violation_shielded / (abs(violation_unshielded) + EPSILON)`; a zero/zero pair is assigned the neutral ratio `1.0`. The mean of those ratios must be at most `1.05`. These formulas, the metric set, and `EPSILON` are fixed before real evaluation.

## 9. Evaluation sequence and decision rules

### Stage 1: known deterministic failure

Run the shield from the same checkpoint and episode configuration that produced the captured failure. Confirm that the original action still fails, at least one projected candidate succeeds, the chosen candidate is the smallest successful `lambda`, and the committed trajectory records the executed action. This is a mechanism test, not a performance result.

### Stage 2: 32-episode context A/B diagnostic

Run shielded zero-context and shielded online-context evaluations under the existing diagnostic protocol. Preserve the unshielded run and its failure evidence as a separate comparator. Apply all four preregistered gates and publish complete intervention diagnostics. No full-suite run is authorized by the method unless Stage 2 passes.

### Stage 3: complete 91-task suite

Run the existing full-suite protocol without changing tasks, seeds, checkpoint, inference definitions, solver settings, or thresholds. Report shielded and unshielded completion, performance, constraint, and intervention results separately. A manuscript claim of robust execution requires Stage 3 to pass all four gates.

Failure at any stage triggers analysis of recorded evidence and a new design decision. It does not authorize post hoc changes to the lambda grid or thresholds within the same confirmatory run.

## 10. Test strategy

### Unit tests

- exact conversion from target control to legal reference action;
- validation of shapes, finiteness, and positive `delta_u_max`;
- exact candidate order and convex construction;
- first-feasible selection and early stopping;
- no projection after a successful policy transition;
- all-candidate failure behavior;
- immutable and complete intervention records;
- exact aggregation denominators and threshold boundary behavior.

### Transaction and isolation tests

- identical state, disturbance, and parameters for all attempts;
- a fresh formal integrator for every retry;
- failure of one candidate cannot mutate the next candidate's inputs;
- only the chosen transition updates state, time, reward, normalization, and memory;
- recurrent/context memory receives executed rather than requested action;
- selected integration output is reused without a duplicate call;
- interrupted publication leaves no valid final output.

### Integration and regression tests

- synthetic integrator that fails until a known `lambda` verifies the full coordinator;
- known capsule inputs verify legal-action construction against replay evidence;
- existing unshielded tests remain unchanged and pass;
- deterministic reruns produce identical selected candidates and fingerprints;
- CLI/config paths preserve separate shielded and unshielded artifacts.

### Real-evidence validation

Stage 1, Stage 2, and Stage 3 are run only in that order. Each stage produces a machine-readable decision artifact that lists every gate, observed value, threshold, and pass/fail result. The next stage refuses to start unless the preceding decision artifact is valid and passing.

## 11. Non-goals

This iteration does not:

- retrain or fine-tune the policy;
- modify rewards, constraints, observations, task definitions, or context inference;
- change ODE equations or formal solver tolerances;
- tune the projection grid after viewing confirmatory results;
- treat raw rule control as a formal fallback;
- claim that the shield improves policy quality rather than execution feasibility;
- merge, push, or clean unrelated repository-reorganization changes.

## 12. Expected paper contribution if validated

If all stages pass, the defensible contribution is a transparent, rate-limit-preserving feasibility shield that removes rare policy-induced simulator failures with preregistered minimal intervention and bounded performance/constraint cost. The paper must continue to report the unshielded instability and present the shield as a distinct method, not conceal the failure or silently repair the baseline.
