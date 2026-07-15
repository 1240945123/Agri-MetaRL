# Conservative Feasibility Action Shield v2 Design

## 1. Decision context

The preregistered v1 action shield searched the fixed convex grid in increasing order and selected the smallest one-step-feasible intervention. Its Stage 3 evaluation reduced formal ODE failures from 22/182 unshielded episodes to one remaining failure, but therefore failed the unchanged zero-failure gate.

The remaining failure is deterministic: seed 123 on `uncertainty_2012_d100_u0p10_standard`. The unshielded trajectory fails at step 4168. The v1 shield intervenes at steps 4167, 4168, and 4169 with lambda values 0.0625, 0.25, and 0.0625, then exhausts every candidate at step 4170. At that state, hold, full-range, and per-channel emergency actions also fail. A diagnostic counterfactual using the same candidate set in descending lambda order completes all 5760 steps.

This is a new method iteration after a failed confirmatory run. All v1 artifacts remain immutable and are reported as v1 evidence; no v1 threshold, task, metric, or result is relabelled.

## 2. Considered approaches

### A. Conservative-first reactive projection (selected)

Keep the trigger, reference controller, convex grid, solver settings, and transactional semantics unchanged, but search lambda in the fixed order `[1, 1/2, 1/4, 1/8, 1/16]`. This directly addresses the observed sequence of marginal recoveries, has already completed the unique v1 failure trajectory in a diagnostic counterfactual, and changes the fewest method components.

### B. Multi-step lookahead projection

Evaluate each candidate for two or more future steps and select the smallest candidate with a feasible rollout. This could retain smaller interventions, but it adds a forecast policy for future actions and disturbances, multiplies solver cost, and creates new design choices that are not identified by the single observed failure.

### C. Numerical fallback or state repair

Change tolerances, substep the ODE, clip state, or switch integrators after failure. This may improve numerical completion but changes simulator semantics and confounds controller robustness with a different numerical model. It is rejected for this iteration.

## 3. Method definition

The policy action, rule reference action, and convex candidate equation are unchanged:

```text
a(lambda) = (1 - lambda) * a_policy + lambda * a_reference
```

The v2 priority is fixed before confirmatory reruns:

```text
lambda_priority = [1, 1/2, 1/4, 1/8, 1/16]
```

The original policy action is still attempted exactly once. Projection is invoked only after that action fails. The projector selects the first feasible candidate in the descending priority, stops immediately, and commits the already computed state exactly once. Thus v2 is a conservative recovery policy, not a continuously active rule controller and not a new action set.

The descending order, schema version, method identifier, and candidate-attempt order are fingerprinted evidence. v2 records must never validate as v1 records.

## 4. Scope and invariants

v2 must not change:

- checkpoints, tasks, seeds, inference definitions, weather, uncertainty, economics, or horizons;
- the reference controller or actuator rate limits;
- the candidate lambda values;
- `FORMAL_CVODES_OPTIONS`, model equations, reward, constraints, or observations;
- intervention, return-loss, violation, or zero-failure thresholds;
- unshielded execution behavior.

Every retry uses the identical state, weather, sampled parameters, and fresh formal integrator. Failed attempts cannot mutate later attempts. Existing action and trace validation remains strict, with the expected prefix changed to descending order.

## 5. Evidence and evaluation sequence

New v2 output roots are mandatory. The partial v1 Stage 3 work directory and the nonformal diagnostic scan are retained as redesign evidence, not scored as v2 results.

1. **Mechanism regression:** unit and transaction tests prove exact descending order, early stopping, immutable attempts, and v1/v2 schema separation.
2. **Stage 1 v2:** replay the known deterministic failure and require the selected candidate to be the first successful candidate in descending priority.
3. **Stage 2 v2:** rerun the unchanged 32-episode context A/B protocol with new v2 provenance. All four existing gates must pass.
4. **Stage 3 v2:** rerun the complete 182-episode shielded suite. Because the runtime source fingerprint changes, regenerate the formal unshielded comparator under the same committed source revision rather than rewriting or migrating old evidence.
5. **Gate:** run the unchanged four Stage 3 thresholds. A robust-execution claim is allowed only if all 182 shielded episodes complete with zero ODE failures, intervention rate is at most 0.005, relative return loss is at most 0.02, and mean normalized violation burden is at most 1.05.

The paper reports v1 as a failed minimal-intervention design and v2 as the evidence-driven conservative redesign. This separation prevents post hoc tuning from being hidden.

## 6. Testing and failure handling

Tests must cover descending construction, descending first-feasible selection, all-candidate failure, selected-control/integrator alignment, attempt-record validation, fingerprints, Stage 1 prerequisites, Stage 2 prerequisites, resume identities, and Stage 3 gate compatibility. Existing unshielded tests must remain unchanged and passing.

If v2 still exhausts all candidates, the run is a failed confirmatory attempt. The next redesign must be based on its recorded task and state evidence; the lambda values or gates cannot be changed inside the same run.

## 7. Expected claim if validated

The defensible claim is that rare numerical failures of the fixed policy can be removed by a deterministic, rate-limit-preserving conservative recovery layer over a preregistered action grid, with intervention and performance costs bounded by unchanged gates. The shield remains a distinct evaluation method and does not retroactively repair or conceal the unshielded baseline.
