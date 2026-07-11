# Online Context Evaluation and A/B Diagnostic Design

## Status

Approved in conversation on 2026-07-10. This design defines the next research gate for AgriMetaRL-v3. It does not change the training algorithm or authorize additional long-horizon training.

## Problem

AgriMetaRL-v3 is trained with a policy observation containing the raw greenhouse observation and a support-conditioned context vector. The current evaluation path calls `model.predict()` with raw observations only. `AgriMetaRL.predict()` pads those observations with a zero context, and the evaluator never returns completed transitions to the model. Consequently, the current deterministic suite cannot measure online context adaptation.

Continuing from approximately 196,608 steps to 2,000,000 steps before resolving this mismatch would consume substantial compute without producing valid evidence for the method's central task-adaptation claim.

## Objective

Add a leakage-safe online inference protocol for context-conditioned agents, then compare online-context and zero-context inference using the same existing AgriMetaRL-v3 checkpoints. Use the result as a gate for further training or method redesign.

## Scope

This iteration will:

- add an explicit episode-scoped inference state to `AgriMetaRL`;
- update that state from evaluation transitions only;
- allow deterministic evaluation to select `online_context` or `zero_context` mode;
- preserve the existing behavior of PPO, Recurrent PPO, and other agents;
- run a small stratified A/B diagnostic on seeds 42 and 123;
- write diagnostic results to a new artifact path without modifying the formal C-route result table.

This iteration will not:

- resume long-horizon training;
- change the context encoder, residual head, reward, or constraint weights;
- overwrite `AgriControl_C_2026-06-30` results;
- update manuscript claims or figures;
- treat the diagnostic subset as paper-grade evidence.

## Considered Approaches

### 1. Continue training and retain zero-context evaluation

This is operationally simple but scientifically invalid for a context-conditioned policy. It is rejected.

### 2. Add explicit online inference hooks and perform a same-checkpoint A/B test

This directly tests whether learned context changes deployment behavior, isolates the effect from training randomness, and requires no retraining. This is the selected approach.

### 3. Put evaluation logic inside a model-specific evaluator

This could work quickly, but it would duplicate transition and memory logic outside the algorithm and make leakage control harder to audit. It is rejected in favor of small model hooks called by the generic evaluator.

## Architecture

### Episode-scoped inference state

`AgriMetaRL` will expose a minimal optional protocol:

- `begin_inference_episode(mode)`: validate the mode, create a fresh empty inference support memory, and clear the current inference task key;
- `predict(...)`: when inference mode is `online_context`, augment raw observations with context computed only from the current inference memory; when mode is `zero_context`, preserve current zero-padding behavior;
- `observe_inference_transition(...)`: after each environment step, construct a normalized-observation transition from the action, reward, next observation, done flag, and environment info, then add it to inference memory;
- `end_inference_episode()`: discard the episode-scoped inference state.

These hooks are evaluation-only. They must not mutate training support memory, calibration memory, optimizer state, rollout buffers, or saved model parameters.

### Evaluator integration

`run_deterministic_episode()` will detect the optional inference protocol by capability rather than by algorithm class.

The episode data flow will be:

1. Reset the environment and recurrent state.
2. Begin a fresh inference episode in the requested mode.
3. Predict the first action. Online mode uses zero context because no transition has yet been observed.
4. Step the environment.
5. Pass the pre-step normalized observation, executed action, reward, post-step normalized observation, done flag, and info to `observe_inference_transition()`.
6. Use the updated support memory to construct context for the next prediction.
7. End and clear inference state even when evaluation raises an exception.

The evaluator will continue to work unchanged for models that do not expose these hooks.

### Observation semantics

The inference transition must use observations in the same normalized space used during training. The evaluator therefore passes the observations visible at the model boundary, not values returned by `env.unnormalize_obs()`.

For terminal transitions, the hook will prefer the normalized terminal observation supplied by the vectorized environment when available. The initial implementation targets one evaluation environment, matching the robust suite.

### Task identity and leakage prevention

The first environment step provides `task_instance_key` in `info`. That key becomes the current inference key for subsequent actions. The first action is necessarily context-free.

Every evaluated task starts with a newly constructed empty inference memory. Training memory loaded from a checkpoint must never be read by online evaluation. No support transition may persist across task records, seeds, or evaluation modes.

## Diagnostic Experiment

### Checkpoints

Use the existing `last_model.zip` checkpoints at 196,608 steps for seeds 42 and 123 from `AgriControl_C_2026-07-09-v3-pilot3`. The same checkpoint is evaluated in both modes.

### Task subset

Use these eight deterministic task IDs, selected before looking at A/B outcomes:

- `fixed_2010_d59_u0p00_standard`;
- `heldout_2011_d59_u0p00_standard`;
- `heldout_2012_d59_u0p00_standard`;
- `heldout_2013_d59_u0p00_standard`;
- `uncertainty_2012_d80_u0p05_standard`;
- `uncertainty_2013_d100_u0p15_standard`;
- `economic_2011_d59_u0p00_high_energy_price`;
- `economic_2013_d100_u0p00_combined_stress`.

The diagnostic must fail with a missing-task error if any identifier is absent; it must not substitute another task silently.

This yields 32 episodes: 2 seeds x 8 tasks x 2 inference modes.

### Outputs

Write all generated files below the new result root:

`artifacts/results/AgriControl_C_2026-07-10-v3-context-ab/`

Required outputs:

- a diagnostic manifest containing checkpoint paths, exact task IDs, modes, and software revision;
- raw per-episode metrics;
- paired deltas computed as `online_context - zero_context` for each seed-task pair;
- split-level summaries for episode return, EPI, temperature violation, CO2 violation, and RH violation;
- a short machine-readable decision record containing the gate outcome and reasons.

The original pilot registry and formal C-route result files remain unchanged.

## Decision Gate

The diagnostic is directional and mechanistic, not a significance test.

Continue training to an intermediate target near 500,000 steps only when all of the following hold:

1. Online context changes actions after the support set becomes ready for both seeds.
2. Mean paired return delta across the seven non-fixed diagnostic tasks is positive.
3. Neither seed has a negative non-fixed mean return delta larger than 2% of its zero-context mean absolute return.
4. Online context does not increase the mean normalized violation burden by more than 5%, where normalized burden is the mean of each violation metric divided by its zero-context mean plus a small numerical epsilon.
5. Fixed-task return does not decrease by more than 2% for either seed.

If actions do not change, inspect context propagation, support readiness, and policy sensitivity before any more training. If actions change but the gate fails, redesign the context representation or its optimization before resuming training. If the gate passes, set the next training target to 500,000 steps and repeat the full 91-task, two-seed evaluation before authorizing 2,000,000-step or five-seed experiments.

## Diagnostics

Record per episode:

- the first step at which the support set becomes ready;
- mean and maximum context norm after readiness;
- mean absolute action difference between online and zero modes, computed during paired post-processing;
- total return and economic/climate metrics;
- whether all values remained finite.

Continue tracking existing training diagnostics separately. The current residual magnitude near 1e-4 is a warning that the advantage-correction mechanism may be practically inactive, but changing its scale is outside this iteration.

## Error Handling

- Reject unknown inference modes.
- Raise a clear error when online mode receives incomplete task identity after a step.
- Reject non-finite observations, rewards, contexts, or actions in the inference path.
- Always clear inference state in a `finally` block.
- Fail rather than silently falling back to zero context when online mode was explicitly requested.
- Preserve existing behavior for models without inference hooks.

## Testing

Implementation will be test-driven and cover:

1. A new inference episode starts with empty support memory even after loading a trained checkpoint.
2. The first online prediction uses zero context.
3. Transitions accumulate only in inference memory and make context available at `support_size`.
4. Online prediction uses the learned context after readiness.
5. Zero-context mode remains identical to the current raw-observation prediction path.
6. Ending or failing an episode clears inference state.
7. Generic evaluation invokes hooks in the correct order and remains compatible with PPO-like models without hooks.
8. Terminal observations and recurrent episode-start flags are handled correctly.
9. A smoke A/B evaluation writes distinct mode keys without overwriting formal results.

Focused tests will run before the full repository test suite. A small deterministic environment will be used to verify that online and zero-context actions diverge only after support readiness.

## Research Interpretation

A passing diagnostic will show only that the learned context has a beneficial deployment-time effect on the selected tasks. It will not establish superiority, statistical significance, or paper-grade robustness. Those claims require the subsequent 91-task evaluation, ablations, five seeds, confidence intervals, and matched baseline comparisons.

The manuscript and figures remain frozen until those requirements are met.
