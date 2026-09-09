# Agri-MetaRL 2.0 Monte Carlo Calibration Design

## Purpose

Replace the degenerate residual-only auxiliary objective in Agri-MetaRL 2.0 with a causally valid supervision signal. The revised method learns a bounded correction from rollout-truncated GAE to full-episode Monte Carlo advantage while preserving strict support/query separation and preventing future information from entering policy inference.

This document amends the optimization section of `2026-06-24-agrimetarl-paper-redesign-design.md`. The task descriptor, frozen support set, transition-set encoder, bounded residual, integrity pipeline, baselines, and gated Pilot remain unchanged.

## Root Cause

The current active v2 objective minimizes only a residual penalty. Its unique useful optimum is a zero residual, so the method collapses to Recurrent PPO and provides no learning signal for the transition-set encoder. Directly differentiating the same-batch PPO surrogate through the advantage head is also invalid because the head can improve the surrogate by changing its output without improving the policy.

The correction head therefore requires a detached outcome-based calibration target that cannot be manipulated by the head itself.

## Data Model

Each task instance owns one pending episode trajectory that can span any number of rollout fragments. A trajectory entry records:

- observation;
- raw rollout GAE;
- rollout-time value prediction;
- reward;
- terminal flag;
- task-instance key;
- whether the transition is support or query.

The frozen support transitions remain in `TaskSupportMemory`. Pending episode trajectories are stored separately so that support freezing, query classification, and calibration lifecycle have independent responsibilities.

No calibration sample is emitted before the episode terminates.

## Calibration Target

At episode termination, compute discounted returns backwards over the complete episode:

```text
G_t = r_t + gamma * G_(t+1)
MC_advantage_t = G_t - V_rollout_t
target_residual_t = clip(MC_advantage_t - raw_GAE_t, -alpha, alpha)
```

Only query transitions become calibration samples. The target, value prediction, raw GAE, and stored trajectory data are detached from the policy and value graphs.

The correction head retains the approved inference interface:

```text
predicted_residual_t = alpha * tanh(head(observation_t, raw_GAE_t, support_context))
corrected_advantage_t = raw_GAE_t + predicted_residual_t
```

Full-episode returns are labels used after an episode ends. They are never inputs to the policy, context encoder, or correction head during rollout collection or evaluation.

## Optimization

Completed calibration episodes enter a bounded FIFO queue. Each queued episode contains an immutable snapshot of its frozen support transitions plus its query calibration samples. This snapshot lets gradients continue to train the transition-set encoder even after the live support memory evicts the finished task instance. A meta update requires a configurable minimum number of query samples, and every query group is paired only with the support snapshot carrying the same task-instance key.

The calibration loss is:

```text
Huber(predicted_residual, target_residual)
  + residual_regularization * mean(predicted_residual^2)
```

The meta optimizer updates the transition-set encoder and residual head only. PPO continues to perform its standard policy and value updates, using corrected advantages generated from the current meta-head parameters. PPO minibatch-level advantage normalization remains unchanged.

The implementation uses gradient clipping and rejects non-finite meta batches instead of silently applying them.

## Episode Calibration Memory

Add an `EpisodeCalibrationMemory` component with these responsibilities:

1. append transitions to a pending trajectory keyed by complete task-instance identity;
2. attach raw GAE and rollout-time values after each rollout computes advantages;
3. retain incomplete episodes across rollout resets;
4. finalize a trajectory only after its terminal transition and all stored rows have GAE/value data;
5. compute discounted Monte Carlo targets and emit immutable query samples;
6. snapshot the matching frozen support set when finalizing an episode;
7. bound pending trajectories and completed episodes with deterministic oldest-first eviction.

It must not encode contexts, update neural networks, sample environments, or alter rollout-buffer values.

## Lifecycle

1. During environment stepping, append reward, observation, terminal state, task key, and query status.
2. Store a stable trajectory-entry identifier in the rollout buffer.
3. After `compute_returns_and_advantage`, attach each row's raw GAE and rollout-time value to its trajectory entry.
4. Finalize any now-complete episode and enqueue its query calibration samples.
5. Before PPO training, use the current meta head to correct eligible query advantages in the active rollout.
6. Run one or more supervised meta updates from the completed-sample queue.
7. Run the normal Recurrent PPO update.

Advantage correction is training-only and is not invoked while selecting evaluation actions. Evaluation may encode support sets for mechanism diagnostics, but it never uses future rewards or performs calibration updates.

## Diagnostics

Retain existing diagnostics and add:

```text
train/calibration_queue_size
train/completed_episode_count
train/mc_gae_abs_difference_mean
train/target_residual_clip_fraction
train/nonfinite_meta_batch_count
```

The Pilot gate continues to reject residual collapse, saturation, non-finite losses, cross-task leakage, or zero context variance.

## Failure Handling

- A terminal trajectory missing GAE/value attachments remains pending until attachment completes; it is never partially finalized.
- Duplicate trajectory-entry identifiers raise an error.
- A task key mismatch between a trajectory and its frozen support snapshot raises an error.
- Non-finite rewards, values, GAE, returns, contexts, predictions, or losses reject the affected calibration batch and increment a diagnostic counter.
- Completed calibration episodes are evicted oldest-first. Pending trajectories are limited separately; exceeding their configured capacity raises an error rather than silently dropping an active episode.

## Tests

Tests must demonstrate:

- an episode ending only after three rollout fragments produces complete targets;
- incomplete episodes produce no calibration samples;
- tasks, vector environments, and episode instances never share trajectory data;
- discounted returns and clipped residual targets match hand-calculated values;
- calibration samples contain query transitions only;
- future rewards are absent from inference inputs;
- a supervised meta update decreases loss on a deterministic batch;
- non-finite batches are rejected without parameter mutation;
- a two-rollout CPU smoke train and the full repository suite pass without NaN or Inf.

## Scope Boundaries

This revision does not add MAML, differentiate through PPO updates, inject context into the principal policy, use query rewards to build context, or change evaluation metrics. The context-conditioned Recurrent PPO remains a separate baseline. Confirmatory experiments remain blocked until the three-seed Pilot passes all integrity and mechanism gates.

## Implementation Status — 2026-06-25

Mechanism implementation and local verification are complete:

- `python -m pytest -q`: 43 passed, 5 pre-existing warnings;
- `python -m pytest tests\integrity\test_no_result_scaling.py -q`: 1 passed;
- `python -m compileall -q src tests`: passed;
- `git diff --check`: passed;
- active-v2 legacy-normalization scan: no matches;
- three-rollout CPU lifecycle test: passed with a completed calibration episode, finite meta update, and no non-finite diagnostic count.

These checks establish implementation behavior only. They provide no evidence of agronomic or algorithmic performance. All performance claims and manuscript result revisions remain blocked until the gated three-seed Pilot is complete and its raw records pass validation.
