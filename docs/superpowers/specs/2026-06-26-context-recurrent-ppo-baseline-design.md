# Context-RecurrentPPO Baseline Design

Date: 2026-06-26

## Purpose

Context-RecurrentPPO is a strong baseline for the Agri-MetaRL v2 paper. It tests whether the benefit of task information comes simply from giving the recurrent policy a learned task context, rather than from Agri-MetaRL v2's Monte Carlo advantage calibration.

The comparison target is:

- RecurrentPPO: no explicit task context.
- Context-RecurrentPPO: task context is concatenated to policy/value observations.
- Agri-MetaRL v2: task context is used for residual advantage calibration.

This baseline must not modify rewards, rewrite GAE, use result scaling, or use the Agri-MetaRL advantage residual head.

## Confirmed Approach

Use the same task support mechanism as Agri-MetaRL v2, but inject the inferred context into the policy/value input.

Context-RecurrentPPO will:

- Reuse `TaskSupportMemory`.
- Reuse `TransitionSetEncoder`.
- Maintain task-instance isolation through `task_instance_key`.
- Concatenate `raw_observation` and `context_vector` before policy/value evaluation.
- Train the context encoder end-to-end through PPO policy and value losses.
- Avoid Monte Carlo calibration memory and residual advantage correction.

Rejected approaches:

- Detached rollout-only context: invalid because the context encoder would not learn from PPO loss.
- Reusing a frozen Agri-MetaRL encoder: unfair because the baseline would depend on the proposed method.
- Adding an independent Monte Carlo auxiliary predictor: valid but less clean as a baseline because it introduces an extra learning objective.

## Architecture

The initial implementation should live in a separate baseline class rather than being mixed into Agri-MetaRL v2:

```text
src/gl_gym/RL/context_recurrent_ppo.py
configs/agents/context_recurrentppo.yml
tests/agri_metarl/test_context_recurrent_ppo.py
```

The main class should be named:

```python
ContextRecurrentPPO
```

It may subclass or wrap the existing RecurrentPPO flow, but it must keep these boundaries clear:

- Environment observations remain raw and unchanged.
- Algorithm-side policy input is augmented with context.
- Support memory stores raw transitions, not augmented observations.
- Agri-MetaRL v2 calibration components are not imported into the training path except for shared encoder/memory utilities.

## Data Flow

During rollout:

1. The environment emits a raw observation and task metadata.
2. The algorithm resolves the current `task_instance_key`.
3. The algorithm queries `TaskSupportMemory` for the current task instance.
4. If support is insufficient, it uses a zero context vector.
5. If support is sufficient, it encodes the support transitions with `TransitionSetEncoder`.
6. The policy/value network receives:

```text
augmented_observation = concat(raw_observation, context_vector)
```

7. After the step, the raw transition is written to support memory.
8. Support entries from one task instance must never be reused for another task instance.

The baseline should use a zero context during warm-up instead of a learned no-context token. This avoids adding extra parameters and keeps the comparison simpler.

## Training Path

The context encoder must receive gradients from PPO losses. Therefore the implementation must not rely only on context vectors precomputed during rollout and stored as NumPy arrays.

The rollout buffer or adjacent metadata structure must retain enough information to reconstruct context during minibatch training:

- raw observation
- action
- old log probability
- return
- advantage
- episode start marker
- task instance key
- support snapshot or support entry reference
- context active/query mask

During `train()`:

1. For each minibatch, reconstruct the relevant support set.
2. Recompute `context_tensor` using `TransitionSetEncoder`.
3. Concatenate `raw_observation_tensor` and `context_tensor`.
4. Pass the augmented tensor to recurrent policy evaluation.
5. Backpropagate PPO policy and value losses through both the policy/value network and the context encoder.

This is the key validity requirement. Without training-time context recomputation, the encoder would be detached from PPO and the baseline would become a random or stale feature baseline.

## Fairness Constraints

Context-RecurrentPPO should be matched to Agri-MetaRL v2 as closely as practical:

- Same support memory capacity.
- Same support warm-up rule.
- Same transition-set encoder architecture where possible.
- Same recurrent policy/value backbone.
- Same optimizer family, learning-rate schedule, rollout length, batch size, epochs, discount factor, and GAE lambda unless a deviation is explicitly documented.
- Same task distributions and evaluation seeds.

The only intended mechanism difference is where task context is used:

- Context-RecurrentPPO uses context as policy/value input.
- Agri-MetaRL v2 uses context to calibrate advantages.

## Configuration

Add a dedicated config file:

```text
configs/agents/context_recurrentppo.yml
```

Expected configuration fields include:

```yaml
agent: ContextRecurrentPPO
context_dim: 16
support_capacity: 64
support_min_size: 4
context_encoder_hidden_dim: 64
learning_rate: null
n_steps: null
batch_size: null
n_epochs: null
gamma: null
gae_lambda: null
```

Exact defaults should be selected during implementation by matching the existing RecurrentPPO and Agri-MetaRL v2 configs.

## Diagnostics

Record diagnostics that make the context mechanism auditable:

- `context_active_fraction`
- `context_norm_mean`
- `context_norm_std`
- `no_context_fraction`
- `support_size_mean`
- `context_between_task_variance`

These diagnostics should help distinguish a working context baseline from a baseline that mostly trains with zero context.

## Test Plan

Add focused tests before implementation.

Required coverage:

1. Raw environment observation space remains unchanged.
2. Augmented policy observation dimension equals `raw_obs_dim + context_dim`.
3. Context is zero before support is sufficient.
4. Different task instances have isolated support memory.
5. `TransitionSetEncoder` parameters change after a small PPO training run.
6. The baseline does not use advantage residual correction or Monte Carlo calibration.
7. A small CPU smoke train completes with finite losses and finite context norms.
8. Agent config resolution creates `ContextRecurrentPPO` explicitly and rejects unknown agent names.

The most important test is encoder parameter movement after training, because that proves the baseline is learning context through PPO rather than using detached features.

## Paper Description

Suggested manuscript wording:

```text
Context-RecurrentPPO is a task-context baseline that uses the same support memory and transition-set encoder as Agri-MetaRL, but injects the inferred task context directly into the recurrent policy and value networks. Unlike Agri-MetaRL, it does not perform Monte Carlo advantage calibration or residual advantage correction.
```

## Non-Goals

This design does not include:

- rewriting the manuscript results,
- launching full experiments,
- adding new task distributions,
- changing Agri-MetaRL v2,
- introducing a new auxiliary context prediction loss,
- committing or staging changes automatically.

Those steps should happen after a separate implementation plan is approved.
