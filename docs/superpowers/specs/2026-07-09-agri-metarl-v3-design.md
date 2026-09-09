# AgriMetaRL-v3 Design

## Context

The 2-seed pilot for `AgriControl_C_2026-06-30` is now internally complete: 8 learning runs × 91 evaluation tasks = 728 deterministic evaluation rows. After bounding the AgriMetaRL calibration queue and retraining `agri_metarl seed=42`, AgriMetaRL no longer fails catastrophically. However, the repaired pilot shows that PPO remains stronger on heldout, uncertainty, and economic splits, while AgriMetaRL is only best on the fixed task.

The current AgriMetaRL implementation uses a support-set context encoder mainly for advantage residual correction. The execution policy itself does not directly receive task context as part of its observation. That weakens the scientific claim that the method performs task-adaptive control.

## Goal

Build AgriMetaRL-v3 as a context-conditioned recurrent policy with meta advantage calibration and constraint-aware residual targets, then evaluate whether it can improve heldout, uncertainty, and economic robustness without increasing CO₂/RH violations.

## Design

AgriMetaRL-v3 will keep the existing `AgriMetaRL` public algorithm name and configuration entry, but extend its internals in three focused ways:

1. **Context-conditioned policy input**
   - Store the raw environment observation space separately.
   - Augment the policy observation space with `context_dim` coordinates.
   - During rollout collection, compute task context from the support memory before choosing the action.
   - Store the same support snapshots in the rollout buffer so train-time context can be recomputed with gradients.
   - During evaluation, raw observations are accepted and padded with zero context, matching current deterministic evaluation behavior. This keeps evaluation compatible while preserving trained model loading.

2. **Meta advantage calibration remains auxiliary**
   - The residual head still corrects rollout advantages for query samples.
   - The context encoder is shared by policy conditioning and residual calibration.
   - Existing bounded calibration memory remains in place to prevent memory blow-ups.

3. **Constraint-aware residual target**
   - Extend calibration samples with a scalar constraint penalty derived from environment `info` values: temperature violation, CO₂ violation, and RH violation.
   - Adjust the residual target so high-violation query transitions receive a more conservative corrected advantage.
   - Start with conservative default weights in config, not a large reward rewrite.

## Success Criteria

For a 2-seed pilot:

- Fixed split remains competitive with PPO.
- Heldout, uncertainty, and economic splits improve over current AgriMetaRL-v2.
- CO₂ and RH violations decrease versus current AgriMetaRL-v2 and move closer to PPO/RecurrentPPO.
- No calibration queue growth above the configured cap.
- All AgriMetaRL and experiment tests pass.

For paper-grade results:

- Expand to all manifest seeds: `42, 123, 456, 789, 1024`.
- Validate artifacts only after 5-seed completion.
- Do not update manuscript claims until the 5-seed suite has been summarized.

## Non-goals

- Do not rewrite the greenhouse reward function globally.
- Do not remove PPO/RecurrentPPO baselines.
- Do not claim superiority from 2 seeds.
- Do not change unrelated repository layout during this method iteration.
