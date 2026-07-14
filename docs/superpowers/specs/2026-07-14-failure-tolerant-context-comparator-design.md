# Failure-Tolerant Context Comparator Design

**Date:** 2026-07-14

**Status:** Approved under the user's standing authorization to adopt the recommended design

## Purpose

Produce the exact 32-key unshielded comparator required by the shielded context A/B gate without concealing deterministic ODE failures or fabricating unexecuted rows. The comparator is a separate diagnostic artifact. It does not replace the original fail-fast context A/B run or its evidence.

## Alternatives

The selected approach is a new, failure-tolerant comparator runner. Modifying `run_context_ab.py` was rejected because it would change the semantics of an existing preregistered diagnostic. Deriving missing rows from the interrupted 18-row progress file was rejected because unexecuted episodes are not failures or measurements.

## Protocol

The runner executes exactly two seeds, eight preregistered task IDs, and both inference modes, for 32 unique `(seed, task_id, inference_mode)` keys. It uses the same checkpoints, task selection, environment loader, inference hooks, solver configuration, failure capsule recorder, and provenance sources as the existing context diagnostic.

A successful episode produces finite metrics, context diagnostics, and an executed-action trace. Its row has `completed=true`, `status=completed`, `ode_failure_count=0`, and an empty failure-evidence path.

An incomplete episode is accepted only when the real episode runner raises its expected early-horizon wrapper and the current attempt creates exactly one new, valid ODE failure capsule. The capsule must match the row's seed, task, inference mode, task descriptor, checkpoint path and hash, source checksums, formal solver options, failure timestep, and wrapper step. Its row has `completed=false`, `status=ode_failure`, `ode_failure_count=1`, all scoring metrics set to non-scoring NaN, and a final capsule manifest path.

All other errors remain fatal. This includes prediction and context errors, malformed environment output, missing or multiple capsules, recorder failure, provenance mismatch, cleanup failure, and `KeyboardInterrupt` or `SystemExit`.

## Resume semantics

The runner may import the 18 completed rows from the existing interrupted work file only after strict validation. Each imported row must have an approved key, exact checkpoint and source provenance, a valid canonical trace, finite metrics, valid mode-aware context diagnostics, and no failure evidence. Invalid rows are recomputed.

Native comparator progress is also resumable. Completed rows require valid traces. Failed rows require reloading and fully validating their capsule. Paths are canonical per key and globally unique. A changed checkpoint, source byte, task table, configuration, trace, or capsule invalidates the row.

## Artifact lifecycle

All mutable progress lives below a comparator-specific sibling `.work` directory. Failure attempts use per-output, per-key directories, so different result roots cannot delete one another's evidence.

The final root is published only after all 32 keys are present and valid. It contains:

- `eval_raw.csv` with the explicit comparator protocol;
- `context_ab_manifest.json` binding checkpoints, inputs, runtime source tree, solver settings, and all final artifact hashes;
- `traces/` containing completed-episode action traces;
- `failures/` containing complete immutable failure capsules.

Final CSV paths point only inside the final root. Publication uses sibling staging, backup, atomic replacement, filesystem-inferred recovery, and copy fallback. A partial run cannot create a valid final comparator. Existing final results and the original failure capsule are protected from output, work, staging, and backup overlap.

## Consumer validation

After publication, the runner invokes the existing shielded Stage 2 comparator loader against the final root and expected checkpoint/provenance map. Publication is considered successful only if this independent validation returns all exact 32 keys and validates every failure capsule.

The runner does not compute the shield gate. Its sole output is an immutable unshielded comparator. The existing shielded Stage 2 runner remains the only component authorized to pair shielded and unshielded rows and apply the preregistered thresholds.

## Testing

Tests cover exact key enumeration, successful episodes, real wrapper-versus-underlying ODE exception behavior, multiple independent ODE failures, rejection of arbitrary exceptions and interrupts, strict capsule identity, import of valid legacy progress, stale progress recomputation, canonical paths, resume after interruption, cleanup priority, 31-row refusal, atomic publication and restoration failures, protected-root topology, and round-trip acceptance by `load_unshielded_comparator()`.

## Non-goals

This component does not alter the policy, shield, tasks, metrics, solver, checkpoints, context modes, or thresholds. It does not infer results for unexecuted episodes, convert arbitrary exceptions into failures, generate the original online-versus-zero decision, or authorize Stage 3 directly.
