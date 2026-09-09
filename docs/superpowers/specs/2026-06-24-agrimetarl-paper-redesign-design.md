# Agri-MetaRL 2.0 Paper and Experiment Redesign

## Objective

Produce a defensible, reproducible Q2-level research paper that balances algorithmic novelty with greenhouse-control relevance. Replace the current unreliable result pipeline, redesign Agri-MetaRL around genuine task-aware cross-rollout support/query learning, rerun a fair experiment matrix, and rewrite the manuscript strictly from validated outputs.

## Non-Negotiable Integrity Rules

- Remove every algorithm-dependent scaling, ranking, or result-adjustment path.
- Never alter raw measurements to obtain a preferred method ordering.
- Treat independent training seeds and evaluation tasks as statistical units; never inflate sample size by repeating an identical deterministic episode.
- Require equal seed sets, task grids, training budgets, and evaluation protocols across learned methods.
- Generate tables and figures from validated data rather than copying numbers into LaTeX manually.
- Rewrite conclusions to match observed results even if Agri-MetaRL 2.0 does not outperform every baseline.
- Preserve old models, results, manuscript versions, and submission materials in their existing local archive locations.
- Store all new environments, caches, checkpoints, experiment outputs, and temporary artifacts on the E: drive.
- Do not upload, submit, push, stage, commit, or delete research artifacts automatically.

## Algorithm: Agri-MetaRL 2.0

### Task Definition

Each task has a complete, serializable descriptor:

```text
TaskDescriptor(
    weather_year,
    start_day,
    parameter_uncertainty,
    economic_scenario,
    climate_constraint_scenario,
)
```

The descriptor, not an episode-start marker, defines task membership. The environment must expose the complete descriptor in every transition's `info` dictionary.

### Cross-Rollout Episodic Memory

A task-aware episodic memory persists support transitions across rollout boundaries. It must:

- key all entries by the complete task descriptor;
- accept transitions from multiple rollout fragments of the same episode;
- retain only prior support transitions when constructing context for a query transition;
- prevent data sharing across task descriptors;
- use a bounded per-task capacity and deterministic eviction policy;
- reset or checkpoint its state explicitly at training boundaries.

This removes the current dependency on `episode_starts`, which causes meta-updates to be skipped when a rollout begins in the middle of a 5760-step episode.

### Support Encoder

The support encoder consumes transition tuples `(observation, action, reward, next_observation, done)`. A shared transition network maps each tuple to an embedding; permutation-invariant pooling produces a fixed-size context vector. The first implementation uses mean pooling to keep the mechanism small and auditable. Attention pooling is outside the initial scope unless pilot evidence justifies it.

### Advantage Residual

For each query transition:

```text
delta_t = correction_head(observation_t, raw_gae_t, context_task)
corrected_advantage_t = raw_gae_t + alpha * tanh(delta_t)
```

`alpha` is explicit and bounded. Any normalization is applied consistently at the PPO minibatch level, not separately inside each query fragment. The residual head cannot use query rewards or returns to construct context.

### Optimization

- Standard Recurrent PPO remains the base learner.
- Support transitions construct context.
- Query transitions receive corrected advantages and participate in PPO optimization.
- An auxiliary calibration loss may compare the corrected advantage with a detached return-based target, with a residual regularizer that discourages unnecessary correction.
- Auxiliary-loss weights and residual bounds are declared in configuration and covered by ablation.
- No MAML inner loop, task-specific policy branch, or policy-context injection is added to the principal method.

### Required Baselines

- Rule-based controller.
- PPO.
- Recurrent PPO.
- Original Agri-MetaRL implementation.
- Agri-MetaRL 2.0.
- Context-conditioned Recurrent PPO, where context enters the policy/value input rather than the advantage residual.

Recurrent PPO and both context/meta variants use matched backbones, optimizer settings, training steps, rollout sizes, and seed sets unless a documented method constraint requires otherwise.

## Experimental Design

### Task Distributions

Training uses Amsterdam weather years 2001-2012. Validation uses 2013-2015. Final temporal OOD testing uses 2016-2020. No final-test year may influence checkpoint selection or hyperparameter selection.

Training tasks stratify start dates across multiple seasonal windows and sample:

- parameter uncertainty levels of 0%, 5%, and 10%;
- documented economic scenarios derived from energy, CO2, and fruit-price coefficients;
- standard, strict, and relaxed climate-constraint scenarios.

Scenario values must be declared in configuration and justified from either the inherited benchmark or cited agricultural/economic sources. The evaluation grid records exact task descriptors rather than relying on random resets.

### Evaluation Suites

- In-distribution: seen year range and scenario ranges with held-out task combinations.
- Temporal OOD: years 2016-2020.
- Compositional OOD: unseen combinations of start date, economic scenario, and constraint scenario.
- Stress OOD: higher uncertainty and extreme-weather windows not used in training.

Every method is evaluated on the same explicit task grid. Each model-task pair produces one deterministic record. Optional stochastic evaluation must use declared, shared evaluation seeds and be reported separately.

### Staged Compute Plan

#### Pilot

- Three independent seeds.
- Reduced training budget.
- Small but representative ID and OOD task grid.
- Compare Recurrent PPO, original Agri-MetaRL, Agri-MetaRL 2.0, and the context-conditioned baseline.
- Stop before confirmatory runs if the implementation is unstable, the context is unused, or the residual collapses/saturates.

#### Confirmatory

- Five independent seeds per selected learned method.
- Two million environment steps per seed unless pilot runtime evidence requires one shared revised budget.
- Complete ID, temporal OOD, compositional OOD, and stress OOD grids.
- Rule-based evaluation over the same task grid.

### Ablations

- No cross-rollout episodic memory.
- No task context.
- No observation input to the correction head.
- Unbounded residual in place of `alpha * tanh(delta)`.
- Support-ratio and support-length sensitivity.

Pilot ablations may use three seeds for screening. Any ablation used for a paper conclusion must be rerun with the confirmatory seed set.

### Metrics

- Episode return.
- Revenue, heating, electricity, CO2 cost, and EPI.
- Temperature, relative-humidity, and CO2 violations reported as both steps and time-within-bounds.
- Worst-task and lower-quantile performance in addition to the mean.
- Training stability and convergence.
- Additional parameters, training time, inference time, and memory overhead.

### Statistical Analysis

- Report mean, standard deviation, 95% bootstrap confidence interval, and an effect size.
- Pair methods by training seed and evaluation task wherever the design supports pairing.
- Use a paired hierarchical bootstrap over training seeds and evaluation tasks for the primary confidence interval on method differences. Use an exact Wilcoxon signed-rank test on seed-level task-aggregated scores as a secondary test when at least five nonzero pairs are available.
- Apply Holm correction to planned multiple comparisons.
- Separate descriptive results from inferential claims.
- Avoid `significant`, `best`, `robust`, and equivalent claims unless directly supported by the declared analysis.

## Data Architecture

```text
experiment configuration snapshot
  -> training run manifest
  -> model checkpoint and normalization state
  -> immutable per-model/per-task raw evaluation record
  -> validation gate
  -> aggregate result tables
  -> generated LaTeX tables and paper figures
  -> manuscript
  -> compiled and rendered PDF
```

Each raw result records:

- repository commit or working-tree identifier;
- configuration hash;
- algorithm and implementation version;
- training seed;
- model and normalization-state paths;
- complete task descriptor;
- evaluation mode and evaluation seed when applicable;
- every reported raw metric.

Raw result files are append-only. Aggregation writes new derived files and never overwrites raw observations.

## Validation Gates

The result validator rejects a dataset if any of the following holds:

- algorithms have different confirmatory seed sets;
- a required model-task pair is missing;
- a model-task pair is duplicated without a declared stochastic evaluation seed;
- task descriptors are incomplete;
- configuration hashes do not match the run manifest;
- the number of methods, seeds, or tasks differs from the experiment specification;
- non-finite metrics or physically invalid values appear;
- an aggregation attempts algorithm-dependent scaling.

The current six-PPO/five-other-model fixed-protocol dataset and repeated deterministic rows must fail this validator.

## Testing Strategy

### Unit Tests

- Task descriptor serialization and equality.
- Environment emission of the complete descriptor.
- Cross-rollout memory persistence and deterministic eviction.
- Cross-task isolation.
- Support-before-query temporal isolation.
- Residual bound and gradient flow.
- Corrected advantage shape, normalization, and finite-value checks.
- Rejection of unbalanced seeds, duplicate deterministic records, missing tasks, and result scaling.

### Integration Tests

- Tiny synthetic task environment proving different tasks obtain different contexts.
- Rollout fragments spanning episode boundaries.
- Short CPU training smoke test for Recurrent PPO and Agri-MetaRL 2.0.
- Evaluation-to-validation-to-aggregation-to-LaTeX pipeline using synthetic results.
- Automated comparison between generated LaTeX table values and aggregate CSV values.

### Pilot Diagnostics

- Context variance across tasks.
- Context consistency within a task.
- Residual magnitude and saturation rate.
- Fraction of query transitions receiving correction.
- Meta-loss and PPO-loss stability.
- Training throughput and GPU/CPU memory use.

## Manuscript Redesign

The abstract, highlights, contributions, methods, protocol, results, discussion, and conclusion are rewritten after confirmatory results exist. No current performance claim is retained merely for continuity.

Add:

- complete task-distribution definition;
- algorithm pseudocode;
- cross-rollout memory mechanism;
- support/query leakage controls;
- architecture and parameter-count comparison;
- training hardware and runtime;
- explicit ID/OOD evaluation grids;
- statistical-analysis protocol;
- ablation and sensitivity results;
- threats to internal, external, and construct validity.

Figure 1 is redrawn to show the complete task descriptor, environment, episodic memory, support encoder, query correction, and PPO update. Figure 2 shows the precise cross-rollout data flow and loss paths. All result figures are generated from validator-approved aggregate data.

Use LaTeX-safe currency and degree notation. The final document must contain resolved citations, embedded figures rather than draft placeholders, a complete bibliography, and no broken glyphs.

## Completion Criteria

The redesign is complete only when:

- the integrity validator passes all confirmatory raw results;
- Agri-MetaRL 2.0 behavior is covered by unit and integration tests;
- the pilot diagnostics support continuing to confirmatory training;
- confirmatory methods use matched seeds, task grids, and budgets;
- every paper table and figure is generated from validated data;
- every manuscript number matches its source data automatically;
- the final PDF compiles, renders, and passes a page-by-page visual review;
- claims accurately reflect the observed results and statistical evidence.
