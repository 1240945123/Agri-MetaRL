# Robust Experiment Redesign for a Q2-Target Greenhouse RL Paper

Date: 2026-06-30  
Workspace: `E:\school\Paper\new`  
Scope: redesign the experimental protocol, result pipeline, and manuscript frame for a stronger 2区/Q2-target paper.

## 1. Problem Diagnosis

The current manuscript and artifact set are not strong enough to support a high-quality paper in their present form.

The main issues are:

- The manuscript describes an older Agri-MetaRL design based on `MetaAdvantageHead` and support statistics, while the current implementation uses `TaskSupportMemory`, `TransitionSetEncoder`, `AdvantageResidualHead`, complete-episode Monte Carlo calibration, and MC residual labels.
- Several manuscript claims are not supported by the current CSV artifacts. Current fixed-protocol summaries do not show Agri-MetaRL winning return or all violation metrics.
- `artifacts/results/AgriControl/fixed_protocol/raw.csv` contains repeated deterministic rows. These repeated episodes must not be counted as independent samples.
- Current learning-curve, heldout, few-update, and ablation artifacts are incomplete or uneven across methods.
- Context-RecurrentPPO has been implemented and tested but has not yet been included in trained/evaluated paper results.
- Current figures are likely stale and should not be reused for the redesigned paper.

The redesign therefore treats the existing artifacts as archival only. New conclusions must come from a new, manifest-controlled experiment suite.

## 2. Paper Framing

The paper should be reframed from a single fixed-protocol win claim to a task-distribution-aware robust greenhouse control study.

Recommended core question:

> In greenhouse climate control across weather years, start dates, parameter uncertainty, and economic scenarios, how should task context enter reinforcement learning?

The paper should compare three adaptation mechanisms:

1. Recurrent memory: RecurrentPPO.
2. Explicit task context as policy/value input: Context-RecurrentPPO.
3. Task context as advantage calibration: Agri-MetaRL v2.

The target narrative is:

- Build a robust task-distribution evaluation protocol for greenhouse RL.
- Compare memory, explicit context, and contextual advantage calibration under the same protocol.
- Use diagnostics and ablations to explain when Agri-MetaRL v2 helps, rather than assuming it wins everywhere.

Conclusions must be determined after real results are generated. If Agri-MetaRL v2 wins broadly, the manuscript can make stronger method claims. If it wins only in specific splits or scenarios, the manuscript should emphasize mechanism boundaries and robustness trade-offs.

## 3. Algorithms

The main comparison set is:

- PPO
- RecurrentPPO
- Context-RecurrentPPO
- Agri-MetaRL v2
- Rule-based baseline

Agri-MetaRL v2 must be described according to the current implementation:

- `TaskSupportMemory`
- `TransitionSetEncoder`
- `AdvantageResidualHead`
- complete-episode Monte Carlo calibration
- completed calibration queue
- residual labels from MC-vs-GAE advantage differences

Context-RecurrentPPO must be described as a mechanism comparator:

- It uses the same support-memory and transition-set encoder idea.
- It concatenates context into policy/value observations.
- It does not use advantage residual correction, Monte Carlo calibration, or advantage rewriting.

This distinction is central to the paper: the experiment compares where context enters the learning system, not only whether context exists.

## 4. Task Distribution

The redesigned task distribution has three levels.

### Level 1: Weather and start-day generalization

- Train year: 2010.
- Held-out years: 2011, 2012, 2013.
- Start days: 59, 80, 100.

This creates a basic held-out weather/start-date split and prevents the paper from relying on a single deterministic climate trajectory.

### Level 2: Parameter robustness

Evaluate trained models under uncertainty scales:

- 0.00
- 0.05
- 0.10
- 0.15

The main outputs are degradation slope, violation growth, and failure/tail-risk behavior as uncertainty increases.

### Level 3: Economic scenario shift

Evaluate policies under:

- standard economics
- high energy price
- low tomato price
- high CO2 price
- combined stress

The exact multipliers should be implemented as named scenario definitions in code and recorded in the suite manifest. The paper should report return, EPI, revenue, heat/electricity/CO2 cost breakdown, and climate violations for each scenario.

## 5. Experiment Matrix

The experiment should use a two-stage matrix.

### Stage 1: Pilot

Purpose:

- Check training stability.
- Detect non-finite losses or calibration collapse.
- Confirm the context encoder updates as expected.
- Identify whether the proposed task distribution separates methods.

Algorithms:

- PPO
- RecurrentPPO
- Context-RecurrentPPO
- Agri-MetaRL v2

Seeds:

- 42
- 123

Training budget:

- 300k to 500k timesteps per run.

Pilot results are for go/no-go and protocol debugging, not final paper claims.

### Stage 2: Full experiment

Algorithms:

- PPO
- RecurrentPPO
- Context-RecurrentPPO
- Agri-MetaRL v2
- Rule-based baseline

Seeds:

- 42
- 123
- 456
- 789
- 1024

Training budget:

- 2M timesteps per learning-based method, unless pilot evidence forces a documented adjustment.

Evaluation groups:

1. Fixed protocol: 2010, start day 59, uncertainty 0, standard economics.
2. Held-out weather/start days: 2011-2013 × start days 59/80/100.
3. Parameter robustness: held-out weather/start tasks × uncertainty scales 0.00/0.05/0.10/0.15.
4. Economic scenario shift: named economic scenarios under the held-out task set.

The fixed protocol is a sanity check, not the sole basis for the main conclusion.

## 6. Agri-MetaRL v2 Ablations

The redesigned paper needs real ablations. At minimum:

- Full Agri-MetaRL v2.
- Without Monte Carlo calibration.
- Without residual correction, or with zero residual.
- Shuffled task keys.
- Context-RecurrentPPO as a strong mechanism comparator.

The ablations should test which part of the method matters:

- Does complete-episode MC calibration improve learning signal quality?
- Does the residual head provide useful correction beyond standard GAE?
- Does task identity/context have meaningful signal, or can shuffled task keys perform similarly?
- Is policy-input context enough, or is advantage calibration more robust?

## 7. Metrics and Statistics

Main economic metrics:

- return
- EPI
- revenue
- heating cost
- electricity cost
- CO2 cost

Main safety/control metrics:

- temperature violation
- relative humidity violation
- CO2 violation
- TWB percentage
- 60-day climate/control trajectories for case studies

Main robustness metrics:

- held-out return/EPI
- uncertainty degradation slope
- violation growth under uncertainty
- economic scenario degradation
- seed variance

Mechanism diagnostics:

- context active fraction
- context embedding norm
- calibration loss
- residual correction magnitude
- MC-vs-GAE advantage differences

Statistical rules:

- Repeated deterministic episodes must not be treated as independent samples.
- The default statistical unit is seed/run.
- For held-out tasks, aggregate task-level metrics within each seed first, then compute mean and standard deviation across seeds.
- Report effect sizes such as Cohen's d or Cliff's delta where appropriate.
- P-values are auxiliary and should not replace effect-size interpretation.

## 8. Result Artifact Pipeline

Create a new experiment suite instead of appending to old results.

Recommended paths:

- `artifacts/results/AgriControl_C_2026-06-30/`
- `artifacts/models/AgriControl_C_2026-06-30/`

The suite must be manifest-first. Required files:

- `suite_manifest.json`: suite id, date, branch, dirty flag, config hashes, algorithms, seeds, training budget, task distribution, scenario definitions, and output paths.
- `runs.csv`: one row per trained run, including method, seed, run name, model path, VecNormalize path, status, training steps, wall time, and notes.
- `eval_tasks.csv`: one row per evaluation task, including split, year, start day, uncertainty scale, economic scenario, and scenario parameters.
- `eval_raw.csv`: one row per run-task evaluation, including return, EPI, revenue, costs, violations, TWB percentage, and trajectory path.
- `method_summary.csv`: aggregation by method, split, and scenario.
- `stat_tests.csv`: paired comparisons and effect sizes.
- `diagnostics.csv`: Agri-MetaRL v2 and Context-RecurrentPPO diagnostics.

Evaluation should be deterministic and use one episode per run-task unless task randomness is explicitly introduced and recorded. Repetition is allowed only when it represents genuinely different random seeds or task stochasticity.

Figure and table generators must read only the new suite outputs. They should fail fast when required algorithms, seeds, tasks, or manifest fields are missing.

Old artifacts remain available for audit/history but must not feed the C-route manuscript.

## 9. Figures and Tables

Recommended figures:

1. Task distribution and architecture overview.
2. Training curves across seeds.
3. Fixed and held-out aggregate performance.
4. Parameter robustness curves over uncertainty scale.
5. Economic scenario performance heatmap or grouped bar chart.
6. Agri-MetaRL v2 ablation and mechanism diagnostics.
7. 60-day climate/control trajectory case study.

Recommended tables:

1. Environment, observation, action, reward, and task-distribution specification.
2. Algorithm hyperparameters.
3. Fixed and held-out summary.
4. Statistical comparisons and effect sizes.

Figures must be regenerated from current suite artifacts, not manually edited or scaled.

## 10. Integrity Tests

Add tests or checks for:

- no algorithm-dependent result scaling constants in figure/table generation;
- no duplicate deterministic rows counted as independent samples;
- all expected algorithms, seeds, and tasks are present before final summary generation;
- summaries match raw data within numerical tolerance;
- stale suite ids are rejected by figure/table generation;
- Agri-MetaRL v2 and Context-RecurrentPPO model load/predict smoke tests;
- generated manuscript tables and figures reference the selected suite only.

These tests are part of the paper-quality workflow, not optional engineering polish.

## 11. Manuscript Rewrite Plan

The manuscript should be rewritten after the new suite pipeline is implemented and real results exist.

Provisional title options:

1. `Task-Distribution-Aware Reinforcement Learning for Robust Greenhouse Climate Control`
2. `Contextual Advantage Calibration for Robust Reinforcement Learning in Greenhouse Climate Control`

Use title 1 by default because it remains valid under more result patterns. Use title 2 only if Agri-MetaRL v2 ablation and robustness results strongly support the method-specific claim.

Recommended manuscript structure:

1. Introduction: motivate robust greenhouse RL under weather, start-date, parameter, and economic shifts.
2. Methods: describe environment, algorithms, Agri-MetaRL v2, Context-RecurrentPPO, and task distribution.
3. Experiments: describe pilot/full matrix, evaluation groups, metrics, and statistics.
4. Results: report actual outcomes without assuming Agri-MetaRL v2 wins everywhere.
5. Discussion: explain mechanism boundaries, robustness trade-offs, and deployment implications.
6. Limitations: simulation-only evaluation, GreenLight assumptions, economic modeling assumptions, compute cost, and missing real-greenhouse validation.

The current manuscript should not be patched sentence-by-sentence first. It should be treated as a source of reusable background text, while methods, experiments, results, and claims are rebuilt around the new protocol.

## 12. Implementation Boundaries

This design document authorizes planning, not implementation. Implementation should begin only after user review and approval of this spec, followed by a written implementation plan.

Implementation should proceed in dependency order:

1. Suite manifest and task/scenario definitions.
2. Training orchestration updates.
3. Evaluation runner and raw artifact writer.
4. Aggregation and statistical checks.
5. Plot/table generation from suite artifacts.
6. Integrity tests.
7. Pilot runs.
8. Full experiment runs if pilot passes.
9. Manuscript rewrite based on real results.

No conclusions should be written as final claims until the redesigned experiments have been run and verified.

