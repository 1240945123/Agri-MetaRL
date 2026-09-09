# Experiment scripts

Run all commands from the repository root after installing the package in editable mode, or with `PYTHONPATH=src`.

## Main workflows

Train the paper models:

```bash
python experiments/scripts/train_paper_experiments.py --device cpu
```

Run the full evaluation and reporting pipeline with existing models:

```bash
python experiments/scripts/run_paper_pipeline_after_train.py --skip-train
```

Train and evaluate ablation variants:

```bash
python experiments/scripts/train_ablation_variants.py --device cpu
python experiments/scripts/run_ablation.py
```

Run the fixed evaluation protocol:

```bash
python experiments/scripts/run_all_fixed_protocol.py
```

Generate legacy comparison plots from existing result CSVs:

```bash
python experiments/scripts/generate_basic_plots.py
```

## Local inputs and outputs

- Environment configuration: `configs/envs/`
- Weather data: `datasets/weather/`
- Trained models: `artifacts/models/`
- Evaluation results: `artifacts/results/`
- Generated figures: `artifacts/figures/`
- W&B tracking data: `artifacts/tracking/`

The `artifacts/` tree is intentionally ignored by Git. Training can take hours; use each script's `--help` output to inspect options before launching a long run.

## C-route robust suite workflow

Use `artifacts/results/AgriControl_C_2026-06-30/` as the result source for the redesigned C-route manuscript. Do not use `artifacts/results/AgriControl/` as an active result source; it is archival and contains stale or incomplete experiment outputs.

Create the robust suite manifest and task table:

```powershell
python experiments\scripts\create_experiment_suite.py --suite_id AgriControl_C_2026-06-30
```

Run a pilot training pass first with a smaller budget so stability issues are found before expensive full runs:

```powershell
python experiments\scripts\run_suite_training.py --manifest artifacts\results\AgriControl_C_2026-06-30\suite_manifest.json --algorithms ppo recurrentppo context_recurrentppo agri_metarl --seeds 42 123 --train_timesteps 300000 --device cpu
```

For a registry-only smoke check, add `--dry_run`. After confirming the manifest, task table, pilot stability, and registry expectations, run the full training job. This can take hours and must complete before evaluation can produce manuscript-ready results:

```powershell
python experiments\scripts\run_suite_training.py --manifest artifacts\results\AgriControl_C_2026-06-30\suite_manifest.json --device cpu
```

Evaluate completed training runs, then summarize, validate, and generate figures:

```powershell
python experiments\scripts\evaluate_suite.py --manifest artifacts\results\AgriControl_C_2026-06-30\suite_manifest.json --runs_csv artifacts\results\AgriControl_C_2026-06-30\runs.csv --tasks_csv artifacts\results\AgriControl_C_2026-06-30\eval_tasks.csv
python experiments\scripts\summarize_suite.py --manifest artifacts\results\AgriControl_C_2026-06-30\suite_manifest.json
python experiments\scripts\validate_suite_artifacts.py --manifest artifacts\results\AgriControl_C_2026-06-30\suite_manifest.json
python experiments\scripts\generate_suite_figures.py --manifest artifacts\results\AgriControl_C_2026-06-30\suite_manifest.json
```

Run `summarize_suite.py`, `validate_suite_artifacts.py`, and `generate_suite_figures.py` again before updating manuscript tables or figures so the paper reflects the latest completed full-run artifacts.

The older fixed-protocol scripts, including `run_all_fixed_protocol.py`, `evaluate_fixed_protocol.py`, and legacy plot generation from fixed CSVs, are retained for archival comparisons only. Do not use their summaries as manuscript evidence for the C-route paper.
