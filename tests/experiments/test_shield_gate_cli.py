import importlib.util
from pathlib import Path

import pandas as pd
import pytest
from types import SimpleNamespace

from gl_gym.experiments.suite_schema import create_default_suite_config
from gl_gym.experiments.suite_tasks import build_evaluation_tasks


def _module():
    path = Path(__file__).resolve().parents[2] / "experiments/scripts/evaluate_shield_gate.py"
    spec = importlib.util.spec_from_file_location("evaluate_shield_gate_cli", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_canonical_tasks_are_descriptor_exact_not_count_only(tmp_path):
    cli = _module()
    suite = create_default_suite_config(result_root=tmp_path / "results")
    tasks = pd.DataFrame(vars(item) for item in build_evaluation_tasks(suite))
    assert len(tasks) == 91
    cli.validate_canonical_tasks(suite, tasks)
    tasks.loc[0, "weather_year"] += 1
    with pytest.raises(ValueError, match="canonical"):
        cli.validate_canonical_tasks(suite, tasks)


def test_gate_parser_has_all_required_inputs():
    cli = _module()
    actions = {action.dest: action.required for action in cli.build_parser()._actions}
    for name in (
        "manifest", "tasks_csv", "unshielded_eval", "shielded_eval",
        "interventions", "stage2_decision", "output_root",
    ):
        assert actions[name] is True


def test_exact_key_validation_rejects_duplicate_extra_and_missing(tmp_path):
    cli = _module()
    base = pd.DataFrame(
        [{
            "suite_id": "s", "algorithm": cli.BASE_ALGORITHM, "seed": 42,
            "task_id": "t", "split": "fixed", "weather_year": 2010,
            "start_day": 59, "uncertainty_scale": 0.0,
            "economic_scenario": "standard", "climate_constraint_scenario": "standard",
        }]
    )
    cli._strict_keys(base, label="sample", expected={(42, "t")})
    with pytest.raises(ValueError, match="duplicate"):
        cli._strict_keys(pd.concat([base, base]), label="sample", expected={(42, "t")})
    with pytest.raises(ValueError, match="exactly"):
        cli._strict_keys(base, label="sample", expected={(42, "other")})


def _gate_case(cli, tmp_path: Path, monkeypatch, *, intervention_count=0, shield_return=100.0,
               shield_violation=1.0, ode_failures=0):
    suite = create_default_suite_config(result_root=tmp_path / "suite")
    tasks = pd.DataFrame(vars(item) for item in build_evaluation_tasks(suite))
    seeds = (42, 123)
    descriptors = [
        {**row._asdict(), "seed": seed, "suite_id": suite.suite_id}
        for seed in seeds for row in tasks.itertuples(index=False)
    ]
    shield = pd.DataFrame([
        {**row, "algorithm": cli.SHIELD_ALGORITHM, "episode_return": shield_return,
         "temp_violation": shield_violation, "co2_violation": shield_violation,
         "rh_violation": shield_violation}
        for row in descriptors
    ])
    unshield = pd.DataFrame([
        {**row, "algorithm": cli.BASE_ALGORITHM, "episode_return": 100.0,
         "temp_violation": 1.0, "co2_violation": 1.0, "rh_violation": 1.0,
         "completed": True, "ode_failure_count": 0}
        for row in descriptors
    ])
    interventions = pd.DataFrame([
        {**row, "algorithm": cli.SHIELD_ALGORITHM, "method": cli.SHIELD_METHOD,
         "completed": True, "ode_failure_count": ode_failures,
         "total_steps": 1000, "intervention_count": intervention_count}
        for row in descriptors
    ])
    paths = {}
    for name, frame in (("shielded_eval", shield), ("unshielded_eval", unshield), ("interventions", interventions)):
        paths[name] = tmp_path / f"{name}.csv"; frame.to_csv(paths[name], index=False)
    for name in ("manifest", "tasks_csv", "stage2_decision"):
        paths[name] = tmp_path / f"{name}.json"; paths[name].write_text("{}", encoding="utf-8")
    (tmp_path / "shield_manifest.json").write_text("{}", encoding="utf-8")
    output = tmp_path / "stage3"
    stage2 = {"checkpoints": [
        {"seed": 42, "model_sha256": "a" * 64, "vecnormalize_sha256": "b" * 64},
        {"seed": 123, "model_sha256": "c" * 64, "vecnormalize_sha256": "d" * 64},
    ]}
    args = SimpleNamespace(**{name: str(path) for name, path in paths.items()}, output_root=str(output))
    monkeypatch.setattr(cli, "_prerequisites", lambda args: (suite, tasks, stage2, {}, output))
    monkeypatch.setattr(cli, "_validate_intervention_evidence", lambda *a, **k: None)
    return cli.run(args), output


def test_full_91_by_2_gate_ready_and_writes_only_stage3_artifacts(tmp_path, monkeypatch):
    cli = _module()
    decision, output = _gate_case(cli, tmp_path, monkeypatch)
    assert decision["outcome"] == "paper_evidence_ready"
    assert {path.name for path in output.iterdir()} == {
        "paired_deltas.csv", "summary.csv", "shield_manifest.json", "decision.json"
    }
    assert len(pd.read_csv(output / "paired_deltas.csv")) == 182


@pytest.mark.parametrize(
    "kwargs",
    [
        {"ode_failures": 1},
        {"intervention_count": 6},
        {"shield_return": 97.999999},
        {"shield_violation": 1.0500001},
    ],
)
def test_just_over_each_gate_condition_redesigns(tmp_path, monkeypatch, kwargs):
    cli = _module()
    decision, _ = _gate_case(cli, tmp_path, monkeypatch, **kwargs)
    assert decision["outcome"] == "redesign_before_claim"
    assert decision["reasons"]


def test_atomic_publication_failure_restores_prior_root(tmp_path, monkeypatch):
    cli = _module()
    output = tmp_path / "result"
    output.mkdir()
    (output / "old.txt").write_text("preserve", encoding="utf-8")
    real_replace = cli.os.replace

    def fail_stage(source, destination):
        if Path(destination) == output and ".stage-" in Path(source).name:
            raise OSError("publish failed")
        return real_replace(source, destination)

    monkeypatch.setattr(cli.os, "replace", fail_stage)
    frame = pd.DataFrame([{"seed": 42, "task_id": "t", "value": 1.0}])
    with pytest.raises(OSError, match="publish failed"):
        cli._publish(output, frame, frame, {"ok": True}, {"ok": True})
    assert (output / "old.txt").read_text(encoding="utf-8") == "preserve"
    assert {path.name for path in output.iterdir()} == {"old.txt"}
