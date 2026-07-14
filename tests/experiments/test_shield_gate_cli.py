import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from types import SimpleNamespace

from gl_gym.experiments.suite_schema import create_default_suite_config, write_suite_manifest
from gl_gym.experiments.suite_tasks import build_evaluation_tasks
from gl_gym.experiments.shield_evaluation import aggregate_episode_interventions
from tests.experiments.test_suite_evaluation_cli import _stage2_fixture
from experiments.scripts import evaluate_suite as evaluator
from experiments.scripts import run_shielded_context_ab as stage2_source


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
               shield_violation=1.0, ode_failures=0, tamper=None):
    suite = create_default_suite_config(result_root=tmp_path / "suite-source")
    tasks = pd.DataFrame(vars(item) for item in build_evaluation_tasks(suite))
    manifest_path = write_suite_manifest(suite, tmp_path / "suite_manifest.json")
    tasks_path = tmp_path / "tasks.csv"; tasks.to_csv(tasks_path, index=False)
    stage2_decision = _stage2_fixture(evaluator, tmp_path / "stage2")
    stage2_manifest_path = stage2_decision.parent / "shield_manifest.json"
    stage2_manifest = json.loads(stage2_manifest_path.read_text(encoding="utf-8"))
    _, rule_sha = stage2_source._load_rule_params()
    stage2_manifest.update({
        "source_manifest_sha256": evaluator._sha(manifest_path),
        "source_tasks_sha256": evaluator._sha(tasks_path),
        "rule_config_sha256": rule_sha,
        "env_config_sha256": evaluator._sha(stage2_source.ENV_CONFIG_PATH),
        "runtime_source_tree_sha256": stage2_source._runtime_source_tree_sha256(),
        "evaluator_source_sha256": evaluator._sha(Path(evaluator.__file__)),
        "gate_source_sha256": evaluator._sha(Path(cli.__file__)),
        "fixed_lambdas": list(stage2_source.DEFAULT_LAMBDAS),
        "formal_solver_options": dict(stage2_source.FORMAL_CVODES_OPTIONS),
        **stage2_source._behavior_source_hashes(),
    })
    model_paths, vec_paths = {}, {}
    for seed in (42, 123):
        model_paths[seed] = tmp_path / f"model-{seed}.zip"; model_paths[seed].write_bytes(f"model{seed}".encode())
        vec_paths[seed] = tmp_path / f"vec-{seed}.pkl"; vec_paths[seed].write_bytes(f"vec{seed}".encode())
    stage2_manifest["checkpoints"] = [
        {"seed": seed, "model_sha256": evaluator._sha(model_paths[seed]),
         "vecnormalize_sha256": evaluator._sha(vec_paths[seed]), "checkpoint_steps": 10}
        for seed in (42, 123)
    ]
    stage2_manifest_path.write_text(json.dumps(stage2_manifest), encoding="utf-8")
    stage2_identity = evaluator.load_stage2_evidence(stage2_decision)["stage2_identity_sha256"]
    seeds = (42, 123)
    checkpoint_map = {int(item["seed"]): item for item in stage2_manifest["checkpoints"]}
    source_inputs = {
        "runtime_source_tree_sha256": stage2_source._runtime_source_tree_sha256(),
        "evaluator_source_sha256": evaluator._sha(Path(evaluator.__file__)),
        "gate_source_sha256": evaluator._sha(Path(cli.__file__)),
    }
    source_fingerprint = evaluator._canonical_hash(source_inputs)
    descriptors = [
        {**row._asdict(), "seed": seed, "suite_id": suite.suite_id}
        for seed in seeds for row in tasks.itertuples(index=False)
    ]
    runs = pd.DataFrame([{
        "suite_id": suite.suite_id, "algorithm": "agri_metarl", "seed": seed,
        "run_name": f"agri_metarl_seed{seed}", "model_path": str(model_paths[seed]),
        "vecnormalize_path": str(vec_paths[seed]), "status": "completed",
    } for seed in (42, 123)])
    runs_path = tmp_path / "runs.csv"; runs.to_csv(runs_path, index=False)
    unshield_root = tmp_path / "unshielded-full"
    unshield_args = SimpleNamespace(
        manifest=str(manifest_path), runs_csv=str(runs_path), tasks_csv=str(tasks_path),
        algorithms=["agri_metarl"], seeds=[42, 123], splits=None, task_ids=None, limit_tasks=None,
        resume_eval=False, action_shield=False, formal_unshielded_provenance=True,
        stage2_decision=str(stage2_decision), result_root=str(unshield_root), interventions_out=None,
    )
    class Model:
        num_timesteps = 10
    class Loader:
        @staticmethod
        def load(path, device): return Model()
    class Env:
        def close(self): pass
    evaluator.run(
        unshield_args, model_map={"agri_metarl": Loader},
        env_loader=lambda *args, **kwargs: Env(),
        episode_runner=lambda model, env: {"episode_return": 100.0, "temp_violation": 1.0,
                                           "co2_violation": 1.0, "rh_violation": 1.0},
    )
    unshield_path = unshield_root / "eval_raw.csv"
    work = tmp_path / "shield-work"; work.mkdir()
    shield_rows, intervention_rows = [], []
    base_record = {
        "schema_version": "minimal-feasibility-action-shield-v1", "intervened": False,
        "requested_action": [0.0], "reference_action": None, "executed_action": [0.0],
        "selected_lambda": 0.0, "candidate_attempts": [], "intervention_l1": 0.0,
        "intervention_l2": 0.0, "intervention_linf": 0.0,
        "per_channel_changed": [False], "extra_solver_attempts": 0,
        "elapsed_seconds": 0.0, "original_failure": None,
    }
    for row in descriptors:
        step_count = 1000 if intervention_count else 1
        token = f"{row['seed']}__{row['task_id']}"
        executed_path = work / f"{token}__executed.npy"; requested_path = work / f"{token}__requested.npy"
        records_path = work / f"{token}__records.json"
        executed_values = np.zeros((step_count, 1))
        executed_values[:intervention_count] = stage2_source.DEFAULT_LAMBDAS[0]
        np.save(executed_path, executed_values, allow_pickle=False)
        np.save(requested_path, np.zeros((step_count, 1)), allow_pickle=False)
        records = []
        for index in range(step_count):
            if index < intervention_count:
                selected_lambda = stage2_source.DEFAULT_LAMBDAS[0]
                records.append({
                    **base_record, "step_index": index, "intervened": True,
                    "reference_action": [1.0], "executed_action": [selected_lambda], "selected_lambda": selected_lambda,
                    "candidate_attempts": [{"lambda": selected_lambda, "action": [selected_lambda], "success": True,
                                             "elapsed_seconds": 0.0, "exception_type": None,
                                             "exception_message": None}],
                    "intervention_l1": selected_lambda, "intervention_l2": selected_lambda,
                    "intervention_linf": selected_lambda, "per_channel_changed": [True],
                    "extra_solver_attempts": 1,
                    "original_failure": {"exception_type": "RuntimeError", "exception_message": "failure"},
                })
            else:
                records.append(dict(base_record, step_index=index))
        records_path.write_text(json.dumps(records), encoding="utf-8")
        hashes = {
            "executed_action_trace_sha256": evaluator._sha(executed_path),
            "requested_action_trace_sha256": evaluator._sha(requested_path),
            "intervention_records_sha256": evaluator._sha(records_path),
        }
        common = {
            **row, "algorithm": cli.SHIELD_ALGORITHM, "method": cli.SHIELD_METHOD,
            "model_sha256": checkpoint_map[row["seed"]]["model_sha256"],
            "vecnormalize_sha256": checkpoint_map[row["seed"]]["vecnormalize_sha256"],
            "checkpoint_steps": 10, "source_fingerprint_sha256": source_fingerprint,
            "runtime_source_tree_sha256": stage2_source._runtime_source_tree_sha256(),
            "stage2_identity_sha256": stage2_identity,
            "completed": ode_failures == 0 or bool(shield_rows),
            "formal_complete": True, "ode_failure_count": 0,
            "episode_return": shield_return, "temp_violation": shield_violation,
            "co2_violation": shield_violation, "rh_violation": shield_violation,
            "executed_action_trace_path": str(executed_path),
            "requested_action_trace_path": str(requested_path),
            "intervention_records_path": str(records_path), **hashes,
        }
        identity_payload = {name: common[name] for name in (
            "algorithm", "method", "model_sha256", "vecnormalize_sha256", "checkpoint_steps",
            "source_fingerprint_sha256", "stage2_identity_sha256", "suite_id", "seed", "task_id",
            "runtime_source_tree_sha256",
            "split", "weather_year", "start_day", "uncertainty_scale", "economic_scenario",
            "climate_constraint_scenario", "episode_return", "temp_violation", "co2_violation",
            "rh_violation", "executed_action_trace_sha256", "requested_action_trace_sha256",
            "intervention_records_sha256",
        )}
        common["episode_evidence_identity_sha256"] = evaluator._canonical_hash(identity_payload)
        summary = aggregate_episode_interventions(records, 1)
        shield_rows.append(common); intervention_rows.append({**common, **summary})
    shield_root = tmp_path / "shielded-full"
    evaluator._publish_shield_final(
        shield_root, work, pd.DataFrame(shield_rows), pd.DataFrame(intervention_rows),
        manifest_base={
            "suite_id": suite.suite_id, "method": cli.SHIELD_METHOD, "formal_complete": True,
            "checkpoints": [{**item, "checkpoint_steps": 10} for item in stage2_manifest["checkpoints"]],
            "stage2_identity_sha256": stage2_identity, "source_fingerprint_sha256": source_fingerprint,
            "runtime_source_tree_sha256": stage2_source._runtime_source_tree_sha256(),
            "evaluator_source_sha256": evaluator._sha(Path(evaluator.__file__)),
            "gate_source_sha256": evaluator._sha(Path(cli.__file__)),
            "source_fingerprint_inputs": source_inputs,
        },
    )
    paths = {
        "manifest": manifest_path, "tasks_csv": tasks_path, "stage2_decision": stage2_decision,
        "shielded_eval": shield_root / "eval_raw.csv", "interventions": shield_root / "interventions.csv",
        "unshielded_eval": unshield_path,
    }
    output = tmp_path / "stage3"
    args = SimpleNamespace(**{name: str(path) for name, path in paths.items()}, output_root=str(output))
    if tamper is not None:
        tamper(paths)
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


def test_metric_tamper_after_manifest_is_rejected(tmp_path, monkeypatch):
    cli = _module()
    def tamper(paths):
        frame = pd.read_csv(paths["shielded_eval"]); frame.loc[0, "episode_return"] += 1
        frame.to_csv(paths["shielded_eval"], index=False)
    with pytest.raises(ValueError, match="hash"):
        _gate_case(cli, tmp_path, monkeypatch, tamper=tamper)


def test_wrong_checkpoint_row_rejected_even_with_updated_file_hash(tmp_path, monkeypatch):
    cli = _module()
    def tamper(paths):
        frame = pd.read_csv(paths["shielded_eval"]); frame.loc[0, "model_sha256"] = "0" * 64
        frame.to_csv(paths["shielded_eval"], index=False)
        manifest_path = paths["shielded_eval"].parent / "evaluation_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["eval_raw_sha256"] = evaluator._sha(paths["shielded_eval"])
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="checkpoint"):
        _gate_case(cli, tmp_path, monkeypatch, tamper=tamper)


@pytest.mark.parametrize("field", ["model_sha256", "checkpoint_steps", "source_fingerprint_sha256"])
def test_unshielded_self_declared_provenance_rejected(tmp_path, monkeypatch, field):
    cli = _module()
    def tamper(paths):
        frame = pd.read_csv(paths["unshielded_eval"])
        frame.loc[0, field] = "0" * 64 if field != "checkpoint_steps" else 999
        frame.to_csv(paths["unshielded_eval"], index=False)
        manifest_path = paths["unshielded_eval"].parent / "evaluation_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["eval_raw_sha256"] = evaluator._sha(paths["unshielded_eval"])
        if field == "source_fingerprint_sha256":
            manifest[field] = "0" * 64
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="checkpoint|source"):
        _gate_case(cli, tmp_path, monkeypatch, tamper=tamper)


def test_atomic_double_restore_failure_preserves_sole_backup_and_notes(tmp_path, monkeypatch):
    cli = _module()
    import gl_gym.experiments.shield_evaluation as shield_evaluation
    output = tmp_path / "result"
    output.mkdir(); (output / "old.txt").write_text("preserve", encoding="utf-8")
    real_replace = cli.os.replace

    def fail_publish_and_restore(source, destination):
        source, destination = Path(source), Path(destination)
        if destination == output and (".stage-" in source.name or ".backup-" in source.name):
            raise OSError("rename failed")
        return real_replace(source, destination)

    monkeypatch.setattr(cli.os, "replace", fail_publish_and_restore)
    monkeypatch.setattr(shield_evaluation.shutil, "copytree", lambda *a, **k: (_ for _ in ()).throw(OSError("copy failed")))
    frame = pd.DataFrame([{"seed": 42, "task_id": "t", "value": 1.0}])
    with pytest.raises(OSError, match="rename failed") as captured:
        cli._publish(output, frame, frame, {"ok": True}, {"ok": True})
    assert any("restoration failed" in note for note in captured.value.__notes__)
    backups = list(tmp_path.glob(".result.backup-*"))
    assert len(backups) == 1
    assert (backups[0] / "old.txt").read_text(encoding="utf-8") == "preserve"
