import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from types import SimpleNamespace

from experiments.scripts import run_shielded_context_ab as cli
from gl_gym.experiments.shield_evaluation import write_shield_artifacts_atomic


def _stage1(root: Path) -> dict:
    root.mkdir()
    report = {
        "schema_version": "action-shield-stage1-v1",
        "failure_id": "failure-1",
        "capsule_identity_sha256": "a" * 64,
        "checkpoint_path": str(root.parent / "last_model.zip"),
        "checkpoint_sha256": "b" * 64,
        "source_checksums": {"source": "c" * 64},
        "git_head": "d" * 40,
        "dirty": False,
        "formal_solver_options": dict(cli.FORMAL_CVODES_OPTIONS),
        "env_config_sha256": "e" * 64,
        "rule_config_sha256": "f" * 64,
        "fixed_lambdas": list(cli.DEFAULT_LAMBDAS),
        "selected_lambda": cli.DEFAULT_LAMBDAS[0],
        "conditions": {name: True for name in cli.STAGE1_CONDITIONS},
        "outcome": "continue_to_context_ab",
    }
    (root / "stage1_results.json").write_text(json.dumps(report), encoding="utf-8")
    (root / "decision.json").write_text(
        json.dumps({key: report[key] for key in ("outcome", "conditions", "selected_lambda")}),
        encoding="utf-8",
    )
    np.savez(
        root / "stage1_states.npz",
        x0=np.ones(2, dtype=np.float64),
        selected_final_state=np.ones(2, dtype=np.float64),
        selected_available=np.array(True, dtype=np.bool_),
    )
    return report


def test_load_stage1_rejects_extra_artifact_and_failing_decision(tmp_path: Path):
    root = tmp_path / "stage1"
    report = _stage1(root)
    (root / "extra.txt").write_text("stale", encoding="utf-8")
    with pytest.raises(ValueError, match="exactly three"):
        cli.load_stage1_prerequisite(root)

    (root / "extra.txt").unlink()
    report["conditions"][cli.STAGE1_CONDITIONS[0]] = False
    (root / "stage1_results.json").write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(ValueError, match="conditions"):
        cli.load_stage1_prerequisite(root)


def test_load_stage1_validates_npz_without_pickle_and_returns_identity(tmp_path: Path):
    root = tmp_path / "stage1"
    report = _stage1(root)
    loaded = cli.load_stage1_prerequisite(root)
    assert loaded["report"] == report
    assert len(loaded["stage1_results_sha256"]) == 64
    assert loaded["selected_lambda"] == cli.DEFAULT_LAMBDAS[0]


@pytest.mark.parametrize("field", ["formal_solver_options", "fixed_lambdas"])
def test_stage1_rejects_stale_solver_or_lambda_grid(tmp_path: Path, field: str):
    root = tmp_path / "stage1"
    report = _stage1(root)
    report[field] = {"stale": True} if field == "formal_solver_options" else [0.5]
    (root / "stage1_results.json").write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(ValueError, match="solver|lambda"):
        cli.load_stage1_prerequisite(root)


@pytest.mark.parametrize("stale", ["checkpoint", "source", "rule"])
def test_stage1_provenance_rejects_stale_inputs(tmp_path: Path, stale: str):
    source_manifest = tmp_path / "manifest.json"
    source_tasks = tmp_path / "tasks.csv"
    source_manifest.write_text("{}", encoding="utf-8")
    source_tasks.write_text("task_id\n", encoding="utf-8")
    evaluation = {"git_commit": "d" * 40, "dirty": False}
    sources = {
        str(source_manifest.resolve()): cli.sha256_file(source_manifest),
        str(source_tasks.resolve()): cli.sha256_file(source_tasks),
    }
    for name, path in cli.RELEVANT_SOURCE_FIELDS:
        checksum = cli.sha256_file(cli.ROOT / path)
        evaluation[name] = checksum
        sources[str((cli.ROOT / path).resolve())] = checksum
    report = {
        "checkpoint_path": str(tmp_path / "model123.zip"), "checkpoint_sha256": "b" * 64,
        "rule_config_sha256": "c" * 64, "env_config_sha256": "e" * 64,
        "source_checksums": sources, "git_head": evaluation["git_commit"], "dirty": False,
    }
    if stale == "checkpoint":
        report["checkpoint_sha256"] = "0" * 64
    elif stale == "source":
        report["source_checksums"][str(source_manifest.resolve())] = "0" * 64
    else:
        report["rule_config_sha256"] = "0" * 64
    runs = [
        {"seed": 42, "model_path": tmp_path / "model42.zip", "model_sha256": "a" * 64},
        {"seed": 123, "model_path": tmp_path / "model123.zip", "model_sha256": "b" * 64},
    ]
    with pytest.raises(ValueError, match="checkpoint|source|rule"):
        cli.validate_stage1_provenance(
            {"report": report}, runs=runs, source_manifest=source_manifest,
            source_tasks_csv=source_tasks, evaluation_provenance=evaluation,
            rule_config_sha256="c" * 64, env_config_sha256="e" * 64,
        )


def test_validate_roots_rejects_stage1_and_unshielded_overlap(tmp_path: Path):
    with pytest.raises(ValueError, match="disjoint"):
        cli.validate_output_roots(
            tmp_path / "stage1" / "out",
            tmp_path / "failures",
            protected_roots=[tmp_path / "stage1", tmp_path / "unshielded"],
        )


def test_unshielded_comparator_rejects_partial_progress_with_actionable_error(tmp_path: Path):
    root = tmp_path / "unshielded"
    root.mkdir()
    provenance = {"source_manifest_sha256": "a" * 64}
    (root / "diagnostic_manifest.json").write_text(json.dumps(provenance), encoding="utf-8")
    pd.DataFrame([
        {"seed": 42, "task_id": cli.DIAGNOSTIC_TASK_IDS[0],
         "inference_mode": cli.MODES[0], "episode_return": 1.0}
    ]).to_csv(root / "eval_raw.csv", index=False)
    with pytest.raises(ValueError, match="separate failure-tolerant"):
        cli.load_unshielded_comparator(root, expected_provenance=provenance)
    with pytest.raises(ValueError, match="disjoint"):
        cli.validate_output_roots(
            tmp_path / "out",
            tmp_path / "unshielded" / "failures",
            protected_roots=[tmp_path / "stage1", tmp_path / "unshielded"],
        )


@pytest.mark.parametrize("suffix", ["work", "stage-token", "staging-token", "backup-token"])
@pytest.mark.parametrize("protected_name", ["unshielded", "formal"])
def test_output_roots_reject_protected_lifecycle_siblings(
    tmp_path: Path, suffix: str, protected_name: str
):
    protected = tmp_path / protected_name
    candidate = tmp_path / f".{protected_name}.{suffix}"
    with pytest.raises(ValueError, match="disjoint|lifecycle"):
        cli.validate_output_roots(
            candidate, tmp_path / "failures", protected_roots=[protected]
        )


def test_diagnostics_reject_trace_record_cross_mismatch_before_writes():
    record = {
        "schema_version": "minimal-feasibility-action-shield-v1",
        "intervened": False,
        "requested_action": [0.0, 0.0],
        "reference_action": None,
        "executed_action": [0.0, 0.0],
        "selected_lambda": 0.0,
        "candidate_attempts": [],
        "intervention_l1": 0.0,
        "intervention_l2": 0.0,
        "intervention_linf": 0.0,
        "per_channel_changed": [False, False],
        "extra_solver_attempts": 0,
        "elapsed_seconds": 0.0,
        "original_failure": None,
    }
    base = {
        "action_trace": np.zeros((1, 2), dtype=np.float32),
        "requested_action_trace": np.zeros((1, 2), dtype=np.float32),
        "action_shield_records": [record],
    }
    for field in ("action_trace", "requested_action_trace"):
        malformed = dict(base)
        malformed[field] = np.ones((1, 2), dtype=np.float32)
        with pytest.raises(ValueError, match="record|trace"):
            cli._strict_diagnostics(malformed)


def _explicit_comparator_rows():
    return pd.DataFrame([
        {"seed": seed, "task_id": task, "inference_mode": mode,
         "completed": True, "status": "completed", "ode_failure_count": 0,
         "failure_evidence_path": "", "model_sha256": str(seed)[0] * 64,
         "vecnormalize_sha256": str(seed)[-1] * 64,
         "episode_return": 1.0, "EPI": 1.0, "temp_violation": 1.0,
         "co2_violation": 1.0, "rh_violation": 1.0}
        for seed in cli.APPROVED_SEEDS for task in cli.DIAGNOSTIC_TASK_IDS for mode in cli.MODES
    ])


def test_comparator_requires_explicit_protocol_columns(tmp_path: Path):
    root = tmp_path / "unshielded"
    root.mkdir()
    provenance = {"source_manifest_sha256": "a" * 64}
    (root / "diagnostic_manifest.json").write_text(json.dumps(provenance), encoding="utf-8")
    table = _explicit_comparator_rows().drop(columns="status")
    table.to_csv(root / "eval_raw.csv", index=False)
    with pytest.raises(ValueError, match="protocol columns"):
        cli.load_unshielded_comparator(root, expected_provenance=provenance)


def test_stage2_gate_outcome_mapping_preserves_generic_evidence():
    gate = {"outcome": "fail", "conditions": {"zero_ode_failures": False},
            "evidence": {"shielded_ode_failure_count": 1}, "reasons": ["zero_ode_failures"]}
    decision = cli._stage2_decision(gate)
    assert decision == {
        "outcome": "redesign_action_shield", "stage": "stage2_shielded_context_ab",
        "conditions": gate["conditions"], "evidence": gate["evidence"],
        "reasons": gate["reasons"],
    }


@pytest.mark.parametrize("capsule_case", ["bogus", "mismatch"])
def test_comparator_rejects_bogus_or_mismatched_failure_capsule(
    tmp_path: Path, capsule_case: str
):
    root = tmp_path / "unshielded"
    root.mkdir()
    provenance = {
        "source_manifest_sha256": "a" * 64, "source_tasks_sha256": "b" * 64,
        "git_commit": "c" * 40, "dirty": False,
    }
    (root / "diagnostic_manifest.json").write_text(json.dumps(provenance), encoding="utf-8")
    table = _explicit_comparator_rows()
    table.loc[0, ["completed", "status", "ode_failure_count"]] = [False, "ode_failure", 1]
    table.loc[0, list(cli.REQUIRED_METRICS)] = np.nan
    capsule_dir = root / ("d" * 64)
    capsule_dir.mkdir()
    evidence = capsule_dir / "manifest.json"
    evidence.write_text("{}", encoding="utf-8")
    table.loc[0, "failure_evidence_path"] = str(evidence)
    table.to_csv(root / "eval_raw.csv", index=False)
    expected = {
        seed: {"model_sha256": str(seed)[0] * 64, "vecnormalize_sha256": str(seed)[-1] * 64,
               "model_path": str(tmp_path / f"model{seed}.zip")}
        for seed in cli.APPROVED_SEEDS
    }
    if capsule_case == "bogus":
        loader = lambda path: (_ for _ in ()).throw(ValueError("bad capsule"))
        match = "valid failure capsule"
    else:
        loader = lambda path: SimpleNamespace(manifest={"context": {}})
        match = "identity/provenance"
    with pytest.raises(ValueError, match=match):
        cli.load_unshielded_comparator(
            root, expected_provenance=provenance,
            expected_checkpoints=expected, capsule_loader=loader,
        )


def test_atomic_writer_copies_evidence_and_preserves_old_root_on_copy_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    frame = pd.DataFrame([{"seed": 42, "task_id": "t", "inference_mode": "m", "value": 1.0}])
    evidence = tmp_path / "trace.npy"
    np.save(evidence, np.ones((1, 2)), allow_pickle=False)
    root = tmp_path / "result"
    root.mkdir()
    (root / "old.txt").write_text("old", encoding="utf-8")

    real_copy = cli.shutil.copy2
    monkeypatch.setattr(cli.shutil, "copy2", lambda *args, **kwargs: (_ for _ in ()).throw(OSError("copy failed")))
    with pytest.raises(OSError, match="copy failed"):
        write_shield_artifacts_atomic(
            frame, frame, frame, {"ok": True}, {"ok": True}, root,
            evidence_files={"traces/trace.npy": evidence},
        )
    assert (root / "old.txt").read_text(encoding="utf-8") == "old"
    monkeypatch.setattr(cli.shutil, "copy2", real_copy)

    paths = write_shield_artifacts_atomic(
        frame, frame, frame, {"ok": True}, {"ok": True}, root,
        evidence_files={"traces/trace.npy": evidence},
    )
    assert (root / "traces" / "trace.npy").is_file()
    assert paths["evidence"] == root


def test_injectable_runner_executes_exact_32_with_shield_params_and_publishes_relative_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    source = tmp_path / "source"
    source.mkdir()
    manifest_path = source / "manifest.json"
    tasks_path = source / "tasks.csv"
    manifest_path.write_text("{}", encoding="utf-8")
    tasks_path.write_text("task_id\n", encoding="utf-8")
    stage = {"root": tmp_path / "stage1", "report": {"capsule_identity_sha256": "4" * 64}, "selected_lambda": 0.0625,
             "stage1_results_sha256": "1" * 64, "stage1_states_sha256": "2" * 64,
             "stage1_decision_sha256": "3" * 64}
    stage["root"].mkdir()
    monkeypatch.setattr(cli, "load_stage1_prerequisite", lambda root: stage)
    monkeypatch.setattr(cli, "validate_stage1_provenance", lambda *args, **kwargs: None)
    provenance = {"git_commit": "a" * 40, "dirty": False}
    evaluation = cli._evaluation_provenance(manifest_path, tasks_path, provenance)
    unshielded = pd.DataFrame(
        [{"seed": seed, "task_id": task, "inference_mode": mode, "completed": True,
          "ode_failure_count": 0, "episode_return": 100.0, "EPI": 1.0,
          "temp_violation": 1.0, "co2_violation": 1.0, "rh_violation": 1.0}
         for seed in cli.APPROVED_SEEDS for task in cli.DIAGNOSTIC_TASK_IDS for mode in cli.MODES]
    )
    monkeypatch.setattr(cli, "load_unshielded_comparator", lambda *args, **kwargs: (unshielded, {}))
    unshielded_root = tmp_path / "unshielded"
    unshielded_root.mkdir()
    (unshielded_root / "diagnostic_manifest.json").write_text("{}", encoding="utf-8")
    suite = SimpleNamespace(suite_id="suite", result_root=str(source), env_id="Fake")
    tasks = pd.DataFrame([
        {"suite_id": "suite", "task_id": task, "split": task.split("_", 1)[0],
         "weather_year": 2010, "start_day": 59, "uncertainty_scale": 0.0,
         "economic_scenario": "standard", "climate_constraint_scenario": "standard"}
        for task in cli.DIAGNOSTIC_TASK_IDS
    ])
    runs = [
        {"seed": seed, "model_path": tmp_path / f"model{seed}.zip",
         "vecnormalize_path": tmp_path / f"vec{seed}.pkl",
         "model_sha256": str(seed)[0] * 64, "vecnormalize_sha256": str(seed)[-1] * 64}
        for seed in cli.APPROVED_SEEDS
    ]
    calls = []
    closed = []

    class Model:
        num_timesteps = 12

    class Env:
        def close(self):
            self.closed = True
            closed.append(self)

    def env_loader(suite, task, path, *, shield_params):
        assert shield_params["lamps_on"] == 0
        calls.append((task.task_id, path))
        return Env()

    record = {
        "schema_version": "minimal-feasibility-action-shield-v1", "intervened": False,
        "requested_action": [0.0, 0.0], "reference_action": None,
        "executed_action": [0.0, 0.0], "selected_lambda": 0.0,
        "candidate_attempts": [], "intervention_l1": 0.0, "intervention_l2": 0.0,
        "intervention_linf": 0.0, "per_channel_changed": [False, False],
        "extra_solver_attempts": 0, "elapsed_seconds": 0.0, "original_failure": None,
    }

    def episode_runner(model, env, *, inference_mode, return_diagnostics, failure_recorder):
        assert return_diagnostics and failure_recorder is not None
        return ({"episode_return": 100.0, "EPI": 1.0, "temp_violation": 1.0,
                 "co2_violation": 1.0, "rh_violation": 1.0},
                {"action_trace": np.zeros((1, 2)), "requested_action_trace": np.zeros((1, 2)),
                 "action_shield_records": [dict(record)]})

    attempt = {"count": 0}

    def fail_on_32nd(*args, **kwargs):
        attempt["count"] += 1
        if attempt["count"] == 32:
            raise RuntimeError("episode 32 failed")
        return episode_runner(*args, **kwargs)

    with pytest.raises(RuntimeError, match="episode 32"):
        cli.run_shielded_diagnostic(
            suite=suite, tasks=tasks, runs=runs, source_manifest=manifest_path,
            source_tasks_csv=tasks_path, stage1_root=stage["root"],
            unshielded_result_root=unshielded_root, result_root=tmp_path / "incomplete",
            failure_root=tmp_path / "incomplete-failures", device="cpu", resume=False,
            model_loader=lambda path, device: Model(), env_loader=env_loader,
            episode_runner=fail_on_32nd, provenance_loader=lambda: provenance,
            recorder_factory=lambda root, context: object(),
        )
    assert len(pd.read_csv(tmp_path / ".incomplete.work" / "progress.csv")) == 31
    assert not (tmp_path / "incomplete").exists()
    assert len(closed) == 32
    calls.clear()
    closed.clear()

    result = cli.run_shielded_diagnostic(
        suite=suite, tasks=tasks, runs=runs, source_manifest=manifest_path,
        source_tasks_csv=tasks_path, stage1_root=stage["root"],
        unshielded_result_root=unshielded_root, result_root=tmp_path / "shielded",
        failure_root=tmp_path / "failures", device="cpu", resume=False,
        model_loader=lambda path, device: Model(), env_loader=env_loader,
        episode_runner=episode_runner, provenance_loader=lambda: provenance,
        recorder_factory=lambda root, context: object(),
    )
    assert len(calls) == len(result) == 32
    assert result["method"].eq(cli.METHOD).all()
    assert result["executed_action_trace_path"].str.startswith("traces/").all()
    assert len(list((tmp_path / "shielded" / "traces").glob("*.npy"))) == 64
    assert len(list((tmp_path / "shielded" / "intervention_records").glob("*.json"))) == 32
    decision = json.loads((tmp_path / "shielded" / "decision.json").read_text(encoding="utf-8"))
    assert decision["outcome"] == "continue_to_full_suite"

    calls.clear()
    resumed = cli.run_shielded_diagnostic(
        suite=suite, tasks=tasks, runs=runs, source_manifest=manifest_path,
        source_tasks_csv=tasks_path, stage1_root=stage["root"],
        unshielded_result_root=unshielded_root, result_root=tmp_path / "shielded",
        failure_root=tmp_path / "failures", device="cpu", resume=True,
        model_loader=lambda path, device: Model(), env_loader=env_loader,
        episode_runner=episode_runner, provenance_loader=lambda: provenance,
        recorder_factory=lambda root, context: object(),
    )
    assert len(resumed) == 32
    assert calls == []

    progress_path = tmp_path / ".shielded.work" / "progress.csv"
    stale = pd.read_csv(progress_path)
    stale.loc[0, "shield_fingerprint"] = "0" * 64
    stale.to_csv(progress_path, index=False)
    calls.clear()
    cli.run_shielded_diagnostic(
        suite=suite, tasks=tasks, runs=runs, source_manifest=manifest_path,
        source_tasks_csv=tasks_path, stage1_root=stage["root"],
        unshielded_result_root=unshielded_root, result_root=tmp_path / "shielded",
        failure_root=tmp_path / "failures", device="cpu", resume=True,
        model_loader=lambda path, device: Model(), env_loader=env_loader,
        episode_runner=episode_runner, provenance_loader=lambda: provenance,
        recorder_factory=lambda root, context: object(),
    )
    assert len(calls) == 1

    refreshed = pd.read_csv(progress_path)
    Path(refreshed.loc[0, "requested_action_trace_path"]).write_bytes(b"corrupt")
    calls.clear()
    cli.run_shielded_diagnostic(
        suite=suite, tasks=tasks, runs=runs, source_manifest=manifest_path,
        source_tasks_csv=tasks_path, stage1_root=stage["root"],
        unshielded_result_root=unshielded_root, result_root=tmp_path / "shielded",
        failure_root=tmp_path / "failures", device="cpu", resume=True,
        model_loader=lambda path, device: Model(), env_loader=env_loader,
        episode_runner=episode_runner, provenance_loader=lambda: provenance,
        recorder_factory=lambda root, context: object(),
    )
    assert len(calls) == 1
