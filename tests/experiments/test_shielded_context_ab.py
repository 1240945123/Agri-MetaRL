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

    class Model:
        num_timesteps = 12

    class Env:
        def close(self):
            self.closed = True

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
