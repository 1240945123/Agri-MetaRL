from pathlib import Path
import importlib.util
from types import SimpleNamespace

import pytest
import json
import pandas as pd
import numpy as np

from gl_gym.experiments.shield_evaluation import (
    aggregate_episode_interventions, build_paired_shield_deltas, evaluate_shield_gate,
)
from gl_gym.experiments.suite_schema import create_default_suite_config, write_suite_manifest
from gl_gym.experiments.suite_tasks import build_evaluation_tasks


def _module():
    path = Path(__file__).resolve().parents[2] / "experiments/scripts/evaluate_suite.py"
    spec = importlib.util.spec_from_file_location("evaluate_suite_cli", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_parser_keeps_legacy_arguments_and_adds_opt_in_shield():
    cli = _module()
    args = cli.build_parser().parse_args(
        ["--manifest", "m", "--runs_csv", "r", "--tasks_csv", "t"]
    )
    assert args.action_shield is False
    assert args.stage2_decision is None
    assert args.result_root is None
    assert args.interventions_out is None


def test_shield_arguments_are_all_or_nothing():
    cli = _module()
    parser = cli.build_parser()
    args = parser.parse_args(
        ["--manifest", "m", "--runs_csv", "r", "--tasks_csv", "t", "--action_shield"]
    )
    with pytest.raises(ValueError, match="stage2_decision.*result_root"):
        cli.validate_cli_mode(args)


def test_environment_close_error_is_not_allowed_to_replace_episode_error():
    cli = _module()

    class Env:
        def close(self):
            raise RuntimeError("close failed")

    primary = RuntimeError("episode failed")
    cli.close_environment(Env(), primary)
    assert any("close failed" in note for note in primary.__notes__)

    with pytest.raises(RuntimeError, match="close failed"):
        cli.close_environment(Env(), None)


def _stage2_fixture(cli, root: Path) -> Path:
    root.mkdir()
    unshielded_root = root.parent / f"{root.name}-unshielded"
    unshielded_root.mkdir()
    from experiments.scripts.run_context_ab import APPROVED_SEEDS
    from gl_gym.experiments.context_ab import DIAGNOSTIC_TASK_IDS, MODES
    keys = [(seed, task, mode) for seed in APPROVED_SEEDS for task in DIAGNOSTIC_TASK_IDS for mode in MODES]
    trace_dir = root / "traces"; trace_dir.mkdir()
    records_dir = root / "intervention_records"; records_dir.mkdir()
    record = {
        "step_index": 0, "schema_version": "minimal-feasibility-action-shield-v1",
        "intervened": False, "requested_action": [0.0], "reference_action": None,
        "executed_action": [0.0], "selected_lambda": 0.0, "candidate_attempts": [],
        "intervention_l1": 0.0, "intervention_l2": 0.0, "intervention_linf": 0.0,
        "per_channel_changed": [False], "extra_solver_attempts": 0,
        "elapsed_seconds": 0.0, "original_failure": None,
    }
    common = []
    for seed, task, mode in keys:
        token = f"seed{seed}__{task}__{mode}"
        executed = trace_dir / f"{token}__executed.npy"
        requested = trace_dir / f"{token}__requested.npy"
        records = records_dir / f"{token}.json"
        np.save(executed, np.zeros((1, 1), dtype=np.float32), allow_pickle=False)
        np.save(requested, np.zeros((1, 1), dtype=np.float32), allow_pickle=False)
        records.write_text(json.dumps([record]), encoding="utf-8")
        common.append({
            "seed": seed, "task_id": task, "inference_mode": mode,
            "completed": True, "ode_failure_count": 0, "episode_return": 100.0,
            "temp_violation": 1.0, "co2_violation": 1.0, "rh_violation": 1.0,
            "executed_action_trace_path": f"traces/{executed.name}",
            "requested_action_trace_path": f"traces/{requested.name}",
            "intervention_records_path": f"intervention_records/{records.name}",
        })
    unshielded = pd.DataFrame(common)
    summary = aggregate_episode_interventions([record], 1)
    raw = pd.DataFrame([{**row, "total_steps": 1, "intervention_count": 0} for row in common])
    interventions = pd.DataFrame([{**{key: row[key] for key in ("seed", "task_id", "inference_mode")}, **summary} for row in common])
    expected = set(keys)
    paired = build_paired_shield_deltas(raw, unshielded, expected)
    gate = evaluate_shield_gate(raw, unshielded, expected)
    decision = {**gate, "stage": "stage2_shielded_context_ab", "outcome": "continue_to_full_suite"}
    unshielded.to_csv(unshielded_root / "eval_raw.csv", index=False)
    unshielded_manifest = {"schema_version": "context-ab-v1", "result_root": str(unshielded_root.resolve())}
    (unshielded_root / "diagnostic_manifest.json").write_text(json.dumps(unshielded_manifest), encoding="utf-8")
    manifest = {
        "schema_version": "shielded-context-ab-stage2-v1", "method": cli.SHIELD_METHOD,
        "result_root": str(root.resolve()), "unshielded_result_root": str(unshielded_root.resolve()),
        "unshielded_manifest_sha256": cli._sha(unshielded_root / "diagnostic_manifest.json"),
        "seeds": list(APPROVED_SEEDS), "task_ids": list(DIAGNOSTIC_TASK_IDS),
        "inference_modes": list(MODES),
        "checkpoints": [
            {"seed": 42, "model_sha256": "a" * 64, "vecnormalize_sha256": "b" * 64, "checkpoint_steps": 10},
            {"seed": 123, "model_sha256": "c" * 64, "vecnormalize_sha256": "d" * 64, "checkpoint_steps": 10},
        ],
    }
    raw.to_csv(root / "eval_raw.csv", index=False)
    paired.to_csv(root / "paired_deltas.csv", index=False)
    interventions.to_csv(root / "interventions.csv", index=False)
    (root / "shield_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    decision_path = root / "decision.json"
    decision_path.write_text(json.dumps(decision), encoding="utf-8")
    return decision_path


def _formal_unshielded_fixture(cli, tmp_path: Path):
    from experiments.scripts import run_shielded_context_ab as stage2_source

    suite = create_default_suite_config(result_root=tmp_path / "suite-source")
    manifest_path = write_suite_manifest(suite, tmp_path / "suite.json")
    tasks = pd.DataFrame(vars(item) for item in build_evaluation_tasks(suite))
    tasks_path = tmp_path / "tasks.csv"
    tasks.to_csv(tasks_path, index=False)
    models, vecs = {}, {}
    for seed in (42, 123):
        models[seed] = tmp_path / f"model-{seed}.zip"
        vecs[seed] = tmp_path / f"vec-{seed}.pkl"
        models[seed].write_bytes(f"model-{seed}".encode())
        vecs[seed].write_bytes(f"vec-{seed}".encode())
    runs = pd.DataFrame([
        {
            "suite_id": suite.suite_id,
            "algorithm": "agri_metarl",
            "seed": seed,
            "run_name": f"agri_metarl_seed{seed}",
            "model_path": str(models[seed]),
            "vecnormalize_path": str(vecs[seed]),
            "status": "completed",
        }
        for seed in (42, 123)
    ])
    runs_path = tmp_path / "runs.csv"
    runs.to_csv(runs_path, index=False)
    decision = _stage2_fixture(cli, tmp_path / "stage2")
    stage2_manifest_path = decision.parent / "shield_manifest.json"
    stage2_manifest = json.loads(stage2_manifest_path.read_text(encoding="utf-8"))
    _, rule_sha = stage2_source._load_rule_params()
    stage2_manifest.update({
        "source_manifest_sha256": cli._sha(manifest_path),
        "source_tasks_sha256": cli._sha(tasks_path),
        "rule_config_sha256": rule_sha,
        "env_config_sha256": cli._sha(stage2_source.ENV_CONFIG_PATH),
        "runtime_source_tree_sha256": stage2_source._runtime_source_tree_sha256(),
        "evaluator_source_sha256": cli._sha(Path(cli.__file__)),
        "gate_source_sha256": cli._sha(
            Path(cli.__file__).with_name("evaluate_shield_gate.py")
        ),
        "fixed_lambdas": list(stage2_source.DEFAULT_LAMBDAS),
        "formal_solver_options": dict(stage2_source.FORMAL_CVODES_OPTIONS),
        "checkpoints": [
            {
                "seed": seed,
                "model_sha256": cli._sha(models[seed]),
                "vecnormalize_sha256": cli._sha(vecs[seed]),
                "checkpoint_steps": 10,
            }
            for seed in (42, 123)
        ],
        **stage2_source._behavior_source_hashes(),
    })
    stage2_manifest_path.write_text(json.dumps(stage2_manifest), encoding="utf-8")
    root = tmp_path / "formal-unshielded"
    args = SimpleNamespace(
        manifest=str(manifest_path),
        runs_csv=str(runs_path),
        tasks_csv=str(tasks_path),
        algorithms=["agri_metarl"],
        seeds=[42, 123],
        splits=None,
        task_ids=None,
        limit_tasks=None,
        resume_eval=False,
        action_shield=False,
        formal_unshielded_provenance=True,
        stage2_decision=str(decision),
        result_root=str(root),
        interventions_out=None,
    )

    class Model:
        num_timesteps = 10

        def predict(self, observation, deterministic):
            return np.array([[0.0]], dtype=np.float32), None

    class Loader:
        @staticmethod
        def load(path, device):
            return Model()

    class Env:
        def __init__(self):
            self.step_count = 0

        def get_attr(self, name):
            if name == "N":
                return [3]
            raise AttributeError(name)

        def env_method(self, name, enabled):
            assert name == "set_ode_diagnostics_enabled"

        def reset(self):
            return np.array([[0.0, 1.0]], dtype=np.float32)

        def step(self, actions):
            from tests.experiments.test_ode_failure import _info
            from experiments.scripts.run_shielded_context_ab import FORMAL_CVODES_OPTIONS

            self.step_count += 1
            info = _info(0, failure=True)
            failure = info["integration_failure"]
            failure["solver_options"] = dict(FORMAL_CVODES_OPTIONS)
            failure["exception_message"] = "CV_CONV_FAILURE"
            failure["traceback"] = "RuntimeError: CV_CONV_FAILURE\n"
            return (
                np.array([[2.0, 3.0]], dtype=np.float32),
                np.array([-1.0], dtype=np.float32),
                np.array([True]),
                [info],
            )

        def close(self):
            pass

    return args, root, {"agri_metarl": Loader}, lambda *unused: Env()


@pytest.mark.parametrize("message", ["model prediction failed", "episode terminated early"])
def test_formal_unshielded_rejects_ordinary_failure_without_new_capsule(tmp_path, message):
    cli = _module()
    args, root, model_map, env_loader = _formal_unshielded_fixture(cli, tmp_path)

    def fail_without_capsule(model, env, *, failure_recorder):
        raise RuntimeError(message)

    with pytest.raises(RuntimeError, match=message):
        cli.run(
            args,
            model_map=model_map,
            env_loader=env_loader,
            episode_runner=fail_without_capsule,
        )
    assert not root.exists()
    assert not (root.parent / f".{root.name}.work" / "eval_raw.csv").exists()


def test_formal_unshielded_rejects_capsule_with_non_horizon_runtime_error(tmp_path):
    from tests.experiments.test_ode_failure import _info
    from experiments.scripts.run_shielded_context_ab import FORMAL_CVODES_OPTIONS

    cli = _module()
    args, root, model_map, env_loader = _formal_unshielded_fixture(cli, tmp_path)

    def unrelated_wrapper(model, env, *, failure_recorder):
        info = _info(0, failure=True)
        failure = info["integration_failure"]
        failure["solver_options"] = dict(FORMAL_CVODES_OPTIONS)
        failure["exception_message"] = "model prediction failed after capsule"
        failure["traceback"] = "RuntimeError: model prediction failed after capsule\n"
        failure_recorder.record_step(
            0, np.array([0.0, 1.0]), -1.0, True, info
        )
        raise RuntimeError("model prediction failed after capsule")

    with pytest.raises(RuntimeError, match="model prediction failed after capsule") as caught:
        cli.run(
            args,
            model_map=model_map,
            env_loader=env_loader,
            episode_runner=unrelated_wrapper,
        )
    assert any("not the exact early-horizon" in note for note in caught.value.__notes__)
    assert not root.exists()
    assert not (root.parent / f".{root.name}.work" / "eval_raw.csv").exists()


def test_formal_unshielded_never_classifies_keyboard_interrupt(tmp_path):
    cli = _module()
    args, root, model_map, env_loader = _formal_unshielded_fixture(cli, tmp_path)

    def interrupt(model, env, *, failure_recorder):
        raise KeyboardInterrupt("stop")

    with pytest.raises(KeyboardInterrupt, match="stop"):
        cli.run(
            args,
            model_map=model_map,
            env_loader=env_loader,
            episode_runner=interrupt,
        )
    assert not root.exists()
    assert not (root.parent / f".{root.name}.work" / "eval_raw.csv").exists()


def test_formal_unshielded_accepts_exactly_one_matching_failure_capsule(tmp_path):
    cli = _module()
    args, root, model_map, env_loader = _formal_unshielded_fixture(cli, tmp_path)
    calls = 0

    def one_wrapped_solver_failure(model, env, *, failure_recorder):
        nonlocal calls
        calls += 1
        if calls == 1:
            return cli.run_deterministic_episode(
                model, env, failure_recorder=failure_recorder
            )
        return {
            "episode_return": 100.0,
            "temp_violation": 1.0,
            "co2_violation": 1.0,
            "rh_violation": 1.0,
        }

    assert cli.run(
        args,
        model_map=model_map,
        env_loader=env_loader,
        episode_runner=one_wrapped_solver_failure,
    ) == 182
    frame = pd.read_csv(root / "eval_raw.csv")
    failed = frame.loc[~frame["completed"]]
    assert len(failed) == 1
    assert failed.iloc[0].status == "ode_failure"
    capsule_manifest = root / failed.iloc[0].failure_evidence_path
    assert capsule_manifest.name == "manifest.json"
    from gl_gym.experiments.ode_failure import load_failure_capsule
    load_failure_capsule(capsule_manifest.parent)


def test_stage2_five_artifact_identity_recomputes_gate_and_rejects_forgery(tmp_path):
    cli = _module()
    decision = _stage2_fixture(cli, tmp_path / "stage2")
    loaded = cli.load_stage2_evidence(decision)
    assert len(loaded["stage2_identity_sha256"]) == 64
    forged = json.loads(decision.read_text(encoding="utf-8"))
    forged["evidence"]["paired_count"] = 999
    decision.write_text(json.dumps(forged), encoding="utf-8")
    with pytest.raises(ValueError, match="recomputed|authentic"):
        cli.load_stage2_evidence(decision)


def test_stage2_rejects_abbreviated_protocol_and_missing_trace(tmp_path):
    cli = _module()
    decision = _stage2_fixture(cli, tmp_path / "stage2")
    manifest_path = decision.parent / "shield_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["task_ids"] = manifest["task_ids"][:2]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="exact approved"):
        cli.load_stage2_evidence(decision)
    decision = _stage2_fixture(cli, tmp_path / "stage2-second")
    first = pd.read_csv(decision.parent / "eval_raw.csv").iloc[0]
    (decision.parent / first.executed_action_trace_path).unlink(missing_ok=True)
    with pytest.raises(ValueError, match="trace|evidence"):
        cli.load_stage2_evidence(decision)


def test_asymmetric_or_malformed_resume_quarantines_both_progress_sides(tmp_path):
    cli = _module()
    work = tmp_path / ".shield.work"; work.mkdir()
    pd.DataFrame([{"seed": 42}]).to_csv(work / "eval_raw.csv", index=False)
    evidence = work / "traces"; evidence.mkdir(); (evidence / "stale.npy").write_bytes(b"bad")
    raw, interventions = cli._read_shield_progress(work)
    assert raw.empty and interventions.empty
    assert list(work.iterdir()) == []


def test_real_shield_smoke_runs_only_in_work_with_provenance(tmp_path):
    cli = _module()
    from experiments.scripts import run_shielded_context_ab as stage2_source
    suite = create_default_suite_config(result_root=tmp_path / "unshielded-suite")
    manifest_path = write_suite_manifest(suite, tmp_path / "suite.json")
    tasks = pd.DataFrame(vars(item) for item in build_evaluation_tasks(suite))
    tasks_path = tmp_path / "tasks.csv"; tasks.to_csv(tasks_path, index=False)
    model_paths, vec_paths = {}, {}
    for seed in (42, 123):
        model_paths[seed] = tmp_path / f"model-{seed}.zip"; model_paths[seed].write_bytes(f"model{seed}".encode())
        vec_paths[seed] = tmp_path / f"vec-{seed}.pkl"; vec_paths[seed].write_bytes(f"vec{seed}".encode())
    runs = pd.DataFrame([{
        "suite_id": suite.suite_id, "algorithm": "agri_metarl", "seed": seed,
        "run_name": f"agri_metarl_seed{seed}", "model_path": str(model_paths[seed]),
        "vecnormalize_path": str(vec_paths[seed]), "status": "completed",
    } for seed in (42, 123)])
    runs_path = tmp_path / "runs.csv"; runs.to_csv(runs_path, index=False)
    decision_path = _stage2_fixture(cli, tmp_path / "stage2")
    stage2_manifest_path = decision_path.parent / "shield_manifest.json"
    stage2_manifest = json.loads(stage2_manifest_path.read_text(encoding="utf-8"))
    _, rule_sha = stage2_source._load_rule_params()
    stage2_manifest.update({
        "source_manifest_sha256": cli._sha(manifest_path), "source_tasks_sha256": cli._sha(tasks_path),
        "rule_config_sha256": rule_sha, "env_config_sha256": cli._sha(stage2_source.ENV_CONFIG_PATH),
        "runtime_source_tree_sha256": stage2_source._runtime_source_tree_sha256(),
        "fixed_lambdas": list(stage2_source.DEFAULT_LAMBDAS),
        "formal_solver_options": dict(stage2_source.FORMAL_CVODES_OPTIONS),
        "checkpoints": [{"seed": seed, "model_sha256": cli._sha(model_paths[seed]),
                         "vecnormalize_sha256": cli._sha(vec_paths[seed]),
                         "checkpoint_steps": 77} for seed in (42, 123)],
        **stage2_source._behavior_source_hashes(),
    })
    stage2_manifest_path.write_text(json.dumps(stage2_manifest), encoding="utf-8")
    result_root = tmp_path / "shield-final"
    args = SimpleNamespace(
        manifest=str(manifest_path), runs_csv=str(runs_path), tasks_csv=str(tasks_path),
        algorithms=["agri_metarl"], seeds=[42, 123], splits=None, task_ids=None,
        limit_tasks=1, resume_eval=False, action_shield=True,
        stage2_decision=str(decision_path), result_root=str(result_root), interventions_out=None,
    )
    calls, closed = [], []
    class Model:
        num_timesteps = 77
    class Loader:
        @staticmethod
        def load(path, device): return Model()
    class Env:
        def close(self): closed.append(self)
    def env_loader(suite_arg, task, vec_path, *, shield_params):
        assert shield_params and task.task_id == tasks.iloc[0].task_id
        calls.append((task.task_id, Path(vec_path)))
        return Env()
    record = {
        "schema_version": "minimal-feasibility-action-shield-v1", "intervened": False,
        "requested_action": [0.0], "reference_action": None, "executed_action": [0.0],
        "selected_lambda": 0.0, "candidate_attempts": [], "intervention_l1": 0.0,
        "intervention_l2": 0.0, "intervention_linf": 0.0, "per_channel_changed": [False],
        "extra_solver_attempts": 0, "elapsed_seconds": 0.0, "original_failure": None,
    }
    def episode_runner(model, env, *, return_diagnostics):
        assert return_diagnostics
        return ({"episode_return": 100.0, "temp_violation": 1.0, "co2_violation": 1.0,
                 "rh_violation": 1.0},
                {"action_trace": np.zeros((1, 1)), "requested_action_trace": np.zeros((1, 1)),
                 "action_shield_records": [record]})
    assert cli.run(args, model_map={"agri_metarl": Loader}, env_loader=env_loader, episode_runner=episode_runner) == 2
    assert not result_root.exists()
    work = tmp_path / ".shield-final.work"
    raw = pd.read_csv(work / "eval_raw.csv")
    assert len(raw) == len(calls) == len(closed) == 2
    assert raw["method"].eq(cli.SHIELD_METHOD).all()
    assert raw["checkpoint_steps"].eq(77).all()
    assert raw["formal_complete"].eq(False).all()
    Path(raw.iloc[0].executed_action_trace_path).write_bytes(b"corrupt")
    calls.clear(); closed.clear(); args.resume_eval = True
    assert cli.run(args, model_map={"agri_metarl": Loader}, env_loader=env_loader, episode_runner=episode_runner) == 2
    assert len(calls) == len(closed) >= 1
    resumed = pd.read_csv(work / "eval_raw.csv")
    assert len(resumed) == 2 and not resumed.duplicated(["seed", "task_id"]).any()
    pd.DataFrame([{"wrong": 1}]).to_csv(work / "eval_raw.csv", index=False)
    pd.DataFrame([{"wrong": 1}]).to_csv(work / "interventions.csv", index=False)
    raw, interventions = cli._read_shield_progress(work)
    assert raw.empty and interventions.empty
    assert list(work.iterdir()) == []
    (work / "eval_raw.csv").write_text('"unterminated', encoding="utf-8")
    pd.DataFrame([{"seed": 42}]).to_csv(work / "interventions.csv", index=False)
    raw, interventions = cli._read_shield_progress(work)
    assert raw.empty and interventions.empty
    assert list(work.iterdir()) == []
