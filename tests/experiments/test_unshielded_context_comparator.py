from __future__ import annotations

from dataclasses import asdict
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from experiments.scripts import run_unshielded_context_comparator as cli
from gl_gym.environments.models.utils import FORMAL_CVODES_OPTIONS


def _tasks() -> pd.DataFrame:
    rows = []
    for index, task_id in enumerate(cli.DIAGNOSTIC_TASK_IDS):
        rows.append(
            {
                "suite_id": "suite-x",
                "task_id": task_id,
                "split": "fixed" if index == 0 else "heldout",
                "weather_year": 2010 + index,
                "start_day": 59,
                "uncertainty_scale": 0.0,
                "economic_scenario": "standard",
                "climate_constraint_scenario": "standard",
            }
        )
    return pd.DataFrame(rows)


class _Env:
    def __init__(self, close_error: BaseException | None = None):
        self.close_error = close_error
        self.closed = False

    def close(self):
        self.closed = True
        if self.close_error is not None:
            raise self.close_error


class _Recorder:
    def __init__(self, root: Path, context, *, malformed: str | None = None):
        self.root = Path(root)
        self.context = context
        self.malformed = malformed
        self.identity = "a" * 64

    def emit(self, step: int = 2):
        first = self.root / "capsule-a"
        first.mkdir(parents=True)
        (first / "manifest.json").write_text("{}", encoding="utf-8")
        if self.malformed == "multiple":
            second = self.root / "capsule-b"
            second.mkdir()
            (second / "manifest.json").write_text("{}", encoding="utf-8")


def _capsule_loader(path: str | Path):
    path = Path(path)
    recorder = _RECORDER_BY_ROOT.get(path.parent)
    if recorder is None:
        recorder = next(
            item for root, item in _RECORDER_BY_ROOT.items()
            if root.name == path.parent.name
            and Path(item.context.formal_result_root).resolve() in path.resolve().parents
        )
    context = asdict(recorder.context)
    if recorder.malformed == "context":
        context["task_id"] = "wrong-task"
    step = 2
    return SimpleNamespace(
        path=path,
        manifest={
            "context": context,
            "exception": {"type": "RuntimeError", "message": "CVODES failed"},
            "solver": {"options": dict(FORMAL_CVODES_OPTIONS)},
            "content_identity_sha256": recorder.identity,
        },
        history_arrays={"step_index": np.asarray([step], dtype=np.int64)},
        failure_inputs={"timestep": np.asarray(step, dtype=np.int64)},
        traceback_text="RuntimeError: CVODES failed",
    )


_RECORDER_BY_ROOT: dict[Path, _Recorder] = {}


def _inputs(tmp_path: Path, *, failure_modes=(), malformed=None, close_error=None):
    tmp_path.mkdir(parents=True, exist_ok=True)
    manifest = tmp_path / "source_manifest.json"
    task_csv = tmp_path / "tasks.csv"
    manifest.write_text("{}", encoding="utf-8")
    _tasks().to_csv(task_csv, index=False)
    suite = SimpleNamespace(suite_id="suite-x", result_root=tmp_path / "formal-suite")
    runs = []
    for seed in cli.APPROVED_SEEDS:
        runs.append(
            {
                "seed": seed,
                "model_path": tmp_path / f"model-{seed}.zip",
                "vecnormalize_path": tmp_path / f"vec-{seed}.pkl",
                "model_sha256": f"{seed % 10}" * 64,
                "vecnormalize_sha256": f"{(seed + 1) % 10}" * 64,
            }
        )
    calls = []
    envs = []

    def model_loader(path, device):
        return SimpleNamespace(num_timesteps=17)

    def env_loader(suite_arg, task, vec_path):
        env = _Env(close_error if not envs else None)
        envs.append(env)
        return env

    def recorder_factory(root, context):
        recorder = _Recorder(Path(root), context, malformed=malformed)
        _RECORDER_BY_ROOT[Path(root)] = recorder
        return recorder

    def episode_runner(model, env, *, inference_mode, return_diagnostics, failure_recorder):
        key = (failure_recorder.context.seed, failure_recorder.context.task_id, inference_mode)
        calls.append(key)
        if key in failure_modes:
            failure_recorder.emit()
            raise RuntimeError(
                "evaluation episode terminated before configured horizon: step 3 of 10"
            )
        metrics = {
            name: float(index + 1)
            for index, name in enumerate(cli.EPISODE_SCORING_METRICS)
        }
        return metrics, {
            "support_ready_step": 1.0,
            "context_norm_mean": 0.5,
            "context_norm_max": 1.0,
            "action_trace": np.ones((3, 2), dtype=np.float32),
        }

    kwargs = dict(
        suite=suite,
        tasks=_tasks(),
        runs=runs,
        result_root=tmp_path / "comparator-final",
        failure_root=tmp_path / "failure-root",
        source_manifest=manifest,
        source_tasks_csv=task_csv,
        device="cpu",
        resume=False,
        model_loader=model_loader,
        env_loader=env_loader,
        episode_runner=episode_runner,
        provenance_loader=lambda: {"git_commit": "b" * 40, "dirty": False},
        recorder_factory=recorder_factory,
        capsule_loader=_capsule_loader,
    )
    return kwargs, calls, envs


def test_bootstrap_help_works_outside_repository(tmp_path):
    script = Path(cli.__file__).resolve()
    result = subprocess.run(
        [sys.executable, str(script), "--help"],
        cwd=tmp_path,
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0, result.stderr
    assert "--legacy_progress" in result.stdout


def test_injected_successes_publish_exact_immutable_result(tmp_path):
    kwargs, calls, envs = _inputs(tmp_path)
    frame = cli.run_unshielded_comparator(**kwargs)

    assert len(frame) == 32
    assert set(zip(frame.seed, frame.task_id, frame.inference_mode)) == {
        (seed, task, mode)
        for seed in cli.APPROVED_SEEDS
        for task in cli.DIAGNOSTIC_TASK_IDS
        for mode in cli.MODES
    }
    assert frame["completed"].eq(True).all()
    assert frame["status"].eq("completed").all()
    assert frame["ode_failure_count"].eq(0).all()
    assert frame["failure_evidence_path"].eq("").all()
    assert np.isfinite(frame[list(cli.REQUIRED_METRICS)].to_numpy()).all()
    assert np.isfinite(frame[["context_norm_mean", "context_norm_max"]].to_numpy()).all()
    root = Path(kwargs["result_root"]).resolve()
    assert all(root in Path(path).resolve().parents for path in frame.action_trace_path)
    assert all(Path(path).is_file() for path in frame.action_trace_path)
    assert {path.name for path in root.iterdir()} == {
        "eval_raw.csv", "context_ab_manifest.json", "traces", "failures"
    }
    manifest = cli._strict_json(root / "context_ab_manifest.json")
    assert Path(manifest["result_root"]).resolve() == root
    assert len(manifest["row_identities"]) == 32
    assert set(manifest["published_file_sha256"]) == {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path.name != "context_ab_manifest.json"
    }
    assert all(
        digest == cli.sha256_file(root / relative)
        for relative, digest in manifest["published_file_sha256"].items()
    )
    assert (Path(kwargs["result_root"]).parent / ".comparator-final.work" / "progress.csv").is_file()
    assert len(calls) == 32 and all(env.closed for env in envs)


def test_publication_rejects_31_rows_without_creating_final_root(tmp_path):
    kwargs, _, _ = _inputs(tmp_path)
    kwargs["episode_runner"] = lambda *args, **inner: (_ for _ in ()).throw(
        KeyboardInterrupt()
    )
    with pytest.raises(KeyboardInterrupt):
        cli.run_unshielded_comparator(**kwargs)
    assert not Path(kwargs["result_root"]).exists()

    with pytest.raises(RuntimeError, match="exact 32-key"):
        cli._publish_comparator(
            [{} for _ in range(31)],
            root=Path(kwargs["result_root"]),
            work=cli._work_root(Path(kwargs["result_root"])),
            failure_work=tmp_path / "failure-work",
            suite=kwargs["suite"],
            runs=kwargs["runs"],
            source_manifest=kwargs["source_manifest"],
            source_tasks_csv=kwargs["source_tasks_csv"],
            provenance={},
            runtime_source_tree_sha256="a" * 64,
            capsule_loader=_capsule_loader,
        )


def test_failed_rows_and_capsules_are_rewritten_beneath_final_root(tmp_path):
    key = (42, cli.DIAGNOSTIC_TASK_IDS[0], cli.MODES[0])
    kwargs, _, _ = _inputs(tmp_path, failure_modes={key})
    frame = cli.run_unshielded_comparator(**kwargs)
    root = Path(kwargs["result_root"]).resolve()
    failed = frame.loc[~frame.completed].iloc[0]
    evidence = Path(failed.failure_evidence_path).resolve()
    assert root in evidence.parents and evidence.is_file()
    assert all(root in Path(path).resolve().parents for path in frame.loc[frame.completed, "action_trace_path"])
    assert not any(".work" in str(path) for path in frame.action_trace_path.astype(str))


def test_old_final_root_is_preserved_when_new_execution_is_partial(tmp_path):
    kwargs, _, _ = _inputs(tmp_path)
    cli.run_unshielded_comparator(**kwargs)
    root = Path(kwargs["result_root"])
    old = (root / "eval_raw.csv").read_bytes()
    kwargs["resume"] = False
    kwargs["episode_runner"] = lambda *args, **inner: (_ for _ in ()).throw(
        RuntimeError("arbitrary runtime")
    )
    with pytest.raises(RuntimeError, match="arbitrary"):
        cli.run_unshielded_comparator(**kwargs)
    assert (root / "eval_raw.csv").read_bytes() == old


def test_publish_rename_failure_uses_copy_fallback(tmp_path, monkeypatch):
    kwargs, _, _ = _inputs(tmp_path)
    original = Path.rename
    monkeypatch.setattr(Path, "rename", lambda self, target: (_ for _ in ()).throw(OSError("rename")))
    frame = cli.run_unshielded_comparator(**kwargs)
    assert len(frame) == 32
    assert (Path(kwargs["result_root"]) / "eval_raw.csv").is_file()
    monkeypatch.setattr(Path, "rename", original)


def test_publish_copy_fallback_failure_restores_unique_old_root(tmp_path, monkeypatch):
    kwargs, _, _ = _inputs(tmp_path)
    cli.run_unshielded_comparator(**kwargs)
    root = Path(kwargs["result_root"])
    old = (root / "eval_raw.csv").read_bytes()
    kwargs["resume"] = False
    monkeypatch.setattr(Path, "rename", lambda self, target: (_ for _ in ()).throw(OSError("rename")))
    real_copytree = cli.shutil.copytree
    calls = {"count": 0}
    def broken_copytree(source, destination, *args, **copy_kwargs):
        calls["count"] += 1
        if calls["count"] == 2:
            raise OSError("copy interrupted")
        return real_copytree(source, destination, *args, **copy_kwargs)
    monkeypatch.setattr(cli.shutil, "copytree", broken_copytree)
    with pytest.raises(OSError, match="copy interrupted"):
        cli.run_unshielded_comparator(**kwargs)
    assert (root / "eval_raw.csv").read_bytes() == old
    backups = list(root.parent.glob(f".{root.name}.backup*"))
    assert len(backups) <= 1


def test_publish_rename_interruption_restores_old_root(tmp_path, monkeypatch):
    kwargs, _, _ = _inputs(tmp_path)
    cli.run_unshielded_comparator(**kwargs)
    root = Path(kwargs["result_root"])
    old = (root / "eval_raw.csv").read_bytes()
    kwargs["resume"] = False
    real_rename = Path.rename
    calls = {"count": 0}
    def interrupted_rename(self, destination):
        calls["count"] += 1
        if calls["count"] == 2:
            raise KeyboardInterrupt()
        return real_rename(self, destination)
    monkeypatch.setattr(Path, "rename", interrupted_rename)
    with pytest.raises(KeyboardInterrupt):
        cli.run_unshielded_comparator(**kwargs)
    assert (root / "eval_raw.csv").read_bytes() == old
    assert not (root.parent / f".{root.name}.backup").exists()


def test_round_trip_rejection_restores_old_final_root(tmp_path, monkeypatch):
    kwargs, _, _ = _inputs(tmp_path)
    cli.run_unshielded_comparator(**kwargs)
    root = Path(kwargs["result_root"])
    old = (root / "eval_raw.csv").read_bytes()
    kwargs["resume"] = False
    monkeypatch.setattr(
        cli, "_load_published_comparator",
        lambda *args, **inner: (_ for _ in ()).throw(ValueError("consumer rejected")),
    )
    with pytest.raises(ValueError, match="consumer rejected"):
        cli.run_unshielded_comparator(**kwargs)
    assert (root / "eval_raw.csv").read_bytes() == old


@pytest.mark.parametrize("source_name", ["source_manifest", "source_tasks_csv"])
@pytest.mark.parametrize("lifecycle", ["work", "stage", "backup", "failure"])
def test_source_inputs_must_not_live_in_mutable_output_topology(
    tmp_path, source_name, lifecycle
):
    kwargs, _, _ = _inputs(tmp_path)
    root = Path(kwargs["result_root"])
    locations = {
        "work": cli._work_root(root),
        "stage": root.parent / f".{root.name}.publish",
        "backup": root.parent / f".{root.name}.backup",
        "failure": Path(kwargs["failure_root"]),
    }
    source = locations[lifecycle] / f"{source_name}.dat"
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_text("x", encoding="utf-8")
    kwargs[source_name] = source
    with pytest.raises(ValueError, match="topology|disjoint"):
        cli.run_unshielded_comparator(**kwargs)


def test_exact_early_wrapper_with_one_matching_capsule_continues_multiple_failures(tmp_path):
    failures = {
        (42, cli.DIAGNOSTIC_TASK_IDS[0], cli.MODES[0]),
        (123, cli.DIAGNOSTIC_TASK_IDS[3], cli.MODES[1]),
    }
    kwargs, calls, _ = _inputs(tmp_path, failure_modes=failures)
    frame = cli.run_unshielded_comparator(**kwargs)

    failed = frame.loc[~frame.completed]
    assert len(failed) == 2 and len(calls) == 32
    assert failed.status.eq("ode_failure").all()
    assert failed.ode_failure_count.eq(1).all()
    assert failed[list(cli.EPISODE_SCORING_METRICS)].isna().all(axis=None)
    assert failed.action_trace_path.eq("").all()
    assert all(Path(path).name == "manifest.json" for path in failed.failure_evidence_path)


@pytest.mark.parametrize(
    "exception",
    [
        RuntimeError("arbitrary runtime"),
        RuntimeError("evaluation episode terminated before configured horizon: step 3 of 10"),
        KeyboardInterrupt(),
        SystemExit(9),
    ],
)
def test_nonclassifiable_and_base_exceptions_propagate(tmp_path, exception):
    kwargs, _, _ = _inputs(tmp_path)

    def runner(*args, **kwargs):
        raise exception

    kwargs["episode_runner"] = runner
    with pytest.raises(type(exception)):
        cli.run_unshielded_comparator(**kwargs)


def test_real_episode_wrapper_and_real_capsule_are_classified_then_run_continues(tmp_path):
    class Model:
        num_timesteps = 17

        def begin_inference_episode(self, mode):
            pass

        def predict(self, obs, **kwargs):
            return np.asarray([[0.0]], dtype=np.float32), None

        def observe_inference_transition(self, *args):
            pass

        def inference_episode_diagnostics(self):
            return {
                "support_ready_step": 1.0,
                "context_norm_mean": 0.5,
                "context_norm_max": 1.0,
            }

        def end_inference_episode(self):
            pass

    class FailureEnv:
        def __init__(self):
            self.step_index = -1

        def get_attr(self, name):
            assert name == "N"
            return [10]

        def env_method(self, name, enabled):
            assert name == "set_ode_diagnostics_enabled"

        def reset(self):
            return np.asarray([[0.0]], dtype=np.float32)

        def step(self, actions):
            self.step_index += 1
            step = self.step_index
            done = step == 2
            transition = {
                "raw_observation": np.asarray([step], dtype=np.float64),
                "requested_action": np.asarray([0.0], dtype=np.float64),
                "previous_control": np.asarray([0.0], dtype=np.float64),
                "executed_control": np.asarray([0.0], dtype=np.float64),
                "raw_next_observation": None if done else np.asarray([step + 1], dtype=np.float64),
                "raw_next_observation_available": not done,
            }
            info = {
                "diagnostic_transition": transition,
                "EPI": 1.0,
                "temp_violation": 0.0,
                "co2_violation": 0.0,
                "rh_violation": 0.0,
            }
            if done:
                weather = np.asarray([19.5], dtype=np.float64)
                parameters = np.asarray([1.1], dtype=np.float64)
                info["terminal_observation"] = np.asarray([3.0], dtype=np.float32)
                info["integration_failure"] = {
                    "x0": np.asarray([4.0], dtype=np.float64),
                    "u": np.asarray([0.0], dtype=np.float64),
                    "previous_control": np.asarray([0.0], dtype=np.float64),
                    "requested_action": np.asarray([0.0], dtype=np.float64),
                    "weather": weather,
                    "sampled_parameters": parameters,
                    "p_dyn": np.concatenate((weather, parameters)),
                    "timestep": step,
                    "day_of_year": 151.0,
                    "hour_of_day": 12.5,
                    "dt": 300.0,
                    "nx": 1,
                    "nu": 1,
                    "nd": 1,
                    "n_params": 1,
                    "solver_options": dict(FORMAL_CVODES_OPTIONS),
                    "exception_type": "RuntimeError",
                    "exception_message": "CVODES failed",
                    "traceback": "Traceback (most recent call last):\nRuntimeError: CVODES failed\n",
                }
            return (
                np.asarray([[step + 1.0]], dtype=np.float32),
                np.asarray([1.0], dtype=np.float32),
                np.asarray([done]),
                [info],
            )

        def close(self):
            pass

    kwargs, calls, _ = _inputs(tmp_path)
    attempt = 0

    def env_loader(*args):
        nonlocal attempt
        attempt += 1
        return FailureEnv() if attempt == 1 else _Env()

    fake_success = kwargs["episode_runner"]

    def runner(model, env, **runner_kwargs):
        if isinstance(env, FailureEnv):
            return cli.run_deterministic_episode(model, env, **runner_kwargs)
        return fake_success(model, env, **runner_kwargs)

    kwargs.update(
        model_loader=lambda *args: Model(),
        env_loader=env_loader,
        episode_runner=runner,
        recorder_factory=cli.FailureCapsuleRecorder,
        capsule_loader=cli.load_failure_capsule,
    )
    frame = cli.run_unshielded_comparator(**kwargs)
    assert len(frame) == 32
    assert (~frame.completed).sum() == 1
    failure = frame.loc[~frame.completed].iloc[0]
    assert Path(failure.failure_evidence_path).is_file()
    loaded = cli.load_failure_capsule(Path(failure.failure_evidence_path).parent)
    assert loaded.manifest["solver"]["options"] == dict(FORMAL_CVODES_OPTIONS)


@pytest.mark.parametrize("malformed", ["multiple", "context"])
def test_multiple_or_mismatched_capsules_rethrow_original_wrapper(tmp_path, malformed):
    failure = {(42, cli.DIAGNOSTIC_TASK_IDS[0], cli.MODES[0])}
    kwargs, _, _ = _inputs(tmp_path, failure_modes=failure, malformed=malformed)
    with pytest.raises(RuntimeError, match="terminated before") as caught:
        cli.run_unshielded_comparator(**kwargs)
    assert any("classification rejected" in note for note in caught.value.__notes__)


def test_success_with_capsule_is_rejected(tmp_path):
    kwargs, _, _ = _inputs(tmp_path)
    original = kwargs["episode_runner"]

    def runner(*args, **runner_kwargs):
        runner_kwargs["failure_recorder"].emit()
        return original(*args, **runner_kwargs)

    kwargs["episode_runner"] = runner
    with pytest.raises(ValueError, match="unexpectedly produced"):
        cli.run_unshielded_comparator(**kwargs)


def test_recorder_factory_and_close_errors_propagate(tmp_path):
    kwargs, _, _ = _inputs(tmp_path)
    kwargs["recorder_factory"] = lambda *args: (_ for _ in ()).throw(OSError("recorder"))
    with pytest.raises(OSError, match="recorder"):
        cli.run_unshielded_comparator(**kwargs)

    kwargs, _, _ = _inputs(tmp_path / "close", close_error=OSError("close"))
    with pytest.raises(OSError, match="close"):
        cli.run_unshielded_comparator(**kwargs)


@pytest.mark.parametrize("kind", ["model", "metrics", "context", "trace"])
def test_model_or_malformed_success_output_propagates(tmp_path, kind):
    kwargs, _, _ = _inputs(tmp_path)
    if kind == "model":
        kwargs["model_loader"] = lambda *args: (_ for _ in ()).throw(ValueError("model"))
    else:
        def runner(*args, **runner_kwargs):
            metrics = {name: 1.0 for name in cli.REQUIRED_METRICS}
            diagnostics = {
                "support_ready_step": 1.0,
                "context_norm_mean": 0.5,
                "context_norm_max": 1.0,
                "action_trace": np.ones((3, 2)),
            }
            if kind == "metrics":
                metrics[cli.REQUIRED_METRICS[0]] = np.inf
            elif kind == "context":
                diagnostics["context_norm_mean"] = np.nan
            else:
                diagnostics["action_trace"] = np.ones(3)
            return metrics, diagnostics
        kwargs["episode_runner"] = runner
    with pytest.raises((ValueError, TypeError)):
        cli.run_unshielded_comparator(**kwargs)


@pytest.mark.parametrize(
    ("collision", "value"),
    [
        ("episode_return", np.nan),
        ("status", "forged"),
        ("model_sha256", "f" * 64),
        ("failure_evidence_path", "forged/manifest.json"),
    ],
)
def test_diagnostics_cannot_override_metrics_or_reserved_row_fields(
    tmp_path, collision, value
):
    kwargs, _, _ = _inputs(tmp_path)

    def runner(*args, **runner_kwargs):
        metrics = {name: 1.0 for name in cli.REQUIRED_METRICS}
        diagnostics = {
            "support_ready_step": 1.0,
            "context_norm_mean": 0.5,
            "context_norm_max": 1.0,
            "action_trace": np.ones((3, 2)),
            collision: value,
        }
        return metrics, diagnostics

    kwargs["episode_runner"] = runner
    with pytest.raises(ValueError, match="collide"):
        cli.run_unshielded_comparator(**kwargs)


def test_parser_requires_explicit_failure_root():
    parser = cli.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "--source_manifest",
                "manifest.json",
                "--source_tasks_csv",
                "tasks.csv",
                "--model_root",
                "models",
                "--seeds",
                "42",
                "123",
            ]
        )


def test_output_roots_must_be_pairwise_disjoint(tmp_path):
    kwargs, _, _ = _inputs(tmp_path)
    kwargs["result_root"] = kwargs["suite"].result_root / "child"
    with pytest.raises(ValueError, match="disjoint"):
        cli.run_unshielded_comparator(**kwargs)


def _progress_path(kwargs) -> Path:
    root = Path(kwargs["result_root"])
    return root.parent / f".{root.name}.work" / "progress.csv"


def _legacy_row(frame: pd.DataFrame) -> pd.DataFrame:
    row = frame.iloc[[0]].copy()
    row["EPI"] = 1.0
    row["revenue"] = 2.0
    return row.drop(
        columns=[
            *cli.STATUS_FIELDS,
            "action_trace_sha256",
            "failure_capsule_identity_sha256",
            "row_identity_sha256",
        ]
    )


def test_resume_skips_canonical_completed_rows(tmp_path):
    kwargs, calls, _ = _inputs(tmp_path)
    first = cli.run_unshielded_comparator(**kwargs)
    assert len(calls) == 32

    calls.clear()
    kwargs["resume"] = True
    resumed = cli.run_unshielded_comparator(**kwargs)

    assert calls == []
    pd.testing.assert_frame_equal(
        resumed.sort_index(axis=1), first.sort_index(axis=1), check_dtype=False
    )


@pytest.mark.parametrize("tamper", ["checkpoint", "source", "diagnostic", "trace"])
def test_resume_recomputes_completed_row_when_evidence_changes(tmp_path, tamper):
    kwargs, calls, _ = _inputs(tmp_path)
    cli.run_unshielded_comparator(**kwargs)
    progress_path = _progress_path(kwargs)
    progress = pd.read_csv(progress_path)
    target = progress.iloc[0]
    key = (int(target.seed), str(target.task_id), str(target.inference_mode))
    if tamper == "checkpoint":
        progress.loc[0, "checkpoint_steps"] = 18
    elif tamper == "source":
        progress.loc[0, "runtime_source_tree_sha256"] = "f" * 64
    elif tamper == "diagnostic":
        progress.loc[0, "context_norm_mean"] = 9.0
    else:
        np.save(Path(target.action_trace_path), np.zeros((3, 2)), allow_pickle=False)
    progress.to_csv(progress_path, index=False)

    calls.clear()
    kwargs["resume"] = True
    cli.run_unshielded_comparator(**kwargs)

    assert calls == [key]


def test_resume_recomputes_row_with_nonfinite_optional_scoring_metric(tmp_path):
    kwargs, calls, _ = _inputs(tmp_path)
    original = kwargs["episode_runner"]

    def runner(*args, **runner_kwargs):
        metrics, diagnostics = original(*args, **runner_kwargs)
        metrics["revenue"] = 2.0
        return metrics, diagnostics

    kwargs["episode_runner"] = runner
    cli.run_unshielded_comparator(**kwargs)
    progress_path = _progress_path(kwargs)
    progress = pd.read_csv(progress_path)
    target = progress.iloc[0]
    key = (int(target.seed), str(target.task_id), str(target.inference_mode))
    progress.loc[0, "revenue"] = np.inf
    progress.loc[0, "row_identity_sha256"] = cli._row_identity(
        progress.iloc[0].to_dict()
    )
    progress.to_csv(progress_path, index=False)

    calls.clear()
    kwargs["resume"] = True
    cli.run_unshielded_comparator(**kwargs)
    assert calls == [key]


def test_resume_skips_valid_failed_capsule(tmp_path):
    failure = {(42, cli.DIAGNOSTIC_TASK_IDS[0], cli.MODES[0])}
    kwargs, calls, _ = _inputs(tmp_path, failure_modes=failure)
    cli.run_unshielded_comparator(**kwargs)
    calls.clear()
    kwargs["resume"] = True

    resumed = cli.run_unshielded_comparator(**kwargs)

    assert calls == []
    assert (~resumed.completed).sum() == 1


def test_resume_recomputes_failed_row_with_finite_optional_scoring_metrics(tmp_path):
    key = (42, cli.DIAGNOSTIC_TASK_IDS[0], cli.MODES[0])
    kwargs, calls, _ = _inputs(tmp_path, failure_modes={key})
    cli.run_unshielded_comparator(**kwargs)
    progress_path = _progress_path(kwargs)
    progress = pd.read_csv(progress_path)
    failed_index = progress.index[progress.status.eq("ode_failure")][0]
    progress.loc[failed_index, "EPI"] = 1.0
    progress.loc[failed_index, "revenue"] = 2.0
    progress.loc[failed_index, "row_identity_sha256"] = cli._row_identity(
        progress.loc[failed_index].to_dict()
    )
    progress.to_csv(progress_path, index=False)
    calls.clear()
    kwargs["resume"] = True

    def successful_runner(model, env, **runner_kwargs):
        recorder = runner_kwargs["failure_recorder"]
        calls.append(
            (recorder.context.seed, recorder.context.task_id, runner_kwargs["inference_mode"])
        )
        metrics = {name: 1.0 for name in cli.EPISODE_SCORING_METRICS}
        return metrics, {
            "support_ready_step": (
                np.nan if runner_kwargs["inference_mode"] == "zero_context" else 1.0
            ),
            "context_norm_mean": 0.5,
            "context_norm_max": 1.0,
            "action_trace": np.ones((3, 2)),
        }

    kwargs["episode_runner"] = successful_runner
    cli.run_unshielded_comparator(**kwargs)
    assert calls == [key]


def test_resume_recomputes_failed_row_when_capsule_identity_changes(tmp_path):
    key = (42, cli.DIAGNOSTIC_TASK_IDS[0], cli.MODES[0])
    kwargs, calls, _ = _inputs(tmp_path, failure_modes={key})
    frame = cli.run_unshielded_comparator(**kwargs)
    progress = pd.read_csv(_progress_path(kwargs))
    manifest = Path(progress.loc[progress.status.eq("ode_failure"), "failure_evidence_path"].iloc[0])
    _RECORDER_BY_ROOT[manifest.parent.parent].identity = "c" * 64
    calls.clear()
    kwargs["resume"] = True

    def successful_runner(model, env, **runner_kwargs):
        recorder = runner_kwargs["failure_recorder"]
        calls.append(
            (recorder.context.seed, recorder.context.task_id, runner_kwargs["inference_mode"])
        )
        metrics = {name: 1.0 for name in cli.REQUIRED_METRICS}
        return metrics, {
            "support_ready_step": (
                np.nan if runner_kwargs["inference_mode"] == "zero_context" else 1.0
            ),
            "context_norm_mean": 0.5,
            "context_norm_max": 1.0,
            "action_trace": np.ones((3, 2)),
        }

    kwargs["episode_runner"] = successful_runner
    cli.run_unshielded_comparator(**kwargs)
    assert calls == [key]


@pytest.mark.parametrize("damage", ["missing", "corrupt", "mismatch"])
def test_resume_recomputes_failed_row_when_capsule_is_not_valid(tmp_path, damage):
    key = (42, cli.DIAGNOSTIC_TASK_IDS[0], cli.MODES[0])
    kwargs, calls, _ = _inputs(tmp_path, failure_modes={key})
    frame = cli.run_unshielded_comparator(**kwargs)
    progress = pd.read_csv(_progress_path(kwargs))
    manifest = Path(progress.loc[progress.status.eq("ode_failure"), "failure_evidence_path"].iloc[0])
    if damage == "missing":
        manifest.unlink()
    elif damage == "corrupt":
        manifest.write_text("not-json", encoding="utf-8")
        _RECORDER_BY_ROOT[manifest.parent.parent].malformed = "context"
    else:
        _RECORDER_BY_ROOT[manifest.parent.parent].malformed = "context"

    calls.clear()
    kwargs["resume"] = True

    def successful_runner(model, env, **runner_kwargs):
        recorder = runner_kwargs["failure_recorder"]
        calls.append((recorder.context.seed, recorder.context.task_id, runner_kwargs["inference_mode"]))
        metrics = {name: 1.0 for name in cli.REQUIRED_METRICS}
        return metrics, {
            "support_ready_step": np.nan if runner_kwargs["inference_mode"] == "zero_context" else 1.0,
            "context_norm_mean": 0.5,
            "context_norm_max": 1.0,
            "action_trace": np.ones((3, 2)),
        }

    kwargs["episode_runner"] = successful_runner
    cli.run_unshielded_comparator(**kwargs)
    assert calls == [key]


def test_resume_ignores_rows_for_other_keys_and_result_roots(tmp_path):
    kwargs, calls, _ = _inputs(tmp_path / "source")
    cli.run_unshielded_comparator(**kwargs)
    progress = pd.read_csv(_progress_path(kwargs))
    progress.loc[0, "task_id"] = "not-a-target"
    progress.to_csv(_progress_path(kwargs), index=False)
    calls.clear()
    kwargs["resume"] = True
    cli.run_unshielded_comparator(**kwargs)
    assert len(calls) == 1

    other, other_calls, _ = _inputs(tmp_path / "other")
    other["resume"] = True
    cli.run_unshielded_comparator(**other)
    assert len(other_calls) == 32


def test_resume_parse_errors_are_stale_not_fatal(tmp_path):
    kwargs, calls, _ = _inputs(tmp_path)
    work = _progress_path(kwargs).parent
    work.mkdir(parents=True)
    _progress_path(kwargs).write_bytes(b'"unterminated')
    kwargs["resume"] = True

    frame = cli.run_unshielded_comparator(**kwargs)

    assert len(frame) == 32 and len(calls) == 32


def test_checkpoint_steps_must_be_a_nonnegative_exact_integer(tmp_path):
    kwargs, _, _ = _inputs(tmp_path)
    kwargs["model_loader"] = lambda *args: SimpleNamespace(num_timesteps=17.5)
    with pytest.raises(ValueError, match="checkpoint steps"):
        cli.run_unshielded_comparator(**kwargs)


@pytest.mark.parametrize("value", [True, np.bool_(True)])
def test_checkpoint_steps_reject_boolean_values(value):
    with pytest.raises(ValueError, match="checkpoint steps"):
        cli._checkpoint_steps(SimpleNamespace(num_timesteps=value))


def test_checkpoint_steps_preserve_large_integer_exactly():
    value = 9_007_199_254_740_993
    assert cli._checkpoint_steps(SimpleNamespace(num_timesteps=value)) == value


def test_runtime_fingerprint_includes_comparator_and_context_runner(tmp_path):
    source = tmp_path / "src" / "gl_gym"
    scripts = tmp_path / "experiments" / "scripts"
    source.mkdir(parents=True)
    scripts.mkdir(parents=True)
    (source / "runtime.py").write_text("VALUE = 1\n", encoding="utf-8")
    comparator = scripts / "run_unshielded_context_comparator.py"
    context_runner = scripts / "run_context_ab.py"
    comparator.write_text("COMPARATOR = 1\n", encoding="utf-8")
    context_runner.write_text("CONTEXT = 1\n", encoding="utf-8")

    original = cli._runtime_source_tree_sha256(tmp_path)
    comparator.write_text("COMPARATOR = 2\n", encoding="utf-8")
    comparator_changed = cli._runtime_source_tree_sha256(tmp_path)
    comparator.write_text("COMPARATOR = 1\n", encoding="utf-8")
    context_runner.write_text("CONTEXT = 2\n", encoding="utf-8")
    context_changed = cli._runtime_source_tree_sha256(tmp_path)

    assert original != comparator_changed
    assert original != context_changed


@pytest.mark.parametrize("lifecycle", ["resume", "legacy"])
def test_runtime_entrypoint_change_makes_saved_evidence_stale(
    tmp_path, monkeypatch, lifecycle
):
    fingerprint = ["a" * 64]
    monkeypatch.setattr(cli, "_runtime_source_tree_sha256", lambda: fingerprint[0])
    old, calls, _ = _inputs(tmp_path / "old")
    old_frame = cli.run_unshielded_comparator(**old)
    calls.clear()
    fingerprint[0] = "b" * 64

    if lifecycle == "resume":
        old["resume"] = True
        cli.run_unshielded_comparator(**old)
    else:
        legacy = _progress_path(old)
        _legacy_row(old_frame).to_csv(legacy, index=False)
        new, calls, _ = _inputs(tmp_path / "new")
        new["legacy_progress"] = legacy
        cli.run_unshielded_comparator(**new)

    assert len(calls) == 32


def test_native_success_rejects_nonfinite_optional_scoring_metric(tmp_path):
    kwargs, _, _ = _inputs(tmp_path)

    def runner(*args, **runner_kwargs):
        metrics = {name: 1.0 for name in cli.REQUIRED_METRICS}
        metrics["EPI"] = np.inf
        return metrics, {
            "support_ready_step": 1.0,
            "context_norm_mean": 0.5,
            "context_norm_max": 1.0,
            "action_trace": np.ones((3, 2)),
        }

    kwargs["episode_runner"] = runner
    with pytest.raises(ValueError, match="scoring metric"):
        cli.run_unshielded_comparator(**kwargs)


def test_legacy_import_copies_valid_trace_and_resigns_identity(tmp_path):
    old, calls, _ = _inputs(tmp_path / "old")
    old_frame = cli.run_unshielded_comparator(**old)
    legacy = _progress_path(old)
    legacy_row = _legacy_row(pd.read_csv(legacy))
    legacy_row.to_csv(legacy, index=False)
    old_trace = Path(legacy_row.iloc[0].action_trace_path)

    calls.clear()
    new = dict(old)
    new["result_root"] = tmp_path / "new" / "comparator-final"
    new["failure_root"] = tmp_path / "new" / "failure-root"
    new["legacy_progress"] = legacy
    frame = cli.run_unshielded_comparator(**new)

    imported = frame.iloc[0]
    assert len(calls) == 31
    assert Path(imported.action_trace_path) != old_trace
    assert Path(imported.action_trace_path) == (
        Path(new["result_root"]) / "traces"
        / f"seed{int(imported.seed)}__{imported.task_id}__{imported.inference_mode}.npy"
    ).resolve()
    assert Path(imported.action_trace_path).read_bytes() == old_trace.read_bytes()
    assert len(imported.row_identity_sha256) == 64

    calls.clear()
    new["resume"] = True
    new["legacy_progress"] = None
    cli.run_unshielded_comparator(**new)
    assert calls == []


def test_legacy_import_rejects_stale_runtime_source_tree(tmp_path):
    old, _, _ = _inputs(tmp_path / "old")
    frame = cli.run_unshielded_comparator(**old)
    legacy = _progress_path(old)
    row = _legacy_row(frame)
    row["runtime_source_tree_sha256"] = "f" * 64
    row.to_csv(legacy, index=False)

    new, calls, _ = _inputs(tmp_path / "new")
    new["legacy_progress"] = legacy
    cli.run_unshielded_comparator(**new)
    assert len(calls) == 32


def test_real_legacy_schema_without_runtime_fingerprint_is_recomputed(tmp_path):
    old, _, _ = _inputs(tmp_path / "old")
    frame = cli.run_unshielded_comparator(**old)
    legacy = _progress_path(old)
    row = _legacy_row(frame).drop(columns=["runtime_source_tree_sha256"])
    row.to_csv(legacy, index=False)

    new, calls, _ = _inputs(tmp_path / "new")
    new["legacy_progress"] = legacy
    cli.run_unshielded_comparator(**new)
    assert len(calls) == 32


@pytest.mark.parametrize("foreign", ["key", "root"])
def test_legacy_import_rejects_noncanonical_trace_source(tmp_path, foreign):
    old, _, _ = _inputs(tmp_path / "old")
    frame = cli.run_unshielded_comparator(**old)
    legacy = _progress_path(old)
    row = _legacy_row(frame)
    if foreign == "key":
        foreign_trace = Path(frame.iloc[1].action_trace_path)
    else:
        source_trace = Path(frame.iloc[0].action_trace_path)
        foreign_trace = tmp_path / "foreign-work" / "traces" / source_trace.name
        foreign_trace.parent.mkdir(parents=True)
        foreign_trace.write_bytes(source_trace.read_bytes())
    row.loc[row.index[0], "action_trace_path"] = str(foreign_trace)
    row.to_csv(legacy, index=False)

    new, calls, _ = _inputs(tmp_path / "new")
    new["legacy_progress"] = legacy
    cli.run_unshielded_comparator(**new)
    assert len(calls) == 32


def test_legacy_import_rejects_nonfinite_optional_scoring_metric(tmp_path):
    old, _, _ = _inputs(tmp_path / "old")
    frame = cli.run_unshielded_comparator(**old)
    legacy = _progress_path(old)
    row = _legacy_row(frame)
    row["revenue"] = np.inf
    row.to_csv(legacy, index=False)

    new, calls, _ = _inputs(tmp_path / "new")
    new["legacy_progress"] = legacy
    cli.run_unshielded_comparator(**new)
    assert len(calls) == 32


def test_legacy_import_rejects_inferred_or_unproven_rows(tmp_path):
    old, _, _ = _inputs(tmp_path / "old")
    frame = cli.run_unshielded_comparator(**old)
    legacy = _progress_path(old)
    invalid = _legacy_row(frame).drop(columns=["inference_mode"])
    invalid.to_csv(legacy, index=False)

    new, calls, _ = _inputs(tmp_path / "new")
    new["legacy_progress"] = legacy
    cli.run_unshielded_comparator(**new)

    assert len(calls) == 32


def test_legacy_import_rejects_non_fail_fast_status_rows(tmp_path):
    old, _, _ = _inputs(tmp_path / "old")
    row = cli.run_unshielded_comparator(**old).iloc[[0]].copy()
    row["EPI"] = 1.0
    legacy = _progress_path(old)
    row.to_csv(legacy, index=False)

    new, calls, _ = _inputs(tmp_path / "new")
    new["legacy_progress"] = legacy
    cli.run_unshielded_comparator(**new)

    assert len(calls) == 32

    kwargs, _, _ = _inputs(tmp_path / "second")
    kwargs["failure_root"] = kwargs["result_root"] / "failures"
    with pytest.raises(ValueError, match="disjoint"):
        cli.run_unshielded_comparator(**kwargs)
