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
    recorder = _RECORDER_BY_ROOT[path.parent]
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
            "content_identity_sha256": "a" * 64,
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
        metrics = {name: float(index + 1) for index, name in enumerate(cli.REQUIRED_METRICS)}
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


def test_injected_successes_produce_exact_32_complete_rows_and_only_work_output(tmp_path):
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
    assert all(Path(path).is_file() for path in frame.action_trace_path)
    assert not Path(kwargs["result_root"]).exists()
    assert (Path(kwargs["result_root"]).parent / ".comparator-final.work" / "progress.csv").is_file()
    assert len(calls) == 32 and all(env.closed for env in envs)


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
    assert failed[list(cli.REQUIRED_METRICS)].isna().all(axis=None)
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


def test_output_roots_must_be_pairwise_disjoint(tmp_path):
    kwargs, _, _ = _inputs(tmp_path)
    kwargs["result_root"] = kwargs["suite"].result_root / "child"
    with pytest.raises(ValueError, match="disjoint"):
        cli.run_unshielded_comparator(**kwargs)

    kwargs, _, _ = _inputs(tmp_path / "second")
    kwargs["failure_root"] = kwargs["result_root"] / "failures"
    with pytest.raises(ValueError, match="disjoint"):
        cli.run_unshielded_comparator(**kwargs)
