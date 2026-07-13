import subprocess
import sys
from dataclasses import FrozenInstanceError, replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import gl_gym.experiments.suite_evaluation as suite_evaluation
from gl_gym.experiments.suite_schema import create_default_suite_config, write_suite_manifest
from gl_gym.experiments.suite_evaluation import (
    EvaluationMetricRow,
    append_eval_raw,
    completed_eval_keys,
    evaluation_key,
    run_deterministic_episode,
    validate_completed_run_paths,
    write_eval_raw,
)


def load_evaluate_suite_module():
    script_path = Path(__file__).resolve().parents[2] / "experiments" / "scripts" / "evaluate_suite.py"
    spec = __import__("importlib.util").util.spec_from_file_location("evaluate_suite", script_path)
    module = __import__("importlib.util").util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class FakeModel:
    def predict(self, obs, deterministic=True):
        return np.array([[0.0]]), None


class HookedFakeModel(FakeModel):
    def __init__(self, fail_predict=False):
        self.events = []
        self.fail_predict = fail_predict
        self.predict_count = 0

    def begin_inference_episode(self, mode):
        self.events.append(("begin", mode))

    def predict(self, obs, deterministic=True, **kwargs):
        self.events.append(("predict", np.asarray(obs).copy()))
        if self.fail_predict:
            raise RuntimeError("predict failed")
        action = np.array([[float(self.predict_count)]])
        self.predict_count += 1
        return action, None

    def observe_inference_transition(
        self, observation, action, reward, next_observation, done, info
    ):
        self.events.append(
            (
                "observe",
                np.asarray(observation).copy(),
                np.asarray(action).copy(),
                float(reward),
                np.asarray(next_observation).copy(),
                bool(done),
                info,
            )
        )

    def inference_episode_diagnostics(self):
        self.events.append(("diagnostics",))
        return {
            "support_ready_step": 1.0,
            "context_norm_mean": 2.0,
            "context_norm_max": 3.0,
        }

    def end_inference_episode(self):
        self.events.append(("end",))


class FakeEnv:
    def __init__(self):
        self.step_count = 0

    def get_attr(self, name):
        if name == "N":
            return [3]
        raise AttributeError(name)

    def reset(self):
        return np.array([[0.0]])

    def step(self, actions):
        self.step_count += 1
        info = {
            "EPI": 1.0,
            "revenue": 2.0,
            "heat_cost": 0.3,
            "co2_cost": 0.2,
            "elec_cost": 0.1,
            "temp_violation": 1,
            "co2_violation": 2,
            "rh_violation": 3,
        }
        done = self.step_count == 3
        if done:
            info["terminal_observation"] = np.array([0.0])
        return np.array([[0.0]]), np.array([10.0]), np.array([done]), [info]


class DiagnosticFakeEnv(FakeEnv):
    def __init__(self, *, fail_enable=False, fail_disable=False):
        super().__init__()
        self.diagnostic_calls = []
        self.fail_enable = fail_enable
        self.fail_disable = fail_disable

    def env_method(self, name, enabled):
        self.diagnostic_calls.append((name, enabled))
        if enabled and self.fail_enable:
            raise RuntimeError("enable failed")
        if not enabled and self.fail_disable:
            raise RuntimeError("disable failed")


class RecordingFailureRecorder:
    def __init__(self, error=None):
        self.calls = []
        self.error = error

    def record_step(self, **kwargs):
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error


def test_run_deterministic_episode_sums_metrics():
    metrics = run_deterministic_episode(FakeModel(), FakeEnv())

    assert metrics["episode_return"] == 30.0
    assert metrics["EPI"] == 3.0
    assert metrics["revenue"] == 6.0
    assert metrics["temp_violation"] == 3
    assert metrics["co2_violation"] == 6
    assert metrics["rh_violation"] == 9


def test_failure_recorder_enables_diagnostics_and_captures_original_transition():
    class TransitionEnv(DiagnosticFakeEnv):
        def reset(self):
            return np.array([[1.0, 2.0]], dtype=np.float32)

        def step(self, actions):
            self.step_count += 1
            info = {
                "integration_failure": {"message": "exact"},
                "diagnostic_transition": {"step": self.step_count},
            }
            done = self.step_count == 3
            if done:
                info["terminal_observation"] = np.array([9.0, 10.0])
            return (
                np.array([[10.0 + self.step_count, 20.0]], dtype=np.float32),
                np.array([self.step_count], dtype=np.float32),
                np.array([done]),
                [info],
            )

    env = TransitionEnv()
    recorder = RecordingFailureRecorder()
    run_deterministic_episode(FakeModel(), env, failure_recorder=recorder)

    assert env.diagnostic_calls == [
        ("set_ode_diagnostics_enabled", True),
        ("set_ode_diagnostics_enabled", False),
    ]
    assert len(recorder.calls) == 3
    first = recorder.calls[0]
    assert first["step_index"] == 0
    np.testing.assert_array_equal(first["policy_observation"], [1.0, 2.0])
    assert first["reward"] == 1.0
    assert first["done"] is False
    assert first["info"]["integration_failure"] == {"message": "exact"}
    assert first["info"]["diagnostic_transition"] == {"step": 1}


def test_default_episode_does_not_toggle_ode_diagnostics():
    env = DiagnosticFakeEnv()
    run_deterministic_episode(FakeModel(), env)
    assert env.diagnostic_calls == []


def test_nonterminal_capture_error_propagates_before_next_prediction():
    model = HookedFakeModel()
    env = DiagnosticFakeEnv()
    recorder = RecordingFailureRecorder(RuntimeError("capture failed"))

    with pytest.raises(RuntimeError, match="capture failed"):
        run_deterministic_episode(
            model, env, inference_mode="online_context", failure_recorder=recorder
        )

    assert [event[0] for event in model.events] == ["begin", "predict", "end"]
    assert env.step_count == 1
    assert env.diagnostic_calls[-1] == ("set_ode_diagnostics_enabled", False)


def test_early_termination_remains_primary_when_capture_fails():
    class EarlyEnv(DiagnosticFakeEnv):
        def step(self, actions):
            self.step_count += 1
            return (
                np.array([[8.0]]),
                np.array([1.0]),
                np.array([True]),
                [{}],
            )

    with pytest.raises(RuntimeError, match="terminated before") as captured:
        run_deterministic_episode(
            HookedFakeModel(),
            EarlyEnv(),
            inference_mode="online_context",
            failure_recorder=RecordingFailureRecorder(
                RuntimeError("capture failed")
            ),
        )

    assert any("capture failed" in note for note in captured.value.__notes__)


def test_enable_failure_avoids_disable_and_still_ends_inference_episode():
    model = HookedFakeModel()
    env = DiagnosticFakeEnv(fail_enable=True)

    with pytest.raises(RuntimeError, match="enable failed"):
        run_deterministic_episode(
            model,
            env,
            inference_mode="online_context",
            failure_recorder=RecordingFailureRecorder(),
        )

    assert env.diagnostic_calls == [("set_ode_diagnostics_enabled", True)]
    assert [event[0] for event in model.events] == ["begin", "end"]


def test_primary_error_keeps_priority_over_disable_and_inference_cleanup_errors():
    class CleanupFailModel(HookedFakeModel):
        def end_inference_episode(self):
            self.events.append(("end",))
            raise RuntimeError("end failed")

    model = CleanupFailModel(fail_predict=True)
    env = DiagnosticFakeEnv(fail_disable=True)
    with pytest.raises(RuntimeError, match="predict failed") as captured:
        run_deterministic_episode(
            model,
            env,
            inference_mode="online_context",
            failure_recorder=RecordingFailureRecorder(),
        )

    assert any("disable failed" in note for note in captured.value.__notes__)
    assert any("end failed" in note for note in captured.value.__notes__)


def test_disable_error_is_primary_and_inference_cleanup_error_is_noted():
    class CleanupFailModel(HookedFakeModel):
        def end_inference_episode(self):
            self.events.append(("end",))
            raise RuntimeError("end failed")

    with pytest.raises(RuntimeError, match="disable failed") as captured:
        run_deterministic_episode(
            CleanupFailModel(),
            DiagnosticFakeEnv(fail_disable=True),
            inference_mode="online_context",
            failure_recorder=RecordingFailureRecorder(),
        )

    assert any("end failed" in note for note in captured.value.__notes__)


def test_deterministic_episode_invokes_online_hooks_in_lifecycle_order():
    model = HookedFakeModel()

    metrics, diagnostics = run_deterministic_episode(
        model,
        FakeEnv(),
        inference_mode="online_context",
        return_diagnostics=True,
    )

    assert [event[0] for event in model.events] == [
        "begin",
        "predict",
        "observe",
        "predict",
        "observe",
        "predict",
        "observe",
        "diagnostics",
        "end",
    ]
    assert model.events[0] == ("begin", "online_context")
    assert metrics["episode_return"] == 30.0
    assert diagnostics["context_norm_max"] == 3.0
    np.testing.assert_array_equal(
        diagnostics["action_trace"],
        np.array([[0.0], [1.0], [2.0]], dtype=np.float32),
    )
    assert diagnostics["action_trace"].dtype == np.float32


def test_deterministic_episode_passes_executed_transition_and_terminal_observation():
    class TransitionEnv(FakeEnv):
        def reset(self):
            return np.array([[1.0, 2.0]], dtype=np.float32)

        def step(self, actions):
            self.step_count += 1
            done = self.step_count == 3
            info = {"marker": self.step_count}
            if done:
                info["terminal_observation"] = np.array([99.0, 100.0], dtype=np.float32)
            post_step_obs = np.array(
                [[10.0 + self.step_count, 20.0 + self.step_count]], dtype=np.float32
            )
            return post_step_obs, np.array([self.step_count]), np.array([done]), [info]

    model = HookedFakeModel()
    run_deterministic_episode(model, TransitionEnv(), inference_mode="online_context")
    observations = [event for event in model.events if event[0] == "observe"]

    np.testing.assert_array_equal(observations[0][1], np.array([1.0, 2.0]))
    np.testing.assert_array_equal(observations[0][2], np.array([0.0]))
    assert observations[0][3] == 1.0
    np.testing.assert_array_equal(observations[0][4], np.array([11.0, 21.0]))
    assert observations[0][5] is False
    assert observations[0][6]["marker"] == 1

    np.testing.assert_array_equal(observations[-1][1], np.array([12.0, 22.0]))
    np.testing.assert_array_equal(observations[-1][2], np.array([2.0]))
    assert observations[-1][3] == 3.0
    np.testing.assert_array_equal(observations[-1][4], np.array([99.0, 100.0]))
    assert observations[-1][5] is True
    assert observations[-1][6]["marker"] == 3


def test_deterministic_episode_uses_post_step_observation_before_terminal_step():
    model = HookedFakeModel()
    run_deterministic_episode(model, FakeEnv(), inference_mode="online_context")

    first_observe = [event for event in model.events if event[0] == "observe"][0]
    np.testing.assert_array_equal(first_observe[4], np.array([0.0]))
    assert first_observe[5] is False


def test_deterministic_episode_rejects_termination_before_configured_horizon():
    class EarlyTerminationEnv(FakeEnv):
        def step(self, actions):
            self.step_count += 1
            info = {
                "terminal_observation": np.array([9.0], dtype=np.float32),
            }
            return (
                np.array([[0.0]], dtype=np.float32),
                np.array([1.0], dtype=np.float32),
                np.array([True]),
                [info],
            )

    env = EarlyTerminationEnv()
    model = HookedFakeModel()

    with pytest.raises(RuntimeError, match="terminated before configured horizon"):
        run_deterministic_episode(model, env, inference_mode="online_context")

    assert env.step_count == 1
    assert model.events[-1] == ("end",)


@pytest.mark.parametrize("terminal_observation", ["missing", None])
def test_terminal_step_requires_usable_terminal_observation_and_cleans_up(
    terminal_observation,
):
    class InvalidTerminalEnv(FakeEnv):
        def step(self, actions):
            obs, rewards, dones, infos = super().step(actions)
            if bool(dones[0]):
                if terminal_observation == "missing":
                    infos[0].pop("terminal_observation")
                else:
                    infos[0]["terminal_observation"] = None
            return obs, rewards, dones, infos

    model = HookedFakeModel()

    with pytest.raises(ValueError, match="terminal_observation"):
        run_deterministic_episode(
            model, InvalidTerminalEnv(), inference_mode="online_context"
        )

    assert [event[0] for event in model.events].count("observe") == 2
    assert model.events[-1] == ("end",)


def test_deterministic_episode_cleans_up_when_prediction_fails():
    model = HookedFakeModel(fail_predict=True)

    with pytest.raises(RuntimeError, match="predict failed"):
        run_deterministic_episode(model, FakeEnv(), inference_mode="online_context")

    assert [event[0] for event in model.events] == ["begin", "predict", "end"]


def test_predict_internal_type_error_is_not_retried_or_masked():
    class InternalTypeErrorModel:
        def __init__(self):
            self.calls = 0

        def predict(
            self, obs, state=None, episode_start=None, deterministic=True
        ):
            self.calls += 1
            raise TypeError("model internals broke")

    model = InternalTypeErrorModel()

    with pytest.raises(TypeError, match="model internals broke"):
        run_deterministic_episode(model, FakeEnv())

    assert model.calls == 1


def test_begin_failure_still_ends_partially_initialized_episode():
    class BeginFailureModel(HookedFakeModel):
        def begin_inference_episode(self, mode):
            self.events.append(("begin", mode))
            raise RuntimeError("begin failed")

    model = BeginFailureModel()

    with pytest.raises(RuntimeError, match="begin failed"):
        run_deterministic_episode(model, FakeEnv(), inference_mode="online_context")

    assert [event[0] for event in model.events] == ["begin", "end"]


def test_prediction_error_remains_primary_when_cleanup_also_fails():
    class DoubleFailureModel(HookedFakeModel):
        def end_inference_episode(self):
            self.events.append(("end",))
            raise RuntimeError("end failed")

    model = DoubleFailureModel(fail_predict=True)

    with pytest.raises(RuntimeError, match="predict failed"):
        run_deterministic_episode(model, FakeEnv(), inference_mode="online_context")

    assert model.events[-1] == ("end",)


def test_cleanup_error_propagates_after_successful_episode():
    class EndFailureModel(HookedFakeModel):
        def end_inference_episode(self):
            self.events.append(("end",))
            raise RuntimeError("end failed")

    model = EndFailureModel()

    with pytest.raises(RuntimeError, match="end failed"):
        run_deterministic_episode(model, FakeEnv(), inference_mode="online_context")

    assert model.events[-1] == ("end",)


def test_explicit_inference_mode_lists_missing_model_hooks():
    class ModelMissingHooks(FakeModel):
        def begin_inference_episode(self, mode):
            pass

    with pytest.raises(TypeError) as exc_info:
        run_deterministic_episode(
            ModelMissingHooks(), FakeEnv(), inference_mode="online_context"
        )

    message = str(exc_info.value)
    assert "observe_inference_transition" in message
    assert "inference_episode_diagnostics" in message
    assert "end_inference_episode" in message


def test_plain_model_retains_existing_evaluation_contract():
    metrics = run_deterministic_episode(FakeModel(), FakeEnv())

    assert isinstance(metrics, dict)
    assert metrics["episode_return"] == 30.0


def test_plain_model_can_request_generic_action_trace_diagnostics():
    metrics, diagnostics = run_deterministic_episode(
        FakeModel(), FakeEnv(), return_diagnostics=True
    )

    assert metrics["episode_return"] == 30.0
    assert set(diagnostics) == {"action_trace"}
    np.testing.assert_array_equal(
        diagnostics["action_trace"], np.zeros((3, 1), dtype=np.float32)
    )


def test_write_eval_raw_has_one_row_per_run_task(tmp_path: Path):
    rows = [
        EvaluationMetricRow(
            suite_id="suite",
            algorithm="ppo",
            seed=42,
            run_name="ppo_seed42",
            task_id="fixed_2010_d59_u0p00_standard",
            split="fixed",
            weather_year=2010,
            start_day=59,
            uncertainty_scale=0.0,
            economic_scenario="standard",
            climate_constraint_scenario="standard",
            episode_return=30.0,
            EPI=3.0,
            revenue=6.0,
            heat_cost=0.9,
            co2_cost=0.6,
            elec_cost=0.3,
            temp_violation=3.0,
            co2_violation=6.0,
            rh_violation=9.0,
            twb_percent=float("nan"),
            trajectory_path="",
        )
    ]

    out = write_eval_raw(rows, tmp_path / "eval_raw.csv")
    df = pd.read_csv(out)

    assert len(df) == 1
    assert df.loc[0, "algorithm"] == "ppo"
    assert df.loc[0, "task_id"] == "fixed_2010_d59_u0p00_standard"
    assert df.loc[0, "climate_constraint_scenario"] == "standard"


def test_append_eval_raw_preserves_existing_rows_and_header(tmp_path: Path):
    first = EvaluationMetricRow(
        suite_id="suite",
        algorithm="ppo",
        seed=42,
        run_name="ppo_seed42",
        task_id="fixed",
        split="fixed",
        weather_year=2010,
        start_day=59,
        uncertainty_scale=0.0,
        economic_scenario="standard",
        climate_constraint_scenario="standard",
        episode_return=30.0,
        EPI=3.0,
        revenue=6.0,
        heat_cost=0.9,
        co2_cost=0.6,
        elec_cost=0.3,
        temp_violation=3.0,
        co2_violation=6.0,
        rh_violation=9.0,
        twb_percent=float("nan"),
        trajectory_path="",
    )
    second = replace(first, seed=123, run_name="ppo_seed123")

    out = tmp_path / "eval_raw.csv"
    append_eval_raw(first, out)
    append_eval_raw(second, out)
    df = pd.read_csv(out)

    assert len(df) == 2
    assert df["seed"].tolist() == [42, 123]


def test_completed_eval_keys_identifies_existing_run_task_pairs(tmp_path: Path):
    row = EvaluationMetricRow(
        suite_id="suite",
        algorithm="ppo",
        seed=42,
        run_name="ppo_seed42",
        task_id="fixed",
        split="fixed",
        weather_year=2010,
        start_day=59,
        uncertainty_scale=0.0,
        economic_scenario="standard",
        climate_constraint_scenario="standard",
        episode_return=30.0,
        EPI=3.0,
        revenue=6.0,
        heat_cost=0.9,
        co2_cost=0.6,
        elec_cost=0.3,
        temp_violation=3.0,
        co2_violation=6.0,
        rh_violation=9.0,
        twb_percent=float("nan"),
        trajectory_path="",
    )
    out = append_eval_raw(row, tmp_path / "eval_raw.csv")

    assert completed_eval_keys(out) == {evaluation_key("ppo", 42, "fixed")}


def test_evaluation_metric_row_is_frozen_and_slotted():
    row = EvaluationMetricRow(
        suite_id="suite",
        algorithm="ppo",
        seed=42,
        run_name="ppo_seed42",
        task_id="fixed_2010_d59_u0p00_standard",
        split="fixed",
        weather_year=2010,
        start_day=59,
        uncertainty_scale=0.0,
        economic_scenario="standard",
        climate_constraint_scenario="standard",
        episode_return=30.0,
        EPI=3.0,
        revenue=6.0,
        heat_cost=0.9,
        co2_cost=0.6,
        elec_cost=0.3,
        temp_violation=3.0,
        co2_violation=6.0,
        rh_violation=9.0,
        twb_percent=float("nan"),
        trajectory_path="",
    )

    assert hasattr(EvaluationMetricRow, "__slots__")

    try:
        row.algorithm = "recurrentppo"
    except FrozenInstanceError:
        pass
    else:
        raise AssertionError("EvaluationMetricRow should be frozen")


def test_validate_completed_run_paths_fails_fast_for_missing_artifacts(tmp_path: Path):
    run = SimpleNamespace(
        status="completed",
        model_path=str(tmp_path / "missing_model.zip"),
        vecnormalize_path=str(tmp_path / "missing_vecnormalize.pkl"),
    )

    with pytest.raises(FileNotFoundError, match="model_path does not exist"):
        validate_completed_run_paths(run)


def test_validate_completed_run_paths_ignores_dry_run_missing_artifacts(tmp_path: Path):
    run = SimpleNamespace(
        status="dry_run",
        model_path=str(tmp_path / "missing_model.zip"),
        vecnormalize_path=str(tmp_path / "missing_vecnormalize.pkl"),
    )

    validate_completed_run_paths(run)


def test_load_task_env_closes_base_env_when_vecnormalize_load_fails(
    tmp_path: Path, monkeypatch
):
    import gl_gym.RL.utils as rl_utils
    import gl_gym.common.utils as common_utils
    import gl_gym.experiments.suite_tasks as suite_tasks
    from stable_baselines3.common.vec_env import VecNormalize

    class BaseEnv:
        closed = False

        def close(self):
            self.closed = True

    env = BaseEnv()
    vec_path = tmp_path / "vec.pkl"
    vec_path.write_bytes(b"vec")
    monkeypatch.setattr(common_utils, "load_env_params", lambda *args: ({}, {}))
    monkeypatch.setattr(
        suite_tasks, "apply_task_to_env_params", lambda base, specific, task: (base, specific)
    )
    monkeypatch.setattr(rl_utils, "make_vec_env", lambda *args, **kwargs: env)

    def fail_load(path, base_env):
        assert base_env is env
        raise RuntimeError("invalid vecnormalize")

    monkeypatch.setattr(VecNormalize, "load", fail_load)
    with pytest.raises(RuntimeError, match="invalid vecnormalize"):
        suite_evaluation.load_task_env(
            SimpleNamespace(env_id="Fake"), SimpleNamespace(), vec_path
        )
    assert env.closed


def test_evaluate_suite_filters_tasks_for_smoke_runs():
    module = load_evaluate_suite_module()
    tasks = pd.DataFrame(
        [
            {"task_id": "fixed_2010_d59_u0p00_standard", "split": "fixed"},
            {"task_id": "heldout_2011_d59_u0p00_standard", "split": "heldout"},
            {"task_id": "economic_2011_d59_u0p00_high_energy_price", "split": "economic"},
        ]
    )

    filtered = module.filter_tasks(tasks, splits=["heldout", "economic"], limit_tasks=1)

    assert filtered["task_id"].tolist() == ["heldout_2011_d59_u0p00_standard"]


def test_evaluate_suite_dry_run_only_writes_clear_message_without_eval_raw(tmp_path: Path):
    result_root = tmp_path / "results"
    suite = create_default_suite_config(result_root=result_root, model_root=tmp_path / "models")
    manifest = write_suite_manifest(suite, result_root / "suite_manifest.json")
    runs_csv = tmp_path / "runs.csv"
    tasks_csv = tmp_path / "eval_tasks.csv"

    pd.DataFrame(
        [
            {
                "suite_id": suite.suite_id,
                "algorithm": "ppo",
                "seed": 42,
                "run_name": "ppo_seed42",
                "model_path": str(tmp_path / "missing_model.zip"),
                "vecnormalize_path": str(tmp_path / "missing_vecnormalize.pkl"),
                "status": "dry_run",
                "train_steps": 0,
                "wall_time_seconds": 0.0,
                "best_eval_return": float("nan"),
                "notes": "dry-run registry entry",
            }
        ]
    ).to_csv(runs_csv, index=False)
    pd.DataFrame(
        [
            {
                "suite_id": suite.suite_id,
                "task_id": "fixed_2010_d59_u0p00_standard",
                "split": "fixed",
                "weather_year": 2010,
                "start_day": 59,
                "uncertainty_scale": 0.0,
                "economic_scenario": "standard",
                "climate_constraint_scenario": "standard",
            }
        ]
    ).to_csv(tasks_csv, index=False)

    result = subprocess.run(
        [
            sys.executable,
            "experiments/scripts/evaluate_suite.py",
            "--manifest",
            str(manifest),
            "--runs_csv",
            str(runs_csv),
            "--tasks_csv",
            str(tasks_csv),
        ],
        cwd=Path(__file__).resolve().parents[2],
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0
    assert "No completed runs to evaluate; eval_raw.csv was not written." in result.stdout
    assert not (result_root / "eval_raw.csv").exists()
