from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest


SCRIPT = Path(__file__).parents[2] / "experiments" / "scripts" / "run_shield_stage1.py"
SPEC = importlib.util.spec_from_file_location("run_shield_stage1_cli", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
cli = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(cli)


def _config(path: Path, *, delta=0.2) -> Path:
    path.write_text(
        "GreenLightEnv:\n"
        "  u_min: [0.0, 0.0]\n"
        "  u_max: [1.0, 1.0]\n"
        f"  delta_u_max: {delta}\n",
        encoding="utf-8",
    )
    return path


def _capsule(tmp_path: Path):
    formal = (tmp_path / "formal").resolve()
    capsule_dir = (tmp_path / "capsules" / "failure-1").resolve()
    capsule_dir.mkdir(parents=True)
    (capsule_dir / "manifest.json").write_text("{}", encoding="utf-8")
    weather = np.array([10.0, 11.0])
    params = np.array([12.0])
    requested = np.array([1.0, -1.0], dtype=np.float32)
    previous = np.array([0.4, 0.6], dtype=np.float32)
    delta = np.ones(2, dtype=np.float32) * 0.2
    control = np.clip(
        previous + requested * delta,
        np.array([0.0, 0.0], dtype=np.float32),
        np.array([1.0, 1.0], dtype=np.float32),
    )
    return SimpleNamespace(
        path=capsule_dir,
        manifest={
            "failure_id": "failure-1",
            "content_identity_sha256": "a" * 64,
            "checkpoint_path": "models/agent.zip",
            "checkpoint_sha256": "b" * 64,
            "source_checksums": {"tomato_env.py": "c" * 64},
            "git_head": "d" * 40,
            "dirty": True,
            "solver": {"options": dict(cli.FORMAL_CVODES_OPTIONS)},
            "context": {
                "formal_result_root": str(formal),
                "checkpoint_path": "models/agent.zip",
                "checkpoint_sha256": "b" * 64,
                "source_checksums": {"tomato_env.py": "c" * 64},
                "git_head": "d" * 40,
                "dirty": True,
            },
        },
        failure_inputs={
            "x0": np.array([1.0, 2.0, 3.0]),
            "u": control,
            "previous_control": previous,
            "requested_action": requested,
            "weather": weather,
            "sampled_parameters": params,
            "p_dyn": np.concatenate((weather, params)),
            "timestep": np.array(4),
            "day_of_year": np.array(151.0),
            "hour_of_day": np.array(12.5),
            "dt": np.array(300.0),
            "nx": np.array(3),
            "nu": np.array(2),
            "nd": np.array(2),
            "n_params": np.array(1),
        },
    )


def test_parser_and_bootstrap_work_outside_repository(tmp_path):
    parser = cli.build_parser()
    assert isinstance(parser, argparse.ArgumentParser)
    with pytest.raises(SystemExit):
        parser.parse_args([])
    completed = subprocess.run(
        [sys.executable, "-I", str(SCRIPT), "--help"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert "--capsule_manifest" in completed.stdout
    assert "--env_config" in completed.stdout
    assert "--output_root" in completed.stdout


def test_failure_then_fixed_candidates_selects_first_success(tmp_path):
    capsule = _capsule(tmp_path)
    config = _config(tmp_path / "env.yml")
    factory_calls = []
    integrator_calls = []
    outcomes = [
        RuntimeError("original"),
        RuntimeError("c1"),
        RuntimeError("c2"),
        np.array([7.0, 8.0, 9.0]),
    ]

    def factory(**kwargs):
        factory_calls.append(kwargs)
        outcome = outcomes[len(factory_calls) - 1]

        def integrate(**inputs):
            integrator_calls.append(
                {key: np.array(value, copy=True) for key, value in inputs.items()}
            )
            if isinstance(outcome, Exception):
                raise outcome
            return {"xf": outcome.copy()}

        return integrate

    class Controller:
        def predict(self, x, weather, env):
            assert env.nu == 2 and env.day_of_year == 151.0 and env.hour_of_day == 12.5
            x[:] = -99
            weather[:] = -99
            return np.array([0.4, 0.6])

    output = tmp_path / "stage1"
    result = cli.run_stage1(
        capsule.path / "manifest.json",
        config,
        output,
        capsule_loader=lambda path: capsule,
        integrator_factory=factory,
        controller_factory=lambda: Controller(),
    )

    assert result == output.resolve()
    assert len(factory_calls) == 4
    assert all(
        call["nx"] == 3 and call["nu"] == 2 and call["nd"] == 2
        for call in factory_calls
    )
    assert all(call["n_params"] == 1 and call["dt"] == 300.0 for call in factory_calls)
    assert all(
        call["integrator_options"] == dict(cli.FORMAL_CVODES_OPTIONS)
        for call in factory_calls
    )
    assert all(np.array_equal(call["x0"], [1.0, 2.0, 3.0]) for call in integrator_calls)
    assert all(
        np.array_equal(call["p"], [10.0, 11.0, 12.0]) for call in integrator_calls
    )
    previous = np.array([0.4, 0.6], dtype=np.float32)
    delta = np.ones(2, dtype=np.float32) * 0.2
    u_min = np.array([0.0, 0.0], dtype=np.float32)
    u_max = np.array([1.0, 1.0], dtype=np.float32)
    requested = np.array([1.0, -1.0], dtype=np.float32)
    reference = cli.control_to_reference_action(np.array([0.4, 0.6]), previous, delta)
    expected_actions = [
        (1.0 - lam) * requested.astype(np.float64) + lam * reference
        for lam in cli.DEFAULT_LAMBDAS[:3]
    ]
    expected_controls = [
        np.clip(previous + action * delta, u_min, u_max) for action in expected_actions
    ]
    assert all(
        np.array_equal(call["u"], expected)
        for call, expected in zip(integrator_calls[1:], expected_controls, strict=True)
    )
    assert all(
        call["u"].dtype == expected.dtype
        for call, expected in zip(integrator_calls[1:], expected_controls, strict=True)
    )
    assert {path.name for path in result.iterdir()} == {
        "stage1_results.json",
        "stage1_states.npz",
        "decision.json",
    }
    payload = json.loads((result / "stage1_results.json").read_text(encoding="utf-8"))
    assert payload["schema_version"] == "action-shield-stage1-v1"
    assert payload["git_head"] == "d" * 40
    assert payload["dirty"] is True
    assert payload["reference_control"] == [0.4, 0.6]
    assert payload["selected_lambda"] == cli.DEFAULT_LAMBDAS[2]
    assert [item["success"] for item in payload["candidate_attempts"]] == [
        False,
        False,
        True,
    ]
    assert payload["outcome"] == "continue_to_context_ab"
    assert payload["delta_u_max"] == delta.tolist()
    decision = json.loads((result / "decision.json").read_text(encoding="utf-8"))
    assert set(decision) == {"outcome", "conditions", "selected_lambda"}
    assert all(decision["conditions"].values())
    with np.load(result / "stage1_states.npz", allow_pickle=False) as archive:
        assert set(archive.files) == {
            "x0",
            "selected_final_state",
            "selected_available",
        }
        assert archive["selected_available"].shape == ()
        assert bool(archive["selected_available"])
        assert np.array_equal(archive["selected_final_state"], [7.0, 8.0, 9.0])


def test_original_success_skips_controller_and_candidates(tmp_path):
    capsule = _capsule(tmp_path)
    config = _config(tmp_path / "env.yml")
    count = 0

    def factory(**kwargs):
        nonlocal count
        count += 1
        return lambda **inputs: {"xf": np.array([4.0, 5.0, 6.0])}

    def controller_factory():
        raise AssertionError("controller must not be constructed")

    output = tmp_path / "success"
    cli.run_stage1(
        capsule.path / "manifest.json",
        config,
        output,
        capsule_loader=lambda _: capsule,
        integrator_factory=factory,
        controller_factory=controller_factory,
    )
    assert count == 1
    decision = json.loads((output / "decision.json").read_text(encoding="utf-8"))
    assert decision["outcome"] == "redesign_action_shield"
    assert decision["conditions"]["original_reproduced"] is False
    assert decision["selected_lambda"] is None


def test_invalid_inputs_and_overlap_fail_before_factories(tmp_path):
    capsule = _capsule(tmp_path)
    config = _config(tmp_path / "env.yml")
    calls = []
    capsule.failure_inputs["nx"] = np.array(True)
    with pytest.raises(ValueError, match="nx"):
        cli.run_stage1(
            capsule.path / "manifest.json",
            config,
            tmp_path / "out",
            capsule_loader=lambda _: capsule,
            integrator_factory=lambda **kwargs: calls.append(kwargs),
            controller_factory=lambda: calls.append("controller"),
        )
    assert calls == []

    capsule = _capsule(tmp_path / "second")
    config.parent.mkdir(parents=True, exist_ok=True)
    with pytest.raises(ValueError, match="disjoint"):
        cli.run_stage1(
            capsule.path / "manifest.json",
            config,
            capsule.path / "child",
            capsule_loader=lambda _: capsule,
            integrator_factory=lambda **kwargs: calls.append(kwargs),
            controller_factory=lambda: calls.append("controller"),
        )
    assert calls == []


def test_float32_capsule_reconstruction_preserves_exact_numpy_semantics(tmp_path):
    capsule = _capsule(tmp_path)
    config = _config(tmp_path / "env.yml", delta=0.1)
    requested = np.array([0.1234567, -0.1234567], dtype=np.float32)
    previous = np.array([0.4, 0.6], dtype=np.float32)
    u_min = np.array([0.0, 0.0], dtype=np.float32)
    u_max = np.array([1.0, 1.0], dtype=np.float32)
    delta = np.ones(2, dtype=np.float32) * 0.1
    stored = np.clip(previous + requested * delta, u_min, u_max)
    capsule.failure_inputs["requested_action"] = requested
    capsule.failure_inputs["previous_control"] = previous
    capsule.failure_inputs["u"] = stored
    observed = []

    def factory(**kwargs):
        def integrate(**inputs):
            observed.append(np.array(inputs["u"], copy=True))
            return {"xf": np.array([4.0, 5.0, 6.0])}

        return integrate

    output = tmp_path / "float32"
    cli.run_stage1(
        capsule.path / "manifest.json",
        config,
        output,
        capsule_loader=lambda _: capsule,
        integrator_factory=factory,
        controller_factory=lambda: (_ for _ in ()).throw(
            AssertionError("controller must not run")
        ),
    )
    assert len(observed) == 1
    assert observed[0].dtype == stored.dtype == np.float32
    assert np.array_equal(observed[0], stored)


@pytest.mark.parametrize(
    "solver",
    [
        None,
        {},
        {"options": {"abstol": 0.0001, "reltol": 0.0001}},
        {
            "options": {
                "abstol": 0.0001,
                "reltol": 0.0001,
                "max_num_steps": 70000,
                "extra": 1,
            }
        },
        {
            "options": {
                "abstol": True,
                "reltol": 0.0001,
                "max_num_steps": 70000,
            }
        },
        {
            "options": {
                "abstol": float("nan"),
                "reltol": 0.0001,
                "max_num_steps": 70000,
            }
        },
        {
            "options": {
                "abstol": 0.0001,
                "reltol": 0.0001,
                "max_num_steps": 70001,
            }
        },
    ],
)
def test_invalid_formal_solver_provenance_rejected_before_factories(tmp_path, solver):
    capsule = _capsule(tmp_path)
    capsule.manifest["solver"] = solver
    config = _config(tmp_path / "env.yml")
    calls = []
    output = tmp_path / "out"
    with pytest.raises(ValueError, match="solver"):
        cli.run_stage1(
            capsule.path / "manifest.json",
            config,
            output,
            capsule_loader=lambda _: capsule,
            integrator_factory=lambda **kwargs: calls.append("integrator"),
            controller_factory=lambda: calls.append("controller"),
        )
    assert calls == []
    assert not output.exists()


def test_requested_action_outside_closed_unit_interval_rejected_before_factories(
    tmp_path,
):
    capsule = _capsule(tmp_path)
    capsule.failure_inputs["requested_action"] = np.array(
        [np.nextafter(1.0, 2.0), -1.0]
    )
    config = _config(tmp_path / "env.yml")
    calls = []
    output = tmp_path / "out"
    with pytest.raises(ValueError, match="requested_action"):
        cli.run_stage1(
            capsule.path / "manifest.json",
            config,
            output,
            capsule_loader=lambda _: capsule,
            integrator_factory=lambda **kwargs: calls.append("integrator"),
            controller_factory=lambda: calls.append("controller"),
        )
    assert calls == []
    assert not output.exists()


def test_config_and_rule_hashes_use_pre_execution_byte_snapshots(tmp_path, monkeypatch):
    capsule = _capsule(tmp_path)
    env_config = _config(tmp_path / "env.yml")
    rule_config = tmp_path / "rule_based.yml"
    rule_config.write_text("TomatoEnv:\n  lamps_on: 0\n", encoding="utf-8")
    monkeypatch.setattr(cli, "RULE_CONFIG_PATH", rule_config)
    env_bytes = env_config.read_bytes()
    rule_bytes = rule_config.read_bytes()

    def factory(**kwargs):
        env_config.unlink()
        rule_config.write_text("TomatoEnv:\n  lamps_on: 99\n", encoding="utf-8")
        return lambda **inputs: {"xf": np.array([4.0, 5.0, 6.0])}

    output = tmp_path / "snapshot"
    cli.run_stage1(
        capsule.path / "manifest.json",
        env_config,
        output,
        capsule_loader=lambda _: capsule,
        integrator_factory=factory,
        controller_factory=lambda: (_ for _ in ()).throw(
            AssertionError("controller must not run")
        ),
    )
    payload = json.loads((output / "stage1_results.json").read_text(encoding="utf-8"))
    assert payload["env_config_sha256"] == hashlib.sha256(env_bytes).hexdigest()
    assert payload["rule_config_sha256"] == hashlib.sha256(rule_bytes).hexdigest()


def test_default_controller_rejects_rule_mutation_before_construction(
    tmp_path, monkeypatch
):
    capsule = _capsule(tmp_path)
    env_config = _config(tmp_path / "env.yml")
    rule_config = tmp_path / "rule_based.yml"
    rule_config.write_text("TomatoEnv:\n  lamps_on: 0\n", encoding="utf-8")
    monkeypatch.setattr(cli, "RULE_CONFIG_PATH", rule_config)
    real_default = cli.build_rule_based_controller

    def factory(**kwargs):
        rule_config.write_text("TomatoEnv:\n  lamps_on: 99\n", encoding="utf-8")
        return lambda **inputs: (_ for _ in ()).throw(RuntimeError("original"))

    output = tmp_path / "mutated-default"
    with pytest.raises(ValueError, match="rule controller config changed"):
        cli.run_stage1(
            capsule.path / "manifest.json",
            env_config,
            output,
            capsule_loader=lambda _: capsule,
            integrator_factory=factory,
            controller_factory=real_default,
        )
    assert not output.exists()


def test_injected_controller_mutation_does_not_change_snapshotted_hashes(
    tmp_path, monkeypatch
):
    capsule = _capsule(tmp_path)
    env_config = _config(tmp_path / "env.yml")
    rule_config = tmp_path / "rule_based.yml"
    rule_config.write_text("TomatoEnv:\n  lamps_on: 0\n", encoding="utf-8")
    monkeypatch.setattr(cli, "RULE_CONFIG_PATH", rule_config)
    env_bytes = env_config.read_bytes()
    rule_bytes = rule_config.read_bytes()
    factory_calls = 0

    def factory(**kwargs):
        nonlocal factory_calls
        factory_calls += 1
        if factory_calls == 1:
            return lambda **inputs: (_ for _ in ()).throw(RuntimeError("original"))
        return lambda **inputs: {"xf": np.array([4.0, 5.0, 6.0])}

    def controller_factory():
        env_config.unlink()
        rule_config.write_text("TomatoEnv:\n  lamps_on: 99\n", encoding="utf-8")
        return SimpleNamespace(predict=lambda *args: np.array([0.4, 0.6]))

    output = tmp_path / "controller-snapshot"
    cli.run_stage1(
        capsule.path / "manifest.json",
        env_config,
        output,
        capsule_loader=lambda _: capsule,
        integrator_factory=factory,
        controller_factory=controller_factory,
    )
    payload = json.loads((output / "stage1_results.json").read_text(encoding="utf-8"))
    assert payload["env_config_sha256"] == hashlib.sha256(env_bytes).hexdigest()
    assert payload["rule_config_sha256"] == hashlib.sha256(rule_bytes).hexdigest()


def test_noncallable_original_integrator_is_construction_error(tmp_path):
    capsule = _capsule(tmp_path)
    config = _config(tmp_path / "env.yml")
    output = tmp_path / "noncallable-original"
    with pytest.raises(TypeError, match="callable"):
        cli.run_stage1(
            capsule.path / "manifest.json",
            config,
            output,
            capsule_loader=lambda _: capsule,
            integrator_factory=lambda **kwargs: object(),
            controller_factory=lambda: (_ for _ in ()).throw(
                AssertionError("controller must not run")
            ),
        )
    assert not output.exists()


def test_noncallable_candidate_integrator_is_construction_error(tmp_path):
    capsule = _capsule(tmp_path)
    config = _config(tmp_path / "env.yml")
    output = tmp_path / "noncallable-candidate"
    calls = 0

    def factory(**kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            return lambda **inputs: (_ for _ in ()).throw(RuntimeError("original"))
        return object()

    with pytest.raises(TypeError, match="callable"):
        cli.run_stage1(
            capsule.path / "manifest.json",
            config,
            output,
            capsule_loader=lambda _: capsule,
            integrator_factory=factory,
            controller_factory=lambda: SimpleNamespace(
                predict=lambda *args: np.array([0.4, 0.6])
            ),
        )
    assert calls == 2
    assert not output.exists()


def test_malformed_integrator_output_and_baseexception_propagate_without_output(
    tmp_path,
):
    capsule = _capsule(tmp_path)
    config = _config(tmp_path / "env.yml")
    output = tmp_path / "bad"
    with pytest.raises(ValueError, match="final state"):
        cli.run_stage1(
            capsule.path / "manifest.json",
            config,
            output,
            capsule_loader=lambda _: capsule,
            integrator_factory=lambda **kwargs: lambda **inputs: {"xf": [1.0]},
            controller_factory=lambda: None,
        )
    assert not output.exists()

    with pytest.raises(RuntimeError, match="construction failed"):
        cli.run_stage1(
            capsule.path / "manifest.json",
            config,
            output,
            capsule_loader=lambda _: capsule,
            integrator_factory=lambda **kwargs: (_ for _ in ()).throw(
                RuntimeError("construction failed")
            ),
            controller_factory=lambda: None,
        )
    assert not output.exists()

    class Stop(BaseException):
        pass

    with pytest.raises(Stop):
        cli.run_stage1(
            capsule.path / "manifest.json",
            config,
            output,
            capsule_loader=lambda _: capsule,
            integrator_factory=lambda **kwargs: lambda **inputs: (_ for _ in ()).throw(
                Stop()
            ),
            controller_factory=lambda: None,
        )
    assert not output.exists()


def test_candidate_exhaustion_and_controller_error_publish_rules(tmp_path):
    capsule = _capsule(tmp_path)
    config = _config(tmp_path / "env.yml")

    def failed_factory(**kwargs):
        return lambda **inputs: (_ for _ in ()).throw(RuntimeError("solver"))

    output = tmp_path / "exhausted"
    cli.run_stage1(
        capsule.path / "manifest.json",
        config,
        output,
        capsule_loader=lambda _: capsule,
        integrator_factory=failed_factory,
        controller_factory=lambda: SimpleNamespace(
            predict=lambda *args: np.array([0.8, 0.1])
        ),
    )
    decision = json.loads((output / "decision.json").read_text(encoding="utf-8"))
    assert decision["outcome"] == "redesign_action_shield"
    assert decision["conditions"]["legal_candidate_succeeded"] is False
    assert len(
        json.loads((output / "stage1_results.json").read_text())["candidate_attempts"]
    ) == len(cli.DEFAULT_LAMBDAS)

    error_output = tmp_path / "controller-error"
    with pytest.raises(RuntimeError, match="controller unavailable"):
        cli.run_stage1(
            capsule.path / "manifest.json",
            config,
            error_output,
            capsule_loader=lambda _: capsule,
            integrator_factory=failed_factory,
            controller_factory=lambda: (_ for _ in ()).throw(
                RuntimeError("controller unavailable")
            ),
        )
    assert not error_output.exists()


def test_publication_failure_restores_prior_root(tmp_path, monkeypatch):
    capsule = _capsule(tmp_path)
    config = _config(tmp_path / "env.yml")
    output = tmp_path / "existing"
    output.mkdir()
    (output / "old.txt").write_text("old", encoding="utf-8")
    real_replace = cli.os.replace

    def fail_stage(source, destination):
        if (
            Path(destination).resolve() == output.resolve()
            and ".stage-" in Path(source).name
        ):
            raise OSError("publish failed")
        return real_replace(source, destination)

    monkeypatch.setattr(cli.os, "replace", fail_stage)
    with pytest.raises(OSError, match="publish failed"):
        cli.run_stage1(
            capsule.path / "manifest.json",
            config,
            output,
            capsule_loader=lambda _: capsule,
            integrator_factory=lambda **kwargs: lambda **inputs: {
                "xf": [1.0, 2.0, 3.0]
            },
            controller_factory=lambda: None,
        )
    assert (output / "old.txt").read_text(encoding="utf-8") == "old"
    assert not list(tmp_path.glob(".existing.stage-*"))
