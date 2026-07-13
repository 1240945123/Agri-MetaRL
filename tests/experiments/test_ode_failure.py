from __future__ import annotations

import hashlib
import json
from dataclasses import FrozenInstanceError
from pathlib import Path

import numpy as np
import pytest

from gl_gym.experiments.ode_failure import (
    CapsuleContext,
    FailureCapsuleRecorder,
    load_failure_capsule,
)


def _context(**changes) -> CapsuleContext:
    values = {
        "seed": 17,
        "task_id": "heldout/../north plot",
        "inference_mode": "formal eval/zero-shot",
        "task": {"location": "NL", "horizon": 24},
        "checkpoint_path": "checkpoints/agent.zip",
        "checkpoint_sha256": "a" * 64,
        "git_head": "b" * 40,
        "dirty": True,
        "source_checksums": {"tomato_env.py": "c" * 64},
        "package_versions": {"numpy": np.__version__, "casadi": "3.7.0"},
        "formal_result_root": "formal/results",
    }
    values.update(changes)
    return CapsuleContext(**values)


def _transition(step: int, *, next_available: bool = True) -> dict:
    return {
        "raw_observation": np.array([step, step + 0.25], dtype=np.float32),
        "requested_action": np.array([step + 1], dtype=np.float64),
        "previous_control": np.array([step + 2], dtype=np.float64),
        "executed_control": np.array([step + 3], dtype=np.float64),
        "raw_next_observation": (
            np.array([step + 1, step + 1.25], dtype=np.float32)
            if next_available
            else None
        ),
        "raw_next_observation_available": next_available,
    }


def _failure(timestep: int = 8) -> dict:
    weather = np.array([19.5, 0.4], dtype=np.float64)
    parameters = np.array([1.1, 2.2, 3.3], dtype=np.float64)
    return {
        "x0": np.array([4.0, 5.0], dtype=np.float64),
        "u": np.array([0.75], dtype=np.float64),
        "previous_control": np.array([0.5], dtype=np.float64),
        "requested_action": np.array([-0.25], dtype=np.float64),
        "weather": weather,
        "sampled_parameters": parameters,
        "p_dyn": np.concatenate((weather, parameters)),
        "timestep": timestep,
        "day_of_year": 151.0,
        "hour_of_day": 12.5,
        "dt": 300.0,
        "nx": 2,
        "nu": 1,
        "nd": 2,
        "n_params": 3,
        "solver_options": {"abstol": 1e-8, "max_num_steps": 1000},
        "exception_type": "RuntimeError",
        "exception_message": "CVODES failed deterministically",
        "traceback": "Traceback (most recent call last):\nRuntimeError: CVODES failed deterministically\n",
    }


def _info(step: int, *, failure: bool = False, next_available: bool = True) -> dict:
    result = {
        "diagnostic_transition": _transition(step, next_available=next_available),
        "episode_return": float(step) + 0.5,
        "success": step % 2 == 0,
        "ignored_text": "not persisted as a metric",
        "ignored_array": np.array([99.0]),
    }
    if failure:
        result["integration_failure"] = _failure(step)
        result["diagnostic_transition"] = _transition(step, next_available=False)
    return result


def _record_failure(root: Path, *, history_prefix: float = 0.0) -> Path:
    recorder = FailureCapsuleRecorder(root, _context())
    for step in range(3):
        recorder.record_step(
            step,
            np.array([history_prefix + step, 10.0], dtype=np.float32),
            reward=float(step),
            done=False,
            info=_info(step),
        )
    path = recorder.record_step(
        8,
        np.array([history_prefix + 8, 10.0], dtype=np.float32),
        reward=-1.0,
        done=True,
        info=_info(8, failure=True),
    )
    assert path is not None
    return path


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _rewrite_manifest(path: Path, transform) -> None:
    manifest_path = path / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    transform(manifest)
    manifest_path.write_text(
        json.dumps(manifest, sort_keys=True, separators=(",", ":"), allow_nan=False),
        encoding="utf-8",
    )


def _rewrite_history_npz(path: Path, transform) -> None:
    history_path = path / "history.npz"
    with np.load(history_path, allow_pickle=False) as archive:
        arrays = {name: np.array(archive[name], copy=True) for name in archive.files}
    transform(arrays)
    np.savez_compressed(history_path, **arrays)

    def update(manifest):
        manifest["files"]["history.npz"]["sha256"] = _sha256(history_path)
        manifest["arrays"]["history"] = {
            name: {"dtype": array.dtype.str, "shape": list(array.shape)}
            for name, array in sorted(arrays.items())
        }

    _rewrite_manifest(path, update)


def test_context_is_frozen_and_capacity_is_fixed(tmp_path):
    context = _context()
    with pytest.raises(FrozenInstanceError):
        context.seed = 9
    with pytest.raises(ValueError, match="256"):
        FailureCapsuleRecorder(tmp_path, context, capacity=255)


def test_history_retains_last_256_steps_and_copies_inputs(tmp_path):
    recorder = FailureCapsuleRecorder(tmp_path, _context())
    observation = np.array([0.0, 1.0])
    transition = _transition(0)
    info = {"diagnostic_transition": transition, "metric": 2.0}
    recorder.record_step(0, observation, 1.0, False, info)
    observation[:] = -1
    transition["requested_action"][:] = -2
    for step in range(1, 299):
        recorder.record_step(step, np.array([step, step + 1]), 1.0, False, _info(step))
    last_info = _info(299)
    recorder.record_step(299, np.array([299, 300]), 1.0, False, last_info)
    last_info["diagnostic_transition"]["requested_action"][:] = -999

    assert recorder.history_length == 256
    assert recorder.history_step_indices == tuple(range(44, 300))
    np.testing.assert_array_equal(recorder.last_requested_action, np.array([300.0]))
    assert not any(tmp_path.iterdir())


def test_record_step_requires_complete_numeric_diagnostic_transition(tmp_path):
    recorder = FailureCapsuleRecorder(tmp_path, _context())
    with pytest.raises(KeyError, match="diagnostic_transition"):
        recorder.record_step(0, np.zeros(2), 0.0, False, {})
    transition = _transition(0)
    transition["raw_observation"] = np.array(["secret"], dtype=object)
    with pytest.raises((TypeError, ValueError), match="numeric"):
        recorder.record_step(
            0, np.zeros(2), 0.0, False, {"diagnostic_transition": transition}
        )


@pytest.mark.parametrize(
    "field",
    [
        "policy_observation",
        "raw_observation",
        "requested_action",
        "previous_control",
        "executed_control",
        "raw_next_observation",
    ],
)
def test_recorder_rejects_non_finite_transition_arrays(tmp_path, field):
    recorder = FailureCapsuleRecorder(tmp_path, _context())
    transition = _transition(0)
    policy_observation = np.zeros(2)
    if field == "policy_observation":
        policy_observation[0] = np.nan
    else:
        transition[field][0] = np.inf

    with pytest.raises(ValueError, match="finite"):
        recorder.record_step(
            0,
            policy_observation,
            0.0,
            False,
            {"diagnostic_transition": transition},
        )


def test_valid_capsule_round_trip_exact_files_metadata_and_sanitized_path(tmp_path):
    path = _record_failure(tmp_path)
    assert path.parents[2] == tmp_path / "17"
    assert ".." not in path.parts
    assert "/" not in path.parent.name and "\\" not in path.parent.name
    assert {item.name for item in path.iterdir()} == {
        "manifest.json",
        "failure_inputs.npz",
        "history.npz",
        "history.jsonl",
        "traceback.txt",
    }

    loaded = load_failure_capsule(path)
    assert loaded.path == path
    assert loaded.manifest["schema_version"] == 1
    assert loaded.manifest["failure_id"] == path.name
    assert loaded.manifest["context"]["source_checksums"] == _context().source_checksums
    assert loaded.manifest["context"]["package_versions"] == _context().package_versions
    assert loaded.manifest["context"]["formal_result_root"] == "formal/results"
    assert loaded.manifest["context"]["git_head"] == "b" * 40
    assert loaded.manifest["context"]["dirty"] is True
    assert loaded.manifest["source_checksums"] == _context().source_checksums
    assert loaded.manifest["package_versions"] == _context().package_versions
    assert loaded.manifest["formal_result_root"] == "formal/results"
    assert set(loaded.manifest["files"]) == {
        "failure_inputs.npz",
        "history.npz",
        "history.jsonl",
        "traceback.txt",
    }
    np.testing.assert_array_equal(
        loaded.failure_inputs["p_dyn"],
        np.concatenate(
            (
                loaded.failure_inputs["weather"],
                loaded.failure_inputs["sampled_parameters"],
            )
        ),
    )
    assert len(loaded.history_rows) == 4
    assert loaded.history_rows[-1]["step_index"] == 8
    assert loaded.history_rows[-1]["metrics"] == {
        "episode_return": 8.5,
        "success": True,
    }
    assert loaded.history_arrays["raw_next_observation_available"].tolist() == [
        True,
        True,
        True,
        False,
    ]
    assert "RuntimeError: CVODES failed deterministically" in loaded.traceback_text
    assert all(
        array.dtype != object
        for array in (*loaded.failure_inputs.values(), *loaded.history_arrays.values())
    )


def test_fresh_recorder_is_idempotent_for_identical_content(tmp_path):
    first = _record_failure(tmp_path)
    before = {item.name: _sha256(item) for item in first.iterdir()}
    second = _record_failure(tmp_path)
    assert second == first
    assert {item.name: _sha256(item) for item in second.iterdir()} == before


def test_same_stable_id_with_different_history_is_collision(tmp_path):
    first = _record_failure(tmp_path, history_prefix=0.0)
    with pytest.raises(FileExistsError, match=first.name):
        _record_failure(tmp_path, history_prefix=100.0)


def test_writer_cleans_sibling_temp_directory_on_failure(tmp_path, monkeypatch):
    recorder = FailureCapsuleRecorder(tmp_path, _context())

    def explode(*args, **kwargs):
        raise OSError("disk write failed")

    monkeypatch.setattr("gl_gym.experiments.ode_failure.np.savez_compressed", explode)
    with pytest.raises(OSError, match="disk write failed"):
        recorder.record_step(8, np.zeros(2), -1.0, True, _info(8, failure=True))
    assert not list(tmp_path.rglob("*.tmp-*"))


@pytest.mark.parametrize(
    "filename", ["failure_inputs.npz", "history.npz", "history.jsonl", "traceback.txt"]
)
def test_loader_rejects_missing_required_file(tmp_path, filename):
    path = _record_failure(tmp_path)
    (path / filename).unlink()
    with pytest.raises(ValueError, match="missing|required"):
        load_failure_capsule(path)


def test_loader_rejects_checksum_tamper(tmp_path):
    path = _record_failure(tmp_path)
    with (path / "traceback.txt").open("a", encoding="utf-8") as handle:
        handle.write("tampered")
    with pytest.raises(ValueError, match="checksum"):
        load_failure_capsule(path)


def test_loader_rejects_object_array_even_with_updated_checksum(tmp_path):
    path = _record_failure(tmp_path)
    np.savez_compressed(
        path / "failure_inputs.npz", x0=np.array([object()], dtype=object)
    )
    _rewrite_manifest(
        path,
        lambda manifest: manifest["files"]["failure_inputs.npz"].update(
            sha256=_sha256(path / "failure_inputs.npz")
        ),
    )
    with pytest.raises(ValueError, match="object|pickle"):
        load_failure_capsule(path)


def test_loader_uses_allow_pickle_false(tmp_path, monkeypatch):
    path = _record_failure(tmp_path)
    real_load = np.load
    calls = []

    def recording_load(*args, **kwargs):
        calls.append(kwargs.get("allow_pickle"))
        return real_load(*args, **kwargs)

    monkeypatch.setattr("gl_gym.experiments.ode_failure.np.load", recording_load)
    load_failure_capsule(path)
    assert calls == [False, False]


def test_loader_rejects_json_nan_constant(tmp_path):
    path = _record_failure(tmp_path)
    manifest_path = path / "manifest.json"
    text = manifest_path.read_text(encoding="utf-8")
    manifest_path.write_text(text[:-1] + ',"not_finite":NaN}', encoding="utf-8")
    with pytest.raises(ValueError, match="NaN|constant|JSON"):
        load_failure_capsule(path)


def test_loader_rejects_p_dyn_mismatch_with_updated_checksum(tmp_path):
    path = _record_failure(tmp_path)
    with np.load(path / "failure_inputs.npz", allow_pickle=False) as archive:
        arrays = {name: np.array(archive[name], copy=True) for name in archive.files}
    arrays["p_dyn"][0] += 1.0
    np.savez_compressed(path / "failure_inputs.npz", **arrays)
    _rewrite_manifest(
        path,
        lambda manifest: manifest["files"]["failure_inputs.npz"].update(
            sha256=_sha256(path / "failure_inputs.npz")
        ),
    )
    with pytest.raises(ValueError, match="p_dyn"):
        load_failure_capsule(path)


def test_loader_rejects_non_finite_history_array_with_updated_checksum(tmp_path):
    path = _record_failure(tmp_path)

    def corrupt(arrays):
        arrays["policy_observation"][0, 0] = np.nan

    _rewrite_history_npz(path, corrupt)
    with pytest.raises(ValueError, match="finite"):
        load_failure_capsule(path)


@pytest.mark.parametrize(
    "field",
    ["requested_action", "previous_control", "executed_control"],
)
def test_loader_rejects_malformed_history_action_control_dimensions(tmp_path, field):
    path = _record_failure(tmp_path)

    def corrupt(arrays):
        arrays[field] = arrays[field].reshape(len(arrays[field]), 1, 1)

    _rewrite_history_npz(path, corrupt)
    with pytest.raises(ValueError, match="dimension|width"):
        load_failure_capsule(path)


def test_loader_rejects_manifest_failure_id_and_path_mismatch(tmp_path):
    path = _record_failure(tmp_path)
    wrong_id = "0" * 64
    _rewrite_manifest(path, lambda manifest: manifest.update(failure_id=wrong_id))
    renamed = path.with_name(wrong_id)
    path.rename(renamed)
    with pytest.raises(ValueError, match="failure ID"):
        load_failure_capsule(renamed)


def test_loader_rejects_extra_regular_file(tmp_path):
    path = _record_failure(tmp_path)
    (path / "unexpected.txt").write_text("not part of the capsule", encoding="utf-8")
    with pytest.raises(ValueError, match="extra|unexpected|required"):
        load_failure_capsule(path)


def test_optional_raw_next_observation_uses_mask_and_finite_fill(tmp_path):
    path = _record_failure(tmp_path)
    loaded = load_failure_capsule(path)
    mask = loaded.history_arrays["raw_next_observation_available"]
    raw_next = loaded.history_arrays["raw_next_observation"]
    assert mask[-1] == np.bool_(False)
    assert np.isfinite(raw_next).all()
    assert np.count_nonzero(raw_next[-1]) == 0


def test_failure_id_depends_on_exact_dtype_shape_bytes_and_task_identity(tmp_path):
    first = _record_failure(tmp_path / "first")
    changed = _context(task_id="different-task")
    recorder = FailureCapsuleRecorder(tmp_path / "second", changed)
    second = recorder.record_step(8, np.zeros(2), -1.0, True, _info(8, failure=True))
    assert second is not None and second.name != first.name

    info = _info(8, failure=True)
    info["integration_failure"]["x0"] = info["integration_failure"]["x0"].astype(
        np.float32
    )
    third = FailureCapsuleRecorder(tmp_path / "third", _context()).record_step(
        8, np.zeros(2), -1.0, True, info
    )
    assert third is not None and third.name != first.name
