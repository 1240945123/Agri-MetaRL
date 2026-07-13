"""Immutable, self-validating evidence capsules for ODE integration failures."""

from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
from typing import Any
import uuid

import numpy as np


SCHEMA_VERSION = 1
CAPACITY = 256
REQUIRED_FILES = frozenset(
    {
        "manifest.json",
        "failure_inputs.npz",
        "history.npz",
        "history.jsonl",
        "traceback.txt",
    }
)
FAILURE_ARRAYS = (
    "x0",
    "u",
    "previous_control",
    "requested_action",
    "weather",
    "sampled_parameters",
    "p_dyn",
)
FAILURE_SCALARS = (
    "timestep",
    "day_of_year",
    "hour_of_day",
    "dt",
    "nx",
    "nu",
    "nd",
    "n_params",
)
TRANSITION_ARRAYS = (
    "raw_observation",
    "requested_action",
    "previous_control",
    "executed_control",
)
HISTORY_ARRAYS = (
    "step_index",
    "policy_observation",
    "raw_observation",
    "requested_action",
    "previous_control",
    "executed_control",
    "raw_next_observation",
    "raw_next_observation_available",
    "reward",
    "done",
)
_SAFE_PART = re.compile(r"[^A-Za-z0-9._-]+")


@dataclass(frozen=True, slots=True)
class CapsuleContext:
    seed: int
    task_id: str
    inference_mode: str
    task: dict
    checkpoint_path: str
    checkpoint_sha256: str
    git_head: str
    dirty: bool
    source_checksums: dict
    package_versions: dict
    formal_result_root: str | None = None


@dataclass(frozen=True, slots=True)
class LoadedFailureCapsule:
    path: Path
    manifest: dict
    failure_inputs: dict[str, np.ndarray]
    history_arrays: dict[str, np.ndarray]
    history_rows: tuple[dict, ...]
    traceback_text: str


def _strict_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _reject_constant(value: str) -> None:
    raise ValueError(f"invalid JSON constant {value}")


def _read_json(path: Path) -> Any:
    try:
        return json.loads(
            path.read_text(encoding="utf-8"), parse_constant=_reject_constant
        )
    except (UnicodeError, json.JSONDecodeError, ValueError) as error:
        raise ValueError(f"invalid strict JSON in {path.name}: {error}") from error


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _numeric_array(value: Any, name: str) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype.kind not in "biuf":
        raise TypeError(f"{name} must be a real numeric array")
    return np.array(array, copy=True)


def _finite_array(value: Any, name: str) -> np.ndarray:
    array = _numeric_array(value, name)
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values")
    return array


def _finite_scalar(value: Any, name: str) -> int | float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        raise TypeError(f"{name} must be a finite numeric scalar")
    result = value.item() if isinstance(value, np.generic) else value
    if not math.isfinite(float(result)):
        raise ValueError(f"{name} must be finite")
    return result


def _sanitize(value: Any) -> str:
    text = _SAFE_PART.sub("_", str(value)).strip(" ._")
    while ".." in text:
        text = text.replace("..", ".")
    if not text or text in {".", ".."}:
        text = "unnamed"
    return text[:120]


def _update_framed(digest: Any, label: str, payload: bytes) -> None:
    label_bytes = label.encode("utf-8")
    digest.update(len(label_bytes).to_bytes(4, "big"))
    digest.update(label_bytes)
    digest.update(len(payload).to_bytes(8, "big"))
    digest.update(payload)


def _update_array(digest: Any, label: str, array: np.ndarray) -> None:
    contiguous = np.ascontiguousarray(array)
    descriptor = _strict_json_bytes(
        {"dtype": contiguous.dtype.str, "shape": list(contiguous.shape)}
    )
    _update_framed(digest, f"{label}:descriptor", descriptor)
    _update_framed(digest, f"{label}:bytes", contiguous.tobytes(order="C"))


def _failure_id(context: dict, failure_inputs: dict[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    identity = {
        "task_id": context["task_id"],
        "task": context["task"],
        "timestep": int(failure_inputs["timestep"].item()),
    }
    _update_framed(digest, "task_and_timestep", _strict_json_bytes(identity))
    for name in ("x0", "u", "p_dyn"):
        _update_array(digest, name, failure_inputs[name])
    return digest.hexdigest()


def _array_metadata(arrays: dict[str, np.ndarray]) -> dict[str, dict[str, Any]]:
    return {
        name: {"dtype": array.dtype.str, "shape": list(array.shape)}
        for name, array in sorted(arrays.items())
    }


def _content_identity(
    context: dict,
    failure_inputs: dict[str, np.ndarray],
    history_arrays: dict[str, np.ndarray],
    history_rows: tuple[dict, ...],
    solver_options: dict,
    exception_type: str,
    exception_message: str,
    traceback_text: str,
) -> str:
    identity = hashlib.sha256()
    _update_framed(identity, "context", _strict_json_bytes(context))
    for name, array in sorted(failure_inputs.items()):
        _update_array(identity, f"failure:{name}", array)
    for name, array in sorted(history_arrays.items()):
        _update_array(identity, f"history:{name}", array)
    _update_framed(identity, "history_rows", _strict_json_bytes(history_rows))
    _update_framed(identity, "solver_options", _strict_json_bytes(solver_options))
    _update_framed(identity, "exception_type", exception_type.encode("utf-8"))
    _update_framed(identity, "exception_message", exception_message.encode("utf-8"))
    _update_framed(identity, "traceback", traceback_text.encode("utf-8"))
    return identity.hexdigest()


def _context_dict(context: CapsuleContext) -> dict:
    data = asdict(context)
    # Validate at the boundary and detach mutable dictionaries from the caller.
    return json.loads(
        _strict_json_bytes(data).decode("utf-8"), parse_constant=_reject_constant
    )


class FailureCapsuleRecorder:
    def __init__(
        self, root: str | Path, context: CapsuleContext, capacity: int = CAPACITY
    ):
        if capacity != CAPACITY:
            raise ValueError("failure history capacity must be exactly 256")
        self.root = Path(root)
        self.context = context
        self._context = _context_dict(context)
        self._history: deque[dict[str, Any]] = deque(maxlen=CAPACITY)

    @property
    def history_length(self) -> int:
        return len(self._history)

    @property
    def history_step_indices(self) -> tuple[int, ...]:
        return tuple(entry["step_index"] for entry in self._history)

    @property
    def last_requested_action(self) -> np.ndarray | None:
        if not self._history:
            return None
        return np.array(self._history[-1]["requested_action"], copy=True)

    def record_step(
        self,
        step_index: int,
        policy_observation: Any,
        reward: float,
        done: bool,
        info: dict,
    ) -> Path | None:
        if "diagnostic_transition" not in info:
            raise KeyError("info['diagnostic_transition'] is required")
        transition = info["diagnostic_transition"]
        if not isinstance(transition, dict):
            raise TypeError("diagnostic_transition must be a dictionary")
        step = _finite_scalar(step_index, "step_index")
        if not isinstance(step, int):
            raise TypeError("step_index must be an integer")
        scalar_reward = _finite_scalar(reward, "reward")
        if not isinstance(done, (bool, np.bool_)):
            raise TypeError("done must be boolean")
        entry: dict[str, Any] = {
            "step_index": step,
            "policy_observation": _numeric_array(
                policy_observation, "policy_observation"
            ),
            "reward": scalar_reward,
            "done": bool(done),
        }
        for name in TRANSITION_ARRAYS:
            if name not in transition:
                raise KeyError(f"diagnostic_transition['{name}'] is required")
            entry[name] = _numeric_array(transition[name], name)
        if "raw_next_observation_available" not in transition:
            raise KeyError(
                "diagnostic_transition['raw_next_observation_available'] is required"
            )
        available = transition["raw_next_observation_available"]
        if not isinstance(available, (bool, np.bool_)):
            raise TypeError("raw_next_observation_available must be boolean")
        entry["raw_next_observation_available"] = bool(available)
        raw_next = transition.get("raw_next_observation")
        if available:
            if raw_next is None:
                raise ValueError("available raw_next_observation cannot be None")
            entry["raw_next_observation"] = _numeric_array(
                raw_next, "raw_next_observation"
            )
        else:
            if raw_next is not None:
                raise ValueError("unavailable raw_next_observation must be None")
            entry["raw_next_observation"] = None
        entry["metrics"] = self._select_metrics(info)
        self._history.append(entry)
        if "integration_failure" not in info:
            return None
        return self._write_failure(info["integration_failure"])

    @staticmethod
    def _select_metrics(info: dict) -> dict[str, bool | int | float]:
        metrics: dict[str, bool | int | float] = {}
        for key, value in info.items():
            if key in {"diagnostic_transition", "integration_failure"}:
                continue
            if isinstance(value, (bool, np.bool_)):
                metrics[str(key)] = bool(value)
            elif isinstance(value, (int, float, np.integer, np.floating)):
                scalar = value.item() if isinstance(value, np.generic) else value
                if math.isfinite(float(scalar)):
                    metrics[str(key)] = scalar
        return dict(sorted(metrics.items()))

    def _failure_inputs(self, failure: Any) -> dict[str, np.ndarray]:
        if not isinstance(failure, dict):
            raise TypeError("integration_failure must be a dictionary")
        result: dict[str, np.ndarray] = {}
        for name in FAILURE_ARRAYS:
            if name not in failure:
                raise KeyError(f"integration_failure['{name}'] is required")
            result[name] = _finite_array(failure[name], name)
        for name in FAILURE_SCALARS:
            if name not in failure:
                raise KeyError(f"integration_failure['{name}'] is required")
            value = _finite_scalar(failure[name], name)
            if name in {"timestep", "nx", "nu", "nd", "n_params"} and not isinstance(
                value, int
            ):
                raise TypeError(f"{name} must be an integer")
            result[name] = np.asarray(value)
        return result

    def _history_payload(self) -> tuple[dict[str, np.ndarray], tuple[dict, ...]]:
        entries = tuple(self._history)
        if not entries:
            raise ValueError("cannot write an empty failure history")
        arrays: dict[str, np.ndarray] = {}
        for name in (
            "policy_observation",
            "raw_observation",
            "requested_action",
            "previous_control",
            "executed_control",
        ):
            try:
                arrays[name] = np.stack([entry[name] for entry in entries])
            except ValueError as error:
                raise ValueError(f"inconsistent history shape for {name}") from error
        template = entries[0]["raw_observation"]
        raw_next_values = []
        for entry in entries:
            if entry["raw_next_observation_available"]:
                value = entry["raw_next_observation"]
            else:
                value = np.zeros_like(template)
            if value.shape != template.shape:
                raise ValueError(
                    "raw_next_observation shape does not match raw_observation"
                )
            raw_next_values.append(value)
        arrays["raw_next_observation"] = np.stack(raw_next_values)
        arrays["raw_next_observation_available"] = np.asarray(
            [entry["raw_next_observation_available"] for entry in entries],
            dtype=np.bool_,
        )
        arrays["step_index"] = np.asarray(
            [entry["step_index"] for entry in entries], dtype=np.int64
        )
        arrays["reward"] = np.asarray(
            [entry["reward"] for entry in entries], dtype=np.float64
        )
        arrays["done"] = np.asarray(
            [entry["done"] for entry in entries], dtype=np.bool_
        )
        rows = tuple(
            {
                "step_index": entry["step_index"],
                "reward": entry["reward"],
                "done": entry["done"],
                "metrics": entry["metrics"],
            }
            for entry in entries
        )
        return arrays, rows

    def _write_failure(self, failure: dict) -> Path:
        failure_inputs = self._failure_inputs(failure)
        history_arrays, history_rows = self._history_payload()
        failure_id = _failure_id(self._context, failure_inputs)
        target = (
            self.root
            / str(self.context.seed)
            / _sanitize(self.context.task_id)
            / _sanitize(self.context.inference_mode)
            / failure_id
        )
        target.parent.mkdir(parents=True, exist_ok=True)
        # Keep sibling staging names short enough for legacy Windows path limits.
        temporary = target.parent / f".tmp-{uuid.uuid4().hex}"
        temporary.mkdir()
        try:
            np.savez_compressed(temporary / "failure_inputs.npz", **failure_inputs)
            np.savez_compressed(temporary / "history.npz", **history_arrays)
            with (temporary / "history.jsonl").open("wb") as handle:
                for row in history_rows:
                    handle.write(_strict_json_bytes(row) + b"\n")
            traceback_text = failure.get("traceback")
            if not isinstance(traceback_text, str):
                raise TypeError("integration_failure['traceback'] must be text")
            (temporary / "traceback.txt").write_text(traceback_text, encoding="utf-8")
            manifest = self._manifest(
                failure,
                failure_id,
                failure_inputs,
                history_arrays,
                history_rows,
                traceback_text,
                temporary,
            )
            (temporary / "manifest.json").write_bytes(_strict_json_bytes(manifest))
            _load_failure_capsule(temporary, check_directory_name=False)
            if target.exists():
                existing = load_failure_capsule(target)
                if (
                    existing.manifest.get("content_identity_sha256")
                    == manifest["content_identity_sha256"]
                ):
                    return target
                raise FileExistsError(f"failure capsule collision for {failure_id}")
            try:
                os.rename(temporary, target)
            except FileExistsError:
                existing = load_failure_capsule(target)
                if (
                    existing.manifest.get("content_identity_sha256")
                    == manifest["content_identity_sha256"]
                ):
                    return target
                raise FileExistsError(f"failure capsule collision for {failure_id}")
            return target
        finally:
            if temporary.exists():
                shutil.rmtree(temporary)

    def _manifest(
        self,
        failure: dict,
        failure_id: str,
        failure_inputs: dict[str, np.ndarray],
        history_arrays: dict[str, np.ndarray],
        history_rows: tuple[dict, ...],
        traceback_text: str,
        directory: Path,
    ) -> dict:
        solver_options = failure.get("solver_options", {})
        if not isinstance(solver_options, dict):
            raise TypeError("solver_options must be a dictionary")
        exception_type = failure.get("exception_type")
        exception_message = failure.get("exception_message")
        if not isinstance(exception_type, str) or not isinstance(
            exception_message, str
        ):
            raise TypeError("exception metadata must be text")
        content_identity = _content_identity(
            self._context,
            failure_inputs,
            history_arrays,
            history_rows,
            solver_options,
            exception_type,
            exception_message,
            traceback_text,
        )
        files = {}
        for name in (
            "failure_inputs.npz",
            "history.npz",
            "history.jsonl",
            "traceback.txt",
        ):
            file_path = directory / name
            files[name] = {
                "sha256": _sha256_file(file_path),
                "size_bytes": file_path.stat().st_size,
            }
        return {
            "schema_version": SCHEMA_VERSION,
            "failure_id": failure_id,
            "content_identity_sha256": content_identity,
            "context": self._context,
            "task_id": self._context["task_id"],
            "task": self._context["task"],
            "checkpoint_path": self._context["checkpoint_path"],
            "checkpoint_sha256": self._context["checkpoint_sha256"],
            "source_checksums": self._context["source_checksums"],
            "package_versions": self._context["package_versions"],
            "git_head": self._context["git_head"],
            "dirty": self._context["dirty"],
            "formal_result_root": self._context["formal_result_root"],
            "failure_timestep": int(failure_inputs["timestep"].item()),
            "solver": {"options": solver_options},
            "exception": {"type": exception_type, "message": exception_message},
            "arrays": {
                "failure_inputs": _array_metadata(failure_inputs),
                "history": _array_metadata(history_arrays),
            },
            "history_length": len(history_rows),
            "files": files,
        }


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    try:
        with np.load(path, allow_pickle=False) as archive:
            arrays = {
                name: np.array(archive[name], copy=True) for name in archive.files
            }
    except (OSError, ValueError, TypeError) as error:
        raise ValueError(
            f"invalid NPZ (object/pickle arrays are forbidden) in {path.name}: {error}"
        ) from error
    for name, array in arrays.items():
        if array.dtype.kind not in "biuf":
            raise ValueError(
                f"object/pickle or non-real array forbidden: {path.name}:{name}"
            )
    return arrays


def _validate_array_metadata(
    manifest: dict, section: str, arrays: dict[str, np.ndarray]
) -> None:
    expected = manifest.get("arrays", {}).get(section)
    if not isinstance(expected, dict) or set(expected) != set(arrays):
        raise ValueError(f"invalid {section} array metadata")
    for name, array in arrays.items():
        item = expected[name]
        if item.get("dtype") != array.dtype.str or item.get("shape") != list(
            array.shape
        ):
            raise ValueError(f"dtype/shape mismatch for {section}:{name}")


def _validate_failure_inputs(arrays: dict[str, np.ndarray]) -> None:
    required = set(FAILURE_ARRAYS) | set(FAILURE_SCALARS)
    if set(arrays) != required:
        raise ValueError("failure_inputs.npz has missing or unexpected arrays")
    if any(not np.isfinite(array).all() for array in arrays.values()):
        raise ValueError("replay inputs must be finite")
    for name in FAILURE_SCALARS:
        if arrays[name].shape != ():
            raise ValueError(f"{name} must be scalar")
    for name in ("timestep", "nx", "nu", "nd", "n_params"):
        if arrays[name].dtype.kind not in "iu":
            raise ValueError(f"{name} must have integer dtype")
    nx, nu, nd, n_params = (
        int(arrays[name]) for name in ("nx", "nu", "nd", "n_params")
    )
    expected_shapes = {
        "x0": (nx,),
        "u": (nu,),
        "previous_control": (nu,),
        "requested_action": (nu,),
        "weather": (nd,),
        "sampled_parameters": (n_params,),
        "p_dyn": (nd + n_params,),
    }
    for name, shape in expected_shapes.items():
        if arrays[name].shape != shape:
            raise ValueError(f"bad shape for {name}: expected {shape}")
    if not np.array_equal(
        arrays["p_dyn"],
        np.concatenate((arrays["weather"], arrays["sampled_parameters"])),
    ):
        raise ValueError("p_dyn must exactly equal concat(weather, sampled_parameters)")


def _validate_history(arrays: dict[str, np.ndarray], rows: tuple[dict, ...]) -> None:
    if set(arrays) != set(HISTORY_ARRAYS):
        raise ValueError("history.npz has missing or unexpected arrays")
    lengths = {array.shape[0] if array.ndim else -1 for array in arrays.values()}
    if len(lengths) != 1 or -1 in lengths or lengths != {len(rows)}:
        raise ValueError("history row/array length mismatch")
    if arrays["step_index"].ndim != 1 or arrays["step_index"].dtype.kind not in "iu":
        raise ValueError("invalid history step_index")
    if arrays["reward"].ndim != 1 or not np.isfinite(arrays["reward"]).all():
        raise ValueError("invalid history reward")
    for name in ("done", "raw_next_observation_available"):
        if arrays[name].ndim != 1 or arrays[name].dtype.kind != "b":
            raise ValueError(f"invalid history {name}")
    for name in (
        "policy_observation",
        "raw_observation",
        "requested_action",
        "previous_control",
        "executed_control",
        "raw_next_observation",
    ):
        if arrays[name].ndim < 2:
            raise ValueError(f"invalid dimensions for history {name}")
    if arrays["raw_next_observation"].shape != arrays["raw_observation"].shape:
        raise ValueError("history raw-next shape mismatch")
    if not np.isfinite(arrays["raw_next_observation"]).all():
        raise ValueError("history raw-next values must be finite")
    unavailable = ~arrays["raw_next_observation_available"]
    if np.any(arrays["raw_next_observation"][unavailable] != 0):
        raise ValueError("unavailable raw-next rows must use zero fill")
    for index, row in enumerate(rows):
        if not isinstance(row, dict) or row.get("step_index") != int(
            arrays["step_index"][index]
        ):
            raise ValueError("history JSONL is not aligned with arrays")
        if row.get("done") is not bool(arrays["done"][index]):
            raise ValueError("history done value mismatch")
        if float(row.get("reward")) != float(arrays["reward"][index]):
            raise ValueError("history reward value mismatch")
        if not isinstance(row.get("metrics"), dict):
            raise ValueError("history metrics must be a dictionary")


def _load_failure_capsule(
    path: str | Path, *, check_directory_name: bool
) -> LoadedFailureCapsule:
    capsule_path = Path(path)
    if not capsule_path.is_dir():
        raise ValueError(f"failure capsule directory is missing: {capsule_path}")
    present = {item.name for item in capsule_path.iterdir() if item.is_file()}
    missing = REQUIRED_FILES - present
    if missing:
        raise ValueError(f"missing required capsule files: {sorted(missing)}")
    manifest = _read_json(capsule_path / "manifest.json")
    if (
        not isinstance(manifest, dict)
        or manifest.get("schema_version") != SCHEMA_VERSION
    ):
        raise ValueError("unsupported or missing manifest schema_version")
    files = manifest.get("files")
    expected_hashed = REQUIRED_FILES - {"manifest.json"}
    if not isinstance(files, dict) or set(files) != expected_hashed:
        raise ValueError("invalid manifest file checksum table")
    for name in sorted(expected_hashed):
        metadata = files[name]
        if not isinstance(metadata, dict) or metadata.get("sha256") != _sha256_file(
            capsule_path / name
        ):
            raise ValueError(f"checksum mismatch for {name}")
    failure_inputs = _load_npz(capsule_path / "failure_inputs.npz")
    history_arrays = _load_npz(capsule_path / "history.npz")
    _validate_failure_inputs(failure_inputs)
    _validate_array_metadata(manifest, "failure_inputs", failure_inputs)
    _validate_array_metadata(manifest, "history", history_arrays)
    rows = []
    try:
        for line in (
            (capsule_path / "history.jsonl").read_text(encoding="utf-8").splitlines()
        ):
            if not line:
                raise ValueError("blank JSONL row")
            rows.append(json.loads(line, parse_constant=_reject_constant))
    except (UnicodeError, json.JSONDecodeError, ValueError) as error:
        raise ValueError(f"invalid strict JSON in history.jsonl: {error}") from error
    history_rows = tuple(rows)
    _validate_history(history_arrays, history_rows)
    if manifest.get("history_length") != len(history_rows):
        raise ValueError("manifest history length mismatch")
    context = manifest.get("context")
    if not isinstance(context, dict):
        raise ValueError("manifest context is missing")
    context_fields = {
        "seed",
        "task_id",
        "inference_mode",
        "task",
        "checkpoint_path",
        "checkpoint_sha256",
        "git_head",
        "dirty",
        "source_checksums",
        "package_versions",
        "formal_result_root",
    }
    if not context_fields.issubset(context):
        raise ValueError("manifest context is incomplete")
    for name in (
        "task_id",
        "inference_mode",
        "checkpoint_path",
        "checkpoint_sha256",
        "git_head",
    ):
        if not isinstance(context[name], str):
            raise ValueError(f"manifest context {name} must be text")
    if not isinstance(context["task"], dict) or not isinstance(context["dirty"], bool):
        raise ValueError("manifest task/dirty context is invalid")
    if not isinstance(context["source_checksums"], dict) or not isinstance(
        context["package_versions"], dict
    ):
        raise ValueError("manifest source/package metadata is invalid")
    for name in (
        "task_id",
        "task",
        "checkpoint_path",
        "checkpoint_sha256",
        "source_checksums",
        "package_versions",
        "git_head",
        "dirty",
        "formal_result_root",
    ):
        if manifest.get(name) != context[name]:
            raise ValueError(f"manifest context mirror mismatch for {name}")
    expected_id = _failure_id(context, failure_inputs)
    if manifest.get("failure_id") != expected_id or (
        check_directory_name and capsule_path.name != expected_id
    ):
        raise ValueError("manifest failure ID mismatch")
    traceback_text = (capsule_path / "traceback.txt").read_text(encoding="utf-8")
    solver = manifest.get("solver")
    exception = manifest.get("exception")
    if not isinstance(solver, dict) or not isinstance(solver.get("options"), dict):
        raise ValueError("manifest solver metadata is invalid")
    if (
        not isinstance(exception, dict)
        or not isinstance(exception.get("type"), str)
        or not isinstance(exception.get("message"), str)
    ):
        raise ValueError("manifest exception metadata is invalid")
    content_identity = _content_identity(
        context,
        failure_inputs,
        history_arrays,
        history_rows,
        solver["options"],
        exception["type"],
        exception["message"],
        traceback_text,
    )
    if manifest.get("content_identity_sha256") != content_identity:
        raise ValueError("manifest content identity mismatch")
    return LoadedFailureCapsule(
        path=capsule_path,
        manifest=manifest,
        failure_inputs=failure_inputs,
        history_arrays=history_arrays,
        history_rows=history_rows,
        traceback_text=traceback_text,
    )


def load_failure_capsule(path: str | Path) -> LoadedFailureCapsule:
    return _load_failure_capsule(path, check_directory_name=True)


__all__ = [
    "CapsuleContext",
    "FailureCapsuleRecorder",
    "LoadedFailureCapsule",
    "load_failure_capsule",
]
