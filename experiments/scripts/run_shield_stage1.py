#!/usr/bin/env python3
"""Validate the fixed action shield against one captured ODE failure."""

from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
import math
from numbers import Integral, Real
import os
from pathlib import Path
import shutil
import stat
import subprocess
import sys
import tempfile
from time import perf_counter
from types import SimpleNamespace
from typing import Any, Callable, Mapping
import uuid


ROOT = Path(__file__).resolve().parents[2]
RULE_CONFIG_PATH = ROOT / "configs" / "agents" / "rule_based.yml"
SOURCE_ROOT = str(ROOT / "src")
if SOURCE_ROOT not in sys.path:
    sys.path.insert(0, SOURCE_ROOT)


import numpy as np
import yaml

from gl_gym.environments.action_shield import (
    DEFAULT_LAMBDAS,
    ActionShieldConfig,
    control_to_reference_action,
    project_first_feasible,
)
from gl_gym.environments.models.utils import FORMAL_CVODES_OPTIONS, define_model
from gl_gym.experiments.ode_failure import load_failure_capsule
import gl_gym.experiments.ode_replay as ode_replay_module
from gl_gym.experiments.ode_replay import build_rule_based_controller


SCHEMA_VERSION = ActionShieldConfig().schema_version
METHOD = "conservative_feasibility_shield_v2"
OUTPUT_NAMES = frozenset({"stage1_results.json", "stage1_states.npz", "decision.json"})
CONDITION_NAMES = (
    "original_reproduced",
    "legal_candidate_succeeded",
    "first_successful_candidate_selected",
    "intervention_recorded",
)
FINGERPRINT_FIELDS = (
    "schema_version",
    "method",
    "fixed_lambdas",
    "source_checksums",
    "capsule_source_checksums",
    "git_head",
    "dirty",
    "capsule_git_head",
    "capsule_dirty",
    "formal_solver_options",
    "env_config_sha256",
    "rule_config_sha256",
    "capsule_identity_sha256",
    "checkpoint_sha256",
    "failure_timestep",
    "delta_u_max",
    "original_outcome",
    "candidate_attempts",
    "reference_action",
    "reference_control",
    "requested_action",
    "requested_control",
    "executed_action",
    "executed_control",
    "selected_lambda",
    "conditions",
    "outcome",
)

MECHANISM_FIELDS = frozenset(
    {
        "original_outcome", "candidate_attempts", "reference_action",
        "reference_control", "requested_action", "requested_control",
        "executed_action", "executed_control", "selected_lambda",
        "delta_u_max", "conditions", "outcome",
    }
)


class _ConstructionFailure(BaseException):
    def __init__(self, error: Exception) -> None:
        self.error = error
        self.traceback = error.__traceback__


class _PostCallFailure(BaseException):
    def __init__(self, error: Exception) -> None:
        self.error = error
        self.traceback = error.__traceback__


def _shield_fingerprint(evidence: Mapping[str, Any]) -> str:
    missing = [name for name in FINGERPRINT_FIELDS if name not in evidence]
    if missing:
        raise ValueError(f"stage-1 shield fingerprint inputs are missing: {missing}")
    payload = {name: evidence[name] for name in FINGERPRINT_FIELDS}
    return hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capsule_manifest", required=True)
    parser.add_argument(
        "--env_config", default=str(ROOT / "configs" / "envs" / "TomatoEnv.yml")
    )
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--formal_result_root")
    return parser


def _overlaps(first: Path, second: Path) -> bool:
    return first == second or first in second.parents or second in first.parents


def _is_reparse(path: Path) -> bool:
    metadata = path.lstat()
    attributes = getattr(metadata, "st_file_attributes", 0)
    return path.is_symlink() or bool(
        attributes & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    )


def _reject_reparse_components(path: Path, *, include_leaf: bool) -> None:
    candidate = path.absolute()
    parts = (candidate, *candidate.parents)
    for index, component in enumerate(parts):
        if index == 0 and not include_leaf:
            continue
        if component.exists() or component.is_symlink():
            if _is_reparse(component):
                raise ValueError(
                    f"path contains a symlink or reparse point: {component}"
                )


def _formal_root(manifest: Mapping[str, Any]) -> Path:
    context = manifest.get("context")
    if not isinstance(context, Mapping):
        raise ValueError("capsule manifest context must be a mapping")
    value = context.get("formal_result_root")
    if not isinstance(value, (str, os.PathLike)) or not str(value).strip():
        raise ValueError("capsule manifest context formal_result_root is missing")
    unresolved = Path(value).expanduser().absolute()
    _reject_reparse_components(unresolved, include_leaf=True)
    return unresolved.resolve()


def _collides_with_publication_sibling(source: Path, output: Path) -> bool:
    for candidate in (source, *source.parents):
        if candidate.parent != output.parent:
            continue
        prefixes = (
            f".{output.name}.stage-",
            f".{output.name}.staging-",
            f".{output.name}.backup-",
        )
        if candidate.name.startswith(prefixes):
            return True
    return False


def _validate_paths(
    manifest_path: str | Path,
    capsule: Any,
    output_root: str | Path,
    formal_result_root: str | Path | None,
) -> tuple[Path, Path, Path]:
    manifest = Path(manifest_path).expanduser().absolute()
    if manifest.name != "manifest.json":
        raise ValueError("capsule_manifest must resolve to manifest.json")
    _reject_reparse_components(manifest, include_leaf=True)
    manifest = manifest.resolve()
    capsule_directory = Path(capsule.path).expanduser().resolve()
    if manifest != capsule_directory / "manifest.json":
        raise ValueError("loaded capsule path does not match capsule_manifest")
    formal = _formal_root(capsule.manifest)
    if formal_result_root is not None:
        supplied_path = Path(formal_result_root).expanduser().absolute()
        _reject_reparse_components(supplied_path, include_leaf=True)
        supplied = supplied_path.resolve()
        if supplied != formal:
            raise ValueError(
                "formal_result_root must exactly match capsule context formal_result_root"
            )
    output_unresolved = Path(output_root).expanduser().absolute()
    _reject_reparse_components(output_unresolved, include_leaf=True)
    output = output_unresolved.resolve()
    if output.exists() and not output.is_dir():
        raise ValueError("output_root exists as a file")
    for source, label in ((capsule_directory, "capsule"), (formal, "formal result")):
        if _overlaps(output, source) or _collides_with_publication_sibling(
            source, output
        ):
            raise ValueError(f"output_root must be disjoint from the {label} root")
    return manifest, output, formal


def _capture_output_topology(
    output: Path, capsule_directory: Path, formal: Path
) -> dict[str, Any]:
    parent = output.parent.absolute()
    parent.mkdir(parents=True, exist_ok=True)
    _reject_reparse_components(parent, include_leaf=True)
    metadata = parent.lstat()
    if _is_reparse(parent) or not stat.S_ISDIR(metadata.st_mode):
        raise ValueError("output parent identity must be a regular directory")
    topology = {
        "parent": parent,
        "resolved_parent": parent.resolve(),
        "identity": (int(metadata.st_dev), int(metadata.st_ino)),
        "reparse": _is_reparse(parent),
        "output_name": output.name,
        "capsule": capsule_directory.resolve(),
        "formal": formal.resolve(),
    }
    _require_output_topology(topology)
    return topology


def _require_output_topology(topology: Mapping[str, Any]) -> None:
    parent = Path(topology["parent"])
    try:
        _reject_reparse_components(parent, include_leaf=True)
        metadata = parent.lstat()
        observed_identity = (int(metadata.st_dev), int(metadata.st_ino))
        if (
            _is_reparse(parent) != topology["reparse"]
            or topology["reparse"]
            or not stat.S_ISDIR(metadata.st_mode)
            or parent.resolve() != topology["resolved_parent"]
            or observed_identity != topology["identity"]
        ):
            raise ValueError("changed")
        capsule = Path(topology["capsule"])
        formal = Path(topology["formal"])
        for source in (capsule, formal):
            _reject_reparse_components(source, include_leaf=True)
        output = (parent / str(topology["output_name"])).resolve()
        if output.exists() and not output.is_dir():
            raise ValueError("output root became a file")
        for source in (capsule.resolve(), formal.resolve()):
            if _overlaps(output, source) or _collides_with_publication_sibling(
                source, output
            ):
                raise ValueError("output overlap changed")
    except (OSError, ValueError, KeyError, TypeError) as error:
        raise ValueError(
            "output parent identity or protected topology changed"
        ) from error


def _output_topology_matches(topology: Mapping[str, Any]) -> bool:
    try:
        _require_output_topology(topology)
    except ValueError:
        return False
    return True


def _positive_int(inputs: Mapping[str, Any], name: str) -> int:
    value = np.asarray(inputs.get(name))
    if value.shape != () or value.dtype.kind not in "iu":
        raise ValueError(f"{name} must be a positive non-boolean integer")
    result = int(value)
    if result <= 0:
        raise ValueError(f"{name} must be positive")
    return result


def _nonnegative_int(inputs: Mapping[str, Any], name: str) -> int:
    value = np.asarray(inputs.get(name))
    if value.shape != () or value.dtype.kind not in "iu":
        raise ValueError(f"{name} must be a non-negative non-boolean integer")
    result = int(value)
    if result < 0:
        raise ValueError(f"{name} must be non-negative")
    return result


def _finite_scalar(inputs: Mapping[str, Any], name: str) -> float:
    value = np.asarray(inputs.get(name))
    if value.shape != () or value.dtype.kind not in "iuf":
        raise ValueError(f"{name} must be a finite non-boolean scalar")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _finite_vector(value: Any, *, name: str, size: int) -> np.ndarray:
    raw = np.asarray(value)
    if raw.shape != (size,) or raw.dtype.kind not in "iuf":
        raise ValueError(f"{name} must be an exact finite vector of shape ({size},)")
    result = np.array(raw, copy=True)
    if not np.isfinite(result).all():
        raise ValueError(f"{name} must be an exact finite vector of shape ({size},)")
    return result


def _snapshot_inputs(source: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(source, Mapping):
        raise TypeError("capsule failure_inputs must be a mapping")
    required = {
        "x0",
        "u",
        "previous_control",
        "requested_action",
        "weather",
        "sampled_parameters",
        "p_dyn",
        "dt",
        "day_of_year",
        "hour_of_day",
        "timestep",
        "nx",
        "nu",
        "nd",
        "n_params",
    }
    missing = required.difference(source)
    if missing:
        raise ValueError(
            f"capsule failure_inputs missing: {', '.join(sorted(missing))}"
        )
    nx = _positive_int(source, "nx")
    nu = _positive_int(source, "nu")
    nd = _positive_int(source, "nd")
    n_params = _positive_int(source, "n_params")
    timestep = _nonnegative_int(source, "timestep")
    dt = _finite_scalar(source, "dt")
    if dt <= 0.0:
        raise ValueError("dt must be positive")
    day = _finite_scalar(source, "day_of_year")
    hour = _finite_scalar(source, "hour_of_day")
    result: dict[str, Any] = {
        "nx": nx,
        "nu": nu,
        "nd": nd,
        "n_params": n_params,
        "dt": dt,
        "day_of_year": day,
        "hour_of_day": hour,
        "timestep": timestep,
    }
    result["x0"] = _finite_vector(source["x0"], name="x0", size=nx)
    for name in ("u", "previous_control", "requested_action"):
        result[name] = _finite_vector(source[name], name=name, size=nu)
    if np.any(result["requested_action"] < -1) or np.any(
        result["requested_action"] > 1
    ):
        raise ValueError("requested_action must lie within [-1, 1]")
    result["weather"] = _finite_vector(source["weather"], name="weather", size=nd)
    result["sampled_parameters"] = _finite_vector(
        source["sampled_parameters"], name="sampled_parameters", size=n_params
    )
    result["p_dyn"] = _finite_vector(source["p_dyn"], name="p_dyn", size=nd + n_params)
    if not np.array_equal(
        result["p_dyn"],
        np.concatenate((result["weather"], result["sampled_parameters"])),
    ):
        raise ValueError(
            "p_dyn must exactly equal weather concatenated with sampled_parameters"
        )
    return result


def _load_env_config(
    path: str | Path, nu: int
) -> tuple[str, np.ndarray, np.ndarray, np.ndarray]:
    unresolved = Path(path).expanduser().absolute()
    _reject_reparse_components(unresolved, include_leaf=True)
    config_path = unresolved.resolve()
    try:
        snapshot = config_path.read_bytes()
        loaded = yaml.safe_load(snapshot.decode("utf-8", errors="strict"))
    except (OSError, UnicodeError, yaml.YAMLError) as error:
        raise ValueError(f"invalid environment config: {error}") from error
    if not isinstance(loaded, Mapping) or not isinstance(
        loaded.get("GreenLightEnv"), Mapping
    ):
        raise ValueError("environment config requires a GreenLightEnv mapping")
    env = loaded["GreenLightEnv"]
    raw_u_min = np.asarray(env.get("u_min"))
    raw_u_max = np.asarray(env.get("u_max"))
    for raw, name in ((raw_u_min, "u_min"), (raw_u_max, "u_max")):
        if (
            raw.shape != (nu,)
            or raw.dtype.kind not in "iuf"
            or not np.isfinite(raw).all()
        ):
            raise ValueError(f"{name} must be an exact finite vector of shape ({nu},)")
    u_min = np.array(env["u_min"], dtype=np.float32)
    u_max = np.array(env["u_max"], dtype=np.float32)
    if not np.isfinite(u_min).all() or not np.isfinite(u_max).all():
        raise ValueError("u_min and u_max must remain finite as float32 vectors")
    if not np.all(u_min < u_max):
        raise ValueError("u_min must be strictly less than u_max in every channel")
    raw_delta = env.get("delta_u_max")
    delta_array = np.asarray(raw_delta)
    if delta_array.shape != () or delta_array.dtype.kind not in "iuf":
        raise ValueError("delta_u_max must be a positive finite scalar")
    delta_scalar = float(delta_array)
    if not math.isfinite(delta_scalar) or delta_scalar <= 0.0:
        raise ValueError("delta_u_max must be a positive finite scalar")
    delta = np.ones(nu, dtype=np.float32) * delta_scalar
    if not np.isfinite(delta).all() or np.any(delta <= 0):
        raise ValueError("delta_u_max must remain positive and finite as float32")
    return hashlib.sha256(snapshot).hexdigest(), u_min, u_max, delta


def _snapshot_rule_config(
    path: str | Path = RULE_CONFIG_PATH,
) -> tuple[str, dict[str, Any]]:
    unresolved = Path(path).expanduser().absolute()
    _reject_reparse_components(unresolved, include_leaf=True)
    config_path = unresolved.resolve()
    try:
        with config_path.open("rb") as stream:
            snapshot = stream.read()
        loaded = yaml.safe_load(snapshot.decode("utf-8", errors="strict"))
    except (OSError, UnicodeError, yaml.YAMLError) as error:
        raise ValueError(f"invalid rule controller config: {error}") from error
    if (
        not isinstance(loaded, Mapping)
        or not isinstance(loaded.get("TomatoEnv"), Mapping)
        or not loaded["TomatoEnv"]
    ):
        raise ValueError(
            "rule controller config requires a nonempty TomatoEnv parameter mapping"
        )
    return (
        hashlib.sha256(snapshot).hexdigest(),
        deepcopy(dict(loaded["TomatoEnv"])),
    )


def _build_default_controller_from_snapshot(
    controller_factory: Callable[[], Any], params: Mapping[str, Any]
) -> Any:
    """Call the declared builder with its config dependency narrowly frozen."""

    original_loader = ode_replay_module.load_model_hyperparams

    def snapshot_loader(agent_name: str, env_name: str) -> dict[str, Any]:
        if (agent_name, env_name) != ("rule_based", "TomatoEnv"):
            raise ValueError("unexpected model hyperparameter request during stage-1")
        return deepcopy(dict(params))

    ode_replay_module.load_model_hyperparams = snapshot_loader
    try:
        return controller_factory()
    finally:
        ode_replay_module.load_model_hyperparams = original_loader


def _state_from_result(result: Any, nx: int) -> np.ndarray:
    if not isinstance(result, Mapping) or "xf" not in result:
        raise ValueError("integrator result must contain final state 'xf'")
    raw = result["xf"]
    if hasattr(raw, "full"):
        raw = raw.full()
    state = np.asarray(raw)
    if state.shape == (nx, 1):
        state = state[:, 0]
    if state.shape != (nx,) or state.dtype.kind not in "iuf":
        raise ValueError(f"final state must be a finite vector of shape ({nx},)")
    final = np.array(state, dtype=np.float64, copy=True)
    if not np.isfinite(final).all():
        raise ValueError(f"final state must be a finite vector of shape ({nx},)")
    return final


def _control(
    action: np.ndarray,
    previous: np.ndarray,
    delta: np.ndarray,
    u_min: np.ndarray,
    u_max: np.ndarray,
) -> np.ndarray:
    return np.clip(previous + action * delta, u_min, u_max)


def _validate_formal_solver_provenance(manifest: Mapping[str, Any]) -> None:
    solver = manifest.get("solver")
    if not isinstance(solver, Mapping) or set(solver) != {"options"}:
        raise ValueError("capsule solver must contain exactly the options mapping")
    options = solver["options"]
    expected = dict(FORMAL_CVODES_OPTIONS)
    if not isinstance(options, Mapping) or set(options) != set(expected):
        raise ValueError("capsule solver options must use the exact formal key set")
    for name in ("abstol", "reltol"):
        value = options[name]
        if (
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, Real)
            or not math.isfinite(float(value))
            or value != expected[name]
        ):
            raise ValueError(f"capsule solver option {name} must match formal settings")
    max_steps = options["max_num_steps"]
    if (
        isinstance(max_steps, (bool, np.bool_))
        or not isinstance(max_steps, Integral)
        or max_steps != expected["max_num_steps"]
    ):
        raise ValueError(
            "capsule solver option max_num_steps must match formal settings"
        )


def _provenance(manifest: Mapping[str, Any]) -> dict[str, Any]:
    identity = manifest.get("content_identity_sha256")
    failure_id = manifest.get("failure_id")
    if not isinstance(failure_id, str) or not failure_id:
        raise ValueError("capsule failure_id must be non-empty text")
    if (
        not isinstance(identity, str)
        or len(identity) != 64
        or any(character not in "0123456789abcdef" for character in identity)
    ):
        raise ValueError("capsule identity must be lowercase SHA-256 hex")
    checkpoint_path = manifest.get("checkpoint_path")
    checkpoint_sha = manifest.get("checkpoint_sha256")
    sources = manifest.get("source_checksums")
    if not isinstance(checkpoint_path, str) or not checkpoint_path:
        raise ValueError("capsule checkpoint_path must be non-empty text")
    if (
        not isinstance(checkpoint_sha, str)
        or len(checkpoint_sha) != 64
        or any(character not in "0123456789abcdef" for character in checkpoint_sha)
    ):
        raise ValueError("capsule checkpoint_sha256 must be lowercase SHA-256 hex")
    if not isinstance(sources, Mapping):
        raise ValueError("capsule source_checksums must be a mapping")
    if any(
        not isinstance(name, str)
        or not name
        or not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
        for name, value in sources.items()
    ):
        raise ValueError(
            "capsule source_checksums must map names to lowercase SHA-256 hex"
        )
    git_head = manifest.get("git_head")
    dirty = manifest.get("dirty")
    if (
        not isinstance(git_head, str)
        or len(git_head) not in (40, 64)
        or any(character not in "0123456789abcdef" for character in git_head)
    ):
        raise ValueError("capsule git_head must be canonical lowercase hex")
    if type(dirty) is not bool:
        raise ValueError("capsule dirty provenance must be boolean")
    return {
        "failure_id": failure_id,
        "capsule_identity_sha256": identity,
        "checkpoint_path": checkpoint_path,
        "checkpoint_sha256": checkpoint_sha,
        "capsule_source_checksums": dict(sources),
        "capsule_git_head": git_head,
        "capsule_dirty": dirty,
    }


def _repository_provenance() -> dict[str, Any]:
    def git(*arguments: str) -> str:
        completed = subprocess.run(
            ["git", *arguments], cwd=ROOT, text=True,
            capture_output=True, check=True,
        )
        return completed.stdout.strip()

    return {
        "git_commit": git("rev-parse", "HEAD"),
        "dirty": bool(git("status", "--porcelain")),
    }


def _execution_source_checksums(
    capsule_sources: Mapping[str, str]
) -> dict[str, str]:
    current: dict[str, str] = {}
    for name in capsule_sources:
        path = Path(name).expanduser()
        if not path.is_absolute():
            raise ValueError(
                f"Stage-1 execution source path must be absolute: {name}"
            )
        path = path.absolute()
        if path.is_symlink() or not path.is_file() or _is_reparse(path):
            raise ValueError(
                f"Stage-1 execution source must be a regular file and not a symlink: {name}"
            )
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        current[name] = digest.hexdigest()
    if not current:
        raise ValueError("Stage-1 execution sources contain no absolute regular files")
    return current


def _validate_execution_provenance(value: Mapping[str, Any]) -> tuple[str, bool]:
    git_commit = value.get("git_commit")
    dirty = value.get("dirty")
    if (
        not isinstance(git_commit, str)
        or len(git_commit) not in (40, 64)
        or any(character not in "0123456789abcdef" for character in git_commit)
    ):
        raise ValueError("Stage-1 execution git_commit must be canonical lowercase hex")
    if type(dirty) is not bool:
        raise ValueError("Stage-1 execution dirty provenance must be boolean")
    return git_commit, dirty


def _factory_kwargs(inputs: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "nx": inputs["nx"],
        "nu": inputs["nu"],
        "nd": inputs["nd"],
        "n_params": inputs["n_params"],
        "dt": inputs["dt"],
        "integrator_options": dict(FORMAL_CVODES_OPTIONS),
    }


def _run_original(inputs: Mapping[str, Any], integrator_factory: Callable[..., Any]):
    started = perf_counter()
    integrator = integrator_factory(**_factory_kwargs(inputs))
    if not callable(integrator):
        raise TypeError("integrator_factory must return a callable integrator")
    try:
        result = integrator(
            x0=np.array(inputs["x0"], copy=True),
            u=np.array(inputs["u"], copy=True),
            p=np.array(inputs["p_dyn"], copy=True),
        )
    except Exception as error:
        return (
            True,
            None,
            {
                "success": False,
                "exception_type": type(error).__name__,
                "exception_message": str(error),
                "elapsed_seconds": perf_counter() - started,
            },
        )
    state = _state_from_result(result, inputs["nx"])
    return (
        False,
        state,
        {
            "success": True,
            "exception_type": None,
            "exception_message": None,
            "elapsed_seconds": perf_counter() - started,
        },
    )


def _attempt_records(attempts: Any, controls: list[np.ndarray]) -> list[dict[str, Any]]:
    if len(attempts) != len(controls):
        raise ValueError("candidate attempt/control evidence is inconsistent")
    return [
        {
            "lambda": float(attempt.lambda_value),
            "action": attempt.action.tolist(),
            "control": control.tolist(),
            "success": bool(attempt.success),
            "exception_type": attempt.exception_type,
            "exception_message": attempt.exception_message,
            "elapsed_seconds": float(attempt.elapsed_seconds),
        }
        for attempt, control in zip(attempts, controls, strict=True)
    ]


_ATTEMPT_FIELDS = {
    "lambda", "action", "control", "success", "exception_type",
    "exception_message", "elapsed_seconds",
}


def _stage1_lambda(value: Any, *, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite real number")
    try:
        result = float(value)
    except (OverflowError, TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a finite fixed-grid value") from error
    if not math.isfinite(result):
        raise ValueError(f"{name} must be a finite fixed-grid value")
    return result


def _stage1_attempt_vector(value: Any, *, name: str) -> np.ndarray:
    raw = np.asarray(value)
    if (
        raw.ndim != 1
        or raw.size == 0
        or not np.issubdtype(raw.dtype, np.number)
        or np.issubdtype(raw.dtype, np.bool_)
        or np.issubdtype(raw.dtype, np.complexfloating)
    ):
        raise ValueError(f"{name} must be a nonempty finite numeric vector")
    result = np.asarray(raw, dtype=np.float64)
    if not np.isfinite(result).all():
        raise ValueError(f"{name} must be a nonempty finite numeric vector")
    return result


def _validate_stage1_candidate_attempts(
    attempts: Any,
    selected_lambda: Any,
    *,
    require_success: bool,
) -> None:
    if not isinstance(attempts, list):
        raise TypeError("candidate_attempts must be a list")
    if not attempts:
        if require_success:
            raise ValueError("candidate_attempts must contain the successful fixed-grid prefix")
        if selected_lambda is not None:
            raise ValueError("selected_lambda requires successful candidate_attempts")
        return
    if len(attempts) > len(DEFAULT_LAMBDAS):
        raise ValueError("candidate_attempts must be a descending fixed-grid prefix")

    vector_shape: tuple[int, ...] | None = None
    success_indices: list[int] = []
    for index, attempt in enumerate(attempts):
        if not isinstance(attempt, Mapping) or set(attempt) != _ATTEMPT_FIELDS:
            raise ValueError("each candidate attempt must contain the exact Stage-1 fields")
        lam = _stage1_lambda(attempt["lambda"], name="candidate attempt lambda")
        if lam != DEFAULT_LAMBDAS[index]:
            raise ValueError("candidate_attempts must be a descending fixed-grid prefix")
        action = _stage1_attempt_vector(attempt["action"], name="candidate attempt action")
        control = _stage1_attempt_vector(attempt["control"], name="candidate attempt control")
        if action.shape != control.shape:
            raise ValueError("candidate attempt action/control shapes must match")
        if vector_shape is None:
            vector_shape = action.shape
        elif action.shape != vector_shape:
            raise ValueError("candidate attempt vector shapes must be consistent")
        success = attempt["success"]
        if type(success) is not bool:
            raise TypeError("candidate attempt success must be a strict bool")
        elapsed = attempt["elapsed_seconds"]
        if isinstance(elapsed, (bool, np.bool_)) or not isinstance(elapsed, Real):
            raise TypeError("candidate attempt elapsed_seconds must be numeric")
        try:
            finite_elapsed = float(elapsed)
        except (OverflowError, TypeError, ValueError) as error:
            raise ValueError("candidate attempt elapsed_seconds must be finite and nonnegative") from error
        if not math.isfinite(finite_elapsed) or finite_elapsed < 0.0:
            raise ValueError("candidate attempt elapsed_seconds must be finite and nonnegative")
        exception_type = attempt["exception_type"]
        exception_message = attempt["exception_message"]
        if success:
            success_indices.append(index)
            if exception_type is not None or exception_message is not None:
                raise ValueError("successful candidate attempt cannot contain an exception")
        elif not isinstance(exception_type, str) or not isinstance(exception_message, str):
            raise ValueError("failed candidate attempt must contain exception strings")

    if require_success:
        if success_indices != [len(attempts) - 1]:
            raise ValueError("only the final selected candidate attempt may succeed")
        selected = _stage1_lambda(selected_lambda, name="selected_lambda")
        expected_count = DEFAULT_LAMBDAS.index(selected) + 1 if selected in DEFAULT_LAMBDAS else 0
        if expected_count == 0 or len(attempts) != expected_count:
            raise ValueError("selected_lambda must equal the final descending-prefix candidate")
    elif success_indices or selected_lambda is not None:
        raise ValueError("unsuccessful Stage-1 evidence cannot select a successful candidate")


def _validate_stage1_mechanism(report: Mapping[str, Any]) -> dict[str, bool]:
    missing = sorted(MECHANISM_FIELDS.difference(report))
    if missing:
        raise ValueError(f"stage-1 mechanism evidence is missing fields: {missing}")

    original = report["original_outcome"]
    outcome_fields = {
        "success", "exception_type", "exception_message", "elapsed_seconds",
    }
    if not isinstance(original, Mapping) or set(original) != outcome_fields:
        raise ValueError("original_outcome must contain exact mechanism fields")
    if type(original["success"]) is not bool:
        raise ValueError("original_outcome success must be a strict bool")
    if original["success"]:
        raise ValueError("original_outcome must record the original action failure")
    if not isinstance(original["exception_type"], str) or not isinstance(
        original["exception_message"], str
    ):
        raise ValueError("original_outcome failure must contain exception strings")
    elapsed = original["elapsed_seconds"]
    if isinstance(elapsed, (bool, np.bool_)) or not isinstance(elapsed, Real):
        raise ValueError("original_outcome elapsed_seconds must be finite and nonnegative")
    try:
        finite_elapsed = float(elapsed)
    except (OverflowError, TypeError, ValueError) as error:
        raise ValueError(
            "original_outcome elapsed_seconds must be finite and nonnegative"
        ) from error
    if not math.isfinite(finite_elapsed) or finite_elapsed < 0.0:
        raise ValueError("original_outcome elapsed_seconds must be finite and nonnegative")

    _validate_stage1_candidate_attempts(
        report["candidate_attempts"], report["selected_lambda"], require_success=True
    )
    selected = _stage1_lambda(report["selected_lambda"], name="selected_lambda")
    vectors = {
        name: _stage1_attempt_vector(report[name], name=f"stage-1 {name}")
        for name in (
            "requested_action", "reference_action", "executed_action",
            "requested_control", "reference_control", "executed_control",
            "delta_u_max",
        )
    }
    action_shape = vectors["requested_action"].shape
    if any(vectors[name].shape != action_shape for name in vectors):
        raise ValueError("stage-1 mechanism action/control shapes must match")
    if np.any(vectors["delta_u_max"] <= 0.0):
        raise ValueError("stage-1 delta_u_max must be strictly positive")

    for attempt in report["candidate_attempts"]:
        lam = _stage1_lambda(attempt["lambda"], name="candidate attempt lambda")
        expected_action = (
            (1.0 - lam) * vectors["requested_action"]
            + lam * vectors["reference_action"]
        )
        action = _stage1_attempt_vector(
            attempt["action"], name="candidate attempt action"
        )
        if not np.array_equal(action, expected_action):
            raise ValueError("candidate attempt action is inconsistent with mechanism")

    selected_attempt = report["candidate_attempts"][-1]
    selected_action = _stage1_attempt_vector(
        selected_attempt["action"], name="selected candidate action"
    )
    selected_control = _stage1_attempt_vector(
        selected_attempt["control"], name="selected candidate control"
    )
    if not np.array_equal(selected_action, vectors["executed_action"]):
        raise ValueError("executed_action must equal the selected candidate attempt")
    if not np.array_equal(selected_control, vectors["executed_control"]):
        raise ValueError("executed_control must equal the selected candidate attempt")

    conditions = report["conditions"]
    if (
        not isinstance(conditions, dict)
        or set(conditions) != set(CONDITION_NAMES)
        or any(type(value) is not bool for value in conditions.values())
    ):
        raise ValueError("invalid stage-1 decision conditions")
    expected_conditions = {
        "original_reproduced": True,
        "legal_candidate_succeeded": True,
        "first_successful_candidate_selected": True,
        "intervention_recorded": bool(
            selected > 0.0
            and not np.array_equal(
                vectors["requested_action"], vectors["executed_action"]
            )
        ),
    }
    if conditions != expected_conditions:
        raise ValueError("stage-1 conditions do not match recomputed mechanism evidence")
    expected_outcome = (
        "continue_to_context_ab"
        if all(expected_conditions.values())
        else "redesign_action_shield"
    )
    if report["outcome"] != expected_outcome:
        raise ValueError("stage-1 outcome is inconsistent with mechanism conditions")
    return expected_conditions


def _validate_report(
    report: Mapping[str, Any], x0: np.ndarray, selected_state: np.ndarray | None
) -> None:
    if report.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("invalid stage-1 report schema")
    if report.get("method") != METHOD:
        raise ValueError("invalid stage-1 method")
    if report.get("fixed_lambdas") != list(DEFAULT_LAMBDAS):
        raise ValueError("invalid stage-1 descending fixed lambda priority")
    fingerprint = report.get("shield_fingerprint")
    if (
        not isinstance(fingerprint, str)
        or len(fingerprint) != 64
        or any(character not in "0123456789abcdef" for character in fingerprint)
    ):
        raise ValueError("invalid stage-1 shield fingerprint")
    _validate_stage1_mechanism(report)
    if fingerprint != _shield_fingerprint(report):
        raise ValueError("stage-1 shield fingerprint does not bind its provenance")
    if selected_state is not None and selected_state.shape != x0.shape:
        raise ValueError("selected final state dimension is inconsistent with x0")
    json.dumps(report, allow_nan=False)


def _write_outputs(
    stage: Path,
    report: Mapping[str, Any],
    x0: np.ndarray,
    selected_state: np.ndarray | None,
) -> None:
    decision = {
        key: report[key]
        for key in (
            "schema_version",
            "method",
            "fixed_lambdas",
            "shield_fingerprint",
            "outcome",
            "conditions",
            "selected_lambda",
        )
    }
    (stage / "stage1_results.json").write_text(
        json.dumps(
            report, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False
        )
        + "\n",
        encoding="utf-8",
    )
    np.savez(
        stage / "stage1_states.npz",
        x0=np.array(x0, dtype=np.float64, copy=True),
        selected_final_state=(
            np.array(selected_state, dtype=np.float64, copy=True)
            if selected_state is not None
            else np.empty(0, dtype=np.float64)
        ),
        selected_available=np.array(selected_state is not None, dtype=np.bool_),
    )
    (stage / "decision.json").write_text(
        json.dumps(
            decision, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False
        )
        + "\n",
        encoding="utf-8",
    )


def _validate_stage(stage: Path, report: Mapping[str, Any]) -> None:
    entries = list(stage.iterdir())
    if {entry.name for entry in entries} != OUTPUT_NAMES:
        raise ValueError("stage-1 output must contain exactly three artifacts")
    if any(_is_reparse(entry) or not entry.is_file() for entry in entries):
        raise ValueError("stage-1 artifacts must be regular non-reparse files")
    loaded = json.loads(
        (stage / "stage1_results.json").read_text(encoding="utf-8"),
        parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)),
    )
    if loaded != report:
        raise ValueError("serialized stage-1 report does not match validated evidence")
    decision = json.loads((stage / "decision.json").read_text(encoding="utf-8"))
    decision_fields = {
        "schema_version", "method", "fixed_lambdas", "shield_fingerprint",
        "outcome", "conditions", "selected_lambda",
    }
    if set(decision) != decision_fields:
        raise ValueError("decision.json has invalid keys")
    if decision != {key: report[key] for key in decision_fields}:
        raise ValueError("decision.json is inconsistent with stage1_results.json")
    with np.load(stage / "stage1_states.npz", allow_pickle=False) as archive:
        if set(archive.files) != {"x0", "selected_final_state", "selected_available"}:
            raise ValueError("stage1_states.npz has invalid fields")
        x0 = archive["x0"]
        selected = archive["selected_final_state"]
        available = archive["selected_available"]
        if x0.dtype != np.float64 or x0.ndim != 1 or not np.isfinite(x0).all():
            raise ValueError("invalid stored x0")
        if available.dtype != np.bool_ or available.shape != ():
            raise ValueError("invalid selected_available flag")
        if (
            selected.dtype != np.float64
            or selected.ndim != 1
            or not np.isfinite(selected).all()
        ):
            raise ValueError("invalid selected final state")
        if bool(available) != (selected.size > 0):
            raise ValueError("selected state availability is inconsistent")


def _trees_equal(source: Path, target: Path) -> bool:
    left = {item.relative_to(source): item for item in source.rglob("*")}
    right = {item.relative_to(target): item for item in target.rglob("*")}
    if set(left) != set(right):
        return False
    for name, item in left.items():
        other = right[name]
        if item.is_file() != other.is_file() or item.is_dir() != other.is_dir():
            return False
        if item.is_file() and item.read_bytes() != other.read_bytes():
            return False
    return True


def _restore(
    backup: Path,
    output: Path,
    primary: BaseException,
    topology: Mapping[str, Any],
) -> bool:
    try:
        _require_output_topology(topology)
    except ValueError as topology_error:
        if hasattr(primary, "add_note"):
            primary.add_note(
                "backup preserved because output topology changed before restoration: "
                f"{topology_error}"
            )
        return False
    try:
        _require_output_topology(topology)
        os.replace(backup, output)
        return True
    except BaseException as error:
        if output.exists() and not backup.exists():
            return True
        if hasattr(primary, "add_note"):
            primary.add_note(f"atomic backup rename restoration failed: {error}")
    try:
        _require_output_topology(topology)
        if output.exists():
            raise RuntimeError("output root exists before fallback restoration")
        shutil.copytree(backup, output, copy_function=shutil.copy2, symlinks=True)
        if not _trees_equal(backup, output):
            raise OSError("fallback restoration verification failed")
    except Exception as error:
        if hasattr(primary, "add_note"):
            primary.add_note(
                f"fallback copy restoration failed; sole backup preserved: {error}"
            )
        if output.exists() and _output_topology_matches(topology):
            shutil.rmtree(output, ignore_errors=True)
        return False
    try:
        shutil.rmtree(backup)
    except Exception:
        pass
    return True


def _publish_atomic(
    output: Path,
    report: Mapping[str, Any],
    x0: np.ndarray,
    selected_state: np.ndarray | None,
    topology: Mapping[str, Any],
) -> Path:
    _require_output_topology(topology)
    stage = Path(tempfile.mkdtemp(prefix=f".{output.name}.stage-", dir=output.parent))
    backup = output.parent / f".{output.name}.backup-{uuid.uuid4().hex}"
    published = False
    replacement_intended = False
    primary_error: BaseException | None = None
    try:
        _write_outputs(stage, report, x0, selected_state)
        _validate_stage(stage, report)
        if output.exists():
            replacement_intended = True
            _require_output_topology(topology)
            os.replace(output, backup)
        _require_output_topology(topology)
        os.replace(stage, output)
        _require_output_topology(topology)
        published = True
        if _output_topology_matches(topology) and backup.exists():
            try:
                shutil.rmtree(backup)
            except Exception:
                pass
        return output
    except BaseException as error:
        primary_error = error
        if backup.exists() and not output.exists():
            _restore(backup, output, error, topology)
        raise
    finally:
        if _output_topology_matches(topology) and stage.exists():
            shutil.rmtree(stage, ignore_errors=True)
        if (
            replacement_intended
            and not published
            and _output_topology_matches(topology)
            and backup.exists()
            and not output.exists()
        ):
            fallback_primary = primary_error or RuntimeError(
                "stage-1 publication did not complete"
            )
            _restore(backup, output, fallback_primary, topology)


def run_stage1(
    capsule_path: str | Path,
    env_config: str | Path,
    output_root: str | Path,
    formal_result_root: str | Path | None = None,
    capsule_loader: Callable[[str | Path], Any] = load_failure_capsule,
    integrator_factory: Callable[..., Any] = define_model,
    controller_factory: Callable[[], Any] = build_rule_based_controller,
    provenance_loader: Callable[[], Mapping[str, Any]] = _repository_provenance,
) -> Path:
    manifest_path = Path(capsule_path).expanduser().absolute()
    if manifest_path.name != "manifest.json":
        raise ValueError("capsule_path must resolve to manifest.json")
    _reject_reparse_components(manifest_path, include_leaf=True)
    capsule = capsule_loader(manifest_path.parent.resolve())
    _, output, formal = _validate_paths(
        manifest_path, capsule, output_root, formal_result_root
    )
    output_topology = _capture_output_topology(
        output, Path(capsule.path).expanduser().resolve(), formal
    )
    _validate_formal_solver_provenance(capsule.manifest)
    inputs = _snapshot_inputs(capsule.failure_inputs)
    env_config_sha256, u_min, u_max, delta = _load_env_config(env_config, inputs["nu"])
    rule_config_sha256, rule_controller_params = _snapshot_rule_config(RULE_CONFIG_PATH)
    expected_control = _control(
        inputs["requested_action"], inputs["previous_control"], delta, u_min, u_max
    )
    if not np.array_equal(inputs["u"], expected_control):
        raise ValueError(
            "stored original u does not match configured action-to-control mapping"
        )
    provenance = _provenance(capsule.manifest)
    git_head, dirty = _validate_execution_provenance(provenance_loader())
    provenance.update({
        "source_checksums": _execution_source_checksums(
            provenance["capsule_source_checksums"]
        ),
        "git_head": git_head,
        "dirty": dirty,
    })

    reproduced, _, original = _run_original(inputs, integrator_factory)
    reference_action = None
    reference_control = None
    selected_state = None
    selected_lambda = None
    executed_action = np.array(inputs["requested_action"], copy=True)
    executed_control = np.array(inputs["u"], copy=True)
    attempts: list[dict[str, Any]] = []
    if reproduced:
        if controller_factory is build_rule_based_controller:
            controller = _build_default_controller_from_snapshot(
                controller_factory, rule_controller_params
            )
        else:
            controller = controller_factory()
        environment = SimpleNamespace(
            nu=inputs["nu"],
            day_of_year=inputs["day_of_year"],
            hour_of_day=inputs["hour_of_day"],
        )
        target = _finite_vector(
            controller.predict(
                np.array(inputs["x0"], copy=True),
                np.array(inputs["weather"], copy=True),
                environment,
            ),
            name="rule controller target",
            size=inputs["nu"],
        )
        reference_action = control_to_reference_action(
            target, inputs["previous_control"], delta
        )
        reference_control = np.array(target, copy=True)
        candidate_controls: list[np.ndarray] = []

        def integrate(candidate_action: np.ndarray) -> np.ndarray:
            try:
                control = _control(
                    np.array(candidate_action, copy=True),
                    inputs["previous_control"],
                    delta,
                    u_min,
                    u_max,
                )
                candidate_controls.append(np.array(control, copy=True))
                integrator = integrator_factory(**_factory_kwargs(inputs))
                if not callable(integrator):
                    raise TypeError(
                        "integrator_factory must return a callable integrator"
                    )
            except Exception as error:
                raise _ConstructionFailure(error) from None
            result = integrator(
                x0=np.array(inputs["x0"], copy=True),
                u=np.array(control, copy=True),
                p=np.array(inputs["p_dyn"], copy=True),
            )
            try:
                return _state_from_result(result, inputs["nx"])
            except Exception as error:
                raise _PostCallFailure(error) from None

        try:
            projection = project_first_feasible(
                inputs["requested_action"],
                reference_action,
                integrate,
                ActionShieldConfig(),
            )
        except _ConstructionFailure as failure:
            raise failure.error.with_traceback(failure.traceback) from None
        except _PostCallFailure as failure:
            raise failure.error.with_traceback(failure.traceback) from None
        attempts = _attempt_records(projection.attempts, candidate_controls)
        if projection.selected is not None:
            selected_lambda = float(projection.selected.lambda_value)
            selected_state = np.array(projection.final_state, copy=True)
            executed_action = np.array(projection.selected.action, copy=True)
            executed_control = np.array(candidate_controls[-1], copy=True)

    legal_succeeded = selected_state is not None
    first_success = bool(
        legal_succeeded
        and attempts
        and attempts[-1]["success"]
        and selected_lambda == attempts[-1]["lambda"]
        and not any(attempt["success"] for attempt in attempts[:-1])
    )
    intervention = bool(
        selected_lambda is not None
        and selected_lambda > 0.0
        and not np.array_equal(inputs["requested_action"], executed_action)
    )
    conditions = {
        "original_reproduced": bool(reproduced),
        "legal_candidate_succeeded": legal_succeeded,
        "first_successful_candidate_selected": first_success,
        "intervention_recorded": intervention,
    }
    outcome = (
        "continue_to_context_ab"
        if all(conditions.values())
        else "redesign_action_shield"
    )
    report = {
        "schema_version": SCHEMA_VERSION,
        "method": METHOD,
        **provenance,
        "failure_timestep": inputs["timestep"],
        "formal_solver_options": dict(FORMAL_CVODES_OPTIONS),
        "env_config_sha256": env_config_sha256,
        "rule_config_sha256": rule_config_sha256,
        "fixed_lambdas": list(DEFAULT_LAMBDAS),
        "delta_u_max": delta.tolist(),
        "original_outcome": original,
        "candidate_attempts": attempts,
        "reference_action": (
            None if reference_action is None else reference_action.tolist()
        ),
        "reference_control": (
            None if reference_control is None else reference_control.tolist()
        ),
        "requested_action": inputs["requested_action"].tolist(),
        "requested_control": inputs["u"].tolist(),
        "executed_action": executed_action.tolist(),
        "executed_control": executed_control.tolist(),
        "selected_lambda": selected_lambda,
        "conditions": conditions,
        "outcome": outcome,
    }
    report["shield_fingerprint"] = _shield_fingerprint(report)
    _validate_report(report, inputs["x0"], selected_state)
    return _publish_atomic(
        output, report, inputs["x0"], selected_state, output_topology
    )


def main() -> None:
    args = build_parser().parse_args()
    output = run_stage1(
        args.capsule_manifest,
        args.env_config,
        args.output_root,
        formal_result_root=args.formal_result_root,
    )
    print(output)


if __name__ == "__main__":
    main()
