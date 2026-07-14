#!/usr/bin/env python3
"""Run the failure-tolerant unshielded online-context comparator."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import re
import shutil
import stat
import sys
from typing import Any, Callable, Mapping


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd

from experiments.scripts.run_context_ab import (
    APPROVED_SEEDS,
    DEFAULT_RESULT_ROOT as ORIGINAL_DIAGNOSTIC_ROOT,
    DIAGNOSTIC_TASK_IDS,
    HASH_FIELDS,
    MODES,
    PROVENANCE_FIELDS,
    RELEVANT_SOURCE_FIELDS,
    _evaluation_provenance,
    _package_versions,
    _provenance,
    build_diagnostic_runs,
    resume_row_is_complete,
    sha256_file,
)
from gl_gym.RL.agri_metarl import AgriMetaRL
from gl_gym.environments.models.utils import FORMAL_CVODES_OPTIONS
from gl_gym.experiments.context_ab import select_diagnostic_tasks
from gl_gym.experiments.ode_failure import (
    CapsuleContext,
    FailureCapsuleRecorder,
    load_failure_capsule,
)
from gl_gym.experiments.shield_evaluation import REQUIRED_METRICS
from gl_gym.experiments.suite_evaluation import (
    load_task_env,
    run_deterministic_episode,
    task_from_row,
)
from gl_gym.experiments.suite_schema import load_suite_manifest


DEFAULT_RESULT_ROOT = Path(
    "artifacts/results/AgriControl_C_2026-07-10-v3-context-ab-unshielded-comparator"
)
DEFAULT_FAILURE_ROOT = Path("artifacts/failures/context-ab-unshielded-comparator")
EARLY_HORIZON_FAILURE = re.compile(
    r"evaluation episode terminated before configured horizon: "
    r"step ([1-9]\d*) of ([1-9]\d*)"
)
STATUS_FIELDS = (
    "completed",
    "status",
    "ode_failure_count",
    "failure_evidence_path",
)
RESERVED_ROW_FIELDS = frozenset(
    {
        "seed",
        "task_id",
        "split",
        "inference_mode",
        "checkpoint_steps",
        "checkpoint_path",
        "model_path",
        "vecnormalize_path",
        "action_trace_path",
        "source_manifest",
        "source_tasks_csv",
        "runtime_source_tree_sha256",
        "action_trace_sha256",
        "failure_capsule_identity_sha256",
        "row_identity_sha256",
        *STATUS_FIELDS,
        *HASH_FIELDS,
        *PROVENANCE_FIELDS,
    }
)


def _overlaps(first: Path, second: Path) -> bool:
    return first == second or first in second.parents or second in first.parents


def _work_root(result_root: Path) -> Path:
    return result_root.parent / f".{result_root.name}.work"


def _failure_work_root(failure_root: Path, result_root: Path) -> Path:
    identity = hashlib.sha256(str(result_root.resolve()).encode("utf-8")).hexdigest()[:8]
    return failure_root / f".work-{identity}"


def validate_output_roots(
    result_root: str | Path,
    failure_root: str | Path,
    formal_result_root: str | Path,
    *,
    original_diagnostic_root: str | Path = ORIGINAL_DIAGNOSTIC_ROOT,
) -> tuple[Path, Path]:
    """Reject overlap among formal, original, comparator, and capsule lifecycles."""

    result = Path(result_root).resolve()
    failure = Path(failure_root).resolve()
    formal = Path(formal_result_root).resolve()
    original_path = Path(original_diagnostic_root)
    original = (
        original_path
        if original_path.is_absolute()
        else ROOT / original_path
    ).resolve()
    result_work = _work_root(result).resolve()
    failure_work = _failure_work_root(failure, result).resolve()
    protected = (formal, original)
    owned = (result, result_work, failure, failure_work)
    if any(_overlaps(item, source) for item in owned for source in protected):
        raise ValueError(
            "comparator result and failure roots must be disjoint from formal and "
            "original diagnostic roots"
        )
    if _overlaps(result, failure) or _overlaps(result_work, failure) or _overlaps(
        result, failure_work
    ):
        raise ValueError("comparator result and failure roots must be disjoint")
    return result, failure


def _trace_path(work: Path, seed: int, task_id: str, mode: str) -> Path:
    return (work / "traces" / f"seed{seed}__{task_id}__{mode}.npy").resolve()


def _attempt_root(failure_work: Path, seed: int, task_id: str, mode: str) -> Path:
    key = f"{seed}\0{task_id}\0{mode}".encode("utf-8")
    return failure_work / hashlib.sha256(key).hexdigest()[:12]


def _runtime_source_tree_sha256(root: str | Path = ROOT) -> str:
    """Fingerprint every regular Python module used from the runtime source tree."""

    source_root = Path(root).resolve() / "src" / "gl_gym"
    if not source_root.is_dir() or source_root.is_symlink():
        raise ValueError("runtime source tree must be a regular directory")
    reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    files: list[tuple[str, Path]] = []
    for path in source_root.rglob("*"):
        metadata = path.lstat()
        if path.is_symlink() or getattr(metadata, "st_file_attributes", 0) & reparse_flag:
            raise ValueError(f"runtime source tree contains symlink/reparse entry: {path}")
        if path.suffix == ".py":
            if not stat.S_ISREG(metadata.st_mode):
                raise ValueError(f"runtime Python source must be a regular file: {path}")
            files.append((path.relative_to(source_root).as_posix(), path))
    if not files:
        raise ValueError("runtime source tree contains no Python files")
    digest = hashlib.sha256()
    for relative, path in sorted(files):
        relative_bytes = relative.encode("utf-8")
        content = path.read_bytes()
        digest.update(len(relative_bytes).to_bytes(8, "big"))
        digest.update(relative_bytes)
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)
    return digest.hexdigest()


def _key(row: Mapping[str, Any]) -> tuple[int, str, str]:
    return (int(row["seed"]), str(row["task_id"]), str(row["inference_mode"]))


def _load_rows(path: Path | None) -> list[dict[str, Any]]:
    if path is None or not path.is_file():
        return []
    try:
        return pd.read_csv(path).to_dict(orient="records")
    except (OSError, UnicodeError, ValueError, pd.errors.ParserError, pd.errors.EmptyDataError):
        return []


def _canonical_scalar(name: str, value: Any) -> Any:
    if isinstance(value, np.generic):
        value = value.item()
    if name in {
        "failure_evidence_path",
        "action_trace_path",
        "action_trace_sha256",
        "failure_capsule_identity_sha256",
    } and (
        value == "" or (isinstance(value, float) and np.isnan(value))
    ):
        return {"empty": True}
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return {"nan": True}
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, float)):
        numeric = float(value)
        if not np.isfinite(numeric):
            return {"number": str(numeric)}
        return {"number": numeric.hex()}
    return str(value)


def _row_identity(row: Mapping[str, Any]) -> str:
    payload = {
        str(name): _canonical_scalar(str(name), value)
        for name, value in row.items()
        if name != "row_identity_sha256"
        and not (
            (
                value is None
                or (isinstance(value, (float, np.floating)) and np.isnan(value))
            )
            and name
            not in {
                "failure_evidence_path",
                "action_trace_path",
                "action_trace_sha256",
                "failure_capsule_identity_sha256",
            }
        )
    }
    canonical = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _signed_row(row: Mapping[str, Any]) -> dict[str, Any]:
    signed = dict(row)
    signed["row_identity_sha256"] = _row_identity(signed)
    return signed


def _replace_progress(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        pd.DataFrame(rows).to_csv(temporary, index=False)
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _finite_metrics(metrics: Any) -> dict[str, Any]:
    if not isinstance(metrics, Mapping):
        raise TypeError("episode metrics must be a mapping")
    normalized = dict(metrics)
    missing = set(REQUIRED_METRICS).difference(normalized)
    if missing:
        raise KeyError(f"episode metrics are missing required keys: {sorted(missing)}")
    try:
        values = np.asarray(
            [float(normalized[name]) for name in REQUIRED_METRICS], dtype=float
        )
    except (TypeError, ValueError) as error:
        raise TypeError("required episode metrics must be numeric scalars") from error
    if not np.isfinite(values).all():
        raise ValueError("successful episode metrics must be finite")
    for name, value in list(normalized.items()):
        try:
            scalar = float(value)
        except (TypeError, ValueError) as error:
            raise TypeError(f"episode metric {name!r} must be a numeric scalar") from error
        if isinstance(value, np.generic):
            normalized[name] = value.item()
        elif not isinstance(value, (int, float)):
            normalized[name] = scalar
    return normalized


def _checkpoint_steps(model: Any) -> int:
    try:
        value = float(model.num_timesteps)
    except (AttributeError, TypeError, ValueError) as error:
        raise ValueError("checkpoint steps must be a nonnegative exact integer") from error
    if not np.isfinite(value) or not value.is_integer() or value < 0:
        raise ValueError("checkpoint steps must be a nonnegative exact integer")
    return int(value)


def _success_diagnostics(
    diagnostics: Any, *, metric_names: set[str]
) -> tuple[dict[str, Any], np.ndarray]:
    if not isinstance(diagnostics, Mapping):
        raise TypeError("episode diagnostics must be a mapping")
    values = dict(diagnostics)
    missing = {
        "support_ready_step",
        "context_norm_mean",
        "context_norm_max",
        "action_trace",
    }.difference(values)
    if missing:
        raise KeyError(f"episode diagnostics are missing required keys: {sorted(missing)}")
    collisions = set(values).intersection(metric_names | RESERVED_ROW_FIELDS)
    if collisions:
        raise ValueError(
            "episode diagnostics collide with metric or reserved row fields: "
            f"{sorted(collisions)}"
        )
    trace = np.asarray(values.pop("action_trace"), dtype=np.float32)
    if trace.ndim != 2 or not trace.shape[0] or not trace.shape[1]:
        raise ValueError("action_trace must be a nonempty two-dimensional array")
    if not np.isfinite(trace).all():
        raise ValueError("action_trace must contain only finite values")
    for name in ("context_norm_mean", "context_norm_max"):
        if not np.isfinite(float(values[name])):
            raise ValueError(f"{name} must be finite")
    readiness = float(values["support_ready_step"])
    if not (np.isfinite(readiness) or np.isnan(readiness)):
        raise ValueError("support_ready_step must be finite or NaN")
    for name, value in list(values.items()):
        if isinstance(value, np.generic):
            values[name] = value.item()
        elif value is not None and not isinstance(value, (str, bool, int, float)):
            raise TypeError(f"diagnostic {name!r} must be a CSV-safe scalar")
    return values, trace


def _validate_completed_row(row: Mapping[str, Any]) -> None:
    """Validate the final merged row before it becomes resumable progress."""

    evidence = row.get("failure_evidence_path")
    evidence_is_empty = evidence == "" or (
        isinstance(evidence, float) and np.isnan(evidence)
    )
    if (
        row.get("completed") not in (True, np.bool_(True))
        or row.get("status") != "completed"
        or row.get("ode_failure_count") != 0
        or not evidence_is_empty
    ):
        raise ValueError("completed comparator row has an invalid status schema")
    try:
        metrics = np.asarray(
            [float(row[name]) for name in REQUIRED_METRICS], dtype=float
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("completed comparator row has invalid required metrics") from error
    if not np.isfinite(metrics).all():
        raise ValueError("completed comparator row required metrics must be finite")


def _valid_readiness(row: Mapping[str, Any], trace_length: int) -> bool:
    try:
        readiness = float(row["support_ready_step"])
    except (KeyError, TypeError, ValueError):
        return False
    if str(row.get("inference_mode")) == "online_context":
        return bool(
            np.isfinite(readiness)
            and readiness.is_integer()
            and 1 <= readiness < trace_length
        )
    if str(row.get("inference_mode")) != "zero_context":
        return False
    return bool(
        np.isnan(readiness)
        or (
            np.isfinite(readiness)
            and readiness.is_integer()
            and 1 <= readiness < trace_length
        )
    )


def _base_matches(row: Mapping[str, Any], expected: Mapping[str, Any]) -> bool:
    try:
        for name, value in expected.items():
            actual = row[name]
            if isinstance(value, (bool, np.bool_)):
                if isinstance(actual, str):
                    if actual not in {"True", "False"}:
                        return False
                    actual = actual == "True"
                if not isinstance(actual, (bool, np.bool_)) or bool(actual) != bool(value):
                    return False
            elif isinstance(value, int):
                numeric = float(actual)
                if not np.isfinite(numeric) or not numeric.is_integer() or int(numeric) != value:
                    return False
            elif str(actual) != str(value):
                return False
        identity = str(row["row_identity_sha256"])
        return len(identity) == 64 and identity == _row_identity(row)
    except (KeyError, OverflowError, TypeError, ValueError):
        return False


def _completed_row_is_valid(
    row: Mapping[str, Any], *, expected: Mapping[str, Any], trace_path: Path
) -> bool:
    if not _base_matches(row, expected):
        return False
    try:
        _validate_completed_row(row)
        if Path(str(row["action_trace_path"])).resolve() != trace_path.resolve():
            return False
        trace = np.load(trace_path, allow_pickle=False)
        if (
            trace.ndim != 2
            or not trace.shape[0]
            or not trace.shape[1]
            or not np.isfinite(trace).all()
            or sha256_file(trace_path) != str(row["action_trace_sha256"])
            or not _valid_readiness(row, trace.shape[0])
        ):
            return False
        context = np.asarray(
            [float(row["context_norm_mean"]), float(row["context_norm_max"])],
            dtype=float,
        )
        return bool(np.isfinite(context).all())
    except (KeyError, OSError, TypeError, ValueError):
        return False


def _validate_capsule(
    manifest_path: Path,
    *,
    expected_context: CapsuleContext,
    error: Exception,
    capsule_loader: Callable[[str | Path], Any],
) -> Any:
    match = EARLY_HORIZON_FAILURE.fullmatch(str(error))
    if type(error) is not RuntimeError or match is None:
        raise ValueError("caught exception is not the exact early-horizon RuntimeError")
    episode_step, horizon = (int(item) for item in match.groups())
    if episode_step >= horizon:
        raise ValueError("early-horizon RuntimeError has inconsistent step/horizon")
    capsule = _validate_capsule_evidence(
        manifest_path,
        expected_context=expected_context,
        capsule_loader=capsule_loader,
    )
    try:
        recorded_step = int(capsule.history_arrays["step_index"][-1])
        failure_timestep = int(capsule.failure_inputs["timestep"])
    except (KeyError, IndexError, TypeError, ValueError) as capsule_error:
        raise ValueError("failure capsule has malformed step evidence") from capsule_error
    if recorded_step != episode_step - 1 or failure_timestep != recorded_step:
        raise ValueError("failure capsule timestep does not match the early episode step")
    return capsule


def _validate_capsule_evidence(
    manifest_path: Path,
    *,
    expected_context: CapsuleContext,
    capsule_loader: Callable[[str | Path], Any],
) -> Any:
    capsule = capsule_loader(manifest_path.parent)
    expected = asdict(expected_context)
    if capsule.manifest.get("context") != expected:
        raise ValueError("failure capsule context does not match this comparator attempt")
    exception = capsule.manifest.get("exception", {})
    if not exception.get("type") or not exception.get("message"):
        raise ValueError("failure capsule underlying exception fields are empty")
    if (
        exception["type"] not in capsule.traceback_text
        or exception["message"] not in capsule.traceback_text
    ):
        raise ValueError("failure capsule traceback does not bind its underlying exception")
    try:
        recorded_step = int(capsule.history_arrays["step_index"][-1])
        failure_timestep = int(capsule.failure_inputs["timestep"])
    except (KeyError, IndexError, TypeError, ValueError) as capsule_error:
        raise ValueError("failure capsule has malformed step evidence") from capsule_error
    if recorded_step < 0 or failure_timestep != recorded_step:
        raise ValueError("failure capsule has inconsistent step evidence")
    if capsule.manifest.get("solver", {}).get("options") != dict(
        FORMAL_CVODES_OPTIONS
    ):
        raise ValueError("failure capsule does not prove formal CVODES options")
    identity = capsule.manifest.get("content_identity_sha256")
    if not isinstance(identity, str) or len(identity) != 64:
        raise ValueError("failure capsule content identity is invalid")
    return capsule


def _failed_row_is_valid(
    row: Mapping[str, Any],
    *,
    expected: Mapping[str, Any],
    attempt: Path,
    context: CapsuleContext,
    capsule_loader: Callable[[str | Path], Any],
) -> bool:
    if not _base_matches(row, expected):
        return False
    try:
        if (
            row["completed"] not in (False, np.bool_(False))
            or str(row["status"]) != "ode_failure"
            or int(row["ode_failure_count"]) != 1
        ):
            return False
        metrics = np.asarray([float(row[name]) for name in REQUIRED_METRICS])
        diagnostics = np.asarray(
            [
                float(row["support_ready_step"]),
                float(row["context_norm_mean"]),
                float(row["context_norm_max"]),
            ]
        )
        if not np.isnan(metrics).all() or not np.isnan(diagnostics).all():
            return False
        trace_value = row["action_trace_path"]
        if not (
            trace_value == ""
            or (isinstance(trace_value, float) and np.isnan(trace_value))
        ):
            return False
        manifest = Path(str(row["failure_evidence_path"])).resolve()
        manifests = sorted(attempt.rglob("manifest.json")) if attempt.is_dir() else []
        if len(manifests) != 1 or manifests[0].resolve() != manifest:
            return False
        capsule = _validate_capsule_evidence(
            manifest, expected_context=context, capsule_loader=capsule_loader
        )
        return str(row["failure_capsule_identity_sha256"]) == str(
            capsule.manifest["content_identity_sha256"]
        )
    except (KeyError, OSError, TypeError, ValueError):
        return False


def run_unshielded_comparator(
    *,
    suite: Any,
    tasks: pd.DataFrame,
    runs: list[dict[str, Any]],
    result_root: str | Path,
    failure_root: str | Path,
    source_manifest: str | Path,
    source_tasks_csv: str | Path,
    device: str,
    resume: bool,
    legacy_progress: str | Path | None = None,
    model_loader: Callable[[Path, str], Any] | None = None,
    env_loader: Callable[[Any, Any, Path], Any] = load_task_env,
    episode_runner: Callable[..., Any] = run_deterministic_episode,
    provenance_loader: Callable[[], dict[str, Any]] = _provenance,
    recorder_factory: Callable[[str | Path, CapsuleContext], Any] = FailureCapsuleRecorder,
    capsule_loader: Callable[[str | Path], Any] = load_failure_capsule,
) -> pd.DataFrame:
    """Run all approved keys, retaining only strictly proven ODE failures."""

    root, capsule_root = validate_output_roots(
        result_root, failure_root, suite.result_root
    )
    if root.exists():
        raise ValueError("Task-1 comparator must not publish or reuse the final result root")
    work = _work_root(root)
    failure_work = _failure_work_root(capsule_root, root)
    if not resume:
        for path in (work, failure_work):
            if path.exists():
                shutil.rmtree(path)
    progress_path = work / "progress.csv"
    work.joinpath("traces").mkdir(parents=True, exist_ok=True)
    resume_rows = _load_rows(progress_path) if resume else []
    legacy_rows = _load_rows(Path(legacy_progress)) if legacy_progress is not None else []

    selected = select_diagnostic_tasks(tasks)
    task_records = [task_from_row(row) for row in selected.itertuples(index=False)]
    provenance = _evaluation_provenance(
        source_manifest, source_tasks_csv, dict(provenance_loader())
    )
    source_checksums = {
        str((ROOT / path).resolve()): provenance[name]
        for name, path in RELEVANT_SOURCE_FIELDS
    }
    source_checksums.update(
        {
            str(Path(source_manifest).resolve()): provenance["source_manifest_sha256"],
            str(Path(source_tasks_csv).resolve()): provenance["source_tasks_sha256"],
        }
    )
    packages = _package_versions()
    runtime_source_tree_sha256 = _runtime_source_tree_sha256()
    evidence_by_seed = {
        int(run["seed"]): {
            "model_sha256": str(run["model_sha256"]),
            "vecnormalize_sha256": str(run["vecnormalize_sha256"]),
            **provenance,
        }
        for run in runs
    }
    expected_seeds = tuple(int(run["seed"]) for run in runs)
    if expected_seeds != APPROVED_SEEDS:
        raise ValueError("approved comparator requires runs for seeds exactly 42 123")
    load_model = model_loader or (
        lambda path, selected_device: AgriMetaRL.load(str(path), device=selected_device)
    )
    rows: list[dict[str, Any]] = []
    for run in runs:
        seed = int(run["seed"])
        model = load_model(Path(run["model_path"]), device)
        checkpoint_steps = _checkpoint_steps(model)
        for task in task_records:
            for mode in MODES:
                attempt = _attempt_root(failure_work, seed, task.task_id, mode)
                context = CapsuleContext(
                    seed=seed,
                    task_id=task.task_id,
                    inference_mode=mode,
                    task=asdict(task),
                    checkpoint_path=str(Path(run["model_path"]).resolve()),
                    checkpoint_sha256=str(run["model_sha256"]),
                    git_head=str(provenance["git_commit"]),
                    dirty=bool(provenance["dirty"]),
                    source_checksums=dict(source_checksums),
                    package_versions=dict(packages),
                    formal_result_root=str(root),
                )
                base = {
                    "seed": seed,
                    "task_id": task.task_id,
                    "split": task.split,
                    "inference_mode": mode,
                    "checkpoint_steps": checkpoint_steps,
                    "checkpoint_path": str(Path(run["model_path"]).resolve()),
                    "model_path": str(Path(run["model_path"]).resolve()),
                    "vecnormalize_path": str(
                        Path(run["vecnormalize_path"]).resolve()
                    ),
                    "source_manifest": str(Path(source_manifest).resolve()),
                    "source_tasks_csv": str(Path(source_tasks_csv).resolve()),
                    "runtime_source_tree_sha256": runtime_source_tree_sha256,
                    **evidence_by_seed[seed],
                }
                key = (seed, task.task_id, mode)
                trace_path = _trace_path(work, seed, task.task_id, mode)
                reused: dict[str, Any] | None = None
                for candidate in resume_rows:
                    try:
                        if _key(candidate) != key:
                            continue
                    except (KeyError, TypeError, ValueError):
                        continue
                    status = str(candidate.get("status"))
                    if status == "completed" and not attempt.exists() and _completed_row_is_valid(
                        candidate, expected=base, trace_path=trace_path
                    ):
                        reused = dict(candidate)
                        reused["failure_evidence_path"] = ""
                        reused["failure_capsule_identity_sha256"] = ""
                        break
                    if status == "ode_failure" and _failed_row_is_valid(
                        candidate,
                        expected=base,
                        attempt=attempt,
                        context=context,
                        capsule_loader=capsule_loader,
                    ):
                        reused = dict(candidate)
                        reused["action_trace_path"] = ""
                        reused["action_trace_sha256"] = ""
                        break
                if reused is not None:
                    rows.append(reused)
                    continue

                for candidate in legacy_rows:
                    if set(STATUS_FIELDS).intersection(candidate):
                        continue
                    try:
                        candidate_key = _key(candidate)
                    except (KeyError, TypeError, ValueError):
                        continue
                    if candidate_key != key or str(candidate.get("split")) != task.split:
                        continue
                    if not resume_row_is_complete(
                        candidate,
                        checkpoint_steps=checkpoint_steps,
                        expected_hashes=evidence_by_seed[seed],
                    ):
                        continue
                    source_trace = Path(str(candidate["action_trace_path"])).resolve()
                    trace_path.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(source_trace, trace_path)
                    imported = dict(candidate)
                    for name in (
                        "row_identity_sha256",
                        "action_trace_sha256",
                        "failure_capsule_identity_sha256",
                        "failure_evidence_path",
                        "status",
                        "completed",
                        "ode_failure_count",
                    ):
                        imported.pop(name, None)
                    imported.update(base)
                    imported.update(
                        {
                            "action_trace_path": str(trace_path),
                            "action_trace_sha256": sha256_file(trace_path),
                            "failure_capsule_identity_sha256": "",
                            "completed": True,
                            "status": "completed",
                            "ode_failure_count": 0,
                            "failure_evidence_path": "",
                        }
                    )
                    imported = _signed_row(imported)
                    if not _completed_row_is_valid(
                        imported, expected=base, trace_path=trace_path
                    ):
                        trace_path.unlink(missing_ok=True)
                        continue
                    reused = imported
                    break
                if reused is not None:
                    rows.append(reused)
                    _replace_progress(rows, progress_path)
                    continue

                if attempt.exists():
                    shutil.rmtree(attempt)
                recorder = recorder_factory(attempt, context)
                env = env_loader(suite, task, Path(run["vecnormalize_path"]))
                error: Exception | None = None
                close_failed = False
                metrics: Any = None
                diagnostics: Any = None
                try:
                    try:
                        metrics, diagnostics = episode_runner(
                            model,
                            env,
                            inference_mode=mode,
                            return_diagnostics=True,
                            failure_recorder=recorder,
                        )
                    except Exception as caught:
                        error = caught
                finally:
                    primary = sys.exception() or error
                    try:
                        env.close()
                    except Exception as close_error:
                        if primary is None:
                            raise
                        close_failed = True
                        primary.add_note(
                            "environment close also failed: "
                            f"{type(close_error).__name__}: {close_error}"
                        )
                if close_failed:
                    assert error is not None
                    raise error
                manifests = sorted(attempt.rglob("manifest.json")) if attempt.exists() else []
                if error is not None:
                    try:
                        if len(manifests) != 1:
                            raise ValueError(
                                "expected exactly one new failure capsule, "
                                f"found {len(manifests)}"
                            )
                        capsule = _validate_capsule(
                            manifests[0],
                            expected_context=context,
                            error=error,
                            capsule_loader=capsule_loader,
                        )
                    except Exception as classification_error:
                        if attempt.exists():
                            shutil.rmtree(attempt)
                        error.add_note(
                            "ODE failure classification rejected: "
                            f"{type(classification_error).__name__}: {classification_error}"
                        )
                        raise error
                    row = _signed_row({
                        **{name: float("nan") for name in REQUIRED_METRICS},
                        "support_ready_step": float("nan"),
                        "context_norm_mean": float("nan"),
                        "context_norm_max": float("nan"),
                        "action_trace_path": "",
                        "action_trace_sha256": "",
                        "failure_capsule_identity_sha256": str(
                            capsule.manifest["content_identity_sha256"]
                        ),
                        **base,
                        "completed": False,
                        "status": "ode_failure",
                        "ode_failure_count": 1,
                        "failure_evidence_path": str(manifests[0].resolve()),
                    })
                else:
                    if manifests:
                        shutil.rmtree(attempt)
                        raise ValueError(
                            "successful comparator episode unexpectedly produced a failure capsule"
                        )
                    normalized_metrics = _finite_metrics(metrics)
                    normalized_diagnostics, trace = _success_diagnostics(
                        diagnostics, metric_names=set(normalized_metrics)
                    )
                    np.save(trace_path, trace, allow_pickle=False)
                    row = _signed_row({
                        **normalized_metrics,
                        **normalized_diagnostics,
                        "action_trace_path": str(trace_path),
                        "action_trace_sha256": sha256_file(trace_path),
                        "failure_capsule_identity_sha256": "",
                        **base,
                        "completed": True,
                        "status": "completed",
                        "ode_failure_count": 0,
                        "failure_evidence_path": "",
                    })
                    _validate_completed_row(row)
                    if not _valid_readiness(row, trace.shape[0]):
                        raise ValueError(
                            "support_ready_step is invalid for the inference mode"
                        )
                    if attempt.exists():
                        shutil.rmtree(attempt)
                rows.append(row)
                _replace_progress(rows, progress_path)
    if len(rows) != 32:
        raise RuntimeError(f"comparator completed {len(rows)} of 32 required episodes")
    return pd.DataFrame(rows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source_manifest", required=True)
    parser.add_argument("--source_tasks_csv", required=True)
    parser.add_argument("--model_root", required=True)
    parser.add_argument("--result_root", default=str(DEFAULT_RESULT_ROOT))
    parser.add_argument("--failure_root", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--legacy_progress")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    source_manifest = Path(args.source_manifest)
    source_tasks = Path(args.source_tasks_csv)
    if not source_manifest.is_file():
        raise FileNotFoundError(f"source manifest does not exist: {source_manifest}")
    if not source_tasks.is_file():
        raise FileNotFoundError(f"source task CSV does not exist: {source_tasks}")
    suite = load_suite_manifest(source_manifest)
    runs = build_diagnostic_runs(args.model_root, args.seeds)
    run_unshielded_comparator(
        suite=suite,
        tasks=pd.read_csv(source_tasks),
        runs=runs,
        result_root=args.result_root,
        failure_root=args.failure_root,
        source_manifest=source_manifest,
        source_tasks_csv=source_tasks,
        device=args.device,
        resume=args.resume,
        legacy_progress=args.legacy_progress,
    )


if __name__ == "__main__":
    main()
