#!/usr/bin/env python3
"""Run the preregistered Stage-2 shielded context A/B diagnostic."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
from typing import Any, Callable, Mapping


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd
import yaml

from experiments.scripts.run_context_ab import (
    APPROVED_SEEDS,
    RELEVANT_SOURCE_FIELDS,
    _evaluation_provenance,
    _package_versions,
    _provenance,
    build_diagnostic_runs,
    sha256_file,
)
from gl_gym.RL.agri_metarl import AgriMetaRL
from gl_gym.environments.action_shield import DEFAULT_LAMBDAS
from gl_gym.environments.models.utils import FORMAL_CVODES_OPTIONS
from gl_gym.experiments.context_ab import (
    DIAGNOSTIC_TASK_IDS,
    MODES,
    select_diagnostic_tasks,
)
from gl_gym.experiments.ode_failure import (
    CapsuleContext,
    FailureCapsuleRecorder,
    load_failure_capsule,
)
from gl_gym.experiments.shield_evaluation import (
    REQUIRED_METRICS,
    aggregate_episode_interventions,
    build_paired_shield_deltas,
    evaluate_shield_gate,
    write_shield_artifacts_atomic,
)
from gl_gym.experiments.suite_evaluation import (
    load_task_env,
    run_deterministic_episode,
    task_from_row,
)
from gl_gym.experiments.suite_schema import load_suite_manifest


METHOD = "minimal_feasibility_shield_v1"
SCHEMA_VERSION = "shielded-context-ab-stage2-v1"
STAGE1_SCHEMA_VERSION = "action-shield-stage1-v1"
STAGE1_CONDITIONS = (
    "original_reproduced",
    "legal_candidate_succeeded",
    "smallest_success_selected",
    "intervention_recorded",
)
STAGE1_OUTPUTS = frozenset(
    {"stage1_results.json", "stage1_states.npz", "decision.json"}
)
RULE_CONFIG_PATH = ROOT / "configs" / "agents" / "rule_based.yml"
ENV_CONFIG_PATH = ROOT / "configs" / "envs" / "TomatoEnv.yml"
DEFAULT_RESULT_ROOT = Path(
    "artifacts/results/AgriControl_C_2026-07-10-v3-shielded-context-ab"
)
HASH_FIELDS = (
    "model_sha256",
    "vecnormalize_sha256",
    "source_manifest_sha256",
    "source_tasks_sha256",
    *(name for name, _ in RELEVANT_SOURCE_FIELDS),
    "evaluation_provenance_sha256",
    "rule_config_sha256",
    "env_config_sha256",
    "formal_solver_options_sha256",
    "fixed_lambdas_sha256",
    "stage1_results_sha256",
    "stage1_states_sha256",
    "stage1_decision_sha256",
    "stage1_capsule_identity_sha256",
    "shield_fingerprint",
)


def _strict_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)),
        )
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as error:
        raise ValueError(f"invalid strict JSON artifact: {path}") from error
    if not isinstance(value, dict):
        raise ValueError(f"JSON artifact must contain a mapping: {path}")
    return value


def _lower_sha(value: Any, *, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be lowercase SHA-256 hex")
    return value


def load_stage1_prerequisite(stage1_root: str | Path) -> dict[str, Any]:
    """Load and strictly validate the immutable three-file Stage-1 decision."""

    root = Path(stage1_root).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Stage-1 root does not exist: {root}")
    entries = list(root.iterdir())
    if {entry.name for entry in entries} != STAGE1_OUTPUTS or any(
        not entry.is_file() or entry.is_symlink() for entry in entries
    ):
        raise ValueError("Stage-1 root must contain exactly three regular artifacts")
    report = _strict_json(root / "stage1_results.json")
    decision = _strict_json(root / "decision.json")
    if set(decision) != {"outcome", "conditions", "selected_lambda"}:
        raise ValueError("Stage-1 decision must contain exact keys")
    if report.get("schema_version") != STAGE1_SCHEMA_VERSION:
        raise ValueError("invalid Stage-1 results schema")
    conditions = report.get("conditions")
    if (
        not isinstance(conditions, dict)
        or tuple(conditions) != STAGE1_CONDITIONS
        or any(value is not True for value in conditions.values())
    ):
        raise ValueError("Stage-1 conditions must be the four exact passing conditions")
    selected = report.get("selected_lambda")
    if (
        isinstance(selected, bool)
        or not isinstance(selected, (int, float))
        or not np.isfinite(selected)
        or float(selected) <= 0.0
        or float(selected) not in DEFAULT_LAMBDAS
    ):
        raise ValueError("Stage-1 selected_lambda must be a positive fixed-grid value")
    expected_decision = {
        "outcome": "continue_to_context_ab",
        "conditions": conditions,
        "selected_lambda": selected,
    }
    if report.get("outcome") != "continue_to_context_ab" or decision != expected_decision:
        raise ValueError("Stage-1 outcome/decision is inconsistent with passing evidence")
    required = {
        "failure_id", "capsule_identity_sha256", "checkpoint_path",
        "checkpoint_sha256", "source_checksums", "git_head", "dirty",
        "formal_solver_options", "env_config_sha256", "rule_config_sha256",
        "fixed_lambdas",
    }
    missing = sorted(required.difference(report))
    if missing:
        raise ValueError(f"Stage-1 results schema is missing provenance: {missing}")
    _lower_sha(report["capsule_identity_sha256"], name="capsule identity")
    _lower_sha(report["checkpoint_sha256"], name="checkpoint hash")
    _lower_sha(report["env_config_sha256"], name="environment config hash")
    _lower_sha(report["rule_config_sha256"], name="rule config hash")
    if type(report["dirty"]) is not bool:
        raise ValueError("Stage-1 dirty provenance must be boolean")
    if report["formal_solver_options"] != dict(FORMAL_CVODES_OPTIONS):
        raise ValueError("Stage-1 formal solver options are stale")
    if report["fixed_lambdas"] != list(DEFAULT_LAMBDAS):
        raise ValueError("Stage-1 fixed lambda grid is stale")
    sources = report["source_checksums"]
    if not isinstance(sources, dict) or not sources:
        raise ValueError("Stage-1 source checksums must be a nonempty mapping")
    for name, checksum in sources.items():
        if not isinstance(name, str) or not name:
            raise ValueError("Stage-1 source checksum names must be nonempty strings")
        _lower_sha(checksum, name=f"Stage-1 source checksum {name}")
    with np.load(root / "stage1_states.npz", allow_pickle=False) as archive:
        if set(archive.files) != {"x0", "selected_final_state", "selected_available"}:
            raise ValueError("Stage-1 states archive has invalid fields")
        x0 = archive["x0"]
        selected_state = archive["selected_final_state"]
        available = archive["selected_available"]
        if x0.dtype != np.float64 or x0.ndim != 1 or not np.isfinite(x0).all():
            raise ValueError("Stage-1 x0 is invalid")
        if (
            selected_state.dtype != np.float64
            or selected_state.ndim != 1
            or selected_state.shape != x0.shape
            or not np.isfinite(selected_state).all()
        ):
            raise ValueError("Stage-1 selected state is invalid or inconsistent")
        if available.dtype != np.bool_ or available.shape != () or not bool(available):
            raise ValueError("Stage-1 selected state must be available")
    return {
        "root": root,
        "report": report,
        "decision": decision,
        "selected_lambda": float(selected),
        "stage1_results_sha256": sha256_file(root / "stage1_results.json"),
        "stage1_states_sha256": sha256_file(root / "stage1_states.npz"),
        "stage1_decision_sha256": sha256_file(root / "decision.json"),
    }


def _overlaps(first: Path, second: Path) -> bool:
    return first == second or first in second.parents or second in first.parents


def _inside_lifecycle_namespace(candidate: Path, protected: Path) -> bool:
    for component in (candidate, *candidate.parents):
        if component.parent != protected.parent:
            continue
        if component.name == f".{protected.name}.work" or component.name.startswith(
            (
                f".{protected.name}.stage-",
                f".{protected.name}.staging-",
                f".{protected.name}.backup-",
            )
        ):
            return True
    return False


def _collides(first: Path, second: Path) -> bool:
    return (
        _overlaps(first, second)
        or _inside_lifecycle_namespace(first, second)
        or _inside_lifecycle_namespace(second, first)
    )


def validate_output_roots(
    result_root: str | Path,
    failure_root: str | Path,
    *,
    protected_roots: list[str | Path] | tuple[str | Path, ...],
) -> tuple[Path, Path]:
    """Reject output, work, staging, and failure collisions before execution."""

    result = Path(result_root).resolve()
    failure = Path(failure_root).resolve()
    work = result.parent / f".{result.name}.work"
    generated = (result, failure, work)
    if _collides(result, failure) or _collides(work, failure):
        raise ValueError("result, work, and failure roots must be mutually disjoint")
    if failure.parent == result.parent and failure.name.startswith(
        (f".{result.name}.stage-", f".{result.name}.staging-", f".{result.name}.backup-")
    ):
        raise ValueError("failure root must be disjoint from publication staging roots")
    for raw in protected_roots:
        protected = Path(raw).resolve()
        for candidate in generated:
            staging_collision = (
                protected.parent == result.parent
                and protected.name.startswith((f".{result.name}.stage-", f".{result.name}.staging-", f".{result.name}.backup-"))
            )
            if _collides(candidate, protected) or staging_collision:
                raise ValueError("all Stage-2 output roots must be disjoint from protected roots")
    for candidate in (result, failure):
        if candidate.exists() and not candidate.is_dir():
            raise ValueError(f"output root exists as a file: {candidate}")
    return result, failure


def _capture_output_topology(root: Path) -> tuple[Path, int, int]:
    parent = root.parent
    parent.mkdir(parents=True, exist_ok=True)
    if parent.is_symlink() or not parent.is_dir():
        raise ValueError("output parent must be a stable regular directory")
    metadata = parent.stat()
    return parent.resolve(), int(metadata.st_dev), int(metadata.st_ino)


def _require_output_topology(topology: tuple[Path, int, int]) -> None:
    parent, device, inode = topology
    if parent.is_symlink() or not parent.is_dir():
        raise RuntimeError("output parent topology changed before publication")
    metadata = parent.stat()
    if (int(metadata.st_dev), int(metadata.st_ino)) != (device, inode):
        raise RuntimeError("output parent identity changed before publication")


def _load_rule_params(path: str | Path = RULE_CONFIG_PATH) -> tuple[dict[str, Any], str]:
    config = Path(path)
    payload = yaml.safe_load(config.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or set(payload) != {"TomatoEnv"} or not isinstance(payload["TomatoEnv"], dict):
        raise ValueError("rule_based.yml must contain the exact TomatoEnv mapping")
    return dict(payload["TomatoEnv"]), sha256_file(config)


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")


def _fingerprint(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_bytes(dict(payload))).hexdigest()


def _stage2_decision(gate: Mapping[str, Any]) -> dict[str, Any]:
    decision = dict(gate)
    decision["stage"] = "stage2_shielded_context_ab"
    decision["outcome"] = (
        "continue_to_full_suite"
        if gate.get("outcome") == "pass"
        else "redesign_action_shield"
    )
    return decision


def validate_stage1_provenance(
    stage1: Mapping[str, Any],
    *,
    runs: list[dict[str, Any]],
    source_manifest: str | Path,
    source_tasks_csv: str | Path,
    evaluation_provenance: Mapping[str, Any],
    rule_config_sha256: str,
    env_config_sha256: str,
) -> None:
    report = stage1["report"]
    seed123 = [run for run in runs if int(run["seed"]) == 123]
    if len(seed123) != 1:
        raise ValueError("Stage-2 requires the one selected seed-123 checkpoint")
    selected_run = seed123[0]
    if Path(report["checkpoint_path"]).resolve() != Path(selected_run["model_path"]).resolve():
        raise ValueError("Stage-1 checkpoint path does not match selected seed 123")
    if report["checkpoint_sha256"] != selected_run["model_sha256"]:
        raise ValueError("Stage-1 checkpoint hash does not match selected seed 123")
    if report["rule_config_sha256"] != rule_config_sha256:
        raise ValueError("Stage-1 rule configuration is stale")
    if report["env_config_sha256"] != env_config_sha256:
        raise ValueError("Stage-1 environment configuration is stale")
    expected_files = {
        str(Path(source_manifest).resolve()): sha256_file(source_manifest),
        str(Path(source_tasks_csv).resolve()): sha256_file(source_tasks_csv),
        **{
            str((ROOT / path).resolve()): str(evaluation_provenance[name])
            for name, path in RELEVANT_SOURCE_FIELDS
        },
    }
    stage_sources = {str(Path(name).resolve()): value for name, value in report["source_checksums"].items()}
    for name, checksum in expected_files.items():
        if stage_sources.get(name) != checksum:
            raise ValueError(f"Stage-1 source/config provenance is stale or missing: {name}")
    if report["git_head"] != evaluation_provenance["git_commit"] or report["dirty"] != evaluation_provenance["dirty"]:
        raise ValueError("Stage-1 git provenance does not match current source")


def _expected_keys() -> set[tuple[int, str, str]]:
    return {
        (seed, task_id, mode)
        for seed in APPROVED_SEEDS
        for task_id in DIAGNOSTIC_TASK_IDS
        for mode in MODES
    }


def load_unshielded_comparator(
    root: str | Path,
    *,
    expected_provenance: Mapping[str, Any],
    expected_checkpoints: Mapping[int, Mapping[str, str]] | None = None,
    capsule_loader: Callable[[str | Path], Any] = load_failure_capsule,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load an immutable exact-key unshielded comparator, never rerunning it."""

    directory = Path(root).resolve()
    raw_path = directory / "eval_raw.csv"
    if not raw_path.is_file():
        raise FileNotFoundError(f"unshielded comparator is missing eval_raw.csv: {directory}")
    manifest_path = next(
        (path for path in (directory / "shield_manifest.json", directory / "manifest.json", directory / "context_ab_manifest.json", directory / "diagnostic_manifest.json") if path.is_file()),
        None,
    )
    if manifest_path is None:
        raise ValueError("unshielded comparator is missing a provenance manifest")
    manifest = _strict_json(manifest_path)
    for name, expected in expected_provenance.items():
        if name not in manifest or manifest[name] != expected:
            raise ValueError(f"unshielded comparator provenance mismatch: {name}")
    table = pd.read_csv(raw_path)
    key_columns = ["seed", "task_id", "inference_mode"]
    if any(column not in table for column in key_columns):
        raise ValueError("unshielded comparator is missing key columns")
    if table.duplicated(key_columns).any():
        raise ValueError("unshielded comparator contains duplicate keys")
    actual = set(table[key_columns].itertuples(index=False, name=None))
    if actual != _expected_keys():
        raise ValueError(
            "unshielded comparator must contain all exact 32 keys; partial progress is not accepted; run the separate failure-tolerant unshielded completion task"
        )
    if "method" in table and table["method"].eq(METHOD).any():
        raise ValueError("unshielded comparator must be method-separated from the shield")
    if expected_checkpoints is not None:
        for column in ("model_sha256", "vecnormalize_sha256"):
            if column not in table:
                raise ValueError(f"unshielded comparator is missing checkpoint provenance: {column}")
        for seed, expected_hashes in expected_checkpoints.items():
            seed_rows = table.loc[table["seed"] == seed]
            for column in ("model_sha256", "vecnormalize_sha256"):
                if set(seed_rows[column].astype(str)) != {str(expected_hashes[column])}:
                    raise ValueError(f"unshielded comparator checkpoint mismatch for seed {seed}: {column}")
    required_protocol = {
        "completed", "status", "ode_failure_count", "failure_evidence_path",
        *REQUIRED_METRICS,
    }
    missing_protocol = sorted(required_protocol.difference(table.columns))
    if missing_protocol:
        raise ValueError(f"unshielded comparator is missing explicit protocol columns: {missing_protocol}")
    completed_values = list(table["completed"].array)
    if any(not isinstance(value, (bool, np.bool_)) for value in completed_values):
        raise ValueError("unshielded completed values must be strict booleans")
    table["completed"] = [bool(value) for value in completed_values]
    for _, row in table.iterrows():
        count = row["ode_failure_count"]
        if isinstance(count, (bool, np.bool_)) or not isinstance(count, (int, np.integer)) or int(count) < 0:
            raise ValueError("unshielded ode_failure_count must be a nonnegative integer")
        completed = bool(row["completed"])
        status = row["status"]
        evidence_value = row["failure_evidence_path"]
        evidence_empty = pd.isna(evidence_value) or (isinstance(evidence_value, str) and not evidence_value.strip())
        metrics = pd.to_numeric(row[list(REQUIRED_METRICS)], errors="coerce").to_numpy(dtype=float)
        if completed:
            if status != "completed" or int(count) != 0 or not evidence_empty or not np.isfinite(metrics).all():
                raise ValueError("completed unshielded rows require completed status, zero failures, empty evidence, and finite metrics")
            continue
        if status != "ode_failure" or int(count) < 1 or evidence_empty or not np.isnan(metrics).all():
            raise ValueError("incomplete unshielded rows must be explicit ode_failure evidence with non-scoring metrics")
        evidence_path = Path(str(evidence_value))
        if not evidence_path.is_absolute():
            evidence_path = directory / evidence_path
        evidence_path = evidence_path.resolve()
        if evidence_path.name != "manifest.json" or not evidence_path.is_file():
            raise ValueError("failure_evidence_path must point to an existing manifest.json")
        try:
            capsule = capsule_loader(evidence_path.parent)
        except Exception as error:
            raise ValueError("failure_evidence_path is not a valid failure capsule") from error
        context = capsule.manifest.get("context")
        if not isinstance(context, Mapping):
            raise ValueError("failure capsule context is missing")
        seed = int(row["seed"])
        expected_checkpoint = expected_checkpoints.get(seed) if expected_checkpoints is not None else None
        identity_matches = (
            context.get("seed") == seed
            and context.get("task_id") == row["task_id"]
            and context.get("inference_mode") == row["inference_mode"]
            and isinstance(context.get("task"), Mapping)
            and context["task"].get("task_id") == row["task_id"]
            and context.get("git_head") == expected_provenance.get("git_commit")
            and context.get("dirty") == expected_provenance.get("dirty")
            and Path(str(context.get("formal_result_root"))).resolve() == directory
        )
        if expected_checkpoint is not None:
            identity_matches = identity_matches and (
                context.get("checkpoint_sha256") == expected_checkpoint["model_sha256"]
                and context.get("checkpoint_sha256") == str(row["model_sha256"])
                and Path(str(context.get("checkpoint_path"))).resolve()
                == Path(expected_checkpoint["model_path"]).resolve()
            )
        source_values = set(context.get("source_checksums", {}).values()) if isinstance(context.get("source_checksums"), Mapping) else set()
        for name in ("source_manifest_sha256", "source_tasks_sha256"):
            identity_matches = identity_matches and expected_provenance.get(name) in source_values
        if not identity_matches:
            raise ValueError("failure capsule identity/provenance does not match comparator row")
    return table, manifest


def _trace_paths(root: Path, seed: int, task_id: str, mode: str) -> tuple[Path, Path, Path]:
    stem = f"seed{seed}__{task_id}__{mode}"
    return (
        root / "traces" / f"{stem}__executed.npy",
        root / "traces" / f"{stem}__requested.npy",
        root / "intervention_records" / f"{stem}.json",
    )


def _json_records(path: Path) -> list[dict[str, Any]]:
    value = json.loads(path.read_text(encoding="utf-8"), parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)))
    if not isinstance(value, list) or not all(isinstance(item, dict) for item in value):
        raise ValueError("intervention evidence must be a JSON list of mappings")
    return value


def resume_row_is_complete(row: Mapping[str, Any], *, expected: Mapping[str, Any], work_root: Path) -> bool:
    required = {"seed", "task_id", "inference_mode", "checkpoint_steps", "completed", "executed_action_trace_path", "requested_action_trace_path", "intervention_records_path", "method", *HASH_FIELDS, *REQUIRED_METRICS}
    completed_value = row.get("completed")
    if (
        not required.issubset(row)
        or row.get("method") != METHOD
        or not isinstance(completed_value, (bool, np.bool_))
        or not bool(completed_value)
    ):
        return False
    try:
        if any(str(row[name]) != str(expected[name]) for name in HASH_FIELDS):
            return False
        paths = [Path(str(row[name])).resolve() for name in ("executed_action_trace_path", "requested_action_trace_path", "intervention_records_path")]
        if any(not path.is_relative_to(work_root.resolve()) for path in paths):
            return False
        executed = np.load(paths[0], allow_pickle=False)
        requested = np.load(paths[1], allow_pickle=False)
        records = _json_records(paths[2])
        if executed.ndim != 2 or executed.shape != requested.shape or executed.shape[0] != len(records) or not np.isfinite(executed).all() or not np.isfinite(requested).all():
            return False
        summary = aggregate_episode_interventions(records, executed.shape[1])
        for name, value in summary.items():
            observed = row[name]
            if isinstance(value, list):
                if json.loads(str(observed)) != value:
                    return False
            elif value is None:
                if not pd.isna(observed):
                    return False
            elif isinstance(value, float):
                if not np.isfinite(float(observed)) or not np.isclose(float(observed), value, rtol=1e-12, atol=1e-12):
                    return False
            elif int(observed) != value:
                return False
        metrics = np.asarray([float(row[name]) for name in REQUIRED_METRICS])
        return bool(np.isfinite(metrics).all())
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
        return False


def _write_progress(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    pd.DataFrame(rows).to_csv(temporary, index=False)
    os.replace(temporary, path)


def _strict_diagnostics(diagnostics: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]], dict[str, Any]]:
    diagnostics = dict(diagnostics)
    missing = {"action_trace", "requested_action_trace", "action_shield_records"}.difference(diagnostics)
    if missing:
        raise KeyError(f"shielded episode diagnostics are missing: {sorted(missing)}")
    executed = np.asarray(diagnostics.pop("action_trace"), dtype=np.float32)
    requested = np.asarray(diagnostics.pop("requested_action_trace"), dtype=np.float32)
    raw_records = diagnostics.pop("action_shield_records")
    if executed.ndim != 2 or executed.shape[0] == 0 or executed.shape[1] == 0 or executed.shape != requested.shape or not np.isfinite(executed).all() or not np.isfinite(requested).all():
        raise ValueError("executed/requested action traces must be matching finite 2D arrays")
    if not isinstance(raw_records, list) or len(raw_records) != executed.shape[0]:
        raise ValueError("action_shield_records must contain one record per executed step")
    records: list[dict[str, Any]] = []
    for step, record in enumerate(raw_records):
        if not isinstance(record, Mapping):
            raise TypeError("each action shield record must be a mapping")
        detached = dict(record)
        if "step_index" in detached and detached["step_index"] != step:
            raise ValueError("action shield step_index is inconsistent")
        detached["step_index"] = step
        try:
            record_executed = np.asarray(detached["executed_action"], dtype=np.float32)
            record_requested = np.asarray(detached["requested_action"], dtype=np.float32)
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("action shield record actions must be numeric vectors") from error
        if (
            record_executed.shape != (executed.shape[1],)
            or record_requested.shape != (requested.shape[1],)
            or not np.isfinite(record_executed).all()
            or not np.isfinite(record_requested).all()
            or not np.array_equal(executed[step], record_executed)
            or not np.array_equal(requested[step], record_requested)
        ):
            raise ValueError("action traces must exactly match each shield record")
        records.append(detached)
    normalized: dict[str, Any] = {}
    reserved = {"seed", "task_id", "inference_mode", "method", *HASH_FIELDS}
    if reserved.intersection(diagnostics):
        raise ValueError("model diagnostics collide with Stage-2 raw fields")
    for name, value in diagnostics.items():
        if value is None:
            normalized[name] = ""
        elif isinstance(value, np.generic):
            normalized[name] = value.item()
        elif isinstance(value, (str, bool, int, float)) and (not isinstance(value, float) or np.isfinite(value)):
            normalized[name] = value
        else:
            raise TypeError(f"model diagnostic {name!r} must be a finite CSV scalar")
    return executed, requested, records, normalized


def run_shielded_diagnostic(
    *,
    suite: Any,
    tasks: pd.DataFrame,
    runs: list[dict[str, Any]],
    source_manifest: str | Path,
    source_tasks_csv: str | Path,
    stage1_root: str | Path,
    unshielded_result_root: str | Path,
    result_root: str | Path,
    failure_root: str | Path,
    device: str,
    resume: bool,
    model_loader: Callable[[Path, str], Any] | None = None,
    env_loader: Callable[..., Any] = load_task_env,
    episode_runner: Callable[..., Any] = run_deterministic_episode,
    provenance_loader: Callable[[], dict[str, Any]] = _provenance,
    recorder_factory: Callable[[str | Path, CapsuleContext], Any] = FailureCapsuleRecorder,
    rule_config_path: str | Path = RULE_CONFIG_PATH,
    env_config_path: str | Path = ENV_CONFIG_PATH,
) -> pd.DataFrame:
    """Validate prerequisites, execute exactly 32 shielded episodes, and gate."""

    if tuple(int(run.get("seed", -1)) for run in runs) != APPROVED_SEEDS:
        raise ValueError("shielded diagnostic requires exactly one ordered run for seeds 42 123")
    stage1 = load_stage1_prerequisite(stage1_root)
    provenance = dict(provenance_loader())
    evaluation = _evaluation_provenance(source_manifest, source_tasks_csv, provenance)
    params, rule_sha = _load_rule_params(rule_config_path)
    env_sha = sha256_file(env_config_path)
    validate_stage1_provenance(
        stage1, runs=runs, source_manifest=source_manifest,
        source_tasks_csv=source_tasks_csv, evaluation_provenance=evaluation,
        rule_config_sha256=rule_sha, env_config_sha256=env_sha,
    )
    root, capsule_root = validate_output_roots(
        result_root, failure_root,
        protected_roots=[suite.result_root, stage1["root"], unshielded_result_root],
    )
    output_topology = _capture_output_topology(root)
    comparator_provenance = {
        name: evaluation[name]
        for name in ("source_manifest_sha256", "source_tasks_sha256", "git_commit", "dirty", "evaluation_provenance_sha256")
    }
    unshielded, unshielded_manifest = load_unshielded_comparator(
        unshielded_result_root, expected_provenance=comparator_provenance,
        expected_checkpoints={
            int(run["seed"]): {
                "model_sha256": str(run["model_sha256"]),
                "vecnormalize_sha256": str(run["vecnormalize_sha256"]),
                "model_path": str(Path(run["model_path"]).resolve()),
            }
            for run in runs
        },
    )
    selected = select_diagnostic_tasks(tasks)
    task_records = [task_from_row(row) for row in selected.itertuples(index=False)]
    fingerprint_payload = {
        "schema_version": SCHEMA_VERSION, "method": METHOD,
        "checkpoints": [{name: str(run[name]) for name in ("seed", "model_path", "vecnormalize_path", "model_sha256", "vecnormalize_sha256")} for run in runs],
        **evaluation, "rule_config_sha256": rule_sha, "env_config_sha256": env_sha,
        "formal_solver_options": dict(FORMAL_CVODES_OPTIONS), "fixed_lambdas": list(DEFAULT_LAMBDAS),
        "stage1_results_sha256": stage1["stage1_results_sha256"],
        "stage1_selected_lambda": stage1["selected_lambda"],
        "stage1_states_sha256": stage1["stage1_states_sha256"],
        "stage1_decision_sha256": stage1["stage1_decision_sha256"],
        "task_ids": list(DIAGNOSTIC_TASK_IDS), "inference_modes": list(MODES),
        "seeds": list(APPROVED_SEEDS), "device": device,
    }
    shield_fingerprint = _fingerprint(fingerprint_payload)
    formal_solver_sha = _fingerprint(dict(FORMAL_CVODES_OPTIONS))
    fixed_lambdas_sha = hashlib.sha256(_canonical_bytes(list(DEFAULT_LAMBDAS))).hexdigest()
    work = root.parent / f".{root.name}.work"
    progress = work / "progress.csv"
    if not resume and work.exists():
        shutil.rmtree(work)
    prior = pd.read_csv(progress).to_dict("records") if resume and progress.is_file() else []
    evidence_by_seed = {
        int(run["seed"]): {
            "model_sha256": run["model_sha256"], "vecnormalize_sha256": run["vecnormalize_sha256"],
            **evaluation, "rule_config_sha256": rule_sha, "env_config_sha256": env_sha,
            "formal_solver_options_sha256": formal_solver_sha,
            "fixed_lambdas_sha256": fixed_lambdas_sha,
            "stage1_results_sha256": stage1["stage1_results_sha256"],
            "stage1_states_sha256": stage1["stage1_states_sha256"],
            "stage1_decision_sha256": stage1["stage1_decision_sha256"],
            "stage1_capsule_identity_sha256": stage1["report"]["capsule_identity_sha256"],
            "shield_fingerprint": shield_fingerprint,
        }
        for run in runs
    }
    completed: dict[tuple[int, str, str], dict[str, Any]] = {}
    targets = _expected_keys()
    for row in prior:
        try:
            key = (int(row["seed"]), str(row["task_id"]), str(row["inference_mode"]))
        except (KeyError, TypeError, ValueError):
            continue
        if key in targets and resume_row_is_complete(row, expected=evidence_by_seed[key[0]], work_root=work):
            completed[key] = row
    source_checksums = {
        str((ROOT / path).resolve()): evaluation[name]
        for name, path in RELEVANT_SOURCE_FIELDS
    }
    source_checksums.update({str(Path(source_manifest).resolve()): evaluation["source_manifest_sha256"], str(Path(source_tasks_csv).resolve()): evaluation["source_tasks_sha256"]})
    packages = _package_versions()
    load_model = model_loader or (lambda path, selected_device: AgriMetaRL.load(str(path), device=selected_device))
    checkpoint_records: list[dict[str, Any]] = []
    for run in runs:
        model = load_model(Path(run["model_path"]), device)
        checkpoint_steps = int(model.num_timesteps)
        checkpoint_records.append({**{name: str(run[name]) for name in ("model_path", "vecnormalize_path", "model_sha256", "vecnormalize_sha256")}, "seed": int(run["seed"]), "checkpoint_steps": checkpoint_steps})
        for task in task_records:
            for mode in MODES:
                key = (int(run["seed"]), task.task_id, mode)
                if key in completed and int(completed[key]["checkpoint_steps"]) == checkpoint_steps:
                    continue
                env = env_loader(suite, task, Path(run["vecnormalize_path"]), shield_params=params)
                primary: BaseException | None = None
                try:
                    recorder = recorder_factory(
                        capsule_root,
                        CapsuleContext(
                            seed=key[0], task_id=task.task_id, inference_mode=mode,
                            task=asdict(task), checkpoint_path=str(Path(run["model_path"]).resolve()),
                            checkpoint_sha256=run["model_sha256"], git_head=evaluation["git_commit"],
                            dirty=evaluation["dirty"], source_checksums=source_checksums,
                            package_versions=packages, formal_result_root=str(root),
                        ),
                    )
                    metrics, diagnostics = episode_runner(
                        model, env, inference_mode=mode, return_diagnostics=True,
                        failure_recorder=recorder,
                    )
                except BaseException as error:
                    primary = error
                    raise
                finally:
                    try:
                        env.close()
                    except BaseException as close_error:
                        if primary is not None:
                            primary.add_note(f"environment close also failed: {type(close_error).__name__}: {close_error}")
                        else:
                            raise
                executed, requested, records, model_diagnostics = _strict_diagnostics(diagnostics)
                summary = aggregate_episode_interventions(records, executed.shape[1])
                executed_path, requested_path, records_path = _trace_paths(work, *key)
                executed_path.parent.mkdir(parents=True, exist_ok=True)
                records_path.parent.mkdir(parents=True, exist_ok=True)
                np.save(executed_path, executed, allow_pickle=False)
                np.save(requested_path, requested, allow_pickle=False)
                records_path.write_text(json.dumps(records, ensure_ascii=False, allow_nan=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
                scalar_summary = {name: (json.dumps(value, separators=(",", ":")) if isinstance(value, list) else value) for name, value in summary.items()}
                row = {
                    **{name: float(value) for name, value in metrics.items()}, **model_diagnostics,
                    "seed": key[0], "task_id": task.task_id, "split": task.split,
                    "inference_mode": mode, "checkpoint_steps": checkpoint_steps,
                    "method": METHOD, "completed": True,
                    "stage1_selected_lambda": stage1["selected_lambda"],
                    "executed_action_trace_path": str(executed_path.resolve()),
                    "requested_action_trace_path": str(requested_path.resolve()),
                    "intervention_records_path": str(records_path.resolve()),
                    **scalar_summary, **evidence_by_seed[key[0]],
                }
                completed[key] = row
                _write_progress([completed[item] for item in sorted(completed)], progress)
    rows = [completed[key] for key in sorted(targets) if key in completed]
    if len(rows) != 32:
        raise RuntimeError(f"shielded diagnostic completed {len(rows)} of 32 required episodes")
    raw = pd.DataFrame(rows)
    if set(raw[["seed", "task_id", "inference_mode"]].itertuples(index=False, name=None)) != targets:
        raise RuntimeError("shielded diagnostic final keys are not the exact 32-key design")
    gate = evaluate_shield_gate(raw, unshielded, targets)
    paired = build_paired_shield_deltas(raw, unshielded, targets)
    intervention_columns = ["seed", "task_id", "split", "inference_mode", "method", "total_steps", "intervention_count", "intervention_rate", "first_intervention_step", "selected_lambda_mean", "selected_lambda_max", "intervention_l1_mean", "intervention_l1_max", "intervention_l2_mean", "intervention_l2_max", "intervention_linf_mean", "intervention_linf_max", "per_channel_intervention_counts", "extra_solver_attempts", "shield_elapsed_seconds", "ode_failure_count", "intervention_records_path"]
    interventions = raw[intervention_columns].copy()
    evidence_files: dict[str, Path] = {}
    for index, row in raw.iterrows():
        for column, directory in (("executed_action_trace_path", "traces"), ("requested_action_trace_path", "traces"), ("intervention_records_path", "intervention_records")):
            source = Path(row[column]).resolve()
            relative = f"{directory}/{source.name}"
            evidence_files[relative] = source
            raw.at[index, column] = relative
            if column == "intervention_records_path":
                interventions.at[index, column] = relative
    decision = _stage2_decision(gate)
    manifest = {
        **fingerprint_payload, "shield_fingerprint": shield_fingerprint,
        "source_manifest": str(Path(source_manifest).resolve()),
        "source_tasks_csv": str(Path(source_tasks_csv).resolve()),
        "source_suite_id": str(suite.suite_id), "source_suite_result_root": str(Path(suite.result_root).resolve()),
        "unshielded_result_root": str(Path(unshielded_result_root).resolve()),
        "unshielded_manifest_sha256": sha256_file(next(path for path in (Path(unshielded_result_root) / "shield_manifest.json", Path(unshielded_result_root) / "manifest.json", Path(unshielded_result_root) / "context_ab_manifest.json", Path(unshielded_result_root) / "diagnostic_manifest.json") if path.is_file())),
        "stage1_root": str(stage1["root"]), "result_root": str(root),
        "checkpoints": checkpoint_records, "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    _require_output_topology(output_topology)
    write_shield_artifacts_atomic(raw, paired, interventions, manifest, decision, root, evidence_files=evidence_files)
    return pd.read_csv(root / "eval_raw.csv")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source_manifest", required=True)
    parser.add_argument("--source_tasks_csv", required=True)
    parser.add_argument("--model_root", required=True)
    parser.add_argument("--stage1_root", required=True)
    parser.add_argument("--unshielded_result_root", required=True)
    parser.add_argument("--result_root", default=str(DEFAULT_RESULT_ROOT))
    parser.add_argument("--failure_root", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--resume", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    source_manifest = Path(args.source_manifest)
    source_tasks = Path(args.source_tasks_csv)
    if not source_manifest.is_file() or not source_tasks.is_file():
        raise FileNotFoundError("source manifest and task CSV must both exist")
    suite = load_suite_manifest(source_manifest)
    runs = build_diagnostic_runs(args.model_root, args.seeds)
    run_shielded_diagnostic(
        suite=suite, tasks=pd.read_csv(source_tasks), runs=runs,
        source_manifest=source_manifest, source_tasks_csv=source_tasks,
        stage1_root=args.stage1_root, unshielded_result_root=args.unshielded_result_root,
        result_root=args.result_root, failure_root=args.failure_root,
        device=args.device, resume=args.resume,
    )


if __name__ == "__main__":
    main()
