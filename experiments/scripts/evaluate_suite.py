#!/usr/bin/env python3
"""Evaluate completed robust experiment suite runs on deterministic tasks."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
import tempfile
import shutil
import uuid
import sys
import time
from pathlib import Path
from typing import Any, Callable, Mapping


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd
from sb3_contrib import RecurrentPPO
from stable_baselines3 import PPO

from gl_gym.RL.agri_metarl import AgriMetaRL
from gl_gym.RL.context_recurrent_ppo import ContextRecurrentPPO
from gl_gym.experiments.suite_evaluation import (
    EvaluationMetricRow,
    append_eval_raw,
    completed_eval_keys,
    evaluation_key,
    load_task_env,
    run_deterministic_episode,
    task_from_row,
    validate_completed_run_paths,
)
from gl_gym.experiments.suite_schema import load_suite_manifest
from gl_gym.experiments.suite_tasks import build_evaluation_tasks
from gl_gym.experiments.shield_evaluation import (
    aggregate_episode_interventions,
    build_paired_shield_deltas,
    evaluate_shield_gate,
)
from gl_gym.experiments.ode_failure import (
    CapsuleContext,
    FailureCapsuleRecorder,
    load_failure_capsule,
)


SHIELD_SCHEMA_VERSION = "conservative-feasibility-action-shield-v2"
SHIELD_METHOD = "conservative_feasibility_shield_v2"
SHIELD_SUFFIX = "__conservative_feasibility_shield_v2"
STAGE2_CONDITIONS = (
    "zero_ode_failures",
    "intervention_rate_within_0p5pct",
    "paired_return_loss_within_2pct",
    "paired_violation_burden_within_5pct",
)
FORMAL_UNSHIELDED_METHOD = "agri_metarl_unshielded_formal_v1"
EARLY_HORIZON_FAILURE = re.compile(
    r"evaluation episode terminated before configured horizon: step ([1-9]\d*) of ([1-9]\d*)"
)


ALG_MAP = {
    "ppo": PPO,
    "recurrentppo": RecurrentPPO,
    "context_recurrentppo": ContextRecurrentPPO,
    "agri_metarl": AgriMetaRL,
}


def _formal_source_checksums(args: argparse.Namespace, payload: Mapping[str, str]) -> dict[str, str]:
    return {
        str(Path(args.manifest).resolve()): payload["source_manifest_sha256"],
        str(Path(args.runs_csv).resolve()): payload["runs_csv_sha256"],
        str(Path(args.tasks_csv).resolve()): payload["tasks_csv_sha256"],
        str(Path(__file__).resolve()): payload["evaluator_source_sha256"],
        str((ROOT / "experiments/scripts/evaluate_shield_gate.py").resolve()): payload[
            "gate_source_sha256"
        ],
        "runtime_source_tree": payload["runtime_source_tree_sha256"],
    }


def _validate_capsule_content(
    manifest_path: Path,
    *,
    expected_context: CapsuleContext,
    expected_solver_options: Mapping[str, Any],
    episode_step: int,
) -> tuple[str, Any]:
    capsule = load_failure_capsule(manifest_path.parent)
    expected = {
        "seed": int(expected_context.seed),
        "task_id": expected_context.task_id,
        "inference_mode": expected_context.inference_mode,
        "task": expected_context.task,
        "checkpoint_path": expected_context.checkpoint_path,
        "checkpoint_sha256": expected_context.checkpoint_sha256,
        "git_head": expected_context.git_head,
        "dirty": expected_context.dirty,
        "source_checksums": expected_context.source_checksums,
        "package_versions": expected_context.package_versions,
        "formal_result_root": expected_context.formal_result_root,
    }
    manifest = capsule.manifest
    if manifest.get("context") != expected:
        raise ValueError("failure capsule context does not match this evaluation attempt")
    exception = manifest.get("exception", {})
    if not exception.get("type") or not exception.get("message"):
        raise ValueError("failure capsule underlying exception fields are empty")
    if (
        exception["type"] not in capsule.traceback_text
        or exception["message"] not in capsule.traceback_text
    ):
        raise ValueError("failure capsule traceback does not bind its underlying exception")
    recorded_step = int(capsule.history_arrays["step_index"][-1])
    failure_timestep = int(capsule.failure_inputs["timestep"])
    if recorded_step != episode_step - 1 or failure_timestep != recorded_step:
        raise ValueError("failure capsule timestep does not match the early episode step")
    solver_options = manifest.get("solver", {}).get("options")
    if solver_options != dict(expected_solver_options) or not solver_options:
        raise ValueError("failure capsule does not prove a configured solver call")
    identity = manifest.get("content_identity_sha256")
    if not isinstance(identity, str) or len(identity) != 64:
        raise ValueError("failure capsule content identity is invalid")
    return identity, capsule


def _validate_attempt_capsule(
    manifest_path: Path,
    *,
    expected_context: CapsuleContext,
    expected_solver_options: Mapping[str, Any],
    error: Exception,
) -> tuple[str, Any, int, int]:
    match = EARLY_HORIZON_FAILURE.fullmatch(str(error))
    if type(error) is not RuntimeError or match is None:
        raise ValueError("caught exception is not the exact early-horizon RuntimeError")
    episode_step, configured_horizon = (int(value) for value in match.groups())
    if episode_step >= configured_horizon:
        raise ValueError("early-horizon RuntimeError has inconsistent step/horizon")
    identity, capsule = _validate_capsule_content(
        manifest_path,
        expected_context=expected_context,
        expected_solver_options=expected_solver_options,
        episode_step=episode_step,
    )
    return identity, capsule, episode_step, configured_horizon


def _path_has_reparse_between(path: Path, root: Path) -> bool:
    current = path
    while True:
        metadata = current.lstat()
        attributes = getattr(metadata, "st_file_attributes", 0)
        if current.is_symlink() or attributes & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400):
            return True
        if current == root:
            return False
        if root not in current.parents:
            return True
        current = current.parent


def _resumed_failure_capsule_valid(
    row: Mapping[str, Any], *, expected_context: CapsuleContext,
    expected_solver_options: Mapping[str, Any], work: Path, result_root: Path,
) -> bool:
    try:
        if (
            row.get("completed") is not False
            or row.get("status") != "ode_failure"
            or row.get("ode_failure_count") != 1
        ):
            return False
        manifest_path = Path(str(row.get("failure_evidence_path", "")))
        if not manifest_path.is_absolute() or manifest_path.name != "manifest.json":
            return False
        resolved = manifest_path.resolve()
        allowed_roots = (
            _long_path(work / "failures" / "attempts"),
            result_root.resolve(),
        )
        containing_root = next(
            (candidate for candidate in allowed_roots if resolved.is_relative_to(candidate)),
            None,
        )
        if containing_root is None or not resolved.is_file() or _path_has_reparse_between(resolved, containing_root):
            return False
        episode_step = row.get("failure_episode_step")
        horizon = row.get("failure_configured_horizon")
        if (
            isinstance(episode_step, bool) or not isinstance(episode_step, int)
            or isinstance(horizon, bool) or not isinstance(horizon, int)
            or episode_step < 1 or horizon <= episode_step
        ):
            return False
        identity, _ = _validate_capsule_content(
            resolved,
            expected_context=expected_context,
            expected_solver_options=expected_solver_options,
            episode_step=episode_step,
        )
        return identity == row.get("failure_evidence_identity_sha256")
    except (OSError, ValueError, TypeError, KeyError, IndexError):
        return False


def filter_tasks(
    tasks: pd.DataFrame,
    splits: list[str] | None = None,
    task_ids: list[str] | None = None,
    limit_tasks: int | None = None,
) -> pd.DataFrame:
    """Return a deterministic subset of evaluation tasks for smoke/pilot runs."""

    selected = tasks
    if splits:
        selected = selected[selected["split"].isin(splits)]
    if task_ids:
        selected = selected[selected["task_id"].isin(task_ids)]
    if limit_tasks is not None:
        if limit_tasks < 1:
            raise ValueError("--limit_tasks must be positive")
        selected = selected.head(limit_tasks)
    return selected.reset_index(drop=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--runs_csv", required=True)
    parser.add_argument("--tasks_csv", required=True)
    parser.add_argument("--algorithms", nargs="+")
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--splits", nargs="+")
    parser.add_argument("--task_ids", nargs="+")
    parser.add_argument("--limit_tasks", type=int)
    parser.add_argument(
        "--resume_eval",
        action="store_true",
        help="Skip algorithm/seed/task rows already present in eval_raw.csv.",
    )
    parser.add_argument("--action_shield", action="store_true")
    parser.add_argument("--formal_unshielded_provenance", action="store_true")
    parser.add_argument("--stage2_decision")
    parser.add_argument("--result_root")
    parser.add_argument("--interventions_out")
    return parser


def validate_cli_mode(args: argparse.Namespace) -> None:
    shield_only = (args.stage2_decision, args.result_root, args.interventions_out)
    formal_unshielded = getattr(args, "formal_unshielded_provenance", False)
    if args.action_shield and formal_unshielded:
        raise ValueError("shielded and formal unshielded modes are mutually exclusive")
    if args.action_shield or formal_unshielded:
        if args.stage2_decision is None or args.result_root is None:
            raise ValueError("formal evaluation requires --stage2_decision and --result_root")
    elif any(value is not None for value in shield_only):
        raise ValueError("shield-only arguments require --action_shield")


def _sha(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _strict_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"), parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))
    if not isinstance(value, dict):
        raise ValueError(f"strict JSON mapping required: {path}")
    return value


def _overlap(a: Path, b: Path) -> bool:
    return a == b or a in b.parents or b in a.parents


def _long_path(path: Path) -> Path:
    resolved = path.resolve()
    if os.name == "nt" and not str(resolved).startswith("\\\\?\\"):
        return Path("\\\\?\\" + str(resolved))
    return resolved


def _canonical_hash(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


_ROW_INTEGER_FIELDS = frozenset({
    "seed", "weather_year", "start_day", "checkpoint_steps", "ode_failure_count",
})
_ROW_FLOAT_FIELDS = frozenset({
    "uncertainty_scale", "episode_return", "temp_violation", "co2_violation",
    "rh_violation", "EPI", "revenue", "heat_cost", "co2_cost", "elec_cost",
})
_ROW_NULLABLE_METRICS = frozenset({
    "episode_return", "temp_violation", "co2_violation", "rh_violation",
    "EPI", "revenue", "heat_cost", "co2_cost", "elec_cost",
})
_ROW_OPTIONAL_NULLABLE_FLOAT_METRICS = frozenset({"twb_percent"})
_ROW_EMPTY_TEXT_FIELDS = frozenset({
    "failure_evidence_path", "failure_evidence_identity_sha256",
})
_ROW_NULLABLE_INTEGER_FIELDS = frozenset({
    "failure_episode_step", "failure_configured_horizon",
})


def canonical_evaluation_row(row: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize every evaluation-row column for stable JSON/CSV identities."""
    import numpy as np

    result: dict[str, Any] = {}
    for name, raw in dict(row).items():
        value = raw.item() if isinstance(raw, np.generic) else raw
        missing = value is None or (
            isinstance(value, (float, np.floating)) and pd.isna(value)
        )
        if name in _ROW_EMPTY_TEXT_FIELDS:
            result[name] = "" if missing else str(value)
        elif name in _ROW_NULLABLE_INTEGER_FIELDS:
            if missing:
                result[name] = None
            elif isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.integer, np.floating)) or not float(value).is_integer():
                raise TypeError(f"evaluation row {name} must be a nullable strict integer")
            else:
                result[name] = int(value)
        elif name in _ROW_NULLABLE_METRICS and missing:
            result[name] = None
        elif name in _ROW_OPTIONAL_NULLABLE_FLOAT_METRICS:
            if missing:
                result[name] = None
            elif isinstance(value, (bool, np.bool_)) or not isinstance(
                value, (int, float, np.integer, np.floating)
            ):
                raise TypeError(f"evaluation row {name} must be numeric")
            else:
                scalar = float(value)
                if not np.isfinite(scalar):
                    raise ValueError(f"evaluation row {name} must be finite")
                result[name] = scalar
        elif name in _ROW_INTEGER_FIELDS:
            if missing or isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
                raise TypeError(f"evaluation row {name} must be a strict integer")
            result[name] = int(value)
        elif name in _ROW_FLOAT_FIELDS:
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.integer, np.floating)):
                raise TypeError(f"evaluation row {name} must be numeric")
            scalar = float(value)
            if not np.isfinite(scalar):
                raise ValueError(f"evaluation row {name} must be finite")
            result[name] = scalar
        elif name in {"completed", "formal_complete"}:
            if not isinstance(value, (bool, np.bool_)):
                raise TypeError(f"evaluation row {name} must be strict boolean")
            result[name] = bool(value)
        elif isinstance(value, float) and pd.isna(value):
            raise ValueError(f"evaluation row {name} has unsupported missing value")
        else:
            result[name] = value
    completed = result.get("completed")
    if completed is True and any(result.get(name) is None for name in _ROW_NULLABLE_METRICS if name in result):
        raise ValueError("completed evaluation row metrics cannot be null")
    if completed is False and any(result.get(name) is not None for name in _ROW_NULLABLE_METRICS if name in result):
        raise ValueError("failed evaluation row metrics must be null")
    return result


def evaluation_row_identity(row: Mapping[str, Any]) -> str:
    normalized = canonical_evaluation_row(row)
    normalized.pop("episode_evidence_identity_sha256", None)
    return _canonical_hash(normalized)


def shield_method_fingerprint_components(
    *, source_inputs: Mapping[str, str], rule_config_sha256: str,
    env_config_sha256: str, stage2_identity_sha256: str,
    fixed_lambdas: Any, formal_solver_options: Mapping[str, Any],
) -> dict[str, Any]:
    ordered_lambdas = list(fixed_lambdas)
    if ordered_lambdas != sorted(ordered_lambdas, reverse=True):
        raise ValueError("fixed lambdas must use canonical descending order")
    return {
        "schema_version": SHIELD_SCHEMA_VERSION,
        "method": SHIELD_METHOD,
        "rule_config_sha256": rule_config_sha256,
        "env_config_sha256": env_config_sha256,
        "stage2_identity_sha256": stage2_identity_sha256,
        "fixed_lambdas": ordered_lambdas,
        "formal_solver_options": dict(formal_solver_options),
        "source_fingerprint_inputs": dict(source_inputs),
    }


def load_stage2_evidence(decision_path: str | Path) -> dict[str, Any]:
    """Authenticate the complete Stage-2 artifact set and recompute its gate."""
    decision_file = Path(decision_path).resolve()
    root = decision_file.parent
    names = ("eval_raw.csv", "paired_deltas.csv", "interventions.csv", "shield_manifest.json", "decision.json")
    paths = {name: root / name for name in names}
    if decision_file != paths["decision.json"]:
        raise ValueError("Stage-2 decision must be the canonical decision.json in its result root")
    if any(not path.is_file() or path.is_symlink() for path in paths.values()):
        raise FileNotFoundError("Stage-2 requires all five canonical regular artifacts")
    manifest = _strict_json(paths["shield_manifest.json"])
    decision = _strict_json(paths["decision.json"])
    from experiments.scripts import run_shielded_context_ab as stage2_source
    if manifest.get("schema_version") != SHIELD_SCHEMA_VERSION or manifest.get("method") != SHIELD_METHOD:
        raise ValueError("Stage-2 manifest schema/method is invalid")
    if manifest.get("fixed_lambdas") != list(stage2_source.DEFAULT_LAMBDAS):
        raise ValueError("Stage-2 manifest fixed lambda order is invalid")
    shield_fingerprint = manifest.get("shield_fingerprint")
    if (
        not isinstance(shield_fingerprint, str)
        or len(shield_fingerprint) != 64
        or any(character not in "0123456789abcdef" for character in shield_fingerprint)
    ):
        raise ValueError("Stage-2 manifest shield fingerprint is invalid")
    if Path(str(manifest.get("result_root", ""))).resolve() != root:
        raise ValueError("Stage-2 manifest result_root does not bind the artifact root")
    from experiments.scripts.run_context_ab import APPROVED_SEEDS
    from gl_gym.experiments.context_ab import DIAGNOSTIC_TASK_IDS, MODES
    seeds, task_ids, modes = manifest.get("seeds"), manifest.get("task_ids"), manifest.get("inference_modes")
    if seeds != list(APPROVED_SEEDS) or task_ids != list(DIAGNOSTIC_TASK_IDS) or modes != list(MODES):
        raise ValueError("Stage-2 manifest is not the exact approved diagnostic protocol")
    expected = {(seed, task, mode) for seed in seeds for task in task_ids for mode in modes}
    raw = pd.read_csv(paths["eval_raw.csv"])
    stored_paired = pd.read_csv(paths["paired_deltas.csv"])
    interventions = pd.read_csv(paths["interventions.csv"])
    key_columns = ["seed", "task_id", "inference_mode"]
    for label, frame in (("shielded", raw), ("interventions", interventions)):
        if any(column not in frame for column in key_columns) or frame.duplicated(key_columns).any():
            raise ValueError(f"Stage-2 {label} keys are invalid")
        if set(frame[key_columns].itertuples(index=False, name=None)) != expected:
            raise ValueError(f"Stage-2 {label} keys do not match its manifest protocol")
        if "method" not in frame or set(frame["method"]) != {SHIELD_METHOD}:
            raise ValueError(f"Stage-2 {label} row schema/method is invalid")
    if "schema_version" not in raw or set(raw["schema_version"]) != {SHIELD_SCHEMA_VERSION}:
        raise ValueError("Stage-2 shielded row schema/method is invalid")
    trace_columns = (
        "executed_action_trace_path", "requested_action_trace_path", "intervention_records_path"
    )
    if any(column not in raw for column in trace_columns):
        raise ValueError("Stage-2 raw evidence is missing required trace/record paths")
    unshielded_root = Path(str(manifest.get("unshielded_result_root", ""))).resolve()
    unshielded_path = unshielded_root / "eval_raw.csv"
    manifest_candidates = [unshielded_root / name for name in ("shield_manifest.json", "manifest.json", "context_ab_manifest.json", "diagnostic_manifest.json")]
    unshielded_manifest_path = next((path for path in manifest_candidates if path.is_file() and not path.is_symlink()), None)
    if not unshielded_path.is_file() or unshielded_path.is_symlink() or unshielded_manifest_path is None:
        raise ValueError("Stage-2 unshielded comparator binding is missing")
    if manifest.get("unshielded_manifest_sha256") != _sha(unshielded_manifest_path):
        raise ValueError("Stage-2 unshielded manifest binding is stale")
    unshielded = pd.read_csv(unshielded_path)
    if any(column not in unshielded for column in key_columns) or unshielded.duplicated(key_columns).any():
        raise ValueError("Stage-2 unshielded keys are invalid")
    if set(unshielded[key_columns].itertuples(index=False, name=None)) != expected:
        raise ValueError("Stage-2 unshielded keys do not match the shielded protocol")
    recomputed_gate = evaluate_shield_gate(raw, unshielded, expected)
    recomputed_decision = stage2_source._stage2_decision(
        recomputed_gate, shield_fingerprint=shield_fingerprint
    )
    if decision != recomputed_decision:
        raise ValueError("Stage-2 decision is not authentic to the recomputed gate evidence")
    recomputed_paired = build_paired_shield_deltas(raw, unshielded, expected)
    try:
        pd.testing.assert_frame_equal(
            stored_paired.reset_index(drop=True), recomputed_paired.reset_index(drop=True),
            check_dtype=False, check_exact=False, rtol=1e-12, atol=1e-12,
        )
    except AssertionError as error:
        raise ValueError("Stage-2 paired deltas are not authentic to raw evidence") from error
    summary_columns = ["total_steps", "intervention_count", "ode_failure_count"]
    if any(column not in interventions or column not in raw for column in summary_columns):
        raise ValueError("Stage-2 intervention summary schema is incomplete")
    merged = raw[key_columns + summary_columns].merge(
        interventions[key_columns + summary_columns], on=key_columns,
        suffixes=("_raw", "_interventions"), validate="one_to_one",
    )
    for column in summary_columns:
        if not merged[f"{column}_raw"].equals(merged[f"{column}_interventions"]):
            raise ValueError(f"Stage-2 intervention evidence mismatch: {column}")
    from experiments.scripts.run_shielded_context_ab import (
        _json_records, _validate_trace_record_consistency,
    )
    evidence_hashes: dict[str, str] = {}
    for row in raw.itertuples(index=False):
        resolved: list[Path] = []
        for column in trace_columns:
            evidence_path = Path(str(getattr(row, column)))
            if not evidence_path.is_absolute(): evidence_path = root / evidence_path
            evidence_path = evidence_path.resolve()
            if not evidence_path.is_relative_to(root) or not evidence_path.is_file() or evidence_path.is_symlink():
                raise ValueError("Stage-2 trace/record evidence must be a final-root regular descendant")
            resolved.append(evidence_path)
            evidence_hashes[evidence_path.relative_to(root).as_posix()] = _sha(evidence_path)
        executed, requested, records = _validate_trace_record_consistency(
            __import__("numpy").load(resolved[0], allow_pickle=False),
            __import__("numpy").load(resolved[1], allow_pickle=False),
            _json_records(resolved[2]), decorate_steps=False,
        )
        summary = aggregate_episode_interventions(records, executed.shape[1])
        match = interventions.loc[
            (interventions["seed"] == row.seed) & (interventions["task_id"] == row.task_id)
            & (interventions["inference_mode"] == row.inference_mode)
        ]
        if len(match) != 1:
            raise ValueError("Stage-2 trace evidence has no one-to-one intervention summary")
        for name, value in summary.items():
            observed = match.iloc[0][name]
            if isinstance(value, list): observed = json.loads(observed) if isinstance(observed, str) else list(observed)
            if value is None and pd.isna(observed): continue
            if isinstance(value, float):
                if not __import__("numpy").isclose(float(observed), value, rtol=1e-12, atol=1e-12): raise ValueError("Stage-2 aggregate evidence mismatch")
            elif observed != value: raise ValueError("Stage-2 aggregate evidence mismatch")
    artifact_hashes = {name: _sha(path) for name, path in paths.items()}
    identity = _canonical_hash({
        "artifacts": artifact_hashes,
        "unshielded_eval_sha256": _sha(unshielded_path),
        "unshielded_manifest_sha256": _sha(unshielded_manifest_path),
        "evidence": evidence_hashes,
    })
    return {
        "root": root, "manifest": manifest, "decision": decision, "raw": raw,
        "paired": stored_paired, "interventions": interventions,
        "unshielded": unshielded, "stage2_identity_sha256": identity,
        "artifact_sha256": artifact_hashes,
    }


def _validate_shield_prerequisite(
    args: argparse.Namespace, suite: Any, runs: pd.DataFrame
) -> tuple[Path, dict[str, Any], dict[str, Any]]:
    """Validate immutable Stage-2 approval and output topology before execution."""
    decision_path = Path(args.stage2_decision).resolve()
    if not decision_path.is_file() or decision_path.is_symlink():
        raise FileNotFoundError("Stage-2 decision must be a regular file")
    stage2_evidence = load_stage2_evidence(decision_path)
    decision = stage2_evidence["decision"]
    manifest = stage2_evidence["manifest"]
    if set(decision) != {
        "schema_version", "method", "fixed_lambdas", "shield_fingerprint",
        "outcome", "stage", "conditions", "evidence", "reasons",
    }:
        raise ValueError("Stage-2 decision has invalid exact schema")
    if (
        decision["schema_version"] != SHIELD_SCHEMA_VERSION
        or decision["method"] != SHIELD_METHOD
        or decision["fixed_lambdas"] != manifest["fixed_lambdas"]
        or decision["shield_fingerprint"] != manifest["shield_fingerprint"]
    ):
        raise ValueError("Stage-2 decision v2 identity is inconsistent")
    if decision["stage"] != "stage2_shielded_context_ab" or decision["outcome"] != "continue_to_full_suite":
        raise ValueError("Stage-2 did not approve continuation to the full suite")
    conditions = decision["conditions"]
    if not isinstance(conditions, dict) or set(conditions) != set(STAGE2_CONDITIONS) or any(v is not True for v in conditions.values()):
        raise ValueError("Stage-2 requires the four exact passing conditions")
    if not isinstance(decision["evidence"], dict) or decision["reasons"] != []:
        raise ValueError("Stage-2 decision evidence/reasons are inconsistent")
    stage2_root = decision_path.parent
    manifest_path = stage2_root / "shield_manifest.json"
    if not manifest_path.is_file() or manifest_path.is_symlink():
        raise FileNotFoundError("Stage-2 shield_manifest.json is required")
    if manifest.get("method") != SHIELD_METHOD:
        raise ValueError("Stage-2 shield method is stale")
    # Reuse the Stage-2 implementation's canonical, source-sensitive values.
    from experiments.scripts import run_shielded_context_ab as stage2
    if manifest.get("fixed_lambdas") != list(stage2.DEFAULT_LAMBDAS):
        raise ValueError("Stage-2 fixed lambda grid is stale")
    if manifest.get("formal_solver_options") != dict(stage2.FORMAL_CVODES_OPTIONS):
        raise ValueError("Stage-2 formal solver configuration is stale")
    _, rule_sha = stage2._load_rule_params()
    checks = {
        "source_manifest_sha256": _sha(args.manifest),
        "source_tasks_sha256": _sha(args.tasks_csv),
        "rule_config_sha256": rule_sha,
        "env_config_sha256": _sha(stage2.ENV_CONFIG_PATH),
        "runtime_source_tree_sha256": stage2._runtime_source_tree_sha256(),
        **stage2._behavior_source_hashes(),
    }
    for name, expected in checks.items():
        if manifest.get(name) != expected:
            raise ValueError(f"Stage-2 provenance is stale: {name}")
    checkpoints = manifest.get("checkpoints")
    if not isinstance(checkpoints, list):
        raise ValueError("Stage-2 checkpoint provenance is missing")
    approved = {(int(item["seed"]), item["model_sha256"], item["vecnormalize_sha256"]) for item in checkpoints}
    selected = runs[(runs["algorithm"] == "agri_metarl") & (runs["status"] == "completed")]
    current = {(int(row.seed), _sha(row.model_path), _sha(row.vecnormalize_path)) for row in selected.itertuples(index=False)}
    if current != approved:
        raise ValueError("Stage-2 checkpoint provenance does not match selected full-suite runs")
    output = Path(args.result_root).resolve()
    protected = [Path(suite.result_root).resolve(), stage2_root]
    for key in ("unshielded_result_root", "stage1_root", "failure_root"):
        if manifest.get(key):
            protected.append(Path(manifest[key]).resolve())
    work = output.parent / f".{output.name}.work"
    for candidate in (output, work):
        if any(stage2._collides(candidate, root) for root in protected):
            raise ValueError("shield output/work roots must be disjoint from prerequisite roots")
    if output.exists() and not output.is_dir():
        raise ValueError("shield result_root exists as a file")
    return output, manifest, {**decision, "stage2_identity_sha256": stage2_evidence["stage2_identity_sha256"]}


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(value, handle, allow_nan=False, sort_keys=True, indent=2)
            handle.write("\n")
        os.replace(temp, path)
    except BaseException:
        try: os.unlink(temp)
        except OSError: pass
        raise


def close_environment(env: Any, primary_error: BaseException | None) -> None:
    """Close an environment without masking an episode/model failure."""
    try:
        env.close()
    except BaseException as close_error:
        if primary_error is None:
            raise
        primary_error.add_note(
            f"environment close also failed: {type(close_error).__name__}: {close_error}"
        )


def _replace_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    os.close(handle)
    try:
        frame.to_csv(temporary, index=False)
        for attempt in range(5):
            try:
                os.replace(temporary, path)
                break
            except PermissionError:
                if attempt == 4:
                    raise
                time.sleep(0.05 * (attempt + 1))
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _read_progress_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, float_precision="round_trip")


def replace_directory_atomic(stage: Path, root: Path) -> None:
    """Commit a prepared directory with verified fallback restoration."""
    from gl_gym.experiments.shield_evaluation import _restore_prior_root
    backup = root.parent / f".{root.name}.backup-{uuid.uuid4().hex}"
    old_moved = False
    published = False
    try:
        if root.exists():
            os.replace(root, backup)
            old_moved = True
        try:
            os.replace(stage, root)
            published = True
        except BaseException as error:
            if old_moved:
                restored = _restore_prior_root(backup, root, error)
                if restored:
                    old_moved = False
            raise
        if old_moved:
            shutil.rmtree(backup, ignore_errors=True)
            old_moved = False
    finally:
        if stage.exists(): shutil.rmtree(stage, ignore_errors=True)
        if backup.exists() and not published and old_moved and not root.exists():
            try: os.replace(backup, root)
            except Exception: pass


def _atomic_npy(path: Path, array: Any) -> None:
    import numpy as np
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle: np.save(handle, array, allow_pickle=False)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary): os.unlink(temporary)


def _clear_shield_work(work: Path) -> None:
    if work.exists(): shutil.rmtree(work)
    work.mkdir(parents=True, exist_ok=True)


def _read_shield_progress(work: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw_path, evidence_path = work / "eval_raw.csv", work / "interventions.csv"
    if raw_path.is_file() != evidence_path.is_file():
        _clear_shield_work(work)
        return pd.DataFrame(), pd.DataFrame()
    if not raw_path.is_file(): return pd.DataFrame(), pd.DataFrame()
    try:
        raw, evidence = _read_progress_csv(raw_path), _read_progress_csv(evidence_path)
    except Exception:
        _clear_shield_work(work)
        return pd.DataFrame(), pd.DataFrame()
    required = {
        "algorithm", "method", "seed", "task_id", "model_sha256", "vecnormalize_sha256",
        "schema_version",
        "checkpoint_steps", "source_fingerprint_sha256", "stage2_identity_sha256",
        "executed_action_trace_path", "requested_action_trace_path", "intervention_records_path",
        "executed_action_trace_sha256", "requested_action_trace_sha256",
        "intervention_records_sha256", "episode_evidence_identity_sha256",
        "formal_complete", "rule_config_sha256", "env_config_sha256",
        "fixed_lambdas_json", "formal_solver_options_json",
        "shield_method_fingerprint_sha256",
    }
    if (
        not required.issubset(raw.columns) or not required.issubset(evidence.columns)
        or raw.duplicated(["algorithm", "seed", "task_id"]).any()
        or evidence.duplicated(["algorithm", "seed", "task_id"]).any()
    ):
        _clear_shield_work(work)
        return pd.DataFrame(), pd.DataFrame()
    return raw, evidence


def _shield_row_valid(
    row: Mapping[str, Any], evidence: Mapping[str, Any], expected: Mapping[str, Any],
    *, work_root: Path,
) -> bool:
    import numpy as np
    try:
        formal_expected = expected.get("formal_complete")
        if not isinstance(formal_expected, bool):
            return False
        for candidate in (row.get("formal_complete"), evidence.get("formal_complete")):
            if not isinstance(candidate, (bool, np.bool_)) or bool(candidate) is not formal_expected:
                return False
        for name, value in expected.items():
            if row[name] != value or evidence[name] != value: return False
        seed = row["seed"]
        steps = row["checkpoint_steps"]
        if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
            return False
        if isinstance(steps, (bool, np.bool_)) or not isinstance(steps, (int, np.integer)) or int(steps) < 0:
            return False
        if int(evidence["checkpoint_steps"]) != int(steps): return False
        for column in ("executed_action_trace", "requested_action_trace", "intervention_records"):
            path = Path(str(row[f"{column}_path"])).resolve()
            if not path.is_relative_to(work_root.resolve()): return False
            if path != Path(str(evidence[f"{column}_path"])).resolve() or not path.is_file() or path.is_symlink(): return False
            observed = _sha(path)
            if row[f"{column}_sha256"] != observed or evidence[f"{column}_sha256"] != observed: return False
        executed = np.load(row["executed_action_trace_path"], allow_pickle=False)
        requested = np.load(row["requested_action_trace_path"], allow_pickle=False)
        records = json.loads(Path(row["intervention_records_path"]).read_text(encoding="utf-8"))
        if executed.ndim != 2 or requested.shape != executed.shape or len(records) != len(executed): return False
        summary = aggregate_episode_interventions(records, executed.shape[1])
        for name, value in summary.items():
            if isinstance(value, list): continue
            observed = evidence[name]
            if value is None and pd.isna(observed): continue
            if observed != value: return False
        identity_payload = {name: row[name] for name in expected}
        identity_payload.update({name: row[name] for name in (
            "seed", "task_id", "split", "weather_year", "start_day", "uncertainty_scale",
            "economic_scenario", "climate_constraint_scenario", "checkpoint_steps",
            "episode_return", "temp_violation", "co2_violation", "rh_violation",
            "executed_action_trace_sha256", "requested_action_trace_sha256", "intervention_records_sha256",
        )})
        computed = _canonical_hash(identity_payload)
        return row["episode_evidence_identity_sha256"] == computed == evidence["episode_evidence_identity_sha256"]
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
        return False


def _publish_shield_final(
    root: Path, work: Path, raw: pd.DataFrame, interventions: pd.DataFrame,
    *, manifest_base: Mapping[str, Any],
) -> None:
    stage = Path(tempfile.mkdtemp(prefix=f".{root.name}.stage-", dir=root.parent))
    try:
        published_raw, published_interventions = raw.copy(), interventions.copy()
        evidence_hashes: dict[str, str] = {}
        for index, row in raw.iterrows():
            token = f"seed{int(row.seed)}__{row.task_id}"
            for column, directory, suffix in (
                ("executed_action_trace_path", "traces", "executed.npy"),
                ("requested_action_trace_path", "traces", "requested.npy"),
                ("intervention_records_path", "intervention_records", "records.json"),
            ):
                relative = Path(directory) / f"{token}__{suffix}"
                destination = stage / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(Path(row[column]), destination)
                evidence_hashes[relative.as_posix()] = _sha(destination)
                published_raw.at[index, column] = relative.as_posix()
                match = (published_interventions["seed"] == row.seed) & (published_interventions["task_id"] == row.task_id)
                published_interventions.loc[match, column] = relative.as_posix()
        published_raw.to_csv(stage / "eval_raw.csv", index=False)
        published_interventions.to_csv(stage / "interventions.csv", index=False)
        manifest = {
            **dict(manifest_base), "schema_version": SHIELD_SCHEMA_VERSION,
            "result_root": str(root), "eval_raw_sha256": _sha(stage / "eval_raw.csv"),
            "interventions_sha256": _sha(stage / "interventions.csv"),
            "evidence_sha256": evidence_hashes,
        }
        _atomic_json(stage / "evaluation_manifest.json", manifest)
        replace_directory_atomic(stage, root)
    except BaseException:
        if stage.exists(): shutil.rmtree(stage, ignore_errors=True)
        raise


def prepare_shield_resume(
    eval_path: Path, interventions_path: Path, *, stage2_manifest_sha256: str
) -> set[tuple[str, int, str]]:
    """Retain only rows whose full shield evidence and provenance still validate."""
    if not eval_path.is_file() or not interventions_path.is_file():
        return set()
    import numpy as np
    raw = pd.read_csv(eval_path)
    evidence = pd.read_csv(interventions_path)
    required = {
        "algorithm", "method", "seed", "task_id", "completed", "model_sha256",
        "vecnormalize_sha256", "stage2_manifest_sha256", "executed_action_trace_path",
        "requested_action_trace_path", "intervention_records_path",
    }
    valid: set[tuple[str, int, str]] = set()
    valid_evidence_indices: list[int] = []
    if required.issubset(evidence.columns):
        for index, row in evidence.iterrows():
            key = (str(row.algorithm), int(row.seed), str(row.task_id))
            try:
                if (
                    row.schema_version != SHIELD_SCHEMA_VERSION
                    or row.method != SHIELD_METHOD or row.algorithm != "agri_metarl" + SHIELD_SUFFIX
                    or row.completed not in (True, 1) or row.stage2_manifest_sha256 != stage2_manifest_sha256
                ):
                    raise ValueError("stale evidence provenance")
                executed = np.load(Path(row.executed_action_trace_path), allow_pickle=False)
                requested = np.load(Path(row.requested_action_trace_path), allow_pickle=False)
                records = json.loads(Path(row.intervention_records_path).read_text(encoding="utf-8"))
                if executed.ndim != 2 or requested.shape != executed.shape or len(records) != len(executed):
                    raise ValueError("trace shape mismatch")
                summary = aggregate_episode_interventions(records, executed.shape[1])
                if int(row.total_steps) != summary["total_steps"] or int(row.intervention_count) != summary["intervention_count"]:
                    raise ValueError("summary mismatch")
                raw_matches = raw[
                    (raw["algorithm"] == key[0]) & (raw["seed"] == key[1]) & (raw["task_id"] == key[2])
                ]
                if len(raw_matches) != 1:
                    raise ValueError("raw evidence must be one-to-one")
            except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
                continue
            valid.add(key); valid_evidence_indices.append(index)
    cleaned_raw = raw[
        [
            (str(row.algorithm), int(row.seed), str(row.task_id)) in valid
            for row in raw.itertuples(index=False)
        ]
    ]
    cleaned_evidence = evidence.loc[valid_evidence_indices]
    if len(cleaned_raw) != len(raw): _replace_csv(cleaned_raw, eval_path)
    if len(cleaned_evidence) != len(evidence): _replace_csv(cleaned_evidence, interventions_path)
    return valid


def run_shield_evaluation(
    args: argparse.Namespace, suite: Any, runs: pd.DataFrame, tasks: pd.DataFrame,
    *, model_map: Mapping[str, Any], env_loader: Callable[..., Any], episode_runner: Callable[..., Any],
) -> int:
    """Run shield episodes into work storage and publish only a complete formal design."""
    if set(runs["algorithm"]) - {"agri_metarl"}:
        raise ValueError("the approved action-shield experiment supports only agri_metarl")
    root, stage2_manifest, stage2_decision = _validate_shield_prerequisite(args, suite, runs)
    stage2_identity = stage2_decision["stage2_identity_sha256"]
    formal = args.splits is None and args.task_ids is None and args.limit_tasks is None
    selected_tasks = filter_tasks(tasks, splits=args.splits, task_ids=args.task_ids, limit_tasks=args.limit_tasks)
    task_records = [task_from_row(row) for row in selected_tasks.itertuples(index=False)]
    if formal:
        canonical = pd.DataFrame(vars(item) for item in build_evaluation_tasks(suite))
        try:
            pd.testing.assert_frame_equal(
                tasks.reset_index(drop=True), canonical.reset_index(drop=True),
                check_dtype=False, check_exact=True,
            )
        except AssertionError as error:
            raise ValueError("formal shield evaluation requires the exact canonical 91 tasks") from error
    if formal and args.interventions_out is not None:
        expected_interventions = root / "interventions.csv"
        if Path(args.interventions_out).resolve() != expected_interventions:
            raise ValueError("formal shield interventions must be contained in the atomic result root")
    work = root.parent / f".{root.name}.work"
    if not args.resume_eval: _clear_shield_work(work)
    else: work.mkdir(parents=True, exist_ok=True)
    raw, interventions = _read_shield_progress(work)

    from experiments.scripts import run_shielded_context_ab as stage2
    shield_params, rule_sha = stage2._load_rule_params()
    source_payload = {
        "source_manifest_sha256": _sha(args.manifest), "runs_csv_sha256": _sha(args.runs_csv),
        "tasks_csv_sha256": _sha(args.tasks_csv), "rule_config_sha256": rule_sha,
        "env_config_sha256": _sha(stage2.ENV_CONFIG_PATH),
        "runtime_source_tree_sha256": stage2._runtime_source_tree_sha256(),
        "evaluator_source_sha256": _sha(Path(__file__)),
        "gate_source_sha256": _sha(ROOT / "experiments/scripts/evaluate_shield_gate.py"),
    }
    source_identity_inputs = {
        name: source_payload[name] for name in (
            "source_manifest_sha256", "runs_csv_sha256", "tasks_csv_sha256",
            "runtime_source_tree_sha256", "evaluator_source_sha256", "gate_source_sha256",
        )
    }
    source_fingerprint = _canonical_hash(source_identity_inputs)
    method_components = shield_method_fingerprint_components(
        source_inputs=source_identity_inputs,
        rule_config_sha256=rule_sha,
        env_config_sha256=source_payload["env_config_sha256"],
        stage2_identity_sha256=stage2_identity,
        fixed_lambdas=stage2.DEFAULT_LAMBDAS,
        formal_solver_options=stage2.FORMAL_CVODES_OPTIONS,
    )
    method_fingerprint = _canonical_hash(method_components)
    completed: dict[tuple[str, int, str], tuple[dict[str, Any], dict[str, Any]]] = {}
    published_checkpoints: dict[int, dict[str, Any]] = {}
    selected_runs = [row for row in runs.itertuples(index=False) if row.status == "completed"]
    for run_row in selected_runs:
        validate_completed_run_paths(run_row)
        model_sha, vec_sha = _sha(run_row.model_path), _sha(run_row.vecnormalize_path)
        model = model_map[run_row.algorithm].load(run_row.model_path, device="cpu")
        checkpoint_steps = getattr(model, "num_timesteps", None)
        import numpy as np
        if isinstance(checkpoint_steps, (bool, np.bool_)) or not isinstance(checkpoint_steps, (int, np.integer)) or int(checkpoint_steps) < 0:
            raise ValueError("model checkpoint_steps must be a nonnegative strict integer")
        authentic_checkpoint = next(
            (item for item in stage2_manifest["checkpoints"] if int(item["seed"]) == int(run_row.seed)),
            None,
        )
        if (
            authentic_checkpoint is None
            or authentic_checkpoint.get("checkpoint_steps") != int(checkpoint_steps)
            or authentic_checkpoint.get("model_sha256") != model_sha
            or authentic_checkpoint.get("vecnormalize_sha256") != vec_sha
        ):
            raise ValueError("shield checkpoint provenance does not match authentic Stage-2")
        expected_common = {
            "schema_version": SHIELD_SCHEMA_VERSION,
            "algorithm": "agri_metarl" + SHIELD_SUFFIX, "method": SHIELD_METHOD,
            "model_sha256": model_sha, "vecnormalize_sha256": vec_sha,
            "checkpoint_steps": int(checkpoint_steps), "source_fingerprint_sha256": source_fingerprint,
            "stage2_identity_sha256": stage2_identity,
            "runtime_source_tree_sha256": source_payload["runtime_source_tree_sha256"],
            "source_manifest_sha256": source_payload["source_manifest_sha256"],
            "runs_csv_sha256": source_payload["runs_csv_sha256"],
            "tasks_csv_sha256": source_payload["tasks_csv_sha256"],
            "formal_complete": formal,
            "rule_config_sha256": source_payload["rule_config_sha256"],
            "env_config_sha256": source_payload["env_config_sha256"],
            "fixed_lambdas_json": json.dumps(
                method_components["fixed_lambdas"], sort_keys=True, separators=(",", ":")
            ),
            "formal_solver_options_json": json.dumps(
                method_components["formal_solver_options"], sort_keys=True, separators=(",", ":")
            ),
            "shield_method_fingerprint_sha256": method_fingerprint,
        }
        published_checkpoints[int(run_row.seed)] = {
            "seed": int(run_row.seed), "model_sha256": model_sha,
            "vecnormalize_sha256": vec_sha, "checkpoint_steps": int(checkpoint_steps),
        }
        if not raw.empty and not interventions.empty:
            for raw_row in raw.loc[raw.get("seed", pd.Series(dtype=object)) == int(run_row.seed)].to_dict("records"):
                matches = interventions.loc[
                    (interventions.get("seed", pd.Series(dtype=object)) == int(run_row.seed))
                    & (interventions.get("task_id", pd.Series(dtype=object)) == raw_row.get("task_id"))
                ]
                if len(matches) == 1 and _shield_row_valid(
                    raw_row, matches.iloc[0].to_dict(), expected_common, work_root=work
                ):
                    completed[(expected_common["algorithm"], int(run_row.seed), str(raw_row["task_id"]))] = (raw_row, matches.iloc[0].to_dict())
        for task in task_records:
            key = (expected_common["algorithm"], int(run_row.seed), task.task_id)
            if key in completed: continue
            env = env_loader(suite, task, run_row.vecnormalize_path, shield_params=shield_params)
            primary: BaseException | None = None
            try:
                metrics, diagnostics = episode_runner(model, env, return_diagnostics=True)
            except BaseException as error:
                primary = error
                raise
            finally:
                close_environment(env, primary)
            executed = diagnostics.get("action_trace")
            requested = diagnostics.get("requested_action_trace")
            records = diagnostics.get("action_shield_records")
            if not isinstance(executed, np.ndarray) or not isinstance(requested, np.ndarray) or not isinstance(records, (list, tuple)) or executed.ndim != 2 or requested.shape != executed.shape or len(records) != len(executed):
                raise ValueError("incomplete action-shield diagnostics")
            indexed = [dict(record, step_index=index) for index, record in enumerate(records)]
            summary = aggregate_episode_interventions(indexed, executed.shape[1])
            token = f"seed{int(run_row.seed)}__{task.task_id}"
            executed_path = work / "traces" / f"{token}__executed.npy"
            requested_path = work / "traces" / f"{token}__requested.npy"
            records_path = work / "intervention_records" / f"{token}__records.json"
            _atomic_npy(executed_path, executed); _atomic_npy(requested_path, requested); _atomic_json(records_path, indexed)
            hashes = {
                "executed_action_trace_sha256": _sha(executed_path),
                "requested_action_trace_sha256": _sha(requested_path),
                "intervention_records_sha256": _sha(records_path),
            }
            descriptor = {
                "suite_id": suite.suite_id, "seed": int(run_row.seed), "run_name": run_row.run_name,
                "task_id": task.task_id, "split": task.split, "weather_year": task.weather_year,
                "start_day": task.start_day, "uncertainty_scale": task.uncertainty_scale,
                "economic_scenario": task.economic_scenario,
                "climate_constraint_scenario": task.climate_constraint_scenario,
            }
            identity_payload = {
                **expected_common, **{name: descriptor[name] for name in descriptor if name != "run_name"},
                "checkpoint_steps": int(checkpoint_steps),
                **{name: metrics[name] for name in ("episode_return", "temp_violation", "co2_violation", "rh_violation")},
                **hashes,
            }
            row = {
                **descriptor, **expected_common, **metrics, "trajectory_path": "", "completed": True,
                "ode_failure_count": 0,
                "executed_action_trace_path": str(executed_path.resolve()),
                "requested_action_trace_path": str(requested_path.resolve()),
                "intervention_records_path": str(records_path.resolve()), **hashes,
                "episode_evidence_identity_sha256": _canonical_hash(identity_payload),
            }
            evidence = {**row, **summary}
            completed[key] = (row, evidence)
            ordered = [completed[item] for item in sorted(completed)]
            raw = pd.DataFrame([item[0] for item in ordered]); interventions = pd.DataFrame([item[1] for item in ordered])
            _replace_csv(raw, work / "eval_raw.csv"); _replace_csv(interventions, work / "interventions.csv")
            print(f"Evaluated agri_metarl seed={run_row.seed} task={task.task_id}", flush=True)
    if not formal:
        print(f"Wrote nonformal shield work artifacts to {work}")
        return len(completed)
    approved_seeds = {int(item["seed"]) for item in stage2_manifest["checkpoints"]}
    expected_keys = {("agri_metarl" + SHIELD_SUFFIX, seed, task.task_id) for seed in approved_seeds for task in task_records}
    if set(completed) != expected_keys:
        raise RuntimeError(f"shield formal evaluation incomplete: expected {len(expected_keys)}, got {len(completed)}")
    raw = pd.DataFrame([completed[key][0] for key in sorted(completed)])
    interventions = pd.DataFrame([completed[key][1] for key in sorted(completed)])
    manifest_base = {
        "suite_id": suite.suite_id, "algorithm": "agri_metarl" + SHIELD_SUFFIX,
        "method": SHIELD_METHOD, "formal_complete": True,
        "approved_seeds": sorted(approved_seeds), "task_count": 91, "episode_count": len(raw),
        "checkpoints": [published_checkpoints[seed] for seed in sorted(published_checkpoints)],
        "stage2_identity_sha256": stage2_identity,
        "source_fingerprint_sha256": source_fingerprint, **source_payload,
        "source_fingerprint_inputs": source_identity_inputs,
        "source_checksum_mapping": _formal_source_checksums(args, source_payload),
        "source_input_paths": {
            "manifest": str(Path(args.manifest).resolve()),
            "runs_csv": str(Path(args.runs_csv).resolve()),
            "tasks_csv": str(Path(args.tasks_csv).resolve()),
        },
        "shield_method_fingerprint_components": method_components,
        "shield_method_fingerprint_sha256": method_fingerprint,
    }
    _publish_shield_final(root, work, raw, interventions, manifest_base=manifest_base)
    print(f"Published {len(raw)} shield rows atomically to {root}")
    return len(raw)


def run_formal_unshielded_evaluation(
    args: argparse.Namespace, suite: Any, runs: pd.DataFrame, tasks: pd.DataFrame,
    *, model_map: Mapping[str, Any], env_loader: Callable[..., Any], episode_runner: Callable[..., Any],
) -> int:
    """Produce the provenance-bearing full-suite unshielded comparator atomically."""
    if args.interventions_out is not None or args.splits or args.task_ids or args.limit_tasks is not None:
        raise ValueError("formal unshielded evaluation forbids smoke filters/intervention output")
    if set(runs["algorithm"]) != {"agri_metarl"}:
        raise ValueError("formal unshielded evaluation supports only agri_metarl")
    root, stage2_manifest, stage2_decision = _validate_shield_prerequisite(args, suite, runs)
    canonical = pd.DataFrame(vars(item) for item in build_evaluation_tasks(suite))
    try:
        pd.testing.assert_frame_equal(tasks.reset_index(drop=True), canonical, check_dtype=False, check_exact=True)
    except AssertionError as error:
        raise ValueError("formal unshielded evaluation requires exact canonical tasks") from error
    task_records = [task_from_row(row) for row in tasks.itertuples(index=False)]
    task_by_id = {task.task_id: task for task in task_records}
    approved = {int(item["seed"]): item for item in stage2_manifest["checkpoints"]}
    if set(approved) != {42, 123} or any("checkpoint_steps" not in item for item in approved.values()):
        raise ValueError("authentic Stage-2 checkpoints must bind seeds and checkpoint_steps")
    work = root.parent / f".{root.name}.work"
    if not args.resume_eval: _clear_shield_work(work)
    else: work.mkdir(parents=True, exist_ok=True)
    progress_path = work / "eval_raw.csv"
    try:
        progress = _read_progress_csv(progress_path) if args.resume_eval and progress_path.is_file() else pd.DataFrame()
    except Exception:
        _clear_shield_work(work); progress = pd.DataFrame()
    from experiments.scripts import run_shielded_context_ab as source
    runtime_sha = source._runtime_source_tree_sha256()
    source_payload = {
        "source_manifest_sha256": _sha(args.manifest), "runs_csv_sha256": _sha(args.runs_csv),
        "tasks_csv_sha256": _sha(args.tasks_csv), "runtime_source_tree_sha256": runtime_sha,
        "evaluator_source_sha256": _sha(Path(__file__)),
        "gate_source_sha256": _sha(ROOT / "experiments/scripts/evaluate_shield_gate.py"),
    }
    source_identity_inputs = {
        name: source_payload[name] for name in (
            "source_manifest_sha256", "runs_csv_sha256", "tasks_csv_sha256",
            "runtime_source_tree_sha256", "evaluator_source_sha256", "gate_source_sha256",
        )
    }
    source_fingerprint = _canonical_hash(source_identity_inputs)
    from experiments.scripts.run_context_ab import _package_versions, _provenance
    execution_provenance = _provenance()
    package_versions = _package_versions()
    source_checksums = _formal_source_checksums(args, source_payload)
    completed: dict[tuple[int, str], dict[str, Any]] = {}
    required_progress = {
        "seed", "task_id", "algorithm", "method", "model_sha256", "vecnormalize_sha256",
        "checkpoint_steps", "source_fingerprint_sha256", "runtime_source_tree_sha256",
        "source_manifest_sha256", "runs_csv_sha256", "tasks_csv_sha256",
        "stage2_identity_sha256", "episode_evidence_identity_sha256", "completed", "status",
        "ode_failure_count", "failure_evidence_path", "failure_evidence_identity_sha256",
        "failure_episode_step", "failure_configured_horizon",
    }
    if not progress.empty and (
        not required_progress.issubset(progress.columns) or progress.duplicated(["seed", "task_id"]).any()
    ):
        _clear_shield_work(work); progress = pd.DataFrame()
    checkpoint_records: list[dict[str, Any]] = []
    for run_row in [row for row in runs.itertuples(index=False) if row.status == "completed"]:
        validate_completed_run_paths(run_row)
        model_sha, vec_sha = _sha(run_row.model_path), _sha(run_row.vecnormalize_path)
        expected_checkpoint = approved[int(run_row.seed)]
        if (model_sha, vec_sha) != (expected_checkpoint["model_sha256"], expected_checkpoint["vecnormalize_sha256"]):
            raise ValueError("unshielded checkpoint hashes do not match authentic Stage-2")
        model = model_map[run_row.algorithm].load(run_row.model_path, device="cpu")
        import numpy as np
        checkpoint_steps = getattr(model, "num_timesteps", None)
        if isinstance(checkpoint_steps, (bool, np.bool_)) or not isinstance(checkpoint_steps, (int, np.integer)) or int(checkpoint_steps) != expected_checkpoint["checkpoint_steps"]:
            raise ValueError("unshielded checkpoint_steps do not match authentic Stage-2")
        checkpoint_records.append({
            "seed": int(run_row.seed), "model_sha256": model_sha, "vecnormalize_sha256": vec_sha,
            "checkpoint_steps": int(checkpoint_steps), "source_fingerprint_sha256": source_fingerprint,
            "runtime_source_tree_sha256": runtime_sha,
        })
        for old in progress.loc[progress.get("seed", pd.Series(dtype=object)) == int(run_row.seed)].to_dict("records") if not progress.empty else []:
            expected_fields = {
                "algorithm": "agri_metarl", "method": FORMAL_UNSHIELDED_METHOD,
                "model_sha256": model_sha, "vecnormalize_sha256": vec_sha,
                "checkpoint_steps": int(checkpoint_steps), "source_fingerprint_sha256": source_fingerprint,
                "runtime_source_tree_sha256": runtime_sha,
                "source_manifest_sha256": source_payload["source_manifest_sha256"],
                "runs_csv_sha256": source_payload["runs_csv_sha256"],
                "tasks_csv_sha256": source_payload["tasks_csv_sha256"],
                "stage2_identity_sha256": stage2_decision["stage2_identity_sha256"],
            }
            try:
                normalized_old = canonical_evaluation_row(old)
                if all(normalized_old[name] == value for name, value in expected_fields.items()):
                    task = task_by_id[str(normalized_old["task_id"])]
                    expected_context = CapsuleContext(
                        seed=int(run_row.seed), task_id=task.task_id,
                        inference_mode="stage3_unshielded", task=dict(vars(task)),
                        checkpoint_path=str(Path(run_row.model_path).resolve()),
                        checkpoint_sha256=model_sha,
                        git_head=execution_provenance["git_commit"],
                        dirty=execution_provenance["dirty"],
                        source_checksums=source_checksums,
                        package_versions=package_versions,
                        formal_result_root=str(root.resolve()),
                    )
                    status_valid = (
                        normalized_old["completed"] is True
                        and normalized_old["status"] == "completed"
                        and normalized_old["failure_evidence_path"] == ""
                        and normalized_old["failure_evidence_identity_sha256"] == ""
                        and normalized_old["failure_episode_step"] is None
                        and normalized_old["failure_configured_horizon"] is None
                    ) or _resumed_failure_capsule_valid(
                        normalized_old,
                        expected_context=expected_context,
                        expected_solver_options=stage2_manifest["formal_solver_options"],
                        work=work,
                        result_root=root,
                    )
                    if status_valid and normalized_old["episode_evidence_identity_sha256"] == evaluation_row_identity(normalized_old):
                        completed[(int(run_row.seed), str(normalized_old["task_id"]))] = normalized_old
            except (KeyError, TypeError, ValueError): pass
        for task in task_records:
            key = (int(run_row.seed), task.task_id)
            if key in completed: continue
            task_token = _canonical_hash(str(key[1]))[:12]
            attempt_root = _long_path(
                work / "failures" / "attempts" / source_fingerprint[:16]
                / str(key[0]) / task_token / uuid.uuid4().hex
            )
            context = CapsuleContext(
                seed=key[0], task_id=task.task_id, inference_mode="stage3_unshielded",
                task=dict(vars(task)), checkpoint_path=str(Path(run_row.model_path).resolve()),
                checkpoint_sha256=model_sha, git_head=execution_provenance["git_commit"],
                dirty=execution_provenance["dirty"], source_checksums=source_checksums,
                package_versions=package_versions,
                formal_result_root=str(root.resolve()),
            )
            recorder = FailureCapsuleRecorder(attempt_root, context)
            env = env_loader(suite, task, run_row.vecnormalize_path)
            error: Exception | None = None
            try:
                try:
                    metrics = episode_runner(model, env, failure_recorder=recorder)
                except Exception as caught:
                    error = caught
                    metrics = {name: None for name in ("episode_return", "temp_violation", "co2_violation", "rh_violation")}
            finally:
                close_environment(env, sys.exception() or error)
            manifests = sorted(attempt_root.rglob("manifest.json")) if attempt_root.exists() else []
            failure_path = ""
            failure_identity = ""
            failure_episode_step = None
            failure_configured_horizon = None
            if error is not None:
                try:
                    if len(manifests) != 1:
                        raise ValueError(
                            f"expected exactly one new failure capsule, found {len(manifests)}"
                        )
                    (
                        failure_identity, _, failure_episode_step,
                        failure_configured_horizon,
                    ) = _validate_attempt_capsule(
                        manifests[0], expected_context=context,
                        expected_solver_options=stage2_manifest["formal_solver_options"],
                        error=error,
                    )
                    failure_path = str(manifests[0].resolve())
                except Exception as capsule_error:
                    if attempt_root.exists():
                        shutil.rmtree(attempt_root, ignore_errors=True)
                    error.add_note(f"ODE failure classification rejected: {capsule_error}")
                    raise error
            elif manifests:
                shutil.rmtree(attempt_root, ignore_errors=True)
                raise ValueError("successful formal episode unexpectedly produced a failure capsule")
            row = {
                "suite_id": suite.suite_id, "algorithm": "agri_metarl", "method": FORMAL_UNSHIELDED_METHOD,
                "seed": key[0], "run_name": run_row.run_name, "task_id": task.task_id,
                "split": task.split, "weather_year": task.weather_year, "start_day": task.start_day,
                "uncertainty_scale": task.uncertainty_scale, "economic_scenario": task.economic_scenario,
                "climate_constraint_scenario": task.climate_constraint_scenario,
                **metrics, "completed": error is None, "status": "completed" if error is None else "ode_failure",
                "ode_failure_count": 0 if error is None else 1, "failure_evidence_path": failure_path,
                "failure_evidence_identity_sha256": failure_identity,
                "failure_episode_step": failure_episode_step,
                "failure_configured_horizon": failure_configured_horizon,
                "model_sha256": model_sha, "vecnormalize_sha256": vec_sha,
                "checkpoint_steps": int(checkpoint_steps), "source_fingerprint_sha256": source_fingerprint,
                "runtime_source_tree_sha256": runtime_sha,
                "source_manifest_sha256": source_payload["source_manifest_sha256"],
                "runs_csv_sha256": source_payload["runs_csv_sha256"],
                "tasks_csv_sha256": source_payload["tasks_csv_sha256"],
                "stage2_identity_sha256": stage2_decision["stage2_identity_sha256"],
            }
            row = canonical_evaluation_row(row)
            row["episode_evidence_identity_sha256"] = evaluation_row_identity(row)
            completed[key] = row
            _replace_csv(pd.DataFrame([completed[item] for item in sorted(completed)]), progress_path)
    expected = {(seed, task.task_id) for seed in approved for task in task_records}
    if set(completed) != expected:
        raise RuntimeError(f"formal unshielded comparator incomplete: expected 182 keys, got {len(completed)}")
    final = pd.DataFrame([completed[key] for key in sorted(completed)])
    stage = Path(tempfile.mkdtemp(prefix=f".{root.name}.stage-", dir=root.parent))
    try:
        published = final.copy(); evidence_hashes = {}; capsule_identities = {}
        source_attempt_roots: set[Path] = set()
        for index, row in final.loc[~final["completed"]].iterrows():
            source_path = Path(row.failure_evidence_path).parent
            attempts_root = _long_path(work / "failures" / "attempts")
            resolved_source = source_path.resolve()
            if resolved_source.is_relative_to(root.resolve()):
                pass
            elif resolved_source.is_relative_to(attempts_root):
                attempt_root = source_path.parents[3]
                source_attempt_roots.add(attempt_root)
            else:
                raise ValueError("failure capsule is not contained in its evaluation roots")
            relative_dir = (
                Path("failure_evidence")
                / f"seed{int(row.seed)}__{_canonical_hash(str(row.task_id))[:12]}"
                / source_path.name
            )
            shutil.copytree(source_path, stage / relative_dir)
            relative_manifest = relative_dir / "manifest.json"
            published.at[index, "failure_evidence_path"] = relative_manifest.as_posix()
            capsule_identities[relative_manifest.as_posix()] = row.failure_evidence_identity_sha256
            for evidence_file in sorted((stage / relative_dir).iterdir()):
                evidence_hashes[evidence_file.relative_to(stage).as_posix()] = _sha(evidence_file)
        for index, row in published.iterrows():
            normalized = canonical_evaluation_row(row.to_dict())
            published.at[index, "episode_evidence_identity_sha256"] = evaluation_row_identity(normalized)
        published.to_csv(stage / "eval_raw.csv", index=False)
        manifest = {
            "schema_version": "formal-unshielded-evaluation-v1", "result_root": str(root),
            "formal_complete": True, "method": FORMAL_UNSHIELDED_METHOD,
            "eval_raw_sha256": _sha(stage / "eval_raw.csv"), "checkpoints": checkpoint_records,
            "source_fingerprint_sha256": source_fingerprint, "runtime_source_tree_sha256": runtime_sha,
            **source_payload, "source_fingerprint_inputs": source_identity_inputs,
            "source_checksum_mapping": source_checksums,
            "source_input_paths": {
                "manifest": str(Path(args.manifest).resolve()),
                "runs_csv": str(Path(args.runs_csv).resolve()),
                "tasks_csv": str(Path(args.tasks_csv).resolve()),
            },
            "stage2_identity_sha256": stage2_decision["stage2_identity_sha256"],
            "evaluator_source_sha256": source_payload["evaluator_source_sha256"],
            "gate_source_sha256": source_payload["gate_source_sha256"],
            "failure_evidence_sha256": evidence_hashes,
            "failure_capsule_identities": capsule_identities,
        }
        _atomic_json(stage / "evaluation_manifest.json", manifest)
        replace_directory_atomic(stage, root)
        resume_frame = published.copy()
        for index, row in resume_frame.loc[~resume_frame["completed"]].iterrows():
            resume_frame.at[index, "failure_evidence_path"] = str(
                (root / str(row.failure_evidence_path)).resolve()
            )
        resume_records = []
        for record in resume_frame.to_dict("records"):
            normalized = canonical_evaluation_row(record)
            normalized["episode_evidence_identity_sha256"] = evaluation_row_identity(normalized)
            resume_records.append(normalized)
        _replace_csv(pd.DataFrame(resume_records), progress_path)
        for source_attempt_root in source_attempt_roots:
            shutil.rmtree(source_attempt_root, ignore_errors=True)
    finally:
        if stage.exists():
            shutil.rmtree(stage, ignore_errors=True)
    return len(final)


def run(
    args: argparse.Namespace,
    *,
    model_map: Mapping[str, Any] = ALG_MAP,
    env_loader: Callable[..., Any] = load_task_env,
    episode_runner: Callable[..., Any] = run_deterministic_episode,
) -> int:
    validate_cli_mode(args)

    suite = load_suite_manifest(args.manifest)
    runs = pd.read_csv(args.runs_csv)
    tasks = pd.read_csv(args.tasks_csv)

    if args.algorithms:
        runs = runs[runs["algorithm"].isin(args.algorithms)]
    if args.seeds:
        runs = runs[runs["seed"].isin(args.seeds)]
    if args.action_shield:
        return run_shield_evaluation(
            args, suite, runs, tasks, model_map=model_map,
            env_loader=env_loader, episode_runner=episode_runner,
        )
    if args.formal_unshielded_provenance:
        return run_formal_unshielded_evaluation(
            args, suite, runs, tasks, model_map=model_map,
            env_loader=env_loader, episode_runner=episode_runner,
        )
    tasks = filter_tasks(
        tasks,
        splits=args.splits,
        task_ids=args.task_ids,
        limit_tasks=args.limit_tasks,
    )

    out_path = Path(suite.result_root) / "eval_raw.csv"
    completed_keys = completed_eval_keys(out_path) if args.resume_eval else set()
    if out_path.exists() and not args.resume_eval:
        out_path.unlink()

    rows_written = 0
    task_records = [task_from_row(row) for row in tasks.itertuples(index=False)]

    for run in runs.itertuples(index=False):
        if run.algorithm == "rule_based":
            continue
        if run.status == "dry_run":
            continue
        if run.status != "completed":
            continue

        validate_completed_run_paths(run)
        model = model_map[run.algorithm].load(run.model_path, device="cpu")
        for task in task_records:
            output_algorithm = run.algorithm
            key = evaluation_key(run.algorithm, int(run.seed), task.task_id)
            if key in completed_keys:
                print(
                    f"Skipping completed eval: {run.algorithm} seed={run.seed} task={task.task_id}",
                    flush=True,
                )
                continue
            env = env_loader(suite, task, run.vecnormalize_path)
            primary_error: BaseException | None = None
            try:
                metrics = episode_runner(model, env)
            except BaseException as error:
                primary_error = error
                raise
            finally:
                close_environment(env, primary_error)
            row = EvaluationMetricRow(
                suite_id=suite.suite_id,
                algorithm=output_algorithm,
                seed=int(run.seed),
                run_name=run.run_name,
                task_id=task.task_id,
                split=task.split,
                weather_year=task.weather_year,
                start_day=task.start_day,
                uncertainty_scale=task.uncertainty_scale,
                economic_scenario=task.economic_scenario,
                climate_constraint_scenario=task.climate_constraint_scenario,
                trajectory_path="",
                **metrics,
            )
            append_eval_raw(row, out_path)
            completed_keys.add(key)
            rows_written += 1
            print(
                f"Evaluated {run.algorithm} seed={run.seed} task={task.task_id}",
                flush=True,
            )

    if rows_written == 0:
        print("No completed runs to evaluate; eval_raw.csv was not written.")
        return 0

    print(f"Wrote {rows_written} rows to {out_path}")
    return rows_written


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
