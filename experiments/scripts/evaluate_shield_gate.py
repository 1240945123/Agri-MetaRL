#!/usr/bin/env python3
"""Gate the complete paired Stage-3 action-shield suite."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
import uuid
from pathlib import Path
from typing import Any

import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd
import numpy as np

from gl_gym.experiments.shield_evaluation import (
    aggregate_episode_interventions,
    build_paired_shield_deltas,
    evaluate_shield_gate,
)
from gl_gym.experiments.suite_schema import load_suite_manifest
from gl_gym.experiments.suite_tasks import build_evaluation_tasks
from experiments.scripts.evaluate_suite import (
    load_stage2_evidence,
    replace_directory_atomic,
)


BASE_ALGORITHM = "agri_metarl"
SHIELD_ALGORITHM = "agri_metarl__minimal_feasibility_shield_v1"
SHIELD_METHOD = "minimal_feasibility_shield_v1"
STAGE3 = "stage3_full_suite_action_shield"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "manifest", "tasks_csv", "unshielded_eval", "shielded_eval",
        "interventions", "stage2_decision", "output_root",
    ):
        parser.add_argument(f"--{name}", required=True)
    return parser


def _sha(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _strict_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"), parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)))
    if not isinstance(value, dict):
        raise ValueError(f"strict JSON mapping required: {path}")
    return value


def validate_canonical_tasks(suite: Any, tasks: pd.DataFrame) -> pd.DataFrame:
    expected = pd.DataFrame(vars(item) for item in build_evaluation_tasks(suite))
    columns = list(expected.columns)
    if list(tasks.columns) != columns or len(tasks) != 91:
        raise ValueError("tasks CSV must have the exact canonical 91-row schema")
    if tasks["task_id"].duplicated().any():
        raise ValueError("canonical tasks contain duplicate task IDs")
    try:
        pd.testing.assert_frame_equal(
            tasks.reset_index(drop=True), expected.reset_index(drop=True),
            check_dtype=False, check_exact=True,
        )
    except AssertionError as exc:
        raise ValueError("tasks CSV does not match the canonical suite task descriptors") from exc
    return expected


def _collides(a: Path, b: Path) -> bool:
    if a == b or a in b.parents or b in a.parents:
        return True
    for candidate, protected in ((a, b), (b, a)):
        for component in (candidate, *candidate.parents):
            if component.parent == protected.parent and (
                component.name == f".{protected.name}.work"
                or component.name.startswith((f".{protected.name}.stage-", f".{protected.name}.backup-"))
            ):
                return True
    return False


def _prerequisites(args: argparse.Namespace) -> tuple[Any, pd.DataFrame, dict[str, Any], dict[str, Any], Path]:
    inputs = [Path(getattr(args, name)).resolve() for name in (
        "manifest", "tasks_csv", "unshielded_eval", "shielded_eval", "interventions", "stage2_decision"
    )]
    if any(not path.is_file() or path.is_symlink() for path in inputs):
        raise FileNotFoundError("all Stage-3 inputs must be regular files")
    output = Path(args.output_root).resolve()
    stage2_root = inputs[-1].parent
    suite = load_suite_manifest(args.manifest)
    tasks = pd.read_csv(args.tasks_csv)
    validate_canonical_tasks(suite, tasks)
    authenticated_stage2 = load_stage2_evidence(inputs[-1])
    decision = authenticated_stage2["decision"]
    stage2 = authenticated_stage2["manifest"]
    stage2["stage2_identity_sha256"] = authenticated_stage2["stage2_identity_sha256"]
    protected = {
        Path(suite.result_root).resolve(), stage2_root,
        Path(args.unshielded_eval).resolve().parent,
        Path(args.shielded_eval).resolve().parent,
        Path(args.interventions).resolve().parent,
    }
    for name in ("stage1_root", "unshielded_result_root", "failure_root", "result_root"):
        if stage2.get(name): protected.add(Path(stage2[name]).resolve())
    if any(_collides(output, item) for item in protected):
        raise ValueError("Stage-3 output root must be disjoint from suite/evaluation/prerequisite roots")
    if output.exists() and not output.is_dir():
        raise ValueError("output_root exists as a file")
    expected_conditions = {
        "zero_ode_failures", "intervention_rate_within_0p5pct",
        "paired_return_loss_within_2pct", "paired_violation_burden_within_5pct",
    }
    if (
        set(decision) != {"outcome", "stage", "conditions", "evidence", "reasons"}
        or decision["outcome"] != "continue_to_full_suite"
        or decision["stage"] != "stage2_shielded_context_ab"
        or set(decision["conditions"]) != expected_conditions
        or any(value is not True for value in decision["conditions"].values())
        or decision["reasons"] != []
    ):
        raise ValueError("Stage-2 prerequisite is failing or has invalid schema")
    from experiments.scripts import run_shielded_context_ab as source
    expected_provenance = {
        "method": SHIELD_METHOD,
        "source_manifest_sha256": _sha(args.manifest),
        "source_tasks_sha256": _sha(args.tasks_csv),
        "env_config_sha256": _sha(source.ENV_CONFIG_PATH),
        "runtime_source_tree_sha256": source._runtime_source_tree_sha256(),
        **source._behavior_source_hashes(),
    }
    _, rule_hash = source._load_rule_params()
    expected_provenance["rule_config_sha256"] = rule_hash
    for key, value in expected_provenance.items():
        if stage2.get(key) != value:
            raise ValueError(f"Stage-2 provenance is stale: {key}")
    if stage2.get("fixed_lambdas") != list(source.DEFAULT_LAMBDAS) or stage2.get("formal_solver_options") != dict(source.FORMAL_CVODES_OPTIONS):
        raise ValueError("Stage-2 method constants are stale")
    return suite, tasks, stage2, decision, output


def _strict_keys(frame: pd.DataFrame, *, label: str, expected: set[tuple[int, str]]) -> pd.DataFrame:
    required = {"suite_id", "algorithm", "seed", "task_id", "split", "weather_year", "start_day", "uncertainty_scale", "economic_scenario", "climate_constraint_scenario"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{label} is missing required provenance/descriptors: {missing}")
    if frame.duplicated(["seed", "task_id"], keep=False).any():
        raise ValueError(f"{label} contains duplicate keys")
    actual: set[tuple[int, str]] = set()
    for seed, task in frame[["seed", "task_id"]].itertuples(index=False, name=None):
        if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
            raise ValueError(f"{label} seed must be a strict integer")
        if not isinstance(task, str) or not task:
            raise ValueError(f"{label} task_id must be nonempty text")
        actual.add((int(seed), task))
    if actual != expected:
        raise ValueError(f"{label} keys must exactly match the complete 182-key design")
    return frame.copy()


def _validate_descriptors(frame: pd.DataFrame, tasks: pd.DataFrame, label: str) -> None:
    descriptor_columns = ["task_id", "split", "weather_year", "start_day", "uncertainty_scale", "economic_scenario", "climate_constraint_scenario"]
    joined = frame[descriptor_columns].drop_duplicates().merge(tasks[descriptor_columns], on="task_id", suffixes=("", "_canonical"), validate="one_to_one")
    if len(joined) != 91 or any(not joined[name].equals(joined[f"{name}_canonical"]) for name in descriptor_columns[1:]):
        raise ValueError(f"{label} task descriptors do not match tasks CSV")


def _evidence_path(value: Any, table_path: str | Path) -> Path:
    path = Path(str(value))
    return path.resolve() if path.is_absolute() else (Path(table_path).resolve().parent / path).resolve()


def _load_evaluation_manifest(eval_path: str | Path, *, label: str) -> dict[str, Any]:
    path = Path(eval_path).resolve()
    manifest_path = path.parent / "evaluation_manifest.json"
    if not manifest_path.is_file() or manifest_path.is_symlink():
        raise ValueError(f"{label} requires a new provenance-bearing sibling evaluation_manifest.json")
    manifest = _strict_json(manifest_path)
    if Path(str(manifest.get("result_root", ""))).resolve() != path.parent:
        raise ValueError(f"{label} evaluation manifest result_root is invalid")
    if manifest.get("eval_raw_sha256") != _sha(path):
        raise ValueError(f"{label} eval_raw.csv hash does not match its evaluation manifest")
    if (
        not isinstance(manifest.get("checkpoints"), list)
        or not manifest.get("source_fingerprint_sha256")
        or not manifest.get("runtime_source_tree_sha256")
    ):
        raise ValueError(f"{label} evaluation manifest checkpoint/source provenance is incomplete")
    from experiments.scripts.run_shielded_context_ab import _runtime_source_tree_sha256
    if manifest["runtime_source_tree_sha256"] != _runtime_source_tree_sha256():
        raise ValueError(f"{label} runtime source provenance is stale")
    return manifest


def _validate_metric_provenance(frame: pd.DataFrame, manifest: dict[str, Any], *, label: str) -> None:
    required = {"model_sha256", "vecnormalize_sha256", "checkpoint_steps", "source_fingerprint_sha256"}
    if not required.issubset(frame.columns):
        raise ValueError(f"{label} rows are missing checkpoint/source provenance")
    checkpoints = {int(row["seed"]): row for row in manifest["checkpoints"]}
    for row in frame.itertuples(index=False):
        expected = checkpoints.get(int(row.seed))
        if expected is None or row.model_sha256 != expected["model_sha256"] or row.vecnormalize_sha256 != expected["vecnormalize_sha256"]:
            raise ValueError(f"{label} row checkpoint provenance is invalid")
        steps = row.checkpoint_steps
        if isinstance(steps, (bool, np.bool_)) or not isinstance(steps, (int, np.integer)) or int(steps) < 0:
            raise ValueError(f"{label} checkpoint_steps must be a strict nonnegative integer")
        if "checkpoint_steps" in expected and int(steps) != expected["checkpoint_steps"]:
            raise ValueError(f"{label} checkpoint_steps mismatch")
        if row.source_fingerprint_sha256 != manifest["source_fingerprint_sha256"]:
            raise ValueError(f"{label} source fingerprint mismatch")


def _validate_intervention_evidence(
    evidence: pd.DataFrame, *, table_path: str | Path, stage2: dict[str, Any],
    stage2_root: Path, evaluation_manifest: dict[str, Any], shield_metrics: pd.DataFrame,
) -> None:
    required = {
        "formal_complete", "model_sha256", "vecnormalize_sha256", "stage2_identity_sha256",
        "executed_action_trace_path", "requested_action_trace_path", "intervention_records_path",
        "intervention_rate", "total_steps", "intervention_count", "ode_failure_count",
        "executed_action_trace_sha256", "requested_action_trace_sha256",
        "intervention_records_sha256", "episode_evidence_identity_sha256",
    }
    if not required.issubset(evidence.columns):
        raise ValueError(f"intervention evidence schema is incomplete: {sorted(required - set(evidence.columns))}")
    if not evidence["formal_complete"].map(lambda value: value is True or value == 1).all():
        raise ValueError("smoke/nonformal shield evaluations cannot pass the Stage-3 gate")
    expected_stage2 = stage2["stage2_identity_sha256"]
    checkpoints = {int(item["seed"]): (item["model_sha256"], item["vecnormalize_sha256"]) for item in stage2["checkpoints"]}
    referenced_evidence: set[str] = set()
    for row in evidence.itertuples(index=False):
        if row.stage2_identity_sha256 != expected_stage2 or (row.model_sha256, row.vecnormalize_sha256) != checkpoints[int(row.seed)]:
            raise ValueError("intervention checkpoint/Stage-2 provenance is stale")
        executed_path = _evidence_path(row.executed_action_trace_path, table_path)
        requested_path = _evidence_path(row.requested_action_trace_path, table_path)
        records_path = _evidence_path(row.intervention_records_path, table_path)
        if any(not path.is_file() or path.is_symlink() for path in (executed_path, requested_path, records_path)):
            raise ValueError("intervention trace evidence is missing")
        for path, hash_name in (
            (executed_path, "executed_action_trace_sha256"),
            (requested_path, "requested_action_trace_sha256"),
            (records_path, "intervention_records_sha256"),
        ):
            observed_hash = _sha(path)
            if getattr(row, hash_name) != observed_hash:
                raise ValueError("intervention evidence file hash is stale")
            relative = path.relative_to(Path(table_path).resolve().parent).as_posix()
            referenced_evidence.add(relative)
            if evaluation_manifest.get("evidence_sha256", {}).get(relative) != observed_hash:
                raise ValueError("intervention evidence tree is not bound by evaluation manifest")
        executed = np.load(executed_path, allow_pickle=False)
        requested = np.load(requested_path, allow_pickle=False)
        records = json.loads(records_path.read_text(encoding="utf-8"))
        if executed.ndim != 2 or requested.shape != executed.shape or len(records) != len(executed):
            raise ValueError("intervention trace evidence is inconsistent")
        summary = aggregate_episode_interventions(records, executed.shape[1])
        for name, value in summary.items():
            if name not in evidence.columns:
                raise ValueError(f"intervention evidence is missing {name}")
            observed = getattr(row, name)
            if isinstance(value, list):
                try: observed = json.loads(observed) if isinstance(observed, str) else list(observed)
                except (TypeError, ValueError, json.JSONDecodeError) as exc: raise ValueError(f"invalid {name}") from exc
            if value is None and pd.isna(observed):
                continue
            if observed != value and not (
                isinstance(value, float) and np.isclose(float(observed), value, rtol=1e-12, atol=1e-12)
            ):
                raise ValueError(f"intervention evidence summary mismatch: {name}")
        metric_match = shield_metrics.loc[
            (shield_metrics["seed"] == row.seed) & (shield_metrics["task_id"] == row.task_id)
        ]
        if len(metric_match) != 1: raise ValueError("shield metric/evidence rows are not one-to-one")
        metric = metric_match.iloc[0]
        identity_payload = {
            name: getattr(row, name) for name in (
                "algorithm", "method", "model_sha256", "vecnormalize_sha256", "checkpoint_steps",
                "source_fingerprint_sha256", "stage2_identity_sha256", "suite_id", "seed", "task_id",
                "split", "weather_year", "start_day", "uncertainty_scale", "economic_scenario",
                "climate_constraint_scenario", "executed_action_trace_sha256",
                "requested_action_trace_sha256", "intervention_records_sha256",
            )
        }
        identity_payload.update({name: metric[name] for name in ("episode_return", "temp_violation", "co2_violation", "rh_violation")})
        from experiments.scripts.evaluate_suite import _canonical_hash
        if row.episode_evidence_identity_sha256 != _canonical_hash(identity_payload) or metric["episode_evidence_identity_sha256"] != row.episode_evidence_identity_sha256:
            raise ValueError("episode evidence identity is invalid")
    if set(evaluation_manifest.get("evidence_sha256", {})) != referenced_evidence:
        raise ValueError("evaluation manifest evidence tree has missing or extra identities")


def _publish(output: Path, paired: pd.DataFrame, summary: pd.DataFrame, manifest: dict[str, Any], decision: dict[str, Any]) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{output.name}.stage-", dir=output.parent))
    try:
        paired.to_csv(stage / "paired_deltas.csv", index=False)
        summary.to_csv(stage / "summary.csv", index=False)
        for name, payload in (("shield_manifest.json", manifest), ("decision.json", decision)):
            with (stage / name).open("w", encoding="utf-8", newline="\n") as handle:
                json.dump(payload, handle, allow_nan=False, sort_keys=True, indent=2); handle.write("\n")
        replace_directory_atomic(stage, output)
    finally:
        if stage.exists(): shutil.rmtree(stage, ignore_errors=True)


def run(args: argparse.Namespace) -> dict[str, Any]:
    suite, tasks, stage2, stage2_decision, output = _prerequisites(args)
    checkpoints = stage2.get("checkpoints")
    if not isinstance(checkpoints, list):
        raise ValueError("Stage-2 checkpoints are missing")
    seeds = {int(item["seed"]) for item in checkpoints}
    if seeds != {42, 123} or len(checkpoints) != 2:
        raise ValueError("Stage-3 requires the exact approved seeds 42 and 123")
    expected = {(seed, task_id) for seed in seeds for task_id in tasks["task_id"]}
    shield_manifest = _load_evaluation_manifest(args.shielded_eval, label="shielded")
    unshield_manifest = _load_evaluation_manifest(args.unshielded_eval, label="unshielded")
    for label, manifest in (("shielded", shield_manifest), ("unshielded", unshield_manifest)):
        manifest_seeds = [int(item["seed"]) for item in manifest["checkpoints"]]
        if len(manifest_seeds) != len(seeds) or set(manifest_seeds) != seeds:
            raise ValueError(f"{label} evaluation manifest has wrong approved checkpoints")
    if shield_manifest.get("formal_complete") is not True:
        raise ValueError("shielded evaluation manifest is not a complete formal protocol")
    if shield_manifest.get("interventions_sha256") != _sha(args.interventions):
        raise ValueError("shielded interventions hash does not match evaluation manifest")
    if shield_manifest.get("stage2_identity_sha256") != stage2["stage2_identity_sha256"]:
        raise ValueError("shielded evaluation Stage-2 identity is stale")
    shield = _strict_keys(pd.read_csv(args.shielded_eval), label="shielded", expected=expected)
    unshield = _strict_keys(pd.read_csv(args.unshielded_eval), label="unshielded", expected=expected)
    _validate_metric_provenance(shield, shield_manifest, label="shielded")
    _validate_metric_provenance(unshield, unshield_manifest, label="unshielded")
    if set(shield["algorithm"]) != {SHIELD_ALGORITHM} or set(unshield["algorithm"]) != {BASE_ALGORITHM}:
        raise ValueError("shielded/unshielded algorithm identifiers are invalid")
    if "method" in shield and set(shield["method"]) != {SHIELD_METHOD}:
        raise ValueError("shielded method identifier is invalid")
    if set(shield["suite_id"]) != {suite.suite_id} or set(unshield["suite_id"]) != {suite.suite_id}:
        raise ValueError("suite provenance is invalid")
    _validate_descriptors(shield, tasks, "shielded"); _validate_descriptors(unshield, tasks, "unshielded")

    interventions = _strict_keys(pd.read_csv(args.interventions), label="interventions", expected=expected)
    _validate_metric_provenance(interventions, shield_manifest, label="interventions")
    if "method" not in interventions or set(interventions["method"]) != {SHIELD_METHOD}:
        raise ValueError("intervention method provenance is invalid")
    if set(interventions["algorithm"]) != {SHIELD_ALGORITHM}:
        raise ValueError("intervention algorithm provenance is invalid")
    _validate_descriptors(interventions, tasks, "interventions")
    _validate_intervention_evidence(
        interventions, table_path=args.interventions, stage2=stage2,
        stage2_root=Path(args.stage2_decision).resolve().parent,
        evaluation_manifest=shield_manifest, shield_metrics=shield,
    )
    summary_fields = {"completed", "ode_failure_count", "total_steps", "intervention_count"}
    if not summary_fields.issubset(interventions.columns):
        raise ValueError("intervention evidence schema is incomplete")
    shield = shield.drop(columns=[name for name in summary_fields if name in shield], errors="ignore").merge(
        interventions[["seed", "task_id", *sorted(summary_fields)]], on=["seed", "task_id"], validate="one_to_one"
    )
    if "completed" not in unshield: unshield["completed"] = True
    if "ode_failure_count" not in unshield: unshield["ode_failure_count"] = 0
    gate = evaluate_shield_gate(shield, unshield, expected, key_columns=("seed", "task_id"))
    paired = build_paired_shield_deltas(shield, unshield, expected, key_columns=("seed", "task_id"))
    decision = {**gate, "stage": STAGE3, "outcome": "paper_evidence_ready" if gate["outcome"] == "pass" else "redesign_before_claim"}
    summary = pd.DataFrame([{**gate["evidence"], **{f"condition_{k}": v for k, v in gate["conditions"].items()}, "outcome": decision["outcome"]}])
    input_hashes = {name: _sha(getattr(args, name)) for name in ("manifest", "tasks_csv", "unshielded_eval", "shielded_eval", "interventions", "stage2_decision")}
    from experiments.scripts.run_shielded_context_ab import _runtime_source_tree_sha256
    manifest = {
        "schema_version": "full-suite-action-shield-stage3-v1", "stage": STAGE3,
        "suite_id": suite.suite_id, "approved_seeds": sorted(seeds), "task_count": 91,
        "expected_pair_count": 182, "method": SHIELD_METHOD,
        "input_sha256": input_hashes,
        "stage2_identity_sha256": stage2["stage2_identity_sha256"],
        "runtime_source_tree_sha256": _runtime_source_tree_sha256(),
        "gate_source_sha256": _sha(Path(__file__)),
        "suite_evaluator_source_sha256": _sha(ROOT / "experiments/scripts/evaluate_suite.py"),
    }
    _publish(output, paired, summary, manifest, decision)
    return decision


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
