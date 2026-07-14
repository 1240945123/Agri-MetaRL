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
    decision = _strict_json(inputs[-1])
    manifest_path = stage2_root / "shield_manifest.json"
    if not manifest_path.is_file() or manifest_path.is_symlink():
        raise FileNotFoundError("Stage-2 shield_manifest.json is required")
    stage2 = _strict_json(manifest_path)
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
    actual = {(int(seed), str(task)) for seed, task in frame[["seed", "task_id"]].itertuples(index=False, name=None)}
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


def _validate_intervention_evidence(
    evidence: pd.DataFrame, *, table_path: str | Path, stage2: dict[str, Any], stage2_root: Path
) -> None:
    required = {
        "formal_complete", "model_sha256", "vecnormalize_sha256", "stage2_manifest_sha256",
        "executed_action_trace_path", "requested_action_trace_path", "intervention_records_path",
        "intervention_rate", "total_steps", "intervention_count", "ode_failure_count",
    }
    if not required.issubset(evidence.columns):
        raise ValueError(f"intervention evidence schema is incomplete: {sorted(required - set(evidence.columns))}")
    if not evidence["formal_complete"].map(lambda value: value is True or value == 1).all():
        raise ValueError("smoke/nonformal shield evaluations cannot pass the Stage-3 gate")
    expected_stage2 = _sha(stage2_root / "shield_manifest.json")
    checkpoints = {int(item["seed"]): (item["model_sha256"], item["vecnormalize_sha256"]) for item in stage2["checkpoints"]}
    for row in evidence.itertuples(index=False):
        if row.stage2_manifest_sha256 != expected_stage2 or (row.model_sha256, row.vecnormalize_sha256) != checkpoints[int(row.seed)]:
            raise ValueError("intervention checkpoint/Stage-2 provenance is stale")
        executed_path = _evidence_path(row.executed_action_trace_path, table_path)
        requested_path = _evidence_path(row.requested_action_trace_path, table_path)
        records_path = _evidence_path(row.intervention_records_path, table_path)
        if any(not path.is_file() or path.is_symlink() for path in (executed_path, requested_path, records_path)):
            raise ValueError("intervention trace evidence is missing")
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
            if observed != value and not (
                isinstance(value, float) and np.isclose(float(observed), value, rtol=1e-12, atol=1e-12)
            ):
                raise ValueError(f"intervention evidence summary mismatch: {name}")


def _publish(output: Path, paired: pd.DataFrame, summary: pd.DataFrame, manifest: dict[str, Any], decision: dict[str, Any]) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{output.name}.stage-", dir=output.parent))
    backup = output.parent / f".{output.name}.backup-{uuid.uuid4().hex}"
    moved = False
    try:
        paired.to_csv(stage / "paired_deltas.csv", index=False)
        summary.to_csv(stage / "summary.csv", index=False)
        for name, payload in (("shield_manifest.json", manifest), ("decision.json", decision)):
            with (stage / name).open("w", encoding="utf-8", newline="\n") as handle:
                json.dump(payload, handle, allow_nan=False, sort_keys=True, indent=2); handle.write("\n")
        if output.exists():
            if backup.exists(): shutil.rmtree(backup)
            os.replace(output, backup); moved = True
        try:
            os.replace(stage, output)
        except BaseException:
            if moved: os.replace(backup, output); moved = False
            raise
        if moved: shutil.rmtree(backup, ignore_errors=True)
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
    shield = _strict_keys(pd.read_csv(args.shielded_eval), label="shielded", expected=expected)
    unshield = _strict_keys(pd.read_csv(args.unshielded_eval), label="unshielded", expected=expected)
    if set(shield["algorithm"]) != {SHIELD_ALGORITHM} or set(unshield["algorithm"]) != {BASE_ALGORITHM}:
        raise ValueError("shielded/unshielded algorithm identifiers are invalid")
    if "method" in shield and set(shield["method"]) != {SHIELD_METHOD}:
        raise ValueError("shielded method identifier is invalid")
    if set(shield["suite_id"]) != {suite.suite_id} or set(unshield["suite_id"]) != {suite.suite_id}:
        raise ValueError("suite provenance is invalid")
    _validate_descriptors(shield, tasks, "shielded"); _validate_descriptors(unshield, tasks, "unshielded")

    interventions = _strict_keys(pd.read_csv(args.interventions), label="interventions", expected=expected)
    if "method" not in interventions or set(interventions["method"]) != {SHIELD_METHOD}:
        raise ValueError("intervention method provenance is invalid")
    if set(interventions["algorithm"]) != {SHIELD_ALGORITHM}:
        raise ValueError("intervention algorithm provenance is invalid")
    _validate_descriptors(interventions, tasks, "interventions")
    _validate_intervention_evidence(
        interventions, table_path=args.interventions, stage2=stage2,
        stage2_root=Path(args.stage2_decision).resolve().parent,
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
        "input_sha256": input_hashes, "stage2_manifest_sha256": _sha(Path(args.stage2_decision).resolve().parent / "shield_manifest.json"),
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
