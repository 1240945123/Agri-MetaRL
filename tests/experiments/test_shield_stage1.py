import json
from pathlib import Path

import numpy as np
import pytest

from experiments.scripts import run_shield_stage1 as cli


def _report() -> dict:
    conditions = {name: True for name in cli.CONDITION_NAMES}
    report = {
        "schema_version": cli.SCHEMA_VERSION,
        "method": cli.METHOD,
        "fixed_lambdas": list(cli.DEFAULT_LAMBDAS),
        "source_checksums": {"source": "b" * 64},
        "formal_solver_options": {"solver": "fixed"},
        "env_config_sha256": "c" * 64,
        "rule_config_sha256": "d" * 64,
        "capsule_identity_sha256": "e" * 64,
        "checkpoint_sha256": "f" * 64,
        "selected_lambda": cli.DEFAULT_LAMBDAS[1],
        "candidate_attempts": [
            {
                "lambda": cli.DEFAULT_LAMBDAS[0],
                "action": [0.0, 0.0],
                "control": [0.0, 0.0],
                "success": False,
                "exception_type": "RuntimeError",
                "exception_message": "failed",
                "elapsed_seconds": 0.01,
            },
            {
                "lambda": cli.DEFAULT_LAMBDAS[1],
                "action": [0.0, 0.0],
                "control": [0.0, 0.0],
                "success": True,
                "exception_type": None,
                "exception_message": None,
                "elapsed_seconds": 0.01,
            },
        ],
        "conditions": conditions,
        "outcome": "continue_to_context_ab",
    }
    report["shield_fingerprint"] = cli._shield_fingerprint(report)
    return report


def _forge_success_before_last(report: dict) -> None:
    report["candidate_attempts"][0].update(
        success=True,
        exception_type=None,
        exception_message=None,
    )


def _forge_all_failed(report: dict) -> None:
    report["candidate_attempts"][-1].update(
        success=False,
        exception_type="RuntimeError",
        exception_message="failed",
    )


def test_stage1_v2_decision_binds_schema_method_grid_and_fingerprint(tmp_path: Path):
    report = _report()
    x0 = np.ones(2, dtype=np.float64)
    cli._validate_report(report, x0, x0.copy())
    cli._write_outputs(tmp_path, report, x0, x0.copy())

    decision = json.loads((tmp_path / "decision.json").read_text(encoding="utf-8"))
    assert decision == {
        key: report[key]
        for key in (
            "schema_version", "method", "fixed_lambdas", "shield_fingerprint",
            "outcome", "conditions", "selected_lambda",
        )
    }
    assert tuple(decision["fixed_lambdas"]) == tuple(
        sorted(decision["fixed_lambdas"], reverse=True)
    )


def test_stage1_condition_names_first_success_in_descending_priority():
    assert "first_successful_candidate_selected" in cli.CONDITION_NAMES
    assert all("smallest" not in name for name in cli.CONDITION_NAMES)


def test_stage1_report_rejects_v1_schema():
    report = _report()
    report["schema_version"] = "action-shield-stage1-v1"
    with pytest.raises(ValueError, match="schema"):
        cli._validate_report(report, np.ones(2), np.ones(2))


def test_stage1_report_rejects_source_provenance_tampering():
    report = _report()
    report["source_checksums"]["source"] = "0" * 64
    with pytest.raises(ValueError, match="fingerprint"):
        cli._validate_report(report, np.ones(2), np.ones(2))


@pytest.mark.parametrize(
    "mutate",
    [
        lambda report: report.pop("candidate_attempts"),
        lambda report: report.__setitem__("candidate_attempts", report["candidate_attempts"][1:]),
        _forge_success_before_last,
        lambda report: report.__setitem__("selected_lambda", cli.DEFAULT_LAMBDAS[2]),
        _forge_all_failed,
        lambda report: report.__setitem__("candidate_attempts", []),
    ],
    ids=("missing", "nonprefix", "success_before_last", "selected_mismatch", "all_failed", "empty"),
)
def test_stage1_passing_report_rejects_forged_candidate_attempts(mutate):
    report = _report()
    mutate(report)
    with pytest.raises(ValueError, match="candidate|lambda|success"):
        cli._validate_report(report, np.ones(2), np.ones(2))


@pytest.mark.parametrize("value", ["1.0", True, np.nan, 10**309])
def test_stage1_passing_report_rejects_invalid_candidate_lambda(value):
    report = _report()
    report["candidate_attempts"][0]["lambda"] = value
    with pytest.raises(ValueError, match="lambda"):
        cli._validate_report(report, np.ones(2), np.ones(2))


@pytest.mark.parametrize("attempts", [[], "all_failed"])
def test_stage1_rejects_unsuccessful_candidate_evidence_even_with_failing_conditions(
    attempts,
):
    report = _report()
    report["conditions"]["legal_candidate_succeeded"] = False
    report["conditions"]["first_successful_candidate_selected"] = False
    report["selected_lambda"] = None
    report["outcome"] = "redesign_action_shield"
    if attempts == []:
        report["candidate_attempts"] = []
    else:
        _forge_all_failed(report)

    with pytest.raises(ValueError, match="candidate_attempts.*successful|final.*succeed"):
        cli._validate_report(report, np.ones(2), None)
