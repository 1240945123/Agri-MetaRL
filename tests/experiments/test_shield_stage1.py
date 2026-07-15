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
        "conditions": conditions,
        "outcome": "continue_to_context_ab",
    }
    report["shield_fingerprint"] = cli._shield_fingerprint(report)
    return report


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
