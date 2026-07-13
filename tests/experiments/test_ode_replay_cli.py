from __future__ import annotations

import argparse
import importlib.util
import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from gl_gym.experiments.ode_replay import ReplayOutcome, ReplayReport, VARIANT_NAMES


IDENTITY = "a" * 64
SCRIPT = Path(__file__).parents[2] / "experiments" / "scripts" / "replay_ode_failure.py"
SPEC = importlib.util.spec_from_file_location("replay_ode_failure_cli", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
cli = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(cli)


def _outcome(
    name: str, *, success: bool = False, available: bool = True
) -> ReplayOutcome:
    return ReplayOutcome(
        name,
        available,
        success,
        0.125,
        np.array([1.0, 2.0]) if success else None,
        None if success else "RuntimeError",
        None if success else "solver failed",
        ("diagnostic warning",),
    )


def _report(
    classification: str = "solver_step_sensitivity",
    *,
    successes: tuple[str, ...] | None = None,
    unavailable: tuple[str, ...] | None = None,
) -> ReplayReport:
    success_by_classification = {
        "policy_induced_control_instability": ("rule_based_control",),
        "mixed_control_and_solver_sensitivity": (
            "previous_control",
            "original_2x_substeps",
        ),
        "solver_step_sensitivity": ("original_2x_substeps",),
        "state_or_model_domain_failure": (),
        "non_reproduced": ("original",),
        "insufficient_counterfactual_evidence": (),
    }
    unavailable_by_classification = {
        "insufficient_counterfactual_evidence": ("rule_based_control",)
    }
    successes = (
        success_by_classification[classification] if successes is None else successes
    )
    unavailable = (
        unavailable_by_classification.get(classification, ())
        if unavailable is None
        else unavailable
    )
    return ReplayReport(
        "failure-123",
        classification,
        tuple(
            _outcome(
                name,
                success=name in successes,
                available=name not in unavailable,
            )
            for name in VARIANT_NAMES
        ),
    )


def _capsule(tmp_path: Path, formal_root: Path | None = None):
    capsule_path = tmp_path / "capsules" / "failure-123"
    capsule_path.mkdir(parents=True)
    formal = formal_root or tmp_path / "formal"
    return SimpleNamespace(
        path=capsule_path.resolve(),
        manifest={
            "failure_id": "failure-123",
            "content_identity_sha256": IDENTITY,
            "context": {"formal_result_root": str(formal)},
            "formal_result_root": str(formal),
        },
        failure_inputs={"x0": np.array([4.0, 5.0])},
    )


def test_parser_requires_capsule_and_output_root():
    parser = cli.build_parser()
    assert isinstance(parser, argparse.ArgumentParser)
    with pytest.raises(SystemExit):
        parser.parse_args([])
    args = parser.parse_args(["--capsule", "source", "--output_root", "target"])
    assert args.capsule == "source"
    assert args.output_root == "target"


def test_cli_loads_before_replay_and_writes_exactly_three_outputs(
    tmp_path, monkeypatch
):
    capsule = _capsule(tmp_path)
    calls: list[str] = []

    def load(path):
        calls.append(f"load:{path}")
        return capsule

    def replay(loaded):
        assert loaded is capsule
        calls.append("replay")
        return _report()

    monkeypatch.setattr(cli, "load_failure_capsule", load)
    output = tmp_path / "diagnostics" / "replay"
    result = cli.run_replay_cli(capsule.path, output, replay_loader=replay)

    assert result == output.resolve()
    assert calls == [f"load:{capsule.path}", "replay"]
    assert {item.name for item in result.iterdir()} == {
        "replay_results.json",
        "replay_states.npz",
        "replay_summary.md",
    }

    payload = json.loads((result / "replay_results.json").read_text(encoding="utf-8"))
    assert payload["schema_version"] == "ode-replay-report-v1"
    assert payload["failure_id"] == "failure-123"
    assert payload["capsule_identity_sha256"] == IDENTITY
    assert payload["classification"] == "solver_step_sensitivity"
    assert [item["variant"] for item in payload["outcomes"]] == list(VARIANT_NAMES)
    assert [item["configuration"]["substeps"] for item in payload["outcomes"]] == [
        1,
        1,
        1,
        2,
        4,
        1,
    ]
    assert payload["outcomes"][0]["configuration"] == {
        "control_source": "stored_executed_control",
        "substeps": 1,
        "tolerance_mode": "formal",
    }
    assert payload["outcomes"][5]["configuration"]["tolerance_mode"] == "strict_1e-6"
    for item in payload["outcomes"]:
        assert set(item) == {
            "variant",
            "configuration",
            "available",
            "success",
            "elapsed_seconds",
            "warnings",
            "exception_type",
            "exception_message",
        }
        assert math.isfinite(item["elapsed_seconds"])

    with np.load(result / "replay_states.npz", allow_pickle=False) as archive:
        assert set(archive.files) == {"variant_names", "final_states", "finite_masks"}
        assert archive["variant_names"].dtype.kind == "U"
        assert archive["variant_names"].tolist() == ["original_2x_substeps"]
        assert archive["final_states"].shape == (1, 2)
        assert archive["final_states"].dtype.kind == "f"
        assert archive["finite_masks"].shape == (1, 2)
        assert archive["finite_masks"].dtype == np.dtype(bool)


def test_empty_success_npz_preserves_capsule_state_dimension(tmp_path):
    output = tmp_path / "report"
    cli.write_replay_report_atomic(
        _report("state_or_model_domain_failure", successes=()),
        output,
        state_dim=2,
        capsule_identity=IDENTITY,
    )
    with np.load(output / "replay_states.npz", allow_pickle=False) as archive:
        assert archive["variant_names"].shape == (0,)
        assert archive["variant_names"].dtype.kind == "U"
        assert archive["final_states"].shape == (0, 2)
        assert archive["finite_masks"].shape == (0, 2)


def test_in_memory_report_rejects_approved_classification_inconsistent_with_outcomes(
    tmp_path,
):
    inconsistent = _report(
        "policy_induced_control_instability",
        successes=("original_2x_substeps",),
    )
    with pytest.raises(ValueError, match="classification.*outcomes"):
        cli.write_replay_report_atomic(
            inconsistent,
            tmp_path / "report",
            state_dim=2,
            capsule_identity=IDENTITY,
        )
    assert not (tmp_path / "report").exists()


@pytest.mark.parametrize(
    ("classification", "expected"),
    [
        (
            "policy_induced_control_instability",
            "principled action safety layer and return/constraint effects",
        ),
        (
            "mixed_control_and_solver_sensitivity",
            "principled action safety layer and return/constraint effects",
        ),
        ("solver_step_sensitivity", "integration-scheme benchmark redesign/version"),
        (
            "state_or_model_domain_failure",
            "physical bounds and implicated ODE/Jacobian terms",
        ),
        ("non_reproduced", "environment/software nondeterminism before causal change"),
        (
            "insufficient_counterfactual_evidence",
            "acquire missing counterfactual evidence",
        ),
    ],
)
def test_markdown_contains_table_classification_and_exact_action(
    tmp_path, classification, expected
):
    output = tmp_path / classification
    cli.write_replay_report_atomic(
        _report(classification), output, state_dim=2, capsule_identity=IDENTITY
    )
    summary = (output / "replay_summary.md").read_text(encoding="utf-8")
    assert "| Variant | Available | Success | Elapsed (s) | Exception |" in summary
    assert f"Classification: `{classification}`" in summary
    assert expected in summary


@pytest.mark.parametrize("relation", ["equal", "descendant", "ancestor"])
def test_output_must_be_disjoint_from_formal_root(tmp_path, relation):
    formal = tmp_path / "formal" / "results"
    outputs = {
        "equal": formal,
        "descendant": formal / "replay",
        "ancestor": formal.parent,
    }
    manifest = {"context": {"formal_result_root": str(formal)}}
    with pytest.raises(ValueError, match="formal result root"):
        cli.validate_isolated_output_root(outputs[relation], manifest)


def test_output_validation_reads_formal_root_from_manifest_context(tmp_path):
    formal = tmp_path / "formal"
    output = tmp_path / "diagnostics" / "replay"
    manifest = {"context": {"formal_result_root": str(formal)}}
    assert cli.validate_isolated_output_root(output, manifest) == output.resolve()


@pytest.mark.parametrize("relation", ["equal", "descendant", "ancestor"])
def test_cli_rejects_output_overlapping_source_capsule(tmp_path, monkeypatch, relation):
    capsule = _capsule(tmp_path)
    monkeypatch.setattr(cli, "load_failure_capsule", lambda _: capsule)
    outputs = {
        "equal": capsule.path,
        "descendant": capsule.path / "report",
        "ancestor": capsule.path.parent,
    }
    with pytest.raises(ValueError, match="source capsule"):
        cli.run_replay_cli(
            capsule.path, outputs[relation], replay_loader=lambda _: _report()
        )


@pytest.mark.parametrize("as_directory", [False, True])
def test_existing_final_output_is_never_overwritten(tmp_path, as_directory):
    output = tmp_path / "existing"
    output.mkdir() if as_directory else output.write_text("keep", encoding="utf-8")
    with pytest.raises(FileExistsError):
        cli.write_replay_report_atomic(
            _report(), output, state_dim=2, capsule_identity=IDENTITY
        )
    assert (
        output.is_dir()
        if as_directory
        else output.read_text(encoding="utf-8") == "keep"
    )


def test_injected_replay_failure_leaves_no_final_output(tmp_path, monkeypatch):
    capsule = _capsule(tmp_path)
    monkeypatch.setattr(cli, "load_failure_capsule", lambda _: capsule)
    output = tmp_path / "replay"
    with pytest.raises(RuntimeError, match="injected replay failure"):
        cli.run_replay_cli(
            capsule.path,
            output,
            replay_loader=lambda _: (_ for _ in ()).throw(
                RuntimeError("injected replay failure")
            ),
        )
    assert not output.exists()


def test_report_failure_id_must_match_loaded_capsule_before_writing(
    tmp_path, monkeypatch
):
    capsule = _capsule(tmp_path)
    monkeypatch.setattr(cli, "load_failure_capsule", lambda _: capsule)
    writer_called = False

    def writer(*_args, **_kwargs):
        nonlocal writer_called
        writer_called = True

    monkeypatch.setattr(cli, "write_replay_report_atomic", writer)
    mismatched = ReplayReport(
        "different-failure", _report().classification, _report().outcomes
    )
    output = tmp_path / "replay"
    with pytest.raises(ValueError, match="failure_id.*capsule"):
        cli.run_replay_cli(capsule.path, output, replay_loader=lambda _: mismatched)
    assert writer_called is False
    assert not output.exists()


def test_atomic_validation_failure_cleans_temp_and_preserves_primary_exception(
    tmp_path, monkeypatch
):
    output = tmp_path / "report"

    def fail_validation(_path, **_kwargs):
        raise ValueError("primary validation failure")

    monkeypatch.setattr(cli, "_validate_replay_directory", fail_validation)
    with pytest.raises(ValueError, match="primary validation failure"):
        cli.write_replay_report_atomic(
            _report(), output, state_dim=2, capsule_identity=IDENTITY
        )
    assert not output.exists()
    assert not list(tmp_path.glob(".ode-replay-*"))


def test_cleanup_error_does_not_mask_primary_exception(tmp_path, monkeypatch):
    monkeypatch.setattr(
        cli,
        "_validate_replay_directory",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(ValueError("primary")),
    )
    monkeypatch.setattr(
        cli.shutil,
        "rmtree",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("cleanup")),
    )
    with pytest.raises(ValueError, match="primary") as caught:
        cli.write_replay_report_atomic(
            _report(),
            tmp_path / "report",
            state_dim=2,
            capsule_identity=IDENTITY,
        )
    assert any("cleanup" in note for note in getattr(caught.value, "__notes__", ()))


def test_temp_name_is_short_and_independent_of_long_output_basename(
    tmp_path, monkeypatch
):
    output = tmp_path / ("r" * 180)
    observed = None

    def observe_and_fail(directory, **_kwargs):
        nonlocal observed
        observed = directory.name
        raise ValueError("stop after observing temp")

    monkeypatch.setattr(cli, "_validate_replay_directory", observe_and_fail)
    with pytest.raises(ValueError, match="observing temp"):
        cli.write_replay_report_atomic(
            _report(), output, state_dim=2, capsule_identity=IDENTITY
        )
    assert observed is not None
    assert observed.startswith(".ode-replay-")
    assert output.name not in observed
    assert len(observed) < 64


def test_publication_race_preserves_competitor_contents(tmp_path, monkeypatch):
    output = tmp_path / "report"

    def competitor(final):
        final.mkdir(exist_ok=True)
        (final / "competitor.txt").write_text("keep", encoding="utf-8")

    monkeypatch.setattr(cli, "_publication_race_hook", competitor)
    with pytest.raises(FileExistsError):
        cli.write_replay_report_atomic(
            _report(), output, state_dim=2, capsule_identity=IDENTITY
        )
    assert (output / "competitor.txt").read_text(encoding="utf-8") == "keep"
    assert not list(tmp_path.glob(".ode-replay-*"))


def test_strict_validation_rejects_extra_file_and_directory(tmp_path):
    output = tmp_path / "report"
    cli.write_replay_report_atomic(
        _report(), output, state_dim=2, capsule_identity=IDENTITY
    )
    (output / "extra.txt").write_text("unexpected", encoding="utf-8")
    with pytest.raises(ValueError, match="unexpected replay output entries"):
        cli._validate_replay_directory(output, state_dim=2)
    (output / "extra.txt").unlink()
    (output / "extra-dir").mkdir()
    with pytest.raises(ValueError, match="unexpected replay output entries"):
        cli._validate_replay_directory(output, state_dim=2)


def test_strict_validation_rejects_required_file_symlink(tmp_path):
    output = tmp_path / "report"
    cli.write_replay_report_atomic(
        _report(), output, state_dim=2, capsule_identity=IDENTITY
    )
    summary = output / "replay_summary.md"
    outside = tmp_path / "outside-summary.md"
    summary.replace(outside)
    try:
        summary.symlink_to(outside)
    except OSError:
        pytest.skip("symlinks are unavailable on this platform")
    with pytest.raises(ValueError, match="regular non-symlink"):
        cli._validate_replay_directory(output, state_dim=2)


def _mutate_top_extra(payload):
    payload["extra"] = "forbidden"


def _mutate_top_missing(payload):
    del payload["generated_at_utc"]


def _mutate_bad_timestamp(payload):
    payload["generated_at_utc"] = "2026-07-13T12:00:00"


def _mutate_nonzero_timestamp_offset(payload):
    payload["generated_at_utc"] = "2026-07-13T12:00:00+08:00"


def _mutate_empty_identity(payload):
    payload["capsule_identity_sha256"] = ""


def _mutate_short_identity(payload):
    payload["capsule_identity_sha256"] = "a" * 63


def _mutate_nonhex_identity(payload):
    payload["capsule_identity_sha256"] = "g" * 64


def _mutate_uppercase_identity(payload):
    payload["capsule_identity_sha256"] = "A" * 64


def _mutate_empty_failure(payload):
    payload["failure_id"] = ""


def _mutate_bad_classification(payload):
    payload["classification"] = "unsupported"


def _mutate_inconsistent_approved_classification(payload):
    payload["classification"] = "policy_induced_control_instability"


def _mutate_outcome_extra(payload):
    payload["outcomes"][0]["extra"] = 1


def _mutate_outcome_missing(payload):
    del payload["outcomes"][0]["warnings"]


def _mutate_available_int(payload):
    payload["outcomes"][0]["available"] = 1


def _mutate_elapsed_int(payload):
    payload["outcomes"][0]["elapsed_seconds"] = 1


def _mutate_warnings_string(payload):
    payload["outcomes"][0]["warnings"] = "not-a-list"


def _mutate_success_exception(payload):
    payload["outcomes"][3]["exception_type"] = "RuntimeError"


def _mutate_unavailable_success(payload):
    payload["outcomes"][3]["available"] = False


def _mutate_failure_without_exception(payload):
    payload["outcomes"][0]["exception_type"] = None
    payload["outcomes"][0]["exception_message"] = None


@pytest.mark.parametrize(
    "mutate",
    [
        _mutate_top_extra,
        _mutate_top_missing,
        _mutate_bad_timestamp,
        _mutate_nonzero_timestamp_offset,
        _mutate_empty_identity,
        _mutate_short_identity,
        _mutate_nonhex_identity,
        _mutate_uppercase_identity,
        _mutate_empty_failure,
        _mutate_bad_classification,
        _mutate_inconsistent_approved_classification,
        _mutate_outcome_extra,
        _mutate_outcome_missing,
        _mutate_available_int,
        _mutate_elapsed_int,
        _mutate_warnings_string,
        _mutate_success_exception,
        _mutate_unavailable_success,
        _mutate_failure_without_exception,
    ],
)
def test_strict_json_validation_rejects_malformed_fields(tmp_path, mutate):
    output = tmp_path / "report"
    cli.write_replay_report_atomic(
        _report(), output, state_dim=2, capsule_identity=IDENTITY
    )
    json_path = output / "replay_results.json"
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    mutate(payload)
    json_path.write_text(json.dumps(payload, allow_nan=False), encoding="utf-8")
    with pytest.raises(ValueError):
        cli._validate_replay_directory(output, state_dim=2)


@pytest.mark.parametrize(
    "identity",
    ["", "a" * 63, "g" * 64, "A" * 64],
)
def test_writer_rejects_noncanonical_capsule_identity(tmp_path, identity):
    with pytest.raises(ValueError, match="capsule identity"):
        cli.write_replay_report_atomic(
            _report(), tmp_path / "report", state_dim=2, capsule_identity=identity
        )
