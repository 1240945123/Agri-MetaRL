import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from gl_gym.experiments import shield_evaluation as shield_module
from gl_gym.experiments.shield_evaluation import (
    aggregate_episode_interventions,
    build_paired_shield_deltas,
    evaluate_shield_gate,
    write_shield_artifacts_atomic,
)


def _record(step=0, *, intervened=True, selected_lambda=0.125):
    requested = np.array([0.5, -0.5])
    executed = np.array([0.45, -0.45]) if intervened else requested.copy()
    difference = requested - executed
    attempts = []
    if intervened:
        grid = (0.0625, 0.125)
        reference = np.array([0.1, -0.1])
        attempts = [
            {
                "lambda": value,
                "action": ((1.0 - value) * requested + value * reference).tolist(),
                "success": value == selected_lambda,
                "elapsed_seconds": 0.01,
                "exception_type": None if value == selected_lambda else "RuntimeError",
                "exception_message": None if value == selected_lambda else "failed",
            }
            for value in grid[: grid.index(selected_lambda) + 1]
        ]
    return {
        "step_index": step,
        "schema_version": "minimal-feasibility-action-shield-v1",
        "intervened": intervened,
        "requested_action": requested.tolist(),
        "reference_action": reference.tolist() if intervened else None,
        "executed_action": executed.tolist(),
        "selected_lambda": selected_lambda if intervened else 0.0,
        "candidate_attempts": attempts,
        "intervention_l1": float(np.linalg.norm(difference, 1)),
        "intervention_l2": float(np.linalg.norm(difference, 2)),
        "intervention_linf": float(np.linalg.norm(difference, np.inf)),
        "per_channel_changed": (difference != 0).tolist(),
        "extra_solver_attempts": len(attempts),
        "elapsed_seconds": 0.25,
        "original_failure": {
            "exception_type": "RuntimeError",
            "exception_message": "failed",
        } if intervened else None,
    }


def test_aggregate_episode_interventions_happy_path_is_json_safe_and_detached():
    records = [_record(0), _record(1, intervened=False)]
    result = aggregate_episode_interventions(records, 2)

    assert result == {
        "total_steps": 2,
        "intervention_count": 1,
        "intervention_rate": 0.5,
        "first_intervention_step": 0,
        "selected_lambda_mean": 0.125,
        "selected_lambda_max": 0.125,
        "intervention_l1_mean": pytest.approx(0.1),
        "intervention_l1_max": pytest.approx(0.1),
        "intervention_l2_mean": pytest.approx(np.sqrt(0.005)),
        "intervention_l2_max": pytest.approx(np.sqrt(0.005)),
        "intervention_linf_mean": pytest.approx(0.05),
        "intervention_linf_max": pytest.approx(0.05),
        "per_channel_intervention_counts": [1, 1],
        "extra_solver_attempts": 2,
        "shield_elapsed_seconds": 0.5,
        "ode_failure_count": 0,
    }
    json.dumps(result, allow_nan=False)
    records[0]["requested_action"][0] = 999
    assert result["per_channel_intervention_counts"] == [1, 1]


def test_aggregate_no_intervention_uses_zero_summaries():
    result = aggregate_episode_interventions([_record(0, intervened=False)], 2)
    assert result["first_intervention_step"] is None
    for field in (
        "selected_lambda_mean", "selected_lambda_max", "intervention_l1_mean",
        "intervention_l1_max", "intervention_l2_mean", "intervention_l2_max",
        "intervention_linf_mean", "intervention_linf_max",
    ):
        assert result[field] == 0.0


def test_aggregate_rejects_unchanged_flag_with_changed_action_and_bad_candidate_action():
    unchanged = _record(0, intervened=False)
    unchanged["executed_action"] = [0.4, -0.5]
    difference = np.asarray(unchanged["requested_action"]) - unchanged["executed_action"]
    unchanged["intervention_l1"] = float(np.linalg.norm(difference, 1))
    unchanged["intervention_l2"] = float(np.linalg.norm(difference, 2))
    unchanged["intervention_linf"] = float(np.linalg.norm(difference, np.inf))
    unchanged["per_channel_changed"] = (difference != 0).tolist()
    with pytest.raises(ValueError, match="without intervention"):
        aggregate_episode_interventions([unchanged], 2)

    bad_attempt = _record(0)
    bad_attempt["candidate_attempts"][0]["action"] = [-1.0, -1.0]
    with pytest.raises(ValueError, match="candidate attempt action"):
        aggregate_episode_interventions([bad_attempt], 2)

    bad_selected = _record(0)
    bad_selected["executed_action"] = [0.4, -0.4]
    difference = np.asarray(bad_selected["requested_action"]) - bad_selected["executed_action"]
    for order in ("l1", "l2", "linf"):
        ord_value = {"l1": 1, "l2": 2, "linf": np.inf}[order]
        bad_selected[f"intervention_{order}"] = float(np.linalg.norm(difference, ord_value))
    bad_selected["per_channel_changed"] = (difference != 0).tolist()
    with pytest.raises(ValueError, match="executed_action"):
        aggregate_episode_interventions([bad_selected], 2)


def test_aggregate_rejects_nonfinite_elapsed_sum():
    records = [_record(0, intervened=False), _record(1, intervened=False)]
    records[0]["elapsed_seconds"] = np.finfo(float).max
    records[1]["elapsed_seconds"] = np.finfo(float).max
    with pytest.raises(ValueError, match="aggregate"):
        aggregate_episode_interventions(records, 2)


@pytest.mark.parametrize(
    "original_failure",
    [
        {},
        {"exception_type": "RuntimeError"},
        {"exception_message": "failed"},
        {"exception_type": "RuntimeError", "exception_message": "failed", "extra": 1},
        {"exception_type": "", "exception_message": "failed"},
        {"exception_type": 1, "exception_message": "failed"},
        {"exception_type": "RuntimeError", "exception_message": None},
    ],
)
def test_aggregate_rejects_malformed_original_failure_schema(original_failure):
    record = _record(0)
    record["original_failure"] = original_failure
    with pytest.raises((TypeError, ValueError), match="original_failure"):
        aggregate_episode_interventions([record], 2)


def test_aggregate_accepts_empty_original_exception_message():
    record = _record(0)
    record["original_failure"] = {
        "exception_type": "RuntimeError",
        "exception_message": "",
    }
    result = aggregate_episode_interventions([record], 2)
    assert result["intervention_count"] == 1


@pytest.mark.parametrize(
    ("mutator", "match"),
    [
        (lambda rs: rs.clear(), "nonempty"),
        (lambda rs: rs[0].update(step_index=True), "step_index"),
        (lambda rs: rs[1].update(step_index=2), "consecutive"),
        (lambda rs: rs[1].update(step_index=0), "consecutive"),
        (lambda rs: rs[0].update(schema_version="bad"), "schema_version"),
        (lambda rs: rs[0].update(intervened=1), "intervened"),
        (lambda rs: rs[0].update(requested_action=[0.0]), "shape"),
        (lambda rs: rs[0].update(executed_action=[np.inf, 0]), "finite"),
        (lambda rs: rs[0].update(reference_action=[2, 0]), "within"),
        (lambda rs: rs[0].update(selected_lambda=0.2), "selected_lambda"),
        (lambda rs: rs[0].update(extra_solver_attempts=1), "length"),
        (lambda rs: rs[0]["candidate_attempts"][0].pop("lambda"), "lambda"),
        (lambda rs: rs[0]["candidate_attempts"][0].update({"lambda": 0.125}), "prefix"),
        (lambda rs: rs[0].update(original_failure=None), "original_failure"),
        (lambda rs: rs[0].update(intervention_l1=9.0), "intervention_l1"),
        (lambda rs: rs[0].update(per_channel_changed=[True, False]), "per_channel_changed"),
        (lambda rs: rs[0].update(elapsed_seconds=-1), "elapsed_seconds"),
    ],
)
def test_aggregate_rejects_malformed_records(mutator, match):
    records = [_record(0), _record(1, intervened=False)]
    mutator(records)
    with pytest.raises((TypeError, ValueError), match=match):
        aggregate_episode_interventions(records, 2)


@pytest.mark.parametrize("action_dim", [0, -1, 1.5, True])
def test_aggregate_rejects_invalid_action_dim(action_dim):
    with pytest.raises((TypeError, ValueError), match="action_dim"):
        aggregate_episode_interventions([_record(0)], action_dim)


KEYS = {(1, "a", "deterministic"), (2, "b", "deterministic")}


def _tables():
    common = [
        {"seed": 1, "task_id": "a", "inference_mode": "deterministic", "completed": True,
         "ode_failure_count": 0, "episode_return": 98.0, "temp_violation": 1.05,
         "co2_violation": 1.05, "rh_violation": 2.1, "total_steps": 1000,
         "intervention_count": 5},
        {"seed": 2, "task_id": "b", "inference_mode": "deterministic", "completed": True,
         "ode_failure_count": 0, "episode_return": 98.0, "temp_violation": 1.05,
         "co2_violation": 1.05, "rh_violation": 2.1, "total_steps": 1000,
         "intervention_count": 5},
    ]
    shielded = pd.DataFrame(common)
    unshielded = shielded.drop(columns=["total_steps", "intervention_count"]).copy()
    unshielded["episode_return"] = 100.0
    unshielded["temp_violation"] = 1.0
    unshielded["co2_violation"] = 1.0
    unshielded["rh_violation"] = 2.0
    return shielded, unshielded


def test_gate_inclusive_boundaries_and_paired_reporting():
    shielded, unshielded = _tables()
    paired = build_paired_shield_deltas(shielded, unshielded, KEYS)
    decision = evaluate_shield_gate(shielded, unshielded, KEYS)

    assert len(paired) == 2
    assert paired["episode_return_delta"].tolist() == [-2.0, -2.0]
    assert paired["co2_violation_ratio"].tolist() == pytest.approx([1.05, 1.05])
    assert decision["outcome"] == "pass"
    assert decision["reasons"] == []
    assert set(decision["conditions"]) == {
        "zero_ode_failures", "intervention_rate_within_0p5pct",
        "paired_return_loss_within_2pct", "paired_violation_burden_within_5pct",
    }
    assert decision["evidence"]["paired_count"] == 2
    json.dumps(decision, allow_nan=False)


@pytest.mark.parametrize(
    ("change", "failed"),
    [
        (lambda s, u: s.loc[0, "ode_failure_count"] == 1, "zero_ode_failures"),
        (lambda s, u: s.__setitem__("intervention_count", [6, 5]), "intervention_rate_within_0p5pct"),
        (lambda s, u: s.__setitem__("episode_return", [97.999999, 98.0]), "paired_return_loss_within_2pct"),
        (lambda s, u: s.__setitem__(["temp_violation", "co2_violation", "rh_violation"],
                                    [[1.050001, 1.050001, 2.100002]] * 2),
         "paired_violation_burden_within_5pct"),
    ],
)
def test_gate_fails_just_over_each_boundary(change, failed):
    shielded, unshielded = _tables()
    if failed == "zero_ode_failures":
        shielded.loc[0, "ode_failure_count"] = 1
    else:
        change(shielded, unshielded)
    result = evaluate_shield_gate(shielded, unshielded, KEYS)
    assert result["outcome"] == "fail"
    assert failed in result["reasons"]


def test_zero_zero_violation_is_neutral_but_new_violation_is_not():
    shielded, unshielded = _tables()
    shielded["co2_violation"] = 0.0
    unshielded["co2_violation"] = 0.0
    assert evaluate_shield_gate(shielded, unshielded, KEYS)["outcome"] == "pass"
    shielded["co2_violation"] = 1e-6
    result = evaluate_shield_gate(shielded, unshielded, KEYS)
    assert result["evidence"]["co2_violation_ratio_mean"] > 1.05
    assert "paired_violation_burden_within_5pct" in result["reasons"]


def test_violation_gate_uses_preregistered_mean_across_metrics_and_pairs():
    shielded, unshielded = _tables()
    shielded["temp_violation"] = 1.10
    shielded["co2_violation"] = 1.0
    shielded["rh_violation"] = 2.0
    result = evaluate_shield_gate(shielded, unshielded, KEYS)
    assert result["evidence"]["paired_violation_ratio_mean"] == pytest.approx(1.10 / 3 + 2 / 3)
    assert result["conditions"]["paired_violation_burden_within_5pct"] is True


def test_unshielded_incomplete_is_excluded_but_counted_and_shield_incomplete_fails():
    shielded, unshielded = _tables()
    unshielded.loc[1, "completed"] = False
    unshielded.loc[1, ["episode_return", "temp_violation", "co2_violation", "rh_violation"]] = np.nan
    result = evaluate_shield_gate(shielded, unshielded, KEYS)
    assert result["evidence"]["paired_count"] == 1
    assert result["evidence"]["unshielded_completion_count"] == 1
    shielded.loc[1, "completed"] = False
    shielded.loc[1, ["episode_return", "temp_violation", "co2_violation", "rh_violation"]] = np.nan
    result = evaluate_shield_gate(shielded, unshielded, KEYS)
    assert "zero_ode_failures" in result["reasons"]


@pytest.mark.parametrize("malformation", ["missing", "extra", "duplicate", "negative", "nonfinite"])
def test_gate_rejects_malformed_tables(malformation):
    shielded, unshielded = _tables()
    expected = KEYS
    if malformation == "missing":
        shielded = shielded.iloc[:1]
    elif malformation == "extra":
        expected = {(1, "a", "deterministic")}
    elif malformation == "duplicate":
        shielded = pd.concat([shielded, shielded.iloc[[0]]], ignore_index=True)
    elif malformation == "negative":
        shielded.loc[0, "temp_violation"] = -1
    else:
        shielded.loc[0, "episode_return"] = np.inf
    with pytest.raises((TypeError, ValueError)):
        evaluate_shield_gate(shielded, unshielded, expected)


def test_gate_rejects_zero_pairs():
    shielded, unshielded = _tables()
    unshielded["completed"] = False
    unshielded[["episode_return", "temp_violation", "co2_violation", "rh_violation"]] = np.nan
    with pytest.raises(ValueError, match="paired"):
        evaluate_shield_gate(shielded, unshielded, KEYS)


def test_gate_rejects_bool_and_float_seed_instead_of_pairing_by_pandas_coercion():
    shielded, unshielded = _tables()
    shielded["seed"] = shielded["seed"].astype(object)
    shielded.loc[0, "seed"] = True
    with pytest.raises((TypeError, ValueError), match="seed"):
        evaluate_shield_gate(shielded, unshielded, KEYS)

    shielded, unshielded = _tables()
    shielded["seed"] = shielded["seed"].astype(float)
    with pytest.raises((TypeError, ValueError), match="seed"):
        evaluate_shield_gate(shielded, unshielded, KEYS)


def test_gate_rejects_duplicate_or_malformed_expected_keys_before_set_collapse():
    shielded, unshielded = _tables()
    duplicate = [
        (1, "a", "deterministic"),
        (np.int64(1), np.str_("a"), np.str_("deterministic")),
        (2, "b", "deterministic"),
    ]
    with pytest.raises(ValueError, match="duplicate expected"):
        evaluate_shield_gate(shielded, unshielded, duplicate)
    with pytest.raises(ValueError, match="length"):
        evaluate_shield_gate(shielded, unshielded, [(1, "a"), (2, "b")])


@pytest.mark.parametrize(
    "bad_key",
    [
        (True, "a", "deterministic"),
        (1.0, "a", "deterministic"),
        (1, "", "deterministic"),
        (1, "a", None),
        (1, "a", np.nan),
    ],
)
def test_gate_rejects_malformed_expected_key_scalars(bad_key):
    shielded, unshielded = _tables()
    expected = [bad_key, (2, "b", "deterministic")]
    with pytest.raises((TypeError, ValueError), match="expected key"):
        evaluate_shield_gate(shielded, unshielded, expected)


def test_gate_rejects_empty_or_duplicate_key_columns():
    shielded, unshielded = _tables()
    with pytest.raises(ValueError, match="nonempty"):
        evaluate_shield_gate(shielded, unshielded, [(1,)], key_columns=("",))
    with pytest.raises(ValueError, match="unique"):
        evaluate_shield_gate(
            shielded,
            unshielded,
            [(1, 1), (2, 2)],
            key_columns=("seed", "seed"),
        )


@pytest.mark.parametrize(
    ("column", "value"),
    [
        ("task_id", ""),
        ("inference_mode", None),
        ("task_id", np.nan),
        ("inference_mode", 1),
    ],
)
def test_gate_rejects_null_empty_or_unsupported_string_key_scalars(column, value):
    shielded, unshielded = _tables()
    shielded.loc[0, column] = value
    with pytest.raises((TypeError, ValueError), match=column):
        evaluate_shield_gate(shielded, unshielded, KEYS)


def test_gate_canonicalizes_numpy_key_scalars_for_pairing_and_output():
    shielded, unshielded = _tables()
    shielded["seed"] = shielded["seed"].map(np.int64)
    unshielded["seed"] = unshielded["seed"].map(np.int64)
    shielded["task_id"] = shielded["task_id"].map(np.str_)
    unshielded["task_id"] = unshielded["task_id"].map(np.str_)
    paired = build_paired_shield_deltas(shielded, unshielded, KEYS)
    assert all(type(value) is int for value in paired["seed"])
    assert all(type(value) is str for value in paired["task_id"])


def test_gate_rejects_overflowing_derived_evidence():
    shielded, unshielded = _tables()
    shielded["temp_violation"] = np.finfo(float).max
    unshielded["temp_violation"] = 0.0
    with pytest.raises(ValueError, match="finite"):
        evaluate_shield_gate(shielded, unshielded, KEYS)


def test_gate_uses_stable_finite_means_when_naive_numpy_mean_overflows():
    shielded, unshielded = _tables()
    huge = np.finfo(float).max / 2
    shielded["episode_return"] = huge
    unshielded["episode_return"] = 0.0
    for metric in ("temp_violation", "co2_violation", "rh_violation"):
        shielded[metric] = huge
        unshielded[metric] = 1.0

    result = evaluate_shield_gate(shielded, unshielded, KEYS)
    assert np.isfinite(result["evidence"]["mean_paired_return_delta"])
    assert np.isfinite(result["evidence"]["paired_violation_ratio_mean"])
    assert result["evidence"]["paired_violation_ratio_mean"] == pytest.approx(huge)
    json.dumps(result, allow_nan=False)


def test_gate_stably_averages_extreme_unshielded_returns_and_rejects_delta_overflow():
    shielded, unshielded = _tables()
    huge = np.finfo(float).max
    shielded["episode_return"] = huge
    unshielded["episode_return"] = huge
    result = evaluate_shield_gate(shielded, unshielded, KEYS)
    assert result["evidence"]["relative_return_loss"] == 0.0
    json.dumps(result, allow_nan=False)

    shielded["episode_return"] = huge
    unshielded["episode_return"] = -huge
    with pytest.raises(ValueError, match="delta.*finite|finite.*delta"):
        evaluate_shield_gate(shielded, unshielded, KEYS)


def _artifact_frames():
    raw = pd.DataFrame([{"seed": 1, "task_id": "a", "value": 1.0}])
    paired = pd.DataFrame([{"seed": 1, "task_id": "a", "delta": 0.0}])
    interventions = pd.DataFrame([{"seed": 1, "task_id": "a", "first_intervention_step": np.nan,
                                   "intervention_count": 0}])
    return raw, paired, interventions


def test_atomic_writer_publishes_exact_outputs_and_strict_json(tmp_path):
    raw, paired, interventions = _artifact_frames()
    root = tmp_path / "shield"
    paths = write_shield_artifacts_atomic(
        raw, paired, interventions, {"root": Path("portable"), "n": np.int64(1)},
        {"outcome": "pass", "rate": np.float64(0.0)}, root,
    )
    assert set(path.name for path in root.iterdir()) == {
        "eval_raw.csv", "paired_deltas.csv", "interventions.csv",
        "shield_manifest.json", "decision.json",
    }
    assert set(paths) == {"eval_raw", "paired_deltas", "interventions", "shield_manifest", "decision"}
    assert json.loads(paths["shield_manifest"].read_text(encoding="utf-8"))["root"] == "portable"


@pytest.mark.parametrize("where", ["csv", "json"])
def test_atomic_writer_failure_preserves_old_root(monkeypatch, tmp_path, where):
    raw, paired, interventions = _artifact_frames()
    root = tmp_path / "shield"
    root.mkdir()
    (root / "old.txt").write_text("old", encoding="utf-8")
    if where == "csv":
        monkeypatch.setattr(pd.DataFrame, "to_csv", lambda *a, **k: (_ for _ in ()).throw(OSError("boom")))
        manifest = {"ok": True}
    else:
        manifest = {"bad": np.nan}
    with pytest.raises((OSError, ValueError)):
        write_shield_artifacts_atomic(raw, paired, interventions, manifest, {"outcome": "pass"}, root)
    assert (root / "old.txt").read_text(encoding="utf-8") == "old"
    assert list(tmp_path.iterdir()) == [root]


def test_atomic_publish_replace_failure_restores_old_root_without_partial_final(monkeypatch, tmp_path):
    raw, paired, interventions = _artifact_frames()
    root = tmp_path / "shield"
    root.mkdir()
    (root / "old.txt").write_text("old", encoding="utf-8")
    real_replace = shield_module.os.replace

    def fail_stage_publish(source, destination):
        if ".stage-" in Path(source).name and Path(destination) == root:
            raise OSError("injected publish failure")
        return real_replace(source, destination)

    monkeypatch.setattr(shield_module.os, "replace", fail_stage_publish)
    with pytest.raises(OSError, match="injected"):
        write_shield_artifacts_atomic(raw, paired, interventions, {}, {}, root)
    assert set(path.name for path in root.iterdir()) == {"old.txt"}
    assert list(tmp_path.iterdir()) == [root]


def test_post_commit_backup_cleanup_failure_is_best_effort(monkeypatch, tmp_path):
    raw, paired, interventions = _artifact_frames()
    root = tmp_path / "shield"
    root.mkdir()
    (root / "old.txt").write_text("old", encoding="utf-8")
    real_rmtree = shield_module.shutil.rmtree

    def fail_backup_cleanup(path, *args, **kwargs):
        if ".backup-" in Path(path).name:
            raise OSError("injected cleanup failure")
        return real_rmtree(path, *args, **kwargs)

    monkeypatch.setattr(shield_module.shutil, "rmtree", fail_backup_cleanup)
    paths = write_shield_artifacts_atomic(raw, paired, interventions, {}, {}, root)

    assert set(path.name for path in root.iterdir()) == {
        "eval_raw.csv", "paired_deltas.csv", "interventions.csv",
        "shield_manifest.json", "decision.json",
    }
    assert all(path.is_file() for path in paths.values())
    backups = [path for path in tmp_path.iterdir() if ".backup-" in path.name]
    assert len(backups) == 1
    real_rmtree(backups[0])


def test_failed_publish_uses_copy_fallback_when_atomic_restore_also_fails(monkeypatch, tmp_path):
    raw, paired, interventions = _artifact_frames()
    root = tmp_path / "shield"
    root.mkdir()
    (root / "old.txt").write_text("old", encoding="utf-8")
    real_replace = shield_module.os.replace

    def fail_publish_and_restore(source, destination):
        source_path = Path(source)
        destination_path = Path(destination)
        if destination_path == root and ".stage-" in source_path.name:
            raise OSError("injected publication failure")
        if destination_path == root and ".backup-" in source_path.name:
            raise OSError("injected atomic restoration failure")
        return real_replace(source, destination)

    monkeypatch.setattr(shield_module.os, "replace", fail_publish_and_restore)
    with pytest.raises(OSError, match="injected publication failure") as captured:
        write_shield_artifacts_atomic(raw, paired, interventions, {}, {}, root)

    backups = [path for path in tmp_path.iterdir() if ".backup-" in path.name]
    assert set(path.name for path in root.iterdir()) == {"old.txt"}
    assert (root / "old.txt").read_text(encoding="utf-8") == "old"
    assert backups == []
    assert any("atomic backup rename restoration failed" in note for note in captured.value.__notes__)


def test_failed_fallback_copy_preserves_backup_and_annotates_primary_error(monkeypatch, tmp_path):
    raw, paired, interventions = _artifact_frames()
    root = tmp_path / "shield"
    root.mkdir()
    (root / "old.txt").write_text("old", encoding="utf-8")
    real_replace = shield_module.os.replace

    def fail_publish_and_restore(source, destination):
        source_path = Path(source)
        destination_path = Path(destination)
        if destination_path == root and ".stage-" in source_path.name:
            raise OSError("injected publication failure")
        if destination_path == root and ".backup-" in source_path.name:
            raise OSError("injected atomic restoration failure")
        return real_replace(source, destination)

    monkeypatch.setattr(shield_module.os, "replace", fail_publish_and_restore)
    monkeypatch.setattr(
        shield_module.shutil,
        "copytree",
        lambda *args, **kwargs: (_ for _ in ()).throw(OSError("injected fallback copy failure")),
    )
    with pytest.raises(OSError, match="injected publication failure") as captured:
        write_shield_artifacts_atomic(raw, paired, interventions, {}, {}, root)

    backups = [path for path in tmp_path.iterdir() if ".backup-" in path.name]
    assert not root.exists()
    assert len(backups) == 1
    assert (backups[0] / "old.txt").read_text(encoding="utf-8") == "old"
    assert any("atomic backup rename restoration failed" in note for note in captured.value.__notes__)
    assert any("fallback copy restoration failed" in note for note in captured.value.__notes__)


def test_atomic_writer_rejects_empty_nonfinite_duplicates_and_file_root(tmp_path):
    raw, paired, interventions = _artifact_frames()
    with pytest.raises(ValueError):
        write_shield_artifacts_atomic(raw.iloc[:0], paired, interventions, {}, {}, tmp_path / "a")
    bad = pd.concat([raw, raw], ignore_index=True)
    with pytest.raises(ValueError, match="duplicate"):
        write_shield_artifacts_atomic(bad, paired, interventions, {}, {}, tmp_path / "b")
    raw.loc[0, "value"] = np.inf
    with pytest.raises(ValueError, match="finite"):
        write_shield_artifacts_atomic(raw, paired, interventions, {}, {}, tmp_path / "c")
    file_root = tmp_path / "file"
    file_root.write_text("x", encoding="utf-8")
    with pytest.raises(ValueError, match="file"):
        write_shield_artifacts_atomic(*_artifact_frames(), {}, {}, file_root)


@pytest.mark.parametrize("bad", [pd.NaT, np.datetime64("NaT"), pd.NA, [1], {"a": 1}, object()])
def test_atomic_writer_rejects_missing_or_nonscalar_nonnullable_cells(tmp_path, bad):
    raw, paired, interventions = _artifact_frames()
    raw["bad"] = pd.Series([bad], dtype=object)
    with pytest.raises((TypeError, ValueError), match="raw.bad"):
        write_shield_artifacts_atomic(raw, paired, interventions, {}, {}, tmp_path / "bad")


def test_atomic_writer_normalizes_all_nullable_missing_representations(tmp_path):
    for index, missing in enumerate((None, np.nan, pd.NA, pd.NaT, np.datetime64("NaT"))):
        raw, paired, interventions = _artifact_frames()
        interventions["first_intervention_step"] = pd.Series([missing], dtype=object)
        root = tmp_path / f"nullable-{index}"
        paths = write_shield_artifacts_atomic(raw, paired, interventions, {}, {}, root)
        written = pd.read_csv(paths["interventions"])
        assert pd.isna(written.loc[0, "first_intervention_step"])
