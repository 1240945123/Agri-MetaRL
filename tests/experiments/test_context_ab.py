import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from gl_gym.experiments.context_ab import (
    DIAGNOSTIC_TASK_IDS,
    MODES,
    PAIR_METRICS,
    build_paired_deltas,
    evaluate_context_gate,
    select_diagnostic_tasks,
    write_context_ab_artifacts,
)


def _task_table() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "task_id": reversed(DIAGNOSTIC_TASK_IDS),
            "split": ["economic", "economic", "uncertainty", "uncertainty", "heldout", "heldout", "heldout", "fixed"],
        }
    )


def _write_trace(path: Path, value: float, shape: tuple[int, ...] = (3, 2)) -> str:
    np.save(path, np.full(shape, value, dtype=np.float32))
    return str(path)


def _raw_pair(tmp_path: Path) -> pd.DataFrame:
    rows = []
    for mode, return_value, epi, action in (
        ("zero_context", 100.0, 2.0, 0.0),
        ("online_context", 110.0, 3.0, 1.0),
    ):
        rows.append(
            {
                "seed": 42,
                "task_id": "t",
                "split": "heldout",
                "inference_mode": mode,
                "episode_return": return_value,
                "EPI": epi,
                "temp_violation": 10.0 if mode == "zero_context" else 8.0,
                "co2_violation": 20.0 if mode == "zero_context" else 18.0,
                "rh_violation": 30.0 if mode == "zero_context" else 27.0,
                "action_trace_path": _write_trace(tmp_path / f"{mode}.npy", action),
            }
        )
    return pd.DataFrame(rows)


def _passing_paired_fixture() -> pd.DataFrame:
    rows = []
    splits = ["fixed", "heldout", "heldout", "heldout", "uncertainty", "uncertainty", "economic", "economic"]
    for seed in (42, 123):
        for index, (task_id, split) in enumerate(zip(DIAGNOSTIC_TASK_IDS, splits, strict=True)):
            zero_return = 100.0
            delta = 0.0 if split == "fixed" else 1.0
            row = {
                "seed": seed,
                "task_id": task_id,
                "split": split,
                "mean_abs_action_delta": 0.1,
            }
            for metric in PAIR_METRICS:
                if metric == "episode_return":
                    zero, online = zero_return, zero_return + delta
                elif metric == "EPI":
                    zero, online = 10.0, 10.5
                else:
                    zero, online = 10.0, 9.0
                row[f"{metric}_zero"] = zero
                row[f"{metric}_online"] = online
                row[f"{metric}_delta"] = online - zero
            rows.append(row)
    return pd.DataFrame(rows)


def _raw_diagnostic_fixture(tmp_path: Path) -> pd.DataFrame:
    rows = []
    splits = ["fixed", "heldout", "heldout", "heldout", "uncertainty", "uncertainty", "economic", "economic"]
    for seed in (42, 123):
        for index, (task_id, split) in enumerate(zip(DIAGNOSTIC_TASK_IDS, splits, strict=True)):
            for mode_index, mode in enumerate(MODES):
                rows.append(
                    {
                        "seed": seed,
                        "task_id": task_id,
                        "split": split,
                        "inference_mode": mode,
                        "episode_return": 100.0 + mode_index,
                        "EPI": 10.0 + mode_index,
                        "temp_violation": 10.0 - mode_index,
                        "co2_violation": 10.0 - mode_index,
                        "rh_violation": 10.0 - mode_index,
                        "action_trace_path": _write_trace(
                            tmp_path / f"{seed}_{index}_{mode}.npy", float(mode_index)
                        ),
                    }
                )
    return pd.DataFrame(rows)


def test_diagnostic_constants_are_exact_and_ordered():
    assert DIAGNOSTIC_TASK_IDS == (
        "fixed_2010_d59_u0p00_standard",
        "heldout_2011_d59_u0p00_standard",
        "heldout_2012_d59_u0p00_standard",
        "heldout_2013_d59_u0p00_standard",
        "uncertainty_2012_d80_u0p05_standard",
        "uncertainty_2013_d100_u0p15_standard",
        "economic_2011_d59_u0p00_high_energy_price",
        "economic_2013_d100_u0p00_combined_stress",
    )
    assert MODES == ("zero_context", "online_context")
    assert PAIR_METRICS == (
        "episode_return", "EPI", "temp_violation", "co2_violation", "rh_violation"
    )


def test_select_diagnostic_tasks_preserves_approved_order():
    selected = select_diagnostic_tasks(_task_table())
    assert selected["task_id"].tolist() == list(DIAGNOSTIC_TASK_IDS)


def test_select_diagnostic_tasks_lists_missing_ids():
    with pytest.raises(ValueError, match="missing diagnostic task IDs") as error:
        select_diagnostic_tasks(_task_table().iloc[:-1])
    assert DIAGNOSTIC_TASK_IDS[0] in str(error.value)


def test_select_diagnostic_tasks_rejects_duplicate_task_ids():
    tasks = pd.concat([_task_table(), _task_table().iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="duplicate task IDs"):
        select_diagnostic_tasks(tasks)


def test_build_paired_deltas_uses_online_minus_zero_and_action_delta(tmp_path: Path):
    paired = build_paired_deltas(_raw_pair(tmp_path))
    assert paired.loc[0, "episode_return_zero"] == 100.0
    assert paired.loc[0, "episode_return_online"] == 110.0
    assert paired.loc[0, "episode_return_delta"] == 10.0
    assert paired.loc[0, "EPI_delta"] == 1.0
    assert paired.loc[0, "mean_abs_action_delta"] == 1.0


def test_build_paired_deltas_supports_injected_action_loader(tmp_path: Path):
    raw = _raw_pair(tmp_path)
    paired = build_paired_deltas(
        raw,
        load_actions=lambda path: np.zeros((2, 1)) if "zero" in str(path) else np.full((2, 1), 0.5),
    )
    assert paired.loc[0, "mean_abs_action_delta"] == 0.5


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda raw: raw.iloc[:1], "both inference modes"),
        (lambda raw: pd.concat([raw, raw.iloc[[0]]], ignore_index=True), "duplicate"),
    ],
)
def test_build_paired_deltas_rejects_missing_mode_and_duplicate(tmp_path: Path, mutation, message):
    with pytest.raises(ValueError, match=message):
        build_paired_deltas(mutation(_raw_pair(tmp_path)))


def test_build_paired_deltas_rejects_nonfinite_metrics(tmp_path: Path):
    raw = _raw_pair(tmp_path)
    raw.loc[0, "EPI"] = np.nan
    with pytest.raises(ValueError, match="finite"):
        build_paired_deltas(raw)


@pytest.mark.parametrize(
    ("zero", "online", "message"),
    [
        (np.empty((0, 1)), np.empty((0, 1)), "nonempty"),
        (np.zeros((2, 1)), np.zeros((3, 1)), "same shape"),
        (np.array([[np.nan]]), np.zeros((1, 1)), "finite"),
    ],
)
def test_build_paired_deltas_validates_action_arrays(tmp_path: Path, zero, online, message):
    raw = _raw_pair(tmp_path)
    arrays = {"zero": zero, "online": online}
    with pytest.raises(ValueError, match=message):
        build_paired_deltas(raw, load_actions=lambda path: arrays["zero" if "zero" in str(path) else "online"])


def test_gate_passes_only_when_all_five_conditions_hold():
    decision = evaluate_context_gate(_passing_paired_fixture())
    assert decision["outcome"] == "continue_to_500k"
    assert all(decision["conditions"].values())
    json.dumps(decision, allow_nan=False)


@pytest.mark.parametrize(
    "condition",
    [
        "actions_change_both_seeds",
        "positive_nonfixed_return",
        "no_seed_large_return_loss",
        "violation_burden_within_5pct",
        "fixed_return_within_2pct",
    ],
)
def test_gate_fails_each_condition_individually(condition):
    paired = _passing_paired_fixture()
    if condition == "actions_change_both_seeds":
        paired.loc[paired["seed"] == 42, "mean_abs_action_delta"] = 0.0
    elif condition == "positive_nonfixed_return":
        paired.loc[paired["split"] != "fixed", "episode_return_delta"] = -1.0
        paired.loc[paired["split"] != "fixed", "episode_return_online"] = 99.0
    elif condition == "no_seed_large_return_loss":
        mask_42 = (paired["seed"] == 42) & (paired["split"] != "fixed")
        mask_123 = (paired["seed"] == 123) & (paired["split"] != "fixed")
        paired.loc[mask_42, ["episode_return_delta", "episode_return_online"]] = [-3.0, 97.0]
        paired.loc[mask_123, ["episode_return_delta", "episode_return_online"]] = [5.0, 105.0]
    elif condition == "violation_burden_within_5pct":
        for metric in ("temp_violation", "co2_violation", "rh_violation"):
            paired[f"{metric}_online"] = 10.6
            paired[f"{metric}_delta"] = 0.6
    else:
        mask = (paired["seed"] == 42) & (paired["split"] == "fixed")
        paired.loc[mask, ["episode_return_delta", "episode_return_online"]] = [-3.0, 97.0]

    decision = evaluate_context_gate(paired)
    assert decision["outcome"] == "redesign_before_training"
    assert decision["conditions"][condition] is False


def test_gate_handles_nonfinite_evidence_with_strict_json():
    paired = _passing_paired_fixture()
    paired.loc[0, "episode_return_delta"] = np.nan
    decision = evaluate_context_gate(paired)
    assert decision["outcome"] == "redesign_before_training"
    assert decision["reasons"]
    assert "non-finite" in decision["reasons"][0]
    json.dumps(decision, allow_nan=False)


def test_write_context_ab_artifacts_writes_complete_schema_and_strict_json(tmp_path: Path):
    raw = _raw_diagnostic_fixture(tmp_path)
    root = tmp_path / "diagnostic"
    manifest = {"source": "pilot3", "seeds": [np.int64(42), np.int64(123)]}
    paths = write_context_ab_artifacts(raw, root, manifest)

    assert set(paths) == {"eval_raw", "paired_deltas", "split_summary", "diagnostic_manifest", "decision"}
    written_raw = pd.read_csv(root / "eval_raw.csv")
    paired = pd.read_csv(root / "paired_deltas.csv")
    summary = pd.read_csv(root / "split_summary.csv")
    assert len(written_raw) == 32
    assert len(paired) == 16
    assert set(["inference_mode", "split", *PAIR_METRICS]).issubset(summary.columns)
    assert json.loads((root / "diagnostic_manifest.json").read_text(encoding="utf-8"))["seeds"] == [42, 123]
    decision_text = (root / "decision.json").read_text(encoding="utf-8")
    json.loads(decision_text, parse_constant=lambda value: pytest.fail(f"non-standard JSON: {value}"))


def test_write_context_ab_artifacts_requires_existing_action_traces(tmp_path: Path):
    raw = _raw_diagnostic_fixture(tmp_path)
    Path(raw.loc[0, "action_trace_path"]).unlink()
    with pytest.raises(ValueError, match="action trace"):
        write_context_ab_artifacts(raw, tmp_path / "diagnostic", {})


@pytest.mark.parametrize("bad_kind", ["row_count", "duplicate", "mode"])
def test_write_context_ab_artifacts_rejects_invalid_raw_table(tmp_path: Path, bad_kind: str):
    raw = _raw_diagnostic_fixture(tmp_path)
    if bad_kind == "row_count":
        raw = raw.iloc[:-1]
        message = "32 rows"
    elif bad_kind == "duplicate":
        raw.loc[1, ["seed", "task_id", "inference_mode"]] = raw.loc[0, ["seed", "task_id", "inference_mode"]]
        message = "duplicate"
    else:
        raw.loc[0, "inference_mode"] = "adaptive_magic"
        message = "inference modes"
    with pytest.raises(ValueError, match=message):
        write_context_ab_artifacts(raw, tmp_path / "diagnostic", {})
