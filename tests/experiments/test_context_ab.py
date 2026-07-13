import json
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import gl_gym.experiments.context_ab as context_ab
from gl_gym.experiments.context_ab import (
    DIAGNOSTIC_TASK_IDS,
    MODES,
    PAIR_METRICS,
    build_paired_deltas,
    evaluate_context_gate,
    select_diagnostic_tasks,
    write_context_ab_artifacts,
)


def _load_context_cli():
    script = Path(__file__).resolve().parents[2] / "experiments" / "scripts" / "run_context_ab.py"
    spec = importlib.util.spec_from_file_location("run_context_ab", script)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


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
                "support_ready_step": 1.0 if mode == "online_context" else np.nan,
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
                        "support_ready_step": 1.0 if mode == "online_context" else np.nan,
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


def test_build_paired_deltas_uses_only_post_readiness_actions(tmp_path: Path):
    raw = _raw_pair(tmp_path)
    arrays = {
        "zero": np.array([[0.0], [0.0], [0.0]]),
        "online": np.array([[1.0], [0.0], [0.0]]),
    }
    paired = build_paired_deltas(
        raw,
        load_actions=lambda path: arrays["zero" if "zero" in str(path) else "online"],
    )
    assert paired.loc[0, "support_ready_step"] == 1
    assert paired.loc[0, "mean_abs_action_delta"] == 0.0


def test_build_paired_deltas_detects_post_readiness_action_change(tmp_path: Path):
    raw = _raw_pair(tmp_path)
    arrays = {
        "zero": np.array([[0.0], [0.0], [0.0]]),
        "online": np.array([[0.0], [1.0], [1.0]]),
    }
    paired = build_paired_deltas(
        raw,
        load_actions=lambda path: arrays["zero" if "zero" in str(path) else "online"],
    )
    assert paired.loc[0, "mean_abs_action_delta"] == 1.0


def test_pre_readiness_only_changes_fail_the_action_gate(tmp_path: Path):
    raw = _raw_diagnostic_fixture(tmp_path)
    for row in raw.itertuples(index=False):
        trace = np.zeros((3, 2), dtype=np.float32)
        if row.inference_mode == "online_context":
            trace[0] = 1.0
        np.save(row.action_trace_path, trace)
    paired = build_paired_deltas(raw)
    assert (paired["mean_abs_action_delta"] == 0.0).all()
    decision = evaluate_context_gate(paired)
    assert decision["conditions"]["actions_change_both_seeds"] is False


@pytest.mark.parametrize("support_ready_step", [np.nan, np.inf, 0, 1.5, 3, "bad"])
def test_build_paired_deltas_rejects_invalid_support_readiness(tmp_path: Path, support_ready_step):
    raw = _raw_pair(tmp_path)
    if isinstance(support_ready_step, str):
        raw["support_ready_step"] = raw["support_ready_step"].astype(object)
    raw.loc[raw["inference_mode"] == "online_context", "support_ready_step"] = support_ready_step
    with pytest.raises(ValueError, match="support_ready_step"):
        build_paired_deltas(raw)


def test_build_paired_deltas_default_loader_disables_pickle(tmp_path: Path, monkeypatch):
    raw = _raw_pair(tmp_path)
    real_load = np.load
    calls = []

    def recording_load(path, **kwargs):
        calls.append(kwargs)
        return real_load(path, **kwargs)

    monkeypatch.setattr(context_ab.np, "load", recording_load)
    build_paired_deltas(raw)
    assert calls == [{"allow_pickle": False}, {"allow_pickle": False}]


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
        (np.zeros(3), np.zeros(3), "2D"),
        (np.zeros((3, 0)), np.zeros((3, 0)), "positive dimensions"),
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


def test_gate_rejects_asymmetric_seed_task_subset_that_would_otherwise_pass():
    paired = _passing_paired_fixture()
    paired = paired.loc[~((paired["seed"] == 123) & (paired["task_id"] == DIAGNOSTIC_TASK_IDS[-1]))]
    decision = evaluate_context_gate(paired)
    assert decision["outcome"] == "redesign_before_training"
    assert decision["reasons"]
    assert "experiment structure" in decision["reasons"][0]


def test_gate_requires_exact_approved_seeds():
    paired = _passing_paired_fixture()
    paired.loc[paired["seed"] == 123, "seed"] = 999
    decision = evaluate_context_gate(paired)
    assert decision["outcome"] == "redesign_before_training"
    assert "experiment structure" in decision["reasons"][0]


def test_gate_treats_zero_over_zero_violation_as_neutral_without_dilution():
    paired = _passing_paired_fixture()
    for metric in ("co2_violation", "rh_violation"):
        paired[f"{metric}_zero"] = 0.0
        paired[f"{metric}_online"] = 0.0
        paired[f"{metric}_delta"] = 0.0
    paired["temp_violation_zero"] = 10.0
    paired["temp_violation_online"] = 20.0
    paired["temp_violation_delta"] = 10.0

    decision = evaluate_context_gate(paired)
    assert decision["evidence"]["mean_normalized_violation_burden"] == pytest.approx(4.0 / 3.0)
    assert decision["conditions"]["violation_burden_within_5pct"] is False


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
    published_traces = written_raw["action_trace_path"].map(Path)
    assert published_traces.map(lambda path: path.parent == root / "traces").all()
    assert published_traces.map(Path.is_file).all()
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


def test_write_context_ab_artifacts_rejects_substituted_tasks_even_with_32_rows(tmp_path: Path):
    raw = _raw_diagnostic_fixture(tmp_path)
    raw.loc[raw["task_id"] == DIAGNOSTIC_TASK_IDS[-1], "task_id"] = "economic_substitute"
    with pytest.raises(ValueError, match="approved diagnostic task IDs"):
        write_context_ab_artifacts(raw, tmp_path / "diagnostic", {})


def test_write_context_ab_artifacts_preserves_complete_root_on_publish_failure(tmp_path: Path, monkeypatch):
    raw = _raw_diagnostic_fixture(tmp_path)
    root = tmp_path / "diagnostic"
    root.mkdir()
    artifact_names = (
        "eval_raw.csv",
        "paired_deltas.csv",
        "split_summary.csv",
        "diagnostic_manifest.json",
        "decision.json",
    )
    for name in artifact_names:
        (root / name).write_bytes(f"old-{name}".encode())
    trace = root / "traces" / "keep.npy"
    trace.parent.mkdir()
    np.save(trace, np.ones((2, 1)))
    before = _tree_bytes(root)
    real_replace = context_ab.os.replace
    failed = False

    def fail_staging_publish(source, destination):
        nonlocal failed
        source_path = Path(source)
        if not failed and source_path.name.startswith(f".{root.name}.staging-"):
            failed = True
            raise OSError("injected publish failure")
        return real_replace(source, destination)

    monkeypatch.setattr(context_ab.os, "replace", fail_staging_publish)
    with pytest.raises(OSError, match="injected publish failure"):
        write_context_ab_artifacts(raw, root, {"revision": 2})

    assert _tree_bytes(root) == before
    assert not list(tmp_path.glob(".diagnostic.staging-*"))
    assert not list(tmp_path.glob(".diagnostic.backup-*"))


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


def _write_diagnostic_checkpoints(model_root: Path, seeds=(42, 123)) -> None:
    for seed in seeds:
        model = (
            model_root
            / "agri_metarl"
            / "deterministic"
            / "models"
            / f"agri_metarl_seed{seed}"
            / "last_model.zip"
        )
        vec = (
            model_root
            / "agri_metarl"
            / "deterministic"
            / "envs"
            / f"agri_metarl_seed{seed}"
            / "last_vecnormalize.pkl"
        )
        model.parent.mkdir(parents=True, exist_ok=True)
        vec.parent.mkdir(parents=True, exist_ok=True)
        model.write_bytes(b"model")
        vec.write_bytes(b"vec")


def _fake_cli_case(tmp_path: Path):
    cli = _load_context_cli()
    model_root = tmp_path / "models"
    _write_diagnostic_checkpoints(model_root)
    source_manifest = tmp_path / "source" / "suite_manifest.json"
    source_tasks_csv = tmp_path / "source" / "eval_tasks.csv"
    source_manifest.parent.mkdir(parents=True)
    source_manifest.write_text("{}", encoding="utf-8")
    source_tasks_csv.write_text("task_id\n", encoding="utf-8")
    suite = SimpleNamespace(
        suite_id="source", result_root=str(tmp_path / "source"), env_id="Fake"
    )
    tasks = pd.DataFrame(
        [
            {
                "suite_id": "source", "task_id": task_id,
                "split": task_id.split("_", 1)[0], "weather_year": 2010,
                "start_day": 59, "uncertainty_scale": 0.0,
                "economic_scenario": "standard",
                "climate_constraint_scenario": "standard",
            }
            for task_id in DIAGNOSTIC_TASK_IDS
        ]
    )

    class Model:
        num_timesteps = 196608

    class Env:
        def close(self):
            pass

    def successful_episode(model, env, *, inference_mode, return_diagnostics):
        value = float(inference_mode == "online_context")
        return (
            {metric: 100.0 if metric == "episode_return" else 1.0 for metric in PAIR_METRICS},
            {
                "action_trace": np.full((3, 1), value, dtype=np.float32),
                "support_ready_step": 1.0 if value else np.nan,
                "context_norm_mean": value,
                "context_norm_max": value,
            },
        )

    return SimpleNamespace(
        cli=cli, suite=suite, tasks=tasks,
        runs=cli.build_diagnostic_runs(model_root, [42, 123]),
        source_manifest=source_manifest, source_tasks_csv=source_tasks_csv,
        model_loader=lambda path, device: Model(),
        env_loader=lambda suite, task, path: Env(),
        episode_runner=successful_episode,
    )


def _tree_bytes(root: Path) -> dict[str, bytes]:
    return {
        str(path.relative_to(root)): path.read_bytes()
        for path in root.rglob("*") if path.is_file()
    }


def test_build_diagnostic_runs_uses_exact_last_checkpoint_paths(tmp_path: Path):
    cli = _load_context_cli()
    _write_diagnostic_checkpoints(tmp_path)

    runs = cli.build_diagnostic_runs(tmp_path, [42, 123])

    assert str(runs[0]["model_path"]).replace("\\", "/").endswith(
        "agri_metarl/deterministic/models/agri_metarl_seed42/last_model.zip"
    )
    assert str(runs[1]["vecnormalize_path"]).replace("\\", "/").endswith(
        "agri_metarl/deterministic/envs/agri_metarl_seed123/last_vecnormalize.pkl"
    )
    assert runs[0]["model_sha256"] == cli.sha256_file(runs[0]["model_path"])
    assert runs[0]["vecnormalize_sha256"] == cli.sha256_file(runs[0]["vecnormalize_path"])


def test_sha256_file_detects_same_path_replacement(tmp_path: Path):
    cli = _load_context_cli()
    path = tmp_path / "checkpoint.zip"
    path.write_bytes(b"first")
    first = cli.sha256_file(path, chunk_size=2)
    path.write_bytes(b"second")
    assert cli.sha256_file(path, chunk_size=2) != first


@pytest.mark.parametrize("seeds", [[42], [123, 42], [42, 123, 456]])
def test_build_diagnostic_runs_requires_exact_seed_order(tmp_path: Path, seeds):
    cli = _load_context_cli()
    with pytest.raises(ValueError, match="exactly 42 123"):
        cli.build_diagnostic_runs(tmp_path, seeds)


@pytest.mark.parametrize("missing", ["model", "vecnormalize"])
def test_build_diagnostic_runs_requires_both_checkpoint_files(tmp_path: Path, missing: str):
    cli = _load_context_cli()
    _write_diagnostic_checkpoints(tmp_path)
    target = next(tmp_path.rglob("last_model.zip" if missing == "model" else "last_vecnormalize.pkl"))
    target.unlink()
    with pytest.raises(FileNotFoundError, match=missing):
        cli.build_diagnostic_runs(tmp_path, [42, 123])


def test_validate_result_root_rejects_source_suite_collision(tmp_path: Path):
    cli = _load_context_cli()
    with pytest.raises(ValueError, match="diagnostic result root"):
        cli.validate_result_root(tmp_path / "suite", tmp_path / "suite" / ".." / "suite")


@pytest.mark.parametrize("relation", ["equal", "inside"])
def test_failure_root_must_be_isolated_from_formal_result_root(
    tmp_path: Path, relation: str
):
    cli = _load_context_cli()
    formal = tmp_path / "diagnostic"
    failure = formal if relation == "equal" else formal / "failures"
    with pytest.raises(ValueError, match="failure root"):
        cli.validate_failure_root(failure, formal)

    approved = tmp_path / ".diagnostic.work" / "failures"
    assert cli.validate_failure_root(approved, formal) == approved.resolve()


def test_context_cli_parser_accepts_failure_root():
    cli = _load_context_cli()
    args = cli.build_parser().parse_args(
        [
            "--source_manifest", "manifest.json",
            "--source_tasks_csv", "tasks.csv",
            "--model_root", "models",
            "--seeds", "42", "123",
            "--failure_root", "capsules",
        ]
    )
    assert args.failure_root == "capsules"


@pytest.mark.parametrize(
    "relation", ["ancestor", "descendant", "work", "staging", "staging_descendant"]
)
def test_validate_result_root_rejects_source_tree_and_derived_collisions(
    tmp_path: Path, relation: str
):
    cli = _load_context_cli()
    result = tmp_path / "diagnostic"
    if relation == "ancestor":
        source = result / "formal"
    elif relation == "descendant":
        source = tmp_path
    elif relation == "work":
        source = tmp_path / ".diagnostic.work"
    elif relation == "staging":
        source = tmp_path / ".diagnostic.staging-existing"
    else:
        source = tmp_path / ".diagnostic.staging-existing" / "formal"
    with pytest.raises(ValueError, match="diagnostic result root"):
        cli.validate_result_root(result, source)


def test_resume_skip_requires_matching_progress_row_and_valid_trace(tmp_path: Path):
    cli = _load_context_cli()
    trace = tmp_path / "trace.npy"
    np.save(trace, np.ones((2, 1), dtype=np.float32))
    row = {
        "seed": 42,
        "task_id": DIAGNOSTIC_TASK_IDS[0],
        "split": "fixed",
        "inference_mode": "zero_context",
        "checkpoint_steps": 100,
        "action_trace_path": str(trace),
        "support_ready_step": np.nan,
        "context_norm_mean": 0.0,
        "context_norm_max": 0.0,
        "model_sha256": "a" * 64,
        "vecnormalize_sha256": "b" * 64,
        "source_manifest_sha256": "c" * 64,
        "source_tasks_sha256": "d" * 64,
        **{metric: 1.0 for metric in PAIR_METRICS},
    }
    assert cli.resume_row_is_complete(row)
    trace.unlink()
    assert not cli.resume_row_is_complete(row)
    trace.write_bytes(b"not-npy")
    assert not cli.resume_row_is_complete(row)


@pytest.mark.parametrize("bad_steps", [None, np.nan, np.inf, "not-a-step"])
def test_resume_row_requires_valid_checkpoint_steps(tmp_path: Path, bad_steps):
    cli = _load_context_cli()
    trace = tmp_path / "trace.npy"
    np.save(trace, np.ones((2, 1), dtype=np.float32))
    row = {
        "seed": 42,
        "task_id": DIAGNOSTIC_TASK_IDS[0],
        "inference_mode": "zero_context",
        "action_trace_path": str(trace),
        **{metric: 1.0 for metric in PAIR_METRICS},
    }
    if bad_steps is not None:
        row["checkpoint_steps"] = bad_steps

    assert not cli.resume_row_is_complete(row, checkpoint_steps=100)


def test_resume_row_must_match_current_checkpoint_steps(tmp_path: Path):
    cli = _load_context_cli()
    trace = tmp_path / "trace.npy"
    np.save(trace, np.ones((2, 1), dtype=np.float32))
    row = {
        "seed": 42,
        "task_id": DIAGNOSTIC_TASK_IDS[0],
        "split": "fixed",
        "inference_mode": "zero_context",
        "checkpoint_steps": 100,
        "action_trace_path": str(trace),
        "support_ready_step": np.nan,
        "context_norm_mean": 0.0,
        "context_norm_max": 0.0,
        "model_sha256": "a" * 64,
        "vecnormalize_sha256": "b" * 64,
        "source_manifest_sha256": "c" * 64,
        "source_tasks_sha256": "d" * 64,
        **{metric: 1.0 for metric in PAIR_METRICS},
    }

    assert cli.resume_row_is_complete(row, checkpoint_steps=100)
    assert not cli.resume_row_is_complete(row, checkpoint_steps=200)


def test_resume_online_row_requires_valid_readiness_and_context_diagnostics(tmp_path: Path):
    cli = _load_context_cli()
    trace = tmp_path / "trace.npy"
    np.save(trace, np.ones((3, 1), dtype=np.float32))
    base = {
        "seed": 42,
        "task_id": DIAGNOSTIC_TASK_IDS[0],
        "split": "fixed",
        "inference_mode": "online_context",
        "checkpoint_steps": 100,
        "action_trace_path": str(trace),
        "support_ready_step": 1.0,
        "context_norm_mean": 1.0,
        "context_norm_max": 2.0,
        "model_sha256": "a" * 64,
        "vecnormalize_sha256": "b" * 64,
        "source_manifest_sha256": "c" * 64,
        "source_tasks_sha256": "d" * 64,
        **{metric: 1.0 for metric in PAIR_METRICS},
    }
    assert cli.resume_row_is_complete(base)
    for name, bad in (("support_ready_step", np.nan), ("support_ready_step", 3),
                      ("context_norm_mean", np.nan), ("context_norm_max", np.inf)):
        row = dict(base)
        row[name] = bad
        assert not cli.resume_row_is_complete(row)


def test_resume_row_rejects_replaced_same_step_inputs(tmp_path: Path):
    cli = _load_context_cli()
    trace = tmp_path / "trace.npy"
    np.save(trace, np.ones((3, 1), dtype=np.float32))
    expected = {
        "model_sha256": "a" * 64,
        "vecnormalize_sha256": "b" * 64,
        "source_manifest_sha256": "c" * 64,
        "source_tasks_sha256": "d" * 64,
    }
    row = {
        "seed": 42, "task_id": DIAGNOSTIC_TASK_IDS[0],
        "split": "fixed",
        "inference_mode": "zero_context", "checkpoint_steps": 100,
        "action_trace_path": str(trace), "support_ready_step": np.nan,
        "context_norm_mean": 0.0, "context_norm_max": 0.0,
        **expected, **{metric: 1.0 for metric in PAIR_METRICS},
    }
    assert cli.resume_row_is_complete(row, checkpoint_steps=100, expected_hashes=expected)
    for name in expected:
        changed = dict(expected)
        changed[name] = "f" * 64
        assert not cli.resume_row_is_complete(
            row, checkpoint_steps=100, expected_hashes=changed
        )


def test_resume_hashes_detect_replaced_model_vec_and_changed_tasks(tmp_path: Path):
    cli = _load_context_cli()
    paths = {
        "model_sha256": tmp_path / "model.zip",
        "vecnormalize_sha256": tmp_path / "vec.pkl",
        "source_manifest_sha256": tmp_path / "manifest.json",
        "source_tasks_sha256": tmp_path / "tasks.csv",
    }
    for path in paths.values():
        path.write_bytes(b"original")
    expected = {name: cli.sha256_file(path) for name, path in paths.items()}
    trace = tmp_path / "trace.npy"
    np.save(trace, np.ones((3, 1), dtype=np.float32))
    row = {
        "seed": 42, "task_id": DIAGNOSTIC_TASK_IDS[0], "split": "fixed",
        "inference_mode": "zero_context", "checkpoint_steps": 100,
        "action_trace_path": str(trace), "support_ready_step": np.nan,
        "context_norm_mean": 0.0, "context_norm_max": 0.0,
        **expected, **{metric: 1.0 for metric in PAIR_METRICS},
    }
    for name, path in paths.items():
        path.write_bytes(f"replaced-{name}".encode())
        changed = dict(expected)
        changed[name] = cli.sha256_file(path)
        assert not cli.resume_row_is_complete(row, expected_hashes=changed)


def test_validated_diagnostics_requires_all_core_diagnostics():
    cli = _load_context_cli()
    with pytest.raises(KeyError, match="support_ready_step"):
        cli._validated_diagnostics(
            {"context_norm_mean": 0.0, "context_norm_max": 0.0},
            metric_names={"episode_return"},
        )


def test_validated_diagnostics_rejects_core_and_metric_collisions():
    cli = _load_context_cli()
    diagnostics = {
        "support_ready_step": None,
        "context_norm_mean": 0.0,
        "context_norm_max": 0.0,
        "seed": 999,
        "episode_return": -1.0,
    }
    with pytest.raises(ValueError, match="episode_return.*seed"):
        cli._validated_diagnostics(
            diagnostics,
            metric_names={"episode_return"},
        )


def test_validated_diagnostics_normalizes_none_and_numpy_scalars():
    cli = _load_context_cli()
    normalized = cli._validated_diagnostics(
        {
            "support_ready_step": None,
            "context_norm_mean": np.float32(1.5),
            "context_norm_max": 2.0,
        },
        metric_names=set(),
    )
    assert np.isnan(normalized["support_ready_step"])
    assert normalized["context_norm_mean"] == pytest.approx(1.5)
    assert isinstance(normalized["context_norm_mean"], float)


def test_cli_core_fake_smoke_writes_exactly_32_rows(tmp_path: Path):
    cli = _load_context_cli()
    _write_diagnostic_checkpoints(tmp_path / "models")
    stale_trace = tmp_path / "diagnostic" / "traces" / "stale.npy"
    stale_trace.parent.mkdir(parents=True)
    np.save(stale_trace, np.zeros((1, 1), dtype=np.float32))
    unrelated = tmp_path / "diagnostic" / "keep-me.txt"
    unrelated.write_text("unrelated", encoding="utf-8")
    source_manifest = tmp_path / "source" / "suite_manifest.json"
    source_tasks_csv = tmp_path / "source" / "eval_tasks.csv"
    source_manifest.parent.mkdir(parents=True)
    source_manifest.write_text("{}", encoding="utf-8")
    source_tasks_csv.write_text("task_id\n", encoding="utf-8")
    suite = SimpleNamespace(suite_id="source", result_root=str(tmp_path / "source"), env_id="Fake")
    split_by_task = {task_id: task_id.split("_", 1)[0] for task_id in DIAGNOSTIC_TASK_IDS}
    tasks = pd.DataFrame(
        [
            {
                "suite_id": "source",
                "task_id": task_id,
                "split": split_by_task[task_id],
                "weather_year": 2010,
                "start_day": 59,
                "uncertainty_scale": 0.0,
                "economic_scenario": "standard",
                "climate_constraint_scenario": "standard",
            }
            for task_id in DIAGNOSTIC_TASK_IDS
        ]
    )

    class Model:
        num_timesteps = 196608

    class Env:
        def close(self):
            pass

    def episode_runner(model, env, *, inference_mode, return_diagnostics):
        value = float(inference_mode == "online_context")
        return (
            {
                "episode_return": 100.0 + value,
                "EPI": 10.0 + value,
                "revenue": 0.0,
                "heat_cost": 0.0,
                "co2_cost": 0.0,
                "elec_cost": 0.0,
                "temp_violation": 10.0 - value,
                "co2_violation": 10.0 - value,
                "rh_violation": 10.0 - value,
                "twb_percent": 0.0,
            },
            {
                "action_trace": np.full((3, 1), value, dtype=np.float32),
                "support_ready_step": 1.0 if value else np.nan,
                "context_norm_mean": value,
                "context_norm_max": value,
                "context_variance": 4.2,
            },
        )

    result = cli.run_diagnostic(
        suite=suite,
        tasks=tasks,
        runs=cli.build_diagnostic_runs(tmp_path / "models", [42, 123]),
        result_root=tmp_path / "diagnostic",
        source_manifest=source_manifest,
        source_tasks_csv=source_tasks_csv,
        device="cpu",
        resume=False,
        model_loader=lambda path, device: Model(),
        env_loader=lambda suite, task, path: Env(),
        episode_runner=episode_runner,
        provenance_loader=lambda: {"git_commit": "abc", "dirty": False},
    )

    assert len(result) == 32
    assert unrelated.read_text(encoding="utf-8") == "unrelated"
    final_raw = pd.read_csv(tmp_path / "diagnostic" / "eval_raw.csv")
    progress_raw = pd.read_csv(
        tmp_path / ".diagnostic.work" / "progress.csv"
    )
    assert len(final_raw) == 32
    assert result["context_variance"].eq(4.2).all()
    assert final_raw["context_variance"].eq(4.2).all()
    assert progress_raw["context_variance"].eq(4.2).all()
    for name in ("model_sha256", "vecnormalize_sha256", "source_manifest_sha256", "source_tasks_sha256"):
        assert progress_raw[name].str.fullmatch(r"[0-9a-f]{64}").all()
    manifest = json.loads(
        (tmp_path / "diagnostic" / "diagnostic_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["source_manifest_sha256"] == cli.sha256_file(source_manifest)
    assert manifest["source_tasks_sha256"] == cli.sha256_file(source_tasks_csv)
    assert len(list((tmp_path / "diagnostic" / "traces").glob("*.npy"))) == 32


def test_failure_capture_builds_per_episode_context_and_does_not_publish(
    tmp_path: Path
):
    case = _fake_cli_case(tmp_path)
    root = tmp_path / "diagnostic"
    failure_root = tmp_path / ".diagnostic.work" / "failures"
    provenance_calls = 0
    created = []

    class Recorder:
        pass

    def recorder_factory(path, context):
        created.append((Path(path), context))
        return Recorder()

    def provenance_loader():
        nonlocal provenance_calls
        provenance_calls += 1
        return {"git_commit": "a" * 40, "dirty": True}

    def fail_first(
        model, env, *, inference_mode, return_diagnostics, failure_recorder
    ):
        assert isinstance(failure_recorder, Recorder)
        raise RuntimeError("injected early failure")

    with pytest.raises(RuntimeError, match="injected early failure"):
        case.cli.run_diagnostic(
            suite=case.suite,
            tasks=case.tasks,
            runs=case.runs,
            result_root=root,
            failure_root=failure_root,
            source_manifest=case.source_manifest,
            source_tasks_csv=case.source_tasks_csv,
            device="cpu",
            resume=False,
            model_loader=case.model_loader,
            env_loader=case.env_loader,
            episode_runner=fail_first,
            recorder_factory=recorder_factory,
            provenance_loader=provenance_loader,
        )

    assert provenance_calls == 1
    assert len(created) == 1
    recorder_path, context = created[0]
    assert recorder_path == failure_root.resolve()
    assert context.seed == 42
    assert context.task_id == DIAGNOSTIC_TASK_IDS[0]
    assert context.inference_mode == MODES[0]
    assert context.task["task_id"] == DIAGNOSTIC_TASK_IDS[0]
    assert Path(context.checkpoint_path).is_absolute()
    assert context.checkpoint_sha256 == case.runs[0]["model_sha256"]
    assert context.git_head == "a" * 40
    assert context.dirty is True
    assert context.formal_result_root == str(root.resolve())
    assert context.package_versions

    expected_sources = {
        (case.cli.ROOT / "src/gl_gym/environments/tomato_env.py").resolve(),
        (case.cli.ROOT / "src/gl_gym/environments/models/ode.py").resolve(),
        (case.cli.ROOT / "src/gl_gym/environments/models/utils.py").resolve(),
        (case.cli.ROOT / "configs/envs/TomatoEnv.yml").resolve(),
        (case.cli.ROOT / "configs/agents/rule_based.yml").resolve(),
        case.source_manifest.resolve(),
        case.source_tasks_csv.resolve(),
    }
    assert {Path(key) for key in context.source_checksums} == expected_sources
    for path in expected_sources:
        assert context.source_checksums[str(path)] == case.cli.sha256_file(path)

    for name in (
        "eval_raw.csv",
        "paired_deltas.csv",
        "split_summary.csv",
        "diagnostic_manifest.json",
        "decision.json",
    ):
        assert not (root / name).exists()


@pytest.mark.parametrize("failure_stage", ["mid_run", "final_publish"])
def test_diagnostic_failure_preserves_published_root_and_resumable_work(
    tmp_path: Path, monkeypatch, failure_stage: str
):
    case = _fake_cli_case(tmp_path)
    root = tmp_path / "diagnostic"
    (root / "traces").mkdir(parents=True)
    (root / "published.txt").write_bytes(b"published-before")
    np.save(root / "traces" / "old.npy", np.ones((2, 1)))
    before = _tree_bytes(root)
    episode_runner = case.episode_runner
    if failure_stage == "mid_run":
        calls = 0

        def fail_second(*args, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise RuntimeError("injected mid-run failure")
            return episode_runner(*args, **kwargs)

        selected_runner = fail_second
        expected = "mid-run"
    else:
        selected_runner = episode_runner
        real_replace = context_ab.os.replace

        def fail_publish(source, destination):
            if Path(source).name.startswith(".diagnostic.staging-"):
                raise OSError("injected final publish failure")
            return real_replace(source, destination)

        monkeypatch.setattr(context_ab.os, "replace", fail_publish)
        expected = "final publish"

    with pytest.raises((RuntimeError, OSError), match=expected):
        case.cli.run_diagnostic(
            suite=case.suite, tasks=case.tasks, runs=case.runs,
            result_root=root, source_manifest=case.source_manifest,
            source_tasks_csv=case.source_tasks_csv, device="cpu", resume=False,
            model_loader=case.model_loader, env_loader=case.env_loader,
            episode_runner=selected_runner,
            provenance_loader=lambda: {"git_commit": "abc", "dirty": False},
        )

    assert _tree_bytes(root) == before
    work = tmp_path / ".diagnostic.work"
    progress = pd.read_csv(work / "progress.csv")
    assert len(progress) == (1 if failure_stage == "mid_run" else 32)
    assert progress["action_trace_path"].map(lambda path: Path(path).is_file()).all()
