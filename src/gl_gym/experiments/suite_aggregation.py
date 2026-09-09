"""Aggregation helpers for robust experiment suite evaluation outputs."""

from __future__ import annotations

from itertools import combinations
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


DEFAULT_METRICS = (
    "episode_return",
    "EPI",
    "revenue",
    "heat_cost",
    "co2_cost",
    "elec_cost",
    "temp_violation",
    "co2_violation",
    "rh_violation",
)


def assert_no_duplicate_run_task_rows(df: pd.DataFrame) -> None:
    """Reject repeated deterministic evaluation rows for the same run and task."""

    key = ["algorithm", "seed", "run_name", "task_id"]
    missing = [column for column in key if column not in df.columns]
    if missing:
        raise ValueError(f"missing duplicate-check columns: {missing}")

    duplicate_mask = df.duplicated(subset=key, keep=False)
    if not duplicate_mask.any():
        return

    sample = df.loc[duplicate_mask, key].head(10).to_dict(orient="records")
    raise ValueError(f"duplicate deterministic evaluation rows detected: {sample}")


def _suite_group_columns(df: pd.DataFrame) -> list[str]:
    return ["suite_id"] if "suite_id" in df.columns else []


def _present_metrics(df: pd.DataFrame, metrics: Iterable[str] | None) -> list[str]:
    requested = list(metrics) if metrics is not None else list(DEFAULT_METRICS)
    return [metric for metric in requested if metric in df.columns]


def aggregate_seed_first(
    df: pd.DataFrame,
    metrics: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Average tasks within each seed before computing across-seed statistics."""

    assert_no_duplicate_run_task_rows(df)
    metric_columns = _present_metrics(df, metrics)
    if not metric_columns:
        raise ValueError("aggregate_seed_first requires at least one metric column")

    suite_columns = _suite_group_columns(df)
    seed_group = suite_columns + ["algorithm", "seed", "split"]
    method_group = suite_columns + ["algorithm", "split"]

    seed_means = (
        df.groupby(seed_group, dropna=False, sort=True)[metric_columns]
        .mean()
        .reset_index()
    )
    method_groups = seed_means.groupby(method_group, dropna=False, sort=True)

    summary = method_groups["seed"].nunique().reset_index(name="n_seeds")
    for metric in metric_columns:
        stats = (
            method_groups[metric]
            .agg(
                **{
                    f"{metric}_mean": "mean",
                    f"{metric}_std": lambda values: (
                        float(values.std(ddof=1)) if len(values) > 1 else float("nan")
                    ),
                }
            )
            .reset_index()
        )
        summary = summary.merge(stats, on=method_group, how="left")

    return summary


def cohens_d(a: Iterable[float], b: Iterable[float]) -> float:
    """Return signed Cohen's d for two samples."""

    a_array = np.asarray(list(a), dtype=float)
    b_array = np.asarray(list(b), dtype=float)
    a_array = a_array[~np.isnan(a_array)]
    b_array = b_array[~np.isnan(b_array)]

    if len(a_array) < 2 or len(b_array) < 2:
        return 0.0

    a_var = float(a_array.var(ddof=1))
    b_var = float(b_array.var(ddof=1))
    pooled_var = ((len(a_array) - 1) * a_var + (len(b_array) - 1) * b_var) / (
        len(a_array) + len(b_array) - 2
    )
    if pooled_var <= 0:
        return 0.0

    return float((a_array.mean() - b_array.mean()) / np.sqrt(pooled_var))


def paired_effect_table(
    df: pd.DataFrame,
    metric: str = "episode_return",
) -> pd.DataFrame:
    """Compute method-pair effect sizes from seed-level suite summaries."""

    if metric not in df.columns:
        raise KeyError(f"missing effect-size metric column: {metric}")

    suite_columns = _suite_group_columns(df)
    seed_group = suite_columns + ["algorithm", "seed", "split"]
    split_group = suite_columns + ["split"]
    seed_means = df.groupby(seed_group, dropna=False, sort=True)[metric].mean().reset_index()

    rows: list[dict[str, object]] = []
    for group_values, split_df in seed_means.groupby(split_group, dropna=False, sort=True):
        if len(split_group) == 1:
            group_values = (group_values,)
        group_context = dict(zip(split_group, group_values, strict=True))
        methods = sorted(split_df["algorithm"].dropna().unique())
        for method_a, method_b in combinations(methods, 2):
            a_values = split_df[split_df["algorithm"] == method_a][["seed", metric]]
            b_values = split_df[split_df["algorithm"] == method_b][["seed", metric]]
            paired = a_values.merge(
                b_values,
                on="seed",
                how="inner",
                suffixes=("_a", "_b"),
            )
            a_metric = paired[f"{metric}_a"]
            b_metric = paired[f"{metric}_b"]

            rows.append(
                {
                    **group_context,
                    "metric": metric,
                    "method_a": method_a,
                    "method_b": method_b,
                    "cohens_d_a_minus_b": cohens_d(a_metric, b_metric),
                    "n_pairs": int(len(paired)),
                    "mean_a": float(a_metric.mean()) if len(a_metric) else np.nan,
                    "mean_b": float(b_metric.mean()) if len(b_metric) else np.nan,
                }
            )

    return pd.DataFrame(
        rows,
        columns=suite_columns
        + [
            "split",
            "metric",
            "method_a",
            "method_b",
            "cohens_d_a_minus_b",
            "n_pairs",
            "mean_a",
            "mean_b",
        ],
    )


def write_summary_files(eval_raw_path: str | Path, result_root: str | Path) -> dict[str, Path]:
    """Write method summaries, effect sizes, and diagnostics for an evaluation CSV."""

    result_dir = Path(result_root)
    result_dir.mkdir(parents=True, exist_ok=True)

    eval_raw = pd.read_csv(eval_raw_path)
    method_summary = aggregate_seed_first(eval_raw)
    stat_tests = paired_effect_table(eval_raw)
    diagnostics = pd.DataFrame(
        [
            {
                "diagnostic": "not_collected",
                "value": np.nan,
                "notes": "diagnostic hooks are added after pilot validation",
            }
        ]
    )

    paths = {
        "method_summary": result_dir / "method_summary.csv",
        "stat_tests": result_dir / "stat_tests.csv",
        "diagnostics": result_dir / "diagnostics.csv",
    }
    method_summary.to_csv(paths["method_summary"], index=False)
    stat_tests.to_csv(paths["stat_tests"], index=False)
    diagnostics.to_csv(paths["diagnostics"], index=False)
    return paths
