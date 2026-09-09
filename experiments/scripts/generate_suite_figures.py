#!/usr/bin/env python3
"""Generate paper figures from current robust experiment suite summaries."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from gl_gym.experiments.suite_schema import load_suite_manifest  # noqa: E402
from gl_gym.experiments.suite_validation import require_current_suite_id  # noqa: E402


LEARNING_METHODS = (
    "ppo",
    "recurrentppo",
    "context_recurrentppo",
    "agri_metarl",
)


def plot_fixed_heldout(summary: pd.DataFrame, out_path: str | Path) -> Path:
    """Plot fixed-vs-heldout return means for the learning methods."""

    if "episode_return_mean" not in summary.columns:
        raise ValueError("missing episode_return_mean column")

    present_algorithms = set(summary["algorithm"].dropna().astype(str))
    missing_algorithms = sorted(set(LEARNING_METHODS) - present_algorithms)
    if missing_algorithms:
        raise ValueError(f"missing algorithms in method summary: {missing_algorithms}")

    data = summary[summary["split"].isin(["fixed", "heldout"])].copy()
    pivot = data.pivot_table(
        index="algorithm",
        columns="split",
        values="episode_return_mean",
        aggfunc="mean",
    ).reindex(LEARNING_METHODS)

    missing_splits = [split for split in ["fixed", "heldout"] if split not in pivot.columns]
    if missing_splits:
        raise ValueError(f"missing splits in method summary: {missing_splits}")

    missing_algorithm_splits = [
        f"{algorithm}/{split}"
        for algorithm in LEARNING_METHODS
        for split in ["fixed", "heldout"]
        if pd.isna(pivot.loc[algorithm, split])
    ]
    if missing_algorithm_splits:
        raise ValueError(
            f"missing algorithm/split fixed/heldout values in method summary: "
            f"{missing_algorithm_splits}"
        )

    output = Path(out_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    ax = pivot[["fixed", "heldout"]].plot(kind="bar", figsize=(8, 4.8))
    ax.set_xlabel("Algorithm")
    ax.set_ylabel("Episode return mean")
    ax.set_title("Fixed protocol vs heldout task performance")
    ax.legend(title="Split")
    ax.figure.tight_layout()
    ax.figure.savefig(output)
    plt.close(ax.figure)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate robust experiment suite paper figures.",
    )
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--summary", default=None)
    parser.add_argument("--out_dir", default=Path("paper") / "figures", type=Path)
    args = parser.parse_args()

    suite = load_suite_manifest(args.manifest)
    summary_path = Path(args.summary) if args.summary else Path(suite.result_root) / "method_summary.csv"
    summary = require_current_suite_id(summary_path, suite.suite_id)
    out_path = plot_fixed_heldout(summary, args.out_dir / "suite_fixed_heldout_returns.png")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
