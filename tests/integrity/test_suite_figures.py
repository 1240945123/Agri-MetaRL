from pathlib import Path

import pandas as pd
import pytest

from experiments.scripts.generate_suite_figures import plot_fixed_heldout
from gl_gym.experiments.suite_validation import require_current_suite_id


def test_require_current_suite_id_rejects_stale_summary(tmp_path: Path):
    summary = tmp_path / "method_summary.csv"
    pd.DataFrame(
        [
            {
                "suite_id": "old_suite",
                "algorithm": "ppo",
                "split": "fixed",
                "episode_return_mean": 1.0,
            }
        ]
    ).to_csv(summary, index=False)

    with pytest.raises(ValueError, match="stale suite id"):
        require_current_suite_id(summary, "current_suite")


def test_require_current_suite_id_rejects_blank_suite_id_rows(tmp_path: Path):
    summary = tmp_path / "method_summary.csv"
    pd.DataFrame(
        [
            {
                "suite_id": "current_suite",
                "algorithm": "ppo",
                "split": "fixed",
                "episode_return_mean": 1.0,
            },
            {
                "suite_id": None,
                "algorithm": "agri_metarl",
                "split": "heldout",
                "episode_return_mean": 2.0,
            },
        ]
    ).to_csv(summary, index=False)

    with pytest.raises(ValueError, match="stale suite id|invalid suite_id"):
        require_current_suite_id(summary, "current_suite")


def test_plot_fixed_heldout_rejects_missing_learning_algorithms(tmp_path: Path):
    summary = pd.DataFrame(
        [
            {
                "suite_id": "suite",
                "algorithm": "ppo",
                "split": "fixed",
                "episode_return_mean": 1.0,
            },
            {
                "suite_id": "suite",
                "algorithm": "ppo",
                "split": "heldout",
                "episode_return_mean": 2.0,
            },
        ]
    )

    with pytest.raises(ValueError, match="missing algorithms"):
        plot_fixed_heldout(summary, tmp_path / "figure.png")


def test_plot_fixed_heldout_rejects_missing_split_per_algorithm(tmp_path: Path):
    rows = []
    for algorithm in ["ppo", "recurrentppo", "context_recurrentppo", "agri_metarl"]:
        rows.append(
            {
                "suite_id": "suite",
                "algorithm": algorithm,
                "split": "fixed",
                "episode_return_mean": 1.0,
            }
        )
        if algorithm != "agri_metarl":
            rows.append(
                {
                    "suite_id": "suite",
                    "algorithm": algorithm,
                    "split": "heldout",
                    "episode_return_mean": 2.0,
                }
            )
    summary = pd.DataFrame(rows)

    with pytest.raises(ValueError, match="missing algorithm/split|missing fixed/heldout"):
        plot_fixed_heldout(summary, tmp_path / "figure.png")
