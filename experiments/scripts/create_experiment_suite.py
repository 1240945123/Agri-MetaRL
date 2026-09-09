"""Create the manifest and task records for a robust experiment suite."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from gl_gym.experiments.suite_schema import (  # noqa: E402
    create_default_suite_config,
    write_records_csv,
    write_suite_manifest,
)
from gl_gym.experiments.suite_tasks import build_evaluation_tasks  # noqa: E402


def _git_output(args: list[str]) -> str:
    try:
        return subprocess.check_output(
            ["git", *args],
            cwd=ROOT,
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return ""


def _git_branch() -> str:
    return _git_output(["branch", "--show-current"]) or "unknown"


def _git_dirty() -> bool:
    return bool(_git_output(["status", "--porcelain"]))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a robust experiment suite manifest and evaluation task CSV.",
    )
    parser.add_argument("--suite_id", default="AgriControl_C_2026-06-30")
    parser.add_argument("--result_root", type=Path, default=None)
    parser.add_argument("--model_root", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    suite = create_default_suite_config(
        suite_id=args.suite_id,
        result_root=args.result_root,
        model_root=args.model_root,
        branch=_git_branch(),
        dirty=_git_dirty(),
    )

    manifest_path = write_suite_manifest(suite)
    eval_tasks_path = write_records_csv(
        build_evaluation_tasks(suite),
        Path(suite.result_root) / "eval_tasks.csv",
    )

    print(f"manifest: {manifest_path}")
    print(f"eval_tasks: {eval_tasks_path}")


if __name__ == "__main__":
    main()
