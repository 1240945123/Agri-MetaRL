#!/usr/bin/env python3
"""Summarize robust experiment suite deterministic evaluation outputs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from gl_gym.experiments.suite_aggregation import write_summary_files
from gl_gym.experiments.suite_schema import load_suite_manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--eval_raw")
    args = parser.parse_args()

    suite = load_suite_manifest(args.manifest)
    eval_raw = Path(args.eval_raw) if args.eval_raw else Path(suite.result_root) / "eval_raw.csv"
    paths = write_summary_files(eval_raw, suite.result_root)

    for path in paths.values():
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
