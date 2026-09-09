#!/usr/bin/env python3
"""Validate robust experiment suite artifacts before paper use."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from gl_gym.experiments.suite_validation import validate_suite_artifacts  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate robust experiment suite result artifacts.",
    )
    parser.add_argument("--manifest", required=True)
    args = parser.parse_args()

    validate_suite_artifacts(args.manifest)
    print("Suite artifacts validated.")


if __name__ == "__main__":
    main()
