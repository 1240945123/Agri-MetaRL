from pathlib import Path
import importlib.util
from types import SimpleNamespace

import pytest


def _module():
    path = Path(__file__).resolve().parents[2] / "experiments/scripts/evaluate_suite.py"
    spec = importlib.util.spec_from_file_location("evaluate_suite_cli", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_parser_keeps_legacy_arguments_and_adds_opt_in_shield():
    cli = _module()
    args = cli.build_parser().parse_args(
        ["--manifest", "m", "--runs_csv", "r", "--tasks_csv", "t"]
    )
    assert args.action_shield is False
    assert args.stage2_decision is None
    assert args.result_root is None
    assert args.interventions_out is None


def test_shield_arguments_are_all_or_nothing():
    cli = _module()
    parser = cli.build_parser()
    args = parser.parse_args(
        ["--manifest", "m", "--runs_csv", "r", "--tasks_csv", "t", "--action_shield"]
    )
    with pytest.raises(ValueError, match="stage2_decision.*result_root"):
        cli.validate_cli_mode(args)


def test_environment_close_error_is_not_allowed_to_replace_episode_error():
    cli = _module()

    class Env:
        def close(self):
            raise RuntimeError("close failed")

    primary = RuntimeError("episode failed")
    cli.close_environment(Env(), primary)
    assert any("close failed" in note for note in primary.__notes__)

    with pytest.raises(RuntimeError, match="close failed"):
        cli.close_environment(Env(), None)
