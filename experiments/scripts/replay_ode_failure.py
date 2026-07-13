"""Replay one validated ODE failure capsule into an isolated atomic report."""

from __future__ import annotations

import argparse
import ctypes
from datetime import datetime, timedelta, timezone
import errno
import json
import math
import os
from pathlib import Path
import re
import shutil
import stat
import sys
from typing import Any, Callable, Mapping
from uuid import uuid4

import numpy as np

from gl_gym.experiments.ode_failure import LoadedFailureCapsule, load_failure_capsule
from gl_gym.experiments.ode_replay import (
    ReplayOutcome,
    ReplayReport,
    VARIANT_NAMES,
    classify_replay_outcomes,
    replay_failure_capsule,
)


SCHEMA_VERSION = "ode-replay-report-v1"
OUTPUT_NAMES = frozenset(
    {"replay_results.json", "replay_states.npz", "replay_summary.md"}
)
JSON_TOP_LEVEL_KEYS = frozenset(
    {
        "schema_version",
        "generated_at_utc",
        "failure_id",
        "capsule_identity_sha256",
        "classification",
        "outcomes",
    }
)
JSON_OUTCOME_KEYS = frozenset(
    {
        "variant",
        "configuration",
        "available",
        "success",
        "elapsed_seconds",
        "warnings",
        "exception_type",
        "exception_message",
    }
)
CLASSIFICATIONS = frozenset(
    {
        "policy_induced_control_instability",
        "solver_step_sensitivity",
        "state_or_model_domain_failure",
        "mixed_control_and_solver_sensitivity",
        "insufficient_counterfactual_evidence",
        "non_reproduced",
    }
)
VARIANT_CONFIGURATIONS: dict[str, dict[str, str | int]] = {
    "original": {
        "control_source": "stored_executed_control",
        "substeps": 1,
        "tolerance_mode": "formal",
    },
    "previous_control": {
        "control_source": "stored_previous_control",
        "substeps": 1,
        "tolerance_mode": "formal",
    },
    "rule_based_control": {
        "control_source": "deterministic_rule_based_controller",
        "substeps": 1,
        "tolerance_mode": "formal",
    },
    "original_2x_substeps": {
        "control_source": "stored_executed_control",
        "substeps": 2,
        "tolerance_mode": "formal",
    },
    "original_4x_substeps": {
        "control_source": "stored_executed_control",
        "substeps": 4,
        "tolerance_mode": "formal",
    },
    "original_strict_tolerance": {
        "control_source": "stored_executed_control",
        "substeps": 1,
        "tolerance_mode": "strict_1e-6",
    },
}
NEXT_ACTIONS = {
    "policy_induced_control_instability": (
        "design/evaluate principled action safety layer and return/constraint effects"
    ),
    "mixed_control_and_solver_sensitivity": (
        "design/evaluate principled action safety layer and return/constraint effects"
    ),
    "solver_step_sensitivity": (
        "fixed-controller integration-scheme benchmark redesign/version"
    ),
    "state_or_model_domain_failure": (
        "audit physical bounds and implicated ODE/Jacobian terms"
    ),
    "non_reproduced": (
        "investigate environment/software nondeterminism before causal change"
    ),
    "insufficient_counterfactual_evidence": ("acquire missing counterfactual evidence"),
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capsule", required=True)
    parser.add_argument("--output_root", required=True)
    return parser


def _overlaps(first: Path, second: Path) -> bool:
    return first == second or first in second.parents or second in first.parents


def _formal_result_root(manifest: Mapping[str, Any]) -> Path:
    context = manifest.get("context")
    if not isinstance(context, Mapping):
        raise ValueError("capsule manifest context is missing")
    value = context.get("formal_result_root")
    if not isinstance(value, (str, os.PathLike)) or not str(value).strip():
        raise ValueError("capsule manifest formal result root is missing")
    return Path(value).expanduser().resolve()


def validate_isolated_output_root(
    output_root: str | Path, manifest: Mapping[str, Any]
) -> Path:
    """Resolve an output path and reject any overlap with the formal result tree."""
    output = Path(output_root).expanduser().resolve()
    formal = _formal_result_root(manifest)
    if _overlaps(output, formal):
        raise ValueError("replay output must be disjoint from the formal result root")
    return output


def _validate_report(report: ReplayReport, state_dim: int | None) -> int:
    if not isinstance(report.failure_id, str) or not report.failure_id:
        raise ValueError("replay report failure_id must be non-empty text")
    if report.classification not in CLASSIFICATIONS:
        raise ValueError("unsupported replay classification")
    outcomes = tuple(report.outcomes)
    if tuple(item.variant for item in outcomes) != VARIANT_NAMES:
        raise ValueError("replay report outcomes must use the fixed variant order")
    derived_classification = classify_replay_outcomes(outcomes)
    if report.classification != derived_classification:
        raise ValueError("replay report classification is inconsistent with outcomes")

    inferred_dims: set[int] = set()
    for outcome in outcomes:
        if not isinstance(outcome.available, bool) or not isinstance(
            outcome.success, bool
        ):
            raise ValueError("replay availability and success must be boolean")
        elapsed = float(outcome.elapsed_seconds)
        if not math.isfinite(elapsed) or elapsed < 0:
            raise ValueError("replay elapsed time must be finite and non-negative")
        if outcome.success:
            state = np.asarray(outcome.final_state)
            if (
                state.ndim != 1
                or state.dtype.kind not in "biuf"
                or not np.isfinite(state).all()
            ):
                raise ValueError(
                    "successful replay final states must be finite numeric vectors"
                )
            inferred_dims.add(int(state.size))
        elif outcome.final_state is not None:
            raise ValueError("failed replay outcomes cannot contain a final state")
    if len(inferred_dims) > 1:
        raise ValueError("successful replay final states have inconsistent dimensions")
    inferred = next(iter(inferred_dims), None)
    if state_dim is None:
        if inferred is None:
            raise ValueError("state_dim is required when no replay variant succeeds")
        state_dim = inferred
    if (
        isinstance(state_dim, bool)
        or not isinstance(state_dim, (int, np.integer))
        or int(state_dim) <= 0
    ):
        raise ValueError("state_dim must be a positive integer")
    if inferred is not None and inferred != int(state_dim):
        raise ValueError(
            "successful replay final state dimension does not match state_dim"
        )
    return int(state_dim)


def _json_payload(
    report: ReplayReport,
    capsule_identity: str | None,
) -> dict[str, Any]:
    outcomes = []
    for outcome in report.outcomes:
        outcomes.append(
            {
                "variant": outcome.variant,
                "configuration": dict(VARIANT_CONFIGURATIONS[outcome.variant]),
                "available": outcome.available,
                "success": outcome.success,
                "elapsed_seconds": float(outcome.elapsed_seconds),
                "warnings": [str(message) for message in outcome.warnings],
                "exception_type": (
                    None
                    if outcome.exception_type is None
                    else str(outcome.exception_type)
                ),
                "exception_message": (
                    None
                    if outcome.exception_message is None
                    else str(outcome.exception_message)
                ),
            }
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": datetime.now(timezone.utc)
        .isoformat()
        .replace("+00:00", "Z"),
        "failure_id": report.failure_id,
        "capsule_identity_sha256": capsule_identity,
        "classification": report.classification,
        "outcomes": outcomes,
    }


def _state_arrays(
    report: ReplayReport, state_dim: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    successful = [outcome for outcome in report.outcomes if outcome.success]
    width = max((len(outcome.variant) for outcome in successful), default=1)
    names = np.asarray([outcome.variant for outcome in successful], dtype=f"<U{width}")
    if successful:
        states = np.stack(
            [
                np.asarray(outcome.final_state, dtype=np.float64)
                for outcome in successful
            ]
        )
    else:
        states = np.empty((0, state_dim), dtype=np.float64)
    masks = np.isfinite(states).astype(np.bool_, copy=False)
    return names, states, masks


def _markdown_from_payload(payload: Mapping[str, Any]) -> str:
    lines = [
        "# ODE Failure Replay Summary",
        "",
        f"Failure: `{payload['failure_id']}`",
        "",
        f"Classification: `{payload['classification']}`",
        "",
        "| Variant | Available | Success | Elapsed (s) | Exception |",
        "| --- | --- | --- | ---: | --- |",
    ]
    for outcome in payload["outcomes"]:
        exception = ""
        if outcome["exception_type"] is not None:
            exception = str(outcome["exception_type"])
            if outcome["exception_message"]:
                exception += f": {outcome['exception_message']}"
        exception = exception.replace("|", "\\|").replace("\n", " ")
        lines.append(
            f"| {outcome['variant']} | {str(outcome['available']).lower()} | "
            f"{str(outcome['success']).lower()} | "
            f"{outcome['elapsed_seconds']:.6f} | {exception} |"
        )
    lines.extend(
        [
            "",
            "## Next action",
            "",
            NEXT_ACTIONS[payload["classification"]],
            "",
        ]
    )
    return "\n".join(lines)


def _write_outputs(
    report: ReplayReport,
    directory: Path,
    *,
    state_dim: int,
    capsule_identity: str | None,
) -> None:
    payload = _json_payload(report, capsule_identity)
    (directory / "replay_results.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    names, states, masks = _state_arrays(report, state_dim)
    np.savez(
        directory / "replay_states.npz",
        variant_names=names,
        final_states=states,
        finite_masks=masks,
    )
    (directory / "replay_summary.md").write_bytes(
        _markdown_from_payload(payload).encode("utf-8")
    )


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"invalid JSON number: {value}")


def _is_regular_nonreparse(path: Path) -> bool:
    metadata = path.lstat()
    reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    attributes = getattr(metadata, "st_file_attributes", 0)
    return (
        not path.is_symlink()
        and not bool(attributes & reparse_flag)
        and stat.S_ISREG(metadata.st_mode)
    )


def _is_directory_nonreparse(path: Path) -> bool:
    metadata = path.lstat()
    reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    attributes = getattr(metadata, "st_file_attributes", 0)
    return (
        not path.is_symlink()
        and not bool(attributes & reparse_flag)
        and stat.S_ISDIR(metadata.st_mode)
    )


def _parse_aware_timestamp(value: Any) -> datetime:
    if not isinstance(value, str) or not value:
        raise ValueError("generated replay timestamp must be non-empty text")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise ValueError("generated replay timestamp must be ISO-8601") from error
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("generated replay timestamp must be timezone-aware")
    if parsed.utcoffset() != timedelta(0):
        raise ValueError("generated replay timestamp must use UTC")
    return parsed


def _is_canonical_sha256(value: Any) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _validate_json_payload(payload: Any) -> list[str]:
    if not isinstance(payload, dict) or set(payload) != JSON_TOP_LEVEL_KEYS:
        raise ValueError("replay report must have the exact top-level key set")
    if payload["schema_version"] != SCHEMA_VERSION:
        raise ValueError("invalid replay report schema")
    if not isinstance(payload["failure_id"], str) or not payload["failure_id"]:
        raise ValueError("replay failure_id must be non-empty text")
    if not _is_canonical_sha256(payload["capsule_identity_sha256"]):
        raise ValueError("replay capsule identity must be lowercase SHA-256 hex")
    if payload["classification"] not in CLASSIFICATIONS:
        raise ValueError("invalid replay classification")
    _parse_aware_timestamp(payload["generated_at_utc"])

    outcomes = payload["outcomes"]
    if not isinstance(outcomes, list) or len(outcomes) != len(VARIANT_NAMES):
        raise ValueError("replay report must contain exactly six outcomes")
    successful_names: list[str] = []
    replay_outcomes: list[ReplayOutcome] = []
    for expected_variant, item in zip(VARIANT_NAMES, outcomes):
        if not isinstance(item, dict) or set(item) != JSON_OUTCOME_KEYS:
            raise ValueError("replay outcome must have the exact key set")
        if item["variant"] != expected_variant:
            raise ValueError("invalid replay outcome order")
        if item["configuration"] != VARIANT_CONFIGURATIONS[expected_variant]:
            raise ValueError("invalid replay variant configuration")
        available = item["available"]
        success = item["success"]
        if type(available) is not bool or type(success) is not bool:
            raise ValueError("replay availability and success must be boolean")
        elapsed = item["elapsed_seconds"]
        if type(elapsed) is not float or not math.isfinite(elapsed) or elapsed < 0:
            raise ValueError("replay elapsed time must be a finite non-negative float")
        warning_messages = item["warnings"]
        if not isinstance(warning_messages, list) or not all(
            isinstance(message, str) for message in warning_messages
        ):
            raise ValueError("replay warnings must be a list of strings")
        exception_type = item["exception_type"]
        exception_message = item["exception_message"]
        if exception_type is not None and not isinstance(exception_type, str):
            raise ValueError("replay exception_type must be text or null")
        if exception_message is not None and not isinstance(exception_message, str):
            raise ValueError("replay exception_message must be text or null")
        if success:
            if not available:
                raise ValueError("successful replay outcomes must be available")
            if exception_type is not None or exception_message is not None:
                raise ValueError(
                    "successful replay outcomes cannot contain an exception"
                )
            successful_names.append(expected_variant)
        elif exception_type is None or exception_message is None:
            raise ValueError("failed replay outcomes must contain exception metadata")
        replay_outcomes.append(
            ReplayOutcome(
                variant=expected_variant,
                available=available,
                success=success,
                elapsed_seconds=elapsed,
                final_state=np.zeros(1, dtype=np.float64) if success else None,
                exception_type=exception_type,
                exception_message=exception_message,
                warnings=tuple(warning_messages),
            )
        )
    derived_classification = classify_replay_outcomes(replay_outcomes)
    if payload["classification"] != derived_classification:
        raise ValueError("replay classification is inconsistent with outcomes")
    return successful_names


def _validate_replay_directory(directory: Path, *, state_dim: int) -> None:
    if not directory.exists() or not _is_directory_nonreparse(directory):
        raise ValueError("replay output must be a regular directory")
    entries = list(directory.iterdir())
    names = {entry.name for entry in entries}
    if names != OUTPUT_NAMES:
        unexpected = sorted(names - OUTPUT_NAMES)
        missing = sorted(OUTPUT_NAMES - names)
        raise ValueError(
            f"unexpected replay output entries: extra={unexpected}, missing={missing}"
        )
    for entry in entries:
        if not _is_regular_nonreparse(entry):
            raise ValueError(
                f"replay output must be a regular non-symlink file: {entry.name}"
            )

    try:
        payload = json.loads(
            (directory / "replay_results.json").read_text(encoding="utf-8"),
            parse_constant=_reject_json_constant,
        )
    except (UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid replay_results.json: {error}") from error
    successful_names = _validate_json_payload(payload)
    try:
        markdown = (
            (directory / "replay_summary.md")
            .read_bytes()
            .decode("utf-8", errors="strict")
        )
    except UnicodeDecodeError as error:
        raise ValueError("replay Markdown must be valid UTF-8") from error
    if markdown != _markdown_from_payload(payload):
        raise ValueError("replay Markdown does not exactly match validated JSON")

    try:
        with np.load(directory / "replay_states.npz", allow_pickle=False) as archive:
            if set(archive.files) != {"variant_names", "final_states", "finite_masks"}:
                raise ValueError("invalid replay state archive fields")
            variant_names = np.array(archive["variant_names"], copy=True)
            final_states = np.array(archive["final_states"], copy=True)
            finite_masks = np.array(archive["finite_masks"], copy=True)
    except (OSError, ValueError) as error:
        raise ValueError(f"invalid replay state archive: {error}") from error
    if variant_names.dtype.kind != "U" or variant_names.ndim != 1:
        raise ValueError("replay variant names must be fixed-width Unicode")
    expected_shape = (len(successful_names), state_dim)
    if final_states.dtype.kind not in "fiu" or final_states.shape != expected_shape:
        raise ValueError("replay final states have invalid dtype or shape")
    if finite_masks.dtype != np.dtype(bool) or finite_masks.shape != expected_shape:
        raise ValueError("replay finite masks have invalid dtype or shape")
    if variant_names.tolist() != successful_names:
        raise ValueError("replay state variants do not match JSON outcomes")
    if (
        not np.array_equal(finite_masks, np.isfinite(final_states))
        or not finite_masks.all()
    ):
        raise ValueError("replay final states contain non-finite values")


def _publication_race_hook(_final: Path) -> None:
    """Test seam immediately before the no-clobber publication operation."""


def _load_libc() -> ctypes.CDLL:
    return ctypes.CDLL(None, use_errno=True)


def _raise_atomic_rename_error(error_number: int, destination: Path) -> None:
    if error_number in (errno.EEXIST, errno.ENOTEMPTY):
        raise FileExistsError(
            error_number,
            os.strerror(error_number),
            str(destination),
        )
    unsupported = {errno.ENOSYS, errno.EINVAL}
    for name in ("ENOTSUP", "EOPNOTSUPP"):
        value = getattr(errno, name, None)
        if value is not None:
            unsupported.add(value)
    if error_number in unsupported:
        raise RuntimeError(
            "atomic no-replace rename is unsupported by this kernel/filesystem"
        ) from OSError(error_number, os.strerror(error_number), str(destination))
    raise OSError(error_number, os.strerror(error_number), str(destination))


def _atomic_rename_noreplace(source: Path, destination: Path) -> None:
    """Atomically rename a directory, failing closed if no no-clobber API exists."""
    if sys.platform == "win32":
        os.rename(source, destination)
        return
    if not (sys.platform.startswith("linux") or sys.platform == "darwin"):
        raise RuntimeError(
            f"atomic no-replace rename is unsupported on platform {sys.platform!r}"
        )

    libc = _load_libc()
    source_bytes = os.fsencode(os.fspath(source))
    destination_bytes = os.fsencode(os.fspath(destination))
    if sys.platform.startswith("linux"):
        renameat2 = getattr(libc, "renameat2", None)
        if renameat2 is None:
            raise RuntimeError(
                "atomic no-replace rename is unsupported: libc has no renameat2"
            )
        renameat2.argtypes = [
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        ]
        renameat2.restype = ctypes.c_int
        ctypes.set_errno(0)
        result = renameat2(
            -100,
            source_bytes,
            -100,
            destination_bytes,
            1,
        )
    elif sys.platform == "darwin":
        renamex_np = getattr(libc, "renamex_np", None)
        if renamex_np is None:
            raise RuntimeError(
                "atomic no-replace rename is unsupported: libc has no renamex_np"
            )
        renamex_np.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint]
        renamex_np.restype = ctypes.c_int
        ctypes.set_errno(0)
        result = renamex_np(source_bytes, destination_bytes, 0x00000004)
    if result != 0:
        _raise_atomic_rename_error(ctypes.get_errno(), destination)


def _publish_directory_no_clobber(temporary: Path, output: Path) -> None:
    """Publish with one atomic no-replace operation after the test race seam."""
    _publication_race_hook(output)
    _atomic_rename_noreplace(temporary, output)


def write_replay_report_atomic(
    report: ReplayReport,
    output: str | Path,
    *,
    state_dim: int | None = None,
    capsule_identity: str | None = None,
) -> Path:
    """Write, strict-validate, and atomically publish exactly three report files."""
    output_path = Path(output).expanduser().resolve()
    if output_path.exists() or output_path.is_symlink():
        raise FileExistsError(f"replay output already exists: {output_path}")
    resolved_dim = _validate_report(report, state_dim)
    if not _is_canonical_sha256(capsule_identity):
        raise ValueError("capsule identity must be lowercase SHA-256 hex")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.parent / f".ode-replay-{uuid4().hex[:12]}"
    temporary.mkdir()
    published = False
    try:
        _write_outputs(
            report,
            temporary,
            state_dim=resolved_dim,
            capsule_identity=capsule_identity,
        )
        _validate_replay_directory(temporary, state_dim=resolved_dim)
        _publish_directory_no_clobber(temporary, output_path)
        published = True
        return output_path
    except BaseException as primary:
        if not published and temporary.exists():
            try:
                shutil.rmtree(temporary)
            except BaseException as cleanup_error:
                if hasattr(primary, "add_note"):
                    primary.add_note(
                        f"temporary replay cleanup failed: {cleanup_error}"
                    )
        raise


def run_replay_cli(
    capsule_path: str | Path,
    output_root: str | Path,
    replay_loader: Callable[
        [LoadedFailureCapsule], ReplayReport
    ] = replay_failure_capsule,
) -> Path:
    capsule = load_failure_capsule(capsule_path)
    output = validate_isolated_output_root(output_root, capsule.manifest)
    capsule_directory = Path(capsule.path).expanduser().resolve()
    if _overlaps(output, capsule_directory):
        raise ValueError(
            "replay output must be disjoint from the source capsule directory"
        )
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"replay output already exists: {output}")
    report = replay_loader(capsule)
    if report.failure_id != capsule.manifest["failure_id"]:
        raise ValueError("replay report failure_id does not match loaded capsule")
    x0 = np.asarray(capsule.failure_inputs["x0"])
    return write_replay_report_atomic(
        report,
        output,
        state_dim=int(x0.size),
        capsule_identity=capsule.manifest.get("content_identity_sha256"),
    )


def main() -> None:
    args = build_parser().parse_args()
    output = run_replay_cli(args.capsule, args.output_root)
    print(output)


if __name__ == "__main__":
    main()
