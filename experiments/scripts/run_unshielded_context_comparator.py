#!/usr/bin/env python3
"""Run the failure-tolerant unshielded online-context comparator."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from numbers import Integral
from pathlib import Path
import re
import shutil
import stat
import sys
from typing import Any, Callable, Mapping


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd

from experiments.scripts.run_context_ab import (
    APPROVED_SEEDS,
    DEFAULT_RESULT_ROOT as ORIGINAL_DIAGNOSTIC_ROOT,
    DIAGNOSTIC_TASK_IDS,
    HASH_FIELDS,
    MODES,
    PROVENANCE_FIELDS,
    RELEVANT_SOURCE_FIELDS,
    _evaluation_provenance,
    _package_versions,
    _provenance,
    build_diagnostic_runs,
    resume_row_is_complete,
    sha256_file,
)
from gl_gym.RL.agri_metarl import AgriMetaRL
from gl_gym.environments.models.utils import FORMAL_CVODES_OPTIONS
from gl_gym.experiments.context_ab import select_diagnostic_tasks
from gl_gym.experiments.ode_failure import (
    CapsuleContext,
    FailureCapsuleRecorder,
    load_failure_capsule,
)
from gl_gym.experiments.shield_evaluation import REQUIRED_METRICS
from gl_gym.experiments.suite_aggregation import DEFAULT_METRICS
from gl_gym.experiments.suite_evaluation import (
    load_task_env,
    run_deterministic_episode,
    task_from_row,
)
from gl_gym.experiments.suite_schema import load_suite_manifest
from experiments.scripts.run_shielded_context_ab import (
    load_unshielded_comparator as _load_published_comparator,
)


DEFAULT_RESULT_ROOT = Path(
    "artifacts/results/AgriControl_C_2026-07-10-v3-context-ab-unshielded-comparator"
)
DEFAULT_FAILURE_ROOT = Path("artifacts/failures/context-ab-unshielded-comparator")
EARLY_HORIZON_FAILURE = re.compile(
    r"evaluation episode terminated before configured horizon: "
    r"step ([1-9]\d*) of ([1-9]\d*)"
)
STATUS_FIELDS = (
    "completed",
    "status",
    "ode_failure_count",
    "failure_evidence_path",
)
EPISODE_SCORING_METRICS = tuple(DEFAULT_METRICS)
RESERVED_ROW_FIELDS = frozenset(
    {
        "seed",
        "task_id",
        "split",
        "inference_mode",
        "checkpoint_steps",
        "checkpoint_path",
        "model_path",
        "vecnormalize_path",
        "action_trace_path",
        "source_manifest",
        "source_tasks_csv",
        "runtime_source_tree_sha256",
        "action_trace_sha256",
        "failure_capsule_identity_sha256",
        "row_identity_sha256",
        *STATUS_FIELDS,
        *HASH_FIELDS,
        *PROVENANCE_FIELDS,
    }
)


def _overlaps(first: Path, second: Path) -> bool:
    return first == second or first in second.parents or second in first.parents


def _work_root(result_root: Path) -> Path:
    return result_root.parent / f".{result_root.name}.work"


def _failure_work_root(failure_root: Path, result_root: Path) -> Path:
    identity = hashlib.sha256(str(result_root.resolve()).encode("utf-8")).hexdigest()[:8]
    return failure_root / f".work-{identity}"


def validate_output_roots(
    result_root: str | Path,
    failure_root: str | Path,
    formal_result_root: str | Path,
    *,
    original_diagnostic_root: str | Path = ORIGINAL_DIAGNOSTIC_ROOT,
) -> tuple[Path, Path]:
    """Reject overlap among formal, original, comparator, and capsule lifecycles."""

    result = Path(result_root).resolve()
    failure = Path(failure_root).resolve()
    formal = Path(formal_result_root).resolve()
    original_path = Path(original_diagnostic_root)
    original = (
        original_path
        if original_path.is_absolute()
        else ROOT / original_path
    ).resolve()
    result_work = _work_root(result).resolve()
    failure_work = _failure_work_root(failure, result).resolve()
    stage = (result.parent / f".{result.name}.publish").resolve()
    backup = (result.parent / f".{result.name}.backup").resolve()
    transaction = _transaction_path(result).resolve()
    transaction_temporary = _transaction_temporary_path(result).resolve()
    protected = (formal, original)
    owned = (
        result, result_work, failure, failure_work, stage, backup,
        transaction, transaction_temporary,
    )
    if any(_overlaps(item, source) for item in owned for source in protected):
        raise ValueError(
            "comparator result and failure roots must be disjoint from formal and "
            "original diagnostic roots"
        )
    independent = (
        result, result_work, failure, stage, backup,
        transaction, transaction_temporary,
    )
    if any(
        _overlaps(first, second)
        for index, first in enumerate(independent)
        for second in independent[index + 1 :]
    ) or any(
        _overlaps(failure_work, item)
        for item in (
            result, result_work, stage, backup, transaction, transaction_temporary
        )
    ):
        raise ValueError("comparator result and failure roots must be disjoint")
    return result, failure


def _trace_path(work: Path, seed: int, task_id: str, mode: str) -> Path:
    return (work / "traces" / f"seed{seed}__{task_id}__{mode}.npy").resolve()


def _attempt_root(failure_work: Path, seed: int, task_id: str, mode: str) -> Path:
    key = f"{seed}\0{task_id}\0{mode}".encode("utf-8")
    return failure_work / hashlib.sha256(key).hexdigest()[:12]


def _runtime_source_tree_sha256(root: str | Path = ROOT) -> str:
    """Fingerprint all repository Python code directly used by this comparator."""

    repository = Path(root).expanduser().absolute()
    reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)

    def require_directory(path: Path) -> None:
        metadata = path.lstat()
        if (
            path.is_symlink()
            or getattr(metadata, "st_file_attributes", 0) & reparse_flag
            or not stat.S_ISDIR(metadata.st_mode)
        ):
            raise ValueError(f"runtime source directory must be regular: {path}")

    def require_file(path: Path) -> None:
        for parent in (repository, *reversed(path.parents)):
            if parent == repository or repository in parent.parents:
                require_directory(parent)
        metadata = path.lstat()
        if (
            path.is_symlink()
            or getattr(metadata, "st_file_attributes", 0) & reparse_flag
            or not stat.S_ISREG(metadata.st_mode)
        ):
            raise ValueError(f"runtime Python source must be a regular file: {path}")

    require_directory(repository)
    source_root = repository / "src" / "gl_gym"
    require_directory(repository / "src")
    require_directory(source_root)
    files: list[tuple[str, Path]] = []
    for path in source_root.rglob("*"):
        metadata = path.lstat()
        if path.is_symlink() or getattr(metadata, "st_file_attributes", 0) & reparse_flag:
            raise ValueError(f"runtime source tree contains symlink/reparse entry: {path}")
        if path.suffix == ".py":
            if not stat.S_ISREG(metadata.st_mode):
                raise ValueError(f"runtime Python source must be a regular file: {path}")
            files.append((path.relative_to(repository).as_posix(), path))
    for relative in (
        "experiments/scripts/run_unshielded_context_comparator.py",
        "experiments/scripts/run_context_ab.py",
    ):
        path = repository / relative
        require_file(path)
        files.append((relative, path))
    if not files:
        raise ValueError("runtime source tree contains no Python files")
    digest = hashlib.sha256()
    for relative, path in sorted(files):
        relative_bytes = relative.encode("utf-8")
        content = path.read_bytes()
        digest.update(len(relative_bytes).to_bytes(8, "big"))
        digest.update(relative_bytes)
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)
    return digest.hexdigest()


def _key(row: Mapping[str, Any]) -> tuple[int, str, str]:
    return (int(row["seed"]), str(row["task_id"]), str(row["inference_mode"]))


def _load_rows(path: Path | None) -> list[dict[str, Any]]:
    if path is None or not path.is_file():
        return []
    try:
        return pd.read_csv(path).to_dict(orient="records")
    except (OSError, UnicodeError, ValueError, pd.errors.ParserError, pd.errors.EmptyDataError):
        return []


def _canonical_scalar(name: str, value: Any) -> Any:
    if isinstance(value, np.generic):
        value = value.item()
    if name in {
        "failure_evidence_path",
        "action_trace_path",
        "action_trace_sha256",
        "failure_capsule_identity_sha256",
    } and (
        value == "" or (isinstance(value, float) and np.isnan(value))
    ):
        return {"empty": True}
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return {"nan": True}
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, float)):
        numeric = float(value)
        if not np.isfinite(numeric):
            return {"number": str(numeric)}
        return {"number": numeric.hex()}
    return str(value)


def _row_identity(row: Mapping[str, Any]) -> str:
    payload = {
        str(name): _canonical_scalar(str(name), value)
        for name, value in row.items()
        if name != "row_identity_sha256"
        and not (
            (
                value is None
                or (isinstance(value, (float, np.floating)) and np.isnan(value))
            )
            and name
            not in {
                "failure_evidence_path",
                "action_trace_path",
                "action_trace_sha256",
                "failure_capsule_identity_sha256",
            }
        )
    }
    canonical = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _signed_row(row: Mapping[str, Any]) -> dict[str, Any]:
    signed = dict(row)
    signed["row_identity_sha256"] = _row_identity(signed)
    return signed


def _replace_progress(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        pd.DataFrame(rows).to_csv(temporary, index=False)
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _strict_json(path: str | Path) -> dict[str, Any]:
    value = json.loads(
        Path(path).read_text(encoding="utf-8"),
        parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)),
    )
    if not isinstance(value, dict):
        raise ValueError(f"JSON document must be an object: {path}")
    return value


def _relocate_tree(source: Path, destination: Path) -> bool:
    """Move a complete tree, using a recoverable copy when rename is unavailable."""

    if destination.exists():
        raise FileExistsError(f"tree destination already exists: {destination}")
    try:
        source.rename(destination)
        return True
    except OSError:
        pass
    try:
        shutil.copytree(source, destination)
    except BaseException:
        if destination.exists():
            shutil.rmtree(destination)
        raise
    return False


def _transaction_path(root: Path) -> Path:
    return root.parent / f".{root.name}.transaction.json"


def _transaction_temporary_path(root: Path) -> Path:
    return _transaction_path(root).with_suffix(".tmp")


def _write_transaction(
    root: Path, state: str, *, tree_identity_sha256: str | None = None
) -> None:
    marker = _transaction_path(root)
    temporary = _transaction_temporary_path(root)
    payload = {"state": state, "result_root": str(root.resolve())}
    if tree_identity_sha256 is not None:
        payload["tree_identity_sha256"] = tree_identity_sha256
    temporary.write_text(
        json.dumps(payload, sort_keys=True),
        encoding="utf-8",
    )
    temporary.replace(marker)


def _tree_file_hashes(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): sha256_file(path)
        for path in root.rglob("*")
        if path.is_file()
    }


def _tree_identity(root: Path) -> str:
    canonical = json.dumps(
        _tree_file_hashes(root), sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _verified_transaction_matches(
    root: Path, transaction: Mapping[str, Any]
) -> bool:
    try:
        identity = str(transaction["tree_identity_sha256"])
        return (
            transaction.get("state") == "consumer_verified"
            and Path(str(transaction["result_root"])).resolve() == root.resolve()
            and len(identity) == 64
            and identity == _tree_identity(root)
        )
    except (KeyError, OSError, TypeError, ValueError):
        return False


def _installed_path(root: Path, value: Any, *, name: str) -> Path:
    path = Path(str(value))
    if not path.is_absolute():
        path = root / path
    resolved = path.resolve()
    if root.resolve() not in resolved.parents:
        raise ValueError(f"installed comparator {name} escapes final result root")
    return resolved


def _validate_installed_integrity(
    root: Path,
    table: pd.DataFrame,
    manifest: Mapping[str, Any],
    capsule_evidence: list[Mapping[str, Any]],
    *,
    expected_manifest: Mapping[str, Any],
) -> None:
    """Bind the installed bytes, CSV identities, traces, and capsules together."""

    reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    if dict(manifest) != dict(expected_manifest):
        raise ValueError("installed comparator manifest differs from staged provenance")
    actual_files: dict[str, Path] = {}
    for path in root.rglob("*"):
        metadata = path.lstat()
        if path.is_symlink() or getattr(metadata, "st_file_attributes", 0) & reparse_flag:
            raise ValueError("installed comparator integrity rejects symlink/reparse entries")
        if stat.S_ISREG(metadata.st_mode):
            actual_files[path.relative_to(root).as_posix()] = path
        elif not stat.S_ISDIR(metadata.st_mode):
            raise ValueError("installed comparator contains a non-regular filesystem entry")
    manifest_name = "context_ab_manifest.json"
    hashes = manifest.get("published_file_sha256")
    if not isinstance(hashes, Mapping) or not all(
        isinstance(name, str) and isinstance(digest, str)
        for name, digest in hashes.items()
    ):
        raise ValueError("published file integrity hashes are missing or malformed")
    expected_names = set(hashes)
    actual_names = set(actual_files).difference({manifest_name})
    if expected_names != actual_names:
        raise ValueError("published file integrity set has missing or extra files")
    for relative, digest in hashes.items():
        relative_path = Path(relative)
        if (
            relative_path.is_absolute()
            or ".." in relative_path.parts
            or relative_path.as_posix() != relative
            or len(digest) != 64
        ):
            raise ValueError("published file integrity path/hash is noncanonical")
        installed = _installed_path(root, relative_path, name="published file")
        if installed != actual_files[relative] or sha256_file(installed) != digest:
            raise ValueError(f"published file hash mismatch: {relative}")

    identities = manifest.get("row_identities")
    if not isinstance(identities, list) or "row_identity_sha256" not in table:
        raise ValueError("installed comparator row identities are missing")
    csv_identities = table["row_identity_sha256"].astype(str).tolist()
    if identities != csv_identities:
        raise ValueError("manifest row identity order does not match eval_raw.csv")
    capsule_by_manifest = {
        Path(item["manifest_path"]).resolve(): item for item in capsule_evidence
    }
    for _, series in table.iterrows():
        row = series.to_dict()
        if str(row["row_identity_sha256"]) != _row_identity(row):
            raise ValueError("installed comparator row identity does not match CSV bytes")
        if bool(row["completed"]):
            trace = _installed_path(
                root, row["action_trace_path"], name="completed action trace"
            )
            if trace.name == "manifest.json" or sha256_file(trace) != str(
                row["action_trace_sha256"]
            ):
                raise ValueError("completed action trace hash mismatch")
            continue
        evidence_path = _installed_path(
            root, row["failure_evidence_path"], name="failure capsule manifest"
        )
        evidence = capsule_by_manifest.get(evidence_path)
        if evidence is None or str(evidence.get("capsule_identity_sha256")) != str(
            row["failure_capsule_identity_sha256"]
        ):
            raise ValueError("failure capsule identity/path integrity mismatch")


def _restore_backup(root: Path, backup: Path) -> None:
    if root.exists():
        shutil.rmtree(root)
    renamed = _relocate_tree(backup, root)
    if not renamed:
        _write_transaction(root, "restored_copy")
        shutil.rmtree(backup)
    _transaction_path(root).unlink(missing_ok=True)


def _recover_publication(root: Path) -> None:
    """Resolve an interrupted publication before any experiment can execute."""

    backup = root.parent / f".{root.name}.backup"
    marker = _transaction_path(root)
    transaction: dict[str, Any] = {}
    state: str | None = None
    if marker.is_file():
        transaction = _strict_json(marker)
        state = str(transaction.get("state"))
    if not backup.exists():
        if state == "consumer_verified":
            if not root.exists() or not _verified_transaction_matches(root, transaction):
                raise RuntimeError("verified comparator transaction identity mismatch")
            stage = root.parent / f".{root.name}.publish"
            if stage.exists():
                shutil.rmtree(stage)
            _transaction_temporary_path(root).unlink(missing_ok=True)
            marker.unlink(missing_ok=True)
            return
        if state in {"candidate_pending", "candidate_installed"}:
            stage = root.parent / f".{root.name}.publish"
            if root.exists():
                shutil.rmtree(root)
            if stage.exists():
                shutil.rmtree(stage)
            _transaction_temporary_path(root).unlink(missing_ok=True)
            marker.unlink(missing_ok=True)
            return
        if root.exists():
            marker.unlink(missing_ok=True)
            return
        if state is not None:
            raise RuntimeError("publication transaction lost both final root and backup")
        return
    if not root.exists():
        _restore_backup(root, backup)
        return
    if state == "backup_pending":
        shutil.rmtree(backup)
        marker.unlink(missing_ok=True)
        return
    if state == "backup_ready":
        if _tree_file_hashes(root) == _tree_file_hashes(backup):
            shutil.rmtree(backup)
            marker.unlink(missing_ok=True)
        else:
            _restore_backup(root, backup)
        return
    if state == "consumer_verified":
        stage = root.parent / f".{root.name}.publish"
        if root.exists() and _verified_transaction_matches(root, transaction):
            shutil.rmtree(backup)
            if stage.exists():
                shutil.rmtree(stage)
            marker.unlink(missing_ok=True)
        else:
            _restore_backup(root, backup)
            if stage.exists():
                shutil.rmtree(stage)
        return
    if state in {"candidate_pending", "candidate_installed"}:
        _restore_backup(root, backup)
        return
    if state == "restored_copy":
        shutil.rmtree(backup)
        marker.unlink(missing_ok=True)
        return
    raise RuntimeError("ambiguous comparator backup recovery state")


def _published_row(
    row: Mapping[str, Any], *, root: Path, work: Path, failure_work: Path, stage: Path
) -> dict[str, Any]:
    published = dict(row)
    key = _key(row)
    if bool(row["completed"]):
        source = Path(str(row["action_trace_path"])).resolve()
        expected = _trace_path(work, *key)
        if source != expected or not source.is_file():
            raise ValueError("completed row action trace is not canonical work evidence")
        relative = Path("traces") / source.name
        destination = stage / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        published["action_trace_path"] = str((root / relative).resolve())
        published["action_trace_sha256"] = sha256_file(destination)
        published["failure_evidence_path"] = ""
        published["failure_capsule_identity_sha256"] = ""
    else:
        source = Path(str(row["failure_evidence_path"])).resolve()
        try:
            relative_source = source.relative_to(failure_work.resolve())
        except ValueError as error:
            raise ValueError("failure capsule is not canonical work evidence") from error
        if source.name != "manifest.json" or not source.is_file():
            raise ValueError("failed row capsule manifest is missing")
        capsule_dir = source.parent
        relative = Path("failures") / relative_source.parent
        destination = stage / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(capsule_dir, destination)
        published["failure_evidence_path"] = str(
            (root / relative / "manifest.json").resolve()
        )
        published["action_trace_path"] = ""
        published["action_trace_sha256"] = ""
    return _signed_row(published)


def _publish_comparator(
    rows: list[dict[str, Any]],
    *,
    root: Path,
    work: Path,
    failure_work: Path,
    suite: Any,
    runs: list[dict[str, Any]],
    source_manifest: str | Path,
    source_tasks_csv: str | Path,
    provenance: Mapping[str, Any],
    runtime_source_tree_sha256: str,
    capsule_loader: Callable[[str | Path], Any],
) -> pd.DataFrame:
    """Publish one exact comparator while preserving any prior accepted root."""

    if len(rows) != 32 or {_key(row) for row in rows} != {
        (seed, task_id, mode)
        for seed in APPROVED_SEEDS
        for task_id in DIAGNOSTIC_TASK_IDS
        for mode in MODES
    }:
        raise RuntimeError("only the exact 32-key comparator may be published")
    stage = root.parent / f".{root.name}.publish"
    backup = root.parent / f".{root.name}.backup"
    _recover_publication(root)
    if stage.exists():
        shutil.rmtree(stage)
    stage.mkdir(parents=True)
    (stage / "traces").mkdir()
    (stage / "failures").mkdir()
    published_rows: list[dict[str, Any]] = []
    try:
        for row in rows:
            published_rows.append(
                _published_row(
                    row, root=root, work=work, failure_work=failure_work, stage=stage
                )
            )
        table = pd.DataFrame(published_rows)
        table.to_csv(stage / "eval_raw.csv", index=False)
        file_hashes = {
            path.relative_to(stage).as_posix(): sha256_file(path)
            for path in sorted(stage.rglob("*"))
            if path.is_file()
        }
        checkpoints = [
            {
                name: (int(run[name]) if name == "seed" else str(run[name]))
                for name in (
                    "seed", "model_path", "vecnormalize_path",
                    "model_sha256", "vecnormalize_sha256",
                )
            }
            for run in runs
        ]
        manifest = {
            "schema_version": 1,
            "artifact_type": "failure_tolerant_unshielded_context_comparator",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "result_root": str(root.resolve()),
            "source_suite_id": str(suite.suite_id),
            "source_suite_result_root": str(Path(suite.result_root).resolve()),
            "source_manifest": str(Path(source_manifest).resolve()),
            "source_tasks_csv": str(Path(source_tasks_csv).resolve()),
            "checkpoints": checkpoints,
            "solver": {"backend": "CVODES", "options": dict(FORMAL_CVODES_OPTIONS)},
            "runtime_source_tree_sha256": runtime_source_tree_sha256,
            "row_identities": [str(row["row_identity_sha256"]) for row in published_rows],
            "row_keys": [list(_key(row)) for row in published_rows],
            "published_file_sha256": file_hashes,
            **dict(provenance),
        }
        (stage / "context_ab_manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )

        had_old = root.exists()
        if had_old:
            _write_transaction(root, "backup_pending")
            renamed = _relocate_tree(root, backup)
            _write_transaction(root, "backup_ready")
            if not renamed:
                shutil.rmtree(root)
        try:
            _write_transaction(root, "candidate_pending")
            _relocate_tree(stage, root)
            _write_transaction(root, "candidate_installed")
            expected_checkpoints = {int(run["seed"]): run for run in runs}
            loaded, loaded_manifest, loaded_capsules = _load_published_comparator(
                root,
                expected_provenance=dict(provenance),
                expected_checkpoints=expected_checkpoints,
                capsule_loader=capsule_loader,
            )
            if len(loaded) != 32 or set(loaded[["seed", "task_id", "inference_mode"]].itertuples(index=False, name=None)) != {
                _key(row) for row in published_rows
            }:
                raise ValueError("published comparator consumer round-trip changed exact keys")
            _validate_installed_integrity(
                root,
                loaded,
                loaded_manifest,
                loaded_capsules,
                expected_manifest=manifest,
            )
            _write_transaction(
                root,
                "consumer_verified",
                tree_identity_sha256=_tree_identity(root),
            )
        except BaseException:
            if had_old and backup.exists():
                _restore_backup(root, backup)
            elif root.exists():
                shutil.rmtree(root)
                _transaction_path(root).unlink(missing_ok=True)
            raise
        if stage.exists():
            shutil.rmtree(stage)
        if backup.exists():
            shutil.rmtree(backup)
        _transaction_path(root).unlink(missing_ok=True)
        return table
    except BaseException:
        if stage.exists():
            shutil.rmtree(stage)
        raise


def _validate_scoring_metrics(
    row: Mapping[str, Any], *, expected: str
) -> None:
    names = [
        *REQUIRED_METRICS,
        *(
            name
            for name in EPISODE_SCORING_METRICS
            if name not in REQUIRED_METRICS and name in row
        ),
    ]
    try:
        values = np.asarray([float(row[name]) for name in names], dtype=float)
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("episode scoring metrics are missing or nonnumeric") from error
    valid = np.isfinite(values).all() if expected == "finite" else np.isnan(values).all()
    if not valid:
        raise ValueError(f"episode scoring metrics must all be {expected}")


def _finite_metrics(metrics: Any) -> dict[str, Any]:
    if not isinstance(metrics, Mapping):
        raise TypeError("episode metrics must be a mapping")
    normalized = dict(metrics)
    missing = set(REQUIRED_METRICS).difference(normalized)
    if missing:
        raise KeyError(f"episode metrics are missing required keys: {sorted(missing)}")
    _validate_scoring_metrics(normalized, expected="finite")
    for name, value in list(normalized.items()):
        try:
            scalar = float(value)
        except (TypeError, ValueError) as error:
            raise TypeError(f"episode metric {name!r} must be a numeric scalar") from error
        if isinstance(value, np.generic):
            normalized[name] = value.item()
        elif not isinstance(value, (int, float)):
            normalized[name] = scalar
    return normalized


def _checkpoint_steps(model: Any) -> int:
    try:
        value = model.num_timesteps
    except AttributeError as error:
        raise ValueError("checkpoint steps must be a nonnegative exact integer") from error
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise ValueError("checkpoint steps must be a nonnegative exact integer")
    exact = int(value)
    if exact < 0:
        raise ValueError("checkpoint steps must be a nonnegative exact integer")
    return exact


def _success_diagnostics(
    diagnostics: Any, *, metric_names: set[str]
) -> tuple[dict[str, Any], np.ndarray]:
    if not isinstance(diagnostics, Mapping):
        raise TypeError("episode diagnostics must be a mapping")
    values = dict(diagnostics)
    missing = {
        "support_ready_step",
        "context_norm_mean",
        "context_norm_max",
        "action_trace",
    }.difference(values)
    if missing:
        raise KeyError(f"episode diagnostics are missing required keys: {sorted(missing)}")
    collisions = set(values).intersection(metric_names | RESERVED_ROW_FIELDS)
    if collisions:
        raise ValueError(
            "episode diagnostics collide with metric or reserved row fields: "
            f"{sorted(collisions)}"
        )
    trace = np.asarray(values.pop("action_trace"), dtype=np.float32)
    if trace.ndim != 2 or not trace.shape[0] or not trace.shape[1]:
        raise ValueError("action_trace must be a nonempty two-dimensional array")
    if not np.isfinite(trace).all():
        raise ValueError("action_trace must contain only finite values")
    for name in ("context_norm_mean", "context_norm_max"):
        if not np.isfinite(float(values[name])):
            raise ValueError(f"{name} must be finite")
    readiness = float(values["support_ready_step"])
    if not (np.isfinite(readiness) or np.isnan(readiness)):
        raise ValueError("support_ready_step must be finite or NaN")
    for name, value in list(values.items()):
        if isinstance(value, np.generic):
            values[name] = value.item()
        elif value is not None and not isinstance(value, (str, bool, int, float)):
            raise TypeError(f"diagnostic {name!r} must be a CSV-safe scalar")
    return values, trace


def _validate_completed_row(row: Mapping[str, Any]) -> None:
    """Validate the final merged row before it becomes resumable progress."""

    evidence = row.get("failure_evidence_path")
    evidence_is_empty = evidence == "" or (
        isinstance(evidence, float) and np.isnan(evidence)
    )
    if (
        row.get("completed") not in (True, np.bool_(True))
        or row.get("status") != "completed"
        or row.get("ode_failure_count") != 0
        or not evidence_is_empty
    ):
        raise ValueError("completed comparator row has an invalid status schema")
    try:
        _validate_scoring_metrics(row, expected="finite")
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("completed comparator row has invalid scoring metrics") from error


def _valid_readiness(row: Mapping[str, Any], trace_length: int) -> bool:
    try:
        readiness = float(row["support_ready_step"])
    except (KeyError, TypeError, ValueError):
        return False
    if str(row.get("inference_mode")) == "online_context":
        return bool(
            np.isfinite(readiness)
            and readiness.is_integer()
            and 1 <= readiness < trace_length
        )
    if str(row.get("inference_mode")) != "zero_context":
        return False
    return bool(
        np.isnan(readiness)
        or (
            np.isfinite(readiness)
            and readiness.is_integer()
            and 1 <= readiness < trace_length
        )
    )


def _base_matches(row: Mapping[str, Any], expected: Mapping[str, Any]) -> bool:
    try:
        for name, value in expected.items():
            actual = row[name]
            if isinstance(value, (bool, np.bool_)):
                if isinstance(actual, str):
                    if actual not in {"True", "False"}:
                        return False
                    actual = actual == "True"
                if not isinstance(actual, (bool, np.bool_)) or bool(actual) != bool(value):
                    return False
            elif isinstance(value, int):
                numeric = float(actual)
                if not np.isfinite(numeric) or not numeric.is_integer() or int(numeric) != value:
                    return False
            elif str(actual) != str(value):
                return False
        identity = str(row["row_identity_sha256"])
        return len(identity) == 64 and identity == _row_identity(row)
    except (KeyError, OverflowError, TypeError, ValueError):
        return False


def _completed_row_is_valid(
    row: Mapping[str, Any], *, expected: Mapping[str, Any], trace_path: Path
) -> bool:
    if not _base_matches(row, expected):
        return False
    try:
        _validate_completed_row(row)
        if Path(str(row["action_trace_path"])).resolve() != trace_path.resolve():
            return False
        trace = np.load(trace_path, allow_pickle=False)
        if (
            trace.ndim != 2
            or not trace.shape[0]
            or not trace.shape[1]
            or not np.isfinite(trace).all()
            or sha256_file(trace_path) != str(row["action_trace_sha256"])
            or not _valid_readiness(row, trace.shape[0])
        ):
            return False
        context = np.asarray(
            [float(row["context_norm_mean"]), float(row["context_norm_max"])],
            dtype=float,
        )
        return bool(np.isfinite(context).all())
    except (KeyError, OSError, TypeError, ValueError):
        return False


def _validate_capsule(
    manifest_path: Path,
    *,
    expected_context: CapsuleContext,
    error: Exception,
    capsule_loader: Callable[[str | Path], Any],
) -> Any:
    match = EARLY_HORIZON_FAILURE.fullmatch(str(error))
    if type(error) is not RuntimeError or match is None:
        raise ValueError("caught exception is not the exact early-horizon RuntimeError")
    episode_step, horizon = (int(item) for item in match.groups())
    if episode_step >= horizon:
        raise ValueError("early-horizon RuntimeError has inconsistent step/horizon")
    capsule = _validate_capsule_evidence(
        manifest_path,
        expected_context=expected_context,
        capsule_loader=capsule_loader,
    )
    try:
        recorded_step = int(capsule.history_arrays["step_index"][-1])
        failure_timestep = int(capsule.failure_inputs["timestep"])
    except (KeyError, IndexError, TypeError, ValueError) as capsule_error:
        raise ValueError("failure capsule has malformed step evidence") from capsule_error
    if recorded_step != episode_step - 1 or failure_timestep != recorded_step:
        raise ValueError("failure capsule timestep does not match the early episode step")
    return capsule


def _validate_capsule_evidence(
    manifest_path: Path,
    *,
    expected_context: CapsuleContext,
    capsule_loader: Callable[[str | Path], Any],
) -> Any:
    capsule = capsule_loader(manifest_path.parent)
    expected = asdict(expected_context)
    if capsule.manifest.get("context") != expected:
        raise ValueError("failure capsule context does not match this comparator attempt")
    exception = capsule.manifest.get("exception", {})
    if not exception.get("type") or not exception.get("message"):
        raise ValueError("failure capsule underlying exception fields are empty")
    if (
        exception["type"] not in capsule.traceback_text
        or exception["message"] not in capsule.traceback_text
    ):
        raise ValueError("failure capsule traceback does not bind its underlying exception")
    try:
        recorded_step = int(capsule.history_arrays["step_index"][-1])
        failure_timestep = int(capsule.failure_inputs["timestep"])
    except (KeyError, IndexError, TypeError, ValueError) as capsule_error:
        raise ValueError("failure capsule has malformed step evidence") from capsule_error
    if recorded_step < 0 or failure_timestep != recorded_step:
        raise ValueError("failure capsule has inconsistent step evidence")
    if capsule.manifest.get("solver", {}).get("options") != dict(
        FORMAL_CVODES_OPTIONS
    ):
        raise ValueError("failure capsule does not prove formal CVODES options")
    identity = capsule.manifest.get("content_identity_sha256")
    if not isinstance(identity, str) or len(identity) != 64:
        raise ValueError("failure capsule content identity is invalid")
    return capsule


def _failed_row_is_valid(
    row: Mapping[str, Any],
    *,
    expected: Mapping[str, Any],
    attempt: Path,
    context: CapsuleContext,
    capsule_loader: Callable[[str | Path], Any],
) -> bool:
    if not _base_matches(row, expected):
        return False
    try:
        if (
            row["completed"] not in (False, np.bool_(False))
            or str(row["status"]) != "ode_failure"
            or int(row["ode_failure_count"]) != 1
        ):
            return False
        _validate_scoring_metrics(row, expected="NaN")
        diagnostics = np.asarray(
            [
                float(row["support_ready_step"]),
                float(row["context_norm_mean"]),
                float(row["context_norm_max"]),
            ]
        )
        if not np.isnan(diagnostics).all():
            return False
        trace_value = row["action_trace_path"]
        if not (
            trace_value == ""
            or (isinstance(trace_value, float) and np.isnan(trace_value))
        ):
            return False
        manifest = Path(str(row["failure_evidence_path"])).resolve()
        manifests = sorted(attempt.rglob("manifest.json")) if attempt.is_dir() else []
        if len(manifests) != 1 or manifests[0].resolve() != manifest:
            return False
        capsule = _validate_capsule_evidence(
            manifest, expected_context=context, capsule_loader=capsule_loader
        )
        return str(row["failure_capsule_identity_sha256"]) == str(
            capsule.manifest["content_identity_sha256"]
        )
    except (KeyError, OSError, TypeError, ValueError):
        return False


def run_unshielded_comparator(
    *,
    suite: Any,
    tasks: pd.DataFrame,
    runs: list[dict[str, Any]],
    result_root: str | Path,
    failure_root: str | Path,
    source_manifest: str | Path,
    source_tasks_csv: str | Path,
    device: str,
    resume: bool,
    legacy_progress: str | Path | None = None,
    model_loader: Callable[[Path, str], Any] | None = None,
    env_loader: Callable[[Any, Any, Path], Any] = load_task_env,
    episode_runner: Callable[..., Any] = run_deterministic_episode,
    provenance_loader: Callable[[], dict[str, Any]] = _provenance,
    recorder_factory: Callable[[str | Path, CapsuleContext], Any] = FailureCapsuleRecorder,
    capsule_loader: Callable[[str | Path], Any] = load_failure_capsule,
) -> pd.DataFrame:
    """Run all approved keys, retaining only strictly proven ODE failures."""

    root, capsule_root = validate_output_roots(
        result_root, failure_root, suite.result_root
    )
    work = _work_root(root)
    failure_work = _failure_work_root(capsule_root, root)
    source_inputs = (Path(source_manifest).resolve(), Path(source_tasks_csv).resolve())
    mutable_roots = (
        root.resolve(),
        work.resolve(),
        capsule_root.resolve(),
        failure_work.resolve(),
        (root.parent / f".{root.name}.publish").resolve(),
        (root.parent / f".{root.name}.backup").resolve(),
        _transaction_path(root).resolve(),
        _transaction_temporary_path(root).resolve(),
    )
    if any(_overlaps(source, owned) for source in source_inputs for owned in mutable_roots):
        raise ValueError("source inputs and comparator output topology must be disjoint")
    _recover_publication(root)
    if not resume:
        for path in (work, failure_work):
            if path.exists():
                shutil.rmtree(path)
    progress_path = work / "progress.csv"
    work.joinpath("traces").mkdir(parents=True, exist_ok=True)
    resume_rows = _load_rows(progress_path) if resume else []
    legacy_path = Path(legacy_progress).resolve() if legacy_progress is not None else None
    legacy_rows = _load_rows(legacy_path)

    selected = select_diagnostic_tasks(tasks)
    task_records = [task_from_row(row) for row in selected.itertuples(index=False)]
    provenance = _evaluation_provenance(
        source_manifest, source_tasks_csv, dict(provenance_loader())
    )
    source_checksums = {
        str((ROOT / path).resolve()): provenance[name]
        for name, path in RELEVANT_SOURCE_FIELDS
    }
    source_checksums.update(
        {
            str(Path(source_manifest).resolve()): provenance["source_manifest_sha256"],
            str(Path(source_tasks_csv).resolve()): provenance["source_tasks_sha256"],
        }
    )
    packages = _package_versions()
    runtime_source_tree_sha256 = _runtime_source_tree_sha256()
    evidence_by_seed = {
        int(run["seed"]): {
            "model_sha256": str(run["model_sha256"]),
            "vecnormalize_sha256": str(run["vecnormalize_sha256"]),
            **provenance,
        }
        for run in runs
    }
    expected_seeds = tuple(int(run["seed"]) for run in runs)
    if expected_seeds != APPROVED_SEEDS:
        raise ValueError("approved comparator requires runs for seeds exactly 42 123")
    load_model = model_loader or (
        lambda path, selected_device: AgriMetaRL.load(str(path), device=selected_device)
    )
    rows: list[dict[str, Any]] = []
    for run in runs:
        seed = int(run["seed"])
        model = load_model(Path(run["model_path"]), device)
        checkpoint_steps = _checkpoint_steps(model)
        for task in task_records:
            for mode in MODES:
                attempt = _attempt_root(failure_work, seed, task.task_id, mode)
                context = CapsuleContext(
                    seed=seed,
                    task_id=task.task_id,
                    inference_mode=mode,
                    task=asdict(task),
                    checkpoint_path=str(Path(run["model_path"]).resolve()),
                    checkpoint_sha256=str(run["model_sha256"]),
                    git_head=str(provenance["git_commit"]),
                    dirty=bool(provenance["dirty"]),
                    source_checksums=dict(source_checksums),
                    package_versions=dict(packages),
                    formal_result_root=str(root),
                )
                base = {
                    "seed": seed,
                    "task_id": task.task_id,
                    "split": task.split,
                    "inference_mode": mode,
                    "checkpoint_steps": checkpoint_steps,
                    "checkpoint_path": str(Path(run["model_path"]).resolve()),
                    "model_path": str(Path(run["model_path"]).resolve()),
                    "vecnormalize_path": str(
                        Path(run["vecnormalize_path"]).resolve()
                    ),
                    "source_manifest": str(Path(source_manifest).resolve()),
                    "source_tasks_csv": str(Path(source_tasks_csv).resolve()),
                    "runtime_source_tree_sha256": runtime_source_tree_sha256,
                    **evidence_by_seed[seed],
                }
                key = (seed, task.task_id, mode)
                trace_path = _trace_path(work, seed, task.task_id, mode)
                reused: dict[str, Any] | None = None
                for candidate in resume_rows:
                    try:
                        if _key(candidate) != key:
                            continue
                    except (KeyError, TypeError, ValueError):
                        continue
                    status = str(candidate.get("status"))
                    if status == "completed" and not attempt.exists() and _completed_row_is_valid(
                        candidate, expected=base, trace_path=trace_path
                    ):
                        reused = dict(candidate)
                        reused["failure_evidence_path"] = ""
                        reused["failure_capsule_identity_sha256"] = ""
                        break
                    if status == "ode_failure" and _failed_row_is_valid(
                        candidate,
                        expected=base,
                        attempt=attempt,
                        context=context,
                        capsule_loader=capsule_loader,
                    ):
                        reused = dict(candidate)
                        reused["action_trace_path"] = ""
                        reused["action_trace_sha256"] = ""
                        break
                if reused is not None:
                    rows.append(reused)
                    continue

                for candidate in legacy_rows:
                    if set(STATUS_FIELDS).intersection(candidate):
                        continue
                    try:
                        candidate_key = _key(candidate)
                    except (KeyError, TypeError, ValueError):
                        continue
                    if candidate_key != key or str(candidate.get("split")) != task.split:
                        continue
                    if (
                        str(candidate.get("runtime_source_tree_sha256"))
                        != runtime_source_tree_sha256
                        or legacy_path is None
                        or Path(str(candidate.get("action_trace_path"))).resolve()
                        != _trace_path(legacy_path.parent, *key)
                    ):
                        continue
                    if not resume_row_is_complete(
                        candidate,
                        checkpoint_steps=checkpoint_steps,
                        expected_hashes=evidence_by_seed[seed],
                    ):
                        continue
                    source_trace = Path(str(candidate["action_trace_path"])).resolve()
                    trace_path.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(source_trace, trace_path)
                    imported = dict(candidate)
                    for name in (
                        "row_identity_sha256",
                        "action_trace_sha256",
                        "failure_capsule_identity_sha256",
                        "failure_evidence_path",
                        "status",
                        "completed",
                        "ode_failure_count",
                    ):
                        imported.pop(name, None)
                    imported.update(base)
                    imported.update(
                        {
                            "action_trace_path": str(trace_path),
                            "action_trace_sha256": sha256_file(trace_path),
                            "failure_capsule_identity_sha256": "",
                            "completed": True,
                            "status": "completed",
                            "ode_failure_count": 0,
                            "failure_evidence_path": "",
                        }
                    )
                    imported = _signed_row(imported)
                    if not _completed_row_is_valid(
                        imported, expected=base, trace_path=trace_path
                    ):
                        trace_path.unlink(missing_ok=True)
                        continue
                    reused = imported
                    break
                if reused is not None:
                    rows.append(reused)
                    _replace_progress(rows, progress_path)
                    continue

                if attempt.exists():
                    shutil.rmtree(attempt)
                recorder = recorder_factory(attempt, context)
                env = env_loader(suite, task, Path(run["vecnormalize_path"]))
                error: Exception | None = None
                close_failed = False
                metrics: Any = None
                diagnostics: Any = None
                try:
                    try:
                        metrics, diagnostics = episode_runner(
                            model,
                            env,
                            inference_mode=mode,
                            return_diagnostics=True,
                            failure_recorder=recorder,
                        )
                    except Exception as caught:
                        error = caught
                finally:
                    primary = sys.exception() or error
                    try:
                        env.close()
                    except Exception as close_error:
                        if primary is None:
                            raise
                        close_failed = True
                        primary.add_note(
                            "environment close also failed: "
                            f"{type(close_error).__name__}: {close_error}"
                        )
                if close_failed:
                    assert error is not None
                    raise error
                manifests = sorted(attempt.rglob("manifest.json")) if attempt.exists() else []
                if error is not None:
                    try:
                        if len(manifests) != 1:
                            raise ValueError(
                                "expected exactly one new failure capsule, "
                                f"found {len(manifests)}"
                            )
                        capsule = _validate_capsule(
                            manifests[0],
                            expected_context=context,
                            error=error,
                            capsule_loader=capsule_loader,
                        )
                    except Exception as classification_error:
                        if attempt.exists():
                            shutil.rmtree(attempt)
                        error.add_note(
                            "ODE failure classification rejected: "
                            f"{type(classification_error).__name__}: {classification_error}"
                        )
                        raise error
                    row = _signed_row({
                        **{
                            name: float("nan")
                            for name in EPISODE_SCORING_METRICS
                        },
                        "support_ready_step": float("nan"),
                        "context_norm_mean": float("nan"),
                        "context_norm_max": float("nan"),
                        "action_trace_path": "",
                        "action_trace_sha256": "",
                        "failure_capsule_identity_sha256": str(
                            capsule.manifest["content_identity_sha256"]
                        ),
                        **base,
                        "completed": False,
                        "status": "ode_failure",
                        "ode_failure_count": 1,
                        "failure_evidence_path": str(manifests[0].resolve()),
                    })
                else:
                    if manifests:
                        shutil.rmtree(attempt)
                        raise ValueError(
                            "successful comparator episode unexpectedly produced a failure capsule"
                        )
                    normalized_metrics = _finite_metrics(metrics)
                    normalized_diagnostics, trace = _success_diagnostics(
                        diagnostics, metric_names=set(normalized_metrics)
                    )
                    np.save(trace_path, trace, allow_pickle=False)
                    row = _signed_row({
                        **normalized_metrics,
                        **normalized_diagnostics,
                        "action_trace_path": str(trace_path),
                        "action_trace_sha256": sha256_file(trace_path),
                        "failure_capsule_identity_sha256": "",
                        **base,
                        "completed": True,
                        "status": "completed",
                        "ode_failure_count": 0,
                        "failure_evidence_path": "",
                    })
                    _validate_completed_row(row)
                    if not _valid_readiness(row, trace.shape[0]):
                        raise ValueError(
                            "support_ready_step is invalid for the inference mode"
                        )
                    if attempt.exists():
                        shutil.rmtree(attempt)
                rows.append(row)
                _replace_progress(rows, progress_path)
    if len(rows) != 32:
        raise RuntimeError(f"comparator completed {len(rows)} of 32 required episodes")
    return _publish_comparator(
        rows,
        root=root,
        work=work,
        failure_work=failure_work,
        suite=suite,
        runs=runs,
        source_manifest=source_manifest,
        source_tasks_csv=source_tasks_csv,
        provenance=provenance,
        runtime_source_tree_sha256=runtime_source_tree_sha256,
        capsule_loader=capsule_loader,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source_manifest", required=True)
    parser.add_argument("--source_tasks_csv", required=True)
    parser.add_argument("--model_root", required=True)
    parser.add_argument("--result_root", default=str(DEFAULT_RESULT_ROOT))
    parser.add_argument("--failure_root", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--legacy_progress")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    source_manifest = Path(args.source_manifest)
    source_tasks = Path(args.source_tasks_csv)
    if not source_manifest.is_file():
        raise FileNotFoundError(f"source manifest does not exist: {source_manifest}")
    if not source_tasks.is_file():
        raise FileNotFoundError(f"source task CSV does not exist: {source_tasks}")
    suite = load_suite_manifest(source_manifest)
    runs = build_diagnostic_runs(args.model_root, args.seeds)
    run_unshielded_comparator(
        suite=suite,
        tasks=pd.read_csv(source_tasks),
        runs=runs,
        result_root=args.result_root,
        failure_root=args.failure_root,
        source_manifest=source_manifest,
        source_tasks_csv=source_tasks,
        device=args.device,
        resume=args.resume,
        legacy_progress=args.legacy_progress,
    )


if __name__ == "__main__":
    main()
