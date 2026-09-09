"""Diagnostics helpers for context-aware recurrent PPO training."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class ContextDiagnostics:
    """Collect and summarize context encoder diagnostics."""

    _contexts: list[np.ndarray] = field(default_factory=list)
    _active_masks: list[np.ndarray] = field(default_factory=list)
    _support_sizes: list[np.ndarray] = field(default_factory=list)
    _task_instance_keys: list[np.ndarray] = field(default_factory=list)
    last_summary: dict[str, float] = field(default_factory=dict)

    def record_contexts(
        self,
        contexts: np.ndarray,
        active_mask: np.ndarray,
        support_sizes: np.ndarray,
        task_instance_keys: np.ndarray,
    ) -> None:
        contexts = np.asarray(contexts, dtype=np.float64)
        active_mask = np.asarray(active_mask, dtype=bool)
        support_sizes = np.asarray(support_sizes, dtype=np.float64)
        task_instance_keys = np.asarray(task_instance_keys, dtype=object)

        if contexts.ndim != 2:
            raise ValueError(
                "contexts must be a 2D array with one row per environment/task instance"
            )

        row_count = contexts.shape[0]
        self._validate_row_count("active_mask", active_mask, row_count)
        self._validate_row_count("support_sizes", support_sizes, row_count)
        self._validate_row_count("task_instance_keys", task_instance_keys, row_count)

        self._contexts.append(contexts.copy())
        self._active_masks.append(active_mask.reshape(row_count).copy())
        self._support_sizes.append(support_sizes.reshape(row_count).copy())
        self._task_instance_keys.append(task_instance_keys.reshape(row_count).copy())

    def summarize(self) -> dict[str, float]:
        if not self._contexts:
            summary = {
                "train/context_active_fraction": 0.0,
                "train/context_norm_mean": 0.0,
                "train/context_norm_std": 0.0,
                "train/no_context_fraction": 1.0,
                "train/support_size_mean": 0.0,
                "train/context_between_task_variance": 0.0,
            }
            self.last_summary = summary.copy()
            return summary

        contexts = np.concatenate(self._contexts, axis=0)
        active_mask = np.concatenate(self._active_masks, axis=0)
        support_sizes = np.concatenate(self._support_sizes, axis=0)
        task_instance_keys = np.concatenate(self._task_instance_keys, axis=0)
        context_norms = np.linalg.norm(contexts, axis=1)

        summary = {
            "train/context_active_fraction": float(np.mean(active_mask)),
            "train/context_norm_mean": float(np.mean(context_norms)),
            "train/context_norm_std": float(np.std(context_norms)),
            "train/no_context_fraction": float(1.0 - np.mean(active_mask)),
            "train/support_size_mean": float(np.mean(support_sizes)),
            "train/context_between_task_variance": self._between_task_variance(
                contexts, task_instance_keys
            ),
        }
        self.last_summary = summary.copy()
        return summary

    def reset(self) -> None:
        self._contexts.clear()
        self._active_masks.clear()
        self._support_sizes.clear()
        self._task_instance_keys.clear()

    @staticmethod
    def _validate_row_count(name: str, values: np.ndarray, expected: int) -> None:
        if values.ndim != 1:
            raise ValueError(f"{name} must be a 1D array with {expected} rows")
        if values.shape[0] != expected:
            raise ValueError(
                f"{name} row count ({values.shape[0]}) must match contexts row "
                f"count ({expected})"
            )

    @staticmethod
    def _between_task_variance(
        contexts: np.ndarray, task_instance_keys: np.ndarray
    ) -> float:
        task_instance_keys = np.asarray(task_instance_keys, dtype=object).reshape(-1)
        known_key_mask = task_instance_keys != None  # noqa: E711
        known_keys = task_instance_keys[known_key_mask]
        if known_keys.size == 0:
            return 0.0

        unique_keys = tuple(dict.fromkeys(known_keys.tolist()))
        task_means = np.vstack(
            [contexts[task_instance_keys == key].mean(axis=0) for key in unique_keys]
        )
        return float(np.mean(np.var(task_means, axis=0)))
