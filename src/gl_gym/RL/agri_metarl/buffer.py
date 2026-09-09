"""Rollout storage for task-aware recurrent training."""
from typing import NamedTuple, Optional

import numpy as np
from gymnasium import spaces
from sb3_contrib.common.recurrent.buffers import RecurrentRolloutBuffer
from sb3_contrib.common.recurrent.type_aliases import (
    RecurrentRolloutBufferSamples,
    RNNStates,
)


def encode_task_id(year: int, day: int) -> int:
    """Encode (year, day) as single int for buffer storage."""
    return int(year) * 1000 + int(day)


def _empty_support_snapshots(buffer_size: int, n_envs: int) -> np.ndarray:
    snapshots = np.empty((buffer_size, n_envs), dtype=object)
    for index in np.ndindex(snapshots.shape):
        snapshots[index] = ()
    return snapshots


class AgriMetaRLRolloutBufferSamples(NamedTuple):
    observations: object
    actions: object
    old_values: object
    old_log_prob: object
    advantages: object
    returns: object
    lstm_states: RNNStates
    episode_starts: object
    mask: object
    support_snapshots: tuple[tuple, ...]


class AgriMetaRLRolloutBuffer(RecurrentRolloutBuffer):
    """Store task-instance identity and support/query role for every row."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.task_ids = np.zeros((self.buffer_size, self.n_envs), dtype=np.int32)
        self.task_instance_keys = np.full(
            (self.buffer_size, self.n_envs), None, dtype=object
        )
        self.query_mask = np.zeros((self.buffer_size, self.n_envs), dtype=bool)
        self.calibration_entry_ids = np.full(
            (self.buffer_size, self.n_envs), -1, dtype=np.int64
        )
        self.support_sizes = np.zeros((self.buffer_size, self.n_envs), dtype=np.int64)
        self.support_snapshots = _empty_support_snapshots(
            self.buffer_size, self.n_envs
        )
        self._flat_support_snapshots = None

    def reset(self) -> None:
        super().reset()
        self.task_ids = np.zeros((self.buffer_size, self.n_envs), dtype=np.int32)
        self.task_instance_keys = np.full(
            (self.buffer_size, self.n_envs), None, dtype=object
        )
        self.query_mask = np.zeros((self.buffer_size, self.n_envs), dtype=bool)
        self.calibration_entry_ids = np.full(
            (self.buffer_size, self.n_envs), -1, dtype=np.int64
        )
        self.support_sizes = np.zeros((self.buffer_size, self.n_envs), dtype=np.int64)
        self.support_snapshots = _empty_support_snapshots(
            self.buffer_size, self.n_envs
        )
        self._flat_support_snapshots = None

    def add(
        self,
        *args,
        lstm_states: RNNStates,
        task_instance_keys=None,
        query_mask=None,
        calibration_entry_ids=None,
        support_snapshots=None,
        support_sizes=None,
        task_ids=None,
        **kwargs,
    ) -> None:
        row_index = self.pos
        super().add(*args, lstm_states=lstm_states, **kwargs)
        if task_instance_keys is not None:
            keys = np.asarray(task_instance_keys, dtype=object).reshape(-1)
            if keys.size != self.n_envs:
                raise ValueError("task_instance_keys must contain one key per environment")
            self.task_instance_keys[row_index, :] = keys
        if query_mask is not None:
            mask = np.asarray(query_mask, dtype=bool).reshape(-1)
            if mask.size != self.n_envs:
                raise ValueError("query_mask must contain one flag per environment")
            self.query_mask[row_index, :] = mask
        if calibration_entry_ids is not None:
            entry_ids = np.asarray(calibration_entry_ids, dtype=np.int64).reshape(-1)
            if entry_ids.size != self.n_envs:
                raise ValueError(
                    "calibration_entry_ids must contain one ID per environment"
            )
            self.calibration_entry_ids[row_index, :] = entry_ids
        if support_snapshots is not None:
            try:
                snapshot_row = list(support_snapshots)
            except TypeError as exc:
                raise ValueError(
                    "support_snapshots must contain one snapshot per environment"
                ) from exc
            if len(snapshot_row) != self.n_envs:
                raise ValueError(
                    "support_snapshots must contain one snapshot per environment"
                )
            self.support_snapshots[row_index, :] = [
                tuple(snapshot) for snapshot in snapshot_row
            ]
        if support_sizes is not None:
            sizes = np.asarray(support_sizes, dtype=np.int64).reshape(-1)
            if sizes.size != self.n_envs:
                raise ValueError(
                    "support_sizes must contain one size per environment"
                )
            self.support_sizes[row_index, :] = sizes
        if task_ids is not None:
            # task_ids: (n_envs,) of (year, day) tuples or encoded int
            row = np.zeros(self.n_envs, dtype=np.int32)
            for env_idx in range(self.n_envs):
                t = task_ids[env_idx]
                arr = np.asarray(t)
                if arr.size >= 2:
                    row[env_idx] = encode_task_id(int(arr.flat[0]), int(arr.flat[1]))
                else:
                    row[env_idx] = int(arr.flat[0])
            self.task_ids[row_index, :] = row

    def get(self, batch_size: Optional[int] = None):
        if not self.generator_ready:
            self._flat_support_snapshots = self.swap_and_flatten(
                self.support_snapshots
            ).reshape(-1)
        yield from super().get(batch_size)

    def _get_samples(
        self,
        batch_inds: np.ndarray,
        env_change: np.ndarray,
        env=None,
    ) -> AgriMetaRLRolloutBufferSamples:
        samples: RecurrentRolloutBufferSamples = super()._get_samples(
            batch_inds,
            env_change,
            env,
        )
        flat_support_snapshots = self._flat_support_snapshots
        if flat_support_snapshots is None:
            flat_support_snapshots = self.swap_and_flatten(
                self.support_snapshots
            ).reshape(-1)
        support_snapshots = self._pad_support_snapshots(
            flat_support_snapshots[batch_inds],
            samples.observations.shape[0],
        )
        return AgriMetaRLRolloutBufferSamples(
            observations=samples.observations,
            actions=samples.actions,
            old_values=samples.old_values,
            old_log_prob=samples.old_log_prob,
            advantages=samples.advantages,
            returns=samples.returns,
            lstm_states=samples.lstm_states,
            episode_starts=samples.episode_starts,
            mask=samples.mask,
            support_snapshots=support_snapshots,
        )

    def _pad_support_snapshots(
        self,
        selected_snapshots: np.ndarray,
        padded_batch_size: int,
    ) -> tuple[tuple, ...]:
        seq_start_indices = self.seq_start_indices
        seq_end_indices = np.concatenate(
            [(seq_start_indices - 1)[1:], np.array([len(selected_snapshots) - 1])]
        )
        max_length = padded_batch_size // len(seq_start_indices)
        padded: list[tuple] = []
        for start, end in zip(seq_start_indices, seq_end_indices):
            sequence = [
                tuple(snapshot) for snapshot in selected_snapshots[start : end + 1]
            ]
            padded.extend(sequence)
            padded.extend([()] * (max_length - len(sequence)))
        return tuple(padded)
