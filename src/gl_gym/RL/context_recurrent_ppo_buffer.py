"""Context-aware rollout storage for recurrent PPO."""

from typing import NamedTuple, Optional

import numpy as np
from sb3_contrib.common.recurrent.buffers import RecurrentRolloutBuffer
from sb3_contrib.common.recurrent.type_aliases import RecurrentRolloutBufferSamples
from sb3_contrib.common.recurrent.type_aliases import RNNStates


class ContextRecurrentRolloutBufferSamples(NamedTuple):
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


def _empty_support_snapshots(buffer_size: int, n_envs: int) -> np.ndarray:
    snapshots = np.empty((buffer_size, n_envs), dtype=object)
    for index in np.ndindex(snapshots.shape):
        snapshots[index] = ()
    return snapshots


class ContextRecurrentRolloutBuffer(RecurrentRolloutBuffer):
    """Store context metadata alongside recurrent rollout rows."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.task_instance_keys = np.full(
            (self.buffer_size, self.n_envs), None, dtype=object
        )
        self.context_active_mask = np.zeros(
            (self.buffer_size, self.n_envs), dtype=bool
        )
        self.support_sizes = np.zeros((self.buffer_size, self.n_envs), dtype=np.int64)
        self.support_snapshots = _empty_support_snapshots(
            self.buffer_size, self.n_envs
        )
        self._flat_support_snapshots = None

    def reset(self) -> None:
        super().reset()
        self.task_instance_keys = np.full(
            (self.buffer_size, self.n_envs), None, dtype=object
        )
        self.context_active_mask = np.zeros(
            (self.buffer_size, self.n_envs), dtype=bool
        )
        self.support_sizes = np.zeros((self.buffer_size, self.n_envs), dtype=np.int64)
        self.support_snapshots = _empty_support_snapshots(
            self.buffer_size, self.n_envs
        )
        self._flat_support_snapshots = None

    def _metadata_row(self, values, *, name: str, dtype) -> np.ndarray:
        row = np.asarray(values, dtype=dtype).reshape(-1)
        if row.size != self.n_envs:
            raise ValueError(
                f"{name} must contain one value per environment "
                f"(expected {self.n_envs}, got {row.size})"
            )
        return row

    def _support_snapshot_row(self, snapshots) -> list[tuple]:
        try:
            row = list(snapshots)
        except TypeError as exc:
            raise ValueError(
                "support_snapshots must contain one snapshot per environment "
                f"(expected {self.n_envs})"
            ) from exc
        if len(row) != self.n_envs:
            raise ValueError(
                "support_snapshots must contain one snapshot per environment "
                f"(expected {self.n_envs}, got {len(row)})"
            )
        return [tuple(snapshot) for snapshot in row]

    def add(
        self,
        *args,
        lstm_states: RNNStates,
        task_instance_keys,
        context_active_mask,
        support_snapshots,
        support_sizes,
        **kwargs,
    ) -> None:
        key_row = self._metadata_row(
            task_instance_keys, name="task_instance_keys", dtype=object
        )
        active_row = self._metadata_row(
            context_active_mask, name="context_active_mask", dtype=bool
        )
        size_row = self._metadata_row(
            support_sizes, name="support_sizes", dtype=np.int64
        )
        snapshot_row = self._support_snapshot_row(support_snapshots)
        row_index = self.pos
        super().add(*args, lstm_states=lstm_states, **kwargs)

        self.task_instance_keys[row_index, :] = key_row
        self.context_active_mask[row_index, :] = active_row
        self.support_sizes[row_index, :] = size_row
        self.support_snapshots[row_index, :] = snapshot_row

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
    ) -> ContextRecurrentRolloutBufferSamples:
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
        return ContextRecurrentRolloutBufferSamples(
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
            sequence = [tuple(snapshot) for snapshot in selected_snapshots[start : end + 1]]
            padded.extend(sequence)
            padded.extend([()] * (max_length - len(sequence)))
        return tuple(padded)
