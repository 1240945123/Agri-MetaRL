"""
Rollout buffer that stores task_id per step for support/query split in Agri-MetaRL.
"""
import numpy as np
from gymnasium import spaces
from sb3_contrib.common.recurrent.buffers import RecurrentRolloutBuffer
from sb3_contrib.common.recurrent.type_aliases import RNNStates


def encode_task_id(year: int, day: int) -> int:
    """Encode (year, day) as single int for buffer storage."""
    return int(year) * 1000 + int(day)


class AgriMetaRLRolloutBuffer(RecurrentRolloutBuffer):
    """RecurrentRolloutBuffer that also stores task_id (year, day) per step per env for support/query split."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.task_ids = np.zeros((self.buffer_size, self.n_envs), dtype=np.int32)

    def reset(self) -> None:
        super().reset()
        self.task_ids = np.zeros((self.buffer_size, self.n_envs), dtype=np.int32)

    def add(self, *args, lstm_states: RNNStates, task_ids=None, **kwargs) -> None:
        super().add(*args, lstm_states=lstm_states, **kwargs)
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
            self.task_ids[self.pos - 1, :] = row
