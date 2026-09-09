from collections import OrderedDict
from dataclasses import dataclass

import numpy as np


def _readonly_copy(value: np.ndarray) -> np.ndarray:
    copied = np.array(value, copy=True)
    copied.flags.writeable = False
    return copied


@dataclass(frozen=True, slots=True)
class Transition:
    observation: np.ndarray
    action: np.ndarray
    reward: float
    next_observation: np.ndarray
    done: bool

    def __post_init__(self) -> None:
        object.__setattr__(self, "observation", _readonly_copy(self.observation))
        object.__setattr__(self, "action", _readonly_copy(self.action))
        object.__setattr__(
            self, "next_observation", _readonly_copy(self.next_observation)
        )
        object.__setattr__(self, "reward", float(self.reward))
        object.__setattr__(self, "done", bool(self.done))


class TaskSupportMemory:
    def __init__(self, support_size: int, max_instances: int) -> None:
        if support_size <= 0:
            raise ValueError("support_size must be positive")
        if max_instances <= 0:
            raise ValueError("max_instances must be positive")
        self.support_size = support_size
        self.max_instances = max_instances
        self._supports: OrderedDict[str, list[Transition]] = OrderedDict()

    def begin_rollout(self) -> None:
        """Mark a rollout boundary without clearing cross-rollout support."""

    def observe(self, task_instance_key: str, transition: Transition) -> bool:
        if task_instance_key not in self._supports:
            if len(self._supports) >= self.max_instances:
                self._supports.popitem(last=False)
            self._supports[task_instance_key] = []

        support = self._supports[task_instance_key]
        if len(support) < self.support_size:
            support.append(transition)
            return False
        return True

    def support(self, task_instance_key: str) -> tuple[Transition, ...]:
        return tuple(self._supports.get(task_instance_key, ()))
