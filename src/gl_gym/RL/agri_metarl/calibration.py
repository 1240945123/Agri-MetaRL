from collections import OrderedDict, deque
from dataclasses import dataclass

import numpy as np

from gl_gym.RL.agri_metarl.memory import Transition


def _readonly_copy(value: np.ndarray) -> np.ndarray:
    copied = np.array(value, copy=True)
    copied.flags.writeable = False
    return copied


@dataclass(frozen=True, slots=True)
class CalibrationSample:
    observation: np.ndarray
    raw_advantage: float
    target_residual: float

    def __post_init__(self) -> None:
        observation = _readonly_copy(self.observation)
        raw_advantage = float(self.raw_advantage)
        target_residual = float(self.target_residual)
        if not np.isfinite(observation).all() or not np.isfinite(
            [raw_advantage, target_residual]
        ).all():
            raise ValueError("calibration sample contains non-finite values")
        object.__setattr__(self, "observation", observation)
        object.__setattr__(self, "raw_advantage", raw_advantage)
        object.__setattr__(self, "target_residual", target_residual)


@dataclass(frozen=True, slots=True)
class CompletedCalibrationEpisode:
    task_instance_key: str
    support: tuple[Transition, ...]
    samples: tuple[CalibrationSample, ...]
    mc_gae_abs_difference_mean: float
    target_clip_fraction: float


@dataclass(slots=True)
class _PendingStep:
    entry_id: int
    observation: np.ndarray
    reward: float
    value: float
    done: bool
    is_query: bool
    constraint_penalty: float = 0.0
    raw_advantage: float | None = None


class EpisodeCalibrationMemory:
    def __init__(
        self,
        gamma: float,
        residual_alpha: float,
        max_pending_episodes: int,
        max_completed_episodes: int,
        max_completed_query_samples: int | None = None,
        constraint_penalty_weight: float = 0.0,
    ) -> None:
        if not 0 <= gamma <= 1:
            raise ValueError("gamma must be in [0, 1]")
        if residual_alpha <= 0:
            raise ValueError("residual_alpha must be positive")
        if max_pending_episodes <= 0 or max_completed_episodes <= 0:
            raise ValueError("capacities must be positive")
        if (
            max_completed_query_samples is not None
            and max_completed_query_samples <= 0
        ):
            raise ValueError("max_completed_query_samples must be positive")
        if constraint_penalty_weight < 0:
            raise ValueError("constraint_penalty_weight must be non-negative")
        self.gamma = float(gamma)
        self.residual_alpha = float(residual_alpha)
        self.max_pending_episodes = int(max_pending_episodes)
        self.max_completed_episodes = int(max_completed_episodes)
        self.constraint_penalty_weight = float(constraint_penalty_weight)
        self.max_completed_query_samples = (
            int(max_completed_query_samples)
            if max_completed_query_samples is not None
            else None
        )
        self._pending: OrderedDict[str, list[_PendingStep]] = OrderedDict()
        self._entry_index: dict[int, tuple[str, _PendingStep]] = {}
        self._completed: deque[CompletedCalibrationEpisode] = deque(
            maxlen=self.max_completed_episodes
        )
        self._next_entry_id = 0

    def observe(
        self,
        task_instance_key: str,
        transition: Transition,
        value: float,
        is_query: bool,
        constraint_penalty: float = 0.0,
    ) -> int:
        if task_instance_key not in self._pending:
            if len(self._pending) >= self.max_pending_episodes:
                raise RuntimeError("pending episode capacity exceeded")
            self._pending[task_instance_key] = []
        numeric = np.r_[
            transition.observation.reshape(-1),
            transition.reward,
            value,
            constraint_penalty,
        ]
        if not np.isfinite(numeric).all():
            raise ValueError("episode step contains non-finite values")
        entry_id = self._next_entry_id
        self._next_entry_id += 1
        step = _PendingStep(
            entry_id=entry_id,
            observation=_readonly_copy(transition.observation),
            reward=float(transition.reward),
            value=float(value),
            done=bool(transition.done),
            is_query=bool(is_query),
            constraint_penalty=float(constraint_penalty),
        )
        self._pending[task_instance_key].append(step)
        self._entry_index[entry_id] = (task_instance_key, step)
        return entry_id

    def attach_rollout(self, entry_ids, raw_advantages) -> None:
        for entry_id, raw_advantage in zip(
            entry_ids, raw_advantages, strict=True
        ):
            entry_id = int(entry_id)
            if entry_id < 0:
                continue
            if entry_id not in self._entry_index:
                raise KeyError(f"unknown calibration entry id: {entry_id}")
            _, step = self._entry_index[entry_id]
            if step.raw_advantage is not None:
                raise ValueError(f"duplicate GAE attachment: {entry_id}")
            if not np.isfinite(raw_advantage):
                raise ValueError("raw advantage must be finite")
            step.raw_advantage = float(raw_advantage)

    def ready_task_keys(self) -> tuple[str, ...]:
        return tuple(
            key
            for key, steps in self._pending.items()
            if steps
            and steps[-1].done
            and all(step.raw_advantage is not None for step in steps)
        )

    def finalize(
        self, task_instance_key: str, support: tuple[Transition, ...]
    ) -> CompletedCalibrationEpisode:
        if task_instance_key not in self.ready_task_keys():
            raise RuntimeError("episode is not ready for finalization")
        steps = self._pending.pop(task_instance_key)
        returns = np.empty(len(steps), dtype=np.float64)
        running = 0.0
        for index in range(len(steps) - 1, -1, -1):
            running = steps[index].reward + self.gamma * running
            returns[index] = running

        query_differences = []
        samples = []
        clipped = 0
        for step, episode_return in zip(steps, returns, strict=True):
            difference = float(episode_return - step.value - step.raw_advantage)
            if step.is_query:
                penalized_difference = (
                    difference
                    - self.constraint_penalty_weight * step.constraint_penalty
                )
                target = float(
                    np.clip(
                        penalized_difference,
                        -self.residual_alpha,
                        self.residual_alpha,
                    )
                )
                query_differences.append(abs(difference))
                clipped += int(target != difference)
                samples.append(
                    CalibrationSample(
                        step.observation, step.raw_advantage, target
                    )
                )
            self._entry_index.pop(step.entry_id)

        samples = self._bounded_samples(tuple(samples))
        episode = CompletedCalibrationEpisode(
            task_instance_key=task_instance_key,
            support=tuple(support),
            samples=samples,
            mc_gae_abs_difference_mean=(
                float(np.mean(query_differences))
                if query_differences
                else 0.0
            ),
            target_clip_fraction=(
                float(clipped / len(samples)) if samples else 0.0
            ),
        )
        if samples:
            self._completed.append(episode)
            self._prune_completed_query_samples()
        return episode

    def _bounded_samples(
        self, samples: tuple[CalibrationSample, ...]
    ) -> tuple[CalibrationSample, ...]:
        if self.max_completed_query_samples is None:
            return samples
        if len(samples) <= self.max_completed_query_samples:
            return samples
        return samples[-self.max_completed_query_samples :]

    def _prune_completed_query_samples(self) -> None:
        if self.max_completed_query_samples is None:
            return
        while (
            len(self._completed) > 1
            and self.completed_query_sample_count > self.max_completed_query_samples
        ):
            self._completed.popleft()

    def pop_completed(
        self, minimum_query_samples: int, maximum_query_samples: int
    ) -> tuple[CompletedCalibrationEpisode, ...]:
        if minimum_query_samples <= 0 or maximum_query_samples <= 0:
            raise ValueError("sample limits must be positive")
        if minimum_query_samples > maximum_query_samples:
            raise ValueError("minimum samples cannot exceed maximum samples")
        selected = []
        count = 0
        while self._completed and count < maximum_query_samples:
            episode = self._completed[0]
            if selected and count + len(episode.samples) > maximum_query_samples:
                break
            selected.append(self._completed.popleft())
            count += len(episode.samples)
        if count < minimum_query_samples:
            self._completed.extendleft(reversed(selected))
            return ()
        return tuple(selected)

    @property
    def completed_episode_count(self) -> int:
        return len(self._completed)

    @property
    def completed_query_sample_count(self) -> int:
        return sum(len(episode.samples) for episode in self._completed)
