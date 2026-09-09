import numpy as np
import os
import pytest
import subprocess
import sys
from pathlib import Path

from gl_gym.RL.agri_metarl.calibration import (
    CalibrationSample,
    EpisodeCalibrationMemory,
)
from gl_gym.RL.agri_metarl.memory import Transition


def transition(reward: float, done: bool = False) -> Transition:
    return Transition(
        observation=np.array([reward, 0.0], dtype=np.float32),
        action=np.array([0.0], dtype=np.float32),
        reward=reward,
        next_observation=np.array([reward + 1.0, 0.0], dtype=np.float32),
        done=done,
    )


def test_three_rollout_fragments_finalize_only_after_terminal_and_gae_attachment():
    memory = EpisodeCalibrationMemory(
        gamma=0.5,
        residual_alpha=2.0,
        max_pending_episodes=2,
        max_completed_episodes=2,
    )
    entry_ids = []
    entry_ids.append(
        memory.observe("task-a", transition(1.0), value=0.5, is_query=False)
    )
    memory.attach_rollout([entry_ids[-1]], [0.25])
    assert memory.ready_task_keys() == ()
    entry_ids.append(
        memory.observe("task-a", transition(2.0), value=1.0, is_query=True)
    )
    memory.attach_rollout([entry_ids[-1]], [0.75])
    assert memory.ready_task_keys() == ()
    entry_ids.append(
        memory.observe(
            "task-a", transition(4.0, done=True), value=1.5, is_query=True
        )
    )
    assert memory.ready_task_keys() == ()
    memory.attach_rollout([entry_ids[-1]], [1.25])
    assert memory.ready_task_keys() == ("task-a",)

    support = (transition(1.0),)
    completed = memory.finalize("task-a", support)

    assert [sample.target_residual for sample in completed.samples] == [2.0, 1.25]
    assert [sample.raw_advantage for sample in completed.samples] == [0.75, 1.25]
    assert completed.support == support
    assert memory.ready_task_keys() == ()


def test_task_instances_never_share_pending_steps():
    memory = EpisodeCalibrationMemory(0.9, 1.0, 4, 4)
    a = memory.observe("task-a", transition(1.0, done=True), 0.0, True)
    b = memory.observe("task-b", transition(9.0, done=True), 0.0, True)
    memory.attach_rollout([a, b], [0.2, 0.8])
    episode_a = memory.finalize("task-a", (transition(1.0),))
    episode_b = memory.finalize("task-b", (transition(9.0),))
    assert episode_a.task_instance_key == "task-a"
    assert episode_b.task_instance_key == "task-b"
    assert episode_a.samples[0].observation[0] == 1.0
    assert episode_b.samples[0].observation[0] == 9.0


def test_pending_capacity_raises_instead_of_dropping_active_episode():
    memory = EpisodeCalibrationMemory(0.9, 1.0, 1, 2)
    memory.observe("task-a", transition(1.0), 0.0, False)
    with pytest.raises(RuntimeError, match="pending episode capacity"):
        memory.observe("task-b", transition(2.0), 0.0, False)


def test_duplicate_and_unknown_gae_attachments_are_rejected():
    memory = EpisodeCalibrationMemory(0.9, 1.0, 2, 2)
    entry_id = memory.observe("task-a", transition(1.0), 0.0, False)
    memory.attach_rollout([entry_id], [0.2])
    with pytest.raises(ValueError, match="duplicate GAE attachment"):
        memory.attach_rollout([entry_id], [0.3])
    with pytest.raises(KeyError, match="unknown calibration entry id"):
        memory.attach_rollout([999], [0.1])


def test_nonfinite_step_and_sample_are_rejected():
    memory = EpisodeCalibrationMemory(0.9, 1.0, 2, 2)
    with pytest.raises(ValueError, match="non-finite"):
        memory.observe("task-a", transition(np.nan), 0.0, True)
    with pytest.raises(ValueError, match="non-finite"):
        CalibrationSample(np.array([0.0]), 0.0, np.nan)


def test_completed_queue_evicts_oldest_episode():
    memory = EpisodeCalibrationMemory(0.9, 1.0, 3, 1)
    for key, reward in (("task-a", 1.0), ("task-b", 2.0)):
        entry_id = memory.observe(key, transition(reward, done=True), 0.0, True)
        memory.attach_rollout([entry_id], [0.0])
        memory.finalize(key, (transition(reward),))
    episodes = memory.pop_completed(1, 10)
    assert [episode.task_instance_key for episode in episodes] == ["task-b"]


def test_completed_queue_evicts_oldest_episodes_to_bound_query_samples():
    memory = EpisodeCalibrationMemory(
        0.9,
        1.0,
        max_pending_episodes=4,
        max_completed_episodes=4,
        max_completed_query_samples=3,
    )
    for episode_index in range(3):
        key = f"task-{episode_index}"
        ids = [
            memory.observe(key, transition(float(episode_index), done=False), 0.0, True),
            memory.observe(key, transition(float(episode_index), done=True), 0.0, True),
        ]
        memory.attach_rollout(ids, [0.0, 0.0])
        memory.finalize(key, (transition(float(episode_index)),))

    assert memory.completed_query_sample_count == 2
    episodes = memory.pop_completed(1, 10)
    assert [episode.task_instance_key for episode in episodes] == ["task-2"]


def test_single_oversized_completed_episode_is_trimmed_to_query_sample_bound():
    memory = EpisodeCalibrationMemory(
        0.9,
        1.0,
        max_pending_episodes=1,
        max_completed_episodes=2,
        max_completed_query_samples=2,
    )
    ids = [
        memory.observe("task-a", transition(1.0, done=False), 0.0, True),
        memory.observe("task-a", transition(2.0, done=False), 0.0, True),
        memory.observe("task-a", transition(3.0, done=True), 0.0, True),
    ]
    memory.attach_rollout(ids, [0.0, 0.0, 0.0])

    episode = memory.finalize("task-a", (transition(1.0),))

    assert len(episode.samples) == 2
    assert memory.completed_query_sample_count == 2


def test_constraint_penalty_reduces_query_residual_target():
    memory = EpisodeCalibrationMemory(
        gamma=0.0,
        residual_alpha=2.0,
        max_pending_episodes=1,
        max_completed_episodes=2,
        constraint_penalty_weight=0.5,
    )
    entry_id = memory.observe(
        "task-a",
        transition(1.0, done=True),
        value=0.0,
        is_query=True,
        constraint_penalty=2.0,
    )
    memory.attach_rollout([entry_id], [0.0])

    episode = memory.finalize("task-a", (transition(1.0),))

    assert episode.samples[0].target_residual == 0.0


def test_clip_fraction_counts_query_rows_only():
    memory = EpisodeCalibrationMemory(0.0, 1.0, 1, 2)
    support_id = memory.observe("task-a", transition(100.0), 0.0, False)
    query_id = memory.observe("task-a", transition(2.0, done=True), 0.0, True)
    memory.attach_rollout([support_id, query_id], [0.0, 0.0])
    episode = memory.finalize("task-a", (transition(100.0),))
    assert episode.target_clip_fraction == 1.0
    assert episode.mc_gae_abs_difference_mean == 2.0


def test_calibration_sample_owns_read_only_observation():
    source = np.array([1.0, 2.0])
    sample = CalibrationSample(source, 0.5, -0.25)
    source[0] = 99.0
    assert sample.observation.tolist() == [1.0, 2.0]
    assert not sample.observation.flags.writeable


def test_lightweight_calibration_import_does_not_load_training_stack():
    code = (
        "import sys; "
        "from gl_gym.RL.agri_metarl.calibration import EpisodeCalibrationMemory; "
        "assert 'stable_baselines3' not in sys.modules; "
        "assert 'pyarrow' not in sys.modules"
    )
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(Path.cwd() / "src")
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=False,
        env=environment,
    )
    assert result.returncode == 0, result.stderr
