import numpy as np

from gl_gym.RL.agri_metarl.memory import TaskSupportMemory, Transition


def transition(reward):
    value = float(reward)
    return Transition(
        observation=np.array([value]),
        action=np.array([value / 10]),
        reward=value,
        next_observation=np.array([value + 1]),
        done=False,
    )


def test_support_persists_across_rollouts():
    memory = TaskSupportMemory(support_size=3, max_instances=4)
    for step in range(2):
        assert memory.observe("task-a", transition(step)) is False
    memory.begin_rollout()
    assert memory.observe("task-a", transition(2)) is False
    assert memory.observe("task-a", transition(3)) is True
    assert len(memory.support("task-a")) == 3


def test_tasks_never_share_support():
    memory = TaskSupportMemory(support_size=2, max_instances=4)
    memory.observe("task-a", transition(1))
    memory.observe("task-b", transition(2))
    assert memory.support("task-a")[0].reward == 1
    assert memory.support("task-b")[0].reward == 2


def test_support_freezes_before_query():
    memory = TaskSupportMemory(support_size=2, max_instances=4)
    memory.observe("task-a", transition(1))
    memory.observe("task-a", transition(2))
    memory.observe("task-a", transition(99))
    assert [item.reward for item in memory.support("task-a")] == [1, 2]


def test_oldest_task_instance_is_evicted_at_capacity():
    memory = TaskSupportMemory(support_size=2, max_instances=2)
    memory.observe("task-a", transition(1))
    memory.observe("task-b", transition(2))
    memory.observe("task-c", transition(3))
    assert memory.support("task-a") == ()
    assert memory.support("task-b")[0].reward == 2
    assert memory.support("task-c")[0].reward == 3
