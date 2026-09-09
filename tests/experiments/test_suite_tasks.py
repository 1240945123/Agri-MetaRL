from pathlib import Path

from gl_gym.experiments.suite_schema import create_default_suite_config
from gl_gym.experiments.suite_tasks import (
    ECONOMIC_SCENARIOS,
    apply_task_to_env_params,
    build_evaluation_tasks,
    scenario_reward_params,
)


BASE_REWARD = {
    "elec_price": 0.3,
    "heating_price": 0.09,
    "co2_price": 0.3,
    "fruit_price": 1.6,
}


def test_economic_scenarios_are_named_and_reproducible():
    assert sorted(ECONOMIC_SCENARIOS) == [
        "combined_stress",
        "high_co2_price",
        "high_energy_price",
        "low_tomato_price",
        "standard",
    ]
    assert scenario_reward_params(BASE_REWARD, "standard") == BASE_REWARD
    high_energy = scenario_reward_params(BASE_REWARD, "high_energy_price")
    assert high_energy["elec_price"] == 0.45
    assert high_energy["heating_price"] == 0.135
    assert high_energy["fruit_price"] == 1.6


def test_build_evaluation_tasks_counts_fixed_heldout_uncertainty_and_economic(tmp_path: Path):
    suite = create_default_suite_config(
        result_root=tmp_path / "results",
        model_root=tmp_path / "models",
    )

    tasks = build_evaluation_tasks(suite)
    fixed = [task for task in tasks if task.split == "fixed"]
    heldout = [task for task in tasks if task.split == "heldout"]
    uncertainty = [task for task in tasks if task.split == "uncertainty"]
    economic = [task for task in tasks if task.split == "economic"]

    assert len(fixed) == 1
    assert len(heldout) == 9
    assert len(uncertainty) == 36
    assert len(economic) == 45
    assert len({task.task_id for task in tasks}) == len(tasks)


def test_apply_task_to_env_params_sets_single_eval_task_and_prices(tmp_path: Path):
    suite = create_default_suite_config(
        result_root=tmp_path / "results",
        model_root=tmp_path / "models",
    )
    task = next(
        item
        for item in build_evaluation_tasks(suite)
        if item.task_id == "economic_2011_d59_u0p00_high_energy_price"
    )
    base_params = {"training": True}
    specific_params = {"reward_params": dict(BASE_REWARD), "eval_options": {}}

    base_out, specific_out = apply_task_to_env_params(base_params, specific_params, task)

    assert base_out["training"] is False
    assert specific_out["uncertainty_scale"] == 0.0
    assert specific_out["economic_scenario"] == "high_energy_price"
    assert specific_out["eval_options"]["eval_years"] == [2011]
    assert specific_out["eval_options"]["eval_days"] == [59]
    assert specific_out["reward_params"]["elec_price"] == 0.45
    assert specific_out["reward_params"]["heating_price"] == 0.135
