"""Task matrix and economic scenario helpers for robust experiment suites."""

from __future__ import annotations

from copy import deepcopy

from gl_gym.experiments.suite_schema import EvaluationTaskRecord, ExperimentSuiteConfig


ECONOMIC_SCENARIOS = {
    "standard": {
        "elec_price": 1.0,
        "heating_price": 1.0,
        "co2_price": 1.0,
        "fruit_price": 1.0,
    },
    "high_energy_price": {
        "elec_price": 1.5,
        "heating_price": 1.5,
        "co2_price": 1.0,
        "fruit_price": 1.0,
    },
    "low_tomato_price": {
        "elec_price": 1.0,
        "heating_price": 1.0,
        "co2_price": 1.0,
        "fruit_price": 0.7,
    },
    "high_co2_price": {
        "elec_price": 1.0,
        "heating_price": 1.0,
        "co2_price": 2.0,
        "fruit_price": 1.0,
    },
    "combined_stress": {
        "elec_price": 1.5,
        "heating_price": 1.5,
        "co2_price": 2.0,
        "fruit_price": 0.7,
    },
}


def _scale_token(value: float) -> str:
    return f"{value:.2f}".replace(".", "p")


def _task(
    suite: ExperimentSuiteConfig,
    split: str,
    year: int,
    start_day: int,
    uncertainty_scale: float,
    scenario: str,
) -> EvaluationTaskRecord:
    token = _scale_token(uncertainty_scale)
    return EvaluationTaskRecord(
        suite_id=suite.suite_id,
        task_id=f"{split}_{year}_d{start_day}_u{token}_{scenario}",
        split=split,
        weather_year=year,
        start_day=start_day,
        uncertainty_scale=uncertainty_scale,
        economic_scenario=scenario,
        climate_constraint_scenario="standard",
    )


def build_evaluation_tasks(suite: ExperimentSuiteConfig) -> list[EvaluationTaskRecord]:
    """Build fixed, heldout, uncertainty, and economic evaluation task records."""

    tasks = [
        _task(
            suite,
            "fixed",
            suite.fixed_protocol_year,
            suite.fixed_protocol_start_day,
            0.0,
            "standard",
        )
    ]

    for year in suite.evaluation_years:
        for start_day in suite.evaluation_start_days:
            tasks.append(_task(suite, "heldout", year, start_day, 0.0, "standard"))

    for year in suite.evaluation_years:
        for start_day in suite.evaluation_start_days:
            for uncertainty_scale in suite.uncertainty_scales:
                tasks.append(
                    _task(
                        suite,
                        "uncertainty",
                        year,
                        start_day,
                        uncertainty_scale,
                        "standard",
                    )
                )

    for year in suite.evaluation_years:
        for start_day in suite.evaluation_start_days:
            for scenario in suite.economic_scenarios:
                tasks.append(_task(suite, "economic", year, start_day, 0.0, scenario))

    return tasks


def scenario_reward_params(
    base_reward_params: dict[str, float],
    scenario: str,
) -> dict[str, float]:
    """Return reward parameters scaled for an economic scenario."""

    if scenario not in ECONOMIC_SCENARIOS:
        raise ValueError(f"Unknown economic scenario: {scenario}")

    reward_params = dict(base_reward_params)
    for key, multiplier in ECONOMIC_SCENARIOS[scenario].items():
        if key in reward_params:
            reward_params[key] = round(reward_params[key] * multiplier, 12)
    return reward_params


def apply_task_to_env_params(
    env_base_params: dict,
    env_specific_params: dict,
    task: EvaluationTaskRecord,
) -> tuple[dict, dict]:
    """Apply one evaluation task to environment parameter dictionaries."""

    base_out = deepcopy(env_base_params)
    specific_out = deepcopy(env_specific_params)

    base_out["training"] = False
    specific_out["uncertainty_scale"] = task.uncertainty_scale
    specific_out["economic_scenario"] = task.economic_scenario

    eval_options = specific_out.setdefault("eval_options", {})
    eval_options["eval_years"] = [task.weather_year]
    eval_options["eval_days"] = [task.start_day]

    specific_out["reward_params"] = scenario_reward_params(
        specific_out.get("reward_params", {}),
        task.economic_scenario,
    )

    return base_out, specific_out
