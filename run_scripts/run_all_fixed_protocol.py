#!/usr/bin/env python3
"""
Run fixed-protocol evaluation for all algorithms (PPO, RecurrentPPO, Agri-MetaRL) and rule baseline.
Looks for models under train_data/AgriControl/<algo>/deterministic/models/ and appends to data/AgriControl/fixed_protocol/.
"""
import os
import sys
import argparse

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np
import pandas as pd
import importlib.util
_spec = importlib.util.spec_from_file_location("eval_fp", os.path.join(os.path.dirname(__file__), "evaluate_fixed_protocol.py"))
_eval_fp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_eval_fp)
ALG_MAP = _eval_fp.ALG_MAP
load_env = _eval_fp.load_env
run_episode = _eval_fp.run_episode
from gl_gym.common.utils import load_env_params, load_model_hyperparams
from gl_gym.environments.tomato_env import TomatoEnv
from gl_gym.environments.baseline import RuleBasedController


def run_rule_baseline_episode(env, controller):
    """Run one episode with rule-based controller, return same metrics as run_episode."""
    N = env.N
    episode_rewards = 0.0
    epi, revenue, heat_cost, co2_cost, elec_cost = 0.0, 0.0, 0.0, 0.0, 0.0
    temp_violation, co2_violation, rh_violation = 0, 0, 0
    obs, _ = env.reset(seed=666)
    done = False
    t = 0
    while not done and t < N:
        control = controller.predict(env.x, env.weather_data[env.timestep], env)
        obs, r, done, _, info = env.step_raw_control(control)
        episode_rewards += float(r)
        epi += info["EPI"]
        revenue += info["revenue"]
        heat_cost += info["heat_cost"]
        co2_cost += info["co2_cost"]
        elec_cost += info["elec_cost"]
        temp_violation += int(info["temp_violation"])
        co2_violation += int(info["co2_violation"])
        rh_violation += int(info["rh_violation"])
        t += 1
    return {
        "episode_return": episode_rewards,
        "revenue": revenue,
        "heat_cost": heat_cost,
        "co2_cost": co2_cost,
        "elec_cost": elec_cost,
        "EPI": epi,
        "temp_violation": temp_violation,
        "co2_violation": co2_violation,
        "rh_violation": rh_violation,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=str, default="AgriControl")
    parser.add_argument("--env_id", type=str, default="TomatoEnv")
    parser.add_argument("--model_root", type=str, default="train_data/AgriControl")
    parser.add_argument("--algorithms", type=str, nargs="+", default=["ppo", "recurrentppo", "agri_metarl"])
    parser.add_argument("--n_episodes", type=int, default=5)
    parser.add_argument("--out_dir", type=str, default="data/AgriControl/fixed_protocol")
    args = parser.parse_args()

    env_config_path = "gl_gym/configs/envs/"
    env_base_params, env_specific_params = load_env_params(args.env_id, env_config_path)
    env_specific_params = env_specific_params or {}
    env_specific_params["uncertainty_scale"] = 0.0

    all_rows = []

    # Rule baseline (no model, uses eval_options from config)
    base = dict(env_base_params)
    base["training"] = False
    spec = {k: v for k, v in (env_specific_params or {}).items() if k != "eval_options_heldout"}
    spec.setdefault("uncertainty_scale", 0.0)
    rb_env = TomatoEnv(base_env_params=base, **spec)
    rb_params = load_model_hyperparams("rule_based", args.env_id)
    rb_controller = RuleBasedController(**rb_params)
    for _ in range(args.n_episodes):
        m = run_rule_baseline_episode(rb_env, rb_controller)
        all_rows.append({"method": "rule_baseline", "run": "rb", **m})
    rb_env.close()

    # RL algorithms
    for algo in args.algorithms:
        models_dir = os.path.join(args.model_root, algo, "deterministic", "models")
        envs_dir = os.path.join(args.model_root, algo, "deterministic", "envs")
        if not os.path.isdir(models_dir):
            continue
        for run_name in os.listdir(models_dir):
            model_dir = os.path.join(models_dir, run_name)
            model_path = os.path.join(model_dir, "best_model.zip")
            env_path = os.path.join(envs_dir, run_name, "best_vecnormalize.pkl")
            if not os.path.isfile(model_path):
                continue
            env = load_env(args.env_id, env_base_params, env_specific_params, env_path)
            model = ALG_MAP[algo].load(model_path, device="cpu")
            for _ in range(args.n_episodes):
                m = run_episode(model, env)
                all_rows.append({"method": algo, "run": run_name, **m})
            env.close()

    if all_rows:
        df = pd.DataFrame(all_rows)
        agg = df.groupby("method").agg({
            "episode_return": ["mean", "std"],
            "temp_violation": ["mean", "std"],
            "rh_violation": ["mean", "std"],
            "co2_violation": ["mean", "std"],
            "revenue": ["mean", "std"],
            "heat_cost": ["mean", "std"],
            "elec_cost": ["mean", "std"],
            "co2_cost": ["mean", "std"],
            "EPI": ["mean", "std"],
        }).round(2)
        os.makedirs(args.out_dir, exist_ok=True)
        df.to_csv(os.path.join(args.out_dir, "raw.csv"), index=False)
        agg.to_csv(os.path.join(args.out_dir, "summary.csv"))
        print("Wrote", os.path.join(args.out_dir, "raw.csv"), "and summary.csv")
    else:
        print("No models found under", args.model_root)


if __name__ == "__main__":
    main()
