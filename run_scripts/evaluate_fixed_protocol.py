#!/usr/bin/env python3
"""
Fixed protocol evaluation: deterministic single episode (eval 2010 day 59).
Output: climate violations + economic metrics per method -> data/AgriControl/fixed_protocol/summary.csv
"""
import os
import sys
import argparse
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
from sb3_contrib import RecurrentPPO

from gl_gym.RL.utils import make_vec_env
from gl_gym.common.utils import load_env_params, load_model_hyperparams
from gl_gym.RL.agri_metarl import AgriMetaRL

ALG_MAP = {"ppo": PPO, "recurrentppo": RecurrentPPO, "agri_metarl": AgriMetaRL}


def load_env(env_id, env_base_params, env_specific_params, vec_norm_path=None):
    env_base_params = dict(env_base_params)
    env_base_params["training"] = False
    env_specific_params = dict(env_specific_params or {})
    env_specific_params.setdefault("uncertainty_scale", 0.0)
    env = make_vec_env(
        env_id,
        env_base_params,
        env_specific_params,
        seed=666,
        n_envs=1,
        monitor_filename=None,
        vec_norm_kwargs=None,
        eval_env=True,
    )
    if vec_norm_path and os.path.isfile(vec_norm_path):
        env = VecNormalize.load(vec_norm_path, env)
        env.training = False
        env.norm_reward = False
    return env


def run_episode(model, env, deterministic=True):
    N = env.get_attr("N")[0]
    episode_rewards = 0.0
    epi, revenue, heat_cost, co2_cost, elec_cost = 0.0, 0.0, 0.0, 0.0, 0.0
    temp_violation, co2_violation, rh_violation = 0, 0, 0
    result = env.reset()
    obs = result[0] if isinstance(result, (tuple, list)) else result
    states = None
    episode_starts = np.ones((1,), dtype=bool)
    for _ in range(N):
        if hasattr(model, "predict") and "state" in str(model.predict.__code__.co_varnames):
            actions, states = model.predict(obs, state=states, episode_start=episode_starts, deterministic=deterministic)
        else:
            actions, _ = model.predict(obs, deterministic=deterministic)
        obs, rewards, dones, infos = env.step(actions)
        episode_rewards += float(rewards[0])
        epi += infos[0]["EPI"]
        revenue += infos[0]["revenue"]
        heat_cost += infos[0]["heat_cost"]
        co2_cost += infos[0]["co2_cost"]
        elec_cost += infos[0]["elec_cost"]
        temp_violation += int(infos[0]["temp_violation"])
        co2_violation += int(infos[0]["co2_violation"])
        rh_violation += int(infos[0]["rh_violation"])
        episode_starts = dones
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
    parser.add_argument("--model_dirs", type=str, nargs="+", help="e.g. train_data/AgriControl/ppo/deterministic/models/run1")
    parser.add_argument("--algorithm", type=str, default="ppo", choices=list(ALG_MAP))
    parser.add_argument("--n_episodes", type=int, default=5)
    parser.add_argument("--out_dir", type=str, default="data/AgriControl/fixed_protocol")
    args = parser.parse_args()

    env_config_path = "gl_gym/configs/envs/"
    env_base_params, env_specific_params = load_env_params(args.env_id, env_config_path)
    env_specific_params = env_specific_params or {}
    env_specific_params["uncertainty_scale"] = 0.0

    rows = []
    if args.model_dirs:
        for model_dir in args.model_dirs:
            model_path = os.path.join(model_dir, "best_model.zip")
            env_path = os.path.join(model_dir.replace("/models/", "/envs/"), "best_vecnormalize.pkl")
            if not os.path.isfile(model_path):
                print(f"Skip (no model): {model_path}")
                continue
            env = load_env(args.env_id, env_base_params, env_specific_params, env_path)
            model = ALG_MAP[args.algorithm].load(model_path, device="cpu")
            metrics_list = [run_episode(model, env) for _ in range(args.n_episodes)]
            env.close()
            name = Path(model_dir).name
            for m in metrics_list:
                row = {"method": args.algorithm, "run": name, **m}
                rows.append(row)

    if rows:
        df = pd.DataFrame(rows)
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
        summary_path = os.path.join(args.out_dir, "summary.csv")
        df.to_csv(os.path.join(args.out_dir, "raw.csv"), index=False)
        agg.to_csv(summary_path)
        print("Summary written to", summary_path)
    else:
        print("No model dirs or no valid models; create summary CSV manually or pass --model_dirs.")


if __name__ == "__main__":
    main()
