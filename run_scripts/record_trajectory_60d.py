#!/usr/bin/env python3
"""
Record 60-day trajectory (temp_air, rh_air, co2_air, uBoil, uVent, uCo2) for long_horizon_control_real_60d.png.
Output: data/AgriControl/fixed_protocol/trajectory_60d.csv
"""
import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.vec_env import VecNormalize

from gl_gym.RL.utils import make_vec_env
from gl_gym.common.utils import load_env_params
from gl_gym.RL.agri_metarl import AgriMetaRL

ALG_MAP = {"ppo": PPO, "recurrentppo": RecurrentPPO, "agri_metarl": AgriMetaRL}

# obs_names order from TomatoEnv: IndoorClimate, BasicCrop, Control, Weather, Time, WeatherForecast
# IndoorClimate: co2_air(0), temp_air(1), rh_air(2), pipe_temp(3)
# Control: uBoil(7), uCo2(8), uThScr(9), uVent(10), uLamp(11), uBlScr(12)
OBS_IDX = {"temp_air": 1, "rh_air": 2, "co2_air": 0, "uBoil": 7, "uVent": 10, "uCo2": 8}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_id", type=str, default="TomatoEnv")
    parser.add_argument("--algorithm", type=str, default="agri_metarl", choices=list(ALG_MAP))
    parser.add_argument("--model_dir", type=str, default=None, help="e.g. train_data/AgriControl/agri_metarl/deterministic/models/run1. If None, uses first available run.")
    parser.add_argument("--out_path", type=str, default="data/AgriControl/fixed_protocol/trajectory_60d.csv")
    args = parser.parse_args()

    env_config_path = "gl_gym/configs/envs/"
    env_base_params, env_specific_params = load_env_params(args.env_id, env_config_path)
    env_base_params = dict(env_base_params)
    env_base_params["training"] = False
    env_specific_params = dict(env_specific_params or {})
    env_specific_params.setdefault("uncertainty_scale", 0.0)

    env = make_vec_env(
        args.env_id,
        env_base_params,
        env_specific_params,
        seed=666,
        n_envs=1,
        monitor_filename=None,
        vec_norm_kwargs=None,
        eval_env=True,
    )
    model_dir = args.model_dir
    if model_dir is None:
        default_models = os.path.join("train_data", "AgriControl", args.algorithm, "deterministic", "models")
        if os.path.isdir(default_models):
            for run_name in os.listdir(default_models):
                d = os.path.join(default_models, run_name)
                if os.path.isfile(os.path.join(d, "best_model.zip")):
                    model_dir = d
                    break
    model_path = os.path.join(model_dir, "best_model.zip") if model_dir and os.path.isfile(os.path.join(model_dir, "best_model.zip")) else None
    if model_dir:
        env_path = os.path.join(model_dir.replace("models", "envs").rstrip(os.sep), "best_vecnormalize.pkl")
        if os.path.isfile(env_path):
            env = VecNormalize.load(env_path, env)
            env.training = False
            env.norm_reward = False

    model = None
    if model_path:
        model = ALG_MAP[args.algorithm].load(model_path, device="cpu")

    N = env.get_attr("N")[0]
    rows = []
    result = env.reset()
    obs = result[0] if isinstance(result, (tuple, list)) else result
    states = None
    episode_starts = np.ones((1,), dtype=bool)

    for t in range(N):
        if model is not None:
            try:
                actions, states = model.predict(obs, state=states, episode_start=episode_starts, deterministic=True)
            except TypeError:
                actions, _ = model.predict(obs, deterministic=True)
        else:
            actions = np.zeros((1, env.action_space.shape[0]), dtype=np.float32)

        obs, _, dones, _ = env.step(actions)

        # After step: obs is new state, env.u is the control that was applied
        if hasattr(env, "obs_rms") and env.obs_rms is not None:
            raw_obs = env.unnormalize_obs(obs)[0]
        else:
            raw_obs = obs[0] if obs.ndim > 1 else obs
        u = env.get_attr("u")[0]

        row = {
            "temp_air": float(raw_obs[OBS_IDX["temp_air"]]),
            "rh_air": float(raw_obs[OBS_IDX["rh_air"]]),
            "co2_air": float(raw_obs[OBS_IDX["co2_air"]]),
            "uBoil": float(u[0]),
            "uVent": float(u[3]),
            "uCo2": float(u[1]),
        }
        rows.append(row)
        episode_starts = dones
        if dones.any():
            break

    env.close()

    if rows:
        os.makedirs(os.path.dirname(args.out_path) or ".", exist_ok=True)
        pd.DataFrame(rows).to_csv(args.out_path, index=False)
        print("Wrote", args.out_path)
    else:
        print("No trajectory recorded.")


if __name__ == "__main__":
    main()
