#!/usr/bin/env python3
"""
Record 24-hour trajectory for PPO, Recurrent PPO, Agri-MetaRL.
Output: data/AgriControl/fixed_protocol/trajectory_24h.csv (method, step, temp_air, rh_air, co2_air, uBoil, uVent, uCo2)
24h = 96 steps (15-min intervals).
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
OBS_IDX = {"temp_air": 1, "rh_air": 2, "co2_air": 0, "uBoil": 7, "uVent": 10, "uCo2": 8}
STEPS_24H = 96


def record_trajectory(algorithm, model_dir, env, n_steps=STEPS_24H):
    """Record n_steps of trajectory for one algorithm."""
    model_path = os.path.join(model_dir, "best_model.zip") if model_dir else None
    model = ALG_MAP[algorithm].load(model_path, device="cpu") if model_path and os.path.isfile(model_path) else None

    result = env.reset()
    obs = result[0] if isinstance(result, (tuple, list)) else result
    states = None
    episode_starts = np.ones((1,), dtype=bool)
    rows = []

    for t in range(n_steps):
        if model is not None:
            try:
                actions, states = model.predict(obs, state=states, episode_start=episode_starts, deterministic=True)
            except TypeError:
                actions, _ = model.predict(obs, deterministic=True)
        else:
            actions = np.zeros((1, env.action_space.shape[0]), dtype=np.float32)

        obs, _, dones, _ = env.step(actions)

        if hasattr(env, "obs_rms") and env.obs_rms is not None:
            raw_obs = env.unnormalize_obs(obs)[0]
        else:
            raw_obs = obs[0] if obs.ndim > 1 else obs
        u = env.get_attr("u")[0]

        rows.append({
            "method": algorithm,
            "step": t,
            "temp_air": float(raw_obs[OBS_IDX["temp_air"]]),
            "rh_air": float(raw_obs[OBS_IDX["rh_air"]]),
            "co2_air": float(raw_obs[OBS_IDX["co2_air"]]),
            "uBoil": float(u[0]),
            "uVent": float(u[3]),
            "uCo2": float(u[1]),
        })
        episode_starts = dones
        if dones.any():
            break

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_id", type=str, default="TomatoEnv")
    parser.add_argument("--algorithms", type=str, nargs="+", default=["ppo", "recurrentppo", "agri_metarl"])
    parser.add_argument("--model_root", type=str, default="train_data/AgriControl")
    parser.add_argument("--out_path", type=str, default="data/AgriControl/fixed_protocol/trajectory_24h.csv")
    parser.add_argument("--n_steps", type=int, default=STEPS_24H)
    args = parser.parse_args()

    env_config_path = "gl_gym/configs/envs/"
    env_base_params, env_specific_params = load_env_params(args.env_id, env_config_path)
    env_base_params = dict(env_base_params)
    env_base_params["training"] = False
    env_specific_params = dict(env_specific_params or {})
    env_specific_params.setdefault("uncertainty_scale", 0.0)

    all_rows = []
    for algo in args.algorithms:
        models_dir = os.path.join(args.model_root, algo, "deterministic", "models")
        model_dir = None
        if os.path.isdir(models_dir):
            for run_name in sorted(os.listdir(models_dir)):
                d = os.path.join(models_dir, run_name)
                if os.path.isfile(os.path.join(d, "best_model.zip")):
                    model_dir = d
                    break

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
        if model_dir:
            env_path = os.path.join(model_dir.replace("models", "envs").rstrip(os.sep), "best_vecnormalize.pkl")
            if os.path.isfile(env_path):
                env = VecNormalize.load(env_path, env)
                env.training = False
                env.norm_reward = False

        df = record_trajectory(algo, model_dir, env, args.n_steps)
        all_rows.append(df)
        env.close()

    if all_rows:
        os.makedirs(os.path.dirname(args.out_path) or ".", exist_ok=True)
        pd.concat(all_rows, ignore_index=True).to_csv(args.out_path, index=False)
        print("Wrote", args.out_path, "with", sum(len(r) for r in all_rows), "rows")
    else:
        print("No trajectory recorded.")


if __name__ == "__main__":
    main()
