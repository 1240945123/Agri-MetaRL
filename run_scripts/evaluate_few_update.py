#!/usr/bin/env python3
"""
Few-update adaptation: zero-shot on held-out, then 1024/2048/4096/8192 steps of training on held-out, evaluate at each.
Output: data/AgriControl/few_update/few_update_returns.csv (method, n_updates, mean_return, std_return).
"""
import os
import sys
import argparse
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.vec_env import VecNormalize

from gl_gym.RL.utils import make_vec_env
from gl_gym.common.utils import load_env_params, load_model_hyperparams
from gl_gym.RL.agri_metarl import AgriMetaRL

ALG_MAP = {"ppo": PPO, "recurrentppo": RecurrentPPO, "agri_metarl": AgriMetaRL}
UPDATE_STEPS = [0, 1024, 2048, 4096, 8192]


def load_env(env_id, env_base_params, env_specific_params, use_heldout=True, vec_norm_path=None):
    base = dict(env_base_params)
    base["training"] = True  # for adaptation we train on held-out
    spec = dict(env_specific_params or {})
    spec.setdefault("uncertainty_scale", 0.0)
    if use_heldout and "eval_options_heldout" in spec:
        spec = {**spec, "eval_options": spec["eval_options_heldout"]}
    env = make_vec_env(env_id, base, spec, seed=666, n_envs=1, monitor_filename=None, vec_norm_kwargs={"norm_obs": True, "norm_reward": True, "gamma": 0.9631}, eval_env=False)
    if vec_norm_path and os.path.isfile(vec_norm_path):
        env = VecNormalize.load(vec_norm_path, env)
    return env


def mean_return(model, env, n_episodes=3, deterministic=True):
    N = env.get_attr("N")[0]
    returns = []
    for _ in range(n_episodes):
        result = env.reset()
        obs = result[0] if isinstance(result, (tuple, list)) else result
        states = None
        episode_starts = np.ones((1,), dtype=bool)
        ep_rew = 0.0
        for _ in range(N):
            try:
                actions, states = model.predict(obs, state=states, episode_start=episode_starts, deterministic=deterministic)
            except TypeError:
                actions, _ = model.predict(obs, deterministic=deterministic)
            obs, rewards, dones, _ = env.step(actions)
            ep_rew += float(rewards[0])
            episode_starts = dones
        returns.append(ep_rew)
    return np.mean(returns), np.std(returns) if len(returns) > 1 else 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=str, default="AgriControl")
    parser.add_argument("--env_id", type=str, default="TomatoEnv")
    parser.add_argument("--algorithms", type=str, nargs="+", default=["ppo", "recurrentppo", "agri_metarl"])
    parser.add_argument("--model_root", type=str, default="train_data/AgriControl")
    parser.add_argument("--n_episodes", type=int, default=3)
    parser.add_argument("--update_steps", type=int, nargs="+", default=UPDATE_STEPS)
    parser.add_argument("--out_dir", type=str, default="data/AgriControl/few_update")
    args = parser.parse_args()

    env_config_path = "gl_gym/configs/envs/"
    env_base_params, env_specific_params = load_env_params(args.env_id, env_config_path)
    env_specific_params = env_specific_params or {}

    rows = []
    for algo in args.algorithms:
        models_dir = os.path.join(args.model_root, algo, "deterministic", "models")
        envs_dir = os.path.join(args.model_root, algo, "deterministic", "envs")
        run_dirs = [d for d in os.listdir(models_dir) if os.path.isdir(os.path.join(models_dir, d))] if os.path.isdir(models_dir) else []
        if not run_dirs:
            continue
        run_name = run_dirs[0]
        model_path = os.path.join(models_dir, run_name, "best_model.zip")
        env_path = os.path.join(envs_dir, run_name, "best_vecnormalize.pkl")
        if not os.path.isfile(model_path):
            continue
        env = load_env(args.env_id, env_base_params, env_specific_params, use_heldout=True, vec_norm_path=env_path)
        for n_updates in args.update_steps:
            # Load with env=env when adapting (n_updates>0) so n_envs match (trained with 8, eval uses 1)
            if n_updates > 0:
                model = ALG_MAP[algo].load(model_path, env=env, device="cpu")
                model.learn(total_timesteps=n_updates, reset_num_timesteps=True)
            else:
                model = ALG_MAP[algo].load(model_path, device="cpu")
            mean_ret, std_ret = mean_return(model, env, args.n_episodes)
            rows.append({"method": algo, "n_updates": n_updates, "mean_return": mean_ret, "std_return": std_ret})
        env.close()
    if rows:
        os.makedirs(args.out_dir, exist_ok=True)
        out_path = os.path.join(args.out_dir, "few_update_returns.csv")
        pd.DataFrame(rows).to_csv(out_path, index=False)
        print("Wrote", out_path)
    else:
        print("No models found.")


if __name__ == "__main__":
    main()
