#!/usr/bin/env python3
"""
Train-vs-test generalization: evaluate same model on train distribution (2010, day 59-96) and held-out (2011-2013, day 59/80/100).
Output: data/AgriControl/train_test_generalization/train_test_returns.csv (seed, algo, train_return, test_return).
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


def load_env(env_id, env_base_params, env_specific_params, use_heldout=False, vec_norm_path=None):
    base = dict(env_base_params)
    base["training"] = False
    spec = dict(env_specific_params or {})
    spec.setdefault("uncertainty_scale", 0.0)
    if use_heldout and "eval_options_heldout" in spec:
        spec = {**spec, "eval_options": spec["eval_options_heldout"]}
    env = make_vec_env(env_id, base, spec, seed=666, n_envs=1, monitor_filename=None, vec_norm_kwargs=None, eval_env=True)
    if vec_norm_path and os.path.isfile(vec_norm_path):
        env = VecNormalize.load(vec_norm_path, env)
        env.training = False
        env.norm_reward = False
    return env


def mean_return(model, env, n_episodes=5, deterministic=True):
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
    parser.add_argument("--algorithm", type=str, default="agri_metarl", choices=list(ALG_MAP))
    parser.add_argument("--model_root", type=str, default="train_data/AgriControl/agri_metarl/deterministic")
    parser.add_argument("--n_episodes", type=int, default=5)
    parser.add_argument("--out_dir", type=str, default="data/AgriControl/train_test_generalization")
    args = parser.parse_args()

    env_config_path = "gl_gym/configs/envs/"
    env_base_params, env_specific_params = load_env_params(args.env_id, env_config_path)
    env_specific_params = env_specific_params or {}

    models_dir = os.path.join(args.model_root, "models")
    envs_dir = args.model_root.replace("/models", "").rstrip("/") + "/envs"
    if not os.path.isdir(models_dir):
        models_dir = os.path.join(args.model_root, "models") if "models" not in args.model_root else args.model_root
        envs_dir = os.path.dirname(models_dir).replace("models", "envs")

    rows = []
    for run_name in os.listdir(models_dir) if os.path.isdir(models_dir) else []:
        model_path = os.path.join(models_dir, run_name, "best_model.zip")
        env_path = os.path.join(envs_dir, run_name, "best_vecnormalize.pkl")
        if not os.path.isfile(model_path):
            continue
        model = ALG_MAP[args.algorithm].load(model_path, device="cpu")
        env_train = load_env(args.env_id, env_base_params, env_specific_params, use_heldout=False, vec_norm_path=env_path)
        env_test = load_env(args.env_id, env_base_params, env_specific_params, use_heldout=True, vec_norm_path=env_path)
        train_mean, train_std = mean_return(model, env_train, args.n_episodes)
        test_mean, test_std = mean_return(model, env_test, args.n_episodes)
        env_train.close()
        env_test.close()
        seed = run_name.split("seed")[-1] if "seed" in run_name else run_name
        rows.append({"seed": seed, "algo": args.algorithm, "train_return": train_mean, "train_std": train_std, "test_return": test_mean, "test_std": test_std})
    if rows:
        os.makedirs(args.out_dir, exist_ok=True)
        out_path = os.path.join(args.out_dir, "train_test_returns.csv")
        pd.DataFrame(rows).to_csv(out_path, index=False)
        print("Wrote", out_path)
    else:
        print("No runs found under", models_dir)


if __name__ == "__main__":
    main()
