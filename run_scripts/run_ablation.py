#!/usr/bin/env python3
"""
Ablation: run fixed-protocol evaluation on multiple variant model dirs and aggregate to ablation_summary.csv.
Variants (train separately with meta flags): full, no_bn, no_clip, state_context_only, advantage_only.
Usage:
  - Manual: --variant_dirs "full:path/to/full" "no_bn:path/to/no_bn" ...
  - Auto: --ablation_base_dir train_data/AgriControl_ablation/agri_metarl/deterministic
          --full_base_dir train_data/AgriControl/agri_metarl/deterministic (for full variant)
"""
import os
import sys
import argparse
import re

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from stable_baselines3.common.vec_env import VecNormalize
from gl_gym.RL.utils import make_vec_env
from gl_gym.common.utils import load_env_params
from gl_gym.RL.agri_metarl import AgriMetaRL

ALG_MAP = {"agri_metarl": AgriMetaRL}
ABLATION_VARIANTS = ["no_bn", "no_clip", "state_context_only", "advantage_only"]


def _discover_ablation_dirs(ablation_base_dir, full_base_dir):
    """Discover variant -> list of model dirs from train_ablation_variants output."""
    variant_to_dirs = {}
    models_dir = os.path.join(ablation_base_dir, "models")
    envs_dir = os.path.join(ablation_base_dir, "envs")
    if os.path.isdir(models_dir):
        for run_name in os.listdir(models_dir):
            m = re.match(r"ablation_(.+)_seed(\d+)$", run_name)
            if m:
                variant = m.group(1)
                if variant in ABLATION_VARIANTS:
                    d = os.path.join(models_dir, run_name)
                    if os.path.isfile(os.path.join(d, "best_model.zip")):
                        variant_to_dirs.setdefault(variant, []).append(d)
    if full_base_dir:
        full_models = os.path.join(full_base_dir, "models")
        if os.path.isdir(full_models):
            for run_name in os.listdir(full_models):
                d = os.path.join(full_models, run_name)
                if os.path.isfile(os.path.join(d, "best_model.zip")):
                    variant_to_dirs.setdefault("full", []).append(d)
                    break  # use first available full model
    return variant_to_dirs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_id", type=str, default="TomatoEnv")
    parser.add_argument("--algorithm", type=str, default="agri_metarl")
    parser.add_argument("--variant_dirs", type=str, nargs="+", help='e.g. "full:path/to/full" "no_bn:path/to/no_bn"')
    parser.add_argument("--ablation_base_dir", type=str, default=None, help="Auto-discover ablation models")
    parser.add_argument("--full_base_dir", type=str, default=None, help="Path to full agri_metarl models for 'full' variant")
    parser.add_argument("--n_episodes", type=int, default=5)
    parser.add_argument("--out_dir", type=str, default="data/AgriControl/ablation")
    args = parser.parse_args()

    env_config_path = "gl_gym/configs/envs/"
    env_base_params, env_specific_params = load_env_params(args.env_id, env_config_path)
    env_specific_params = env_specific_params or {}
    env_specific_params["uncertainty_scale"] = 0.0

    def load_env(env_id, env_base_params, env_specific_params, vec_norm_path=None):
        base = dict(env_base_params)
        base["training"] = False
        spec = dict(env_specific_params or {})
        env = make_vec_env(env_id, base, spec, seed=666, n_envs=1, monitor_filename=None, vec_norm_kwargs=None, eval_env=True)
        if vec_norm_path and os.path.isfile(vec_norm_path):
            env = VecNormalize.load(vec_norm_path, env)
            env.training = False
            env.norm_reward = False
        return env

    def run_episode(model, env):
        N = env.get_attr("N")[0]
        ep_rew = 0.0
        result = env.reset()
        obs = result[0] if isinstance(result, (tuple, list)) else result
        states = None
        episode_starts = np.ones((1,), dtype=bool)
        for _ in range(N):
            actions, states = model.predict(obs, state=states, episode_start=episode_starts, deterministic=True)
            obs, rewards, dones, _ = env.step(actions)
            ep_rew += float(rewards[0])
            episode_starts = dones
        return {"episode_return": ep_rew}

    # Build variant -> list of model dirs
    if args.ablation_base_dir:
        variant_to_dirs = _discover_ablation_dirs(args.ablation_base_dir, args.full_base_dir)
        specs = []
        for v, dirs in variant_to_dirs.items():
            for d in dirs:
                specs.append(f"{v}:{d}")
        if args.variant_dirs:
            specs = list(args.variant_dirs) + specs
        args.variant_dirs = specs
    elif not args.variant_dirs:
        args.variant_dirs = []

    rows = []
    variant_seed_returns = {}  # variant -> [mean_per_seed, ...]

    for spec in args.variant_dirs:
        if ":" not in spec:
            continue
        name, model_dir = spec.split(":", 1)
        model_path = os.path.join(model_dir, "best_model.zip")
        # envs/ mirrors models/: .../deterministic/envs/<run_name>/best_vecnormalize.pkl
        parent = os.path.dirname(os.path.dirname(model_dir))
        run_name = os.path.basename(model_dir)
        env_path = os.path.join(parent, "envs", run_name, "best_vecnormalize.pkl")
        if not os.path.isfile(model_path):
            print("Skip (no model):", model_path)
            continue
        env = load_env(args.env_id, env_base_params, env_specific_params, vec_norm_path=env_path)
        model = ALG_MAP[args.algorithm].load(model_path, device="cpu")
        returns = [run_episode(model, env)["episode_return"] for _ in range(args.n_episodes)]
        env.close()
        mean_ret = float(pd.Series(returns).mean())
        std_ret = float(pd.Series(returns).std()) if len(returns) > 1 else 0.0
        variant_seed_returns.setdefault(name, []).append(mean_ret)

    # Aggregate: per variant, mean±std across seeds
    for variant, seed_means in variant_seed_returns.items():
        arr = np.array(seed_means)
        rows.append({
            "variant": variant,
            "mean_return": float(arr.mean()),
            "std_return": float(arr.std()) if len(arr) > 1 else 0.0,
        })

    if rows:
        os.makedirs(args.out_dir, exist_ok=True)
        out_path = os.path.join(args.out_dir, "ablation_summary.csv")
        pd.DataFrame(rows).to_csv(out_path, index=False)
        print("Wrote", out_path)
    else:
        print("No variant dirs or no valid models.")


if __name__ == "__main__":
    main()
