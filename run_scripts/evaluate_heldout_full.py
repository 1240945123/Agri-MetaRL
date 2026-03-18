#!/usr/bin/env python3
"""
Held-out (Test) full metrics evaluation: Return, EPI, TWB for all RL algorithms.
Uses eval_options_heldout (2011-2013, day 59/80/100). Output: data/AgriControl/train_test_generalization/heldout_full.csv
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
run_episode = _eval_fp.run_episode

from gl_gym.RL.utils import make_vec_env
from gl_gym.common.utils import load_env_params
from stable_baselines3.common.vec_env import VecNormalize

N_STEPS = 5760  # 60 days * 96 steps/day


def load_heldout_env(env_id, env_base_params, env_specific_params, vec_norm_path=None):
    """Load env with held-out eval options (2011-2013)."""
    base = dict(env_base_params)
    base["training"] = False
    spec = dict(env_specific_params or {})
    spec.setdefault("uncertainty_scale", 0.0)
    if "eval_options_heldout" in spec:
        spec = {**spec, "eval_options": spec["eval_options_heldout"]}
    env = make_vec_env(
        env_id, base, spec,
        seed=666, n_envs=1, monitor_filename=None, vec_norm_kwargs=None, eval_env=True,
    )
    if vec_norm_path and os.path.isfile(vec_norm_path):
        env = VecNormalize.load(vec_norm_path, env)
        env.training = False
        env.norm_reward = False
    return env


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_root", type=str, default="train_data/AgriControl")
    parser.add_argument("--algorithms", type=str, nargs="+", default=["ppo", "recurrentppo", "agri_metarl"])
    parser.add_argument("--n_episodes", type=int, default=3, help="Episodes per seed (reduce if timeout)")
    parser.add_argument("--out_dir", type=str, default="data/AgriControl/train_test_generalization")
    args = parser.parse_args()

    env_config_path = "gl_gym/configs/envs/"
    env_base_params, env_specific_params = load_env_params("TomatoEnv", env_config_path)
    env_specific_params = env_specific_params or {}
    env_specific_params["uncertainty_scale"] = 0.0

    all_rows = []
    for algo in args.algorithms:
        models_dir = os.path.join(args.model_root, algo, "deterministic", "models")
        envs_dir = os.path.join(args.model_root, algo, "deterministic", "envs")
        if not os.path.isdir(models_dir):
            print(f"Skip {algo}: no models dir")
            continue
        for run_name in os.listdir(models_dir):
            model_path = os.path.join(models_dir, run_name, "best_model.zip")
            env_path = os.path.join(envs_dir, run_name, "best_vecnormalize.pkl")
            if not os.path.isfile(model_path):
                continue
            env = load_heldout_env("TomatoEnv", env_base_params, env_specific_params, env_path)
            model = ALG_MAP[algo].load(model_path, device="cpu")
            for ep in range(args.n_episodes):
                m = run_episode(model, env)
                all_rows.append({"method": algo, "run": run_name, **m})
            env.close()
            print(f"  {algo}/{run_name}: done")

    if all_rows:
        df = pd.DataFrame(all_rows)
        agg = df.groupby("method").agg({
            "episode_return": ["mean", "std"],
            "temp_violation": ["mean", "std"],
            "rh_violation": ["mean", "std"],
            "co2_violation": ["mean", "std"],
            "EPI": ["mean", "std"],
        }).round(4)
        os.makedirs(args.out_dir, exist_ok=True)
        out_path = os.path.join(args.out_dir, "heldout_full.csv")
        df.to_csv(out_path, index=False)
        agg_path = os.path.join(args.out_dir, "heldout_summary.csv")
        agg.to_csv(agg_path)
        print(f"Wrote {out_path} and {agg_path}")
    else:
        print("No models found.")


if __name__ == "__main__":
    main()
