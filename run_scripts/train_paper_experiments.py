#!/usr/bin/env python3
"""
Paper experiments: train PPO, Recurrent PPO, Agri-MetaRL with 5 seeds each.
Output: models and logs under train_data/AgriControl/<algo>/deterministic/.
"""
import os
import sys
import argparse

# run from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import copy
import numpy as np
from gl_gym.RL.experiment_manager import ExperimentManager
from gl_gym.common.utils import load_env_params, load_model_hyperparams

SEEDS = [42, 123, 456, 789, 1024]
ALGORITHMS = ["ppo", "recurrentppo", "agri_metarl"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=str, default="AgriControl")
    parser.add_argument("--env_id", type=str, default="TomatoEnv")
    parser.add_argument("--algorithms", type=str, nargs="+", default=ALGORITHMS)
    parser.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    parser.add_argument("--n_evals", type=int, default=10)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    env_config_path = "gl_gym/configs/envs/"
    env_base_params, env_specific_params = load_env_params(args.env_id, env_config_path)

    for algo in args.algorithms:
        hyperparameters = load_model_hyperparams(algo, args.env_id)
        for seed in args.seeds:
            group = f"{algo}_paper_seed{seed}"
            print(f"Training {algo} seed={seed} ...")
            # Pass deep copy: ExperimentManager mutates hyperparameters (deletes n_envs, total_timesteps)
            experiment_manager = ExperimentManager(
                env_id=args.env_id,
                project=args.project,
                env_base_params=env_base_params,
                env_specific_params=env_specific_params,
                hyperparameters=copy.deepcopy(hyperparameters),
                group=group,
                n_eval_episodes=1,
                n_evals=args.n_evals,
                algorithm=algo,
                env_seed=seed,
                model_seed=seed,
                stochastic=False,
                save_model=True,
                save_env=True,
                device=args.device,
            )
            experiment_manager.run_experiment()
    print("Done.")


if __name__ == "__main__":
    main()
