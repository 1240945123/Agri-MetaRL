#!/usr/bin/env python3
"""
Train Agri-MetaRL ablation variants: no_bn, no_clip, state_context_only, advantage_only.
Each variant: 5 seeds, 2e6 steps. Output: train_data/AgriControl_ablation/agri_metarl/deterministic/models/.
"""
import os
import sys

# 在导入 torch 前强制 CPU，避免多进程加载 CUDA DLL 导致 "页面文件太小" 错误
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import argparse
import copy

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gl_gym.RL.experiment_manager import ExperimentManager
from gl_gym.common.utils import load_env_params, load_model_hyperparams

SEEDS = [42, 123, 456, 789, 1024]

# 论文消融：完整 / 无 BN / 无裁剪 / 仅状态上下文 / 仅优势输入
ABLATION_VARIANTS = {
    "no_bn": {"meta_use_batch_norm": False},
    "no_clip": {"meta_use_output_clip": False},
    "state_context_only": {"meta_use_obs_in_correction": True, "meta_use_advantage_in_correction": False},
    "advantage_only": {"meta_use_obs_in_correction": False, "meta_use_advantage_in_correction": True},
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=str, default="AgriControl")
    parser.add_argument("--env_id", type=str, default="TomatoEnv")
    parser.add_argument("--variants", type=str, nargs="+", default=list(ABLATION_VARIANTS.keys()))
    parser.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    parser.add_argument("--n_evals", type=int, default=10)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--total_timesteps", type=int, default=None, help="Override (e.g. 200000 for quick test)")
    parser.add_argument("--skip_existing", action="store_true", help="Skip variant+seed if best_model.zip exists")
    parser.add_argument("--n_envs", type=int, default=None, help="Override n_envs (e.g. 2 to reduce memory)")
    args = parser.parse_args()

    env_config_path = "gl_gym/configs/envs/"
    env_base_params, env_specific_params = load_env_params(args.env_id, env_config_path)
    base_hyperparams = load_model_hyperparams("agri_metarl", args.env_id)
    if args.total_timesteps is not None:
        base_hyperparams["total_timesteps"] = args.total_timesteps
    if args.n_envs is not None:
        base_hyperparams["n_envs"] = args.n_envs

    project = f"{args.project}_ablation"
    models_base = os.path.join("train_data", project, "agri_metarl", "deterministic", "models")

    for variant_name in args.variants:
        if variant_name not in ABLATION_VARIANTS:
            print(f"Unknown variant: {variant_name}, skip")
            continue
        overrides = ABLATION_VARIANTS[variant_name]
        for seed in args.seeds:
            run_name = f"ablation_{variant_name}_seed{seed}"
            if args.skip_existing:
                model_path = os.path.join(models_base, run_name, "best_model.zip")
                if os.path.isfile(model_path):
                    print(f"Skip (exists): {variant_name} seed={seed}")
                    continue
            hyperparameters = copy.deepcopy(base_hyperparams)
            for k, v in overrides.items():
                hyperparameters[k] = v
            group = f"ablation_{variant_name}_seed{seed}"
            print(f"Training ablation {variant_name} seed={seed} ...")
            experiment_manager = ExperimentManager(
                env_id=args.env_id,
                project=project,
                env_base_params=env_base_params,
                env_specific_params=env_specific_params,
                hyperparameters=hyperparameters,
                group=group,
                n_eval_episodes=1,
                n_evals=args.n_evals,
                algorithm="agri_metarl",
                env_seed=seed,
                model_seed=seed,
                stochastic=False,
                save_model=True,
                save_env=True,
                device=args.device,
                run_name=run_name,
            )
            experiment_manager.run_experiment()
    print("Ablation training done.")


if __name__ == "__main__":
    main()
