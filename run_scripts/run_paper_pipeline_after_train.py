#!/usr/bin/env python3
"""
Run paper pipeline after training: fixed protocol -> generalization -> few-update -> ablation -> figures.
Call after train_paper_experiments.py has produced models under train_data/AgriControl/<algo>/deterministic/.
"""
import os
import sys
import subprocess
import argparse

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)


def run(cmd, desc=""):
    print("\n---", desc or cmd)
    r = subprocess.run(cmd, shell=True)
    if r.returncode != 0:
        print("Command failed:", cmd, "exit", r.returncode)
        return False
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-train", action="store_true", help="Skip (training already done)")
    parser.add_argument("--train-ablation", action="store_true", help="Train ablation variants (4 variants x 5 seeds, long)")
    parser.add_argument("--model_root", type=str, default="train_data/AgriControl")
    parser.add_argument("--out_dir", type=str, default="data/AgriControl")
    parser.add_argument("--n_episodes", type=int, default=3, help="Episodes per eval (reduce for speed)")
    args = parser.parse_args()

    if not args.skip_train:
        run("python run_scripts/train_paper_experiments.py --device cpu", "1. Train (PPO, RecurrentPPO, Agri-MetaRL, 5 seeds)")
    if args.train_ablation:
        run("python run_scripts/train_ablation_variants.py --device cpu", "1b. Train ablation variants (no_bn, no_clip, state_context_only, advantage_only)")

    run("python run_scripts/run_all_fixed_protocol.py", "2. Fixed protocol (all algorithms)")

    # 2b. Record 60d trajectory for long_horizon figure
    subprocess.run(
        [sys.executable, "run_scripts/record_trajectory_60d.py", "--algorithm", "agri_metarl"],
        cwd=ROOT
    )

    # 2c. Export learning curves from wandb
    subprocess.run(
        [sys.executable, "run_scripts/export_learning_curves.py", "--project", "AgriControl"],
        cwd=ROOT
    )

    # 3. Train-test generalization: run per algo and merge
    import pandas as pd
    gen_dir = os.path.join(args.out_dir, "train_test_generalization")
    os.makedirs(gen_dir, exist_ok=True)
    all_rows = []
    for algo in ["ppo", "recurrentppo", "agri_metarl"]:
        model_root = os.path.join(args.model_root, algo, "deterministic")
        if not os.path.isdir(os.path.join(model_root, "models")):
            continue
        r = subprocess.run(
            [sys.executable, "run_scripts/evaluate_train_test_generalization.py",
             "--algorithm", algo, "--model_root", model_root, "--out_dir", gen_dir,
             "--n_episodes", str(args.n_episodes)],
            cwd=ROOT, capture_output=True, text=True, timeout=900
        )
        if r.returncode == 0 and os.path.isfile(os.path.join(gen_dir, "train_test_returns.csv")):
            df = pd.read_csv(os.path.join(gen_dir, "train_test_returns.csv"))
            all_rows.append(df)
    if all_rows:
        pd.concat(all_rows, ignore_index=True).to_csv(os.path.join(gen_dir, "train_test_returns.csv"), index=False)
        print("Merged train_test_returns.csv for all algorithms.")

    subprocess.run(
        [sys.executable, "run_scripts/evaluate_few_update.py", "--n_episodes", str(args.n_episodes)],
        cwd=ROOT
    )

    # 5. Ablation: full (from main training) + 4 variants (from train_ablation_variants)
    full_base = os.path.join(args.model_root, "agri_metarl", "deterministic")
    ablation_base = os.path.join(ROOT, "train_data", "AgriControl_ablation", "agri_metarl", "deterministic")
    r = subprocess.run(
        [sys.executable, "run_scripts/run_ablation.py",
         "--ablation_base_dir", ablation_base,
         "--full_base_dir", full_base,
         "--n_episodes", str(args.n_episodes)],
        cwd=ROOT
    )
    if r.returncode == 0:
        print("5. Ablation done.")
    else:
        # Fallback: only full variant if ablation models not trained yet
        first_run_dir = None
        ablate_models = os.path.join(full_base, "models")
        if os.path.isdir(ablate_models):
            for run_name in os.listdir(ablate_models):
                d = os.path.join(ablate_models, run_name)
                if os.path.isfile(os.path.join(d, "best_model.zip")):
                    first_run_dir = d
                    break
        if first_run_dir:
            subprocess.run(
                [sys.executable, "run_scripts/run_ablation.py", "--variant_dirs", f"full:{first_run_dir}",
                 "--n_episodes", str(args.n_episodes)],
                cwd=ROOT
            )
            print("5. Ablation (full only, run train_ablation_variants.py for full ablation).")

    run("python visualisations/plot_paper_figures.py", "6. Plot paper figures")
    print("\nPipeline done.")


if __name__ == "__main__":
    main()
