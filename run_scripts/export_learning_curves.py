#!/usr/bin/env python3
"""
Export learning curves from wandb to data/AgriControl/learning_curves/curves.csv.
Format: step, reward, algo, seed (for plot_paper_figures plot_learning_curves).
Requires: wandb login and runs for ppo, recurrentppo, agri_metarl (groups like ppo_paper_seed42).
"""
import os
import sys
import re
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

OUT_DIR = "data/AgriControl/learning_curves"
OUT_PATH = os.path.join(OUT_DIR, "curves.csv")
ALGOS = ["ppo", "recurrentppo", "agri_metarl"]


def export_from_wandb(entity: str, project: str) -> pd.DataFrame:
    import wandb
    api = wandb.Api()
    all_rows = []
    for algo in ALGOS:
        # Match groups like ppo_paper_seed42, recurrentppo_paper_seed123, etc.
        runs = api.runs(f"{entity}/{project}", filters={"group": {"$regex": f"^{algo}_paper_seed"}})
        for run in runs:
            try:
                hist = run.history()
            except Exception:
                continue
            # Prefer eval/mean_reward (held-out eval), fallback to rollout/ep_rew_mean
            step_col = "time/total_timesteps" if "time/total_timesteps" in hist.columns else ("global_step" if "global_step" in hist.columns else None)
            rew_col = "eval/mean_reward" if "eval/mean_reward" in hist.columns else ("rollout/ep_rew_mean" if "rollout/ep_rew_mean" in hist.columns else None)
            if step_col is None or rew_col is None:
                continue
            steps = hist[step_col].dropna()
            rewards = hist[rew_col].dropna()
            # Align lengths (wandb history can have NaNs)
            valid = steps.notna() & rewards.notna()
            steps = steps[valid].astype(int)
            rewards = rewards[valid].astype(float)
            m = re.search(r"seed(\d+)", run.group or run.name or "")
            seed = int(m.group(1)) if m else 0
            for s, r in zip(steps, rewards):
                all_rows.append({"step": int(s), "reward": float(r), "algo": algo, "seed": seed})
    return pd.DataFrame(all_rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--entity", type=str, default=None, help="W&B entity (default: from wandb)")
    parser.add_argument("--project", type=str, default="AgriControl")
    parser.add_argument("--out_path", type=str, default=OUT_PATH)
    args = parser.parse_args()

    try:
        import wandb
        entity = args.entity or wandb.api.viewer()["entity"] if hasattr(wandb.api, "viewer") else "user"
    except Exception:
        entity = args.entity or "user"

    try:
        df = export_from_wandb(entity, args.project)
    except Exception as e:
        print("wandb export failed:", e)
        print("Create curves.csv manually with columns: step, reward, algo, seed")
        return

    if df.empty:
        print("No runs found. Create curves.csv manually or run train_paper_experiments first.")
        return

    os.makedirs(os.path.dirname(args.out_path) or ".", exist_ok=True)
    df.to_csv(args.out_path, index=False)
    print("Wrote", args.out_path, "with", len(df), "rows")


if __name__ == "__main__":
    main()
