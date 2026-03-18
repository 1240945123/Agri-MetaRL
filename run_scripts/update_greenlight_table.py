#!/usr/bin/env python3
"""
Update GreenLight-Gym table in argi-meta-rl from evaluation data.
Reads: fixed_protocol/summary.csv (Train), heldout_summary.csv (Test), train_test_returns.csv (Test Return).
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import pandas as pd

N_STEPS = 5760
DATA_ROOT = os.path.join(ROOT, "data", "AgriControl")
FIXED = os.path.join(DATA_ROOT, "fixed_protocol", "summary.csv")
HELDOUT = os.path.join(DATA_ROOT, "train_test_generalization", "heldout_summary.csv")
TRAIN_TEST = os.path.join(DATA_ROOT, "train_test_generalization", "train_test_returns.csv")
PAPER = os.path.join(ROOT, "argi-meta-rl")


def twb(violation):
    return round((N_STEPS - violation) / N_STEPS * 100, 1)


def load_fixed():
    """Train data from fixed protocol."""
    if not os.path.isfile(FIXED):
        return None
    df = pd.read_csv(FIXED, index_col=0, header=[0, 1])
    return df


def load_heldout():
    """Test data from held-out evaluation."""
    if not os.path.isfile(HELDOUT):
        return None
    df = pd.read_csv(HELDOUT, index_col=0, header=[0, 1])
    return df


def load_train_test_returns():
    """Test Return per seed from train_test_generalization."""
    if not os.path.isfile(TRAIN_TEST):
        return None
    df = pd.read_csv(TRAIN_TEST)
    return df


def main():
    fixed = load_fixed()
    heldout = load_heldout()
    tt = load_train_test_returns()

    # Build table rows
    rows = []
    for method in ["agri_metarl", "recurrentppo", "ppo", "rule_baseline"]:
        if method not in (fixed.index if fixed is not None else []):
            continue
        name = "Agri-MetaRL" if method == "agri_metarl" else "Recurrent PPO" if method == "recurrentppo" else "PPO" if method == "ppo" else "Rule-based"

        # Train
        ret_mean = fixed.loc[method, ("episode_return", "mean")]
        ret_std = fixed.loc[method, ("episode_return", "std")]
        epi = fixed.loc[method, ("EPI", "mean")]
        tv = fixed.loc[method, ("temp_violation", "mean")]
        rv = fixed.loc[method, ("rh_violation", "mean")]
        cv = fixed.loc[method, ("co2_violation", "mean")]
        epi_str = f"${epi:.2f}$" if epi < 0 else str(round(epi, 2))
        train_row = {
            "name": name,
            "train_return": f"${int(ret_mean)} \\pm {int(ret_std)}$" if ret_std > 0 else f"${int(ret_mean)}$",
            "train_epi": epi_str,
            "train_twb_co2": twb(cv),
            "train_twb_t": twb(tv),
            "train_twb_rh": twb(rv),
        }

        # Test
        if method == "rule_baseline":
            test_return = "---"
            test_epi = "---"
            test_twb_co2 = test_twb_t = test_twb_rh = "---"
        elif heldout is not None and method in heldout.index:
            ret_mean_t = heldout.loc[method, ("episode_return", "mean")]
            ret_std_t = heldout.loc[method, ("episode_return", "std")]
            epi_t = heldout.loc[method, ("EPI", "mean")]
            tv_t = heldout.loc[method, ("temp_violation", "mean")]
            rv_t = heldout.loc[method, ("rh_violation", "mean")]
            cv_t = heldout.loc[method, ("co2_violation", "mean")]
            test_return = f"${int(ret_mean_t)} \\pm {int(ret_std_t)}$" if ret_std_t > 0 else f"${int(ret_mean_t)}$"
            test_epi = f"${epi_t:.2f}$" if epi_t < 0 else round(epi_t, 2)
            test_twb_co2 = twb(cv_t)
            test_twb_t = twb(tv_t)
            test_twb_rh = twb(rv_t)
        elif tt is not None and method in tt["algo"].values:
            sub = tt[tt["algo"] == method]
            ret_mean_t = sub["test_return"].mean()
            ret_std_t = sub["test_return"].std() if len(sub) > 1 else 0
            test_return = f"${int(ret_mean_t)} \\pm {int(ret_std_t)}$" if ret_std_t > 0 else f"${int(ret_mean_t)}$"
            test_epi = test_twb_co2 = test_twb_t = test_twb_rh = "---"
        else:
            test_return = test_epi = test_twb_co2 = test_twb_t = test_twb_rh = "---"

        train_row.update({
            "test_return": test_return,
            "test_epi": test_epi,
            "test_twb_co2": test_twb_co2,
            "test_twb_t": test_twb_t,
            "test_twb_rh": test_twb_rh,
        })
        rows.append(train_row)

    # Print LaTeX table body
    print("% LaTeX table rows (copy to argi-meta-rl):")
    print()
    for r in rows:
        print(f"{r['name']} & {r['train_return']} & {r['train_epi']} & {r['train_twb_co2']} & {r['train_twb_t']} & {r['train_twb_rh']} & {r['test_return']} & {r['test_epi']} & {r['test_twb_co2']} & {r['test_twb_t']} & {r['test_twb_rh']} \\\\")

    # Auto-update paper if requested
    if len(sys.argv) > 1 and sys.argv[1] == "--update":
        if not rows:
            print("No data to update.")
            return
        # Read paper content
        with open(PAPER, "r", encoding="utf-8") as f:
            content = f.read()
        # Build new table rows
        new_lines = []
        for r in rows:
            line = f"{r['name']} & {r['train_return']} & {r['train_epi']} & {r['train_twb_co2']} & {r['train_twb_t']} & {r['train_twb_rh']} & {r['test_return']} & {r['test_epi']} & {r['test_twb_co2']} & {r['test_twb_t']} & {r['test_twb_rh']} \\\\\n"
            new_lines.append(line)
        # Find and replace table body
        import re
        pattern = r"(Agri-MetaRL & \$[^$]+\$ & [\d.-]+ & [\d.]+ & [\d.]+ & [\d.]+ & )[^&]+( & )[^&]+( & )[^&]+( & )[^&]+( & )[^&]+( \\\\\nRecurrent PPO.*?\\\\\nPPO.*?\\\\\nRule-based.*?\\\\\n)"
        # Simpler: replace each line
        old_lines = [
            "Agri-MetaRL & $3812 \\pm 13$ & 3.36 & 99.5 & 98.1 & 97.1 & --- & --- & --- & --- & --- \\\\\n",
            "Recurrent PPO & $3808 \\pm 9$ & 3.29 & 91.4 & 98.3 & 97.0 & $3982 \\pm 7$ & --- & --- & --- & --- \\\\\n",
            "PPO & $3819 \\pm 2$ & 3.34 & 99.8 & 98.0 & 98.3 & --- & --- & --- & --- & --- \\\\\n",
            "Rule-based & $3094$ & $-4.76$ & 100.0 & 92.6 & 36.2 & --- & --- & --- & --- & --- \\\\\n",
        ]
        for i, r in enumerate(rows):
            new_line = f"{r['name']} & {r['train_return']} & {r['train_epi']} & {r['train_twb_co2']} & {r['train_twb_t']} & {r['train_twb_rh']} & {r['test_return']} & {r['test_epi']} & {r['test_twb_co2']} & {r['test_twb_t']} & {r['test_twb_rh']} \\\\\n"
            for old in old_lines:
                if old.startswith(r["name"].split()[0]):  # Match by first word
                    content = content.replace(old, new_line)
                    break
        with open(PAPER, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"Updated {PAPER}")


if __name__ == "__main__":
    main()
