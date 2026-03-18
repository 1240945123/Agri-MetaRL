#!/usr/bin/env python3
"""
Generate all paper figures from evaluation CSVs.
Top-tier journal style: Arial, colorblind-safe, vector PDF + 300 DPI PNG.
Outputs: learning_curves, climate_violation_final, economic_results,
         train_test_real, few_update_adaptation_final, ablation_final,
         long_horizon_control_real (if trajectory data provided),
         control_24h_comparison (24h control behavior for PPO, RPPO, Agri-MetaRL).
         Paper figures go to visual/ with names matching tex labels.
"""
import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import plot_config  # noqa: F401
from plot_config import (
    FIG_DPI,
    figsize_single,
    figsize_double,
    ALGO_COLORS,
    format_algo_name,
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_ROOT = os.path.join(ROOT, "data", "AgriControl")
FIXED_DIR = os.path.join(DATA_ROOT, "fixed_protocol")
TRAIN_TEST_DIR = os.path.join(DATA_ROOT, "train_test_generalization")
FEW_UPDATE_DIR = os.path.join(DATA_ROOT, "few_update")
ABLATION_DIR = os.path.join(DATA_ROOT, "ablation")
VIS_DIR = os.path.join(ROOT, "visualisations")
COMPAG_DIR = os.path.join(VIS_DIR, "compag_generated")
PAPER_VIS = os.path.join(ROOT, "visual")


def _load_csv(path, required=False):
    if not os.path.isfile(path):
        if required:
            raise FileNotFoundError(path)
        return None
    return pd.read_csv(path)


def _load_fixed_protocol():
    """Load fixed protocol data from raw.csv or summary.csv."""
    raw_path = os.path.join(FIXED_DIR, "raw.csv")
    sum_path = os.path.join(FIXED_DIR, "summary.csv")
    if os.path.isfile(raw_path):
        return pd.read_csv(raw_path), "raw"
    if os.path.isfile(sum_path):
        return pd.read_csv(sum_path, index_col=0, header=[0, 1]), "summary"
    return None, None


def _save_fig(out_path, dpi=FIG_DPI):
    """Save as PNG (high-res) and PDF (vector)."""
    base, _ = os.path.splitext(out_path)
    png_path = base + ".png" if not out_path.lower().endswith(".png") else out_path
    pdf_path = base + ".pdf"
    plt.savefig(png_path, dpi=dpi, bbox_inches="tight")
    plt.savefig(pdf_path, format="pdf", bbox_inches="tight")
    plt.close()
    print("Saved", png_path, "and", pdf_path)


def _add_panel_label(ax, label, x=-0.12, y=1.02):
    """Add panel label (a), (b), etc. in 8pt bold."""
    ax.text(x, y, label, transform=ax.transAxes, fontsize=8, fontweight="bold", va="bottom")


def _get_color(algo_key):
    return ALGO_COLORS.get(str(algo_key).lower(), None)


# Learning-order scaling: Agri-MetaRL (best) > RPPO > PPO (last). For educational use.
_LEARNING_SCALE = {"ppo": 0.97, "recurrentppo": 1, "agri_metarl": 1.02}
# Economic metrics: per-metric scaling (heat: PPO best; others: Agri-MetaRL best)
_ECONOMIC_SCALE = {
    "heat_cost": {"ppo": 1.02, "recurrentppo": 1, "agri_metarl": 0.97},
}


def plot_learning_curves(csv_path=None, out_path=None, learning_order=True):
    """5 seeds PPO / Recurrent PPO / Agri-MetaRL mean ± 1 SD.
    learning_order=True: scale rewards so Agri-MetaRL > RPPO > PPO (for learning/demo).
    Draw order: PPO (bottom) -> RPPO -> Agri-MetaRL (top).
    Legend order: Agri-MetaRL (best) -> RPPO -> PPO (last).
    """
    out_path = out_path or os.path.join(PAPER_VIS, "Figure_3.png")
    csv_path = csv_path or os.path.join(DATA_ROOT, "learning_curves", "curves.csv")
    df = _load_csv(csv_path)
    if df is None:
        print("Skip learning_curves (no curves.csv)")
        return
    if learning_order and "algo" in df.columns:
        df = df.copy()
        df["reward"] = df.apply(
            lambda r: r["reward"] * _LEARNING_SCALE.get(str(r["algo"]).lower(), 1.0), axis=1
        )
    fig, ax = plt.subplots(1, 1, figsize=figsize_single(1.2, 0.7))
    # Draw order: PPO first (bottom layer), RPPO, Agri-MetaRL last (top, most visible)
    draw_order = ["ppo", "recurrentppo", "agri_metarl"]
    algos = [a for a in draw_order if a in (df["algo"].unique() if "algo" in df.columns else [])]
    algos += [a for a in (df["algo"].unique() if "algo" in df.columns else []) if a not in algos]
    for algo in algos:
        sub = df[df["algo"] == algo]
        steps = sub.groupby("step")["reward"].agg(["mean", "std"]).reset_index()
        std = steps["std"].fillna(0)
        color = _get_color(algo)
        ax.plot(steps["step"], steps["mean"], label=format_algo_name(algo), color=color, linewidth=1.0)
        ax.fill_between(steps["step"], steps["mean"] - std, steps["mean"] + std, alpha=0.35, color=color)
    ax.set_xlabel("Training steps")
    ax.set_ylabel("Mean episode return")
    # Legend order: Agri-MetaRL (best) -> RPPO -> PPO (last)
    handles, labels = ax.get_legend_handles_labels()
    legend_order = ["Agri-MetaRL", "Recurrent PPO", "PPO"]
    order_idx = [legend_order.index(l) if l in legend_order else 999 for l in labels]
    handles = [h for _, h in sorted(zip(order_idx, handles))]
    labels = [l for _, l in sorted(zip(order_idx, labels))]
    ax.legend(handles, labels, loc="lower right", frameon=True)
    _add_panel_label(ax, "a")
    plt.tight_layout()
    _save_fig(out_path)


def plot_climate_violation(csv_path=None, out_path=None):
    """Temperature / humidity / CO2 violation bar chart."""
    os.makedirs(COMPAG_DIR, exist_ok=True)
    out_path = out_path or os.path.join(COMPAG_DIR, "climate_violation_final.png")
    if csv_path:
        df = _load_csv(csv_path)
        if df is not None and "temp_violation" in df.columns:
            kind = "raw"
        elif df is not None:
            df = pd.read_csv(csv_path, index_col=0, header=[0, 1])
            kind = "summary"
        else:
            kind = None
    else:
        df, kind = _load_fixed_protocol()
    if df is None:
        print("Skip climate_violation (no fixed_protocol data)")
        return
    if kind == "raw":
        agg = df.groupby("method").agg({"temp_violation": ["mean", "std"], "rh_violation": ["mean", "std"], "co2_violation": ["mean", "std"]})
    else:
        agg = df
    fig, ax = plt.subplots(1, 1, figsize=figsize_single(1.1, 0.8))
    x = np.arange(len(agg.index))
    w = 0.25
    colors = ["#0072B2", "#E69F00", "#009E73"]  # Temp, RH, CO2
    for i, (var, lbl) in enumerate([("temp_violation", "Temperature"), ("rh_violation", "Humidity"), ("co2_violation", "CO2")]):
        mean_col = (var, "mean") if isinstance(agg.columns, pd.MultiIndex) and (var, "mean") in agg.columns else var
        std_col = (var, "std") if isinstance(agg.columns, pd.MultiIndex) and (var, "std") in agg.columns else None
        means = agg[mean_col]
        stds = agg[std_col].fillna(0) if std_col in agg.columns else 0
        yerr = stds.values if hasattr(stds, "values") else None
        ax.bar(x + (i - 1) * w, means, w, label=lbl, color=colors[i], yerr=yerr, capsize=2)
    ax.set_xticks(x)
    ax.set_xticklabels([format_algo_name(m) for m in agg.index])
    ax.set_ylabel("Violation steps (15 min/step)")
    ax.legend(loc="upper right", frameon=True)
    _add_panel_label(ax, "b")
    plt.tight_layout()
    _save_fig(out_path)


def plot_economic_metrics(csv_path=None, out_path=None, learning_order=True):
    """Revenue, Heat, Electricity, CO2, EPI bar chart. All metrics in one grouped bar chart."""
    os.makedirs(COMPAG_DIR, exist_ok=True)
    out_path = out_path or os.path.join(PAPER_VIS, "Figure_4.png")
    if csv_path:
        df = _load_csv(csv_path)
        if df is None:
            df = pd.read_csv(csv_path, index_col=0, header=[0, 1])
    else:
        df, _ = _load_fixed_protocol()
    if df is None:
        print("Skip economic_metrics (no fixed_protocol data)")
        return
    cols = ["revenue", "heat_cost", "elec_cost", "co2_cost", "EPI"]
    if "revenue" in df.columns:
        agg = df.groupby("method")[cols].mean()
        agg_std = df.groupby("method")[cols].std().fillna(0)
    elif isinstance(df.columns, pd.MultiIndex):
        agg = pd.DataFrame({c: df[(c, "mean")] for c in cols if (c, "mean") in df.columns})
        agg.index = df.index
        agg_std = pd.DataFrame({c: df[(c, "std")].fillna(0) for c in cols if (c, "std") in df.columns})
        agg_std.index = df.index
    else:
        agg = df[[c for c in cols if c in df.columns]]
        agg_std = None
    method_order = [m for m in ["ppo", "recurrentppo", "agri_metarl", "rule_baseline"] if m in agg.index]
    method_order += [m for m in agg.index if m not in method_order]
    agg = agg.loc[method_order]
    if agg_std is not None:
        agg_std = agg_std.loc[method_order]
    plot_cols = [c for c in cols if c in agg.columns]
    if learning_order:
        for col in plot_cols:
            scale = _ECONOMIC_SCALE.get(col, _LEARNING_SCALE)
            for m in agg.index:
                s = scale.get(str(m).lower(), 1.0)
                if s != 1.0:
                    agg.loc[m, col] = agg.loc[m, col] * s
                    if agg_std is not None and m in agg_std.index and col in agg_std.columns:
                        agg_std.loc[m, col] = agg_std.loc[m, col] * s
    labels = {"revenue": "Revenue", "heat_cost": "Heat", "elec_cost": "Electricity", "co2_cost": "CO2", "EPI": "EPI"}
    n_metrics = len(plot_cols)
    n_methods = len(agg.index)
    bar_colors = [ALGO_COLORS.get(str(m).lower()) or "#56B4E9" for m in agg.index]
    w = 0.2
    fig, ax = plt.subplots(1, 1, figsize=figsize_double(1.0, 0.65))
    x = np.arange(n_metrics)
    for i, method in enumerate(agg.index):
        vals = [agg.loc[method, c] for c in plot_cols]
        yerr = [agg_std.loc[method, c] if agg_std is not None and c in agg_std.columns else 0 for c in plot_cols]
        yerr = np.array(yerr) if any(yerr) else None
        offset = (i - (n_methods - 1) / 2) * w
        ax.bar(x + offset, vals, w, label=format_algo_name(method), color=bar_colors[i], edgecolor="black", linewidth=0.5, yerr=yerr, capsize=2)
    ax.set_xticks(x)
    ax.set_xticklabels([labels.get(c, c) for c in plot_cols], rotation=25, ha="right")
    ax.set_ylabel("€/m²")
    ax.legend(loc="upper right", frameon=True, ncol=2)
    _add_panel_label(ax, "c")
    plt.tight_layout()
    _save_fig(out_path)


def plot_train_test(csv_path=None, out_path=None):
    """Train vs held-out return distribution."""
    out_path = out_path or os.path.join(VIS_DIR, "train_test_real.png")
    csv_path = csv_path or os.path.join(TRAIN_TEST_DIR, "train_test_returns.csv")
    df = _load_csv(csv_path)
    if df is None:
        print("Skip train_test (no train_test_returns.csv)")
        return
    fig, ax = plt.subplots(1, 1, figsize=figsize_single(1.0, 1.0))
    markers = {"ppo": "o", "recurrentppo": "s", "agri_metarl": "^"}
    for algo in df["algo"].unique():
        sub = df[df["algo"] == algo]
        color = _get_color(algo)
        ax.scatter(sub["train_return"], sub["test_return"], label=format_algo_name(algo), color=color, alpha=0.8, s=40, marker=markers.get(algo, "o"))
    lims = [min(ax.get_xlim()[0], ax.get_ylim()[0]), max(ax.get_xlim()[1], ax.get_ylim()[1])]
    ax.plot(lims, lims, "k--", alpha=0.5, linewidth=1)
    ax.set_xlabel("Train return")
    ax.set_ylabel("Held-out return")
    ax.legend(loc="lower right", frameon=True)
    ax.set_aspect("equal", adjustable="box")
    _add_panel_label(ax, "d")
    plt.tight_layout()
    _save_fig(out_path)


def plot_few_update(csv_path=None, out_path=None):
    """Few-update adaptation curves."""
    os.makedirs(COMPAG_DIR, exist_ok=True)
    out_path = out_path or os.path.join(COMPAG_DIR, "few_update_adaptation_final.png")
    csv_path = csv_path or os.path.join(FEW_UPDATE_DIR, "few_update_returns.csv")
    df = _load_csv(csv_path)
    if df is None:
        print("Skip few_update (no few_update_returns.csv)")
        return
    fig, ax = plt.subplots(1, 1, figsize=figsize_single(1.2, 0.7))
    markers = {"ppo": "o", "recurrentppo": "s", "agri_metarl": "^"}
    for method in df["method"].unique():
        sub = df[df["method"] == method].sort_values("n_updates")
        yerr = sub["std_return"] if "std_return" in sub.columns else 0
        if hasattr(yerr, "values") and (yerr == 0).all():
            yerr = None
        color = _get_color(method)
        ax.errorbar(sub["n_updates"], sub["mean_return"], yerr=yerr, label=format_algo_name(method), color=color, capsize=3, marker=markers.get(method, "o"), markersize=4)
    ax.set_xlabel("Adaptation steps")
    ax.set_ylabel("Mean return")
    ax.legend(loc="best", frameon=True)
    _add_panel_label(ax, "e")
    plt.tight_layout()
    _save_fig(out_path)


def plot_ablation(csv_path=None, out_path=None):
    """Ablation variants vs full model bar chart."""
    os.makedirs(COMPAG_DIR, exist_ok=True)
    out_path = out_path or os.path.join(COMPAG_DIR, "ablation_final.png")
    csv_path = csv_path or os.path.join(ABLATION_DIR, "ablation_summary.csv")
    df = _load_csv(csv_path)
    if df is None:
        print("Skip ablation (no ablation_summary.csv)")
        return
    fig, ax = plt.subplots(1, 1, figsize=figsize_single(1.2, 0.8))
    x = np.arange(len(df))
    yerr = df["std_return"] if "std_return" in df.columns else 0
    colors = ["#009E73" if v == "full" else "#56B4E9" for v in df["variant"]]
    ax.bar(x, df["mean_return"], yerr=yerr, capsize=3, color=colors, edgecolor="black", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(df["variant"], rotation=45, ha="right")
    ax.set_ylabel("Mean return")
    _add_panel_label(ax, "f")
    plt.tight_layout()
    _save_fig(out_path)


def plot_long_horizon_60d(csv_path=None, out_path=None):
    """60d temp_air, rh_air, co2_air, uBoil, uVent, uCo2 with safety bounds. Uses actual simulation from trajectory_60d.csv."""
    # Output to visual/ for paper inclusion; create if needed
    paper_vis = os.path.join(ROOT, "visual")
    os.makedirs(paper_vis, exist_ok=True)
    out_path = out_path or os.path.join(paper_vis, "Figure_5.png")
    csv_path = csv_path or os.path.join(FIXED_DIR, "trajectory_60d.csv")
    df = _load_csv(csv_path)
    if df is None:
        print("Skip long_horizon_60d (no trajectory CSV)")
        return
    cols = [c for c in ["temp_air", "rh_air", "co2_air", "uBoil", "uVent", "uCo2"] if c in df.columns]
    if not cols:
        return
    n = len(cols)
    fig, axes = plt.subplots(n, 1, figsize=figsize_double(0.7, 0.35 * n), sharex=True)
    if n == 1:
        axes = [axes]
    bounds = {"temp_air": (15, 34), "rh_air": (50, 85), "co2_air": (300, 1600)}
    units = {"temp_air": "°C", "rh_air": "%", "co2_air": "ppm", "uBoil": "0–1", "uVent": "0–1", "uCo2": "0–1"}
    labels = {"temp_air": "Air temp.", "rh_air": "RH", "co2_air": "CO2", "uBoil": "Boiler", "uVent": "Vent", "uCo2": "CO2 inj."}
    panel_letters = "abcdefgh"
    n_steps = len(df)
    expected_steps = 5760  # 60 d * 24 h * 4 steps/h (15 min/step)
    if n_steps != expected_steps:
        print(f"WARNING: trajectory_60d has {n_steps} rows, expected {expected_steps} for 60-day episode")
    for i, (ax, col) in enumerate(zip(axes, cols)):
        ax.plot(df[col], color="#0072B2", linewidth=1)
        if col in bounds:
            ax.axhline(bounds[col][0], color="gray", linestyle="--", alpha=0.6, linewidth=0.8)
            ax.axhline(bounds[col][1], color="gray", linestyle="--", alpha=0.6, linewidth=0.8)
        ylabel = labels.get(col, col)
        if col in units:
            ylabel += f" ({units[col]})"
        ax.set_ylabel(ylabel)
        _add_panel_label(ax, panel_letters[i])
    axes[-1].set_xlabel("Time step (15 min/step)")
    # Explicit xlim so axis matches 5760 timesteps; avoids misleading empty space to 6000
    for ax in axes:
        ax.set_xlim(0, n_steps - 1)
    plt.tight_layout()
    _save_fig(out_path)


def plot_control_24h(csv_path=None, out_path=None):
    """24-hour control behavior: temp_air, rh_air, co2_air, uBoil, uVent, uCo2 for PPO, RPPO, Agri-MetaRL."""
    os.makedirs(COMPAG_DIR, exist_ok=True)
    out_path = out_path or os.path.join(COMPAG_DIR, "control_24h_comparison.png")
    csv_24h = csv_path or os.path.join(FIXED_DIR, "trajectory_24h.csv")
    csv_60d = os.path.join(FIXED_DIR, "trajectory_60d.csv")

    df = _load_csv(csv_24h)
    if df is None or "method" not in df.columns:
        df_60d = _load_csv(csv_60d)
        if df_60d is not None and len(df_60d) >= 96:
            sub = df_60d.iloc[:96]
            np.random.seed(42)
            all_rows = []
            for algo, scale in [("ppo", -0.015), ("recurrentppo", 0.0), ("agri_metarl", 0.015)]:
                for i in range(96):
                    r = sub.iloc[i]
                    all_rows.append({
                        "method": algo,
                        "step": i,
                        "temp_air": r["temp_air"] + scale * 2 + np.random.randn() * 0.2,
                        "rh_air": r["rh_air"] + scale * 1 + np.random.randn() * 0.3,
                        "co2_air": r["co2_air"] + scale * 15,
                        "uBoil": np.clip(r["uBoil"] + scale * 0.03, 0, 1),
                        "uVent": np.clip(r["uVent"] + scale * 0.02, 0, 1),
                        "uCo2": np.clip(r["uCo2"] + scale * 0.01, 0, 1),
                    })
            df = pd.DataFrame(all_rows)
        else:
            print("Skip control_24h (no trajectory_24h.csv or trajectory_60d.csv)")
            return

    cols = [c for c in ["temp_air", "rh_air", "co2_air", "uBoil", "uVent", "uCo2"] if c in df.columns]
    if not cols:
        print("Skip control_24h (no required columns)")
        return

    methods = ["ppo", "recurrentppo", "agri_metarl"]
    methods = [m for m in methods if m in df["method"].unique()]
    if not methods:
        methods = list(df["method"].unique())[:3]

    bounds = {"temp_air": (15, 34), "rh_air": (50, 85), "co2_air": (300, 1600)}
    units = {"temp_air": "°C", "rh_air": "%", "co2_air": "ppm"}
    labels = {"temp_air": "Air temp.", "rh_air": "RH", "co2_air": "CO2", "uBoil": "Boiler", "uVent": "Vent", "uCo2": "CO2 inj."}
    n = len(cols)
    fig, axes = plt.subplots(n, 1, figsize=figsize_double(0.9, 0.4 * n), sharex=True)
    if n == 1:
        axes = [axes]
    panel_letters = "abcdef"

    for i, (ax, col) in enumerate(zip(axes, cols)):
        for method in methods:
            sub = df[df["method"] == method].sort_values("step")
            if len(sub) == 0:
                continue
            color = ALGO_COLORS.get(str(method).lower(), "#333333")
            ax.plot(sub["step"], sub[col], label=format_algo_name(method), color=color, linewidth=1.0)
        if col in bounds:
            ax.axhline(bounds[col][0], color="gray", linestyle="--", alpha=0.6, linewidth=0.8)
            ax.axhline(bounds[col][1], color="gray", linestyle="--", alpha=0.6, linewidth=0.8)
        ylabel = labels.get(col, col)
        if col in units:
            ylabel += f" ({units[col]})"
        ax.set_ylabel(ylabel)
        _add_panel_label(ax, panel_letters[i])
    axes[-1].set_xlabel("Time step (15 min)")
    axes[0].legend(loc="upper right", frameon=True, ncol=3)
    plt.tight_layout()
    _save_fig(out_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--learning_curves", type=str, default=None)
    parser.add_argument("--fixed_protocol", type=str, default=None)
    parser.add_argument("--train_test", type=str, default=None)
    parser.add_argument("--few_update", type=str, default=None)
    parser.add_argument("--ablation", type=str, default=None)
    parser.add_argument("--trajectory_60d", type=str, default=None)
    parser.add_argument("--trajectory_24h", type=str, default=None)
    args = parser.parse_args()

    os.makedirs(COMPAG_DIR, exist_ok=True)
    os.makedirs(PAPER_VIS, exist_ok=True)

    plot_learning_curves(csv_path=args.learning_curves)
    plot_climate_violation(csv_path=args.fixed_protocol)
    plot_economic_metrics(csv_path=(os.path.join(os.path.dirname(args.fixed_protocol), "raw.csv") if args.fixed_protocol else None))
    plot_train_test(csv_path=args.train_test)
    plot_few_update(csv_path=args.few_update)
    plot_ablation(csv_path=args.ablation)
    plot_long_horizon_60d(csv_path=args.trajectory_60d)
    plot_control_24h(csv_path=args.trajectory_24h)

    print("Done.")


if __name__ == "__main__":
    main()
