#!/usr/bin/env python3
"""
Agri-MetaRL 架构图：支持集-查询集式优势修正。
输出: architecture_agri_metarl.png, architecture_agri_metarl.pdf
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

import plot_config  # noqa: F401
from plot_config import FIG_DPI

# 中文字体（在 plot_config 之后覆盖）：Windows 常用 Microsoft YaHei / SimHei
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "SimSun", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VIS_DIR = os.path.join(ROOT, "visualisations")
COMPAG_DIR = os.path.join(VIS_DIR, "compag_generated")


def _save_fig(out_path, dpi=FIG_DPI):
    """Save as PNG and PDF."""
    base, _ = os.path.splitext(out_path)
    png_path = base + ".png" if not out_path.lower().endswith(".png") else out_path
    pdf_path = base + ".pdf"
    plt.savefig(png_path, dpi=dpi, bbox_inches="tight")
    plt.savefig(pdf_path, format="pdf", bbox_inches="tight")
    plt.close()
    print("Saved", png_path, "and", pdf_path)


def draw_architecture():
    """Draw Agri-MetaRL architecture diagram."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.set_aspect("equal")
    ax.axis("off")

    # Colors (Wong palette, consistent with paper)
    col_rollout = "#56B4E9"  # sky blue
    col_support = "#E69F00"  # orange
    col_query = "#009E73"  # green
    col_meta = "#0072B2"  # blue
    col_ppo = "#CC79A7"  # purple
    col_arrow = "#333333"

    def box(x, y, w, h, label, color, fontsize=7):
        """Draw a rounded box with label."""
        p = FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.02,rounding_size=0.15",
            facecolor=color, edgecolor="black", linewidth=1.0, alpha=0.85
        )
        ax.add_patch(p)
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", fontsize=fontsize)

    def arrow(x1, y1, x2, y2, label=""):
        """Draw arrow from (x1,y1) to (x2,y2)."""
        ax.annotate(
            "", xy=(x2, y2), xytext=(x1, y1),
            arrowprops=dict(arrowstyle="->", color=col_arrow, lw=1.2),
            annotation_clip=False,
        )
        if label:
            mx, my = (x1 + x2) / 2, (y1 + y2) / 2
            ax.text(mx, my, label, fontsize=5, ha="center", va="bottom", color=col_arrow)

    # === Layout ===
    # Row 0 (top): Task fragment -> Support/Query split
    box(0.3, 4.2, 1.4, 0.9, "任务片段 τ\n(episode)", col_rollout)
    box(2.2, 4.2, 1.2, 0.9, "支持集 D_sup\n(前 50%)", col_support)
    box(3.9, 4.2, 1.2, 0.9, "查询集 D_qry\n(后 50%)", col_query)

    arrow(1.7, 4.65, 2.2, 4.65)
    arrow(3.4, 4.65, 3.9, 4.65)

    # Row 1: Support -> statistics -> context encoder
    box(2.0, 2.8, 1.6, 0.8, "统计量提取\nμ_s, σ_s, adv, r, G", col_support, fontsize=6)
    box(4.0, 2.8, 1.4, 0.8, "上下文编码器\nMLP → c_τ", col_meta)

    arrow(2.8, 4.2, 2.8, 3.6)
    arrow(3.6, 3.2, 4.0, 3.2)

    # Row 2: Query -> advantage correction
    box(3.6, 1.4, 2.2, 0.9, "优势修正头 δ_φ(o_t, A_t, c_τ)\nMLP(128→1)", col_meta)
    box(6.2, 1.4, 1.2, 0.9, "Ã_t = A_t + δ\n标准化 & 裁剪", col_query)

    arrow(5.1, 4.2, 4.7, 3.2)  # Query -> correction (obs, A_t)
    arrow(4.7, 2.8, 4.7, 2.3)  # c_τ -> correction
    arrow(4.7, 1.85, 6.2, 1.85)
    ax.text(4.0, 3.0, "c_τ", fontsize=6, ha="center", color=col_meta)
    ax.text(5.0, 3.8, "o_t, A_t", fontsize=5, ha="center", color=col_query)

    # Row 3: PPO update
    box(6.0, 0.2, 1.6, 0.7, "PPO 策略更新\n(Recurrent PPO)", col_ppo)
    arrow(6.8, 1.4, 6.8, 0.9)

    # Recurrent PPO base (left side)
    box(0.2, 0.2, 1.8, 1.2, "Recurrent PPO 主体\nActor(LSTM)+Critic(LSTM)\nGAE → A_t", col_rollout, fontsize=6)
    arrow(2.0, 0.8, 3.6, 1.4)
    ax.text(2.6, 1.0, "A_t", fontsize=5, ha="center")

    # Title
    ax.text(5, 5.6, "Agri-MetaRL: 支持集-查询集式优势修正", fontsize=10, fontweight="bold", ha="center")
    ax.text(5, 5.25, "MetaAdvantageHead = 上下文编码器 + 优势修正头", fontsize=7, ha="center", style="italic")

    plt.tight_layout()
    return fig


def main():
    os.makedirs(COMPAG_DIR, exist_ok=True)
    out_path = os.path.join(COMPAG_DIR, "architecture_agri_metarl.png")
    fig = draw_architecture()
    _save_fig(out_path)


if __name__ == "__main__":
    main()
