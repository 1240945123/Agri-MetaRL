@echo off
REM 完整消融实验: 4 变体 x 5 种子, 2e6 步/run
REM 使用 n_envs=2 减少内存占用，--skip_existing 支持断点续跑
set CUDA_VISIBLE_DEVICES=-1
cd /d "%~dp0\.."
python experiments/scripts/train_ablation_variants.py --n_envs 2 --skip_existing --device cpu
echo.
echo 训练完成，运行消融评估...
python experiments/scripts/run_ablation.py --ablation_base_dir artifacts/models/AgriControl_ablation/agri_metarl/deterministic --full_base_dir artifacts/models/AgriControl/agri_metarl/deterministic --n_episodes 5
echo.
echo 生成消融图...
python artifacts/figures/generators/plot_paper_figures.py
echo Done.
