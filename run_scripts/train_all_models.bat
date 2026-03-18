@echo off
REM 训练所有模型的脚本 - Windows版本
REM 用于论文复现的完整模型训练

echo 开始训练所有模型...

REM 设置环境变量
set WANDB_MODE=disabled

REM 训练PPO模型（确定性）
echo 训练PPO模型（确定性模式）...
python gl_gym/RL/experiment_manager.py --project AgriControl --env_id TomatoEnv --algorithm ppo --group ppo_det --n_eval_episodes 1 --n_evals 10 --env_seed 666 --model_seed 666 --device cpu --save_model --save_env

REM 训练PPO模型（随机性）
echo 训练PPO模型（随机性模式）...
python gl_gym/RL/experiment_manager.py --project AgriControl --env_id TomatoEnv --algorithm ppo --group ppo_stoch --n_eval_episodes 1 --n_evals 10 --env_seed 666 --model_seed 666 --device cpu --save_model --save_env --stochastic

REM 训练SAC模型（确定性）
echo 训练SAC模型（确定性模式）...
python gl_gym/RL/experiment_manager.py --project AgriControl --env_id TomatoEnv --algorithm sac --group sac_det --n_eval_episodes 1 --n_evals 10 --env_seed 666 --model_seed 666 --device cpu --save_model --save_env

REM 训练SAC模型（随机性）
echo 训练SAC模型（随机性模式）...
python gl_gym/RL/experiment_manager.py --project AgriControl --env_id TomatoEnv --algorithm sac --group sac_stoch --n_eval_episodes 1 --n_evals 10 --env_seed 666 --model_seed 666 --device cpu --save_model --save_env --stochastic

REM 训练RecurrentPPO模型（确定性）
echo 训练RecurrentPPO模型（确定性模式）...
python gl_gym/RL/experiment_manager.py --project AgriControl --env_id TomatoEnv --algorithm recurrentppo --group rec_ppo_det --n_eval_episodes 1 --n_evals 10 --env_seed 666 --model_seed 666 --device cpu --save_model --save_env

echo 所有模型训练完成！
pause
