@echo off
REM Complete evaluation script for paper reproduction
REM Runs all necessary evaluations automatically

echo ==========================================
echo Starting complete evaluation for paper reproduction
echo ==========================================

REM Project configuration
set project_name=AgriControl
set env_id=TomatoEnv

echo.
echo Step 1: Running deterministic evaluation...
python gl_gym/experiments/evaluate_rl.py --project %project_name% --env_id %env_id% --model_name dummy-cg4axdls --algorithm ppo --mode deterministic --uncertainty_scale 0.0

echo.
echo Step 2: Running stochastic evaluations for all uncertainty scales...
for %%s in (0.0 0.05 0.1 0.15 0.2 0.25 0.3) do (
    echo Evaluating uncertainty scale: %%s
    python gl_gym/experiments/evaluate_rl.py --project %project_name% --env_id %env_id% --model_name dummy-nbw883du --algorithm ppo --mode stochastic --uncertainty_scale %%s
)

echo.
echo Step 3: Running rule-based controller evaluations...
echo Deterministic rule-based evaluation...
python gl_gym/experiments/evaluate_baseline.py --project %project_name% --env_id %env_id% --mode deterministic --uncertainty_scale 0.0

echo Stochastic rule-based evaluations...
for %%s in (0.0 0.05 0.1 0.15 0.2 0.25 0.3) do (
    echo Evaluating uncertainty scale: %%s
    python gl_gym/experiments/evaluate_baseline.py --project %project_name% --env_id %env_id% --mode stochastic --uncertainty_scale %%s
)

echo.
echo ==========================================
echo All evaluations completed!
echo ==========================================
echo.
echo Results saved in:
echo - Deterministic PPO: data/AgriControl/deterministic/ppo/
echo - Stochastic PPO: data/AgriControl/stochastic/ppo/
echo - Rule-based baseline: data/AgriControl/deterministic/rb_baseline/ and data/AgriControl/stochastic/rb_baseline/
echo.
echo Next step: Generate visualizations
echo.
pause
