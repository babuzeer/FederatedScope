@echo off
setlocal

set REPO_DIR=D:\Projects\FederatedScope
set PYTHON_BIN=D:\ProgramData\anaconda3\envs\pi_fmd_gpu_py39\python.exe
set CONFIG_PATH=scripts/headonly_cache_generation/officehome_vit_cachegen_400c_random25_gen20_8g.yaml
set RUN_ROOT=D:\Projects\FederatedScope\exp\headonly_cache_generation\direct_400c_random25_20r_gen20_8g
set LOG_DIR=%RUN_ROOT%\launcher_logs
set STDOUT_LOG=%LOG_DIR%\stdout.log
set STDERR_LOG=%LOG_DIR%\stderr.log
set EXIT_CODE_FILE=%LOG_DIR%\exit_code.txt

if not exist "%LOG_DIR%" mkdir "%LOG_DIR%"
if exist "%EXIT_CODE_FILE%" del "%EXIT_CODE_FILE%"
type nul > "%STDERR_LOG%"

cd /d "%REPO_DIR%"
set PYTHONPATH=%REPO_DIR%
set HF_ENDPOINT=https://hf-mirror.com
set WANDB_MODE=disabled
set WANDB_DISABLED=true

echo started_at=%date% %time% > "%STDOUT_LOG%"
echo repo_dir=%REPO_DIR% >> "%STDOUT_LOG%"
echo config=%CONFIG_PATH% >> "%STDOUT_LOG%"
echo cache_dir=D:\Projects\FederatedScope\exp\ggeur_headonly_real_cache\officehome_vitb16_400c_random25_gen20_fcache_v1 >> "%STDOUT_LOG%"

"%PYTHON_BIN%" -m federatedscope.main --cfg "%CONFIG_PATH%" >> "%STDOUT_LOG%" 2>> "%STDERR_LOG%"
echo exit_code=%ERRORLEVEL% > "%EXIT_CODE_FILE%"
echo finished_at=%date% %time% >> "%STDOUT_LOG%"

endlocal
