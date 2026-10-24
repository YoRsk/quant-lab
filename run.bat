@echo off
rem ============================================
rem quant-lab 一键运行（Windows CMD / 双击运行）
rem   双击本文件        = 跑主策略回测
rem   cmd 中带参数      = run.bat scripts\run_factor_test.py
rem ============================================
cd /d "%~dp0"
set PYTHONPATH=src

if "%~1"=="" (
  set "TARGET=strategies\multifactor.py"
  set "PAUSE_AFTER=1"
) else (
  set "TARGET=%*"
)

echo [quant-lab] 运行 %TARGET% ...
"C:\Users\liuyi\.workbuddy\binaries\python\envs\default\Scripts\python.exe" %TARGET%
echo.
echo 完成。报告与图表在 outputs\ 文件夹。
if defined PAUSE_AFTER pause
