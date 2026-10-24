# Windows PowerShell 一键运行脚本
# 用法: .\run.ps1 strategies\multifactor.py
#       .\run.ps1 scripts\run_factor_test.py
#       .\run.ps1 scripts\run_sensitivity.py
$env:PYTHONPATH = "src"
& "C:\Users\liuyi\.workbuddy\binaries\python\envs\default\Scripts\python.exe" @args
