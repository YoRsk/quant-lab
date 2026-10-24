#!/usr/bin/env bash
# WSL / Git Bash 一键运行脚本
# 用法: ./run.sh strategies/multifactor.py
#       ./run.sh scripts/run_factor_test.py
#       ./run.sh scripts/run_sensitivity.py
cd "$(dirname "$0")"

if command -v python3 >/dev/null 2>&1 && python3 -c "import pandas" 2>/dev/null; then
  PY=python3                      # 本机已有装好依赖的 Linux Python，优先用
else
  # 回退到 Windows venv：WSL 挂载点为 /mnt/c，Git Bash 为 /c
  WIN_PY="Users/liuyi/.workbuddy/binaries/python/envs/default/Scripts/python.exe"
  if [ -f "/mnt/c/$WIN_PY" ]; then
    PY="/mnt/c/$WIN_PY"
  elif [ -f "/c/$WIN_PY" ]; then
    PY="/c/$WIN_PY"
  else
    echo "找不到可用的 Python：请先 pip install -r requirements.txt" >&2
    exit 1
  fi
fi

PYTHONPATH=src "$PY" "$@"
