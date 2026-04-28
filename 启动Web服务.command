#!/usr/bin/env bash
# 追色 Web 服务启动脚本
cd "$(dirname "$0")/web_app"

PYTHON="/opt/homebrew/bin/python3.14"

echo "📦 检查依赖…"
"$PYTHON" -m pip install -q -r requirements.txt

echo "🚀 启动追色 Web 服务 → http://localhost:8000"
"$PYTHON" -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload
