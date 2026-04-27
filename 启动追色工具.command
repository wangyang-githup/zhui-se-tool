#!/usr/bin/env bash
# 追色工具启动脚本
cd "$(dirname "$0")"

PYTHON="/opt/homebrew/bin/python3.14"

# 检查依赖
check_pkg() {
    "$PYTHON" -c "import $1" 2>/dev/null
}

MISSING=""
for pkg in cv2 numpy PIL scipy; do
    if ! check_pkg "$pkg"; then
        MISSING="$MISSING $pkg"
    fi
done

if [ -n "$MISSING" ]; then
    echo "正在安装缺失依赖：$MISSING"
    "$PYTHON" -m pip install opencv-python numpy Pillow scipy
fi

# 确保 Tkinter 可用
"$PYTHON" -c "import tkinter" 2>/dev/null || arch -arm64 brew install python-tk@3.14

echo "启动追色工具..."
"$PYTHON" app.py
