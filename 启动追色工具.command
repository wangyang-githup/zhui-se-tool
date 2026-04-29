#!/usr/bin/env bash
# 追色工具 v2.0 启动脚本
cd "$(dirname "$0")"

PYTHON="/opt/homebrew/bin/python3.14"

# 检查依赖
check_pkg() {
    "$PYTHON" -c "import $1" 2>/dev/null
}

MISSING=""
for pkg in numpy PIL scipy customtkinter; do
    if ! check_pkg "$pkg"; then
        MISSING="$MISSING $pkg"
    fi
done

# sklearn 可选但推荐
if ! check_pkg sklearn; then
    MISSING="$MISSING scikit-learn"
fi

if [ -n "$MISSING" ]; then
    echo "正在安装缺失依赖：$MISSING"
    "$PYTHON" -m pip install numpy Pillow scipy customtkinter scikit-learn
fi

echo "启动追色工具 v2.0..."
"$PYTHON" app.py
