#!/usr/bin/env bash
# 追色工具 v2.1 启动脚本
cd "$(dirname "$0")"

PYTHON="python3"

# 检查必要依赖
check_pkg() {
    "$PYTHON" -c "import $1" 2>/dev/null
}

MISSING=""
for pkg in numpy PIL scipy; do
    if ! check_pkg "$pkg"; then
        MISSING="$MISSING $pkg"
    fi
done

if ! check_pkg sklearn; then
    MISSING="$MISSING scikit-learn"
fi

if [ -n "$MISSING" ]; then
    echo "正在安装缺失依赖：$MISSING"
    "$PYTHON" -m pip install numpy Pillow scipy scikit-learn --break-system-packages
fi

echo "启动追色工具 v2.1..."
"$PYTHON" app.py
