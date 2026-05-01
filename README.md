# 追色工具 v2.1

> **AI 色彩迁移工具** | Lab 空间 Reinhard 色彩迁移 + ΔE 肤色保护 + 8 组预设

基于参考图的一键追色工具。导入参考图和原图，自动匹配色彩风格，并支持精细参数微调。

---

## 快速启动

```bash
# 安装依赖
pip install numpy Pillow scipy scikit-learn

# 启动
python3 app.py
```

或双击 **「启动追色工具.command」**。

## 系统要求

- Python 3.9+
- macOS / Windows / Linux
- 8GB 内存（推荐 16GB+）
- 无 OpenCV 依赖

## 界面布局

```
┌─────────────────────────────────────────────────────┐
│  追色    预设  批量  历史                           │
├──────────────────────────────┬──────────────────────┤
│                              │  预设（横向滚动）     │
│  参考图   原图   结果图      │  ┌──┐┌──┐┌──┐┌──┐   │
│  (三联对比预览)              │  │  ││  ││  ││  │   │
│                              │  └──┘└──┘└──┘└──┘   │
│                              │                       │
│                              │  参数                 │
│                              │  曝光  ─────●───     │
│                              │  对比度 ────●────    │
│                              │  饱和度 ────●────    │
│                              │  色调   ──────●──    │
│                              │                       │
│                              │  肤色保护 [开关]      │
│  色差 肤色 速度 覆盖度       │                       │
│  ΔE   偏差  142ms  92%       │  ▶ 开始追色           │
│                              │  ⌘+Enter              │
│                              │                       │
│                              │  导出  [图片] [LUT]   │
├──────────────────────────────┴──────────────────────┤
│  就绪 — 请导入参考图和原图                           │
└─────────────────────────────────────────────────────┘
```

## 预设列表（8组）

| 预设 | 风格来源 | 适用场景 |
|------|---------|---------|
| 🌿 日系清新 | 滨田英明 | 日常写真、少女系 |
| 🎬 暗调电影 | 王家卫/Wim Wenders | 人像、城市、情绪片 |
| 📷 复古胶片 | Kodak Gold 160 | 人文、旅拍、怀旧 |
| 🎭 青橙色调 | 好莱坞 T&O | 商业广告、人像、风光 |
| 🎞 Kodak Portra | Portra 400 人像卷 | 婚礼、肖像、儿童 |
| 🖤 水墨风 | 中国传统水墨画 | 古风、人文、静物 |
| 🌃 港风复古 | 90年代香港电影 | 复古人像、城市题材 |
| 🍬 INS马卡龙 | Instagram 莫兰迪 | 少女系、美食、静物 |

## 技术架构

```
app.py (Tkinter GUI)
  ├── 快速模式 → color_engine.ColorTransfer → Reinhard Lab 迁移
  └── 专业模式 → feature_extractor.FeatureExtractor
                   → grading_generator.ColorGradingGenerator
                   → color_renderer.ColorRenderer (含肤色保护)
                 evaluator.Evaluator (评价指标)
```

### 核心模块

| 模块 | 功能 | 核心类/函数 |
|------|------|-------------|
| `color_engine.py` | Lab 色彩迁移、LUT 生成 | `ColorTransfer`, `LUTGenerator` |
| `feature_extractor.py` | 图像色彩特征提取 | `FeatureExtractor`, `ColorFeatures` |
| `grading_generator.py` | LR 风格参数生成 | `ColorGradingGenerator`, `ColorGradingParams` |
| `color_renderer.py` | 渲染 + 肤色保护 | `ColorRenderer`, `delta_e_cie76` |
| `evaluator.py` | 评价指标 | `Evaluator` (ΔE/直方图/匹配度) |
| `presets.json` | 8 组内置预设参数 | JSON 格式 |

### 肤色保护策略

| ΔE 范围 | 策略 |
|---------|------|
| < 2 | 安全，不处理 |
| [2, 5] | 保守混合（70% 结果 + 30% 原图） |
| > 5 | 自动回滚（仅半曝光 + 轻微提亮） |

### 性能分级

| 图片尺寸 | 预览模式 | 内存预估 |
|---------|---------|---------|
| ≤ 2000px | 原图 | < 300MB |
| 2000-4000px | 下采样 800px | < 500MB |
| 4000-8000px | 下采样 600px | < 800MB |
| > 8000px | 下采样 400px | < 1GB |

## 运行测试

```bash
python3 -m pytest tests/ -v
```

## 构建桌面应用

```bash
pip install pyinstaller
pyinstaller app.spec --clean -y
```

构建产物位于 `dist/追色工具/` 目录。

## 快捷键

| 快捷键 | 操作 |
|--------|------|
| ⌘ + Enter | 开始追色 |
| ⌘ + S | 保存结果 |

## 版本历史

- v2.1 — 新 70/30 布局、预设系统、评价指标、模块化架构
- v2.0 — 去 OpenCV + 模块化重构 + 专家管线
- v0.1.0 — 初始版本
