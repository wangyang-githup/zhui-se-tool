# 追色工具 · 更新日志

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/).
This project uses [Semantic Versioning](https://semver.org/).

## [2.0.0] - 2026-04-29

### Changed — 重大架构重构

- **统一引擎 (color_engine.py)**：将桌面版简化引擎 + Web版5模块专家管线整合为 `ZhuiseEngine`
  - 两种工作模式：简化模式 (Reinhard Lab 迁移) + 专家模式 (Lightroom 风格完整参数面板)
  - 新增 `ZhuiseEngine` 统一入口，一键切换模式
  - 新增 `FeatureExtractor`：影调分位数 + Lab 统计 + K-Means 色板 + 肤色锚点 + 环境主色
  - 新增 `ColorGradingGenerator`：自动生成曝光/对比度/高光/阴影/白色/黑色 + RGB曲线 + HSL 8色通道 + 分离色调 + 白平衡校准
  - 新增 `ColorRenderer`：模拟 Lightroom 处理管线，含 ΔE 监控肤色保护 + 自动回滚
  - 新增 `Evaluator`：直方图相似度 + 主色色相差 + ΔE + Lab 匹配度综合评价
  - 新增 `StylePresets`：5 组内置风格预设（日系清新/暗调电影/复古胶片/青橙色调/Kodak Portra）

- **彻底去除 OpenCV 依赖**
  - RGB↔Lab 转换：纯 NumPy 实现（sRGB D65 → XYZ → CIE Lab 标准路径），往返精度完美
  - RGB↔HSV 转换：纯 NumPy 实现
  - 肤色检测：从 YCrCb+HSV 改为 HSV+Lab 双空间（更精准的亚洲人肤色范围）
  - 图像 I/O：统一使用 PIL（更鲁棒的文件路径处理，修复 .app 打包后加载失败）
  - 形态学操作：scipy.ndimage 替代 cv2.morphologyEx
  - K-Means：sklearn（可选，缺失时自动降级为 NumPy 简易实现）

- **GUI 升级为 CustomTkinter**
  - 紫色暗色主题 (Catppuccin Mocha)，解决标准 Tkinter 版本 UI 不显示问题
  - 标签页式参数面板：基础 / 进阶 / 风格
  - 基础标签：影调/色调强度 + 肤色保护 + 全局强度
  - 进阶标签：曝光/对比度/高光/阴影/白色/黑色 + 分离色调 + 白平衡 + 自动分析按钮
  - 风格标签：5 组预设一键应用 + 参考图特征显示
  - 评价指标实时显示

- **打包配置更新**
  - app.spec 移除 cv2 依赖，添加 customtkinter 资源和 sklearn hidden imports
  - 版本号升级至 2.0.0

## [0.1.0] - 2026-04-27

### Added
- **追色引擎 (color_engine.py)**
  - Lab 色彩迁移（Reinhard 方法）：在 L*a*b* 空间精准迁移影调与色调
  - 影调强度 / 色调强度独立滑块控制
  - 肤色保护遮罩：YCrCb + HSV 双空间检测，高斯羽化边缘，自然过渡
  - 3D LUT 生成器：输出标准 .cube 文件，可导入 PR / AE / 达芬奇

- **桌面应用 (app.py)**
  - 三格实时预览：参考图 / 原图 / 结果图
  - 一键导入图片、滑动调参自动预览
  - 保存结果图（JPG / PNG / TIFF）
  - 导出 .cube LUT（三档精度：17 / 33 / 65）
  - macOS 原生深色界面（Catppuccin 配色）

- **启动器**
  - `启动追色工具.command`：自动检测依赖，一键启动
