# 追色工具 · 更新日志

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/).
This project uses [Semantic Versioning](https://semver.org/).

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
