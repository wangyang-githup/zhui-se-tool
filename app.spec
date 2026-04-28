# -*- mode: python ; coding: utf-8 -*-
# 追色工具 打包配置
# 用法: pyinstaller app.spec

import os
import sys
import shutil

block_cipher = None

# 项目根目录（spec 文件所在目录）
project_root = os.path.dirname(os.path.abspath(SPEC))

a = Analysis(
    [os.path.join(project_root, 'app.py')],
    pathex=[project_root],
    binaries=[],
    datas=[
        # 字体文件打包（如果有的话）
    ],
    hiddenimports=[
        'numpy',
        'PIL',
        'PIL._tkinter_finder',
        'scipy',
        'scipy.ndimage',
        'scipy.special',
        'customtkinter',
        'color_engine',
    ],
    hookspath=[],
    hooksconfig={},
    keys=[],
    privatescripts=[],
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='追色工具',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,            # 不显示终端窗口
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=None,               # 如有 .icns 图标可在此指定
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name='追色工具',
)

# ── 生成 macOS .app Bundle ──────────────────────────────────
app = BUNDLE(
    coll,
    name='追色工具.app',
    icon=None,                # 可在此指定 app.icns
    bundle_identifier='com.zhuisse.color-tool',
    info_plist={
        'CFBundleName': '追色工具',
        'CFBundleDisplayName': '追色工具',
        'CFBundleIdentifier': 'com.zhuisse.color-tool',
        'CFBundleVersion': '1.0.0',
        'CFBundleShortVersionString': '1.0.0',
        'CFBundlePackageType': 'APPL',
        'CFBundleExecutable': '追色工具',
        'LSMinimumSystemVersion': '10.15',
        'NSHighResolutionCapable': True,
        'LSApplicationCategoryType': 'public.app-category.photography',
    },
    force=False,
)
