# -*- mode: python ; coding: utf-8 -*-
# 追色工具 打包配置 (v2.0 — 无OpenCV)
# 用法: pyinstaller app.spec

import os
import sys

block_cipher = None

project_root = os.path.dirname(os.path.abspath(SPEC))

# CustomTkinter 资源文件
import customtkinter as ctk
ctk_assets = os.path.join(os.path.dirname(ctk.__file__), 'assets')

a = Analysis(
    [os.path.join(project_root, 'app.py')],
    pathex=[project_root],
    binaries=[],
    datas=[
        # CustomTkinter 资源（必须打包，否则界面灰色）
        (ctk_assets, 'customtkinter/assets'),
    ],
    hiddenimports=[
        'numpy',
        'PIL',
        'PIL._tkinter_finder',
        'scipy',
        'scipy.special',
        'scipy.ndimage',
        'scipy.cluster',
        'sklearn',
        'sklearn.cluster',
        'sklearn.utils',
        'sklearn.utils._typedefs',
        'sklearn.utils._heap',
        'sklearn.utils._sorting',
        'sklearn.utils._vector_sentinel',
        'sklearn.neighbors',
        'sklearn.neighbors._partition_nodes',
        'customtkinter',
        'color_engine',
        'threading',
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
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=None,
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

app = BUNDLE(
    coll,
    name='追色工具.app',
    icon=None,
    bundle_identifier='com.zhuisse.color-tool',
    info_plist={
        'CFBundleName': '追色工具',
        'CFBundleDisplayName': '追色工具',
        'CFBundleIdentifier': 'com.zhuisse.color-tool',
        'CFBundleVersion': '2.0.0',
        'CFBundleShortVersionString': '2.0.0',
        'CFBundlePackageType': 'APPL',
        'CFBundleExecutable': '追色工具',
        'LSMinimumSystemVersion': '10.15',
        'NSHighResolutionCapable': True,
        'LSApplicationCategoryType': 'public.app-category.photography',
        'NSDocumentsFolderUsageDescription': '追色工具需要访问图片文件以进行色彩迁移',
        'NSPhotoLibraryUsageDescription': '追色工具需要访问照片以进行色彩迁移',
    },
    force=False,
)
