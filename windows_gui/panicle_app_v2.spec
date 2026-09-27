# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller 打包配置（Windows GUI v2）
用法（在 windows_gui 目录下、与 GrainNuber.onnx / calibration_model.json 同目录）：
    python -m PyInstaller panicle_app_v2.spec
产物：dist/水稻穗粒数计数工具_v2.exe（单文件，内嵌模型与标定参数）
"""
block_cipher = None

a = Analysis(
    ['panicle_app_v2.py'],
    pathex=[],
    binaries=[],
    datas=[
        ('GrainNuber.onnx', '.'),
        ('calibration_model.json', '.'),
    ],
    hiddenimports=['onnxruntime', 'cv2', 'numpy'],
    hookspath=[],
    runtime_hooks=[],
    excludes=['matplotlib', 'PyQt5', 'PySide2', 'scipy', 'pandas'],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='水稻穗粒数计数工具_v2',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,          # GUI 程序；CLI 模式日志写入 输出目录/cli_log.txt
    disable_windowed_traceback=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
