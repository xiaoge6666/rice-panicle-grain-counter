# -*- coding: utf-8 -*-
"""PyInstaller 打包配置: 水稻穗粒数计数工具
说明: 在本仓库根目录执行 `pyinstaller panicle_app.spec --noconfirm`
即可打包出 dist/水稻穗粒数计数工具.exe (模型 GrainNuber.onnx 自动内嵌)。
"""
import os

# 模型权重(与 spec 同目录)
onnx = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'GrainNuber.onnx')

a = Analysis(
    ['panicle_app.py'],
    pathex=[],
    binaries=[],
    datas=[(onnx, '.')],   # 内嵌 GrainNuber.onnx 到 exe 根目录
    hiddenimports=[
        'onnxruntime',
        'onnxruntime.capi._pybind_state',
        'onnxruntime.capi.onnxruntime_pybind11_state',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['matplotlib', 'PIL', 'IPython', 'notebook', 'pandas'],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='水稻穗粒数计数工具',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,          # GUI 模式,不弹黑框
    disable_windowed_traceback=False,
    icon=None,
)