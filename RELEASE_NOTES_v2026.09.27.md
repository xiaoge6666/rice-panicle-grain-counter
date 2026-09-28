# 水稻穗粒数计数工具 v2026.09.27

**整穗不脱粒、逐粒计数**（EOPT 检测权重 + 自标定遮挡校正 + 逐粒编号）

## 精度

| 指标 | 本版 v2 | 官方模型裸检测 |
|---|---|---|
| 单穗 MAPE（32 株人工真值，嵌套 LOO） | **10.3%** | 19.6% |
| 密穗段（>90 粒） | 13.8% | 22.1% |
| 疏穗段（≤50 粒） | 7.0% | 18.1% |

## 下载哪个

- **Windows 用户** → `panicle-grain-counter-v2-windows-x64.exe`（免安装，双击即用；无需 Python、无需联网）
- **手机用户** → `panicle-grain-counter-v2.9-android-arm64.apk`（Android 7.0+，arm64）
- **跑源码** → 另需 `GrainNuber.onnx`，放到项目根目录或 `data/`

## 本版要点

- 检测参数重标定：**conf=0.5 / iou=0.5**（iou 由 0.8 收紧是关键修复）
- **逐穗遮挡校正**：ridge 回归（框面积中位数/谷粒像素占比/框覆盖重数），含
  **尺度域保护**（`med_area ∈ [1197, 4455]` 才启用，避免换成像尺度外推失真）与
  自适应开关（检出<50 粒视为无遮挡，不校正）
- 输出：逐穗 CSV、整图标注、单穗检测框图、**逐粒编号图**（顶栏 `检测N / 校正M`）
- Android 端：分穗多阈值重试、EXIF 方向纠正、崩溃自愈、**结果图点击全屏放大**

## 文件校验（SHA256）

| 文件 | 大小 | SHA256 |
|---|---|---|
| `panicle-grain-counter-v2-windows-x64.exe` | 118.1 MB | `2135e22ec3d7d3a6ad57211b7fb71b72f1a706dd4569061af781e79fd0965db7` |
| `panicle-grain-counter-v2.9-android-arm64.apk` | 46.0 MB | `3a3c675e39eeeffd7ec63c63c2dba7dac47b4d71cd8415c05da044df282e0320` |
| `GrainNuber.onnx` | 45.2 MB | `f38d671cdabc78915dc56a83782ca4b69918d62a8cbe524add3ca42a168bc55c` |

## 注意

- **不要只下载 Source code**（那是源码压缩包，不含程序）。
- 权重 `GrainNuber.onnx` 版权属 EOPT 作者团队（Sun J, et al. *Plant Phenomics* 2024, 6:0213, PMID 39091338），请按其许可使用并引用原论文。
- 手机拍照请用**纯黑/纯白背景**，不要拍电脑屏幕或杂色桌面（否则背景会被并入穗区域）。

更新日期：2026-09-27
