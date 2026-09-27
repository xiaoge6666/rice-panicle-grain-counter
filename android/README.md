# Android App（穗粒数计数 v2）

拍照/选图 → 自动分穗 → 逐粒检测 + 遮挡校正 → 图上逐粒编号 + 每穗「检测N / 校正M」→ 可保存到相册；结果图可点击全屏放大（双指缩放）。

## 技术要点

- **Chaquopy 无法安装 onnxruntime**（无 Android wheel），因此本 App 采用
  **ONNX Runtime Android（`com.microsoft.onnxruntime:onnxruntime-android:1.19.2`）+ Java 自行实现后处理**
  （letterbox、NMS、特征提取、编号渲染全部在 Java 侧）。
- 算法与桌面版**同口径**（同一份标定参数、同样的 conf=0.5/iou=0.5、同样的尺度域保护）。
- 模型与标定放在 `app/src/main/assets/`：`GrainNuber.onnx`（从 Release 下载）+ `calibration_model.json`。

## 构建

```bash
cd android
# 需要 JDK 17 与 Android SDK（compileSdk 34, minSdk 24, abiFilters arm64-v8a）
gradle assembleDebug --no-daemon
# 产物 app/build/outputs/apk/debug/app-debug.apk
```

## 已知边界

- 遮挡校正仅在**与标定一致的成像尺度**下启用（`med_area ∈ [1197, 4455]`）；
  手机拍摄的照片尺度不同 → 自动跳过校正、直接输出检测数（防止线性模型外推失真）
- 拍照/选图时**不要拍电脑屏幕或杂色桌面**（背景会被并入穗区域）；纯黑/纯白背景最稳
- 若希望手机照片也带校正，需要用手机拍的「配对真值」重新标定一版参数（见 `calibration/`）

## 版本记录

| 版本 | 修复/新增 |
|------|-----------|
| v2.0 | 首版（分穗 + 检测 + 校正 + 编号） |
| v2.1 | 修「二次拍照闪退」（内存：采样解码 + 释放旧结果 + 捕获 Throwable） |
| v2.2 | 修「校正模型跨尺度外推失真」（尺度域保护）；EXIF 方向纠正；编号加粗 |
| v2.3 | 修「拍照即闪退」（移除 CAMERA 权限声明陷阱；模型改为用时加载、用完释放） |
| v2.4 | 异常记录显示改为「异常类型+位置」；修 Bitmap 回收竞态 |
| v2.5 | 分穗多阈值自动重试；「背景太杂」提示 |
| v2.6 | **结果图点击全屏放大**（双指缩放/拖动/双击） |
