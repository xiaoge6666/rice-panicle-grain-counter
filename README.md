# Rice Panicle Grain Counter 水稻穗粒数计数工具

> **不脱粒、直接在整穗图像上逐粒计数**——基于 [EOPT](https://github.com/SUNJHZAU/EOPT)（Plant Phenomics 2024）官方检测权重，
> 外加**自标定的遮挡校正**与**逐粒编号可视化**。

| 指标 | 本工具 v2 | 官方模型直接检测（基线） |
|------|-----------|--------------------------|
| 单穗计数 MAPE（32 株人工真值，嵌套留一验证） | **10.3%** | 19.6% |
| 密穗段（>90 粒） | **13.8%** | 22.1% |
| 疏穗段（≤50 粒） | 7.0% | 18.1% |

> 官方论文的 PMI 遮挡校正模块**未公开权重**，本仓库用「32 株人工真值 + 图像特征回归」自行标定替代（见 `calibration/`）。

---

## 1. 三种使用方式

| 形态 | 适合谁 | 获取方式 |
|------|--------|----------|
| **Windows 免安装 exe** | 批量处理扫描图/照片（本机无 Python 也能跑） | 见 [Releases](../../releases) 的 `panicle-grain-counter-v2-windows-x64.exe` |
| **Android App (APK)** | 手机拍照即数、现场用 | 见 Releases 的 `panicle-grain-counter-v2.9-android-arm64.apk` |
| **源码** | 要改算法/重标定/集成 | 本仓库 |

**两个二进制的输出都包含**：逐穗粒数 CSV、整图标注、单穗检测框图、**逐粒编号图**（每粒一个框 + 序号，顶栏 `检测N / 校正M`）。

---

## 2. 算法流程

```
输入图 → ① 分穗（HSV 颜色掩码 + 形态学闭运算 + 连通域，自动排除尺子/杂色背景）
        → ② 每穗裁切 → YOLOv8 单类谷粒检测（GrainNuber.onnx，letterbox 保比例到 1280）
        → ③ 计数框 = NMS(conf=0.5, iou=0.5)
        → ④ 遮挡校正：med_area / 谷粒像素占比 / 框覆盖重数 → ridge 回归（λ=3）
              仅当 med_area 落在标定尺度域 [1197, 4455] 内才启用（防止换尺度外推失真）
        → ⑤ 输出：CSV + 标注图 + 逐粒编号图
```

### 关键经验（踩坑记录）

- **IoU 阈值比 conf 更关键**：0.8 → 0.5 使 MAPE 从 19.6% 降到 12.1%（同一粒会被检出多个重叠框）
- **官方默认 conf=0.7 不可用**：实测 MAPE 34%
- **误差方向随密度翻转**：疏穗被**高估**（颖壳/枝梗假阳性）、密穗被**低估**（遮挡），所以单一全局系数无效（实测仅从 19.6%→17.4%）
- **遮挡校正必须逐穗**：同密度箱内校正系数仍差 ±15~29%
- **校正模型含绝对像素量（框面积），换成像尺度即失效**：手机照片与扫描图尺度不同，需加尺度域保护

---

## 3. 目录结构

```
panicle_count_v2.py          # v2 生产流程（分穗→检测→校正→输出，109 图全量跑批示例）
eopt_count.py                # EOPT 权重推理封装（YOLOv8 ONNX，letterbox + NMS）
panicle_app.py               # Windows GUI v1（早期版本）
windows_gui/panicle_app_v2.py# Windows GUI v2（含逐粒编号输出、模型内嵌打包）
calibration/
  calibration_model.json     # ⭐ 标定模型（ridge 系数 + 尺度域 + 开关阈值）
  calibrate_model.py         # 用「配对真值」重标定 → 生成上面的 JSON
  nested_validate.py         # 嵌套留一交叉验证（诚实评估泛化误差）
  sweep_conf.py              # conf × IoU 网格搜索
  validate_recompute.py      # 与标定口径一致性自检
android/                     # Android App 源码（ONNX Runtime Android + Java 实现后处理）
docs/                        # 方法说明、各端使用说明
data/                        # 模型权重放置处（权重请从 Release 下载，勿提交进仓库）
```

---

## 4. 快速开始（源码）

```bash
pip install -r requirements.txt           # numpy opencv-python onnxruntime

# 1) 下载模型权重 GrainNuber.onnx（45MB，来自 EOPT 官方发布），放到项目根目录或 data/
#    见 Releases 的 GrainNuber.onnx 附件

# 2) 批量计数（输入文件夹 → 输出目录）
python panicle_count_v2.py               # 脚本内修改输入/输出路径（默认按扫描图流程）
# 或单张测试：见 windows_gui/panicle_app_v2.py 的 CLI 模式
python windows_gui/panicle_app_v2.py --cli <输入目录> <输出目录> 0.5
```

## 5. 重新标定（换成你自己的材料/成像条件）

```bash
# 1) 人工数一批穗的真实粒数（推荐 30~100 株，覆盖疏/中/密）
#    配对表格式：图片,穗序号,N_pred,N_true（N_pred 由第 4 步的检测流程给出）
# 2) 生成特征并拟合
python calibration/calibrate_model.py
# 3) 诚实评估
python calibration/nested_validate.py
```

---

## 6. Android 构建

```bash
cd android
# 需要 JDK 17 + Android SDK（compileSdk 34）
gradle assembleDebug        # 产物: app/build/outputs/apk/debug/app-debug.apk
```
模型与标定参数放在 `android/app/src/main/assets/`（`GrainNuber.onnx` + `calibration_model.json`）。

---

## 7. 引用

- Sun J, Ren Z, Cui J, Tang C, Luo T, Yang W, Song P. **A High-Throughput Method for Accurate Extraction of Intact Rice Panicle Traits.** *Plant Phenomics*, 2024, 6:0213. DOI: 10.34133/plantphenomics.0213（PMID 39091338）
- 范圣哲, 贡亮, 杨智宇, 等. 面向水稻穗上谷粒原位计数与遮挡还原的轻量级 I2I 深度学习方法. *华南农业大学学报*, 2023, 44(1): 74-83.

## 8. 许可与致谢

- 本仓库代码：科研用途，欢迎自取修改。
- `GrainNuber.onnx` 权重版权属 EOPT 作者团队，请遵循其原始许可并引用原论文。
