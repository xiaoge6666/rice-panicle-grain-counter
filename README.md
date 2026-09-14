# Rice Panicle Grain Counter 水稻穗粒数计数工具

基于深度学习的穗扫描图**逐穗谷粒计数**工具。输入黑底穗扫描图片文件夹，自动分穗、逐穗计数，输出 CSV 统计表 + 标注图。带图形界面（Windows 免安装 exe）和命令行批量模式。

## 🧬 来源与致谢 (Source & Credits)

**本项目为开源项目 EOPT 的改造与封装**，核心深度学习模型与推理代码源自：

- **原始开源项目**: [SUNJHZAU/EOPT](https://github.com/SUNJHZAU/EOPT)
- **原论文**: *High-throughput and separating-free phenotyping method for on-panicle rice grains based on deep learning*, Plant Phenomics (2024), DOI: [10.34133/PlantPhenomics.0030](https://doi.org/10.34133/PlantPhenomics.0030), PMC: [PMC11292034](https://pmc.ncbi.nlm.nih.gov/articles/PMC11292034/)
- **模型**: EOPT 作者发布的官方 YOLOv8 单类谷粒检测权重 `GrainNuber.onnx`（计数准确率 93.57%，论文报告）
- **推理代码**: `eopt_count.py` 修改自 EOPT 官方 `ui2pyshow1014.py` 的 YOLOv8 ONNX 推理类

> ⚠️ EOPT 论文中的遮挡补偿模块（Pix2Pix）权重作者未发布，因此密集紧贴谷粒存在 10–20% 漏检——这是原作者模型本身的边界，非本封装引入。

## 🔧 本项目的改造 (Modifications)

在原 EOPT 基础上做了以下改造：

| 模块 | 改造内容 |
|---|---|
| **图像预处理** | 原版直接 resize 拉伸至 1280×1280（细长穗畸变严重、漏检）；改为 **letterbox 保比例** + 坐标逆映射回原图（细长穗 4 粒 → 38 粒） |
| **自动分穗** | 原版需人工框选穗；新增 `find_panicles`：HSV 谷粒掩码 + 闭运算连通域实现自动分穗 |
| **尺子排除** | 原版会把图上标尺误当穗；新增双规则排除（绿色穗轴占比 <0.5% 判为尺子；长宽比 >25 排除） |
| **图形界面** | 新增 tkinter GUI：选文件夹 → 开始计数 → CSV + 标注图 |
| **命令行模式** | 新增 `--cli 输入 输出 [conf]` 批量模式（windowed exe 日志落盘 cli_log.txt） |
| **封装分发** | PyInstaller 单文件 exe（模型内嵌），拷到任意 Win10/11 64 位电脑免安装直接运行 |
| **置信度** | 默认 conf=0.5（官方 0.7 对扫描图漏检多；实测 0.5 更稳） |

## 📦 安装与使用

### 方式 A：免安装 exe（推荐，无需 Python）

从 **GitHub Releases** 下载 `rice-panicle-grain-counter_v*.zip`，解压后双击 `水稻穗粒数计数工具.exe`：

1. 点【浏览...】选择穗扫描图片文件夹（支持 jpg/png/bmp/tif）
2. 点【浏览...】选择结果保存位置
3. 置信度保持默认 0.5，点【开始计数】
4. 输出：`穗粒数统计.csv`（Excel 可开）+ `labeled\` 整图标注 + `per_panicle\` 单穗标注

命令行批量：`水稻穗粒数计数工具.exe --cli "输入文件夹" "输出文件夹" 0.5`

### 方式 B：源码运行

```bash
pip install -r requirements.txt
# 模型权重下载: GitHub Releases Assets 里下载 GrainNuber.onnx 放到本目录
# (exe 用户无需下载, 模型已内嵌)
python panicle_app.py                     # GUI
python panicle_app.py --cli 输入 输出 0.5 # CLI
```

### 重新打包 exe

```bash
pip install pyinstaller
pyinstaller panicle_app.spec --noconfirm
# 产物: dist/水稻穗粒数计数工具.exe
```

## 📋 输入输出

- **输入**: 黑底、穗完整、谷粒黄色的扫描图片（本工具按此成像条件调参）
- **输出**: CSV 逐穗统计（每行=图，每列=穗粒数，末列合计）；labeled 整图标注；per_panicle 单穗标注（绿框=每粒）

## 🔬 方法要点（写论文引用用）

1. **分穗**: HSV 颜色掩码 (S>35, V>70, H<50|>140, 排除绿 H40-90) → 闭运算 31×31 → 连通域
2. **检测**: YOLOv8 单类谷粒，letterbox 保比例 → 1280×1280 → onnxruntime CPU 推理 → NMS (conf=0.5, iou=0.8)
3. **统计**: 每穗检测框数 = 谷粒数；尺子双规则排除

## 📁 仓库结构

```
├─ panicle_app.py      # 主程序（GUI + CLI + 分穗 + 主流程）
├─ eopt_count.py       # YOLOv8 ONNX 推理类（改造自 EOPT ui2pyshow1014.py）
├─ panicle_app.spec    # PyInstaller 打包配置
├─ requirements.txt    # 运行依赖
└─ README.md
```

> `GrainNuber.onnx`（43MB，EOPT 官方权重）在 **GitHub Releases Assets** 下载：<https://github.com/xiaoge6666/rice-panicle-grain-counter/releases>。exe 免安装版已内置模型，无需单独下载。

## ⚖️ License

- 本项目代码基于 EOPT 改造，遵循 EOPT 原项目许可。
- `GrainNuber.onnx` 为 EOPT 作者发布，版权归原作者；学术使用请引用 EOPT 原论文。

> 来自科研助手（镰刀君）定制开发的田间表型工具，有改进建议欢迎 issue。