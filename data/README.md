# data/ — 模型权重放置处

**`GrainNuber.onnx` 不随仓库分发（45MB）**，请从 [Releases](../../releases) 下载附件后放到本目录（或项目根目录）。

- 来源：EOPT 官方发布（`Paniclee_Analyzer.rar` 内提取），YOLOv8 单类谷粒检测权重
- 输入：`images` `[1, 3, 1280, 1280]`（RGB、0~1，letterbox 保比例）
- 输出：`output0` `[1, 5, 33600]`，每行为 `(cx, cy, w, h, score)`
- 论文：Sun J, et al. *Plant Phenomics* 2024, 6:0213（PMID 39091338）

放好后目录应为：

```
data/GrainNuber.onnx
data/README.md
```
