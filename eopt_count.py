# -*- coding: utf-8 -*-
"""
EOPT 深度学习计数接入脚本 (等 GrainNuber.onnx 权重就位后使用)
来源: SUNJHZAU/EOPT (Plant Phenomics 2024, 计数准确率 93.57%)
用法:
  1. 从百度网盘下载 Panicle_Analyzer.rar (链接见 README_zh.md)
  2. 解压, 把 GrainNuber.onnx 放到本脚本同目录
  3. python eopt_count.py "图片文件夹" [--out 输出]
"""
import cv2
import numpy as np
import os
import sys
import glob
import csv
import time

def imread_unicode(path):
    data = np.fromfile(path, dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)

class YOLOv8:
    """从 EOPT ui2pyshow1014.py 提取的 YOLOv8 ONNX 推理类
    改进: letterbox 预处理保持宽高比(适配细长穗图), 坐标逆映射回原图
    """
    def __init__(self, onnx_model, confidence_thres=0.7, iou_thres=0.8):
        import onnxruntime as ort
        self.session = ort.InferenceSession(
            onnx_model, providers=["CPUExecutionProvider"])
        self.confidence_thres = confidence_thres
        self.iou_thres = iou_thres
        # 输入尺寸
        self.input_width = 640
        self.input_height = 640
        self.model_inputs = self.session.get_inputs()
        self.model_outputs = self.session.get_outputs()
        self.input_shape = self.model_inputs[0].shape
        if len(self.input_shape) == 4 and self.input_shape[2]:
            self.input_height, self.input_width = self.input_shape[2], self.input_shape[3]

    def preprocess(self, img_bgr):
        """letterbox 保持宽高比, 返回 (输入张量, 映射参数)"""
        self.img_height, self.img_width = img_bgr.shape[:2]
        img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        # letterbox: 长边缩放到输入尺寸, 短边 pad
        scale = min(self.input_width / img.shape[1], self.input_height / img.shape[0])
        nw, nh = int(round(img.shape[1] * scale)), int(round(img.shape[0] * scale))
        img = cv2.resize(img, (nw, nh))
        canvas = np.full((self.input_height, self.input_width, 3), 114, dtype=np.uint8)
        dx, dy = (self.input_width - nw) // 2, (self.input_height - nh) // 2
        canvas[dy:dy + nh, dx:dx + nw] = img
        self._pad = (dx, dy, scale)
        img = canvas.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))
        return np.expand_dims(img, axis=0).astype(np.float32)

    def postprocess(self, outputs):
        outputs = np.transpose(np.squeeze(outputs[0]))
        boxes, scores, class_ids = [], [], []
        dx, dy, sc = self._pad
        rows = outputs.shape[0]
        for i in range(rows):
            classes_scores = outputs[i][4:]
            max_score = np.amax(classes_scores)
            if max_score >= self.confidence_thres:
                class_id = int(np.argmax(classes_scores))
                x, y, w, h = outputs[i][0], outputs[i][1], outputs[i][2], outputs[i][3]
                # 逆映射回原图坐标
                left = int((x - w / 2 - dx) / sc)
                top = int((y - h / 2 - dy) / sc)
                width = int(w / sc)
                height = int(h / sc)
                boxes.append([left, top, width, height])
                scores.append(float(max_score))
                class_ids.append(class_id)
        # NMS
        indices = cv2.dnn.NMSBoxes(boxes, scores, self.confidence_thres, self.iou_thres)
        result = []
        if len(indices) > 0:
            indices = indices.flatten() if hasattr(indices, 'flatten') else indices
            for i in indices:
                b = boxes[i]
                result.append({'box': b, 'score': float(scores[i]), 'class_id': class_ids[i]})
        return result

    def detect(self, img_bgr):
        inp = self.preprocess(img_bgr)
        outputs = self.session.run([o.name for o in self.model_outputs],
                                   {self.model_inputs[0].name: inp})
        return self.postprocess(outputs)

def process_folder(in_dir, out_dir, onnx_path, conf=0.7):
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(os.path.join(out_dir, 'labeled'), exist_ok=True)
    model = YOLOv8(onnx_path, confidence_thres=conf)
    exts = ('*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG')
    files = []
    for ex in exts:
        files.extend(glob.glob(os.path.join(in_dir, '**', ex), recursive=True))
    files = [f for f in files if imread_unicode(f) is not None]
    rows = []
    t0 = time.time()
    for i, f in enumerate(sorted(files), 1):
        rel = os.path.relpath(f, in_dir)
        img = imread_unicode(f)
        dets = model.detect(img)
        n = len(dets)
        rows.append([rel, n])
        vis = img.copy()
        for d in dets:
            x, y, w, h = d['box']
            cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 200, 0), 2)
            cv2.putText(vis, '%.2f' % d['score'], (x, max(0, y-5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 0), 1)
        out_name = rel.replace(os.sep, '__').replace('/', '__')
        cv2.imwrite(os.path.join(out_dir, 'labeled', out_name + '.png'), vis)
        print('[%d/%d] %s -> %d 粒 (%.0fs)' % (i, len(files), rel, n, time.time()-t0))
    csv_path = os.path.join(out_dir, 'eopt_counts.csv')
    with open(csv_path, 'w', newline='', encoding='utf-8-sig') as fo:
        w = csv.writer(fo)
        w.writerow(['图片', '粒数(深度学习)'])
        w.writerows(rows)
    print('DONE %d 张 %.0fs' % (len(files), time.time()-t0))
    print('CSV:', csv_path)

if __name__ == '__main__':
    in_dir = sys.argv[1]
    out = sys.argv[2] if len(sys.argv) > 2 else in_dir + '_eopt'
    onnx = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'GrainNuber.onnx')
    if not os.path.exists(onnx):
        print('未找到 GrainNuber.onnx → 请先下载 Panicle_Analyzer.rar 并解压权重到本目录')
        sys.exit(1)
    process_folder(in_dir, out, onnx)