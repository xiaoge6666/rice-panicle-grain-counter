# -*- coding: utf-8 -*-
"""
conf 阈值扫描：32 个人工真值穗 → 找最优检测策略
- 每个穗图推理一次 (conf=0.05 全召回)，离线扫阈值 + NMS
- 对比各阈值下的 MAPE（总体 / 疏 / 中密）
"""
import sys, os, csv, time
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
import cv2
import numpy as np
import onnxruntime as ort

sys.path.insert(0, r'J:\claw\水稻穗粒数计数_2026-09-12\EOPT版')
from eopt_count import YOLOv8, imread_unicode

BASE = r'J:\claw\水稻穗粒数计数_2026-09-12'
CROPS = os.path.join(BASE, '人工真值核对', 'crops')
ONNX = os.path.join(BASE, 'EOPT版', 'GrainNuber.onnx')

# 真值
truth = {}
for r in csv.DictReader(open(os.path.join(BASE, '人工真值核对', '配对结果_32穗.csv'), encoding='utf-8-sig')):
    truth[r['编号']] = int(r['N_true'])

model = YOLOv8(ONNX, confidence_thres=0.05, iou_thres=0.99)  # 近关闭NMS，拿原始框
print('模型输入尺寸:', model.input_width, 'x', model.input_height)

raw = {}   # pid -> list[(box, score)]
t0 = time.time()
for pid in sorted(truth):
    p = os.path.join(CROPS, f'{pid}.png')
    img = imread_unicode(p)
    dets = model.detect(img)
    raw[pid] = [(d['box'], d['score']) for d in dets]
    print(f'{pid}: {len(dets)} raw boxes  ({time.time()-t0:.0f}s)', flush=True)

np.save(os.path.join(r'C:\Users\Administrator\.openclaw\workspace\tmp_rice', 'raw_boxes.npy'),
        np.array([(r['编号'] if False else 0) for r in []], dtype=object), allow_pickle=True)

# 离线扫阈值
def count_at(pid, thr, iou=0.8):
    boxes = [b for b, s in raw[pid] if s >= thr]
    scores = [s for b, s in raw[pid] if s >= thr]
    if not boxes: return 0
    idx = cv2.dnn.NMSBoxes(boxes, scores, thr, iou)
    return len(idx) if idx is not None and len(idx) > 0 else 0

print(f'\n{"conf":<6}{"总MAPE":<9}{"疏(n=14)":<10}{"中密(n=18)":<11}{"过检穗":<7}{"漏检穗"}')
best = {}
for thr in [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]:
    errs, errs_s, errs_m = [], [], []
    n_over = n_under = 0
    for pid, t in truth.items():
        c = count_at(pid, thr)
        e = abs(c - t) / t
        errs.append(e)
        (errs_s if t <= 50 else errs_m).append(e)
        if c > t: n_over += 1
        elif c < t: n_under += 1
    line = f'{thr:<6.2f}{np.mean(errs)*100:<9.1f}{np.mean(errs_s)*100:<10.1f}{np.mean(errs_m)*100:<11.1f}{n_over:<7}{n_under}'
    print(line)
    best[thr] = np.mean(errs)

opt = min(best, key=best.get)
print(f'\n最优全局阈值: conf={opt} (MAPE {best[opt]*100:.1f}%)  —— 当前部署用 0.5/0.7，对照: 0.5→{best[0.5]*100:.1f}%, 0.7→{best[0.7]*100:.1f}%')

# 逐穗明细（最优阈值 vs 0.5）
print(f'\n{"编号":<5}{"人工":<5}{"c=0.5":<7}{"c=opt":<7}{"最优conf":<9}')
for pid in sorted(truth):
    c5 = count_at(pid, 0.5)
    co = count_at(pid, opt)
    # 该穗自己的最优阈值
    cand = [(abs(count_at(pid, th) - truth[pid]), th) for th in [0.3,0.4,0.5,0.6,0.7,0.8]]
    cand.sort()
    print(f'{pid:<5}{truth[pid]:<5}{c5:<7}{co:<7}{cand[0][1]}')
print('DONE')
