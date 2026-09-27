# -*- coding: utf-8 -*-
"""验证 recompute_v2 路径在 32 GT 穗上的表现"""
import sys, os, json
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
import numpy as np
sys.path.insert(0, r'C:\Users\Administrator\.openclaw\workspace\tmp_rice')
import recompute_v2 as R

BASE = R.BASE
feats = json.load(open(r'C:\Users\Administrator\.openclaw\workspace\tmp_rice\features_32.json', encoding='utf-8'))
model = R.YOLO(R.ONNX)
CROPS = os.path.join(BASE, '人工真值核对', 'crops')
true_all, fin_all, det_all = [], [], []
print(f'{"pid":<5}{"真值":<6}{"检测":<6}{"校正后":<7}{"误差%"}')
for r in feats:
    img = R.imread_u(os.path.join(CROPS, r['pid'] + '.png'))
    boxes, scores = model.raw_boxes(img)
    n_det, kept = R.nms_count(boxes, scores)
    ma, rg, oc = R.features(boxes, scores, img)
    x = np.array([n_det, ma, rg, oc], float)
    est = float(max(1.0, R.YM + ((x - R.MU) / R.SD) @ R.W))
    final = est if n_det >= R.T else float(n_det)
    e = abs(final - r['n_true']) / r['n_true'] * 100
    true_all.append(r['n_true']); fin_all.append(final); det_all.append(n_det)
    print(f'{r["pid"]:<5}{r["n_true"]:<6}{n_det:<6}{final:<7.1f}{e:.0f}')
t = np.array(true_all, float); f = np.array(fin_all); d = np.array(det_all, float)
print(f'\n部署路径 MAPE = {np.mean(np.abs(f-t)/t)*100:.1f}%  (纯检测 {np.mean(np.abs(d-t)/t)*100:.1f}%)')
print(f'检测数与标定时一致吗:', np.allclose(d, [r["n_use"] * 1.0 for r in feats]) or '（conf 不同，n_use 来自 0.55）')
