# -*- coding: utf-8 -*-
"""
1) 用 32 对真值拟合最终校正模型 → 保存 JSON
   检测: conf=0.5, iou=0.5 | 校正特征: [n_det, med_area, r_grain_medgrain, occ_mean] 的 ridge
2) 打印最终系数与自检
"""
import sys, os, json
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
import numpy as np
import cv2

OUT = r'C:\Users\Administrator\.openclaw\workspace\tmp_rice'
BASE = r'J:\claw\水稻穗粒数计数_2026-09-12'
feats = json.load(open(os.path.join(OUT, 'features_32.json'), encoding='utf-8'))
npz = np.load(os.path.join(OUT, 'raw_boxes.npz'))
pids = [r['pid'] for r in feats]
y = np.array([r['n_true'] for r in feats], float)
med_area = np.array([r['med_area'] for r in feats], float)
r_grain = np.array([r['r_grain_medgrain'] for r in feats], float)
occ = np.array([r['occ_mean'] for r in feats], float)

CONF, IOU = 0.5, 0.5
def n_det(pid):
    b, s = npz[f'{pid}_b'], npz[f'{pid}_s']
    m = s >= CONF
    if m.sum() == 0: return 0
    idx = cv2.dnn.NMSBoxes(b[m].tolist(), s[m].tolist(), CONF, IOU)
    return len(idx) if idx is not None and len(idx) > 0 else 0

X = np.column_stack([[n_det(p) for p in pids], med_area, r_grain, occ])
n = len(y)

def loo_err(lam):
    errs = []
    for i in range(n):
        tr = np.arange(n) != i
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-9
        Xt = (X[tr] - mu) / sd
        A = Xt.T @ Xt + lam * np.eye(4)
        w = np.linalg.solve(A, Xt.T @ (y[tr] - y[tr].mean()))
        p = y[tr].mean() + ((X[i] - mu) / sd) @ w
        errs.append(abs(p - y[i]) / y[i])
    return float(np.mean(errs))

lam_scores = {l: loo_err(l) for l in [0.3, 1, 3, 10, 30]}
LAM = min(lam_scores, key=lam_scores.get)
print('各 lam 的 LOO MAPE:', {k: f'{v*100:.1f}%' for k, v in lam_scores.items()})
print(f'选用 lam={LAM} → LOO MAPE={lam_scores[LAM]*100:.1f}%')

# 全量拟合
mu, sd = X.mean(0), X.std(0) + 1e-9
Xn = (X - mu) / sd
A = Xn.T @ Xn + LAM * np.eye(4)
w = np.linalg.solve(A, Xn.T @ (y - y.mean()))
ym = float(y.mean())

model = {
    'version': 'v2-2026-09-26',
    'detection': {'conf': CONF, 'iou': IOU, 'input': 1280},
    'features': ['n_det', 'med_area', 'r_grain_medgrain', 'occ_mean'],
    'mu': mu.tolist(), 'sd': sd.tolist(), 'w': w.tolist(), 'y_mean': ym,
    'lam': LAM,
    'calibration': {'n_pairs': n, 'loo_mape_pct': round(lam_scores[LAM] * 100, 2),
                    'baseline_detection_mape_pct': round(float(np.mean(np.abs(y - X[:, 0]) / y)) * 100, 2)},
    'note': 'k=N_true/N_pred 遮挡校正；特征来自单穗crop(20px边距)；med_area=conf>=0.3框面积中位; r_grain=谷粒像素面积/med_area; occ_mean=框覆盖重数均值'
}
p = os.path.join(BASE, 'EOPT版', 'calibration_model.json')
json.dump(model, open(p, 'w', encoding='utf-8'), ensure_ascii=False, indent=2)
print('模型已存:', p)

# 全样本预测自检
pred = np.maximum(1.0, ym + Xn @ w)
print(f'\n全样本(含拟合) MAPE: {np.mean(np.abs(y-pred)/y)*100:.1f}%  | 纯检测: {np.mean(np.abs(y-X[:,0])/y)*100:.1f}%')
print(f'{"pid":<5}{"true":<6}{"det":<6}{"est":<7}')
for i, pid in enumerate(pids):
    print(f'{pid:<5}{y[i]:<6.0f}{X[i,0]:<6.0f}{pred[i]:<7.0f}')
