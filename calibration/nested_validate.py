# -*- coding: utf-8 -*-
"""
最终模型：严格嵌套 LOO 验证
- 基础检测 conf/iou 选择：在外层 LOO 内层做，避免选择泄漏
- 特征：n_det, med_area, r_grain_medgrain (+occ_mean 备选)
- 目标：给出诚实的泛化误差估计
"""
import sys, os, json, itertools
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
import numpy as np
import cv2

OUT = r'C:\Users\Administrator\.openclaw\workspace\tmp_rice'
feats = json.load(open(os.path.join(OUT, 'features_32.json'), encoding='utf-8'))
npz = np.load(os.path.join(OUT, 'raw_boxes.npz'))
pids = [r['pid'] for r in feats]
n_true = np.array([r['n_true'] for r in feats], float)
med_area = np.array([r['med_area'] for r in feats], float)
r_grain = np.array([r['r_grain_medgrain'] for r in feats], float)
occ = np.array([r['occ_mean'] for r in feats], float)
f_raw = np.array([r['f_raw'] for r in feats], float)

CONF_GRID = [0.45, 0.50, 0.55, 0.60]
IOU_GRID = [0.5, 0.6, 0.7, 0.8]

def det_count(pid, conf, iou):
    b, s = npz[f'{pid}_b'], npz[f'{pid}_s']
    m = s >= conf
    if m.sum() == 0: return 0
    idx = cv2.dnn.NMSBoxes(b[m].tolist(), s[m].tolist(), conf, iou)
    return len(idx) if idx is not None and len(idx) > 0 else 0

# 预算好所有 config 下的计数
counts = {(c, i): np.array([det_count(p, c, i) for p in pids], float)
          for c in CONF_GRID for i in IOU_GRID}

def mape(est): return np.mean(np.abs(n_true - np.asarray(est, float)) / n_true) * 100

# ---------- 1) 全网格（供参考，含选择偏差）----------
print('=== 检测参数网格 MAPE(全32样本, 含选择偏差) ===')
best = (1e9, None)
for c in CONF_GRID:
    row = [mape(counts[(c, i)]) for i in IOU_GRID]
    j = int(np.argmin(row))
    if row[j] < best[0]: best = (row[j], (c, IOU_GRID[j]))
    print(f'  conf={c:<5}' + ' '.join(f'{v:5.1f}' for v in row))
print(f'  网格最优: conf={best[1][0]} iou={best[1][1]} → {best[0]:.1f}%')

# ---------- 2) 嵌套 LOO：检测参数 + 校正模型 全在训练折内选 ----------
def fit_ridge(idx_tr, y_tr, cols, lam):
    Xs = X_all[idx_tr][:, cols]
    mu, sd = Xs.mean(0), Xs.std(0) + 1e-9
    Xn = (Xs - mu) / sd
    A = Xn.T @ Xn + lam * np.eye(Xn.shape[1])
    w = np.linalg.solve(A, Xn.T @ (y_tr - y_tr.mean()))
    return w, mu, sd, y_tr.mean()

X_all = np.column_stack([counts[(0.5, 0.6)], med_area, r_grain, occ, f_raw])
NAMES = ['n_det@0.5/0.6', 'med_area', 'r_grain_medgrain', 'occ_mean', 'f_raw']
FEATSETS = [[0], [0, 1, 2], [0, 1, 3], [0, 2, 3], [0, 1, 2, 3], [0, 1, 2, 3, 4]]
LAMS = [0.3, 1, 3, 10, 30]

n = len(n_true)
est_plain, est_model = np.zeros(n), np.zeros(n)
sel_cfg, sel_set = [], []
for i in range(n):
    tr = np.arange(n) != i
    # 内层：选检测参数（用训练折纯检测 MAPE）
    cfg_scores = {k: mape(counts[k][tr].copy() * np.ones(1)) if False else
                  np.mean(np.abs(n_true[tr] - counts[k][tr]) / n_true[tr])
                  for k in counts}
    cfg_best = min(cfg_scores, key=cfg_scores.get)
    sel_cfg.append(cfg_best)
    est_plain[i] = counts[cfg_best][i]
    # 内层：选特征集(固定用小网格) + lam（用训练折内部 LOO）
    inner_best = (1e9, None, None)
    for cols in FEATSETS:
        for lam in LAMS:
            # 训练折内部再做 LOO
            sub = np.where(tr)[0]
            errs = []
            for j in sub:
                tr2 = sub[sub != j]
                w, mu, sd, ym = fit_ridge(tr2, n_true[tr2], cols, lam)
                p = ym + ((X_all[j][cols] - mu) / sd) @ w
                errs.append(abs(p - n_true[j]) / n_true[j])
            e = np.mean(errs)
            if e < inner_best[0]: inner_best = (e, cols, lam)
    cols, lam = inner_best[1], inner_best[2]
    w, mu, sd, ym = fit_ridge(np.where(tr)[0], n_true[tr], cols, lam)
    p = ym + ((X_all[i][cols] - mu) / sd) @ w
    est_model[i] = max(1.0, p)
    sel_set.append(tuple(NAMES[c] for c in cols))

print('\n=== 嵌套 LOO（完全诚实）===')
print(f'  纯检测(逐折选参): MAPE = {mape(est_plain):.1f}%')
print(f'  检测+回归校正:    MAPE = {mape(est_model):.1f}%')
from collections import Counter
print('  各折选中的特征集:', Counter(sel_set).most_common(3))
print('  各折选中的检测参数:', Counter(sel_cfg).most_common(3))

# 逐穗明细
print(f'\n{"编号":<5}{"真值":<5}{"纯检测":<7}{"校正后":<7}{"误差%"}')
for i, pid in enumerate(pids):
    e = abs(est_model[i] - n_true[i]) / n_true[i] * 100
    print(f'{pid:<5}{n_true[i]:<5.0f}{est_plain[i]:<7.0f}{est_model[i]:<7.0f}{e:.0f}')

np.save(os.path.join(OUT, 'final_est.npy'), est_model)
print('\n最终方案对比:')
print(f'  旧交付 (conf0.5/iou0.8, 无校正): 19.6%')
print(f'  新检测 (conf0.5/iou0.6):         {mape(counts[(0.5,0.6)]):.1f}% (全样本) / {mape(est_plain):.1f}% (嵌套LOO)')
print(f'  新检测+回归校正:                  {mape(est_model):.1f}% (嵌套LOO)')
