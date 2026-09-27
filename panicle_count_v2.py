# -*- coding: utf-8 -*-
"""
v2 最终一遍出（顺序与旧交付一致：find_panicles 按 -area 排序）
产出:
  穗粒数_v2校正版.csv / 穗粒数_v2原始检测.csv / features_v2_all.csv
  labeled/                 整图标注 (P#=校正后粒数)
  per_panicle_dl/          单穗检测框图 (实际计入的 NMS 框)
  per_panicle_numbered/    单穗逐粒编号图 (序号 + 顶栏 检测/校正后)
"""
import os, sys, csv, json, time
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
import cv2
import numpy as np
import onnxruntime as ort

BASE = r'J:\claw\水稻穗粒数计数_2026-09-12'
SRC = r'J:\素材\RNAI\RNAi\农艺性状\2026水稻表型'
OUT = os.path.join(BASE, 'EOPT版_v2_校正版')
ONNX = os.path.join(BASE, 'GrainNuber.onnx')
CAL = json.load(open(os.path.join(BASE, 'EOPT版', 'calibration_model.json'), encoding='utf-8'))
MU = np.array(CAL['mu']); SD = np.array(CAL['sd']); W = np.array(CAL['w']); YM = CAL['y_mean']
CONF, IOU, T = CAL['detection']['conf'], CAL['detection']['iou'], CAL['switch_threshold']

def imread_u(p): return cv2.imdecode(np.fromfile(p, dtype=np.uint8), cv2.IMREAD_COLOR)
def imwrite_u(p, img):
    ok, buf = cv2.imencode(os.path.splitext(p)[1], img)
    if ok: buf.tofile(p)

class YOLO:
    def __init__(self, path):
        self.sess = ort.InferenceSession(path, providers=['CPUExecutionProvider'])
        self.inp = self.sess.get_inputs()[0]
        self.outs = [o.name for o in self.sess.get_outputs()]
        s = self.inp.shape
        self.iw = s[3] if isinstance(s[3], int) else 1280
        self.ih = s[2] if isinstance(s[2], int) else 1280

    def raw_boxes(self, img):
        h, w = img.shape[:2]
        s = min(self.iw / w, self.ih / h)
        nw, nh = int(round(w * s)), int(round(h * s))
        r = cv2.resize(img, (nw, nh))
        canvas = np.full((self.ih, self.iw, 3), 114, np.uint8)
        dx, dy = (self.iw - nw) // 2, (self.ih - nh) // 2
        canvas[dy:dy + nh, dx:dx + nw] = r
        t = canvas[:, :, ::-1].astype(np.float32) / 255.0
        inp = np.expand_dims(np.transpose(t, (2, 0, 1)), 0)
        o = np.squeeze(self.sess.run(self.outs, {self.inp.name: inp})[0]).T
        sc = o[:, 4]; m = sc >= 0.05
        o, sc = o[m], sc[m]
        cx, cy, bw, bh = o[:, 0], o[:, 1], o[:, 2], o[:, 3]
        boxes = np.stack([((cx - bw / 2 - dx) / s).astype(int), ((cy - bh / 2 - dy) / s).astype(int),
                          (bw / s).astype(int), (bh / s).astype(int)], 1).astype(np.int32)
        scores = sc.astype(np.float32)
        idx = cv2.dnn.NMSBoxes(boxes.tolist(), scores.tolist(), 0.05, 0.99)
        if idx is not None and len(idx) > 0:
            idx = np.array(idx).flatten(); boxes, scores = boxes[idx], scores[idx]
        return boxes, scores

def nms_count(boxes, scores):
    m = scores >= CONF
    if not m.any(): return 0, np.zeros((0, 4), np.int32)
    b, s = boxes[m], scores[m]
    idx = cv2.dnn.NMSBoxes(b.tolist(), s.tolist(), CONF, IOU)
    if idx is None or len(idx) == 0: return 0, np.zeros((0, 4), np.int32)
    idx = np.array(idx).flatten()
    return len(idx), b[idx]

def feats_of(boxes, scores, img):
    h, w = img.shape[:2]
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hch, sch, vch = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    grain = float((((hch < 35) | (hch > 160)) & (sch > 50) & (vch > 60)).sum())
    m3 = scores >= 0.3
    med = float(np.median(boxes[m3][:, 2].astype(float) * boxes[m3][:, 3])) if m3.any() else 1.0
    cover = np.zeros((h, w), np.uint16)
    for x, y, bw, bh in boxes[m3]:
        if bw > 0 and bh > 0: cover[max(0, y):y + bh, max(0, x):x + bw] += 1
    occ = float(cover[cover > 0].mean()) if cover.max() > 0 else 0.0
    return med, grain / max(med, 1.0), occ

def find_panicles(img, close_k=31, min_area=15000):
    """与 eopt_batch109.find_panicles 一致（按 -area 排序）"""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    H, S, V = cv2.split(hsv)
    g = ((S > 35) & (V > 70) & ((H < 50) | (H > 140))).astype(np.uint8) * 255
    g[(H >= 40) & (H <= 90)] = 0
    g = cv2.morphologyEx(g, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    closed = cv2.morphologyEx(g, cv2.MORPH_CLOSE, np.ones((close_k, close_k), np.uint8))
    green = ((H >= 35) & (H <= 95) & (S > 50) & (V > 50)).astype(np.uint8)
    n, labels, stats, cent = cv2.connectedComponentsWithStats(closed)
    pans = []
    for i in range(1, n):
        x, y, w, h, a = stats[i]
        if a >= min_area and max(w, h) / max(1, min(w, h)) < 25 and w > 60 and h > 60:
            if green[y:y + h, x:x + w].sum() / (w * h) < 0.005: continue
            pans.append({'x': int(x), 'y': int(y), 'w': int(w), 'h': int(h), 'area': int(a)})
    pans.sort(key=lambda p: -p['area'])
    return pans

def main():
    for sub in ['labeled', 'per_panicle_dl', 'per_panicle_numbered']:
        os.makedirs(os.path.join(OUT, sub), exist_ok=True)
    files = open(os.path.join(BASE, 'EOPT版', 'scan_files.txt'), encoding='utf-8').read().splitlines()
    model = YOLO(ONNX)
    rows, rows_raw, feats = [], [], []
    colors = [(0, 0, 255), (0, 200, 0), (255, 0, 0), (0, 255, 255), (255, 0, 255),
              (255, 200, 0), (200, 100, 255), (0, 200, 200), (200, 200, 0)]
    t0 = time.time()
    for i, f in enumerate(files, 1):
        rel = os.path.relpath(f, SRC)
        tag = rel.replace(os.sep, '__').rsplit('.', 1)[0]
        img = imread_u(f)
        if img is None:
            rows.append([rel]); rows_raw.append([rel]); continue
        pans = find_panicles(img)
        vis = img.copy()
        cc, rc = [], []
        for pi, p in enumerate(pans, 1):
            crop = img[max(0, p['y'] - 20):p['y'] + p['h'] + 20, max(0, p['x'] - 20):p['x'] + p['w'] + 20]
            boxes, scores = model.raw_boxes(crop)
            n_det, kept = nms_count(boxes, scores)
            med, rg, occ = feats_of(boxes, scores, crop)
            x = np.array([n_det, med, rg, occ], float)
            est = float(max(1.0, YM + ((x - MU) / SD) @ W))
            final = int(round(est if n_det >= T else float(n_det)))
            cc.append(final); rc.append(int(n_det))
            feats.append([rel, pi, n_det, round(med, 1), round(rg, 3), round(occ, 3), round(est, 2), final])
            col = colors[(pi - 1) % len(colors)]
            cv2.rectangle(vis, (p['x'], p['y']), (p['x'] + p['w'], p['y'] + p['h']), col, 4)
            cv2.putText(vis, 'P%d=%d' % (pi, final), (p['x'] + 6, p['y'] - 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.3, col, 4)
            # 单穗检测框图
            cvis = crop.copy()
            for bx, by, bw, bh in kept:
                cv2.rectangle(cvis, (bx, by), (bx + bw, by + bh), col, 2)
            imwrite_u(os.path.join(OUT, 'per_panicle_dl', f'{tag}_P{pi}.png'), cvis)
            # 单穗逐粒编号图
            nvis = crop.copy()
            order = np.lexsort((kept[:, 0], kept[:, 1])) if len(kept) else np.arange(0)
            for k, bi in enumerate(order, 1):
                bx, by, bw, bh = kept[bi]
                cv2.rectangle(nvis, (bx, by), (bx + bw, by + bh), (0, 220, 0), 1)
                lab = str(k)
                fs = 0.45 if len(kept) > 120 else 0.55
                cv2.putText(nvis, lab, (bx + 1, by + 14), cv2.FONT_HERSHEY_SIMPLEX, fs, (0, 0, 0), 3)
                cv2.putText(nvis, lab, (bx + 1, by + 14), cv2.FONT_HERSHEY_SIMPLEX, fs, (0, 255, 255), 1)
            head = f'P{pi}: 检测{len(kept)} / 校正后{final}'
            cv2.rectangle(nvis, (0, 0), (nvis.shape[1], 42), (0, 0, 0), -1)
            cv2.putText(nvis, head, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                        (0, 255, 0) if final == len(kept) else (0, 200, 255), 2)
            imwrite_u(os.path.join(OUT, 'per_panicle_numbered', f'{tag}_P{pi}.png'), nvis)
        rows.append([rel] + cc); rows_raw.append([rel] + rc)
        imwrite_u(os.path.join(OUT, 'labeled', tag + '.png'),
                  cv2.resize(vis, None, fx=0.3, fy=0.3))
        print(f'[{i}/{len(files)}] {rel} 穗{len(pans)} ({time.time()-t0:.0f}s)', flush=True)

    mp = max(len(r) - 1 for r in rows)
    for name, data in [('穗粒数_v2校正版.csv', rows), ('穗粒数_v2原始检测.csv', rows_raw)]:
        with open(os.path.join(OUT, name), 'w', newline='', encoding='utf-8-sig') as fo:
            wr = csv.writer(fo)
            wr.writerow(['图片'] + [f'穗{j+1}粒数' for j in range(mp)] + ['合计'])
            for r in data:
                tail = r[1:] + [None] * (mp - (len(r) - 1))
                wr.writerow([r[0]] + tail + [sum(v for v in r[1:] if isinstance(v, int))])
    with open(os.path.join(OUT, 'features_v2_all.csv'), 'w', newline='', encoding='utf-8-sig') as fo:
        wr = csv.writer(fo); wr.writerow(['图片', '穗', 'n_det', 'med_area', 'r_grain', 'occ', 'est', 'final'])
        wr.writerows(feats)
    print(f'DONE {time.time()-t0:.0f}s 穗数={len(feats)}')

if __name__ == '__main__':
    main()
