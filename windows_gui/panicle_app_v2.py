# -*- coding: utf-8 -*-
"""水稻穗粒数计数工具 v2 (遮挡校正版) — GUI 入口
功能: 选择图片文件夹 -> 自动分穗 -> EOPT 检测(iou=0.5) + 逐穗遮挡校正 -> CSV + 标注图 + 逐粒编号图
打包: PyInstaller --onefile --windowed (内嵌 GrainNuber.onnx + calibration_model.json)
版本: v2-2026-09-26  标定: 32 穗人工真值, 嵌套 LOO MAPE 10.3% (旧版 19.6%)
"""
import os, sys, csv, json, time, traceback
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext

import cv2
import numpy as np

# ---------- 路径处理(打包后资源在 _MEIPASS) ----------
def resource_path(rel):
    base = getattr(sys, '_MEIPASS', os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base, rel)

def imread_u(p):
    return cv2.imdecode(np.fromfile(p, dtype=np.uint8), cv2.IMREAD_COLOR)

def imwrite_u(p, img):
    try:
        ok, buf = cv2.imencode(os.path.splitext(p)[1], img)
        if ok:
            buf.tofile(p)
    except Exception:
        pass

# ---------- 模型 ----------
import onnxruntime as ort

class YOLOv8:
    """YOLOv8 ONNX 推理 (letterbox 保比例, 输出逆映射回原图)"""
    def __init__(self, onnx_model):
        self.session = ort.InferenceSession(onnx_model, providers=['CPUExecutionProvider'])
        self.model_inputs = self.session.get_inputs()
        self.model_outputs = self.session.get_outputs()
        shp = self.model_inputs[0].shape
        self.input_width = shp[3] if isinstance(shp[3], int) else 1280
        self.input_height = shp[2] if isinstance(shp[2], int) else 1280

    def raw_boxes(self, img_bgr, lo_conf=0.05):
        """返回 conf>=lo_conf 的框 (int 截断 + iou=0.99 轻量 NMS, 与标定口径一致)"""
        h, w = img_bgr.shape[:2]
        s = min(self.input_width / w, self.input_height / h)
        nw, nh = int(round(w * s)), int(round(h * s))
        r = cv2.resize(img_bgr, (nw, nh))
        canvas = np.full((self.input_height, self.input_width, 3), 114, np.uint8)
        dx, dy = (self.input_width - nw) // 2, (self.input_height - nh) // 2
        canvas[dy:dy + nh, dx:dx + nw] = r
        t = canvas[:, :, ::-1].astype(np.float32) / 255.0
        inp = np.expand_dims(np.transpose(t, (2, 0, 1)), 0)
        out = self.session.run([o.name for o in self.model_outputs],
                               {self.model_inputs[0].name: inp})[0]
        o = np.squeeze(out).T
        sc = o[:, 4]
        m = sc >= lo_conf
        o, sc = o[m], sc[m]
        cx, cy, bw, bh = o[:, 0], o[:, 1], o[:, 2], o[:, 3]
        boxes = np.stack([((cx - bw / 2 - dx) / s).astype(int), ((cy - bh / 2 - dy) / s).astype(int),
                          (bw / s).astype(int), (bh / s).astype(int)], 1).astype(np.int32)
        scores = sc.astype(np.float32)
        idx = cv2.dnn.NMSBoxes(boxes.tolist(), scores.tolist(), lo_conf, 0.99)
        if idx is not None and len(idx) > 0:
            idx = np.array(idx).flatten()
            boxes, scores = boxes[idx], scores[idx]
        return boxes, scores

# ---------- 分穗 ----------
def find_panicles(img, close_k=31, min_area=15000):
    """HSV 闭运算分穗 + 尺子排除(绿色穗轴占比 + 长宽比); 按面积降序"""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    H, S, V = cv2.split(hsv)
    g = ((S > 35) & (V > 70) & ((H < 50) | (H > 140))).astype(np.uint8) * 255
    g[(H >= 40) & (H <= 90)] = 0
    g = cv2.morphologyEx(g, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    closed = cv2.morphologyEx(g, cv2.MORPH_CLOSE, np.ones((close_k, close_k), np.uint8))
    green = ((H >= 35) & (H <= 95) & (S > 50) & (V > 50)).astype(np.uint8)
    n, labels, stats, cent = cv2.connectedComponentsWithStats(closed)
    panicles = []
    for i in range(1, n):
        x, y, w, h, a = stats[i]
        if a >= min_area:
            ratio = max(w, h) / max(1, min(w, h))
            if ratio < 25 and w > 60 and h > 60:
                g_px = green[y:y + h, x:x + w].sum()
                if g_px / (w * h) < 0.005:
                    continue
                panicles.append({'x': int(x), 'y': int(y), 'w': int(w), 'h': int(h), 'area': int(a)})
    panicles.sort(key=lambda p: -p['area'])
    return panicles

# ---------- 主流程 ----------
def run_analysis(src_dir, out_dir, conf=0.5, cal_path=None, log_cb=None, progress_cb=None):
    """遍历图片 -> 分穗 -> 检测 -> 特征校正 -> 输出 CSV/标注图/编号图
    返回 (n_images, n_panicles, csv_path)
    """
    def log(m):
        if log_cb: log_cb(m)
    os.makedirs(out_dir, exist_ok=True)
    lab = os.path.join(out_dir, 'labeled')
    per = os.path.join(out_dir, 'per_panicle')
    num = os.path.join(out_dir, 'per_panicle_numbered')
    for d in (lab, per, num):
        os.makedirs(d, exist_ok=True)

    # 标定模型
    cal_file = cal_path or resource_path('calibration_model.json')
    if not os.path.exists(cal_file):
        cal_file = os.path.join(os.path.dirname(os.path.abspath(sys.argv[0])), 'calibration_model.json')
    CAL = None
    if os.path.exists(cal_file):
        CAL = json.load(open(cal_file, encoding='utf-8'))
        log('已加载标定模型: %s (版本 %s, MAPE %.1f%%)' % (
            os.path.basename(cal_file), CAL.get('version', '?'),
            CAL.get('calibration', {}).get('nested_loo_mape_pct', 0)))
    else:
        log('⚠️ 未找到 calibration_model.json → 只输出原始检测数(不校正)')

    MU = np.array(CAL['mu']) if CAL else None
    SD = np.array(CAL['sd']) if CAL else None
    W = np.array(CAL['w']) if CAL else None
    YM = CAL['y_mean'] if CAL else 0.0
    T = CAL.get('switch_threshold', 0) if CAL else 0

    onnx_path = resource_path('GrainNuber.onnx')
    if not os.path.exists(onnx_path):
        onnx_path = os.path.join(os.path.dirname(os.path.abspath(sys.argv[0])), 'GrainNuber.onnx')
    if not os.path.exists(onnx_path):
        raise FileNotFoundError('找不到模型权重 GrainNuber.onnx')
    model = YOLOv8(onnx_path)

    exts = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')
    files = []
    for root, dirs, fs in os.walk(src_dir):
        for f in sorted(fs):
            if f.lower().endswith(exts):
                files.append(os.path.join(root, f))
    files.sort()
    if not files:
        raise ValueError('输入文件夹中没有找到图片(jpg/png/bmp/tif)')
    log('找到图片 %d 张, 置信度阈值 %.2f, IoU 0.5' % (len(files), conf))

    rows, rows_raw = [], []
    t0 = time.time()
    colors = [(0, 0, 255), (0, 200, 0), (255, 0, 0), (0, 255, 255), (255, 0, 255),
              (255, 200, 0), (200, 100, 255), (0, 200, 200), (200, 200, 0)]

    for i, f in enumerate(files, 1):
        tag = os.path.splitext(os.path.basename(f))[0]
        img = imread_u(f)
        if img is None:
            log('  [%d/%d] 跳过(无法读取): %s' % (i, len(files), os.path.basename(f)))
            continue
        panicles = find_panicles(img)
        if not panicles:
            log('  [%d/%d] 无穗: %s' % (i, len(files), os.path.basename(f)))
            continue
        vis = img.copy()
        cc, rc = [], []
        for pi, p in enumerate(panicles):
            x, y, w, h = p['x'], p['y'], p['w'], p['h']
            crop = img[max(0, y - 20):min(img.shape[0], y + h + 20),
                       max(0, x - 20):min(img.shape[1], x + w + 20)]
            boxes, scores = model.raw_boxes(crop)
            # 计数框 = NMS(conf, 0.5)
            mm = scores >= conf
            if mm.any():
                idx = cv2.dnn.NMSBoxes(boxes[mm].tolist(), scores[mm].tolist(), conf, 0.5)
                idx = np.array(idx).flatten() if idx is not None and len(idx) > 0 else np.zeros(0, int)
                kept = boxes[mm][idx] if len(idx) else np.zeros((0, 4), np.int32)
            else:
                kept = np.zeros((0, 4), np.int32)
            n_det = len(kept)
            # 特征
            ch, cw = crop.shape[:2]
            hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
            hch, sch, vch = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
            grain_area = float((((hch < 35) | (hch > 160)) & (sch > 50) & (vch > 60)).sum())
            m3 = scores >= 0.3
            if m3.any():
                b3 = boxes[m3]
                med_area = float(np.median(b3[:, 2].astype(np.float64) * b3[:, 3]))
            else:
                med_area = 1.0
            cover = np.zeros((ch, cw), np.uint16)
            for bx, by, bw, bh in boxes[m3]:
                if bw > 0 and bh > 0:
                    cover[max(0, by):by + bh, max(0, bx):bx + bw] += 1
            occ = float(cover[cover > 0].mean()) if cover.max() > 0 else 0.0
            # 校正
            if CAL is not None:
                xx = np.array([n_det, med_area, grain_area / max(med_area, 1.0), occ], float)
                est = float(max(1.0, YM + ((xx - MU) / SD) @ W))
                final = int(round(est if n_det >= T else float(n_det)))
            else:
                final = n_det
            cc.append(final); rc.append(n_det)
            col = colors[pi % len(colors)]
            # 单穗检测框图
            cvis = crop.copy()
            for bx, by, bw, bh in kept:
                cv2.rectangle(cvis, (bx, by), (bx + bw, by + bh), col, 2)
            imwrite_u(os.path.join(per, '%s_P%d.png' % (tag, pi + 1)), cvis)
            # 逐粒编号图 (自上而下、自左而右)
            nvis = crop.copy()
            order = np.lexsort((kept[:, 0], kept[:, 1])) if len(kept) else np.arange(0)
            for k, bi in enumerate(order, 1):
                bx, by, bw, bh = kept[bi]
                cv2.rectangle(nvis, (bx, by), (bx + bw, by + bh), (0, 220, 0), 1)
                lab_s = str(k)
                fs = 0.45 if len(kept) > 120 else 0.55
                cv2.putText(nvis, lab_s, (bx + 1, by + 14), cv2.FONT_HERSHEY_SIMPLEX, fs, (0, 0, 0), 3)
                cv2.putText(nvis, lab_s, (bx + 1, by + 14), cv2.FONT_HERSHEY_SIMPLEX, fs, (0, 255, 255), 1)
            head = 'P%d: 检测%d / 校正后%d' % (pi + 1, n_det, final)
            cv2.rectangle(nvis, (0, 0), (nvis.shape[1], 42), (0, 0, 0), -1)
            cv2.putText(nvis, head, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                        (0, 255, 0) if final == n_det else (0, 200, 255), 2)
            imwrite_u(os.path.join(num, '%s_P%d.png' % (tag, pi + 1)), nvis)
            # 整图标注
            cv2.rectangle(vis, (x, y), (x + w, y + h), col, 4)
            cv2.putText(vis, 'P%d=%d' % (pi + 1, final), (x + 6, y - 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.3, col, 4)
        rows.append([os.path.basename(f)] + cc)
        rows_raw.append([os.path.basename(f)] + rc)
        imwrite_u(os.path.join(lab, tag + '.png'), cv2.resize(vis, None, fx=0.3, fy=0.3))
        log('  [%d/%d] %s : 校正%s 原始%s 合计 %d' % (
            i, len(files), os.path.basename(f), cc, rc, sum(cc)))
        if progress_cb:
            progress_cb(i, len(files))

    max_p = max((len(r) - 1 for r in rows), default=0)
    csv_path = os.path.join(out_dir, '穗粒数统计.csv')
    for path, data, title in [(csv_path, rows, '穗粒数统计.csv'),
                              (os.path.join(out_dir, '穗粒数统计_原始检测.csv'), rows_raw, '')]:
        with open(path, 'w', newline='', encoding='utf-8-sig') as fo:
            w = csv.writer(fo)
            w.writerow(['图片'] + ['穗%d粒数' % (j + 1) for j in range(max_p)] + ['合计'])
            for r in data:
                tail = r[1:] + [None] * (max_p - (len(r) - 1))
                w.writerow([r[0]] + tail + [sum(r[1:])])
    n_pan = sum(len(r) - 1 for r in rows)
    log('完成: %d 张图, %d 株穗, 用时 %.0f 秒' % (len(rows), n_pan, time.time() - t0))
    log('CSV: %s' % csv_path)
    return len(rows), n_pan, csv_path

# ---------- GUI ----------
class App:
    def __init__(self, root):
        self.root = root
        root.title('水稻穗粒数计数工具 v2 (EOPT + 遮挡校正)')
        root.geometry('760x580')
        root.resizable(True, True)

        pad = {'padx': 8, 'pady': 4}
        frm = ttk.Frame(root, padding=10)
        frm.pack(fill='both', expand=True)

        row1 = ttk.Frame(frm)
        row1.pack(fill='x', **pad)
        ttk.Label(row1, text='图片文件夹:').pack(side='left')
        self.src_var = tk.StringVar()
        ttk.Entry(row1, textvariable=self.src_var).pack(side='left', fill='x', expand=True, padx=4)
        ttk.Button(row1, text='浏览...', command=self.pick_src).pack(side='left')

        row2 = ttk.Frame(frm)
        row2.pack(fill='x', **pad)
        ttk.Label(row2, text='结果保存到:').pack(side='left')
        self.out_var = tk.StringVar()
        ttk.Entry(row2, textvariable=self.out_var).pack(side='left', fill='x', expand=True, padx=4)
        ttk.Button(row2, text='浏览...', command=self.pick_out).pack(side='left')

        row3 = ttk.Frame(frm)
        row3.pack(fill='x', **pad)
        ttk.Label(row3, text='置信度阈值:').pack(side='left')
        self.conf_var = tk.DoubleVar(value=0.5)
        ttk.Spinbox(row3, from_=0.2, to=0.9, increment=0.05, textvariable=self.conf_var,
                    width=6).pack(side='left', padx=4)
        ttk.Label(row3, text='(默认 0.5，已标定；建议不要随意改)').pack(side='left')

        row4 = ttk.Frame(frm)
        row4.pack(fill='x', **pad)
        self.run_btn = ttk.Button(row4, text='开始计数', command=self.start)
        self.run_btn.pack(side='left')
        self.progress = ttk.Progressbar(row4, mode='determinate')
        self.progress.pack(side='left', fill='x', expand=True, padx=8)

        lf = ttk.LabelFrame(frm, text='运行日志')
        lf.pack(fill='both', expand=True, **pad)
        self.log_txt = scrolledtext.ScrolledText(lf, height=16, state='disabled', font=('Consolas', 9))
        self.log_txt.pack(fill='both', expand=True, padx=4, pady=4)

        row5 = ttk.Frame(frm)
        row5.pack(fill='x', **pad)
        ttk.Button(row5, text='打开结果文件夹', command=self.open_out).pack(side='left')
        ttk.Label(row5, text='输出: 穗粒数统计.csv + 整图标注 + 单穗框图 + 逐粒编号图').pack(side='right')

    def pick_src(self):
        d = filedialog.askdirectory(title='选择穗扫描图片文件夹')
        if d:
            self.src_var.set(d)
            if not self.out_var.get():
                self.out_var.set(os.path.join(d, '计数结果'))

    def pick_out(self):
        d = filedialog.askdirectory(title='选择结果保存文件夹')
        if d:
            self.out_var.set(d)

    def open_out(self):
        d = self.out_var.get().strip()
        if d and os.path.isdir(d):
            os.startfile(d)

    def log(self, m):
        self.log_txt.configure(state='normal')
        self.log_txt.insert('end', m + '\n')
        self.log_txt.see('end')
        self.log_txt.configure(state='disabled')
        self.root.update_idletasks()

    def start(self):
        src = self.src_var.get().strip()
        out = self.out_var.get().strip()
        if not src or not os.path.isdir(src):
            messagebox.showerror('错误', '请选择有效的图片文件夹')
            return
        if not out:
            out = os.path.join(src, '计数结果')
            self.out_var.set(out)
        conf = float(self.conf_var.get())
        self.run_btn.configure(state='disabled')
        self.progress['value'] = 0
        self.log_txt.configure(state='normal')
        self.log_txt.delete('1.0', 'end')
        self.log_txt.configure(state='disabled')
        self.log('开始计数... 输入: %s' % src)
        t = threading.Thread(target=self._work, args=(src, out, conf), daemon=True)
        t.start()

    def _work(self, src, out, conf):
        try:
            def log_cb(m):
                self.root.after(0, lambda: self.log(m))
            def progress_cb(i, n):
                self.root.after(0, lambda: (self.progress.configure(maximum=n, value=i)))
            n_img, n_pan, csv_path = run_analysis(src, out, conf, log_cb=log_cb, progress_cb=progress_cb)
            self.root.after(0, lambda: self._done(n_img, n_pan, csv_path))
        except Exception as e:
            tb = traceback.format_exc()
            self.root.after(0, lambda: self._fail(str(e), tb))

    def _done(self, n_img, n_pan, csv_path):
        self.run_btn.configure(state='normal')
        messagebox.showinfo('完成', '处理完成!\n%d 张图, %d 株穗\n结果已保存:\n%s' % (n_img, n_pan, csv_path))

    def _fail(self, msg, tb):
        self.run_btn.configure(state='normal')
        self.log('错误: %s\n%s' % (msg, tb))
        messagebox.showerror('出错', msg)

def main():
    # CLI: 程序 --cli <输入目录> <输出目录> [conf]
    if len(sys.argv) >= 4 and sys.argv[1] == '--cli':
        src, out = sys.argv[2], sys.argv[3]
        conf = float(sys.argv[4]) if len(sys.argv) >= 5 else 0.5
        os.makedirs(out, exist_ok=True)
        logf = open(os.path.join(out, 'cli_log.txt'), 'w', encoding='utf-8')
        try:
            def log(m):
                logf.write(m + '\n')
                logf.flush()
            n_img, n_pan, csv_path = run_analysis(src, out, conf, log_cb=log)
            log('CLI_DONE %d %d %s' % (n_img, n_pan, csv_path))
        except Exception as e:
            log('CLI_ERROR %s' % traceback.format_exc())
        finally:
            logf.close()
        return
    root = tk.Tk()
    App(root)
    root.mainloop()

if __name__ == '__main__':
    main()
