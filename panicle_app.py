# -*- coding: utf-8 -*-
"""水稻穗粒数计数小程序 — GUI 入口
功能: 选择图片文件夹 -> 自动分穗 + EOPT 深度学习逐穗计数 -> 输出 CSV + 标注图
打包: PyInstaller --onefile --windowed (内嵌 GrainNuber.onnx)
"""
import os, sys, csv, time, traceback
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext

# ---------- 路径处理(打包后资源在 _MEIPASS) ----------
def resource_path(rel):
    base = getattr(sys, '_MEIPASS', os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base, rel)

# ---------- 核心逻辑 ----------
import cv2
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eopt_count import YOLOv8

def imread_u(p):
    return cv2.imdecode(np.fromfile(p, dtype=np.uint8), cv2.IMREAD_COLOR)

def find_panicles(img, close_k=31, min_area=15000):
    """HSV 闭运算分穗 + 尺子排除(绿色穗轴占比 + 长宽比)"""
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
                panicles.append({'x': x, 'y': y, 'w': w, 'h': h, 'area': a})
    panicles.sort(key=lambda p: -p['area'])
    return panicles

def run_analysis(src_dir, out_dir, conf=0.5, log_cb=None, progress_cb=None):
    """主流程: 遍历图片 -> 分穗 -> 检测 -> 输出 CSV/标注图
    返回 (n_images, n_panicles, csv_path) 或抛异常
    """
    def log(m):
        if log_cb: log_cb(m)
    os.makedirs(out_dir, exist_ok=True)
    lab = os.path.join(out_dir, 'labeled')
    per = os.path.join(out_dir, 'per_panicle')
    os.makedirs(lab, exist_ok=True)
    os.makedirs(per, exist_ok=True)

    # 模型(优先 exe 内置, 其次同目录)
    onnx_path = resource_path('GrainNuber.onnx')
    if not os.path.exists(onnx_path):
        onnx_path = os.path.join(os.path.dirname(os.path.abspath(sys.argv[0])), 'GrainNuber.onnx')
    if not os.path.exists(onnx_path):
        raise FileNotFoundError('找不到模型权重 GrainNuber.onnx')
    model = YOLOv8(onnx_path, confidence_thres=conf, iou_thres=0.8)

    # 收集图片
    exts = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')
    files = []
    for root, dirs, fs in os.walk(src_dir):
        for f in sorted(fs):
            if f.lower().endswith(exts):
                files.append(os.path.join(root, f))
    files.sort()
    if not files:
        raise ValueError('输入文件夹中没有找到图片(jpg/png/bmp/tif)')
    log('找到图片 %d 张' % len(files))

    rows = []
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
        per_counts = []
        for pi, p in enumerate(panicles):
            x, y, w, h = p['x'], p['y'], p['w'], p['h']
            crop = img[max(0, y - 15):min(img.shape[0], y + h + 15),
                       max(0, x - 15):min(img.shape[1], x + w + 15)]
            dets = model.detect(crop)
            n = len(dets)
            per_counts.append(n)
            col = colors[pi % len(colors)]
            # 单穗标注图(检测框)
            cvis = crop.copy()
            for d in dets:
                bx, by, bw, bh = d['box']
                cv2.rectangle(cvis, (bx, by), (bx + bw, by + bh), col, 2)
            try:
                cv2.imencode('.png', cvis)[1].tofile(os.path.join(per, '%s_P%d.png' % (tag, pi + 1)))
            except Exception:
                pass
            # 整图标注
            cv2.rectangle(vis, (x, y), (x + w, y + h), col, 4)
            cv2.putText(vis, 'P%d=%d' % (pi + 1, n), (x + 6, y - 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.3, col, 4)
        rows.append([os.path.basename(f)] + per_counts)
        try:
            cv2.imencode('.png', cv2.resize(vis, None, fx=0.3, fy=0.3))[1].tofile(
                os.path.join(lab, tag + '.png'))
        except Exception:
            pass
        log('  [%d/%d] %s : %s 合计 %d' % (
            i, len(files), os.path.basename(f), per_counts, sum(per_counts)))
        if progress_cb:
            progress_cb(i, len(files))

    # 写 CSV
    max_p = max((len(r) - 1 for r in rows), default=0)
    csv_path = os.path.join(out_dir, '穗粒数统计.csv')
    with open(csv_path, 'w', newline='', encoding='utf-8-sig') as fo:
        w = csv.writer(fo)
        w.writerow(['图片'] + ['穗%d粒数' % (j + 1) for j in range(max_p)] + ['合计'])
        for r in rows:
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
        root.title('水稻穗粒数计数工具 v1.0 (EOPT 深度学习)')
        root.geometry('720x560')
        root.resizable(True, True)

        pad = {'padx': 8, 'pady': 4}
        frm = ttk.Frame(root, padding=10)
        frm.pack(fill='both', expand=True)

        # 输入
        row1 = ttk.Frame(frm)
        row1.pack(fill='x', **pad)
        ttk.Label(row1, text='图片文件夹:').pack(side='left')
        self.src_var = tk.StringVar()
        ttk.Entry(row1, textvariable=self.src_var).pack(side='left', fill='x', expand=True, padx=4)
        ttk.Button(row1, text='浏览...', command=self.pick_src).pack(side='left')

        # 输出
        row2 = ttk.Frame(frm)
        row2.pack(fill='x', **pad)
        ttk.Label(row2, text='结果保存到:').pack(side='left')
        self.out_var = tk.StringVar()
        ttk.Entry(row2, textvariable=self.out_var).pack(side='left', fill='x', expand=True, padx=4)
        ttk.Button(row2, text='浏览...', command=self.pick_out).pack(side='left')

        # 参数
        row3 = ttk.Frame(frm)
        row3.pack(fill='x', **pad)
        ttk.Label(row3, text='置信度阈值:').pack(side='left')
        self.conf_var = tk.DoubleVar(value=0.5)
        ttk.Spinbox(row3, from_=0.2, to=0.9, increment=0.05, textvariable=self.conf_var,
                    width=6).pack(side='left', padx=4)
        ttk.Label(row3, text='(越小检得越多, 默认0.5)').pack(side='left')

        # 运行
        row4 = ttk.Frame(frm)
        row4.pack(fill='x', **pad)
        self.run_btn = ttk.Button(row4, text='开始计数', command=self.start)
        self.run_btn.pack(side='left')
        self.progress = ttk.Progressbar(row4, mode='determinate')
        self.progress.pack(side='left', fill='x', expand=True, padx=8)

        # 日志
        lf = ttk.LabelFrame(frm, text='运行日志')
        lf.pack(fill='both', expand=True, **pad)
        self.log_txt = scrolledtext.ScrolledText(lf, height=16, state='disabled', font=('Consolas', 9))
        self.log_txt.pack(fill='both', expand=True, padx=4, pady=4)

        # 底部
        row5 = ttk.Frame(frm)
        row5.pack(fill='x', **pad)
        ttk.Button(row5, text='打开结果文件夹', command=self.open_out).pack(side='left')
        ttk.Label(row5, text='支持 jpg/png/bmp/tif; 自动分穗逐穗计数; 结果=CSV+标注图').pack(side='right')

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
            n_img, n_pan, csv_path = run_analysis(src, out, conf, log_cb, progress_cb)
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
    # CLI 模式: panicle_app(.exe) --cli <输入目录> <输出目录> [conf]
    # 日志写 输出目录/cli_log.txt (windowed exe 无 stdout, 一律落盘)
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