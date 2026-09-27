package com.example.ricecounter;

import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.graphics.Typeface;

import org.json.JSONObject;

import java.io.ByteArrayOutputStream;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;

import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtSession;

/**
 * 穗粒数计数核心（与桌面标定版口径一致）
 * 流程: 分穗(HSV+形态学+连通域) -> 每穗 crop(20px) -> YOLOv8 检测(iou=0.5)
 *      -> 4 特征 -> ridge 遮挡校正(开关 T=50) -> 逐粒编号渲染
 */
public class PanicleCounter {

    public static final int INPUT = 1280;
    public static final float CONF = 0.5f;      // 计数阈值(标定)
    public static final float IOU = 0.5f;       // NMS 阈值(标定)
    public static final float LO = 0.05f;       // 原始框下限
    public static final float F3 = 0.3f;        // 特征用框下限

    // ---------------- 标定模型 ----------------
    public static class Calib {
        double[] mu = new double[4], sd = new double[4], w = new double[4];
        double ymean = 72.5625, T = 50;
        double medLo = 1197, medHi = 4455;   // 标定尺度域（med_area 必须落在此区间才启用校正）
        boolean ok = false;
    }

    public static Calib loadCalib(AssetManager am) {
        Calib c = new Calib();
        try {
            String s = readAssetText(am, "calibration_model.json");
            JSONObject o = new JSONObject(s);
            c.mu = arr(o.getJSONArray("mu"));
            c.sd = arr(o.getJSONArray("sd"));
            c.w = arr(o.getJSONArray("w"));
            c.ymean = o.getDouble("y_mean");
            c.T = o.optDouble("switch_threshold", 50);
            if (o.has("med_area_domain")) {
                org.json.JSONArray d = o.getJSONArray("med_area_domain");
                c.medLo = d.getDouble(0);
                c.medHi = d.getDouble(1);
            }
            c.ok = true;
        } catch (Exception e) {
            c.ok = false;
        }
        return c;
    }

    private static double[] arr(org.json.JSONArray a) throws Exception {
        double[] r = new double[a.length()];
        for (int i = 0; i < a.length(); i++) r[i] = a.getDouble(i);
        return r;
    }

    static String readAssetText(AssetManager am, String name) throws Exception {
        InputStream is = am.open(name);
        ByteArrayOutputStream bo = new ByteArrayOutputStream();
        byte[] b = new byte[8192];
        int n;
        while ((n = is.read(b)) != -1) bo.write(b, 0, n);
        is.close();
        return bo.toString("UTF-8");
    }

    static byte[] readAssetBytes(AssetManager am, String name) throws Exception {
        InputStream is = am.open(name);
        ByteArrayOutputStream bo = new ByteArrayOutputStream();
        byte[] b = new byte[1 << 16];
        int n;
        while ((n = is.read(b)) != -1) bo.write(b, 0, n);
        is.close();
        return bo.toByteArray();
    }

    // ---------------- 检测 ----------------
    public static class Box {
        public float x, y, w, h, score;
        Box(float x, float y, float w, float h, float s) { this.x = x; this.y = y; this.w = w; this.h = h; this.score = s; }
        float area() { return w * h; }
    }

    /** letterbox 到 INPUT x INPUT，返回缩放/填充参数 */
    static float[] letterboxParams(int w, int h) {
        float s = Math.min((float) INPUT / w, (float) INPUT / h);
        int nw = Math.round(w * s), nh = Math.round(h * s);
        int dx = (INPUT - nw) / 2, dy = (INPUT - nh) / 2;
        return new float[]{s, dx, dy};
    }

    // 复用缓冲区（避免每次推理重新分配 ~20MB）
    private static java.nio.FloatBuffer BUF = null;
    private static int[] PX = null;
    private static float[] PLANE = null;

    /** 运行一次推理，返回原始框（conf>=0.05，int 截断 + iou=0.99 轻量 NMS） */
    public static List<Box> rawBoxes(OrtSession sess, OrtEnvironment env, String inputName,
                                     Bitmap crop) throws Exception {
        int w = crop.getWidth(), h = crop.getHeight();
        float[] lb = letterboxParams(w, h);
        float s = lb[0];
        int dx = (int) lb[1], dy = (int) lb[2];
        int nw = Math.round(w * s), nh = Math.round(h * s);

        Bitmap rs = Bitmap.createScaledBitmap(crop, nw, nh, true);
        int need = nw * nh;
        if (PX == null || PX.length < need) PX = new int[need];
        rs.getPixels(PX, 0, nw, 0, 0, nw, nh);
        if (rs != crop) rs.recycle();
        final int PLANE_SZ = INPUT * INPUT;
        if (BUF == null) BUF = java.nio.ByteBuffer.allocateDirect(3 * PLANE_SZ * 4)
                .order(java.nio.ByteOrder.nativeOrder()).asFloatBuffer();
        if (PLANE == null) PLANE = new float[PLANE_SZ];
        final float BG = 114f / 255f;
        java.util.Arrays.fill(PLANE, BG);
        // 先清三个平面为背景色
        BUF.clear();
        BUF.put(PLANE); BUF.put(PLANE); BUF.put(PLANE);
        // 逐像素写入 letterbox 区域（直接写浮点缓冲）
        for (int yy = 0; yy < nh; yy++) {
            int base = (yy + dy) * INPUT + dx;
            int srcRow = yy * nw;
            for (int xx = 0; xx < nw; xx++) {
                int p = PX[srcRow + xx];
                int r = (p >> 16) & 0xFF, g = (p >> 8) & 0xFF, b = p & 0xFF;
                BUF.put(base + xx, r / 255f);
                BUF.put(PLANE_SZ + base + xx, g / 255f);
                BUF.put(2 * PLANE_SZ + base + xx, b / 255f);
            }
        }

        long[] shape = new long[]{1, 3, INPUT, INPUT};
        BUF.flip();   // 相对写入后翻转游标，让 ORT 能读到全部 4915200 个元素
        OnnxTensor t = OnnxTensor.createTensor(env, BUF, shape);
        OrtSession.Result res = sess.run(Collections.singletonMap(inputName, t));
        float[][][] out = (float[][][]) res.get(0).getValue();   // [1,5,N]
        int C = out[0].length, N = out[0][0].length;
        List<Box> boxes = new ArrayList<>();
        for (int i = 0; i < N; i++) {
            float sc = out[0][4][i];
            if (sc < LO) continue;
            float cx = out[0][0][i], cy = out[0][1][i], bw = out[0][2][i], bh = out[0][3][i];
            int left = (int) ((cx - bw / 2 - dx) / s);
            int top = (int) ((cy - bh / 2 - dy) / s);
            int wi = (int) (bw / s), hi = (int) (bh / s);
            boxes.add(new Box(left, top, wi, hi, sc));
        }
        t.close(); res.close();
        return nms(boxes, 0.99f);
    }

    /** IoU-NMS */
    public static List<Box> nms(List<Box> boxes, float thr) {
        List<Box> sorted = new ArrayList<>(boxes);
        Collections.sort(sorted, new Comparator<Box>() {
            public int compare(Box a, Box b) { return Float.compare(b.score, a.score); }
        });
        List<Box> keep = new ArrayList<>();
        boolean[] dead = new boolean[sorted.size()];
        for (int i = 0; i < sorted.size(); i++) {
            if (dead[i]) continue;
            Box A = sorted.get(i);
            keep.add(A);
            for (int j = i + 1; j < sorted.size(); j++) {
                if (dead[j]) continue;
                if (iou(A, sorted.get(j)) > thr) dead[j] = true;
            }
        }
        return keep;
    }

    static float iou(Box a, Box b) {
        float x1 = Math.max(a.x, b.x), y1 = Math.max(a.y, b.y);
        float x2 = Math.min(a.x + a.w, b.x + b.w), y2 = Math.min(a.y + a.h, b.y + b.h);
        float iw = x2 - x1, ih = y2 - y1;
        if (iw <= 0 || ih <= 0) return 0f;
        float inter = iw * ih;
        float ua = a.area() + b.area() - inter;
        return ua <= 0 ? 0 : inter / ua;
    }

    public static List<Box> keepByScore(List<Box> boxes, float thr) {
        List<Box> r = new ArrayList<>();
        for (Box b : boxes) if (b.score >= thr) r.add(b);
        return r;
    }

    // ---------------- 特征 ----------------
    public static double medianArea(List<Box> boxes) {
        if (boxes.isEmpty()) return 1.0;
        double[] a = new double[boxes.size()];
        for (int i = 0; i < a.length; i++) a[i] = boxes.get(i).area();
        java.util.Arrays.sort(a);
        int n = a.length;
        return n % 2 == 1 ? a[n / 2] : (a[n / 2 - 1] + a[n / 2]) / 2.0;
    }

    /** 谷粒像素面积（HSV: H<35 或 H>160, S>50, V>60；H 为 0..180） */
    public static double grainArea(Bitmap crop) {
        int w = crop.getWidth(), h = crop.getHeight();
        int[] px = new int[w * h];
        crop.getPixels(px, 0, w, 0, 0, w, h);
        long cnt = 0;
        for (int p : px) {
            int r = (p >> 16) & 0xFF, g = (p >> 8) & 0xFF, b = p & 0xFF;
            int max = Math.max(r, Math.max(g, b)), min = Math.min(r, Math.min(g, b));
            int v = max;
            int sat = max == 0 ? 0 : (int) (255.0 * (max - min) / max);
            int hue = 0;
            if (max != min) {
                float d = max - min;
                float hh;
                if (max == r) hh = 60f * ((g - b) / d);
                else if (max == g) hh = 60f * (2 + (b - r) / d);
                else hh = 60f * (4 + (r - g) / d);
                if (hh < 0) hh += 360f;
                hue = (int) (hh / 2f);           // 0..180, 与 OpenCV 一致
            }
            if ((hue < 35 || hue > 160) && sat > 50 && v > 60) cnt++;
        }
        return cnt;
    }

    /** 框覆盖重数均值（score>=0.3 的框，不 NMS） */
    public static double occMean(List<Box> boxes, int w, int h) {
        if (boxes.isEmpty() || w <= 0 || h <= 0) return 0.0;
        short[] cover = new short[w * h];
        for (Box b : boxes) {
            int x0 = Math.max(0, (int) b.x), y0 = Math.max(0, (int) b.y);
            int x1 = Math.min(w, (int) (b.x + b.w)), y1 = Math.min(h, (int) (b.y + b.h));
            for (int y = y0; y < y1; y++) {
                int row = y * w;
                for (int x = x0; x < x1; x++) cover[row + x]++;
            }
        }
        long sum = 0, n = 0;
        for (short c : cover) if (c > 0) { sum += c; n++; }
        return n == 0 ? 0.0 : (double) sum / n;
    }

    /** 校正：返回 {检测数, 校正后}；不满足域条件时返回原始检测数 */
    public static int[] correct(Calib cal, int nDet, double medArea, double rGrain, double occ) {
        if (!cal.ok) return new int[]{nDet, nDet};
        // 尺度域保护：手机的成像尺度与扫描标定不同，外推会失真 → 域外不校正
        if (medArea < cal.medLo || medArea > cal.medHi) return new int[]{nDet, nDet};
        double[] x = {nDet, medArea, rGrain, occ};
        double est = cal.ymean;
        for (int i = 0; i < 4; i++) {
            double z = (x[i] - cal.mu[i]) / cal.sd[i];
            if (z > 2.5) z = 2.5;
            if (z < -2.5) z = -2.5;
            est += z * cal.w[i];
        }
        est = Math.max(1.0, est);
        // 校正幅度限制：最多下调 15%，最多上调 80%
        double lo = nDet * 0.85, hi = nDet * 1.8;
        if (est < lo) est = lo;
        if (est > hi) est = hi;
        int fin = (nDet >= cal.T) ? (int) Math.round(est) : nDet;
        return new int[]{nDet, fin};
    }

    // ---------------- 分穗 ----------------
    /** 分穗：先用常规阈值；若只分出≤1个区域（常见于拍了屏幕/杂背景）则用更严阈值重试 */
    public static List<int[]> findPanicles(Bitmap bmp) {
        List<int[]> best = pass(bmp, 35, 70);
        if (best.size() <= 1) {
            List<int[]> alt = pass(bmp, 70, 95);
            if (alt.size() > best.size()) best = alt;
            if (best.size() <= 1) {
                List<int[]> alt2 = pass(bmp, 90, 110);
                if (alt2.size() > best.size()) best = alt2;
            }
        }
        return best;
    }

    /** 单次分穗（sMin/vMin 为饱和度/亮度阈值） */
    static List<int[]> pass(Bitmap bmp, int sMin, int vMin) {
        int W = bmp.getWidth(), H = bmp.getHeight();
        float sc = 1f;
        int mw = W, mh = H;
        if (Math.max(W, H) > 1400) {
            sc = 1400f / Math.max(W, H);
            mw = Math.max(1, Math.round(W * sc));
            mh = Math.max(1, Math.round(H * sc));
        }
        Bitmap small = (sc < 1f) ? Bitmap.createScaledBitmap(bmp, mw, mh, true) : bmp;
        int[] px = new int[mw * mh];
        small.getPixels(px, 0, mw, 0, 0, mw, mh);

        byte[] grain = new byte[mw * mh];   // 谷粒候选(非绿)
        byte[] green = new byte[mw * mh];   // 绿色(穗轴/叶)
        for (int i = 0; i < px.length; i++) {
            int p = px[i];
            int r = (p >> 16) & 0xFF, g = (p >> 8) & 0xFF, b = p & 0xFF;
            int max = Math.max(r, Math.max(g, b)), min = Math.min(r, Math.min(g, b));
            int v = max;
            int sat = max == 0 ? 0 : (int) (255.0 * (max - min) / max);
            int hue = 0;
            if (max != min) {
                float d = max - min, hh;
                if (max == r) hh = 60f * ((g - b) / d);
                else if (max == g) hh = 60f * (2 + (b - r) / d);
                else hh = 60f * (4 + (r - g) / d);
                if (hh < 0) hh += 360f;
                hue = (int) (hh / 2f);
            }
            grain[i] = (byte) ((sat > sMin && v > vMin && (hue < 50 || hue > 140)) ? 1 : 0);
            if (hue >= 40 && hue <= 90) grain[i] = 0;
            green[i] = (byte) ((hue >= 35 && hue <= 95 && sat > 50 && v > 50) ? 1 : 0);
        }
        // open(3x3) -> close(31x31)，用可分离方框形态学近似
        byte[] m = grain;
        m = morphMin(m, mw, mh, 1);
        m = morphMax(m, mw, mh, 1);
        m = morphMax(m, mw, mh, 15);
        m = morphMin(m, mw, mh, 15);

        // 连通域
        int[] lab = new int[mw * mh];
        int nlab = 0;
        int[] stack = new int[mw * mh];
        List<int[]> comps = new ArrayList<>();   // {x0,y0,x1,y1,area}
        for (int i = 0; i < m.length; i++) {
            if (m[i] != 1 || lab[i] != 0) continue;
            nlab++;
            int sp = 0;
            stack[sp++] = i;
            lab[i] = nlab;
            int x0 = mw, y0 = mh, x1 = 0, y1 = 0, area = 0;
            while (sp > 0) {
                int cur = stack[--sp];
                int cy = cur / mw, cx = cur - cy * mw;
                area++;
                if (cx < x0) x0 = cx;
                if (cy < y0) y0 = cy;
                if (cx > x1) x1 = cx;
                if (cy > y1) y1 = cy;
                for (int dy = -1; dy <= 1; dy++) {
                    for (int dx = -1; dx <= 1; dx++) {
                        int nx = cx + dx, ny = cy + dy;
                        if (nx < 0 || ny < 0 || nx >= mw || ny >= mh) continue;
                        int ni = ny * mw + nx;
                        if (m[ni] == 1 && lab[ni] == 0) { lab[ni] = nlab; stack[sp++] = ni; }
                    }
                }
            }
            comps.add(new int[]{x0, y0, x1, y1, area});
        }
        // 过滤（面积阈值按降采样平方缩放）
        int minArea = (int) (15000 * sc * sc);
        int minSide = Math.max(10, (int) (60 * sc));
        List<int[]> out = new ArrayList<>();
        for (int[] c : comps) {
            int w = c[2] - c[0] + 1, h = c[3] - c[1] + 1, a = c[4];
            if (a < minArea) continue;
            if (Math.max(w, h) / (float) Math.max(1, Math.min(w, h)) >= 25) continue;
            if (w <= minSide || h <= minSide) continue;
            int gc = 0;
            for (int y = c[1]; y <= c[3]; y++)
                for (int x = c[0]; x <= c[2]; x++) if (green[y * mw + x] == 1) gc++;
            if (gc / (double) (w * h) < 0.005) continue;
            out.add(new int[]{c[0], c[1], w, h, a});
        }
        Collections.sort(out, new Comparator<int[]>() {
            public int compare(int[] a, int[] b) { return Integer.compare(b[4], a[4]); }
        });
        // 还原到原图坐标
        List<int[]> res = new ArrayList<>();
        for (int[] c : out) {
            res.add(new int[]{(int) (c[0] / sc), (int) (c[1] / sc), (int) (c[2] / sc), (int) (c[3] / sc)});
        }
        return res;
    }

    /** 1D 方框最小/最大滤波（半径 r），返回新数组 */
    static byte[] morphMin(byte[] src, int w, int h, int r) {
        byte[] tmp = new byte[src.length], out = new byte[src.length];
        for (int y = 0; y < h; y++) {
            for (int x = 0; x < w; x++) {
                byte best = 1;
                for (int k = -r; k <= r; k++) {
                    int xx = x + k;
                    if (xx < 0 || xx >= w) continue;
                    if (src[y * w + xx] == 0) { best = 0; break; }
                }
                tmp[y * w + x] = best;
            }
        }
        for (int x = 0; x < w; x++) {
            for (int y = 0; y < h; y++) {
                byte best = 1;
                for (int k = -r; k <= r; k++) {
                    int yy = y + k;
                    if (yy < 0 || yy >= h) continue;
                    if (tmp[yy * w + x] == 0) { best = 0; break; }
                }
                out[y * w + x] = best;
            }
        }
        return out;
    }

    static byte[] morphMax(byte[] src, int w, int h, int r) {
        byte[] tmp = new byte[src.length], out = new byte[src.length];
        for (int y = 0; y < h; y++) {
            for (int x = 0; x < w; x++) {
                byte best = 0;
                for (int k = -r; k <= r; k++) {
                    int xx = x + k;
                    if (xx < 0 || xx >= w) continue;
                    if (src[y * w + xx] == 1) { best = 1; break; }
                }
                tmp[y * w + x] = best;
            }
        }
        for (int x = 0; x < w; x++) {
            for (int y = 0; y < h; y++) {
                byte best = 0;
                for (int k = -r; k <= r; k++) {
                    int yy = y + k;
                    if (yy < 0 || yy >= h) continue;
                    if (tmp[yy * w + x] == 1) { best = 1; break; }
                }
                out[y * w + x] = best;
            }
        }
        return out;
    }

    // ---------------- 渲染 ----------------
    public static Bitmap renderNumbered(Bitmap crop, List<Box> kept, int nDet, int finalCount) {
        Bitmap out = crop.copy(Bitmap.Config.ARGB_8888, true);
        Canvas cv = new Canvas(out);
        Paint box = new Paint();
        box.setStyle(Paint.Style.STROKE);
        box.setStrokeWidth(Math.max(1f, crop.getWidth() / 400f));
        box.setColor(Color.rgb(0, 220, 0));
        Paint txt = new Paint(Paint.ANTI_ALIAS_FLAG);
        txt.setTypeface(Typeface.create(Typeface.SANS_SERIF, Typeface.BOLD));
        txt.setTextSize(Math.max(10f, crop.getWidth() / 60f));
        txt.setColor(Color.YELLOW);
        txt.setShadowLayer(2f, 1f, 1f, Color.BLACK);

        // 按上->下、左->右编号
        List<Box> seq = new ArrayList<>(kept);
        Collections.sort(seq, new Comparator<Box>() {
            public int compare(Box a, Box b) {
                if (Math.abs(a.y - b.y) > 8) return Float.compare(a.y, b.y);
                return Float.compare(a.x, b.x);
            }
        });
        for (int i = 0; i < seq.size(); i++) {
            Box b = seq.get(i);
            cv.drawRect(b.x, b.y, b.x + b.w, b.y + b.h, box);
            cv.drawText(String.valueOf(i + 1), b.x + 2, b.y + txt.getTextSize(), txt);
        }
        // 顶栏
        Paint bar = new Paint();
        bar.setColor(Color.BLACK);
        float barH = txt.getTextSize() * 1.8f;
        cv.drawRect(0, 0, crop.getWidth(), barH, bar);
        Paint head = new Paint(Paint.ANTI_ALIAS_FLAG);
        head.setTypeface(Typeface.create(Typeface.SANS_SERIF, Typeface.BOLD));
        head.setTextSize(txt.getTextSize() * 1.2f);
        head.setColor(finalCount == nDet ? Color.rgb(0, 255, 0) : Color.rgb(255, 190, 0));
        cv.drawText("检测" + nDet + " / 校正后" + finalCount, 8, barH * 0.75f, head);
        return out;
    }

    public static Bitmap cropBitmap(Bitmap src, int x, int y, int w, int h) {
        x = Math.max(0, x); y = Math.max(0, y);
        w = Math.min(w, src.getWidth() - x);
        h = Math.min(h, src.getHeight() - y);
        if (w <= 2 || h <= 2) return null;
        return Bitmap.createBitmap(src, x, y, w, h);
    }
}
