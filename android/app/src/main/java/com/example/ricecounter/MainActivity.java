package com.example.ricecounter;

import android.app.Activity;
import android.content.ContentValues;
import android.content.Intent;
import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.graphics.Typeface;
import android.net.Uri;
import android.os.Build;
import android.os.Bundle;
import android.os.Environment;
import android.provider.MediaStore;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.LinearLayout;
import android.widget.ScrollView;
import android.widget.TextView;
import android.widget.Toast;

import androidx.core.content.FileProvider;

import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.FileOutputStream;
import java.io.InputStream;
import java.io.OutputStream;
import java.io.PrintWriter;
import java.io.StringWriter;
import java.util.ArrayList;
import java.util.List;

import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtSession;

public class MainActivity extends Activity {
    private static final int PICK_IMAGE = 1;
    private static final int TAKE_PHOTO = 2;

    private TextView resultText;
    private ImageView resultImage;
    private TextView zoomHint;
    private Button saveBtn;
    private Uri photoUri;
    private Bitmap annotated;          // 最近一次结果图
    private final List<String> lastLog = new ArrayList<>();

    private OrtEnvironment env;
    private OrtSession session;
    private PanicleCounter.Calib calib;
    private String lastEngine = "?";
    private volatile boolean ready = false;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

        // 崩溃日志落盘（便于远程排查）
        final Thread.UncaughtExceptionHandler def = Thread.getDefaultUncaughtExceptionHandler();
        Thread.setDefaultUncaughtExceptionHandler(new Thread.UncaughtExceptionHandler() {
            public void uncaughtException(Thread t, Throwable e) {
                try {
                    StringWriter sw = new StringWriter();
                    e.printStackTrace(new PrintWriter(sw));
                    File f = new File(getExternalFilesDir(null), "crash.log");
                    FileOutputStream fo = new FileOutputStream(f, true);
                    fo.write(("\n==== " + new java.util.Date() + " ====\n" + sw.toString()).getBytes("UTF-8"));
                    fo.close();
                } catch (Throwable ignore) { }
                if (def != null) def.uncaughtException(t, e);
            }
        });

        ScrollView scroll = new ScrollView(this);
        LinearLayout layout = new LinearLayout(this);
        layout.setOrientation(LinearLayout.VERTICAL);
        layout.setPadding(40, 70, 40, 40);

        TextView title = new TextView(this);
        title.setTextSize(22);
        title.setTypeface(Typeface.DEFAULT_BOLD);
        title.setText("穗粒数计数 v2（遮挡校正）");
        layout.addView(title);

        TextView sub = new TextView(this);
        sub.setTextSize(13);
        sub.setText("拍照或选图 → 自动分穗 → 逐粒计数并校正 → 输出编号图\n标定 32 穗人工真值，MAPE 10.3%（旧版 19.6%）");
        sub.setPadding(0, 8, 0, 16);
        layout.addView(sub);

        Button photoBtn = new Button(this);
        photoBtn.setText("📷 拍照计数");
        photoBtn.setTextSize(18);
        photoBtn.setOnClickListener(new View.OnClickListener() {
            public void onClick(View v) { takePhoto(); }
        });
        layout.addView(photoBtn);

        Button pickBtn = new Button(this);
        pickBtn.setText("🖼 从相册选图计数");
        pickBtn.setTextSize(18);
        pickBtn.setOnClickListener(new View.OnClickListener() {
            public void onClick(View v) {
                Intent intent = new Intent(Intent.ACTION_GET_CONTENT);
                intent.setType("image/*");
                startActivityForResult(Intent.createChooser(intent, "选择图片"), PICK_IMAGE);
            }
        });
        layout.addView(pickBtn);

        resultText = new TextView(this);
        resultText.setTextSize(18);
        resultText.setPadding(0, 30, 0, 10);
        resultText.setText("首次使用请点按钮，模型载入约需 3~10 秒");
        layout.addView(resultText);

        saveBtn = new Button(this);
        saveBtn.setText("💾 保存编号图到相册");
        saveBtn.setEnabled(false);
        saveBtn.setOnClickListener(new View.OnClickListener() {
            public void onClick(View v) { saveToGallery(); }
        });
        layout.addView(saveBtn);

        resultImage = new ImageView(this);
        resultImage.setAdjustViewBounds(true);
        resultImage.setLayoutParams(new LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT, LinearLayout.LayoutParams.WRAP_CONTENT));
        resultImage.setOnClickListener(new View.OnClickListener() {
            public void onClick(View v) { showZoom(); }
        });
        layout.addView(resultImage);

        zoomHint = new TextView(this);
        zoomHint.setTextSize(13);
        zoomHint.setText("👆 点上方结果图可放大查看（双指缩放 / 单指拖动 / 双击放大）");
        zoomHint.setPadding(0, 4, 0, 12);
        layout.addView(zoomHint);

        scroll.addView(layout);
        setContentView(scroll);

        // 异步加载标定参数（模型改为按需加载/用后释放，避免占用大内存被系统回收）
        new Thread(new Runnable() {
            public void run() {
                try {
                    AssetManager am = getAssets();
                    calib = PanicleCounter.loadCalib(am);
                    runOnUiThread(new Runnable() {
                        public void run() {
                            String tail = lastCrashTail();
                            StringBuilder sb = new StringBuilder();
                            sb.append(calib.ok ? "就绪 ✅（含遮挡校正）" : "就绪（未找到标定参数，仅输出原始检测数）");
                            sb.append("\n点上方按钮拍照/选图");
                            if (tail != null) sb.append("\n\n⚠ 上次异常记录：\n").append(tail);
                            resultText.setText(sb.toString());
                        }
                    });
                } catch (final Exception e) {
                    runOnUiThread(new Runnable() {
                        public void run() { resultText.setText("初始化失败: " + e.getMessage()); }
                    });
                }
            }
        }).start();
    }

    /** 读 crash.log 最后一条异常的“开头几行”（异常类型+位置），供用户截图回传 */
    private String lastCrashTail() {
        try {
            File f = new File(getExternalFilesDir(null), "crash.log");
            if (!f.exists() || f.length() == 0) return null;
            java.io.RandomAccessFile ra = new java.io.RandomAccessFile(f, "r");
            long len = ra.length();
            long from = Math.max(0, len - 8000);
            ra.seek(from);
            byte[] buf = new byte[(int) (len - from)];
            ra.readFully(buf);
            ra.close();
            String s = new String(buf, "UTF-8");
            String[] lines = s.split("\n");
            // 找最后一条记录的起始
            int start = 0;
            for (int i = 0; i < lines.length; i++) {
                if (lines[i].startsWith("====")) start = i;
            }
            StringBuilder sb = new StringBuilder();
            int end = Math.min(lines.length, start + 9);
            for (int i = start; i < end; i++) sb.append(lines[i].trim()).append("\n");
            return sb.toString();
        } catch (Throwable e) {
            return null;
        }
    }

    private void takePhoto() {
        try {
            File dir = new File(getCacheDir(), "images");
            dir.mkdirs();
            File photoFile = new File(dir, "photo_" + System.currentTimeMillis() + ".jpg");
            photoUri = FileProvider.getUriForFile(this,
                    getApplicationContext().getPackageName() + ".fileprovider", photoFile);
            Intent intent = new Intent(MediaStore.ACTION_IMAGE_CAPTURE);
            intent.putExtra(MediaStore.EXTRA_OUTPUT, photoUri);
            intent.addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION | Intent.FLAG_GRANT_WRITE_URI_PERMISSION);
            startActivityForResult(intent, TAKE_PHOTO);
        } catch (Throwable e) {
            resultText.setText("无法调用相机: " + e.getClass().getSimpleName() + " " + e.getMessage()
                    + "\n可改用【从相册选图】");
        }
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        if (resultCode != RESULT_OK) return;
        try {
            byte[] bytes = null;
            if (requestCode == TAKE_PHOTO) {
                if (photoUri == null) { resultText.setText("相片路径丢失（App 可能在后台被系统回收过一次）\n请改用【从相册选图】或重拍"); return; }
                bytes = readBytes(photoUri);
            } else if (requestCode == PICK_IMAGE && data != null && data.getData() != null) {
                bytes = readBytes(data.getData());
            }
            if (bytes == null) { resultText.setText("读取图片失败"); return; }
            resultText.setText("正在分析...（大图约 10~60 秒）");
            final byte[] b = bytes;
            new Thread(new Runnable() {
                public void run() { analyze(b); }
            }).start();
        } catch (Throwable e) {
            resultText.setText("处理失败: " + e.getClass().getSimpleName() + " " + e.getMessage());
        }
    }

    private void analyze(byte[] bytes) {
        OrtSession sess = null;
        final long t0 = System.currentTimeMillis();
        try {
            // 释放上一次结果（回收在 UI 线程做，避免 ImageView 引用被回收的 Bitmap）
            final Bitmap old = annotated;
            annotated = null;
            runOnUiThread(new Runnable() {
                public void run() {
                    resultImage.setImageBitmap(null);
                    saveBtn.setEnabled(false);
                    resultText.setText("正在分析...（大图约 10~60 秒）");
                    if (old != null && !old.isRecycled()) { try { old.recycle(); } catch (Throwable ignore) { } }
                }
            });
            System.gc();

            // 按需采样解码（避免整张 4000x3000 ARGB 直接进内存）
            BitmapFactory.Options bounds = new BitmapFactory.Options();
            bounds.inJustDecodeBounds = true;
            BitmapFactory.decodeByteArray(bytes, 0, bytes.length, bounds);
            int mxSide = Math.max(bounds.outWidth, bounds.outHeight);
            int sample = 1;
            while (mxSide / sample > 2200) sample *= 2;
            BitmapFactory.Options o = new BitmapFactory.Options();
            o.inSampleSize = sample;
            o.inPreferredConfig = Bitmap.Config.ARGB_8888;
            Bitmap src = BitmapFactory.decodeByteArray(bytes, 0, bytes.length, o);
            if (src == null) {
                runOnUiThread(new Runnable() { public void run() { resultText.setText("图片解码失败"); } });
                return;
            }
            // EXIF 方向纠正（相机竖拍常见）
            int exifDeg = exifRotation(bytes);
            if (exifDeg != 0) {
                android.graphics.Matrix mtx = new android.graphics.Matrix();
                mtx.postRotate(exifDeg);
                Bitmap rot = Bitmap.createBitmap(src, 0, 0, src.getWidth(), src.getHeight(), mtx, true);
                if (rot != src) { src.recycle(); src = rot; }
            }
            // 若采样后仍偏大，再压到 2200
            int W = src.getWidth(), H = src.getHeight();
            int mx = Math.max(W, H);
            if (mx > 2200) {
                float k = 2200f / mx;
                Bitmap s2 = Bitmap.createScaledBitmap(src, Math.round(W * k), Math.round(H * k), true);
                src.recycle();
                src = s2;
            }

            List<int[]> pans = PanicleCounter.findPanicles(src);
            // 复用常驻会话（首次创建后才加载模型；不再每次重载）
            sess = getSession();
            env = OrtEnvironment.getEnvironment();
            StringBuilder sb = new StringBuilder();
            if (pans.size() == 1) {
                int[] p0 = pans.get(0);
                double cov = (double) p0[2] * p0[3] / (src.getWidth() * (double) src.getHeight());
                if (cov > 0.45) {
                    sb.append("⚠ 只认出 1 个穗区域且几乎覆盖整图（背景可能太杂，如拍了屏幕/桌面/杂色台面）。\n")
                      .append("建议：用【从相册选图】选原图，或在纯黑/纯白背景下拍穗。\n\n");
                }
            }
            if (pans.isEmpty()) {
                // 没分到穗 → 整图当一穗处理
                pans = new ArrayList<>();
                pans.add(new int[]{0, 0, src.getWidth(), src.getHeight()});
                sb.append("未识别到独立穗，按整图单穗处理\n");
            }
            Bitmap canvasBmp = src.copy(Bitmap.Config.ARGB_8888, true);
            Canvas cv = new Canvas(canvasBmp);
            Paint frame = new Paint();
            frame.setStyle(Paint.Style.STROKE);
            frame.setStrokeWidth(Math.max(3f, src.getWidth() / 300f));
            frame.setColor(Color.rgb(255, 60, 60));
            Paint label = new Paint(Paint.ANTI_ALIAS_FLAG);
            label.setTypeface(Typeface.create(Typeface.SANS_SERIF, Typeface.BOLD));
            label.setTextSize(Math.max(24f, src.getWidth() / 22f));
            label.setColor(Color.WHITE);
            label.setShadowLayer(4f, 2f, 2f, Color.BLACK);

            int total = 0, totalRaw = 0;
            StringBuilder detail = new StringBuilder();
            for (int i = 0; i < pans.size(); i++) {
                int[] p = pans.get(i);
                Bitmap crop = PanicleCounter.cropBitmap(src,
                        p[0] - 20, p[1] - 20, p[2] + 40, p[3] + 40);
                if (crop == null) continue;
                List<PanicleCounter.Box> raw = PanicleCounter.rawBoxes(sess, env, "images", crop);
                List<PanicleCounter.Box> kept = PanicleCounter.nms(PanicleCounter.keepByScore(raw, PanicleCounter.CONF), PanicleCounter.IOU);
                List<PanicleCounter.Box> f3 = PanicleCounter.keepByScore(raw, PanicleCounter.F3);
                double med = PanicleCounter.medianArea(f3);
                double ga = PanicleCounter.grainArea(crop);
                double occ = PanicleCounter.occMean(f3, crop.getWidth(), crop.getHeight());
                int[] rc = PanicleCounter.correct(calib, kept.size(), med, ga / Math.max(med, 1.0), occ);
                totalRaw += rc[0];
                total += rc[1];
                detail.append(String.format("穗%d: 检测 %d / 校正 %d\n", i + 1, rc[0], rc[1]));

                // 在整图上画编号（按上->下）
                List<PanicleCounter.Box> seq = new ArrayList<>(kept);
                final int offX = Math.max(0, p[0] - 20), offY = Math.max(0, p[1] - 20);
                java.util.Collections.sort(seq, new java.util.Comparator<PanicleCounter.Box>() {
                    public int compare(PanicleCounter.Box a, PanicleCounter.Box b) {
                        if (Math.abs(a.y - b.y) > 12) return Float.compare(a.y, b.y);
                        return Float.compare(a.x, b.x);
                    }
                });
                Paint nb = new Paint();
                nb.setStyle(Paint.Style.STROKE);
                nb.setStrokeWidth(Math.max(2f, src.getWidth() / 500f));
                nb.setColor(Color.rgb(0, 230, 0));
                Paint nt = new Paint(Paint.ANTI_ALIAS_FLAG);
                nt.setTypeface(Typeface.create(Typeface.SANS_SERIF, Typeface.BOLD));
                nt.setTextSize(Math.max(22f, src.getWidth() / 40f));
                nt.setColor(Color.YELLOW);
                nt.setShadowLayer(4f, 1.5f, 1.5f, Color.BLACK);
                for (int k = 0; k < seq.size(); k++) {
                    PanicleCounter.Box b = seq.get(k);
                    float x = offX + b.x, y = offY + b.y;
                    cv.drawRect(x, y, x + b.w, y + b.h, nb);
                    cv.drawText(String.valueOf(k + 1), x + 2, y + nt.getTextSize(), nt);
                }
                // 穗框 + 穗标签
                cv.drawRect(p[0], p[1], p[0] + p[2], p[1] + p[3], frame);
                cv.drawText("P" + (i + 1) + "=" + rc[1], p[0] + 6, Math.max(p[1] - 8, label.getTextSize()), label);
                crop.recycle();
                if ((i + 1) % 3 == 0) System.gc();
            }
            annotated = canvasBmp;
            src.recycle();
            System.gc();
            // 诊断日志（分穗框 + 每穗检出/校正），便于排查
            try {
                StringBuilder lg = new StringBuilder();
                lg.append("图 ").append(W).append("x").append(H)
                  .append("  exif=").append(exifDeg).append("\n分穗 ").append(pans.size()).append(" 个\n");
                for (int i = 0; i < pans.size(); i++) lg.append("  [" + pans.get(i)[0] + "," + pans.get(i)[1] + "," + pans.get(i)[2] + "," + pans.get(i)[3] + "]\n");
                lg.append(detail);
                lg.append(String.format("合计 校正%d 原始%d\n", total, totalRaw));
                File f = new File(getExternalFilesDir(null), "debug_last.txt");
                FileOutputStream fo = new FileOutputStream(f);
                fo.write(lg.toString().getBytes("UTF-8"));
                fo.close();
            } catch (Throwable ignore) { }
            final String msg = String.format("%s共 %d 株穗　校正合计 %d 粒（原始检测 %d）\n耗时 %.1f 秒（引擎 %s，均 %.1f 秒/株）\n%s点击下方按钮保存编号图",
                    sb.toString(), pans.size(), total, totalRaw,
                    (System.currentTimeMillis() - t0) / 1000.0, lastEngine,
                    (System.currentTimeMillis() - t0) / 1000.0 / Math.max(1, pans.size()),
                    detail.toString());
            runOnUiThread(new Runnable() {
                public void run() {
                    resultText.setText(msg);
                    resultImage.setImageBitmap(annotated);
                    saveBtn.setEnabled(true);
                }
            });
        } catch (Throwable e) {
            System.gc();
            final String em = e.getClass().getSimpleName() + " " + e.getMessage();
            try {
                StringWriter sw = new StringWriter();
                e.printStackTrace(new PrintWriter(sw));
                File f = new File(getExternalFilesDir(null), "crash.log");
                FileOutputStream fo = new FileOutputStream(f, true);
                fo.write(("\n==== " + new java.util.Date() + " (analyze) ====\n" + sw.toString()).getBytes("UTF-8"));
                fo.close();
            } catch (Throwable ignore) { }
            runOnUiThread(new Runnable() {
                public void run() { resultText.setText("分析失败: " + em + "\n（已记录，可在重开 App 后看到详情）"); }
            });
        } finally {
            // 会话常驻复用（下次分析不再重新加载）；离开前台时在 onStop 里释放
            System.gc();
        }
    }

    /** 创建推理会话（尝试 XNNPACK 移动端加速，失败回退 CPU） */
    private OrtSession createSession() throws Exception {
        if (env == null) env = OrtEnvironment.getEnvironment();
        OrtSession.SessionOptions opt = new OrtSession.SessionOptions();
        opt.setIntraOpNumThreads(Math.max(2, Runtime.getRuntime().availableProcessors() - 1));
        opt.setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT);
        String eng = "CPU";
        try {
            opt.addXnnpack(new java.util.HashMap<String, String>());
            eng = "XNNPACK";
        } catch (Throwable t) {
            eng = "CPU";
        }
        lastEngine = eng;
        byte[] model = PanicleCounter.readAssetBytes(getAssets(), "GrainNuber.onnx");
        return env.createSession(model, opt);
    }

    /** 取会话：首次创建后常驻，避免每次分析都重新加载模型 */
    private synchronized OrtSession getSession() throws Exception {
        if (session == null) session = createSession();
        return session;
    }

    private synchronized void closeSession() {
        if (session != null) {
            try { session.close(); } catch (Throwable ignore) { }
            session = null;
        }
    }

    @Override
    protected void onStop() {
        super.onStop();
        // 离开前台（例如去拍照）就释放模型内存，回来再按需加载
        closeSession();
    }

    /** 全屏放大查看结果图（双指缩放 / 拖动 / 双击） */
    private void showZoom() {
        if (annotated == null || annotated.isRecycled()) return;
        try {
            final android.app.Dialog d = new android.app.Dialog(this,
                    android.R.style.Theme_Black_NoTitleBar_Fullscreen);
            LinearLayout ll = new LinearLayout(this);
            ll.setOrientation(LinearLayout.VERTICAL);
            ZoomImageView zv = new ZoomImageView(this, annotated);
            ll.addView(zv, new LinearLayout.LayoutParams(
                    LinearLayout.LayoutParams.MATCH_PARENT, 0, 1f));
            TextView hint = new TextView(this);
            hint.setTextSize(14);
            hint.setTextColor(0xFFFFFFFF);
            hint.setText("双指缩放 · 单指拖动 · 双击放大/还原");
            hint.setPadding(24, 16, 24, 8);
            ll.addView(hint);
            Button close = new Button(this);
            close.setText("关闭");
            close.setOnClickListener(new View.OnClickListener() {
                public void onClick(View v) { d.dismiss(); }
            });
            ll.addView(close);
            d.setContentView(ll);
            d.show();
        } catch (Throwable e) {
            Toast.makeText(this, "放大失败: " + e.getMessage(), Toast.LENGTH_SHORT).show();
        }
    }

    private void saveToGallery() {
        if (annotated == null) return;
        try {
            String name = "穗粒数编号图_" + System.currentTimeMillis() + ".png";
            ContentValues v = new ContentValues();
            v.put(MediaStore.Images.Media.DISPLAY_NAME, name);
            v.put(MediaStore.Images.Media.MIME_TYPE, "image/png");
            if (Build.VERSION.SDK_INT >= 29) {
                v.put(MediaStore.Images.Media.RELATIVE_PATH, Environment.DIRECTORY_PICTURES + "/穗粒数计数");
            }
            Uri uri = getContentResolver().insert(MediaStore.Images.Media.EXTERNAL_CONTENT_URI, v);
            if (uri == null) throw new Exception("无法创建相册条目");
            OutputStream os = getContentResolver().openOutputStream(uri);
            annotated.compress(Bitmap.CompressFormat.PNG, 100, os);
            os.close();
            Toast.makeText(this, "已保存到相册：Pictures/穗粒数计数", Toast.LENGTH_LONG).show();
        } catch (Exception e) {
            // 兜底：写应用外部目录
            try {
                File dir = getExternalFilesDir(null);
                File f = new File(dir, "穗粒数编号图_" + System.currentTimeMillis() + ".png");
                FileOutputStream fo = new FileOutputStream(f);
                annotated.compress(Bitmap.CompressFormat.PNG, 100, fo);
                fo.close();
                Toast.makeText(this, "已保存: " + f.getAbsolutePath(), Toast.LENGTH_LONG).show();
            } catch (Exception e2) {
                Toast.makeText(this, "保存失败: " + e2.getMessage(), Toast.LENGTH_LONG).show();
            }
        }
    }

    /** 读取 EXIF 方向 → 需旋转角度（0/90/180/270） */
    private int exifRotation(byte[] bytes) {
        try {
            java.io.ByteArrayInputStream is = new java.io.ByteArrayInputStream(bytes);
            android.media.ExifInterface ex = new android.media.ExifInterface(is);
            int ori = ex.getAttributeInt(android.media.ExifInterface.TAG_ORIENTATION,
                    android.media.ExifInterface.ORIENTATION_NORMAL);
            is.close();
            switch (ori) {
                case android.media.ExifInterface.ORIENTATION_ROTATE_90: return 90;
                case android.media.ExifInterface.ORIENTATION_ROTATE_180: return 180;
                case android.media.ExifInterface.ORIENTATION_ROTATE_270: return 270;
                default: return 0;
            }
        } catch (Throwable e) {
            return 0;
        }
    }

    private byte[] readBytes(Uri uri) {
        try {
            InputStream is = getContentResolver().openInputStream(uri);
            ByteArrayOutputStream bo = new ByteArrayOutputStream();
            byte[] buf = new byte[1 << 16];
            int n;
            while ((n = is.read(buf)) != -1) bo.write(buf, 0, n);
            is.close();
            return bo.toByteArray();
        } catch (Exception e) {
            return null;
        }
    }
}
