from flask import Flask, render_template, send_from_directory, jsonify, request
import os, random, json, time
from werkzeug.utils import secure_filename

import cv2
import numpy as np

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = os.path.join('static', 'uploads')
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

try:
    FACE_CAS = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    PROF_CAS = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_profileface.xml')
except:
    FACE_CAS = None
    PROF_CAS = None

DATA_ROOT     = r"D:\DeepFashion2-master\DeepFashion2-master"
VAL_IMAGE_DIR = os.path.join(DATA_ROOT, "validation", "image")
VAL_ANNO_DIR  = os.path.join(DATA_ROOT, "validation", "annos")
try:
    available_images = [f for f in os.listdir(VAL_IMAGE_DIR) if f.endswith('.jpg')]
except FileNotFoundError:
    available_images = []

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/random_sample')
def get_random_sample():
    if not available_images:
        return jsonify({"error": "Dataset not found"}), 404
    fn = random.choice(available_images)
    iid = fn.replace('.jpg', '')
    try:
        with open(os.path.join(VAL_ANNO_DIR, iid + '.json')) as f:
            anno = json.load(f)
    except Exception as e:
        anno = {"error": str(e)}
    return jsonify({"image_url": f"/dataset/image/{fn}",
                    "annotations": anno, "image_id": iid})

@app.route('/dataset/image/<filename>')
def serve_image(filename):
    return send_from_directory(VAL_IMAGE_DIR, filename)

@app.route('/api/upload', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        time.sleep(0.5)

        import cv2, numpy as np

        # ── Load & scale ──────────────────────────────────────────────────────
        image = cv2.imread(filepath)
        if image is None:
            return jsonify({'error': 'Could not read image'}), 400
        h_img, w_img = image.shape[:2]
        MAX_W = 800
        scale = MAX_W / w_img if w_img > MAX_W else 1.0
        work  = cv2.resize(image, (int(w_img*scale), int(h_img*scale))) if scale < 1.0 else image.copy()
        wh, ww = work.shape[:2]

        gray_w  = cv2.cvtColor(work, cv2.COLOR_BGR2GRAY)
        ycrcb_w = cv2.cvtColor(work, cv2.COLOR_BGR2YCrCb)
        hsv_w   = cv2.cvtColor(work, cv2.COLOR_BGR2HSV)

        # ── Skin mask ─────────────────────────────────────────────────────────
        skin_mask = cv2.inRange(ycrcb_w,
                                np.array([0, 133, 77],   dtype=np.uint8),
                                np.array([255, 173, 127], dtype=np.uint8))

        # ── Face detection ─────────────────────────────────────────────────────
        raw = []
        if FACE_CAS is not None:
            # Resize image to max 400px width just for face detection to speed it up massively
            f_scale = 400 / ww if ww > 400 else 1.0
            f_gray = cv2.resize(gray_w, (int(ww*f_scale), int(wh*f_scale))) if f_scale < 1.0 else gray_w
            
            f_raw = list(FACE_CAS.detectMultiScale(f_gray, 1.1, 4)) + \
                    list(PROF_CAS.detectMultiScale(f_gray, 1.1, 4))
            
            # Map back to working scale
            for (fx, fy, fw, fh) in f_raw:
                raw.append((int(fx/f_scale), int(fy/f_scale), int(fw/f_scale), int(fh/f_scale)))

        valid_faces = []
        for (fx, fy, fw, fh) in raw:
            roi_skin  = skin_mask[fy:fy+fh, fx:fx+fw]
            skin_ratio = np.sum(roi_skin > 0) / float(fw * fh + 1)
            if fw >= 0.04 * ww and skin_ratio >= 0.15:
                valid_faces.append((fw * fh * skin_ratio, fx, fy, fw, fh))

        # ── Build garment mask (always — used for width profile) ──────────────
        white_bg = (hsv_w[:, :, 2] > 240) & (hsv_w[:, :, 1] < 30)
        dark_bg  =  hsv_w[:, :, 2] < 25
        grey_bg  = (hsv_w[:, :, 1] < 12) & (hsv_w[:, :, 2] > 195)
        navy_bg  = (hsv_w[:, :, 0] > 100) & (hsv_w[:, :, 0] < 135) & (hsv_w[:, :, 2] < 60)
        bg_mask  = (white_bg | dark_bg | grey_bg | navy_bg).astype(np.uint8) * 255
        g_mask   = cv2.bitwise_not(bg_mask)
        k9       = np.ones((9, 9), np.uint8)
        g_mask   = cv2.morphologyEx(g_mask, cv2.MORPH_CLOSE, k9, iterations=3)
        g_mask   = cv2.morphologyEx(g_mask, cv2.MORPH_OPEN,  k9, iterations=1)

        # ── Garment bounding box ──────────────────────────────────────────────
        # Compute g_mask contour bbox (used for product photos & as fallback)
        cnts, _ = cv2.findContours(g_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if cnts:
            best = max(cnts, key=cv2.contourArea)
            rx, ry, rw2, rh2 = cv2.boundingRect(best)
            gbx = max(0, int(rx / scale)); gby = max(0, int(ry / scale))
            gbw = min(int(rw2 / scale), w_img - gbx)
            gbh = min(int(rh2 / scale), h_img - gby)
        else:
            gbx = int(w_img*0.10); gby = int(h_img*0.10)
            gbw = int(w_img*0.80); gbh = int(h_img*0.80)

        g_aspect = gbh / float(gbw + 1)

        # When face detected: ALWAYS use face anchor (shirt region below chin).
        # This prevents full-body silhouette from being mistaken as trousers.
        # For trouser/bottom product photos, no face will be detected → g_mask used.
        if valid_faces:
            valid_faces.sort(key=lambda x: x[0], reverse=True)
            _, fx, fy, fw, fh = valid_faces[0]
            fx_o = int(fx / scale); fy_o = int(fy / scale)
            fw_o = int(fw / scale); fh_o = int(fh / scale)
            chin_y = int(fy_o + fh_o * 0.90)
            sw     = int(fw_o * 4.5)
            sx     = max(0, fx_o + fw_o // 2 - sw // 2)
            bx = max(0, sx);           by = max(0, chin_y)
            bw = min(sw, w_img - bx);  bh = min(int(fh_o * 4.2), h_img - by)
        else:
            bx, by, bw, bh = gbx, gby, gbw, gbh



        bx = max(0, bx); by = max(0, by)
        bw = max(1, min(bw, w_img - bx)); bh = max(1, min(bh, h_img - by))
        bx2 = bx + bw;  by2 = by + bh

        # Scaled coords for g_mask sampling
        bx_s = min(int(bx * scale), ww - 1)
        by_s = min(int(by * scale), wh - 1)
        bw_s = min(int(bw * scale), ww - bx_s)
        bh_s = min(int(bh * scale), wh - by_s)

        # ── Width-at-row helper using g_mask ──────────────────────────────────
        def width_at(frac):
            row  = min(int(by_s + bh_s * frac), g_mask.shape[0] - 1)
            line = g_mask[row, bx_s:bx_s + bw_s]
            nz   = np.where(line > 0)[0]
            return int(nz[-1] - nz[0] + 1) if len(nz) >= 2 else 0

        def row_y(frac):
            return int(by + bh * frac)

        # ── Classification ────────────────────────────────────────────────────
        aspect_ratio = bh / float(bw) if bw > 0 else 1.0

        # Scan widths 10%..90% every 5%
        fracs  = [f / 100.0 for f in range(10, 92, 5)]
        widths = {f: width_at(f) for f in fracs}
        wmax   = max(widths.values()) if widths else 1

        # Normalised widths
        def wn(f): return widths.get(f, 0) / float(wmax + 1)

        # Peak: row with maximum width in the UPPER half (10%..50%)
        top_fracs = [f for f in fracs if f <= 0.50]
        peak_f  = max(top_fracs, key=lambda f: widths[f]) if top_fracs else 0.30
        peak_wn = wn(peak_f)
        w50_n   = wn(0.50)
        w10_n   = wn(0.10)
        w20_n   = wn(0.20)

        # Drop from peak to mid-torso
        peak_drop = (peak_wn - w50_n) / (peak_wn + 0.001)
        # Shoulder flare: does width grow from collar (10%) to shoulder (20%)?
        shoulder_flare = (wn(0.22) - wn(0.10)) / (wn(0.10) + 0.001)
        # Bottom width for dress detection
        w90_n = wn(0.90)

        if aspect_ratio > 1.55:
            # Tall: trouser vs dress
            if w90_n / (w50_n + 0.001) > 1.10:
                category_name, category_id = "dress", 6
            else:
                category_name, category_id = "trousers", 8
        elif aspect_ratio < 0.55:
            category_name, category_id = "shorts", 7
        else:
            # Shirt/top range
            # Short sleeve: peak occurs early (< 45%) AND big drop to mid-torso
            if peak_f < 0.45 and peak_drop > 0.20:
                category_name, category_id = "short_sleeved_shirt", 1
            # Sleeveless/tank: collar zone (10%) is much narrower than shoulder zone (25%)
            # This means there are no sleeves — just straps. wn(0.10) will be low.
            elif wn(0.10) < 0.42 and wn(0.25) > wn(0.10) * 2.0:
                category_name, category_id = "sling", 5
            else:
                category_name, category_id = "long_sleeved_shirt", 2

        # ── Landmark generation ───────────────────────────────────────────────
        landmarks = [0] * (294 * 3)

        def set_lm(idx, x, y, v=2):
            x = int(max(bx, min(int(x), bx2)))
            y = int(max(by, min(int(y), by2)))
            landmarks[idx * 3]     = x
            landmarks[idx * 3 + 1] = y
            landmarks[idx * 3 + 2] = v

        cx = int(bx + bw * 0.50)

        if category_name == "trousers":
            set_lm(0, bx+bw*0.20, row_y(0.04)); set_lm(1, cx, row_y(0.04))
            set_lm(2, bx+bw*0.80, row_y(0.04)); set_lm(3, cx, row_y(0.45))
            set_lm(4, bx+bw*0.23, row_y(0.95)); set_lm(5, bx+bw*0.77, row_y(0.95))

        elif category_name == "shorts":
            set_lm(0, bx+bw*0.20, row_y(0.06)); set_lm(1, bx+bw*0.80, row_y(0.06))
            set_lm(2, cx, row_y(0.55))
            set_lm(3, bx+bw*0.22, row_y(0.92)); set_lm(4, bx+bw*0.78, row_y(0.92))

        elif category_name in ("dress", "skirt"):
            set_lm(0, bx+bw*0.38, row_y(0.08)); set_lm(1, bx+bw*0.62, row_y(0.08))
            set_lm(2, bx+bw*0.20, row_y(0.18)); set_lm(3, bx+bw*0.80, row_y(0.18))
            set_lm(4, bx+bw*0.28, row_y(0.50)); set_lm(5, bx+bw*0.72, row_y(0.50))
            set_lm(6, bx+bw*0.15, row_y(0.93)); set_lm(7, bx+bw*0.85, row_y(0.93))

        elif category_name == "sling":
            # Tank top / sleeveless vest
            set_lm(0, bx+bw*0.37, row_y(0.07)); set_lm(1, bx+bw*0.63, row_y(0.07))  # Straps
            set_lm(2, bx+bw*0.24, row_y(0.20)); set_lm(3, bx+bw*0.76, row_y(0.20))  # Armholes
            set_lm(4, bx+bw*0.22, row_y(0.50)); set_lm(5, bx+bw*0.78, row_y(0.50))  # Side seams
            set_lm(6, bx+bw*0.25, row_y(0.93)); set_lm(7, bx+bw*0.75, row_y(0.93))  # Hem

        elif category_name == "long_sleeved_shirt":
            set_lm(0, bx+bw*0.42, row_y(0.10)); set_lm(1, bx+bw*0.58, row_y(0.10))  # Collar
            set_lm(2, bx+bw*0.22, row_y(0.18)); set_lm(3, bx+bw*0.78, row_y(0.18))  # Shoulders
            set_lm(4, bx+bw*0.06, row_y(0.85)); set_lm(5, bx+bw*0.94, row_y(0.85))  # Cuffs
            set_lm(6, bx+bw*0.28, row_y(0.52)); set_lm(7, bx+bw*0.72, row_y(0.52))  # Mid-torso
            set_lm(11, bx+bw*0.28, row_y(0.93)); set_lm(12, bx+bw*0.72, row_y(0.93))  # Hem

        else:  # short_sleeved_shirt
            set_lm(0, bx+bw*0.40, row_y(0.10)); set_lm(1, bx+bw*0.60, row_y(0.10))  # Collar
            set_lm(2, bx+bw*0.22, row_y(0.18)); set_lm(3, bx+bw*0.78, row_y(0.18))  # Shoulders
            # Cuffs at actual peak row (end of sleeve)
            set_lm(4, bx+bw*0.06, row_y(peak_f)); set_lm(5, bx+bw*0.94, row_y(peak_f))
            # Armpits just below sleeve end
            ap_f = min(peak_f + 0.12, 0.55)
            set_lm(6, bx+bw*0.27, row_y(ap_f)); set_lm(7, bx+bw*0.73, row_y(ap_f))
            set_lm(11, bx+bw*0.27, row_y(0.93)); set_lm(12, bx+bw*0.73, row_y(0.93))  # Hem

        # ── Response ──────────────────────────────────────────────────────────
        return jsonify({
            "image_url": f"/static/uploads/{filename}",
            "annotations": {
                "item1": {
                    "category_name": category_name,
                    "category_id":   int(category_id),
                    "style": 1,
                    "bounding_box":  [int(bx), int(by), int(bx2), int(by2)],
                    "landmarks":     [int(x) for x in landmarks],
                    "segmentation":  [],
                    "scale": 2, "occlusion": 1, "zoom_in": 1, "viewpoint": 2
                }
            },
            "image_id": f"UPLOADED: {filename}"
        })

if __name__ == '__main__':
    app.run(debug=False, port=5000)
