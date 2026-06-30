"""
predict.py — Screen-photo (recapture) detector.

Usage:
    python predict.py some_image.jpg
Prints ONE number from 0 to 1:
    0 = real photo,  1 = photo of a screen (recapture / fraud)

"""

import sys, os, time, pickle
import numpy as np
from PIL import Image
import cv2

_DIR   = os.path.dirname(os.path.abspath(__file__))
_MODEL = os.path.join(_DIR, "model.pkl")

# model.pkl is the 3-model soft-voting ensemble from train.py, trained on
# 154 labelled photos using multi-view augmentation and validated at
# ~92.4% per-photo accuracy under StratifiedGroupKFold (no leakage
# between views of the same photo).
PREFER_ML_IF_AVAILABLE = True


# ────────────────────────────────────────────────────────────────
#  Pure-CV heuristic (no model needed)
# ────────────────────────────────────────────────────────────────

def _heuristic(image_path: str) -> float:
    """
    Returns a score in [0, 1]: higher = more screen-like.

    Combines three signals, weighted by how reliably each one separated
    real vs. screen photos during calibration:

      1. Multi-scale FFT peak-to-mean ratio (weight 0.60) 
      2. Chromatic-aberration proxy / channel-edge alignment (weight 0.25)
      3. Scanline / subpixel banding (weight 0.15) 
    
    """
    img_pil = Image.open(image_path).convert("RGB")
    rgb_full = np.array(img_pil)

    # ── Signal 1: multi-scale FFT peak-to-mean ──────────────────
    ptms = []
    for size in [(256, 192), (512, 384), (1024, 768)]:
        rgb_s = cv2.resize(rgb_full, size, interpolation=cv2.INTER_AREA)
        gray_s = cv2.cvtColor(rgb_s, cv2.COLOR_RGB2GRAY).astype(np.float32)
        h, w = gray_s.shape
        win = np.hanning(h)[:, None] * np.hanning(w)[None, :]
        mag = np.abs(np.fft.fftshift(np.fft.fft2(gray_s * win))) + 1e-9
        cy, cx = h // 2, w // 2
        Y, X = np.ogrid[:h, :w]
        r = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)
        lo, hi = min(h, w) * 0.05, min(h, w) * 0.45
        mid = (r > lo) & (r < hi)
        band = mag[mid]
        ptms.append(float(band.max() / band.mean()))
    fft_ptm = max(ptms)
    s1 = float(np.clip((fft_ptm - 12) / 24, 0, 1))

    # ── Signal 2: chromatic-aberration / channel-edge alignment ─
    rgb_m = cv2.resize(rgb_full, (512, 384), interpolation=cv2.INTER_AREA)
    r_lap = np.abs(cv2.Laplacian(rgb_m[:, :, 0].astype(np.float32), cv2.CV_32F))
    g_lap = np.abs(cv2.Laplacian(rgb_m[:, :, 1].astype(np.float32), cv2.CV_32F))
    b_lap = np.abs(cv2.Laplacian(rgb_m[:, :, 2].astype(np.float32), cv2.CV_32F))
    rg = np.corrcoef(r_lap.ravel(), g_lap.ravel())[0, 1]
    rb = np.corrcoef(r_lap.ravel(), b_lap.ravel())[0, 1]
    gb = np.corrcoef(g_lap.ravel(), b_lap.ravel())[0, 1]
    chrom_mean = (rg + rb + gb) / 3
    s2 = float(np.clip((chrom_mean - 0.965) / 0.025, 0, 1))

    # ── Signal 3: row/column scanline banding ───────────────────
    gray_m = cv2.cvtColor(rgb_m, cv2.COLOR_RGB2GRAY).astype(np.float32)

    def short_lag_energy(signal_1d):
        s = signal_1d - signal_1d.mean()
        ac = np.correlate(s, s, mode='full')
        ac = ac[len(ac) // 2:]
        ac /= (ac[0] + 1e-9)
        return float(np.sum(ac[2:21] ** 2))

    row_e = short_lag_energy(gray_m.mean(axis=1))
    col_e = short_lag_energy(gray_m.mean(axis=0))
    scan_e = max(row_e, col_e)
    s3 = float(np.clip((scan_e - 8) / 12, 0, 1))

    score = 0.60 * s1 + 0.25 * s2 + 0.15 * s3
    return float(np.clip(score, 0.0, 1.0))


# ────────────────────────────────────────────────────────────────
#  ML predictor
# ────────────────────────────────────────────────────────────────

_model_cache = None

def _ml_predict(image_path: str) -> float:
    """
    Multi-view test-time averaging
    """
    global _model_cache
    if _model_cache is None:
        with open(_MODEL, 'rb') as f:
            _model_cache = pickle.load(f)
    from features import extract_features
    import tempfile

    img_pil = Image.open(image_path).convert("RGB")
    w, h = img_pil.size

    views = [img_pil, img_pil.transpose(Image.FLIP_LEFT_RIGHT)]
    cw, ch = int(w * 0.55), int(h * 0.55)
    xs = [0, (w - cw) // 2, w - cw]
    ys = [0, (h - ch) // 2, h - ch]
    for x in xs:
        for y in ys:
            views.append(img_pil.crop((x, y, x + cw, y + ch)))

    probs = []
    with tempfile.TemporaryDirectory() as td:
        for i, v in enumerate(views):
            vp = os.path.join(td, f"view_{i}.jpg")
            v.save(vp, quality=95)
            feats = extract_features(vp).reshape(1, -1)
            probs.append(_model_cache.predict_proba(feats)[0, 1])

    return float(np.mean(probs))


# ────────────────────────────────────────────────────────────────
#  Public API
# ────────────────────────────────────────────────────────────────

def predict(image_path: str) -> float:
    """
    Returns a fraud score in [0, 1].
      0 = definitely a real photo
      1 = definitely a photo of a screen
    """
    if PREFER_ML_IF_AVAILABLE and os.path.exists(_MODEL):
        try:
            return _ml_predict(image_path)
        except Exception as e:
            print(f"[predict] ML model failed ({e}), falling back to heuristic",
                  file=sys.stderr)
    return _heuristic(image_path)


# ────────────────────────────────────────────────────────────────
#  CLI entry point
# ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python predict.py <image_path>", file=sys.stderr)
        sys.exit(1)

    path = sys.argv[1]
    if not os.path.exists(path):
        print(f"Error: file not found: {path}", file=sys.stderr)
        sys.exit(1)

    t0 = time.perf_counter()
    score = predict(path)
    ms = (time.perf_counter() - t0) * 1000

    print(f"{score:.4f}")
    label = "SCREEN (fraud)" if score >= 0.5 else "REAL"
    print(f"  -> {label}  (latency: {ms:.1f} ms)", file=sys.stderr)
