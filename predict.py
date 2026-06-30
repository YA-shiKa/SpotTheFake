import sys, os, time, pickle
import numpy as np
from PIL import Image
import cv2

_DIR = os.path.dirname(os.path.abspath(__file__))
_MODEL = os.path.join(_DIR, "model.pkl")

PREFER_ML_IF_AVAILABLE = True


def _heuristic(image_path: str) -> float:
    img_pil = Image.open(image_path).convert("RGB")
    rgb_full = np.array(img_pil)

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

    rgb_m = cv2.resize(rgb_full, (512, 384), interpolation=cv2.INTER_AREA)
    r_lap = np.abs(cv2.Laplacian(rgb_m[:, :, 0].astype(np.float32), cv2.CV_32F))
    g_lap = np.abs(cv2.Laplacian(rgb_m[:, :, 1].astype(np.float32), cv2.CV_32F))
    b_lap = np.abs(cv2.Laplacian(rgb_m[:, :, 2].astype(np.float32), cv2.CV_32F))
    rg = np.corrcoef(r_lap.ravel(), g_lap.ravel())[0, 1]
    rb = np.corrcoef(r_lap.ravel(), b_lap.ravel())[0, 1]
    gb = np.corrcoef(g_lap.ravel(), b_lap.ravel())[0, 1]
    chrom_mean = (rg + rb + gb) / 3
    s2 = float(np.clip((chrom_mean - 0.965) / 0.025, 0, 1))

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


_model_cache = None

def _ml_predict(image_path: str) -> float:
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


def predict(image_path: str) -> float:
    if PREFER_ML_IF_AVAILABLE and os.path.exists(_MODEL):
        try:
            return _ml_predict(image_path)
        except Exception:
            pass
    return _heuristic(image_path)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(1)

    path = sys.argv[1]
    if not os.path.exists(path):
        sys.exit(1)

    t0 = time.perf_counter()
    score = predict(path)
    ms = (time.perf_counter() - t0) * 1000

    print(f"{score:.4f}")