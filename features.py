import numpy as np
from PIL import Image
import cv2


def _to_gray(rgb: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)


def _resize(rgb: np.ndarray, size) -> np.ndarray:
    return cv2.resize(rgb, size, interpolation=cv2.INTER_AREA)


def _fft_peak_to_mean(gray: np.ndarray) -> tuple[float, float]:
    h, w = gray.shape
    win = np.hanning(h)[:, None] * np.hanning(w)[None, :]
    mag = np.abs(np.fft.fftshift(np.fft.fft2(gray * win))) + 1e-9
    cy, cx = h // 2, w // 2
    Y, X = np.ogrid[:h, :w]
    r = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)
    lo, hi = min(h, w) * 0.05, min(h, w) * 0.45
    mid = (r > lo) & (r < hi)
    band = mag[mid]
    ptm = float(band.max() / band.mean())
    top20 = float(np.sort(band.ravel())[-20:].mean() / band.mean())
    return ptm, top20


def fft_features(rgb: np.ndarray) -> np.ndarray:
    feats = []
    for size in [(256, 192), (512, 384), (1024, 768)]:
        rgb_s = _resize(rgb, size)
        gray_s = _to_gray(rgb_s)
        ptm, top20 = _fft_peak_to_mean(gray_s)
        feats.extend([ptm, top20])
    ptms = feats[0::2]
    feats.append(max(ptms))
    return np.array(feats, dtype=np.float32)


def tile_periodicity_features(rgb: np.ndarray) -> np.ndarray:
    feats = []
    for grid, res in [(4, (1024, 768)), (6, (1200, 900)), (8, (1024, 768))]:
        rgb_s = _resize(rgb, res)
        gray_s = _to_gray(rgb_s)
        h, w = gray_s.shape
        th, tw = h // grid, w // grid
        if th < 24 or tw < 24:
            feats.extend([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
            continue
        tile_ptms = []
        for r in range(grid):
            for c in range(grid):
                tile = gray_s[r*th:(r+1)*th, c*tw:(c+1)*tw]
                th2, tw2 = tile.shape
                win = np.hanning(th2)[:, None] * np.hanning(tw2)[None, :]
                mag = np.abs(np.fft.fftshift(np.fft.fft2(tile * win))) + 1e-9
                cy, cx = th2 // 2, tw2 // 2
                Y, X = np.ogrid[:th2, :tw2]
                rr = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)
                mid = (rr > min(th2, tw2) * 0.1) & (rr < min(th2, tw2) * 0.45)
                if mid.sum() < 5:
                    continue
                band = mag[mid]
                tile_ptms.append(float(band.max() / band.mean()))
        tile_ptms = np.array(tile_ptms) if tile_ptms else np.array([0.0])
        feats.extend([
            float(np.mean(tile_ptms > 12)),
            float(np.mean(tile_ptms > 15)),
            float(np.mean(tile_ptms > 20)),
            float(tile_ptms.mean()),
            float(tile_ptms.min()),
            float(np.median(tile_ptms)),
        ])
    return np.array(feats, dtype=np.float32)


def chromatic_features(rgb: np.ndarray) -> np.ndarray:
    r = rgb[:, :, 0].astype(np.float32)
    g = rgb[:, :, 1].astype(np.float32)
    b = rgb[:, :, 2].astype(np.float32)

    r_lap = np.abs(cv2.Laplacian(r, cv2.CV_32F))
    g_lap = np.abs(cv2.Laplacian(g, cv2.CV_32F))
    b_lap = np.abs(cv2.Laplacian(b, cv2.CV_32F))

    rg = np.corrcoef(r_lap.ravel(), g_lap.ravel())[0, 1]
    rb = np.corrcoef(r_lap.ravel(), b_lap.ravel())[0, 1]
    gb = np.corrcoef(g_lap.ravel(), b_lap.ravel())[0, 1]

    return np.array([rg, rb, gb, (rg + rb + gb) / 3], dtype=np.float32)


def noise_features(gray: np.ndarray) -> np.ndarray:
    smooth = cv2.GaussianBlur(gray, (5, 5), 1.0)
    noise = gray - smooth
    flat = noise.ravel()
    std = flat.std() + 1e-9
    kurtosis = float(np.mean(((flat - flat.mean()) / std) ** 4))

    n64 = cv2.resize(noise, (64, 64))
    row_ac = float(np.corrcoef(n64[:-1, :].ravel(), n64[1:, :].ravel())[0, 1])
    col_ac = float(np.corrcoef(n64[:, :-1].ravel(), n64[:, 1:].ravel())[0, 1])
    noise_var = float(np.var(noise))

    return np.array([kurtosis, row_ac, col_ac, noise_var], dtype=np.float32)


def scanline_features(gray: np.ndarray) -> np.ndarray:
    def short_lag_energy(signal_1d):
        s = signal_1d - signal_1d.mean()
        ac = np.correlate(s, s, mode='full')
        ac = ac[len(ac) // 2:]
        ac /= (ac[0] + 1e-9)
        return float(np.sum(ac[2:21] ** 2))

    row_e = short_lag_energy(gray.mean(axis=1))
    col_e = short_lag_energy(gray.mean(axis=0))
    return np.array([row_e, col_e, max(row_e, col_e)], dtype=np.float32)


def colour_features(rgb: np.ndarray) -> np.ndarray:
    r = rgb[:, :, 0].astype(np.float32) / 255.0
    g = rgb[:, :, 1].astype(np.float32) / 255.0
    b = rgb[:, :, 2].astype(np.float32) / 255.0

    max_c = np.maximum(np.maximum(r, g), b)
    min_c = np.minimum(np.minimum(r, g), b)
    sat = (max_c - min_c) / (max_c + 1e-9)

    mean_sat = float(np.mean(sat))
    high_sat_frac = float(np.mean(sat > 0.6))
    rg_corr = float(np.corrcoef(r.ravel(), g.ravel())[0, 1])
    rb_corr = float(np.corrcoef(r.ravel(), b.ravel())[0, 1])

    def entropy(ch):
        hist, _ = np.histogram(ch.ravel(), bins=64, range=(0, 1))
        p = hist / (hist.sum() + 1e-9)
        p = p[p > 0]
        return float(-np.sum(p * np.log2(p)))

    ent = (entropy(r) + entropy(g) + entropy(b)) / 3

    return np.array([mean_sat, high_sat_frac, rg_corr, rb_corr, ent], dtype=np.float32)


def contrast_sharpness_features(gray: np.ndarray) -> np.ndarray:
    kernel = np.ones((9, 9), np.float32) / 81
    mean_sq = cv2.filter2D(gray ** 2, -1, kernel)
    sq_mean = cv2.filter2D(gray, -1, kernel) ** 2
    local_std = np.sqrt(np.clip(mean_sq - sq_mean, 0, None))
    lc_mean = float(np.mean(local_std))
    lc_cov = float(np.std(local_std) / (lc_mean + 1e-9))

    h, w = gray.shape
    tile_vars = np.zeros((4, 4), dtype=np.float32)
    for rr in range(4):
        for cc in range(4):
            tile = gray[rr * h // 4:(rr + 1) * h // 4, cc * w // 4:(cc + 1) * w // 4]
            tile_vars[rr, cc] = np.var(cv2.Laplacian(tile, cv2.CV_32F))

    sharp_cov = float(tile_vars.std() / (tile_vars.mean() + 1e-9))
    grad = float((np.mean(np.abs(np.diff(tile_vars, axis=0))) +
                  np.mean(np.abs(np.diff(tile_vars, axis=1)))) / (tile_vars.mean() + 1e-9))

    return np.array([lc_mean, lc_cov, sharp_cov, grad], dtype=np.float32)


def edge_features(gray: np.ndarray) -> np.ndarray:
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag_e = np.sqrt(gx ** 2 + gy ** 2) + 1e-9
    angle = np.arctan2(np.abs(gy), np.abs(gx))

    strong = mag_e > np.percentile(mag_e, 75)
    if strong.sum() == 0:
        hv_frac = 0.5
    else:
        a = angle[strong]
        hv_frac = float(np.mean((a < 0.3) | (a > np.pi / 2 - 0.3)))

    edge_density = float(strong.mean())
    return np.array([hv_frac, edge_density, float(mag_e.mean())], dtype=np.float32)


def extract_features(image_path: str) -> np.ndarray:
    img_pil = Image.open(image_path).convert("RGB")
    rgb_full = np.array(img_pil)

    rgb = _resize(rgb_full, (512, 384))
    gray = _to_gray(rgb)

    feats = np.concatenate([
        fft_features(rgb_full),
        tile_periodicity_features(rgb_full),
        chromatic_features(rgb),
        noise_features(gray),
        scanline_features(gray),
        colour_features(rgb),
        contrast_sharpness_features(gray),
        edge_features(gray),
    ])
    return feats


def feature_names() -> list[str]:
    names = [
        "fft_ptm_256", "fft_top20_256",
        "fft_ptm_512", "fft_top20_512",
        "fft_ptm_1024", "fft_top20_1024",
        "fft_max_ptm",
    ]
    for grid in [4, 6, 8]:
        names.extend([
            f"tile_g{grid}_frac_p12", f"tile_g{grid}_frac_p15", f"tile_g{grid}_frac_p20",
            f"tile_g{grid}_ptm_mean", f"tile_g{grid}_ptm_min", f"tile_g{grid}_ptm_median",
        ])
    names.extend([
        "chrom_rg", "chrom_rb", "chrom_gb", "chrom_mean",
        "noise_kurtosis", "noise_row_ac", "noise_col_ac", "noise_var",
        "scanline_row_e", "scanline_col_e", "scanline_max_e",
        "colour_sat_mean", "colour_high_sat_frac", "colour_rg_corr",
        "colour_rb_corr", "colour_entropy",
        "lc_mean", "lc_cov", "sharp_cov", "sharp_smoothness",
        "edge_hv_frac", "edge_density", "edge_mean_mag",
    ])
    return names