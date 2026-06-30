# Spot the Fake Photo — Submission

## Approach

The detector is an ensemble of three classic ML classifiers (Logistic Regression,
linear-kernel SVM, Random Forest — soft voting) trained on hand-crafted image
features rather than raw pixels, so it stays tiny and fast enough for on-device use.

**Features extracted per image (`features.py`):**
- **Tiled FFT periodicity (strongest signal)** — splitting the frame into a
  grid of tiles (4×4, 6×6, 8×8) and running FFT peak-analysis on each tile
  separately. A screen's pixel/subpixel grid is periodic *everywhere* in the
  frame, while a real photo containing one patterned object (a railing, a
  tiled floor) is only periodic in the tiles that overlap that object — the
  *fraction* of periodic tiles separates the classes far better than any
  whole-image statistic.
- **Multi-scale FFT periodicity** — the same peak-to-mean check on the whole
  image at three resolutions, useful in combination with the tiled version but
  weak alone (real photos with one patterned texture can fool it).
- **Chromatic correlation** — RGB channel edge-correlation; screens often desync
  channels slightly differently than real-world optics.
- **Noise statistics** — kurtosis and row/col autocorrelation of the high-frequency
  residual (sensor noise vs. re-captured noise behaves differently).
- **Scanline periodicity** — autocorrelation of row/column intensity profiles,
  which can pick up display refresh/scan artifacts.
- **Colour stats** — saturation, channel correlation, entropy (screens tend to be
  more saturated/uniform).
- **Local contrast & sharpness** — variation in local contrast and Laplacian
  variance across tiles (uniform sharpness/blur across a screen vs. natural depth
  variation in a real scene).
- **Edge stats** — edge density and horizontal/vertical edge bias (screen bezels
  and pixel grids skew edges to be axis-aligned).

**Training augmentation:** each source photo is expanded into 8 views (full image,
horizontal flip, and 6 overlapping 55%-crops) before features are extracted. This
multiplies a modest dataset into more training examples and makes the model robust
to framing/cropping, while `StratifiedGroupKFold` keeps all views of the same
original photo together so evaluation never leaks across folds.

**Inference (`predict.py`):** the same 8-view augmentation is applied at prediction
time and the ensemble's probabilities are averaged across views for a more stable
score. If `model.pkl` is missing or fails to load, `predict.py` falls back to a
pure heuristic version of the FFT/chromatic/scanline signal (no training required),
so the script never hard-fails.

## Data

- 94 real photos, 61 photos-of-a-screen, taken on a phone across a range of
  lighting conditions, angles, and screens.

## Accuracy

Evaluated with grouped, repeated 5-fold cross-validation (8 repeats,
`StratifiedGroupKFold` on the source photo, not the augmented views, to avoid
leakage), aggregating each photo's held-out views by mean predicted probability:

```
acc = 92.4% (+/- ~4%)   — 154 photos, 40 held-out per-photo estimates
```

This is below the 95% target stated in the brief. It is the validated ceiling
found for this dataset size — a non-augmented, single-view baseline tops out
around 83–87% under the same honest cross-validation, and tree-only ensembles
without grouping looked better (~96%) only because they were overfitting and
leaking across views of the same photo. The augmentation + grouped CV +
soft-voting ensemble closes most, but not all, of that gap. Getting past ~92%
reliably would need more labelled photos rather than more clever code — see
"What I'd improve" below.

## Latency & cost

- **Latency:** ~240–260 ms per image, measured on this container's CPU
  (10-run average, ML mode, model loaded once and cached). Feature extraction
  dominates — the model itself is a small ensemble pipeline, so inference is
  effectively free. Most of the cost is the multi-scale FFT and the 8×8-grid
  tile-periodicity check at 1024×768, the strongest single feature group, run
  across all 8 augmented views per prediction. An easy win: short-circuit out
  of the 1024 px pass (and/or use fewer views) once the cheaper, faster scales
  already give a confident verdict.
- **Cost per image:** on-device ≈ free (runs locally, no network call, no
  per-image API fee). For a cloud server: at ~250 ms/image, a single CPU core
  processes ~14,400 images/hour. A small on-demand instance like an AWS
  t3.medium (~$0.0416/hr) running this serially works out to roughly
  **$0.003 per 1,000 images** (~$2.90 per million), before accounting for
  request overhead, idle time, or batching gains from running multiple
  workers per box — so treat this as a rough floor, not a guaranteed price.

## Demo

`server.py` + `demo.html` provide a small local web demo (image upload and live
camera capture) that calls the same `predict()` function. A preview video of the
live-camera demo is included separately.

To run:
```
python server.py
# open http://127.0.0.1:5000 in a browser
```

## What I'd improve with more time

- More data and more screen *types* (different phone/laptop/TV panels, different
  moiré patterns) — the model currently risks overfitting to the screens used to
  collect training photos.
- Calibrate the score threshold against a precision/recall tradeoff suited to the
  fraud use case (e.g. bias toward fewer false "REAL" verdicts) rather than the
  default 0.5 cutoff.
- Distill the feature pipeline down to the few highest-signal features (FFT +
  tile periodicity look like the strongest) to cut latency further for on-device
  deployment.
- Add adversarial robustness: test against cheaters who photograph a screen at an
  angle, with reduced brightness, or print-and-rephotograph to defeat moiré-based
  cues, and add corresponding training examples.
