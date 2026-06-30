import argparse, os, pickle, time
import numpy as np
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedGroupKFold
from features import extract_features, feature_names


def make_views(img_path: str, tmp_dir: str, prefix: str) -> list:
    im = Image.open(img_path).convert("RGB")
    w, h = im.size
    paths = []

    full_p = os.path.join(tmp_dir, f"{prefix}_full.jpg")
    im.save(full_p, quality=95)
    paths.append(full_p)

    flip_p = os.path.join(tmp_dir, f"{prefix}_flip.jpg")
    im.transpose(Image.FLIP_LEFT_RIGHT).save(flip_p, quality=95)
    paths.append(flip_p)

    cw, ch = int(w * 0.55), int(h * 0.55)
    xs = [0, (w - cw) // 2, w - cw]
    ys = [0, (h - ch) // 2, h - ch]
    i = 0
    for x in xs:
        for y in ys:
            crop_p = os.path.join(tmp_dir, f"{prefix}_g{i}.jpg")
            im.crop((x, y, x + cw, y + ch)).save(crop_p, quality=95)
            paths.append(crop_p)
            i += 1

    return paths


def load_dataset_augmented(real_dir: str, screen_dir: str, tmp_dir: str = "_views_tmp"):
    os.makedirs(tmp_dir, exist_ok=True)
    X, y, groups = [], [], []
    gid = 0

    def _add(folder, label):
        nonlocal gid
        files = [f for f in os.listdir(folder)
                 if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp', '.bmp'))]
        for fname in sorted(files):
            path = os.path.join(folder, fname)
            try:
                view_paths = make_views(path, tmp_dir, f"{label}_{gid}")
                for vp in view_paths:
                    feats = extract_features(vp)
                    X.append(feats)
                    y.append(label)
                    groups.append(gid)
                gid += 1
            except Exception:
                pass

    _add(real_dir, 0)
    _add(screen_dir, 1)

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int32)
    groups = np.array(groups, dtype=np.int32)
    return X, y, groups


def build_ensemble():
    lr = Pipeline([('sc', StandardScaler()),
                   ('clf', LogisticRegression(C=0.5, max_iter=3000, random_state=42))])
    svc = Pipeline([('sc', StandardScaler()),
                     ('clf', SVC(C=0.5, kernel='linear', probability=True, random_state=42))])
    rf = Pipeline([('sc', StandardScaler()),
                    ('clf', RandomForestClassifier(n_estimators=300, max_depth=8,
                                                    min_samples_leaf=3, random_state=42, n_jobs=-1))])
    return VotingClassifier(estimators=[('lr', lr), ('svc', svc), ('rf', rf)],
                             voting='soft', weights=[2, 2, 1])


def evaluate_grouped(X, y, groups, n_repeats=8):
    accs = []
    for rep in range(n_repeats):
        sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=rep)
        for tr_idx, te_idx in sgkf.split(X, y, groups):
            m = build_ensemble()
            m.fit(X[tr_idx], y[tr_idx])
            proba = m.predict_proba(X[te_idx])[:, 1]
            te_groups = groups[te_idx]
            agg = {}
            for g, p, lab in zip(te_groups, proba, y[te_idx]):
                agg.setdefault(g, {'p': [], 'y': lab})
                agg[g]['p'].append(p)
            correct = sum((np.mean(d['p']) >= 0.5) == d['y'] for d in agg.values())
            accs.append(correct / len(agg))
    accs = np.array(accs)
    print(f"acc={accs.mean():.4f} +/- {accs.std():.4f}")
    return accs.mean()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--real', default='data/real/')
    ap.add_argument('--screen', default='data/screen/')
    ap.add_argument('--out', default='model.pkl')
    ap.add_argument('--keep-tmp', action='store_true')
    args = ap.parse_args()

    X, y, groups = load_dataset_augmented(args.real, args.screen)
    acc = evaluate_grouped(X, y, groups)

    model = build_ensemble()
    model.fit(X, y)

    with open(args.out, 'wb') as f:
        pickle.dump(model, f, protocol=5)
    print(f"saved {args.out}  acc={acc*100:.1f}%")

    if not args.keep_tmp:
        import shutil
        shutil.rmtree('_views_tmp', ignore_errors=True)


if __name__ == '__main__':
    main()