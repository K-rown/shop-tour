"""Independent check of exposure matching: brightness of the SAME physical points in neighbouring stops, with and without the gains.
Points come from SIFT matches verified by an essential-matrix RANSAC (hop_geometry's own code), not from the floor projection that the
gains were fitted on. Compares two built tours (e.g. tour_noexp vs tour).   python qa/exposure_check.py tour_noexp tour
"""
import json, os, sys
import cv2, numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
import hop_geometry as hg

W = hg.W_SIFT


def main(folders, window=3):
    pts = json.load(open(os.path.join(HERE, "frames", "points.json")))
    n = len(pts)
    sift = cv2.SIFT_create(nfeatures=6000)
    feats = [hg.features(os.path.join(HERE, "frames", "points", p["filename"]), sift) for p in pts]
    bf = cv2.BFMatcher()
    pairs = []
    for i in range(n):
        for j in range(i + 1, min(n, i + window + 1)):
            k1, d1, w, h = feats[i]; k2, d2, _, _ = feats[j]
            m = [x for x, y in bf.knnMatch(d1, d2, k=2) if x.distance < 0.75 * y.distance]
            if len(m) < 40:
                continue
            b1 = hg.bearings(k1[[x.queryIdx for x in m]], w, h); b2 = hg.bearings(k2[[x.trainIdx for x in m]], w, h)
            E, inl = hg.ransac(b1, b2, seed=i * 100 + j)
            if inl.sum() < 40:
                continue
            idx = np.nonzero(inl)[0]
            pairs.append((i, j, k1[[m[t].queryIdx for t in idx]], k2[[m[t].trainIdx for t in idx]]))
    print(f"{len(pairs)} stop pairs with >= 40 verified matches", flush=True)
    out = {}
    for f in folders:
        imgs = [cv2.cvtColor(cv2.resize(cv2.imread(os.path.join(HERE, f, "panos_lo", p["filename"])), (W, W // 2), interpolation=cv2.INTER_AREA),
                             cv2.COLOR_BGR2GRAY).astype(np.float32) for p in pts]
        sm = [cv2.blur(g, (9, 9)) for g in imgs]
        res = []
        for i, j, a, b in pairs:
            la = sm[i][np.clip(a[:, 1].astype(int), 0, W // 2 - 1), np.clip(a[:, 0].astype(int), 0, W - 1)]
            lb = sm[j][np.clip(b[:, 1].astype(int), 0, W // 2 - 1), np.clip(b[:, 0].astype(int), 0, W - 1)]
            ok = (la > 25) & (la < 235) & (lb > 25) & (lb < 235)
            if ok.sum() >= 30:
                res.append(abs(float(np.median(np.log(lb[ok] / la[ok])))))
        out[f] = np.array(res)
        print(f"{f}: median |log brightness ratio| of matched points between stops: mean {out[f].mean():.4f}  median {np.median(out[f]):.4f}  ({len(res)} pairs)")
    if len(folders) == 2:
        a, b = out[folders[0]], out[folders[1]]
        print(f"{folders[1]} better on {(b < a).sum()}/{len(a)} pairs; mean change {100 * (b.mean() / a.mean() - 1):+.0f}%")


if __name__ == "__main__":
    main(sys.argv[1:])
