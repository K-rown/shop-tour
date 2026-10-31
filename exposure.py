"""Even brightness/colour between stops (so the glide blend does not flash).

For every pair of stops that see the same floor (floorlink.py), compare the colour of the same floor points in both panoramas
(median of log-ratios, so objects standing on the floor and occlusions are outvoted). A least-squares solve over all pairs gives one
gain per stop and colour channel such that gain_i * colour_i == gain_j * colour_j on the shared floor, with the mean gain fixed at 1
and every gain clamped to 0.7 .. 1.4. build_tour.py multiplies each pano by its gain.
"""
import json, os
from concurrent.futures import ThreadPoolExecutor
import cv2
import numpy as np

from floorlink import FloorLinks
from nadir import body_mask
from imutil import imread

W = 2048
LO, HI = 0.7, 1.4


def load_small(path):
    im = imread(path)
    return cv2.GaussianBlur(cv2.resize(im, (W, W // 2), interpolation=cv2.INTER_AREA).astype(np.float32), (0, 0), 2.0)


def pair_offsets(FL, G, i, j, bm):
    """Median log-ratio (j over i, per BGR channel) of the same floor points seen from both stops, and the sample count."""
    az, el = np.meshgrid(np.radians(np.arange(-180, 180, 3.0)), np.radians(np.arange(-65, -24, 3.0)))
    aj, ej, hz = FL.project(i, j, az, el)
    ok = (ej < np.radians(-18)) & (hz < 9.0)
    h = W // 2
    ui, vi = FL.raw_uv(i, az, el, W, h); uj, vj = FL.raw_uv(j, aj, ej, W, h)
    ui, vi, uj, vj = [x.astype(np.float32) for x in (ui, vi, uj, vj)]
    a = cv2.remap(G[i], ui, vi, cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
    b = cv2.remap(G[j], uj, vj, cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
    inbody = lambda u, v: cv2.remap(bm, u, v, cv2.INTER_NEAREST, borderMode=cv2.BORDER_WRAP) > 0
    ok &= ~inbody(ui, vi) & ~inbody(uj, vj)
    ya, yb = a.mean(-1), b.mean(-1)
    ok &= (ya > 20) & (ya < 235) & (yb > 20) & (yb < 235)
    if ok.sum() < 150:
        return None, int(ok.sum())
    lr = np.log((b[ok] + 1.0) / (a[ok] + 1.0))
    return np.median(lr, axis=0), int(ok.sum())


def gains_from_offsets(rows, offs, wts, n):
    """Least-squares gains from pair offsets: log g_i - log g_j = offs (log c_j - log c_i), weighted, mean log gain 0, clamped to LO..HI.
    rows: [(i, j)], offs: (P, 3), wts: (P,). Returns (n, 3)."""
    offs = np.asarray(offs).reshape(-1, 3); wts = np.asarray(wts)
    A = np.zeros((len(rows) + 1, n))
    for r, (i, j) in enumerate(rows):
        A[r, i] = wts[r]; A[r, j] = -wts[r]
    A[-1, :] = 1.0                                 # mean log gain = 0
    sol = np.zeros((n, 3))
    for c in range(3):
        b = np.concatenate([offs[:, c] * wts, [0.0]])
        sol[:, c] = np.linalg.lstsq(A, b, rcond=None)[0]
    return np.clip(np.exp(sol), LO, HI)


def solve_gains(frames="frames", files=None, verbose=True):
    FL = FloorLinks(frames)
    n = FL.n
    pts = json.load(open(os.path.join(frames, "points.json")))
    with ThreadPoolExecutor(max_workers=min(8, os.cpu_count() or 1)) as ex:
        G = list(ex.map(lambda p: load_small(os.path.join(frames, "points", p["filename"])), pts))
    bm = body_mask(W // 2, W)
    rows, offs, wts, info = [], [], [], []
    for i in range(n):
        for j in FL.neighbours(i, 1.0, 6.0):
            if j <= i:
                continue
            o, cnt = pair_offsets(FL, G, i, j, bm)
            if o is None:
                continue
            rows.append((i, j)); offs.append(o); wts.append(np.sqrt(cnt)); info.append(cnt)
    if not rows:                                   # stops that share no floor (very sparse tour): nothing to match, leave the pictures as they are
        if verbose:
            print("exposure: no stops share enough floor; exposure left as recorded")
        return np.ones((n, 3)), {"pairs": 0, "median_samples": 0, "mean_abs_log_ratio_before": 0.0, "mean_abs_log_ratio_after": 0.0,
                                 "gains_bgr": [[1.0, 1.0, 1.0]] * n, "gain_min": 1.0, "gain_max": 1.0}
    offs = np.array(offs); wts = np.array(wts)
    g = gains_from_offsets(rows, offs, wts, n)
    # residual disagreement between linked stops before / after (mean abs log ratio over pairs and channels)
    before = np.abs(offs).mean(); after = np.abs(offs - (np.log(g[[i for i, _ in rows]]) - np.log(g[[j for _, j in rows]]))).mean()
    rep = {"pairs": len(rows), "median_samples": int(np.median(info)), "mean_abs_log_ratio_before": round(float(before), 4),
           "mean_abs_log_ratio_after": round(float(after), 4), "gains_bgr": [[round(float(x), 3) for x in r] for r in g],
           "gain_min": round(float(g.min()), 3), "gain_max": round(float(g.max()), 3)}
    if verbose:
        print(f"exposure: {len(rows)} pairs, residual {before:.4f} -> {after:.4f}, gains {g.min():.2f}..{g.max():.2f}")
    return g, rep


if __name__ == "__main__":
    g, rep = solve_gains()
    json.dump(rep, open(os.path.join("frames", "exposure.json"), "w"), indent=1)
    for k, r in enumerate(g):
        print(f"{k + 1:03d}  B {r[0]:.3f}  G {r[1]:.3f}  R {r[2]:.3f}")
