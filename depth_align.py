"""Put the single-photo depth on the multi-view depth: the dense picture comes from the depth model, its scale comes from what several stops agree on.

Usage:  python depth_align.py [--frames frames] [--sigma 40] [--only 003 004]
Reads:  frames/depth/eq_<id>.png (the depth model's range, tied to the sparse map: place_rings.py), frames/mvs/<id>.npz (mvs.py: range where >= 2 other stops agree)
Writes: frames/depth_mvs/eq_<id>.png (same encoding as frames/depth/eq_<id>.png, range channel replaced; the floor/rug channels are kept),
        frames/depth_mvs/<id>_anchor.png (255 where the range is the multi-view one), frames/depth_mvs/stats.json

How: where the multi-view range exists it is used as it is. Everywhere else the depth model's range is multiplied by a smooth field: the Gaussian-weighted (sigma
pixels, wrapping round the azimuth) median-free mean of log(multi-view / model) over the anchors nearby, falling back to the global median ratio where there are
none. Multi-view is never changed by the model, only the other way round.
"""
import argparse, json, os, sys
import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from imutil import imread       # noqa: E402
import dollhouse as D           # noqa: E402


def wrap_blur(x, sigma):
    W = x.shape[1]; pad = int(4 * sigma)
    big = np.concatenate([x[:, -pad:], x, x[:, :pad]], 1)
    return cv2.GaussianBlur(big, (0, 0), sigma, borderType=cv2.BORDER_REPLICATE)[:, pad:pad + W]


def encode(r):
    return np.clip(np.round(255.0 * np.log(np.clip(r, D.DMIN, D.DMAX) / D.DMIN) / np.log(D.DMAX / D.DMIN)), 0, 255).astype(np.uint8)


def align_one(eq, mvs_range, sigma=40.0, min_anchors=150):
    r_da = D.decode_range(eq)
    a = (mvs_range > 0) & (r_da > 0)
    out = {"anchors": int(a.sum())}
    if a.sum() < min_anchors:
        out.update(mode="model only (too few anchors)", field_spread=None); return r_da, np.zeros(a.shape, bool), out
    L = np.zeros(a.shape, np.float32); L[a] = np.log(mvs_range[a] / r_da[a])
    med = float(np.median(L[a]))
    keep = a & (np.abs(L - med) < 0.7)                                              # a few wrong anchors must not drag the field
    w = keep.astype(np.float32)
    num = wrap_blur(L * w, sigma); den = wrap_blur(w, sigma)
    field = np.where(den > 0.02, num / np.maximum(den, 1e-6), med)
    field = np.clip(0.5 * field + 0.5 * wrap_blur(field.astype(np.float32), sigma), -0.7, 0.7)
    r = r_da * np.exp(field)
    r = np.where(keep, mvs_range, r)
    out.update(mode="anchored", global_ratio=round(float(np.exp(med)), 3), field_spread=round(float(np.exp(np.percentile(field, 95) - np.percentile(field, 5))), 3), anchors_used=int(keep.sum()))
    return r.astype(np.float32), keep, out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--frames", default="frames"); ap.add_argument("--sigma", type=float, default=40.0); ap.add_argument("--only", nargs="*")
    a = ap.parse_args()
    od = os.path.join(a.frames, "depth_mvs")
    try:
        skipped = json.load(open(os.path.join(a.frames, "mvs", "stats.json"))).get("skipped")
    except (OSError, ValueError):
        skipped = None
    if skipped:                                                      # stereo did not run: the depth model's own depth is used everywhere (dollhouse.py and build_tour.py fall back per stop)
        import shutil
        shutil.rmtree(od, ignore_errors=True); os.makedirs(od)
        json.dump(dict(skipped=skipped), open(os.path.join(od, "stats.json"), "w"), indent=1)
        print("depth not anchored:", skipped)
        return
    os.makedirs(od, exist_ok=True)
    stats = {}
    for f in sorted(os.listdir(os.path.join(a.frames, "mvs"))):
        if not f.endswith(".npz"):
            continue
        sid = f[:-4]
        if a.only and sid not in a.only:
            continue
        eq = imread(os.path.join(a.frames, "depth", f"eq_{sid}.png"))
        m = np.load(os.path.join(a.frames, "mvs", f))["range"].astype(np.float32)
        if m.shape != eq.shape[:2]:
            m = cv2.resize(m, (eq.shape[1], eq.shape[0]), interpolation=cv2.INTER_NEAREST)
        r, keep, st = align_one(eq, m, a.sigma)
        eq2 = eq.copy(); eq2[..., 2] = encode(r)
        cv2.imwrite(os.path.join(od, f"eq_{sid}.png"), eq2)
        cv2.imwrite(os.path.join(od, f"{sid}_anchor.png"), keep.astype(np.uint8) * 255)
        stats[sid] = st
        print(sid, st)
    json.dump(stats, open(os.path.join(od, "stats.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
