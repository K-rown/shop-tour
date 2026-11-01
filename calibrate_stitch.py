"""Fit the lens model by making the two lenses agree where they overlap.

Usage: python calibrate_stitch.py --src frames/src --out frames/calib.json
Needs pairs f_<t>.png / b_<t>.png (see PLAN.md for the ffmpeg line that makes them).
"""
import argparse, glob, os
import numpy as np
import cv2
from scipy.optimize import least_squares
import stitch_lib as sl

HALF = 0.5  # work at half resolution


def load_pairs(src):
    pairs = []
    for fp in sorted(glob.glob(os.path.join(src, "f_*.png"))):
        bp = os.path.join(src, "b_" + os.path.basename(fp)[2:])
        if not os.path.exists(bp):
            continue
        F = cv2.imread(fp, cv2.IMREAD_GRAYSCALE); A = cv2.imread(bp, cv2.IMREAD_GRAYSCALE)
        F = cv2.GaussianBlur(cv2.resize(F, None, fx=HALF, fy=HALF, interpolation=cv2.INTER_AREA), (0, 0), 1.0)
        A = cv2.GaussianBlur(cv2.resize(A, None, fx=HALF, fy=HALF, interpolation=cv2.INTER_AREA), (0, 0), 1.0)
        pairs.append((A.astype(np.float32), F.astype(np.float32)))
    return pairs


def band_rays(step=0.15):
    """Directions near the great circle where the two lens fields of view overlap (azimuth +-90)."""
    out = []
    for c in (90.0, -90.0):
        az = np.arange(c - 16, c + 16.01, step)
        el = np.arange(-72, 72.01, step)
        AZ, EL = np.meshgrid(az, el)
        out.append(sl.rays(AZ, EL))
    return out


def local_norm(x, valid):
    m = cv2.blur(x * valid, (31, 31)) / (cv2.blur(valid, (31, 31)) + 1e-6)
    d = (x - m) * valid
    s = np.sqrt(np.maximum(cv2.blur(d * d, (31, 31)), 0) / (cv2.blur(valid, (31, 31)) + 1e-6)) + 4.0
    return d / s


def residuals(x, pairs, bands, P0):
    P = dict(P0)
    for name, val in zip(sl.PARAM_NAMES, x):
        P[name] = val
    # the band geometry depends on the parameters only, not on the frame pair: compute it once per evaluation
    # (it used to be recomputed for every pair, which was ~3/4 of the 4 minutes this step took)
    geo = []
    for v in bands:
        uA, vA, thA = sl.project(sl.to_lens(v, "A", P), "A", P)
        uF, vF, thF = sl.project(sl.to_lens(v, "F", P), "F", P)
        geo.append((uA, vA, uF, vF, ((thA < np.radians(94)) & (thF < np.radians(94))).astype(np.float32)))
    out = []
    for A, F in pairs:
        for uA, vA, uF, vF, valid in geo:
            a = cv2.remap(A, uA, vA, cv2.INTER_LINEAR); f = cv2.remap(F, uF, vF, cv2.INTER_LINEAR)
            r = (local_norm(a, valid) - local_norm(f, valid)) * valid   # invalid pixels contribute 0, so the length is fixed
            # a penalty for shrinking the overlap, otherwise the fit can 'win' by making it disappear
            out.append(r[::3, ::3].ravel()); out.append(np.array([(1.0 - valid.mean()) * 40.0]))
    return np.concatenate(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="frames/src"); ap.add_argument("--out", default="frames/calib.json")
    a = ap.parse_args()
    pairs = load_pairs(a.src)
    if not pairs:
        raise SystemExit("no f_*.png / b_*.png pairs in " + a.src)
    bands = band_rays()
    size = int(round(pairs[0][0].shape[0] / HALF))            # lens frame size in pixels (3840 on the X5, 2880 on 5.7K cameras, ...)
    print(f"lens frames are {size}x{size}")
    P0 = sl.scaled(sl.default_params(size), HALF)
    x0 = np.array([P0[n] for n in sl.PARAM_NAMES])
    # scale each parameter so one unit ~ a small, comparable change
    scale = np.array([20, 20, 6, 6, 6, 6, 0.02, 0.02, 0.01, 0.01, 0.01])
    n0 = np.linalg.norm(residuals(x0, pairs, bands, P0))
    print("start residual %.1f  (%d pairs)" % (n0, len(pairs)))
    x = x0.copy()
    for rnd in range(int(os.environ.get("ROUNDS", 4))):      # 8 rounds only improve the residual by <1%; held-out score is the same from 5 rounds down to 3
        sol = least_squares(lambda z: residuals(x0 * 0 + x + z * scale, pairs, bands, P0), np.zeros(len(x)),
                            x_scale=1.0, diff_step=0.3, max_nfev=40, verbose=0)
        x = x + sol.x * scale
        print("round %d residual %.1f" % (rnd + 1, np.linalg.norm(sol.fun)))
    P = dict(P0)
    for name, val in zip(sl.PARAM_NAMES, x):
        P[name] = float(val)
    P = sl.scaled(P, 1.0 / HALF)          # back to full-resolution pixels
    P["size"] = float(size)
    sl.save_params(P, a.out)
    print("wrote", a.out)
    for k in sl.PARAM_NAMES:
        print("  %-4s %10.4f" % (k, P[k]))
    print("  fov A %.1f deg  fov F %.1f deg" % (np.degrees(size / 2 / P["fA"]) * 2, np.degrees(size / 2 / P["fF"]) * 2))


if __name__ == "__main__":
    main()
