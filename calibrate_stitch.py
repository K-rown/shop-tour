"""Fit the lens model by making the two lenses agree where they overlap.

Usage: python calibrate_stitch.py --src frames/src --out frames/calib.json
Needs pairs f_<t>.png / b_<t>.png (tour.py makes them; run it, or stops/calib, rather than building them by hand).
"""
import argparse, glob, os, threading
from concurrent.futures import ThreadPoolExecutor
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


_SIDE = {}                  # (lens, its parameters, the problem) -> that lens's band geometry and the pictures seen through it
_SIDE_LOCK = threading.Lock()
SIDE_KEEP = 40
LENS_PARAMS = {"A": ("fA", "cxA", "cyA", "kA"), "F": ("fF", "cxF", "cyF", "kF", "rx", "ry", "rz")}      # what each lens's geometry depends on (lens A is the pano's own frame)


def lens_side(which, P, bands, pairs):
    """For one lens: the pixel coordinates of every band, where the lens sees (theta under 94 degrees), and each pair's picture of that lens remapped into the band. They depend
    on this lens's own parameters only, so a finite-difference step of the fit that moves a parameter of the other lens (or the same lens's twin, for the three rotation
    angles) finds them made already: the same numbers, half the work of a Jacobian."""
    key = (which, tuple(P[n] for n in LENS_PARAMS[which]), id(bands), id(pairs))
    with _SIDE_LOCK:
        hit = _SIDE.get(key)
    if hit is not None and hit[2] is bands and hit[3] is pairs:          # (the entry holds the problem, so an id is never taken for another problem's)
        return hit[0], hit[1]
    geo = []
    for v in bands:
        u, vv, th = sl.project(sl.to_lens(v, which, P), which, P)
        geo.append((u, vv, th < np.radians(94)))
    col = 0 if which == "A" else 1
    rem = [[cv2.remap(pr[col], u, vv, cv2.INTER_LINEAR) for (u, vv, _) in geo] for pr in pairs]
    with _SIDE_LOCK:
        if len(_SIDE) >= SIDE_KEEP:
            _SIDE.pop(next(iter(_SIDE)))
        _SIDE[key] = (geo, rem, bands, pairs)
    return geo, rem


def residuals(x, pairs, bands, P0):
    P = dict(P0)
    for name, val in zip(sl.PARAM_NAMES, x):
        P[name] = val
    # the band geometry depends on the parameters only, not on the frame pair: compute it once per evaluation
    # (it used to be recomputed for every pair, which was ~3/4 of the 4 minutes this step took)
    gA, rA = lens_side("A", P, bands, pairs); gF, rF = lens_side("F", P, bands, pairs)
    valid = [(a[2] & f[2]).astype(np.float32) for a, f in zip(gA, gF)]
    out = []
    for pi in range(len(pairs)):
        for bi in range(len(bands)):
            a, f, vd = rA[pi][bi], rF[pi][bi], valid[bi]
            r = (local_norm(a, vd) - local_norm(f, vd)) * vd   # invalid pixels contribute 0, so the length is fixed
            # a penalty for shrinking the overlap, otherwise the fit can 'win' by making it disappear
            out.append(r[::3, ::3].ravel()); out.append(np.array([(1.0 - vd.mean()) * 40.0]))
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
    import inspect
    kw = {}
    pool = None
    if "workers" in inspect.signature(least_squares).parameters:      # scipy 1.16+: the Jacobian's columns are independent evaluations; cv2 releases the GIL, so they run beside each other
        pool = ThreadPoolExecutor(max_workers=min(len(x), os.cpu_count() or 1))
        kw["workers"] = pool.map                                      # (the same differences, bit for bit, as the serial Jacobian: only who computes which column changes)
    for rnd in range(int(os.environ.get("ROUNDS", 4))):      # 8 rounds only improve the residual by <1%; held-out score is the same from 5 rounds down to 3
        sol = least_squares(lambda z: residuals(x0 * 0 + x + z * scale, pairs, bands, P0), np.zeros(len(x)),
                            x_scale=1.0, diff_step=0.3, max_nfev=40, verbose=0, **kw)
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
