"""Dual-fisheye -> equirectangular stitching with a fitted lens model and a feathered seam.

Conventions (identical to the ffmpeg panos this project already uses, so nothing downstream shifts):
  pano centre (azimuth 0) = optical axis of lens A, the second video stream (b_*.png)
  azimuth 180            = optical axis of lens F, the first video stream (f_*.png)
Lens model, equidistant with one cubic term:  r = f * (theta + k * theta^3), theta = angle from the lens axis.
"""
import json
import numpy as np
import cv2

PARAM_NAMES = ["fA", "fF", "cxA", "cyA", "cxF", "cyF", "kA", "kF", "rx", "ry", "rz"]
THETA_MAX = np.radians(97.0)   # beyond this the fisheye image is vignetted / unusable


def rodrigues(rv):
    R, _ = cv2.Rodrigues(np.asarray(rv, float).reshape(3, 1))
    return R


def rays(az_deg, el_deg):
    az, el = np.radians(az_deg), np.radians(el_deg)
    return np.stack([np.cos(el) * np.sin(az), np.sin(el), np.cos(el) * np.cos(az)], -1)


def to_lens(v, which, P):
    """Pano ray (x right, y up, z forward) -> lens frame (z = optical axis, x right, y up in the lens image)."""
    if which == "A":
        return v
    # lens F looks backwards: turn 180 deg about the vertical, then apply its small calibration rotation
    w = v * np.array([-1.0, 1.0, -1.0])
    return w @ rodrigues([P["rx"], P["ry"], P["rz"]]).T


def project(vl, which, P):
    """Lens-frame rays -> pixel coords (u, v) in that lens image, and theta."""
    f, cx, cy, k = P["f" + which], P["cx" + which], P["cy" + which], P["k" + which]
    th = np.arccos(np.clip(vl[..., 2], -1, 1))
    ph = np.arctan2(vl[..., 1], vl[..., 0])
    r = f * (th + k * th ** 3)
    return (cx + r * np.cos(ph)).astype(np.float32), (cy - r * np.sin(ph)).astype(np.float32), th


EDGE_PX = 48.0   # fade a lens out over this many pixels before the edge of its square frame


def edge_weight(u, v, P):
    """1 well inside the lens frame, 0 at/after the frame edge (the image circle is clipped top and bottom)."""
    size = P.get("size", 2 * P["cxA"])
    d = np.minimum(np.minimum(u, v), np.minimum(size - 1 - u, size - 1 - v))
    t = np.clip(d / EDGE_PX, 0, 1)
    return (t * t * (3 - 2 * t)).astype(np.float32)


def default_params(size=3840):
    f = (size / 2) / np.radians(97.5)
    return dict(fA=f, fF=f, cxA=size / 2, cyA=size / 2, cxF=size / 2, cyF=size / 2, kA=0.0, kF=0.0, rx=0.0, ry=0.0, rz=0.0)


def scaled(P, s):
    Q = dict(P)
    for key in ("fA", "fF", "cxA", "cyA", "cxF", "cyF"):
        Q[key] = P[key] * s
    return Q


def build_maps(P, out_w=7680, feather_deg=14.0, rows=256):
    """Everything that depends only on the lens model, computed once and reused for every frame.
    Done in row chunks in float32 so an 8K pano does not need gigabytes of RAM."""
    out_h = out_w // 2
    az = ((np.arange(out_w) + 0.5) / out_w * 360.0 - 180.0).astype(np.float32)
    M = {k: np.empty((out_h, out_w), np.float32) for k in ("uA", "vA", "uF", "vF", "wA", "wF")}
    ov = np.empty((out_h, out_w), bool)
    lo = THETA_MAX - np.radians(feather_deg)

    def weight(th):
        t = np.clip((THETA_MAX - th) / (THETA_MAX - lo), 0, 1)
        return t * t * (3 - 2 * t)

    for r0 in range(0, out_h, rows):
        r1 = min(out_h, r0 + rows)
        el = (90.0 - (np.arange(r0, r1) + 0.5) / out_h * 180.0).astype(np.float32)
        AZ, EL = np.meshgrid(az, el)
        v = rays(AZ, EL).astype(np.float32)
        uA, vA, thA = project(to_lens(v, "A", P), "A", P)
        uF, vF, thF = project(to_lens(v, "F", P), "F", P)
        eA, eF = edge_weight(uA, vA, P), edge_weight(uF, vF, P)
        okA, okF = (thA < THETA_MAX) & (eA > 0), (thF < THETA_MAX) & (eF > 0)
        wA, wF = weight(thA) * eA, weight(thF) * eF
        wA = np.where(~okF, eA, wA); wF = np.where(~okA, eF, wF)
        tot = wA + wF + 1e-6
        M["uA"][r0:r1], M["vA"][r0:r1], M["uF"][r0:r1], M["vF"][r0:r1] = uA, vA, uF, vF
        M["wA"][r0:r1], M["wF"][r0:r1] = wA / tot, wF / tot
        ov[r0:r1] = okA & okF
    M["ov"] = ov
    return M


def stitch_maps(imgA, imgF, M, gain=True):
    """Blend both lenses with precomputed maps. imgA/imgF: BGR uint8, square. Returns BGR uint8."""
    sA = cv2.remap(imgA, M["uA"], M["vA"], cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT)
    sF = cv2.remap(imgF, M["uF"], M["vF"], cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT)
    ov = M["ov"]
    g = np.ones(3, np.float32)
    if gain and ov.sum() > 1000:
        sub = ov[::4, ::4]
        a, f = sA[::4, ::4][sub].astype(np.float32), sF[::4, ::4][sub].astype(np.float32)
        g = np.clip((a.mean(0) + 1e-3) / (f.mean(0) + 1e-3), 0.7, 1.45).astype(np.float32)
    out = np.empty_like(sA)
    for c in range(3):  # per channel keeps peak memory low
        out[..., c] = np.clip(sA[..., c] * M["wA"] + sF[..., c] * g[c] * M["wF"], 0, 255)
    return out


def stitch(imgA, imgF, P, out_w=7680, gain=True, feather_deg=14.0):
    """One-off stitch (builds the maps every call; use build_maps + stitch_maps for many frames)."""
    return stitch_maps(imgA, imgF, build_maps(P, out_w, feather_deg), gain)


def save_params(P, path):
    json.dump({k: float(v) for k, v in P.items()}, open(path, "w"), indent=1)


def load_params(path):
    return {k: float(v) for k, v in json.load(open(path)).items()}


# ---------------------------------------------------------------------------------------------
# Optical-flow seam: the two lenses sit ~3 cm apart, so close things (ceiling ducts, people)
# land in slightly different places in each lens. Plain blending makes a ghost. Instead we
# find, in the overlap band, how far each feature moved between the lenses and slide both
# halves toward each other across the band, so the edges meet. (What Insta360 calls flow stitching.)
# The band is resampled in lens-A polar coords (phi around the lens axis, theta 83..97 deg),
# where the seam is a straight horizontal strip with no pole distortion.
# ---------------------------------------------------------------------------------------------
BAND_LO = THETA_MAX - np.radians(14.0)


def build_flow_maps(P, out_w=7680, feather_deg=14.0, strip_px_per_deg=None):
    M = build_maps(P, out_w, feather_deg)
    ppd = strip_px_per_deg or out_w / 360.0
    lo = THETA_MAX - np.radians(feather_deg)
    SW = int(round(360 * ppd)); SH = int(round(np.degrees(THETA_MAX - lo) * 2 * ppd)) | 1
    # strip covers theta_A in [90 - span, 90 + span] so both feather ramps fit
    span = np.degrees(THETA_MAX) - 90.0 + np.degrees(THETA_MAX - lo)
    phi = (np.arange(SW) + 0.5) / SW * 2 * np.pi
    th = np.radians(90.0 - span + (np.arange(SH) + 0.5) / SH * 2 * span)
    PH, TH = np.meshgrid(phi, th)
    v = np.stack([np.sin(TH) * np.cos(PH), np.sin(TH) * np.sin(PH), np.cos(TH)], -1).astype(np.float32)
    S = {}
    S["uA"], S["vA"], thA = project(to_lens(v, "A", P), "A", P)
    S["uF"], S["vF"], thF = project(to_lens(v, "F", P), "F", P)

    def weight(t):
        t = np.clip((THETA_MAX - t) / (THETA_MAX - lo), 0, 1)
        return t * t * (3 - 2 * t)
    wA, wF = weight(thA), weight(thF)
    wA = np.where(thF >= THETA_MAX, 1.0, wA); wF = np.where(thA >= THETA_MAX, 1.0, wF)
    S["eA"], S["eF"] = edge_weight(S["uA"], S["vA"], P), edge_weight(S["uF"], S["vF"], P)
    S["wA"] = (wA / (wA + wF + 1e-6)).astype(np.float32)
    S["span"], S["SW"], S["SH"] = span, SW, SH
    # where each overlap pano pixel sits in the strip
    out_h = out_w // 2
    ov = M["ov"]
    ys, xs = np.nonzero(ov)
    az = np.radians((xs + 0.5) / out_w * 360.0 - 180.0); el = np.radians(90.0 - (ys + 0.5) / out_h * 180.0)
    x, y, z = np.cos(el) * np.sin(az), np.sin(el), np.cos(el) * np.cos(az)
    p = np.mod(np.arctan2(y, x), 2 * np.pi); t = np.degrees(np.arccos(np.clip(z, -1, 1)))
    S["ov_idx"] = (ys, xs)
    S["ov_sx"] = (p / (2 * np.pi) * SW - 0.5).astype(np.float32)
    S["ov_sy"] = ((t - (90.0 - span)) / (2 * span) * SH - 0.5).astype(np.float32)
    M["strip"] = S
    return M


def _wrap_pad(img, pad):
    return np.concatenate([img[:, -pad:], img, img[:, :pad]], 1)


def _seam_rows(cost, lo, hi, step=2, smooth=0.15):
    """Cheapest left-to-right path through the strip (one row per column), confined to rows lo..hi."""
    H, W = cost.shape
    c = cost.copy(); c[:lo] = 1e6; c[hi:] = 1e6
    acc = c[:, 0].copy(); back = np.zeros((H, W), np.int16)
    rows = np.arange(H)
    for x in range(1, W):
        best = np.full(H, np.inf); arg = np.zeros(H, np.int16)
        for d in range(-step, step + 1):
            sh = np.roll(acc, d) + smooth * abs(d)
            if d > 0: sh[:d] = np.inf
            if d < 0: sh[d:] = np.inf
            m = sh < best; best[m] = sh[m]; arg[m] = (rows - d)[m]
        acc = best + c[:, x]; back[:, x] = arg
    path = np.zeros(W, np.int32); path[-1] = int(np.argmin(acc))
    for x in range(W - 1, 0, -1):
        path[x - 1] = back[path[x], x]
    return path


def stitch_seam(imgA, imgF, M, gain=True, blend_deg=1.5):
    """Cut along the line where the two lenses agree best, with only a narrow blend, so close objects
    (ceiling ducts) are taken from one lens instead of ghosted from both."""
    out = stitch_maps(imgA, imgF, M, gain)
    S = M["strip"]
    a = cv2.remap(imgA, S["uA"], S["vA"], cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)
    f = cv2.remap(imgF, S["uF"], S["vF"], cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)
    if gain:
        mid = slice(S["SH"] // 3, 2 * S["SH"] // 3)
        g = np.clip((a[mid].reshape(-1, 3).mean(0) + 1e-3) / (f[mid].reshape(-1, 3).mean(0) + 1e-3), 0.7, 1.45)
        f = np.clip(f.astype(np.float32) * g, 0, 255).astype(np.uint8)
    H, W = a.shape[:2]
    # cost on a quarter-size strip: colour difference plus edge difference, blurred a little
    q = 4
    sa = cv2.resize(a, (W // q, H // q), interpolation=cv2.INTER_AREA).astype(np.float32)
    sf = cv2.resize(f, (W // q, H // q), interpolation=cv2.INTER_AREA).astype(np.float32)
    cost = np.abs(sa - sf).sum(2)
    ga = cv2.Sobel(cv2.cvtColor(sa, cv2.COLOR_BGR2GRAY), cv2.CV_32F, 1, 1); gf = cv2.Sobel(cv2.cvtColor(sf, cv2.COLOR_BGR2GRAY), cv2.CV_32F, 1, 1)
    cost = cv2.GaussianBlur(cost + 0.5 * np.abs(ga - gf), (0, 0), 1.0)
    eAq = cv2.resize(S["eA"], (W // q, H // q), interpolation=cv2.INTER_AREA)
    eFq = cv2.resize(S["eF"], (W // q, H // q), interpolation=cv2.INTER_AREA)
    cost = cost + 2000.0 * ((eAq < 0.99) | (eFq < 0.99))      # never cut where a lens has no picture
    # stay where both lenses still have good pixels (theta_A between ~85 and ~95 deg)
    span, Hq = S["span"], cost.shape[0]
    row_of = lambda deg: int(round((deg - (90.0 - span)) / (2 * span) * Hq))
    path = _seam_rows(cost, row_of(85.0), row_of(95.0))
    seam = cv2.resize(path.astype(np.float32).reshape(1, -1) * q + q / 2, (W, 1), interpolation=cv2.INTER_LINEAR)[0]
    seam = cv2.GaussianBlur(seam.reshape(1, -1), (0, 0), 6)[0]
    half = blend_deg * W / 360.0
    yy = np.arange(H, dtype=np.float32)[:, None]
    t = np.clip((seam[None, :] + half - yy) / (2 * half), 0, 1)       # 1 above the seam (lens A side), 0 below
    t = t * t * (3 - 2 * t)
    wA = (t * S["eA"] / (t * S["eA"] + (1 - t) * S["eF"] + 1e-6)).astype(np.float32)
    strip = np.clip(a * wA[..., None] + f * (1 - wA[..., None]), 0, 255).astype(np.uint8)
    strip = _wrap_pad(strip, 4)
    n = len(S["ov_sx"]); C = 8192; r = -(-n // C)
    sx = np.zeros(r * C, np.float32); sy = np.zeros(r * C, np.float32)
    sx[:n] = S["ov_sx"] + 4; sy[:n] = S["ov_sy"]
    vals = cv2.remap(strip, sx.reshape(r, C), sy.reshape(r, C), cv2.INTER_LINEAR).reshape(-1, 3)[:n]
    ys, xs = S["ov_idx"]
    out[ys, xs] = vals
    return out
