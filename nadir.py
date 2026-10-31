"""Fill the bottom of a panorama (the stick and the person holding it) with plausible floor.

The cap used to be a dark disc. Here the region under the camera is rebuilt from the floor around it:
  1. the person holding the stick is found per stop by the segmentation model (floor_panos.py, straight-down view); the fixed
     body-shaped region below is only the fallback when no mask is available,
  2. the bottom 50 degrees of the pano are unwrapped into a top-down disc (floor is continuous there, objects run
     radially, unlike in the stretched equirect rows), filled with the average of the floor-labelled pixels around the hole,
  3. the fill is mapped back, given fine grain matched to the surrounding floor, and feathered into the real pixels.
Used by build_tour.py (--nadir-mode fill). Pure OpenCV (Apache-2.0).
"""
import os
import cv2
import numpy as np

FILL_MODE = os.environ.get("NADIR_FILL", "harmonic")    # "blur": smooth floor average only; "harmonic": the same, bent to meet the floor all round the edge

RADIUS_DEG = 50.0       # the disc covers elevations -90 .. -40
# person + stick outline in pano fractions (x, y), measured on the walk (head ~x .47, arms out to ~.31/.63)
BODY = [(0.405, 0.790), (0.545, 0.790), (0.600, 0.830), (0.640, 0.885), (0.685, 0.945), (0.700, 1.0),
        (0.300, 1.0), (0.315, 0.945), (0.345, 0.885), (0.375, 0.830)]
POLE_Y = 0.965          # the last rows are one stretched smear across the whole width


def body_mask(h, w, grow=0):
    m = np.zeros((h, w), np.uint8)
    cv2.fillPoly(m, [np.array([(int(x * w), int(y * h)) for x, y in BODY], np.int32)], 255)
    m[int(h * POLE_Y):, :] = 255
    if grow:
        m = cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * grow + 1, 2 * grow + 1)))
    return m


def _maps(h, w, S):
    """Disc pixel -> equirect pixel, and equirect rows (nadir cap) -> disc pixel."""
    y, x = np.mgrid[0:S, 0:S].astype(np.float32)
    dx, dy = (x + 0.5 - S / 2) / (S / 2), (y + 0.5 - S / 2) / (S / 2)
    rho = np.hypot(dx, dy)
    az = np.arctan2(dx, -dy)                                   # disc "up" = azimuth 0
    el = np.radians(-90.0 + rho * RADIUS_DEG)
    u = ((az / (2 * np.pi) + 0.5) % 1.0) * w - 0.5
    v = (0.5 - el / np.pi) * h - 0.5
    return u.astype(np.float32), v.astype(np.float32), rho


NADIR_FOV = 130.0       # the straight-down view floor_panos.py cuts (pitch -90): covers elevation -90 .. -25 in every direction


def view_to_equirect(mask_view, h, w, fov=NADIR_FOV):
    """Straight-down view mask (n x n, 255 = set) -> equirect mask (h x w). Rows above elevation -25 are 0."""
    n = mask_view.shape[0]
    f = n / 2 / np.tan(np.radians(fov) / 2)
    az = ((np.arange(w) + 0.5) / w * 2 * np.pi - np.pi).astype(np.float32)
    el = (np.pi / 2 - (np.arange(h) + 0.5) / h * np.pi).astype(np.float32)
    AZ, EL = np.meshgrid(az, el)
    x, y, z = np.cos(EL) * np.sin(AZ), np.sin(EL), np.cos(EL) * np.cos(AZ)
    # camera looks straight down: view x = pano x, view y = pano z (forward), view depth = -pano y
    dz = -y
    ok = dz > 0.05
    col = (f * x / np.where(ok, dz, 1) + n / 2 - 0.5).astype(np.float32)
    row = (n / 2 - 0.5 - f * z / np.where(ok, dz, 1)).astype(np.float32)
    out = cv2.remap(mask_view, col, row, cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    out[~ok] = 0
    return out


def central_person(mask_view, touch=0.25):
    """Keep only person regions that come within `touch` of the view radius from the middle of the straight-down view (the person
    holding the stick is right under the camera: on 24 stops the nearest pixel was 0-0.15). The model also calls orange cable
    spools, helmets and red tool racks 'person'; the nearest of those starts at 0.45 or more (0.5 let a rack at a very close stop through)."""
    m = (mask_view > 127).astype(np.uint8)
    n = m.shape[0]
    cnt, lab = cv2.connectedComponents(m)
    yy, xx = np.mgrid[0:n, 0:n]
    near = np.hypot(xx - n / 2, yy - n / 2) < touch * n / 2
    keep = np.unique(lab[near & (m > 0)])
    keep = keep[keep > 0]
    return (np.isin(lab, keep) * 255).astype(np.uint8)


def person_hole(person_view, h, w, grow_deg=1.6):
    """Hole = the person (grown a little: hair, hands, soft edges) plus the pole rows. Returns None if there is no person."""
    pv = central_person(person_view)
    if (pv > 0).sum() < 40:
        return None
    m = view_to_equirect(pv, h, w)
    k = max(3, int(round(w * grow_deg / 360)) * 2 + 1)
    m = cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    m[int(h * POLE_Y):, :] = 255
    return m


def harmonic_correction(res, disc, hole, ok_s, fill_all, band=13, iters=600):
    """Make the patch meet the real floor all round its edge, without inventing anything inside.
    The smooth fill is an average of nearby floor, so at the edge it can be a little lighter, darker or tinted compared with the real floor
    next to it (a visible outline). Where the ring around the hole is floor, measure that mismatch (real floor, lightly blurred, minus the smooth
    estimate at the same spot) and spread it over the hole by relaxation (Laplace, solved at quarter size). Where the ring is a cable or a
    rack, no mismatch is assumed there. Zero detail is added: only a slowly varying correction that equals the mismatch at the edge.
    fill_all: the smooth estimate over the whole disc (floor_fill computes it everywhere, not only inside the hole)."""
    S = disc.shape[0]; q = S // 4
    hole_q = (cv2.resize(hole, (q, q), interpolation=cv2.INTER_AREA) > 40).astype(np.uint8)
    if hole_q.sum() < 20:
        return res
    ring_q = (cv2.dilate(hole_q, np.ones((band, band), np.uint8)) > 0) & (hole_q == 0)
    lab_real = cv2.cvtColor(cv2.resize(cv2.GaussianBlur(disc, (0, 0), 3), (q, q), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2LAB).astype(np.float32)
    lab_est = cv2.cvtColor(cv2.resize(fill_all, (q, q), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2LAB).astype(np.float32)
    okq = cv2.resize(ok_s.astype(np.uint8) * 255, (q, q), interpolation=cv2.INTER_AREA) > 127
    known = ring_q & okq
    if known.sum() < 10:
        return res
    delta = np.zeros_like(lab_est)
    delta[known] = np.clip((lab_real - lab_est)[known], -25, 25)          # a floor-like ring never differs by more than this; guards against a mislabelled pixel
    free = ((hole_q > 0) | (ring_q & ~known)).astype(np.float32)
    kern = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], np.float32) / 4
    d = delta.copy()
    d[free > 0] = delta[known].mean(axis=0)                              # start from the average mismatch: far from the edge the correction then settles at that average, not at zero
    for _ in range(iters):                                               # known ring pixels stay put, everything else takes the mean of its neighbours
        nb = np.stack([cv2.filter2D(d[..., c], -1, kern, borderType=cv2.BORDER_REFLECT) for c in range(3)], -1)
        d = np.where(free[..., None] > 0, nb, d)
    d = cv2.GaussianBlur(d, (0, 0), 2)
    dfull = cv2.resize(d, (S, S), interpolation=cv2.INTER_CUBIC)
    lab = cv2.cvtColor(res, cv2.COLOR_BGR2LAB).astype(np.float32)
    m = hole > 0
    lab[m] = lab[m] + dfull[m]
    out = res.copy()
    out[m] = cv2.cvtColor(np.clip(lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)[m]
    return out


def seam_score(disc, filled, hole, band=7):
    """How visible is the outline of the patch? Mean colour difference (CIE76 in LAB) between a thin band just inside the hole and a thin
    band just outside it, both lightly blurred along the edge. Lower = the patch meets the floor better. None if there is no hole."""
    h = (hole > 0).astype(np.uint8)
    if h.sum() < 50:
        return None
    k = np.ones((2 * band + 1,) * 2, np.uint8)
    inn = (h > 0) & (cv2.erode(h, k) == 0)
    out = (cv2.dilate(h, k) > 0) & (h == 0)
    A = cv2.cvtColor(cv2.GaussianBlur(filled, (0, 0), 3), cv2.COLOR_BGR2LAB).astype(np.float32)
    B = cv2.cvtColor(cv2.GaussianBlur(disc, (0, 0), 3), cv2.COLOR_BGR2LAB).astype(np.float32)
    # compare each inner-band pixel with the mean of the outer band pixels within 2*band of it
    wi, wo = cv2.GaussianBlur(inn.astype(np.float32), (0, 0), band * 2), cv2.GaussianBlur(out.astype(np.float32), (0, 0), band * 2)
    ok = inn & (wo > 1e-4)
    if not ok.any():
        return None
    mo = np.stack([cv2.GaussianBlur(B[..., c] * out, (0, 0), band * 2) for c in range(3)], -1) / np.maximum(wo[..., None], 1e-6)
    return float(np.linalg.norm((A - mo)[ok], axis=1).mean())


def floor_fill(disc, hole, radius, floor_disc=None):
    """Fill the hole with a smooth average of the *floor-like* pixels around it (grey, floor brightness), so that
    saturated or dark objects next to the hole (cables, machines) cannot bleed into the patch the way inpainting lets them.
    The smooth estimate is computed at quarter size (it is smooth anyway) and scaled back up."""
    S = disc.shape[0]; q = S // 3
    d = cv2.resize(disc, (q, q), interpolation=cv2.INTER_AREA)
    hl = (cv2.resize(hole, (q, q), interpolation=cv2.INTER_AREA) > 40).astype(np.uint8) * 255
    lab = cv2.cvtColor(d, cv2.COLOR_BGR2LAB).astype(np.float32)
    ring = (cv2.dilate(hl, np.ones((31, 31), np.uint8)) > 0) & (hl == 0)
    ref = np.median(lab[ring], axis=0) if ring.any() else np.array([128.0, 128.0, 128.0], np.float32)     # empty ring (hole fills the view): mid grey
    fd = None
    if floor_disc is not None:
        fd = (cv2.resize(floor_disc, (q, q), interpolation=cv2.INTER_AREA) > 127) & (hl == 0)
    if fd is not None and fd.sum() > 150:                    # the floor model says which pixels around the hole are floor
        ref = np.median(lab[fd], axis=0)
        ok = fd & (np.abs(lab[..., 0] - ref[0]) < 55)
    else:                                                    # no floor mask: take grey, floor-bright pixels around the hole
        chroma = np.hypot(lab[..., 1] - ref[1], lab[..., 2] - ref[2])
        ok = (hl == 0) & (chroma < 9) & (np.abs(lab[..., 0] - ref[0]) < 38)
    ok_raw = ok.copy()                                       # floor-like pixels right up to the hole (harmonic_correction needs the ring next to the edge)
    ok = cv2.erode(ok.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(np.float32)   # keep away from object edges
    out = np.zeros_like(lab); wsum = np.zeros(lab.shape[:2], np.float32)
    for sig in (q / 10, q / 4, q / 1.5):                                                  # near floor first, then wider
        num = cv2.GaussianBlur(lab * ok[..., None], (0, 0), sig)
        den = cv2.GaussianBlur(ok, (0, 0), sig)
        est = num / np.maximum(den[..., None], 1e-6)
        take = (wsum < 0.02) & (den > 0.02)
        out[take] = est[take]; wsum[take] = den[take]
    fill = cv2.cvtColor(np.clip(out, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)
    fill = cv2.resize(fill, (S, S), interpolation=cv2.INTER_CUBIC)
    have = cv2.resize((wsum > 0.02).astype(np.uint8) * 255, (S, S), interpolation=cv2.INTER_NEAREST)
    res = disc.copy()
    flat = np.full_like(disc, np.clip(cv2.cvtColor(np.clip(ref, 0, 255).astype(np.uint8)[None, None], cv2.COLOR_LAB2BGR)[0, 0], 0, 255))
    res[hole > 0] = flat[hole > 0]                           # nothing floor-like nearby: one plain floor colour (inpainting made spokes)
    m = (hole > 0) & (have > 0)
    res[m] = fill[m]
    floor_fill.last_ok = cv2.resize(ok_raw.astype(np.uint8) * 255, (S, S), interpolation=cv2.INTER_NEAREST)
    if FILL_MODE == "harmonic":
        res = harmonic_correction(res, disc, hole, floor_fill.last_ok > 0, np.where(have[..., None] > 0, fill, disc))
    return res


def fill_nadir(img, lo_w=2048, S=768, radius=5, person=None, floor=None):
    """person / floor: straight-down view masks from floor_panos.py (frames/depth/<id>_n.png, _nf.png), or None."""
    h, w = img.shape[:2]
    lh = lo_w // 2
    small = cv2.resize(img, (lo_w, lh), interpolation=cv2.INTER_AREA)
    mask = person_hole(person, lh, lo_w) if person is not None else None
    fill_nadir.last_mask = "person" if mask is not None else "fixed"
    if mask is None and person is not None:
        # the model looked and found nobody under the camera (an export with the stick and person already removed, or a clean frame):
        # only the stretched pole rows need filling, the floor around must not be painted over
        mask = np.zeros((lh, lo_w), np.uint8); mask[int(lh * POLE_Y):, :] = 255
        fill_nadir.last_mask = "nobody"
    elif mask is None:
        mask = body_mask(lh, lo_w, grow=5)                   # no per-stop mask file: the fixed outline measured on the first stops
    floor_eq = view_to_equirect(floor, lh, lo_w) if floor is not None else None
    u, v, rho = _maps(lh, lo_w, S)
    disc = cv2.remap(small, u, v, cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
    dmask = cv2.remap(mask, u, v, cv2.INTER_NEAREST, borderMode=cv2.BORDER_WRAP)
    dmask[rho > 1.0] = 0
    fdisc = cv2.remap(floor_eq, u, v, cv2.INTER_NEAREST, borderMode=cv2.BORDER_WRAP) if floor_eq is not None else None
    filled = floor_fill(disc, dmask, radius, fdisc)
    fill_nadir.last_seam = seam_score(disc, filled, dmask)
    # back to equirect: for each pano pixel in the cap, where is it in the disc?
    ys, xs = np.mgrid[int(lh * (1 - RADIUS_DEG / 180)) - 1:lh, 0:lo_w].astype(np.float32)
    az = (xs + 0.5) / lo_w * 2 * np.pi - np.pi
    el = np.pi / 2 - (ys + 0.5) / lh * np.pi
    rr = (np.degrees(el) + 90.0) / RADIUS_DEG
    bx = (np.sin(az) * rr * (S / 2) + S / 2 - 0.5).astype(np.float32)
    by = (-np.cos(az) * rr * (S / 2) + S / 2 - 0.5).astype(np.float32)
    cap = cv2.remap(filled, bx, by, cv2.INTER_LINEAR)
    fill_lo = small.copy(); fill_lo[int(lh * (1 - RADIUS_DEG / 180)) - 1:] = cap
    # everything below happens only in the bottom band (rows >= y0), the rest of the pano is left untouched
    y0 = int(h * (1 - RADIUS_DEG / 180)) - 2
    band = img[y0:]
    bh = h - y0
    big = cv2.resize(fill_lo[int(lh * y0 / h):], (w, bh), interpolation=cv2.INTER_CUBIC).astype(np.float32)
    full_m = cv2.resize(mask[int(lh * y0 / h):], (w, bh), interpolation=cv2.INTER_LINEAR)
    # fine grain from the real floor just outside the hole, so the patch is not glassy-smooth
    ring = (cv2.dilate(full_m, np.ones((41, 41), np.uint8)) > 0) & (full_m == 0)
    detail = band.astype(np.float32) - cv2.GaussianBlur(band, (0, 0), 4).astype(np.float32)
    sd = float(detail[ring].std()) if ring.any() else 0.0
    rng = np.random.default_rng(7)
    grain = cv2.GaussianBlur(rng.standard_normal((bh, w)).astype(np.float32), (0, 0), 0.9)
    grain *= sd / (grain.std() + 1e-6)
    big = np.clip(big + grain[..., None], 0, 255)
    alpha = cv2.GaussianBlur(full_m.astype(np.float32) / 255, (0, 0), max(2.0, h * 0.005))[..., None]
    alpha = np.maximum(alpha, (full_m > 127)[..., None])
    out = img.copy()
    out[y0:] = (band.astype(np.float32) * (1 - alpha) + big * alpha).astype(np.uint8)
    return out
