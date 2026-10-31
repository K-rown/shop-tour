"""Paint out the OTHER people in every stop (ON by default since 2026-10-22; SHOP_TOUR_REMOVE_OTHERS=0 turns it off) and let the privacy gate look again.

Usage:  python remove_others.py [--frames frames] [--only 003 004] [--debug]
Runs after remove_capturer.py (which paints out the person holding the stick) and before privacy_gate.py. Reads frames/clean/<id>.jpg, frames/points/<id>.jpg (the
neighbours' ORIGINAL pictures), frames/up.json, frames/hops.json; writes frames/clean/<id>.jpg again (the people painted out) and frames/capturer/others.json.
On by default (SHOP_TOUR_REMOVE_OTHERS=0 turns it off): painting a person out is the safe direction. With the switch off nothing here runs and no file changes.

For every person or face the gate's own scan (privacy_gate.scan + nadir_persons, the same detector, the same bars) finds in a stop:
  1. a rectilinear view is cut round him (wide enough to hold him with room), the detector's box found again in it (a box from the scan's angles when it is not),
     SAM 2 is prompted with the box (Models.segment_box), the mask is dilated and his shadow on the ground added;
  2. the hole is filled from the NEIGHBOURING stops: each pixel's ray is put on the ground plane (the stick height below the horizon; a far plane above and
     beyond it), looked at again from each neighbour (hops.json: azimuth, distance, turn - tried at a few distances and headings, kept where the neighbour's picture
     lines up with this stop's own on a ring round the hole), with the neighbour's own people and its own capturer left out; best-aligned neighbour first, colour-matched,
     Poisson-blended at the edge. What no neighbour saw is filled from the colours round it, with the grain of the picture put back (capturer.soft_fill);
  3. the gate looks again (privacy_chain.py / privacy_gate.py: a stop where a person or face is still found stays HELD - nothing with a person in it ships);
     this module also looks once itself and paints what is left a second time (ROUNDS).
A poster of a face is a face to this: it is painted out the same way (a bus-shelter ad is not a person, but it is a photograph of one).
Nothing reads vio or the map: the neighbours come from privacy_inputs.py (own mode).
"""
import argparse, json, math, os, threading, time, collections
import cv2
import numpy as np
import capturer as C
import privacy_gate as PG
import privacy_inputs as PI
import synth_fill as SF
from imutil import imread, imwrite

HERE = os.path.dirname(os.path.abspath(__file__))
PX_PER_DEG = 7680 / 360.0         # the pano's own resolution (the views are cut at about this)
ROUNDS = 4                        # scan -> paint -> scan again: a leftover found after the first painting is painted in a second round
DMAX = 60.0                       # the plane beyond which a ray is "far" (the sky and the distant street are looked at from the neighbour in the same direction)
MAX_NBRS = 6
NCC_MIN = 0.3                     # a neighbour whose picture lines up with this stop's own on the ring at least this well may supply pixels
SINGLE_NCC = 0.65                 # where no two neighbours agree, one that lines up this well on the ring may supply whole patches (0.5 left a dark speckled smudge from a neighbour that did not line up; what it could not supply is made by synth_fill.py)
STRATA = (8.0, 0.25)              # synth_fill's weights for copying from far above/below the hole and from far to either side of it
NBR_MIN_DEG = 8.0                 # a person smaller than this across (more than about 6 m off) is not filled from the neighbours: a ground-plane look from a stop a few metres away cannot tell him from the scene behind him,
                                  # and the neighbours' pictures often hold the same far person (his vest came back as a ghost); the picture round him makes the fill (synth_fill.py)
SYNTH_MIN_PX = 150                # a part of the hole no neighbour saw is made by patch synthesis (synth_fill.py) when it is at least this many pixels of the view; a smaller one by the old fill
SHADOW_DARK = 0.78                # ground pixels this much darker than the ground round him, touching him, are his shadow
MIN_N, MAX_N = 512, 1536
# Where the painted-out picture still holds a person-sized lump under the camera (the gate's check C, the same one, at every camera height the margin tries), the capturer's own hole is made
# again, from the picture round it, by patch synthesis (synth_fill.py): up to REFILL_TRIES ways, the first that leaves no lump wins; none: the picture stays as it was and the gate holds it.
REFILL_METHODS = (("synth", {}), ("synth-free", dict(local=0.7, patch=9, seed=1)), ("synth-near", dict(local=6.0, patch=5, seed=2)))
REFILL_TRIES = len(REFILL_METHODS)


def enabled():
    """The paint-out of the other people runs only with SHOP_TOUR_PEOPLE=remove (people_mode.py; the default is to smudge them, people_blur.py) and not with SHOP_TOUR_REMOVE_OTHERS=0."""
    import people_mode
    return people_mode.remove() and os.environ.get("SHOP_TOUR_REMOVE_OTHERS", "").strip().lower() not in ("0", "false", "no", "off")


def rounds():
    try:
        return max(1, int(os.environ.get("SHOP_TOUR_REMOVE_OTHERS_ROUNDS", "") or ROUNDS))
    except ValueError:
        return ROUNDS


# ---------------------------------------------------------------------------------------------------------------------------------
# geometry of a view cut round a direction of the pano

def _basis(az, el):
    """Rx, Ry of depth_panos.view_rays: a pano ray = Ry @ Rx @ (camera ray)."""
    p, y = np.radians(el), np.radians(az)
    Rx = np.array([[1, 0, 0], [0, np.cos(p), np.sin(p)], [0, -np.sin(p), np.cos(p)]])
    Ry = np.array([[np.cos(y), 0, np.sin(y)], [0, 1, 0], [-np.sin(y), 0, np.cos(y)]])
    return Rx, Ry


class View:
    """A rectilinear n x n view of the pano centred on (az, el) degrees, fov degrees wide (the same maths as depth_panos.view_rays)."""
    def __init__(self, az, el, fov, n):
        from depth_panos import view_rays
        self.az, self.el, self.fov, self.n = float(az), float(el), float(fov), int(n)
        self.f = n / 2 / np.tan(np.radians(fov) / 2)
        self.rays = view_rays(az, el, fov, n).astype(np.float32)             # (n, n, 3) in the pano frame
        Rx, Ry = _basis(az, el)
        self.M = (Rx.T @ Ry.T).astype(np.float32)                           # pano ray -> camera ray

    def cut(self, pano):
        az = np.arctan2(self.rays[..., 0], self.rays[..., 2]); el = np.arcsin(np.clip(self.rays[..., 1], -1, 1))
        h, w = pano.shape[:2]
        return cv2.remap(pano, ((az + np.pi) / (2 * np.pi) * w - 0.5).astype(np.float32), ((np.pi / 2 - el) / np.pi * h - 0.5).astype(np.float32),
                         cv2.INTER_AREA if w > 4 * self.n else cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)

    def project(self, r):
        """Pano rays (..., 3) -> (col, row, in front) in this view's pixels."""
        v = r @ self.M.T
        vz = np.maximum(v[..., 2], 1e-6)
        return (self.f * v[..., 0] / vz + self.n / 2 - 0.5).astype(np.float32), (self.n / 2 - 0.5 - self.f * v[..., 1] / vz).astype(np.float32), v[..., 2] > 1e-3

    def lowres(self, n):
        return View(self.az, self.el, self.fov, n)


def angles_to_ray(az, el):
    a, e = np.radians(az), np.radians(el)
    return np.array([np.cos(e) * np.sin(a), np.sin(e), np.cos(e) * np.cos(a)], np.float32)


def view_size(deg):
    """Field of view and size of the view cut round a person who is `deg` degrees across: wide enough to hold him with room, about the pano's own resolution."""
    fov = float(np.clip(deg * 2.6, 22.0, 110.0))
    n = int(np.clip(round(fov * PX_PER_DEG / 16) * 16, MIN_N, MAX_N))
    return fov, n


def window_pano(view, H, W):
    """(rows, cols) index arrays of the part of the pano a view round (az, el) covers, with room to spare (columns wrap round the seam)."""
    el = view.el
    half = view.fov * 0.85
    r0 = int(max(0, (90 - (el + half)) / 180 * H)); r1 = int(min(H, (90 - (el - half)) / 180 * H + 1))
    cosel = max(math.cos(math.radians(min(abs(el) + half * 0.5, 85.0))), 0.08)
    spanw = min(180.0, half / cosel)
    c = (view.az + 180) / 360 * W
    c0 = int(math.floor(c - spanw / 360 * W)); c1 = int(math.ceil(c + spanw / 360 * W))
    if c1 - c0 >= W:
        cols = np.arange(W)
    else:
        cols = np.arange(c0, c1 + 1) % W
    return np.arange(r0, r1), cols


def paste_view(pano, view, patch, hole, feather_px=7):
    """Write the view's filled picture into the pano wherever the pano sees the hole, feathered at the edge (only a window of the pano is worked on)."""
    H, W = pano.shape[:2]
    rows, cols = window_pano(view, H, W)
    az = ((cols + 0.5) / W * 2 * np.pi - np.pi).astype(np.float32); el = (np.pi / 2 - (rows + 0.5) / H * np.pi).astype(np.float32)
    A, E = np.meshgrid(az, el)
    r = np.stack([np.cos(E) * np.sin(A), np.sin(E), np.cos(E) * np.cos(A)], -1)
    col, row, front = view.project(r)
    n = view.n
    inside = front & (col > 1) & (col < n - 2) & (row > 1) & (row < n - 2)
    wm = cv2.GaussianBlur(hole.astype(np.float32), (0, 0), feather_px)
    w = cv2.remap(wm, col, row, cv2.INTER_LINEAR, borderValue=0) * inside
    w = np.where(cv2.remap(hole.astype(np.float32), col, row, cv2.INTER_LINEAR, borderValue=0) > 0.5, 1.0, w) * inside
    new = cv2.remap(patch, col, row, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    out = pano.copy()
    sub = out[np.ix_(rows, cols)].astype(np.float32)
    out[np.ix_(rows, cols)] = (sub * (1 - w[..., None]) + new.astype(np.float32) * w[..., None]).astype(np.uint8)
    return out


# ---------------------------------------------------------------------------------------------------------------------------------
# what to paint: the gate's own scan

def people_in(det, pano, up):
    """Persons and faces anywhere in the picture, exactly as privacy_gate.Gate.stop finds them: [{kind, score, az, el, deg}] (faces that the gate would let go, small and with
    no person near them, are left out: they are blurred, not held)."""
    found = PG.dedupe(PG.scan(det, pano) + PG.nadir_persons(det, pano, up), tol=4.0)     # (the gate's own tolerance, 12 degrees, joins a crew standing together into one; each is painted here)
    persons = [f for f in found if f["kind"] == "person"]; faces = [f for f in found if f["kind"] == "face"]
    faces = [f for f in faces if not (PG.face_is_small_and_alone(f, persons) and not PG.face_has_body(det, pano, f))]
    return persons + faces


def _iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0])) * max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - ix
    return ix / ua if ua > 0 else 0.0


def find_box(det, view_img, person, view):
    """The detector's box round this person in the view cut round him (pixels), or one drawn from the angles the scan gave when the detector does not see him here."""
    n = view.n; c = n / 2
    deg_px = person["deg"] / view.fov * n
    if person["kind"] == "face":
        best = None
        for sc, (x, y, w, h) in det.faces(view_img):
            d = math.hypot(x + w / 2 - c, y + h / 2 - c)
            if d < 0.3 * n and (best is None or d < best[0]):
                best = (d, [x, y, x + w, y + h])
        b = best[1] if best else [c - deg_px / 2, c - deg_px / 2, c + deg_px / 2, c + deg_px / 2]
        px = 0.35 * (b[2] - b[0]); py = 0.35 * (b[3] - b[1])
        return [max(0, b[0] - px), max(0, b[1] - py), min(n - 1, b[2] + px), min(n - 1, b[3] + py)], "face"
    best = None
    for sc, b in det.persons_at(view_img, 0.25):
        cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
        d = math.hypot(cx - c, cy - c)
        if d < 0.25 * n and (best is None or sc - d / n > best[0]):
            best = (sc - d / n, b)
    if best is not None:
        return [float(v) for v in best[1]], "detector"
    h = deg_px; w = 0.5 * h
    return [max(0.0, c - w / 2), max(0.0, c - h / 2), min(n - 1.0, c + w / 2), min(n - 1.0, c + h / 2)], "angles"


def person_mask(models, view_img, box):
    """SAM 2 with the box as the prompt: the candidate that is the most believable body (its predicted quality, sized like the box), else the box's ellipse."""
    n = view_img.shape[0]
    x0, y0, x1, y1 = box
    area = max(1.0, (x1 - x0) * (y1 - y0))
    best = None
    for m, sc in models.segment_box(view_img, box):
        a = float(m.sum())
        if a < 0.12 * area or a > 1.5 * area:
            continue
        if best is None or sc > best[0]:
            best = (sc, m)
    if best is None:
        m = np.zeros((n, n), np.uint8)
        cv2.ellipse(m, (int((x0 + x1) / 2), int((y0 + y1) / 2)), (max(2, int((x1 - x0) / 2)), max(2, int((y1 - y0) / 2))), 0, 0, 360, 1, -1)
        return m > 0, "ellipse", 0.0
    return best[1].astype(bool), "sam", float(best[0])


def add_shadow_view(view_img, mask, ground, reach):
    """His shadow: ground pixels touching the mask that are clearly darker than the ground round him."""
    g = cv2.GaussianBlur(cv2.cvtColor(view_img, cv2.COLOR_BGR2GRAY).astype(np.float32), (0, 0), 2.5)
    k = lambda r: cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    m8 = mask.astype(np.uint8)
    dist = cv2.distanceTransform(1 - m8, cv2.DIST_L2, 3)
    near = dist < reach
    far = dist < 2 * reach
    ref = far & ~near & ground
    if ref.sum() < 400:
        return mask
    base = float(np.median(g[ref]))
    dark = cv2.morphologyEx((near & ground & (g < SHADOW_DARK * base)).astype(np.uint8), cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    cnt, lab = cv2.connectedComponents(np.maximum(dark, cv2.dilate(m8, np.ones((7, 7), np.uint8))))
    keep = np.unique(lab[mask]); keep = keep[keep > 0]
    sh = np.isin(lab, keep) & (dark > 0)
    if sh.sum() > 4.0 * mask.sum():                                           # a dark road feature, not a shadow
        return mask
    return mask | sh


# ---------------------------------------------------------------------------------------------------------------------------------
# the neighbours

class Neighbour:
    def __init__(self, j, pano, up, cam_h, capt_mask, people, nb):
        self.j, self.pano, self.up, self.h, self.capt, self.people, self.nb = j, pano, up, cam_h, capt_mask, people, nb
        self.R = C.rot_to_y(up).astype(np.float32)


def reproject(view, rays, n_up, up_k, h_k, nbr, d, dyaw, small=None):
    """What neighbour `nbr` sees along `rays` (pano frame of this stop; (m, m, 3)) if this stop's ground plane is the ground and everything above the horizon is far: returns
    (picture crop m x m, valid mask). d: the distance to the neighbour, dyaw: a heading correction (degrees)."""
    a = math.radians(nbr.nb["az"])
    Lk = rays @ C.rot_to_y(up_k).astype(np.float32).T                           # this stop's level frame
    down = -Lk[..., 1]
    t = np.where(down > 1e-3, np.minimum(h_k / np.maximum(down, 1e-3), DMAX), DMAX)
    P = Lk * t[..., None]
    Cj = np.array([math.sin(a) * d, nbr.h - h_k, math.cos(a) * d], np.float32)
    D = (P - Cj) @ C.roty(math.radians(nbr.nb["turn"] + dyaw)).T.astype(np.float32)
    rj = D @ nbr.R                                                              # neighbour's level frame -> its pano frame
    u, v = C.pano_uv(rj, *nbr.pano.shape[:2])
    img = cv2.remap(nbr.pano, u, v, cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
    rn = rj / np.maximum(np.linalg.norm(rj, axis=-1, keepdims=True), 1e-9)
    ok = np.ones(rn.shape[:2], bool)
    for p in nbr.people:                                                        # his neighbour's own people (and its capturer) are not a picture of what is behind this one
        c = angles_to_ray(p["az"], p["el"])
        rad = math.radians((0.5 * p["deg"] + 1.5) if p["el"] < -55 else (0.8 * p["deg"] + 2.0))
        ok &= (rn @ c) < math.cos(min(rad, 3.0))
    if nbr.capt is not None:
        Dn = D / np.maximum(np.linalg.norm(D, axis=-1, keepdims=True), 1e-9)
        dz = -Dn[..., 1]
        f1 = C.VIEW_N / 2 / np.tan(np.radians(C.VIEW_FOV) / 2)
        col = f1 * Dn[..., 0] / np.maximum(dz, 1e-3) + C.VIEW_N / 2 - 0.5; row = C.VIEW_N / 2 - 0.5 - f1 * Dn[..., 2] / np.maximum(dz, 1e-3)
        inside = (dz > 0.05) & (col > 0) & (col < C.VIEW_N - 1) & (row > 0) & (row < C.VIEW_N - 1)
        mk = cv2.remap(nbr.capt.astype(np.uint8), col.astype(np.float32), row.astype(np.float32), cv2.INTER_NEAREST, borderValue=0) > 0
        ok &= ~(inside & mk)
    return img, ok


def _ncc(a, b, m):
    if m.sum() < 300:
        return -1.0
    x, y = a[m], b[m]
    x = x - x.mean(); y = y - y.mean()
    den = math.sqrt(float((x * x).sum() * (y * y).sum()))
    return float((x * y).sum() / den) if den > 1e-6 else -1.0


def _gray(img, s=1.5):
    return cv2.GaussianBlur(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32), (0, 0), s)



def texture_from_nearby(view_img, hole, rest, lowf, lp, grain):
    """The fine detail for a patch that has only its structure (`lowf`): the detail (picture minus its own blur) of the place near the hole, displaced to cover it, whose
    colours and shapes at the patch's scale are closest to the patch (a road, a hedge, a wall are all stationary at this scale, so real texture of the same material); the
    random grain when nothing near is clear of the hole."""
    n = hole.shape[0]
    f = view_img.astype(np.float32)
    det = f - cv2.GaussianBlur(f, (0, 0), lp)
    low_src = cv2.GaussianBlur(f, (0, 0), lp)
    ys, xs = np.nonzero(rest)
    if len(ys) < 50:
        return None, "grain"
    rng = np.random.default_rng(7)
    sel = rng.choice(len(ys), min(len(ys), 6000), replace=False)
    sy, sx = ys[sel], xs[sel]
    size = max(ys.max() - ys.min(), xs.max() - xs.min()) + 1
    free = ~hole
    best = None
    for r in (0.8, 1.2, 1.7, 2.4, 3.2):
        for ang in range(0, 360, 20):
            dx, dy = int(round(math.cos(math.radians(ang)) * r * size)), int(round(math.sin(math.radians(ang)) * r * size))
            ty, tx = sy + dy, sx + dx
            ok = (ty >= 0) & (ty < n) & (tx >= 0) & (tx < n)
            if ok.mean() < 0.95:
                continue
            ty, tx = np.clip(ty, 0, n - 1), np.clip(tx, 0, n - 1)
            if free[ty, tx].mean() < 0.97:
                continue
            dd = np.abs(low_src[ty, tx] - lowf[sy, sx]).mean(-1)
            cost = float(dd.mean() + np.percentile(dd, 90))                        # (a cone or a sign inside the source shows in the 90th percentile)
            if best is None or cost < best[0]:
                best = (cost, dx, dy)
    if best is None:
        return None, "grain"
    _, dx, dy = best
    M = np.float32([[1, 0, -dx], [0, 1, -dy]])
    shifted = cv2.warpAffine(view_img, M, (n, n), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    return shifted, f"copied from {dx:+d},{dy:+d} px (colour cost {best[0]:.1f})"


def fill_hole(view_img, hole, view, nbrs, up_k, h_k, stats, copy_ok=True, avoid=None, person_deg=99.0, seed=0, synth=True):
    """The hole of the view filled from the neighbours' pictures (see the module's header). Returns (picture, got mask, rest mask)."""
    n = view.n
    ker = lambda r: cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    ys, xs = np.nonzero(hole)
    size = max(ys.max() - ys.min(), xs.max() - xs.min()) + 1
    r_in, r_out = max(3, int(0.06 * size)), max(12, int(0.55 * size))
    ring = (cv2.dilate(hole.astype(np.uint8), ker(r_out)) > 0) & ~(cv2.dilate(hole.astype(np.uint8), ker(r_in)) > 0)
    # the search at a quarter of the size
    lo = max(160, n // 4); lv = view.lowres(lo); s = lo / n
    gk_lo = _gray(cv2.resize(view_img, (lo, lo), interpolation=cv2.INTER_AREA))
    ring_lo = cv2.resize(ring.astype(np.uint8), (lo, lo), interpolation=cv2.INTER_NEAREST) > 0
    cand = []
    for nbr in (nbrs[:MAX_NBRS] if (avoid is None or person_deg >= NBR_MIN_DEG) else []):
        ds = nbr.nb.get("ds") or [nbr.nb["d"]]
        ds = sorted({round(d * f, 2) for d in ds for f in (0.75, 1.0, 1.35)})
        best = None
        for d in ds:
            for dyaw in (-6, -3, 0, 3, 6):
                img, ok = reproject(lv, lv.rays, lo, up_k, h_k, nbr, d, dyaw)
                sc = _ncc(gk_lo, _gray(img, 1.0 * s * 2), ring_lo & ok)
                if best is None or sc > best[0]:
                    best = (sc, d, dyaw)
        if best is not None and best[0] >= NCC_MIN:
            cand.append((best[0], nbr, best[1], best[2]))
        else:
            stats["neighbours"].append(dict(j=nbr.j, ncc=round(best[0], 2) if best else None, used=0))
    cand.sort(key=lambda c: -c[0])
    ground = view.rays[..., 1] < -0.02                                          # below the horizon: the road under him, seen from a neighbour at another height of look-angle
    lp = max(2.0, 0.004 * n)                                                    # the neighbour's colours and shapes (markings, shade) are kept; its finer grain is made new
    imgs = []
    for sc, nbr, d, dyaw in cand:
        img, ok = reproject(view, view.rays, n, up_k, h_k, nbr, d, dyaw)
        both = ring & ok
        if both.sum() < 300:
            continue
        g = np.ones(3, np.float32)
        for c in range(3):
            a_, b_ = view_img[..., c][both].astype(np.float32), img[..., c][both].astype(np.float32)
            if np.median(b_) > 5:
                g[c] = np.clip(np.median(a_) / np.median(b_), 0.7, 1.4)
        img = np.clip(img.astype(np.float32) * g, 0, 255).astype(np.uint8)
        imgs.append((sc, nbr, d, dyaw, img, ok, cv2.GaussianBlur(img, (0, 0), 4).astype(np.float32)))
    got = np.zeros((n, n), bool); patch = view_img.copy(); used = {}
    min_area = max(400, int(0.004 * hole.sum()))
    # where two neighbours that look from different sides agree on a pixel, it is what is there (an object between them lands in different places in each)
    import itertools
    for (sa, na, da, ya, ia, oa, ba), (sb, nb_, db, yb, ib, ob, bb) in itertools.combinations(imgs, 2):
        if abs((na.nb["az"] - nb_.nb["az"] + 180) % 360 - 180) < 25:
            continue
        agree = C.clean_regions(oa & ob & hole & ~got & (np.abs(ba - bb).mean(-1) < 22), 15, 7, min_area)
        if agree.any():
            patch = np.where(agree[..., None], ia, patch); got |= agree; used[na.j] = used.get(na.j, 0) + int(agree.sum())
    agreed = int(got.sum())
    for sc, nbr, d, dyaw, img, ok, _ in imgs:                                    # then, where none agreed, the one that lines up best (whole patches)
        if sc < SINGLE_NCC:
            continue
        take = C.clean_regions(ok & hole & ~got, 15, 7, min_area)
        if take.any():
            patch = np.where(take[..., None], img, patch); got |= take; used[nbr.j] = used.get(nbr.j, 0) + int(take.sum())
    for sc, nbr, d, dyaw, img, ok, _ in imgs:
        stats["neighbours"].append(dict(j=nbr.j, ncc=round(sc, 2), d=d, dyaw=dyaw, sees_pct=round(100.0 * (ok & hole).sum() / max(1, hole.sum()), 1), used=used.get(nbr.j, 0)))
    rest = hole & ~got
    gr_sigma = 3.0
    gsrc = cv2.cvtColor(view_img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    hp = gsrc - cv2.GaussianBlur(gsrc, (0, 0), 2.0)
    rg = ring & ground & (np.abs(hp) < 40)
    if rg.sum() > 400:
        gr_sigma = float(1.4826 * np.median(np.abs(hp[rg] - np.median(hp[rg]))))
    rng = np.random.default_rng(1234)
    grain = cv2.GaussianBlur(rng.normal(0, 1, (n, n)).astype(np.float32), (0, 0), 0.9)
    grain *= gr_sigma / max(float(grain.std()), 1e-6)
    grain = np.clip(grain, -3 * gr_sigma, 3 * gr_sigma)[..., None]
    soft_g = got & ground                                                       # borrowed road: its low frequencies, this picture's grain
    if soft_g.any():
        low = cv2.GaussianBlur(patch.astype(np.float32), (0, 0), lp)
        patch = np.where(soft_g[..., None], np.clip(low + grain, 0, 255), patch).astype(np.uint8)
    if got.any():                                                              # Poisson blend: the borrowed picture takes its colour at the edge from the picture round the hole
        hm = (cv2.erode(got.astype(np.uint8), np.ones((3, 3), np.uint8)) * 255)
        yy, xx = np.nonzero(hm)
        if len(yy):
            cx, cy = int((xx.min() + xx.max()) / 2), int((yy.min() + yy.max()) / 2)
            try:
                bl = cv2.seamlessClone(np.ascontiguousarray(patch), np.ascontiguousarray(view_img), hm, (cx, cy), cv2.NORMAL_CLONE)
                patch = np.where(got[..., None], bl, patch)
            except cv2.error:
                pass
    synthed = False
    if synth and rest.sum() >= SYNTH_MIN_PX:
        # what no neighbour saw: texture of the place, made by patch synthesis from the picture round the hole (never from another person or from the hole itself)
        ys, xs = np.nonzero(hole)
        sz_h = max(ys.max() - ys.min(), xs.max() - xs.min())
        m = int(max(40, 1.5 * sz_h)); mx = int(max(60, 4.0 * sz_h))              # (wider across than up: a street is made of horizontal strata)
        y0, y1, x0, x1 = max(0, ys.min() - m), min(n, ys.max() + m + 1), max(0, xs.min() - mx), min(n, xs.max() + mx + 1)
        src = ~(cv2.dilate(hole.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * max(6, int(0.05 * size)) + 1,) * 2)) > 0)
        if avoid is not None:
            src &= ~avoid
        out, note = SF.synth(patch[y0:y1, x0:x1], rest[y0:y1, x0:x1], src[y0:y1, x0:x1], local=STRATA, seed=seed)
        if note.startswith("synth"):
            patch = patch.copy(); patch[y0:y1, x0:x1] = out
            stats["texture"] = note; synthed = True
    if rest.any() and not synthed:
        # what no neighbour saw: the structure continued from the picture round it (a shade band, a lane marking carry on through) at a quarter of the size, and this picture's
        # grain put back on it, as strong as the grain is under that much light (a flat smudge is the thing a person would still see)
        q = max(128, n // 4)
        small = cv2.resize(patch, (q, q), interpolation=cv2.INTER_AREA)
        hs = cv2.dilate(cv2.resize(rest.astype(np.uint8), (q, q), interpolation=cv2.INTER_NEAREST), np.ones((7, 7), np.uint8))
        ink = cv2.resize(cv2.inpaint(small, hs, 7, cv2.INPAINT_TELEA), (n, n), interpolation=cv2.INTER_CUBIC).astype(np.float32)
        lowf = cv2.GaussianBlur(ink, (0, 0), lp)
        shifted, how_t = texture_from_nearby(view_img, hole, rest, lowf, lp, grain) if copy_ok else (None, "grain")      # (a small thing, a cone mistaken for a person: no piece of another place is pasted over it, that is a blob a detector sees again)
        if isinstance(shifted, np.ndarray) and shifted.ndim == 3 and shifted.shape[2] == 3 and shifted.dtype == np.uint8:
            # a real piece of the same material from beside the hole, Poisson-blended so that its colour at the edge follows the picture round the hole
            lowpatch = np.clip(lowf, 0, 255).astype(np.uint8)
            src = np.where(rest[..., None], shifted, lowpatch)
            body = cv2.erode(rest.astype(np.uint8), np.ones((3, 3), np.uint8)) * 255
            yy, xx = np.nonzero(body)
            new = np.where(rest[..., None], shifted, patch)
            if len(yy):
                try:
                    dst = np.where(rest[..., None], lowpatch, patch)
                    new = cv2.seamlessClone(np.ascontiguousarray(src), np.ascontiguousarray(dst), body, (int((xx.min() + xx.max()) / 2), int((yy.min() + yy.max()) / 2)), cv2.NORMAL_CLONE)
                except cv2.error:
                    pass
            patch = np.where(rest[..., None], new, patch)
        else:
            patch = np.where(rest[..., None], np.clip(lowf + grain, 0, 255).astype(np.uint8), patch)
        stats["texture"] = how_t
    stats["got_pct"] = round(100.0 * got.sum() / max(1, hole.sum()), 1)
    stats["rest_pct"] = round(100.0 * rest.sum() / max(1, hole.sum()), 1)
    return patch, got, rest


# ---------------------------------------------------------------------------------------------------------------------------------

def jpeg_roundtrip(pano, quality=95):
    """The picture as it will be when it is written (frames/clean/<id>.jpg is a JPEG): the gate looks at the file, so this module's own look must see the same pixels."""
    ok, buf = cv2.imencode(".jpg", pano, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return cv2.imdecode(buf, cv2.IMREAD_COLOR), buf


def paint_person(pano, person, det, models, nbrs, up_k, h_k, grow=1.0, others=(), seed=0, synth=True):
    """Paint one person (a scan() result) out of the pano; returns (new pano, a record of what was done)."""
    t0 = time.time()
    H, W = pano.shape[:2]
    fov, n = view_size(person["deg"])
    view = View(person["az"], person["el"], fov, n)
    vimg = view.cut(pano)
    box, how = find_box(det, vimg, person, view)
    mask, mhow, msc = person_mask(models, vimg, box)
    size = max(box[2] - box[0], box[3] - box[1])
    ground = (view.rays[..., 1] < -0.02)
    core = np.zeros(mask.shape, np.uint8)                                      # the box's inscribed ellipse is him too (SAM sometimes leaves an arm or a head out)
    cv2.ellipse(core, (int((box[0] + box[2]) / 2), int((box[1] + box[3]) / 2)), (max(2, int(0.42 * (box[2] - box[0]))), max(2, int(0.46 * (box[3] - box[1])))), 0, 0, 360, 1, -1)
    mask = mask | (core > 0)
    mask = add_shadow_view(vimg, mask, ground, max(10, int(1.4 * size)))
    r = max(5, int(0.035 * size * grow)) if person["deg"] >= 9.0 else max(10, int(0.07 * size * grow))      # (a small thing: a wider hole, its edge pixels are most of it)
    hole = cv2.dilate(mask.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))) > 0
    hole = cv2.morphologyEx(hole.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8)) > 0
    stats = dict(neighbours=[])
    avoid = np.zeros(hole.shape, bool)
    for o in others:                                                           # every person found in the picture: nothing is copied from where one stands
        rad = math.radians(0.8 * o["deg"] + 2.0)
        avoid |= (view.rays @ angles_to_ray(o["az"], o["el"])) > math.cos(min(rad, 1.2))
    patch, got, rest = fill_hole(vimg, hole, view, nbrs, up_k, h_k, stats, copy_ok=person["deg"] >= 9.0, avoid=avoid, person_deg=person["deg"], seed=seed, synth=synth)
    out = paste_view(pano, view, patch, hole)
    dbg = os.environ.get("SHOP_TOUR_REMOVE_OTHERS_DEBUG")
    if dbg:
        os.makedirs(dbg, exist_ok=True)
        ov = vimg.copy(); cnts, _ = cv2.findContours(hole.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE); cv2.drawContours(ov, cnts, -1, (0, 0, 255), 3)
        cnts, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE); cv2.drawContours(ov, cnts, -1, (0, 255, 0), 2)
        cv2.rectangle(ov, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (255, 0, 0), 2)
        cv2.imwrite(os.path.join(dbg, f"dbg_{person['az']:.0f}_{person['el']:.0f}.jpg"), cv2.resize(np.hstack([ov, patch]), None, fx=0.5, fy=0.5), [cv2.IMWRITE_JPEG_QUALITY, 85])
    rec = dict(texture=stats.get("texture", "neighbours only"), kind=person["kind"], score=person["score"], az=person["az"], el=person["el"], deg=person["deg"], box_from=how, mask_from=mhow, view_fov=round(fov, 1), view_n=n,
               hole_px=int(hole.sum()), got_pct=stats["got_pct"], rest_pct=stats["rest_pct"], neighbours=stats["neighbours"], seconds=round(time.time() - t0, 1))
    return out, rec


def refill_capturer(pano, up, cam_h, hole1024, kw):
    """The capturer's hole (frames/capturer/<id>_hole.png, 1024 view) made again from the picture round it by patch synthesis. Returns (new pano or None, note, the hole in the 1024 view that
    was made again)."""
    fill = C.Fill(hole1024, up, cam_h)
    vc = C.level_view_crop(pano, fill)
    hole = cv2.dilate(fill.hole.astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
    out, note = SF.synth(vc, hole, **kw)
    if not note.startswith("synth"):
        return None, note, None
    return C.paste_back(pano, out, hole, fill), note, cv2.dilate(hole1024.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0


class Others:
    """The paint-out of every stop of one walk: holds what the stops share (the neighbours' pictures and the people found in them) so that a stop's neighbours are worked out once."""
    def __init__(self, frames, pts, UP, P):
        self.frames, self.pts, self.UP, self.P = frames, pts, UP, P
        self.ids = [p["id"] for p in pts]
        self.lock = threading.RLock()
        self.pics = collections.OrderedDict(); self.people = {}; self.capt = {}
        self.report = {}

    def _pic(self, j):
        with self.lock:
            if j in self.pics:
                self.pics.move_to_end(j); return self.pics[j]
        img = imread(os.path.join(self.frames, "points", self.pts[j]["filename"]))
        with self.lock:
            self.pics[j] = img
            while len(self.pics) > 5:
                self.pics.popitem(last=False)
        return img

    def _capt(self, j):
        if j not in self.capt:
            mp = os.path.join(self.frames, "capturer", self.ids[j] + "_mask.png")
            m = cv2.imread(mp, 0) if os.path.exists(mp) else None
            self.capt[j] = (m > 0) if m is not None and (m > 0).any() else None
        return self.capt[j]

    def _people_of(self, j, det):
        """The people the gate's scan finds in stop j's ORIGINAL picture (what must not be borrowed from it); kept in frames/capturer/others_scan.json."""
        with self.lock:
            if j in self.people:
                return self.people[j]
            sp = os.path.join(self.frames, "capturer", "others_scan.json")
            if not self.people and os.path.exists(sp):
                try:
                    self.people = {int(k): v for k, v in json.load(open(sp)).items()}
                except Exception:
                    self.people = {}
            if j in self.people:
                return self.people[j]
            pano = self._pic(j)
            found = [f for f in PG.dedupe(PG.scan(det, pano) + PG.nadir_persons(det, pano, self.UP[j]), tol=4.0)]
            self.people[j] = found
            json.dump({str(k): v for k, v in self.people.items()}, open(sp, "w"))
            return found

    def neighbours(self, k, det):
        out = []
        for nb in self.P.neighbours(k)[:MAX_NBRS + 2]:
            j = nb["j"]
            out.append(Neighbour(j, self._pic(j), self.UP[j], self.P.cam_h(j), self._capt(j), self._people_of(j, det), nb))
        return out

    def _lump(self, k, pano, models):
        """The gate's check C and the margin's (every camera height, every person-sized shape) on this picture: the first problem found, else None. Reads the stop's masks as the gate does."""
        import privacy_margin as PM
        sid = self.ids[k]
        scales = PM.scales_to_try(None) if PM.enabled() else (1.0,)
        res = PM.sweep(self.frames, scales, ids={sid}, log=lambda *a, **kw: None, candidates=PM.enabled(), models=models, rep_cap=self._report(), panos={sid: pano})
        for s, h, problems, which, *_ in res.get(sid, []):
            if problems:
                return problems[0]
        return None

    def _report(self):
        rp = os.path.join(self.frames, "capturer", "report.json")
        return json.load(open(rp)) if os.path.exists(rp) else {}

    def _refill(self, k, pano, jbuf, up, h, models, rec, log):
        """If the picture has a person-sized lump where the capturer stood, make his hole again (see REFILL_METHODS). Returns (pano, jbuf, whether the picture was made again)."""
        sid = self.ids[k]
        hp = os.path.join(self.frames, "capturer", sid + "_hole.png")
        if rec.get("left") or not os.path.exists(hp):                                # (a person still found is held anyway)
            return pano, jbuf, False                                                 # (nobody was found under the camera: there is nothing of his to make again)
        hole1024 = cv2.imread(hp, 0) > 0
        if not hole1024.any():
            return pano, jbuf, False
        t0 = time.time()
        problem = self._lump(k, pano, models)
        info = dict(first_check=problem or "ok", check_seconds=round(time.time() - t0, 1), tries=[])
        rec["refill"] = info
        if not problem:
            return pano, jbuf, False
        sp = os.path.join(self.frames, "capturer", sid + "_soft.png")
        soft0 = cv2.imread(sp, 0) if os.path.exists(sp) else None
        for name, kw in REFILL_METHODS[:REFILL_TRIES]:
            t1 = time.time()
            cand, note, made = refill_capturer(pano, up, h, hole1024, kw)
            if cand is None:
                info["tries"].append(dict(method=name, note=note)); continue
            cand, cbuf = jpeg_roundtrip(cand)
            if soft0 is not None:                                                    # what was made again is no soft smudge: the gate's grain and flatness checks have nothing to say about it
                s2 = soft0.copy(); s2[made] = 0; cv2.imwrite(sp, s2)
            left = self._lump(k, cand, models)
            info["tries"].append(dict(method=name, note=note, lump=left or "none", seconds=round(time.time() - t1, 1)))
            log(f"{sid}: person-sized lump under the camera ({problem}); made again by {name}: " + (f"still {left}" if left else "none left"))
            if not left:
                info["won"] = name
                return cand, cbuf, True
        if soft0 is not None:
            cv2.imwrite(sp, soft0)                                                   # nothing worked: the picture and its record stay as they were (the gate holds the stop)
        info["won"] = None
        return pano, jbuf, False

    def paint_stop(self, k, det, models, log=print):
        """Paint every other person out of frames/clean/<id>.jpg (in place), and look again; returns the record for the stop."""
        sid = self.ids[k]; t0 = time.time()
        cp = os.path.join(self.frames, "clean", self.pts[k]["filename"])
        pano = imread(cp)
        up, h = self.UP[k], self.P.cam_h(k)
        rec = dict(painted=[], rounds=0, left=[])
        nbrs = None; real = np.zeros(pano.shape[:2], bool); jbuf = None
        ever = []                                                               # everyone found in any round: nothing is copied from where one stood, in a later round too
        for rd in range(rounds()):
            found = people_in(det, pano, up)
            if rd == 0:
                rec["found"] = [dict(kind=f["kind"], az=f["az"], el=f["el"], deg=f["deg"], score=f["score"]) for f in found]
            if not found:
                rec["left"] = []; break
            if nbrs is None:
                nbrs = self.neighbours(k, det)
            done = []
            ever.extend(f for f in found if not any(PG._ang((f['az'], f['el']), (e['az'], e['el'])) < 1.0 for e in ever))
            for f in sorted(found, key=lambda f: -f["deg"]):
                if any(PG._ang((f["az"], f["el"]), (d["az"], d["el"])) < 0.35 * d["deg"] for d in done):
                    continue                                                    # (inside what was just painted: the next scan says whether anything of him is left)
                try:
                    prev = pano
                    # a spot painted by synthesis in an earlier round that the scan finds a person in again (the texture round him held a person-like shape): the old fill, which is smooth, is tried
                    again = rd >= 2 and any(PG._ang((f["az"], f["el"]), (p["az"], p["el"])) < max(2.0, 0.6 * p["deg"]) and str(p.get("texture", "")).startswith("synth") for p in rec["painted"])
                    pano, pr = paint_person(pano, f, det, models, nbrs, up, h, grow=1.0 + 0.6 * rd, others=ever, seed=rd, synth=not again)
                    if pr.get("texture") != "grain":
                        real |= (pano != prev).any(-1)
                except Exception as e:                                          # a person who cannot be painted stays in the picture, and the gate holds the stop
                    pr = dict(kind=f["kind"], az=f["az"], el=f["el"], deg=f["deg"], error=repr(e)[:160])
                    log(f"{sid}: could not paint out the {f['kind']} at {f['az']:+.0f},{f['el']:+.0f}: {e!r}")
                pr["round"] = rd + 1; rec["painted"].append(pr); done.append(f)
            pano, jbuf = jpeg_roundtrip(pano)
            rec["rounds"] = rd + 1
        else:
            rec["left"] = [dict(kind=f["kind"], az=f["az"], el=f["el"], deg=f["deg"], score=f["score"]) for f in people_in(det, pano, up)]
        self._unsoften(sid, up, real)
        pano, jbuf, again = self._refill(k, pano, jbuf, up, h, models, rec, log)
        if jbuf is not None:                                                          # (the very bytes this module looked at: the gate reads the same pixels)
            tmp = cp + ".part"
            with open(tmp, "wb") as fh:
                fh.write(jbuf.tobytes())
            os.replace(tmp, cp)
        rec["seconds"] = round(time.time() - t0, 1)
        with self.lock:
            self.report[sid] = rec
        log(f"{sid}: painted out {len(rec['painted'])} ({', '.join(str(p.get('kind')) for p in rec['painted'])}) in {rec['rounds']} round(s), {len(rec['left'])} left in view of this module's own look ({rec['seconds']}s)")
        return rec

    def _unsoften(self, sid, up, real):
        """Where this module put a real piece of the picture over what the capturer had filled softly (the person at the nadir), that part is no longer a soft fill: the gate's
        4b check ('the soft fill is grainier than the floor round it') is about the capturer's smudge, not about a copied piece of the road."""
        sp = os.path.join(self.frames, "capturer", sid + "_soft.png")
        if not real.any() or not os.path.exists(sp):
            return
        soft = cv2.imread(sp, 0)
        if soft is None or not (soft > 0).any():
            return
        down = C.level_view((real.astype(np.uint8) * 255), up, soft.shape[0]) > 100
        down = cv2.dilate(down.astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
        if (soft[down] > 0).any():
            soft[down] = 0
            cv2.imwrite(sp, soft)

    def write_report(self):
        p = os.path.join(self.frames, "capturer", "others.json")
        old = json.load(open(p)) if os.path.exists(p) else {}
        old.update(self.report)
        json.dump(old, open(p, "w"), indent=1)


def stage(frames, pts, UP, P, todo, det, models, log=print):
    """The paint-out of the stops in `todo` (indices), one after the other (the three-script path: remove_capturer.py calls this after the floors are made)."""
    O = Others(frames, pts, UP, P)
    for k in todo:
        O.paint_stop(k, det, models, log)
    O.write_report()
    return O


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--frames", default="frames")
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--force", action="store_true", help="run although SHOP_TOUR_REMOVE_OTHERS=0")
    a = ap.parse_args()
    if not (enabled() or a.force):
        print("SHOP_TOUR_REMOVE_OTHERS=0: the other people stay in the pictures (and the gate holds the stops they are in). Nothing done.")
        return
    pts = json.load(open(os.path.join(a.frames, "points.json")))
    UP = json.load(open(os.path.join(a.frames, "up.json"))); P = PI.Inputs(a.frames, len(pts))
    todo = [k for k, p in enumerate(pts) if not a.only or p["id"] in a.only]
    det = PG.Detectors(); models = C.Models()
    stage(a.frames, pts, UP, P, todo, det, models)


if __name__ == "__main__":
    main()
