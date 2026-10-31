"""A second opinion on check C ("a person-sized shape is still near the camera"): is the shape an OBJECT that the neighbouring stops see too?

The capturer is never in two stops' pictures at the same place on the floor: he walks on between the stops. A shape that a stop sees at (x, y) on the floor
and that two neighbouring stops - looking from different sides - ALSO see there, in their ORIGINAL pictures (nothing painted in, their own capturer's mask and
nadir cone left out), is not him: it is a toolbox, a chair, a spool, a rack. qa/check_capturer.py asks this once check C has found a person-sized shape; the
shape is dropped (the hold is not made) only when NEED_SHARE of its pixels are places where two neighbours agree with each other AND with this stop's picture.
Where he is left in a stop the neighbours see bare floor or clutter, not him, so they do not agree with the stop and the hold stands.

How "agree" is worked out (CPU only: numpy and OpenCV, no model; nothing is read from vio or the map - the stops, levelling, hop azimuth/distance and the
alignment search are the ones capturer.fill_stop uses to rebuild the floor, the agreement is the one it uses to trust a floor pixel):
  - each neighbour j is lined up with this stop on the floor ring round the stop's own hole (capturer.align_search: heading, distance, sideways offset);
    a neighbour that does not line up with the floor (NCC under ALIGN_NCC) is not asked, so a stop whose neighbours do not line up is never judged an object;
  - every neighbour's picture is looked up at the floor points of this stop's straight-down view and at a few heights above them (HEIGHTS: a thing that stands
    h above the floor lies on another floor point from every stop), brightness matched to this stop's over the window (the median is the floor: him and an object
    are a small part of it), lightly blurred; a place counts only where j sees it between ASK_MIN_DEG and 72 degrees off its own straight down (never at its nadir,
    where its own capturer stands) and outside its own capturer's mask;
  - two neighbours whose directions differ by at least MIN_SPREAD_DEG (the same rule as the fill: two stops on one side see something above the floor in the
    same wrong place) AGREE at a place when their colours differ by less than AGREE_TOL and both are within STOP_TOL of this stop's picture there.
Judged: the share of the shape's pixels that are agreed places (small speckles of agreement dropped), and how many of its pixels two neighbours saw at all.
It never touches the person pass, the face pass or the horizon rule, only ever REMOVES a check-C hold, and SHOP_TOUR_NEIGHBOUR_CHECK=0 turns it off.
"""
import collections
import hashlib
import itertools
import os
import threading

import numpy as np
import cv2

import capturer as C
import privacy_inputs as PI
from imutil import imread

ALIGN_NCC = 0.3            # a neighbour is asked only when it lines up with this stop's floor this well (capturer.SINGLE_NCC)
MAX_ASKED = 6              # neighbours lined up and asked
MIN_SPREAD_DEG = 35        # two neighbours vouch together only from directions this far apart (capturer.MIN_SPREAD_DEG)
AGREE_TOL = 18             # two neighbours agree on a place when their (blurred) colours differ by less than this (0-255, capturer.AGREE_TOL)
STOP_TOL = 28              # ... and each is this close to this stop's own picture
NEED_SHARE = 0.65          # the shape is an object when at least this share of it is an agreed place (walk 1: his objects 0.8-0.9, the capturer in the original pictures 0.5 at most)
NEED_SEEN = 0.7            # ... and at least this share of it was seen by two neighbours (that could agree) at all
ASK_MIN_DEG = 22           # a neighbour is not asked about places within this many degrees of its own straight down (where its capturer stands)
STEP = 2                   # the work grid: every 2nd pixel of the 1024 view (about 2 cm)
GRAY_W = 2880              # a neighbour's picture is looked at at this width
SPECKLE_PX = 1             # agreed places narrower than this many grid cells are dropped (1: none; an opening of 5 cells, 10 cm, ate a chair's legs)
HEIGHTS = (0.0, 0.2, 0.4, 0.6, 0.8)   # metres above the floor the places are looked at (a thing that stands h above the floor lies on another floor point from every stop)
WINDOW_PX = 120            # the agreement is worked out in the capturer's box and this many view pixels (about 1.2 m) round it
BLUR = 2.0                 # the pictures are compared after this much smoothing (sigma, in 1024-view pixels; the fill uses 2 cm)
_LOCK = threading.RLock()


def enabled():
    """SHOP_TOUR_NEIGHBOUR_CHECK=0 (or off/no/false) turns the second opinion off."""
    return os.environ.get("SHOP_TOUR_NEIGHBOUR_CHECK", "").strip().lower() not in ("0", "false", "no", "off")


def _remap_flat(img, u, v, **kw):
    """cv2.remap for any number of points (it takes at most 32,767 columns and rows): the points are laid out 512 to a row."""
    shp = u.shape; n = u.size
    rows = -(-n // 512)
    uu = np.zeros(rows * 512, np.float32); vv = np.zeros(rows * 512, np.float32)
    uu[:n] = u.ravel(); vv[:n] = v.ravel()
    out = cv2.remap(img, uu.reshape(rows, 512), vv.reshape(rows, 512), cv2.INTER_LINEAR, **kw)
    return out.reshape(-1, *out.shape[2:])[:n].reshape(*shp, *out.shape[2:])


def _small(pano, w=GRAY_W):
    """The neighbour's picture as the agreement looks at it: w wide, a little smoothed, 8 bits (remap works on them and keeps the memory small)."""
    im = cv2.resize(pano, (w, int(round(pano.shape[0] * w / pano.shape[1]))), interpolation=cv2.INTER_AREA) if pano.shape[1] > w else pano
    return cv2.GaussianBlur(im, (0, 0), 1.0)


class Context:
    """What the check needs from one walk's frames: the original pictures of the neighbours, their capturer masks, the stop geometry (privacy_inputs: hops.json) and,
    per stop, which neighbours line up with it. Pictures are read when first asked for and a few are kept. Safe to share between threads."""
    KEEP = 12

    def __init__(self, frames, pts=None, UP=None, PIN=None):
        import json
        self.frames = frames
        self.pts = pts if pts is not None else json.load(open(os.path.join(frames, "points.json")))
        self.UP = UP if UP is not None else json.load(open(os.path.join(frames, "up.json")))
        self.PIN = PIN if PIN is not None else PI.Inputs(frames, len(self.pts))
        self.ids = [p["id"] for p in self.pts]
        self._pic = collections.OrderedDict(); self._aligned = {}; self._agree = {}

    # -- pictures of the neighbours (originals) ------------------------------------------------------------------------------------------------------
    def _mask(self, j):
        mp = os.path.join(self.frames, "capturer", self.ids[j] + "_mask.png")
        m = cv2.imread(mp, 0) if os.path.exists(mp) else None
        return None if m is None or not (m > 0).any() else (m > 0)

    def picture(self, j):
        """(small BGR picture of the original, up, cam_h, capturer mask 1024 or None, small grey for the alignment search) of stop j. (The full picture is not kept.)"""
        with _LOCK:
            if j in self._pic:
                self._pic.move_to_end(j)
                return self._pic[j]
            pano = imread(os.path.join(self.frames, "points", self.pts[j]["filename"]))
            got = (_small(pano), self.UP[j], self.PIN.cam_h(j), self._mask(j), C.small_gray(pano))
            self._pic[j] = got
            while len(self._pic) > self.KEEP:
                self._pic.popitem(last=False)
            return got

    # -- which neighbours line up with stop k's floor ------------------------------------------------------------------------------------------------
    def aligned(self, k):
        """[{j, d, az, turn, par}] for the neighbours of stop k that line up with its floor (the search capturer.fill_stop makes, on the same ring), best first.
        Worked out once per stop."""
        with _LOCK:
            if k in self._aligned:
                return self._aligned[k]
            out = []
            mk = self._mask(k)
            if mk is not None:
                pano_k = imread(os.path.join(self.frames, "points", self.pts[k]["filename"])); up_k, h_k = self.UP[k], self.PIN.cam_h(k)
                fill = C.Fill(mk, up_k, h_k)
                vc = C.level_view_crop(pano_k, fill)
                ker = lambda r: cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
                rpx = lambda m: max(1, int(m * fill.f / fill.h))
                hole = fill.hole.astype(np.uint8)
                ring = (cv2.dilate(hole, ker(rpx(C.HALO_M))) > 0) & ~(cv2.dilate(hole, ker(rpx(0.05))) > 0)
                rs = C.RingSearch(fill, vc, ring)
                for nb in self.PIN.neighbours(k):
                    if len(out) >= MAX_ASKED:
                        break
                    _, uj, hj, mj, gs = self.picture(nb["j"])
                    par, d = max(((C.align_search(rs, gs, uj, hj, nb["az"], d, nb["turn"], mj), d) for d in (nb.get("ds") or [nb["d"]])), key=lambda r: r[0]["ncc"])
                    if par["ncc"] >= ALIGN_NCC:
                        out.append(dict(j=nb["j"], d=d, az=nb["az"], turn=nb["turn"], par=par))
                out.sort(key=lambda o: -o["par"]["ncc"])
            self._aligned[k] = out
            return out

    # -- what neighbour o sees at the floor points of stop k's view -----------------------------------------------------------------------------------
    def sample(self, k, o, ys, xs, h=0.0):
        """(BGR colours (..., 3), valid) that neighbour o sees where stop k's pixels (rows ys, columns xs of the 1024 view, any shape) would be if they stood h above the floor."""
        sj, uj, hj, mj, _ = self.picture(o["j"])
        H = self.PIN.cam_h(k)
        d, f = C.view_dirs()
        dd = d[ys, xs]
        P = dd * ((H - h) / -dd[..., 1])[..., None]
        a = np.radians(o["az"]); par = o["par"]
        Cv = np.array([np.sin(a) * (o["d"] + par["dd"]) - np.cos(a) * par["dl"], hj - H, np.cos(a) * (o["d"] + par["dd"]) + np.sin(a) * par["dl"]], np.float32)
        D = ((P - Cv).astype(np.float32)) @ C.roty(np.radians(o["turn"] + par["dyaw"])).T.astype(np.float32)
        rj = D @ C.rot_to_y(uj).astype(np.float32)
        H2, W2 = sj.shape[:2]
        n = np.linalg.norm(rj, axis=-1)
        u = ((np.arctan2(rj[..., 0], rj[..., 2]) + np.pi) / (2 * np.pi) * W2 - 0.5).astype(np.float32)
        v = ((np.pi / 2 - np.arcsin(np.clip(rj[..., 1] / n, -1, 1))) / np.pi * H2 - 0.5).astype(np.float32)
        col = _remap_flat(sj, u, v, borderMode=cv2.BORDER_WRAP).astype(np.float32)
        Dn = D / np.linalg.norm(D, axis=-1, keepdims=True)
        down = -Dn[..., 1]
        ok = (down > np.cos(np.radians(72))) & (down < np.cos(np.radians(ASK_MIN_DEG)))
        if mj is not None:                                                        # where j's own capturer stands
            f1 = C.VIEW_N / 2 / np.tan(np.radians(C.VIEW_FOV) / 2)
            ccol = f1 * Dn[..., 0] / np.maximum(down, 1e-3) + C.VIEW_N / 2 - 0.5; crow = C.VIEW_N / 2 - 0.5 - f1 * Dn[..., 2] / np.maximum(down, 1e-3)
            inside = (ccol > 0) & (ccol < C.VIEW_N - 1) & (crow > 0) & (crow < C.VIEW_N - 1)
            mk = _remap_flat(cv2.dilate(mj.astype(np.uint8), np.ones((15, 15), np.uint8)).astype(np.float32), ccol.astype(np.float32), crow.astype(np.float32), borderValue=0) > 0
            ok &= ~(inside & mk)
        return col, ok

    # -- where two neighbours agree with each other and with this stop -------------------------------------------------------------------------------------------
    def agreement(self, k, pano):
        """(agreed (bool, the 1024 view at every STEP-th pixel), seen (bool: two neighbours that could agree both saw the place), neighbours used) for stop k whose
        picture is `pano`. Worked out once per stop and picture."""
        key = (k, hashlib.blake2b(np.ascontiguousarray(pano[::61, ::61]).tobytes(), digest_size=8).digest(), pano.shape)
        with _LOCK:
            if key in self._agree:
                return self._agree[key]
            n = C.VIEW_N // STEP
            agreed = np.zeros((n, n), bool); seen = np.zeros((n, n), bool); used = []
            al = self.aligned(k)
            mk = self._mask(k)
            if len(al) >= 2 and mk is not None:
                ys, xs = np.nonzero(mk)
                r0, r1 = max(0, int(ys.min()) - WINDOW_PX), min(C.VIEW_N, int(ys.max()) + WINDOW_PX); c0, c1 = max(0, int(xs.min()) - WINDOW_PX), min(C.VIEW_N, int(xs.max()) + WINDOW_PX)
                r0 -= r0 % STEP; c0 -= c0 % STEP
                gy, gx = np.meshgrid(np.arange(r0, r1, STEP), np.arange(c0, c1, STEP), indexing="ij")
                v = C.level_view(pano, self.UP[k])
                ck = cv2.GaussianBlur(v, (0, 0), BLUR).astype(np.float32)[gy, gx]
                wagreed = np.zeros(gy.shape, bool); wseen = np.zeros(gy.shape, bool)
                for h in HEIGHTS:
                    views = []
                    for o in al:
                        col, ok = self.sample(k, o, gy, gx, h)
                        col = cv2.GaussianBlur(col, (0, 0), BLUR / STEP)
                        if ok.sum() < 1000:
                            continue
                        gain = (np.median(ck[ok], axis=0) + 1e-3) / (np.median(col[ok], axis=0) + 1e-3)    # the floor is most of the window: the median of each channel
                        views.append((o, np.clip(col * gain, 0, 255), ok))
                        if o["j"] not in used:
                            used.append(o["j"])
                    for (oa, ca, oka), (ob, cb, okb) in itertools.combinations(views, 2):
                        if abs((oa["az"] - ob["az"] + 180) % 360 - 180) < MIN_SPREAD_DEG:
                            continue
                        both = oka & okb
                        wseen |= both
                        wagreed |= both & (np.abs(ca - cb).mean(-1) < AGREE_TOL) & (np.abs(ca - ck).mean(-1) < STOP_TOL) & (np.abs(cb - ck).mean(-1) < STOP_TOL)
                if wagreed.any():
                    wagreed = cv2.morphologyEx(wagreed.astype(np.uint8), cv2.MORPH_OPEN, np.ones((SPECKLE_PX, SPECKLE_PX), np.uint8)) > 0
                a0, b0 = r0 // STEP, c0 // STEP
                agreed[a0:a0 + gy.shape[0], b0:b0 + gy.shape[1]] = wagreed; seen[a0:a0 + gy.shape[0], b0:b0 + gy.shape[1]] = wseen
            got = (agreed, seen, used)
            if len(self._agree) > 16:
                self._agree.pop(next(iter(self._agree)))
            self._agree[key] = got
            return got

    def judge(self, sid, shape, pano):
        """Is `shape` (bool, the 1024 straight-down view of the cleaned picture `pano` of stop `sid`) an object the neighbours see too?
        Returns dict(object=bool, share=share of the shape that is an agreed place, seen=share two neighbours saw, neighbours=[stop ids])."""
        res = dict(object=False, share=0.0, seen=0.0, neighbours=[])
        if not enabled() or sid not in self.ids:
            return res
        k = self.ids.index(sid)
        agreed, seen, used = self.agreement(k, pano)
        sub = shape[::STEP, ::STEP]
        n = int(sub.sum())
        res["neighbours"] = [self.ids[j] for j in used]
        if n < 50 or not seen.any():
            return res
        res["share"] = round(float((agreed & sub).sum()) / n, 2); res["seen"] = round(float((seen & sub).sum()) / n, 2)
        res["object"] = res["share"] >= NEED_SHARE and res["seen"] >= NEED_SEEN
        return res


_CONTEXTS = {}


def get_context(frames):
    """One Context per walk's frames folder in this process (the gate and the margin share the pictures and the alignments)."""
    key = os.path.abspath(frames)
    with _LOCK:
        if key not in _CONTEXTS:
            _CONTEXTS[key] = Context(frames)
        return _CONTEXTS[key]
