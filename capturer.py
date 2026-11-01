"""Find the person holding the stick (the capturer) in a stop's straight-down view WITHOUT a person detector, and rebuild the floor under them.

Mask (find_mask):
  a) depth: Depth Anything V2 Small (Apache-2.0) on a straight-down view in the stop's own levelled frame. The floor is a plane at the camera height, so its
     relative depth is a smooth surface; a pixel whose depth sits more than HEIGHT_M above that surface (within 60 degrees of straight down) and whose
     blob comes close to the nadir is the capturer,
  b) SAM 2.1 (Apache-2.0) with a point prompt at the nadir and at every blob of (a); points are added on (a) pixels it left out,
  the union of both, plus his shadow (darker floor touching the mask), dilated slightly.
Fill (fill_patch): see that function. Pure numpy / OpenCV / torch; every model is Apache-2.0 or MIT.
"""
import numpy as np
import cv2

NEAR_M = 0.12            # depth pixels count as part of him only this close to what SAM found (beyond that it is shelves and tools)
HEIGHT_M = 0.15          # anything higher than this above the floor plane, near the nadir, is the capturer
CONE_DEG = 60.0          # "near the nadir"
VIEW_N = 1024            # straight-down view: n x n pixels
VIEW_FOV = 130.0         # covers 65 degrees from straight down in every direction
CORE_DEG = 34.0          # a blob has to come this close to the nadir to count as the capturer (he stands 10-40 degrees off it)
SAM_ID = "facebook/sam2.1-hiera-small"
DA_ID = "depth-anything/Depth-Anything-V2-Small-hf"


def rot_to_y(u):
    """Minimal rotation taking the unit vector u to (0, 1, 0)."""
    u = np.asarray(u, float); u = u / np.linalg.norm(u); y = np.array([0, 1.0, 0]); v = np.cross(u, y); c = float(u @ y)
    if np.linalg.norm(v) < 1e-9:
        return np.eye(3)
    K = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + K + K @ K / (1 + c)


def pano_uv(r, h, w):
    """Pano rays (..., 3) -> pixel coordinates in an h x w equirect picture."""
    n = np.linalg.norm(r, axis=-1)
    az = np.arctan2(r[..., 0], r[..., 2]); el = np.arcsin(np.clip(r[..., 1] / n, -1, 1))
    return ((az + np.pi) / (2 * np.pi) * w - 0.5).astype(np.float32), ((np.pi / 2 - el) / np.pi * h - 0.5).astype(np.float32)


def view_dirs(n=VIEW_N, fov=VIEW_FOV):
    """Level-frame ray of every pixel of the straight-down view (x right, y up, z forward), image up = forward. Also the focal length in pixels."""
    f = n / 2 / np.tan(np.radians(fov) / 2)
    c, r = np.meshgrid(np.arange(n) + 0.5, np.arange(n) + 0.5)
    d = np.stack([(c - n / 2) / f, -np.ones_like(c), (n / 2 - r) / f], -1)
    return d / np.linalg.norm(d, axis=-1, keepdims=True), f


def level_view(pano, up, n=VIEW_N, fov=VIEW_FOV):
    d, _ = view_dirs(n, fov)
    u, v = pano_uv(d @ rot_to_y(up), *pano.shape[:2])
    return cv2.remap(pano, u, v, cv2.INTER_AREA if pano.shape[1] > 4 * n else cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)


def view_to_pano(view, up, h, w, n=VIEW_N, fov=VIEW_FOV):
    """The straight-down view (single or multi channel) -> an equirect picture (h x w) with the same colours; second result: where it covers."""
    R = rot_to_y(up)
    az = ((np.arange(w) + 0.5) / w * 2 * np.pi - np.pi).astype(np.float32); el = (np.pi / 2 - (np.arange(h) + 0.5) / h * np.pi).astype(np.float32)
    A, E = np.meshgrid(az, el)
    r = np.stack([np.cos(E) * np.sin(A), np.sin(E), np.cos(E) * np.cos(A)], -1)
    L = r @ R.T
    ok = L[..., 1] < -np.cos(np.radians(fov / 2)) * 1.02
    f = n / 2 / np.tan(np.radians(fov) / 2)
    dy = np.where(ok, -L[..., 1], 1)
    col = (f * L[..., 0] / dy + n / 2 - 0.5).astype(np.float32); row = (n / 2 - 0.5 - f * L[..., 2] / dy).astype(np.float32)
    ok &= (col > 0) & (col < n - 1) & (row > 0) & (row < n - 1)
    out = cv2.remap(view, col, row, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return out, ok


class Models:
    """Depth and SAM 2, loaded once."""
    def __init__(self, device=None):
        import os
        os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1"); os.environ.setdefault("HF_HUB_VERBOSITY", "error")
        import torch
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation, Sam2Model, Sam2Processor
        self.torch = torch; self.dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dp = AutoImageProcessor.from_pretrained(DA_ID); self.dm = AutoModelForDepthEstimation.from_pretrained(DA_ID).to(self.dev).eval()
        self.sp = Sam2Processor.from_pretrained(SAM_ID); self.sm = Sam2Model.from_pretrained(SAM_ID).to(self.dev).eval()

    def disparity(self, bgr):
        from PIL import Image
        t = self.torch
        x = self.dp(images=Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)), return_tensors="pt").to(self.dev)
        with t.no_grad():
            o = self.dm(**x).predicted_depth
        return t.nn.functional.interpolate(o[None], size=bgr.shape[:2], mode="bicubic")[0, 0].cpu().numpy()

    def segment(self, bgr, points):
        """SAM 2 mask for positive point prompts [(x, y), ...] given together; the highest-scoring of its three candidates."""
        t = self.torch
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        inp = self.sp(images=rgb, input_points=[[[list(map(float, p)) for p in points]]], input_labels=[[[1] * len(points)]], return_tensors="pt").to(self.dev)
        with t.no_grad():
            o = self.sm(**inp, multimask_output=True)
        m = self.sp.post_process_masks(o.pred_masks.cpu(), inp["original_sizes"].cpu())[0][0].numpy() > 0
        return m[int(np.argmax(o.iou_scores.cpu().numpy().ravel()))]


    def segment_many(self, bgr, points, chunk=8):
        """SAM 2 candidates for single-point prompts: the picture is encoded once. Returns [(point index, mask bool, predicted IoU)], 3 per point."""
        t = self.torch
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        out = []
        for c0 in range(0, len(points), chunk):
            pts = points[c0:c0 + chunk]
            inp = self.sp(images=rgb, input_points=[[[[float(x), float(y)]] for x, y in pts]], input_labels=[[[1]] * len(pts)], return_tensors="pt").to(self.dev)
            with t.no_grad():
                o = self.sm(**inp, multimask_output=True)
            m = self.sp.post_process_masks(o.pred_masks.cpu(), inp["original_sizes"].cpu())[0].numpy() > 0       # (objects, 3, H, W)
            sc = o.iou_scores.cpu().numpy()[0]
            for k in range(len(pts)):
                for j in range(3):
                    out.append((c0 + k, m[k, j], float(sc[k, j])))
        return out


def depth_flags(disp, cam_h, reach_m=0.9, n=VIEW_N, fov=VIEW_FOV):
    """Height of every pixel above the floor around it, in metres. The depth model gives relative depth (inverse, up to scale and with a drifting
    floor), so the floor under each pixel is taken as the farthest surface within about a metre of it (grayscale erosion of the inverse depth,
    smoothed); the floor is a plane CAM_H below the camera, so a pixel nearer than that by a fraction q has height cam_h * q. Returns
    (flags: higher than HEIGHT_M and within CONE_DEG of straight down, heights, angle from straight down)."""
    d, f = view_dirs(n, fov)
    ang = np.degrees(np.arccos(np.clip(-d[..., 1], -1, 1)))
    r = int(reach_m / (cam_h / f))
    base = cv2.erode(disp, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1)))
    base = cv2.GaussianBlur(base, (0, 0), r / 3)
    base = np.minimum(base, disp)                                       # the floor is never nearer than the pixel itself
    q = 1 - np.clip(base, 1e-3, None) / np.clip(disp, 1e-3, None)       # fraction of the way from the floor up to the camera
    height = cam_h * q
    return (height > HEIGHT_M) & (ang < CONE_DEG), height, ang


def prior_points(cam_h, n=VIEW_N, fov=VIEW_FOV):
    """Where the capturer can stand in the view: the camera is on his stick, so he is always at about the same side of it (image up) and 8-46 degrees
    off straight down. Prompt points on a coarse grid of that window; the pictures decide which one is him."""
    f = n / 2 / np.tan(np.radians(fov) / 2)
    pts = []
    for pol in range(8, 47, 6):
        for az in range(-40, 41, 10):
            t = np.tan(np.radians(pol)); a = np.radians(az)
            pts.append((n / 2 + f * t * np.sin(a), n / 2 - f * t * np.cos(a)))
    return pts


def score_candidates(cands, disp, cam_h, mpp, n=VIEW_N):
    """Pick the mask that looks like a standing person seen from above: person sized, solid, and standing well above a ring of floor all round it
    (by depth, not by looking for a person). Returns (best mask or None, its score, the number of candidates that passed)."""
    best, bs, passed, found = None, 0.0, 0, []
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * int(0.3 / mpp) + 1,) * 2)
    seen = []
    for k, m, iou in cands:
        area = m.sum() * mpp * mpp
        if not 0.12 <= area <= 1.8 or iou < 0.5:
            continue
        if any((m & o).sum() > 0.9 * (m | o).sum() for o in seen):
            continue
        seen.append(m)
        mu = m.astype(np.uint8)
        cnts, _ = cv2.findContours(mu, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        hull = cv2.contourArea(cv2.convexHull(max(cnts, key=cv2.contourArea))) if cnts else 0
        if hull == 0 or m.sum() / hull < 0.55:
            continue
        ring = (cv2.dilate(mu, ker) > 0) & ~(cv2.dilate(mu, np.ones((9, 9), np.uint8)) > 0)
        if ring.sum() < 200:
            continue
        inside = np.median(disp[m]); out = disp[ring]
        h = cam_h * (1 - np.median(out) / inside); share = (out < inside * 0.93).mean()
        if h < 0.35 or share < 0.7:
            continue
        passed += 1
        sc = h + 0.5 * iou + 0.3 * min(area, 1.0)
        found.append((sc, m))
        if sc > bs:
            best, bs = m, sc
    if best is not None:                                                    # his head and shoulders can come out as separate blobs: join the ones right next to the best
        ys, xs = np.nonzero(best); c0 = np.array([xs.mean(), ys.mean()]); u = best.copy()
        for sc, m in sorted(found, key=lambda t: -t[0]):
            ys, xs = np.nonzero(m)
            if np.hypot(xs.mean() - c0[0], ys.mean() - c0[1]) * mpp < 0.7 and (u | m).sum() * mpp * mpp <= 2.0:
                u |= m
        best = u
    return best, bs, passed


def find_mask(models, pano, up, cam_h, log=None, hint_fn=None):
    """The capturer's mask in the straight-down view (bool, VIEW_N square) plus debug pieces. mask is None when no person-like blob was found (the
    stop has to be held back for review)."""
    n = VIEW_N
    v = level_view(pano, up)
    disp = cv2.medianBlur(models.disparity(v).astype(np.float32), 5)
    d, f = view_dirs(); mpp = cam_h / f
    ang = np.degrees(np.arccos(np.clip(-d[..., 1], -1, 1)))
    flag_s, height_s, _ = depth_flags(disp, cam_h, reach_m=0.45)           # (a) height above the floor right around each pixel
    flag, height, _ = depth_flags(disp, cam_h)
    dbg = dict(view=v, disp=disp, height=height, height_s=height_s, flag=flag_s, ang=ang, mpp=mpp, prompts=[])
    pts = prior_points(cam_h)
    cands = models.segment_many(v, pts)
    sam, sc, passed = score_candidates(cands, disp, cam_h, mpp)
    dbg.update(score=sc, passed=passed, prompts=pts)
    if sam is None and hint_fn is not None:
        # fallback: the depth contrast found nobody (he can be a small grey head far below the lens). A person detector is asked where a person is near the
        # nadir; its box centre is only a point prompt for SAM, which draws the outline. privacy_gate.py runs the same detector on the result anyway.
        hint = hint_fn(v, pano, up)
        if hint is not None:
            cand = models.segment(v, [hint])
            if 0.04 <= cand.sum() * mpp * mpp <= 1.8:
                sam = cand; dbg["fallback"] = "person detector point prompt"
    if sam is None:
        return None, dbg
    ellipse = lambda rr: cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * rr + 1, 2 * rr + 1))
    # he holds the stick right under the camera: his body runs from the nadir (hand) to his head. Prompt SAM along that line.
    ys, xs = np.nonzero(sam); head = (float(xs.mean()), float(ys.mean())); c0 = (n / 2.0, n / 2.0)
    line = [c0] + [(c0[0] + t * (head[0] - c0[0]), c0[1] + t * (head[1] - c0[1])) for t in (0.3, 0.55, 0.8)] + [head]
    body = models.segment(v, line)
    if body.sum() * mpp * mpp <= 2.0 and (body & sam).sum() >= 0.6 * sam.sum():
        sam = sam | body
    else:
        dbg["body_rejected"] = float(body.sum() * mpp * mpp)
    used = list(line)
    for _ in range(2):                                                      # (b) SAM again with extra points on the depth pixels it left out
        near = cv2.dilate(sam.astype(np.uint8), ellipse(int(NEAR_M / mpp))) > 0
        a = cv2.morphologyEx((flag_s & near).astype(np.uint8), cv2.MORPH_OPEN, np.ones((5, 5), np.uint8)) > 0
        left = cv2.morphologyEx((a & ~(cv2.dilate(sam.astype(np.uint8), np.ones((7, 7), np.uint8)) > 0)).astype(np.uint8), cv2.MORPH_OPEN, ellipse(5))
        cnt, lab, st, cen = cv2.connectedComponentsWithStats(left)
        if cnt <= 1 or st[1:, cv2.CC_STAT_AREA].max() * mpp * mpp < 0.02:
            break
        kk = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA])); used.append((float(cen[kk][0]), float(cen[kk][1])))
        more = models.segment(v, used)
        if more.sum() * mpp * mpp > 2.0 or (more & sam).sum() < 0.7 * sam.sum():
            used.pop(); break
        sam = sam | more
    dbg["prompts"] = used
    near = cv2.dilate(sam.astype(np.uint8), ellipse(int(NEAR_M / mpp))) > 0
    a = cv2.morphologyEx((flag_s & near).astype(np.uint8), cv2.MORPH_OPEN, np.ones((5, 5), np.uint8)) > 0
    mask = cv2.morphologyEx((a | sam).astype(np.uint8), cv2.MORPH_CLOSE, ellipse(8)) > 0
    # the stick: from the camera (the nadir) to the hand
    ys, xs = np.nonzero(sam); j = int(np.argmin(np.hypot(xs - n / 2, ys - n / 2)))
    stick = np.zeros((n, n), np.uint8); cv2.line(stick, (n // 2, n // 2), (int(xs[j]), int(ys[j])), 255, max(3, int(0.07 / mpp)))
    mask |= stick > 0
    cnt, lab = cv2.connectedComponents(mask.astype(np.uint8))                # one body: drop pieces that do not connect to the stick or the head
    keep = np.unique(lab[(stick > 0) | sam]); keep = keep[keep > 0]
    ok = np.isin(lab, keep)
    cnt2, lab2, st2, _ = cv2.connectedComponentsWithStats(ok.astype(np.uint8))
    mask = lab2 == (1 + int(np.argmax(st2[1:, cv2.CC_STAT_AREA]))) if cnt2 > 1 else ok
    dbg.update(sam=sam, a=a)
    return mask, dbg


def add_shadow(view, mask, height, cam_h, mpp, reach_m=0.7):
    """His shadow: floor pixels touching the mask that are clearly darker than the floor around."""
    g = cv2.GaussianBlur(cv2.cvtColor(view, cv2.COLOR_BGR2GRAY).astype(np.float32), (0, 0), 3)
    n = mask.shape[0]; r = int(reach_m / mpp)
    near = cv2.dilate(mask.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))) > 0
    ring = near & ~cv2.dilate(mask.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1)), iterations=2).astype(bool)
    ring_far = cv2.dilate(mask.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4 * r + 1, 4 * r + 1))) > 0
    ref = ring_far & ~near & (height < HEIGHT_M)
    if ref.sum() < 500:
        return mask
    base = np.median(g[ref])
    dark = near & (height < HEIGHT_M) & (g < 0.82 * base)
    dark = cv2.morphologyEx(dark.astype(np.uint8), cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    grown = cv2.dilate(mask.astype(np.uint8), np.ones((5, 5), np.uint8))
    cnt, lab = cv2.connectedComponents(np.maximum(dark, grown))
    keep = np.unique(lab[mask]); keep = keep[keep > 0]
    return mask | (np.isin(lab, keep) & dark.astype(bool))


# ----------------------------------------------------------------------------------------------------------------------------------
# rebuilding the floor under him from the neighbouring stops

FILL_N = 3072                     # the fill is made at this (virtual) size of the straight-down view: about the pano's own resolution on the floor
MAX_VIEWS = 5                     # neighbours compared for the floor
MIN_SPREAD_DEG = 35               # two neighbours only vouch for the floor together if they look at it from directions at least this far apart
DIFF_TOL = 30                     # this stop's floor differs from the neighbours' agreed floor by more than this (0-255, blurred): something lies on it
AGREE_BLUR, AGREE_TOL = 6, 18     # two views agree on a pixel when their blurred colours differ by less than this (0-255)
HALO_M = 0.45                     # width of the floor ring around the hole that the neighbours are lined up and colour-matched against


def roty(t):
    c, s = np.cos(t), np.sin(t)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


class Fill:
    """The hole of one stop in a crop of the straight-down view at FILL_N size, and everything needed to look at the same floor from another stop."""
    def __init__(self, mask1024, up, cam_h, margin_m=1.0):
        self.up, self.h = up, cam_h
        self.f = FILL_N / 2 / np.tan(np.radians(VIEW_FOV) / 2)
        s = FILL_N / VIEW_N
        ys, xs = np.nonzero(mask1024)
        mg = int(margin_m * self.f / cam_h)
        self.x0, self.x1 = max(0, int(xs.min() * s) - mg), min(FILL_N, int((xs.max() + 1) * s) + mg)
        self.y0, self.y1 = max(0, int(ys.min() * s) - mg), min(FILL_N, int((ys.max() + 1) * s) + mg)
        self.hole = cv2.resize(mask1024.astype(np.uint8), (FILL_N, FILL_N), interpolation=cv2.INTER_LINEAR)[self.y0:self.y1, self.x0:self.x1] > 0
        c, r = np.meshgrid(np.arange(self.x0, self.x1) + 0.5, np.arange(self.y0, self.y1) + 0.5)
        d = np.stack([(c - FILL_N / 2) / self.f, -np.ones_like(c), (FILL_N / 2 - r) / self.f], -1)
        self.dirs = (d / np.linalg.norm(d, axis=-1, keepdims=True)).astype(np.float32)
        self.P = self.dirs * (cam_h / -self.dirs[..., 1])[..., None]           # the floor point each pixel sees, in this stop's level frame
        self.shape = self.hole.shape


def sample_from(fill, pano_j, up_j, h_j, az, d, turn, mask_j1024, dyaw=0.0, dd=0.0, dl=0.0, nf_j=None):
    """What stop j sees at the floor points of `fill`: (bgr crop, valid mask). az/d/turn: the link from this stop to j (degrees, metres).
    With nf_j (j's floor mask of the segmentation step) the valid mask also says: j itself sees FLOOR there (a pasted shelf or box is not floor)."""
    a = np.radians(az)
    C = np.array([np.sin(a) * (d + dd) - np.cos(a) * dl, h_j - fill.h, np.cos(a) * (d + dd) + np.sin(a) * dl], np.float32)
    D = (fill.P - C) @ roty(np.radians(turn + dyaw)).T.astype(np.float32)
    rj = D @ rot_to_y(up_j).astype(np.float32)                                  # level frame of j -> its pano frame
    u, v = pano_uv(rj, *pano_j.shape[:2])
    img = cv2.remap(pano_j, u, v, cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
    Dn = D / np.linalg.norm(D, axis=-1, keepdims=True)
    down = -Dn[..., 1]                                                          # cos of the angle from straight down at j
    ok = (down > np.cos(np.radians(72))) & (down < 0.999)
    if mask_j1024 is not None:                                                  # where j's own capturer stands (its nadir cone)
        f1 = VIEW_N / 2 / np.tan(np.radians(VIEW_FOV) / 2)
        col = f1 * Dn[..., 0] / np.maximum(down, 1e-3) + VIEW_N / 2 - 0.5; row = VIEW_N / 2 - 0.5 - f1 * Dn[..., 2] / np.maximum(down, 1e-3)
        inside = (col > 0) & (col < VIEW_N - 1) & (row > 0) & (row < VIEW_N - 1)
        mk = cv2.remap(mask_j1024.astype(np.uint8), col.astype(np.float32), row.astype(np.float32), cv2.INTER_NEAREST, borderValue=0) > 0
        ok &= ~(inside & mk)
    if nf_j is not None:
        n = nf_j.shape[0]; f2 = n / 2 / np.tan(np.radians(130.0) / 2); dz = np.maximum(-rj[..., 1] / np.linalg.norm(rj, axis=-1), 1e-3)
        r1 = rj / np.linalg.norm(rj, axis=-1, keepdims=True)
        col = (f2 * r1[..., 0] / dz + n / 2 - 0.5).astype(np.float32); row = (n / 2 - 0.5 - f2 * r1[..., 2] / dz).astype(np.float32)
        fl = cv2.remap(cv2.dilate(nf_j, np.ones((5, 5), np.uint8)), col, row, cv2.INTER_NEAREST, borderValue=0) > 0
        ok &= fl
    return img, ok


def _gray(img, s=2.0):
    return cv2.GaussianBlur(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32), (0, 0), s)


def align(fill, view_crop, ring, pano_j, up_j, h_j, az, d, turn, mask_j1024, nf_j=None):
    """Line stop j up with this stop on the floor ring (small search over its heading and distance); returns (parameters, correlation, image, valid)."""
    A = _gray(view_crop)
    # coarse-to-fine on a quarter size crop: much cheaper than the full one
    s = 4
    sub = lambda a: a[::s, ::s]
    sf = Fill.__new__(Fill); sf.__dict__.update(fill.__dict__)
    sf.P = fill.P[::s, ::s]; sf.hole = fill.hole[::s, ::s]
    As = sub(A); rs = ring[::s, ::s]
    def score(dyaw, dd, dl):
        img, ok = sample_from(sf, pano_j, up_j, h_j, az, d, turn, mask_j1024, dyaw, dd, dl)
        m = rs & ok
        if m.sum() < 300:
            return -1, 0
        g = _gray(img, 1.0)
        a, b = As[m], g[m]
        return float(np.corrcoef(a, b)[0, 1]), int(m.sum())
    best = (-2, 0, 0, 0)
    for dyaw in (-5, -2.5, 0, 2.5, 5):
        for dd in (-0.4, -0.2, 0, 0.2, 0.4):
            for dl in (-0.2, 0, 0.2):
                sc, n = score(dyaw, dd, dl)
                if sc > best[0]:
                    best = (sc, dyaw, dd, dl)
    sc, dyaw, dd, dl = best
    img, ok = sample_from(fill, pano_j, up_j, h_j, az, d, turn, mask_j1024, dyaw, dd, dl, nf_j)
    return dict(dyaw=dyaw, dd=dd, dl=dl, ncc=sc), img, ok


def floor_label_crop(fill, nf, fov=130.0):
    """The floor mask of the segmentation step (frames/depth/<id>_nf.png: a straight-down view cut from the pano as it is, fov 130) in this crop."""
    n = nf.shape[0]; f2 = n / 2 / np.tan(np.radians(fov) / 2)
    r = fill.dirs @ rot_to_y(fill.up).astype(np.float32)                         # level frame -> pano frame
    dz = np.maximum(-r[..., 1], 1e-3)
    col = (f2 * r[..., 0] / dz + n / 2 - 0.5).astype(np.float32); row = (n / 2 - 0.5 - f2 * r[..., 2] / dz).astype(np.float32)
    return cv2.remap(nf, col, row, cv2.INTER_NEAREST, borderValue=0) > 0


def dominant_smooth(img, band, grad_max=12.0):
    """The smooth pixels of `band` (floors and walls are smooth; shelves and tools are not): the colours a soft fill may take from the edge of the hole."""
    g = cv2.GaussianBlur(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32), (0, 0), 1.5)
    grad = np.hypot(cv2.Sobel(g, cv2.CV_32F, 1, 0), cv2.Sobel(g, cv2.CV_32F, 0, 1))
    sm = band & (grad < grad_max)
    return sm if sm.sum() > 800 else band


def clean_regions(m, close_px=31, open_px=15, min_px=12000):
    """Whole patches only: close small gaps, drop thin bits and small islands, so that the copied floor never shows up as speckle."""
    e = lambda r: cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (r, r))
    u = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_CLOSE, e(close_px))
    u = cv2.morphologyEx(u, cv2.MORPH_OPEN, e(open_px))
    cnt, lab, st, _ = cv2.connectedComponentsWithStats(u)
    keep = [k for k in range(1, cnt) if st[k, cv2.CC_STAT_AREA] >= min_px]
    return np.isin(lab, keep)


def fill_gain(view_crop, img, ring_ok):
    """Per-channel gain that makes the neighbour's floor as bright as this stop's floor on the ring."""
    g = np.ones(3, np.float32)
    for k in range(3):
        a, b = view_crop[..., k][ring_ok].astype(np.float32), img[..., k][ring_ok].astype(np.float32)
        if len(a) > 200 and np.median(b) > 5:
            g[k] = np.clip(np.median(a) / np.median(b), 0.6, 1.6)
    return g


def soft_fill(img, known, sigma=40):
    """A smooth fill from the colours around: the known pixels, blurred at a wide and then a narrower scale (normalised so that holes do not go dark)."""
    k = known.astype(np.float32)
    out = img.astype(np.float32)
    for sg in (sigma * 2, sigma, sigma / 3):
        num = cv2.GaussianBlur(out * k[..., None], (0, 0), sg); den = cv2.GaussianBlur(k, (0, 0), sg)
        est = num / np.maximum(den[..., None], 1e-4)
        fillable = (den > 1e-3)
        out = np.where((k[..., None] > 0), out, np.where(fillable[..., None], est, out))
        k = np.maximum(k, fillable.astype(np.float32) * 0 + k)
        k = np.where(fillable, 1.0, k).astype(np.float32)
    return np.clip(out, 0, 255).astype(np.uint8)


def fill_stop(i, pano_i, up_i, h_i, mask1024, nbrs, load, log=None, ncc_min=0.12, floor_mask=None):
    """Rebuild the floor under the capturer from the neighbouring stops.
    nbrs: [{'j', 'az', 'd', 'turn'}] in priority order (previous, next, then nearest); load(j) -> (pano, up, cam_h, mask1024 of that stop, floor mask of that stop or None).
    Returns (new pano with the hole filled, stats dict, debug dict)."""
    H, W = pano_i.shape[:2]
    fill = Fill(mask1024, up_i, h_i)
    vc = level_view_crop(pano_i, fill)                                         # this stop's own floor in the crop (hole still has him)
    hole = fill.hole
    ker = lambda r: cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    rpx = lambda m: max(1, int(m * fill.f / fill.h))
    ring = (cv2.dilate(hole.astype(np.uint8), ker(rpx(HALO_M))) > 0) & ~(cv2.dilate(hole.astype(np.uint8), ker(rpx(0.05))) > 0)
    patch = np.zeros_like(vc); got = np.zeros(hole.shape, bool); used = []
    # up to MAX_VIEWS neighbours that line up with this stop's floor; a floor pixel is trusted only where two of them agree (anything above the floor
    # lands in different places from different stops, so it does not agree and is not copied)
    good = []
    for nb in nbrs:
        if len(good) >= MAX_VIEWS:
            break
        pj, uj, hj, mj, nfj = load(nb["j"])
        par, img, ok = align(fill, vc, ring, pj, uj, hj, nb["az"], nb["d"], nb["turn"], mj)        # (the segmentation's floor mask is not used: it calls white walls floor and concrete not)
        if par["ncc"] < ncc_min:
            used.append(dict(j=nb["j"], ncc=round(par["ncc"], 2), used=0, why="does not line up with this stop's floor"))
            continue
        g = fill_gain(vc, img, ring & ok)
        img = np.clip(img.astype(np.float32) * g, 0, 255).astype(np.uint8)
        good.append((nb["j"], par, img, ok, cv2.GaussianBlur(img, (0, 0), AGREE_BLUR).astype(np.float32), nb["az"]))
    import itertools
    cnt = {j: 0 for j, *_ in good}
    # the floor as two or more stops agree on it, over the whole crop (not only the hole)
    cons = np.zeros_like(vc); cons_ok = np.zeros(hole.shape, bool)
    for (ja, pa, ia, oa, ba, aza), (jb, pb, ib, ob, bb, azb) in itertools.combinations(good, 2):
        if abs((aza - azb + 180) % 360 - 180) < MIN_SPREAD_DEG:                      # two stops on the same side see an object above the floor in the same wrong place: they would 'agree'
            continue
        agree = clean_regions(oa & ob & (np.abs(ba - bb).mean(-1) < AGREE_TOL)) & ~cons_ok
        cons[agree] = ia[agree]; cons_ok |= agree; cnt[ja] += int((agree & hole).sum())
    for j, par, img, ok, _, _az in good:
        used.append(dict(j=j, ncc=round(par["ncc"], 2), dyaw=par["dyaw"], dd=par["dd"], dl=par["dl"], used=cnt[j]))
    # what is left of him: floor that the other stops agree on, but that THIS stop sees differently right next to the hole (a leg, an arm, a head,
    # a shadow lying on the floor). It is part of the hole too, and the hole grows over it.
    grown = 0.0
    own_b = cv2.GaussianBlur(vc, (0, 0), 3).astype(np.float32); cons_b = cv2.GaussianBlur(cons, (0, 0), 3).astype(np.float32)
    differs = cv2.morphologyEx(((np.abs(own_b - cons_b).mean(-1) > DIFF_TOL) & cons_ok).astype(np.uint8), cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    cntc, labc = cv2.connectedComponents(differs)
    near = cv2.dilate(hole.astype(np.uint8), ker(rpx(0.15))) > 0
    keep = np.unique(labc[near & (differs > 0)]); keep = keep[keep > 0]
    if len(keep):
        extra = np.isin(labc, keep)
        hole2 = cv2.dilate(cv2.morphologyEx((hole | extra).astype(np.uint8), cv2.MORPH_CLOSE, ker(rpx(0.1))), ker(rpx(0.02))) > 0
        if hole2.sum() * (fill.h / fill.f) ** 2 <= 4.5:                              # never let it run away
            grown = float((hole2.sum() - hole.sum()) * (fill.h / fill.f) ** 2)
            hole = hole2; fill.hole = hole
    ring = (cv2.dilate(hole.astype(np.uint8), ker(rpx(HALO_M))) > 0) & ~(cv2.dilate(hole.astype(np.uint8), ker(rpx(0.05))) > 0)
    band = (cv2.dilate(hole.astype(np.uint8), ker(rpx(0.3))) > 0) & ~(cv2.dilate(hole.astype(np.uint8), ker(rpx(0.02))) > 0)
    resid = cv2.morphologyEx(((np.abs(own_b - cons_b).mean(-1) > DIFF_TOL) & cons_ok & band).astype(np.uint8), cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    leftover = float(resid.sum()) * (fill.h / fill.f) ** 2           # floor the other stops agree on that this stop still sees differently, right next to the hole
    got = cons_ok & hole; patch = cons
    ring_floor = cons_ok & ring                                                      # floor round the hole that two stops agree on
    known = (~hole) | got
    base = np.where(got[..., None], patch, vc)
    src = clean_regions(ring_floor, 15, 7, 3000)                                    # what no neighbour saw: a soft fill from the FLOOR round it (not the shelves)
    src = src if src.sum() > 4000 else dominant_smooth(vc, (cv2.dilate(hole.astype(np.uint8), ker(rpx(0.2))) > 0) & ~hole)             # nobody vouched for any floor: the smooth colours right at the edge of the hole
    filled = soft_fill(base, src | got)
    mix = np.where(got[..., None], patch, filled)
    mix = np.where(hole[..., None], mix, vc)
    # Poisson blend: the patch takes its colour at the edge from the floor around the hole (a seam disappears), its detail stays
    # only the copied floor is Poisson-blended (its colour at the edge follows the floor around the hole; a soft fill taken from the same edge would
    # drag the colours of shelves and tools into the middle of the floor)
    hm = (cv2.erode(got.astype(np.uint8), np.ones((3, 3), np.uint8)) * 255)
    ys, xs = np.nonzero(hm)
    if len(ys):
        cx, cy = int((xs.min() + xs.max()) / 2), int((ys.min() + ys.max()) / 2)
        try:
            blended = cv2.seamlessClone(np.ascontiguousarray(mix), np.ascontiguousarray(np.where(hole[..., None], mix, vc)), hm, (cx, cy), cv2.NORMAL_CLONE)
        except cv2.error:
            blended = mix
        mix = np.where(got[..., None], blended, mix)
    new_pano = paste_back(pano_i, mix, hole, fill)
    stats = dict(hole_px_view=int(hole.sum()), real_pct=round(100.0 * got.sum() / max(1, hole.sum()), 1), blur_pct=round(100.0 * (hole & ~got).sum() / max(1, hole.sum()), 1),
                 neighbours=used, hole_m2=round(float(hole.sum()) * (fill.h / fill.f) ** 2, 2), grown_m2=round(grown, 2), leftover_m2=round(leftover, 3))
    return new_pano, stats, dict(before=vc, after=mix, hole=hole, got=got, fill=fill, floor_ref=(src | got))


def level_view_crop(pano, fill):
    u, v = pano_uv(fill.dirs @ rot_to_y(fill.up).astype(np.float32), *pano.shape[:2])
    return cv2.remap(pano, u, v, cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)


def paste_back(pano, crop_img, hole, fill, feather_px=6):
    """Write the filled crop into the pano wherever the pano sees the hole. Feathered at the edge so that no line shows."""
    H, W = pano.shape[:2]
    r0 = int(H * 0.5)                                                              # nadir cone only: rows below the horizon
    az = ((np.arange(W) + 0.5) / W * 2 * np.pi - np.pi).astype(np.float32); el = (np.pi / 2 - (np.arange(r0, H) + 0.5) / H * np.pi).astype(np.float32)
    A, E = np.meshgrid(az, el)
    r = np.stack([np.cos(E) * np.sin(A), np.sin(E), np.cos(E) * np.cos(A)], -1)
    L = r @ rot_to_y(fill.up).T.astype(np.float32)
    dy = np.maximum(-L[..., 1], 1e-3)
    col = (fill.f * L[..., 0] / dy + FILL_N / 2 - fill.x0 - 0.5).astype(np.float32)
    row = (FILL_N / 2 - fill.y0 - 0.5 - fill.f * L[..., 2] / dy).astype(np.float32)
    inside = (L[..., 1] < -0.3) & (col > 1) & (col < fill.shape[1] - 2) & (row > 1) & (row < fill.shape[0] - 2)
    wmask = cv2.GaussianBlur(hole.astype(np.float32), (0, 0), feather_px)
    w = cv2.remap(wmask, col, row, cv2.INTER_LINEAR, borderValue=0) * inside
    w = np.where(cv2.remap(hole.astype(np.float32), col, row, cv2.INTER_LINEAR, borderValue=0) > 0.5, 1.0, w) * inside
    new = cv2.remap(crop_img, col, row, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    out = pano.copy()
    sub = out[r0:H]
    sub[:] = (sub.astype(np.float32) * (1 - w[..., None]) + new.astype(np.float32) * w[..., None]).astype(np.uint8)
    return out
