"""Put every ring on the real floor and hide the ones behind things, using per-photo depth.

1. Tie each depth view to the real 3D points COLMAP found (map_stops.py), so depth is in map units
   and every view of every stop agrees:  1/z = a * disparity + b, fitted robustly per view.
2. Floor: from the depth, the lowest well-supported horizontal layer around each stop = camera height.
3. Each ring goes where the floor under the other stop appears:  el = atan(camera height / distance).
4. Occlusion: if the depth in a ring's direction is clearly shorter than the ring's distance,
   something (a shelf, the generator) is in the way, so that ring is hidden.

Run after map_stops.py and depth_panos.py:   python place_rings.py
Updates frames/poses.json in place (adds el, occ per link and cam_h per stop).
"""
import argparse, json, os
import cv2
import numpy as np
import pycolmap
from hop_geometry import level_matrix
from depth_panos import view_rays

F = np.diag([1.0, -1.0, 1.0])


def bearings(uv, w, h):
    th = uv[:, 0] / w * 2 * np.pi - np.pi
    ph = np.pi / 2 - uv[:, 1] / h * np.pi
    return np.stack([np.cos(ph) * np.sin(th), np.sin(ph), np.cos(ph) * np.cos(th)], 1)


def view_basis(v):
    """Rows: view right, up, forward, in pano coords (same rotation depth_panos used)."""
    p, yv = np.radians(v["pitch"]), np.radians(v["yaw"])
    Rx = np.array([[1, 0, 0], [0, np.cos(p), np.sin(p)], [0, -np.sin(p), np.cos(p)]])
    Ry = np.array([[np.cos(yv), 0, np.sin(yv)], [0, 1, 0], [-np.sin(yv), 0, np.cos(yv)]])
    return (Ry @ Rx).T          # maps pano ray -> view coords


def to_view_px(rays, v, n):
    """Pano rays -> (col, row, z-component) in the n x n pinhole view; z <= 0 means behind."""
    q = rays @ view_basis(v).T
    f = n / 2 / np.tan(np.radians(v["fov"]) / 2)
    z = q[:, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        col = q[:, 0] / z * f + n / 2 - 0.5
        row = n / 2 - 0.5 - q[:, 1] / z * f
    return col, row, z


def robust_affine(x, y, iters=300, thr=0.08, seed=0):
    """Fit y = a*x + b with RANSAC on relative error, then least squares on inliers."""
    rng = np.random.default_rng(seed)
    best, best_in = None, None
    for _ in range(iters):
        i, j = rng.choice(len(x), 2, replace=False)
        if abs(x[i] - x[j]) < 1e-6:
            continue
        a = (y[i] - y[j]) / (x[i] - x[j]); b = y[i] - a * x[i]
        if a <= 0:
            continue
        inl = np.abs(a * x + b - y) < thr * np.abs(y)
        if best_in is None or inl.sum() > best_in.sum():
            best, best_in = (a, b), inl
    if best is None or best_in.sum() < 8:
        return None
    A = np.stack([x[best_in], np.ones(best_in.sum())], 1)
    a, b = np.linalg.lstsq(A, y[best_in], rcond=None)[0]
    return (float(a), float(b), int(best_in.sum())) if a > 0 else None


DMIN, DMAX = 0.2, 60.0     # metres; 8-bit log encoding: r = DMIN * (DMAX/DMIN) ** (v/255)


def write_depth_png(path, ranges, views, n, cam_h, L, W=1024, H=512, fmasks=None):
    """Equirect depth (pano's own frame, same layout as the photo) for the viewer's occlusion mesh.
    Red = 8-bit log depth, green = 255 where that pixel is visible floor (its 3D point sits at floor height).
    Returns the floor mask too."""
    import cv2
    az = np.radians((np.arange(W) + 0.5) / W * 360 - 180); el = np.radians(90 - (np.arange(H) + 0.5) / H * 180)
    AZ, EL = np.meshgrid(az, el)
    rays = np.stack([np.cos(EL) * np.sin(AZ), np.sin(EL), np.cos(EL) * np.cos(AZ)], -1).reshape(-1, 3)
    out = np.full(len(rays), DMAX, np.float32); best = np.full(len(rays), -1.0); sem = np.zeros(len(rays), bool)
    for vi, v in enumerate(views):
        c, r_, zc = to_view_px(rays, v, n)
        ok = (zc > 0.2) & (c >= 0) & (c < n - 1) & (r_ >= 0) & (r_ < n - 1)
        # prefer the view where the pixel is most central
        cen = np.where(ok, zc, -1.0)
        take = cen > best
        out[take] = ranges[vi][r_[take].astype(int), c[take].astype(int)]
        if fmasks is not None:
            sem[take] = fmasks[vi][r_[take].astype(int), c[take].astype(int)] > 127
        best[take] = cen[take]
    # below the views (straight down) the floor plane; above them, far away
    lv = rays @ L.T
    floor = np.where(lv[:, 1] < -0.05, cam_h / np.maximum(-lv[:, 1], 0.05), DMAX)
    out = np.where(best < 0, floor, out)
    out = np.clip(out, DMIN, DMAX)
    # Depth Anything puts the floor a bit too close. Nothing can be seen *through* the floor, so cap every ray
    # at the floor plane, and where the floor model says "floor", snap the depth onto the plane exactly
    # (4% beyond, so a ring lying on the floor is never hidden by the floor itself).
    fd = np.where(lv[:, 1] < -0.02, cam_h / np.maximum(-lv[:, 1], 0.02), DMAX)
    out = np.minimum(out, fd * 1.04)
    if fmasks is not None:
        out = np.where(sem & (lv[:, 1] < -0.03), fd * 1.04, out)
    out = np.clip(out, DMIN, DMAX)
    v8 = cv2.medianBlur(np.round(np.log(out / DMIN) / np.log(DMAX / DMIN) * 255).astype(np.uint8).reshape(H, W), 3)
    # floor: the 3D point is within a distance-dependent band of the floor plane
    y = (lv[:, 1] * out)                                         # levelled height of each pixel's point, metres
    tol = 0.10 + 0.04 * out
    if fmasks is not None:     # semantic floor (floor_panos.py) is far more reliable than depth height
        floor_m = (sem & (lv[:, 1] < -0.03)).reshape(H, W).astype(np.uint8) * 255
    else:
        floor_m = ((np.abs(y + cam_h) < tol) & (lv[:, 1] < -0.05) & (best >= 0)).reshape(H, W).astype(np.uint8) * 255
    floor_m = cv2.morphologyEx(floor_m, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))   # fill specks and thin gaps in the floor
    floor_m = cv2.morphologyEx(floor_m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    img = np.zeros((H, W, 3), np.uint8); img[..., 2] = v8; img[..., 1] = floor_m   # BGR on disk -> R depth, G floor
    cv2.imwrite(path, img)
    return floor_m > 0, out.reshape(H, W)


def floor_fraction(mask, L, centre, radius=0.35, k=24):
    """Share of a ring's footprint (circle on the floor, levelled metres) that is visible floor in this pano."""
    H, W = mask.shape
    ang = np.linspace(0, 2 * np.pi, k, endpoint=False)
    pts = [centre] + [centre + radius * np.array([np.cos(t), 0, np.sin(t)]) for t in ang]
    hits = 0
    for p in pts:
        d = L.T @ (p / np.linalg.norm(p))                         # back to the pano's own frame
        az = np.degrees(np.arctan2(d[0], d[2])); el = np.degrees(np.arcsin(np.clip(d[1], -1, 1)))
        u = int((az + 180) / 360 * W) % W; v = min(H - 1, max(0, int((90 - el) / 180 * H)))
        hits += mask[v, u]
    return hits / len(pts)


def sparse_floor(rec, up, cams, min_points=1500):
    """Where the floor is, from the sparse map itself (no depth model): the height (along `up`, in COLMAP units) at which the cloud of triangulated points starts. A floor has speckle
    and clutter on it, so the point density below the camera jumps from nothing to a plateau at the floor and stays there; the first two bins that reach half of the plateau mark it.
    The depth model's own floor (the densest layer in its ranges) picked a layer of clutter half a metre above the real floor on the first test walk, which lifted the floor plane and
    made everything above it 0.5 m too short. None if there are too few points or no clear edge."""
    P = np.array([p.xyz for p in rec.points3D.values() if p.track.length() >= 3 and p.error < 1.0])
    return sparse_floor_pts(P, up, cams, min_points)


def sparse_floor_pts(P, up, cams, min_points=1500):
    """sparse_floor on an array of points (N, 3)."""
    up = np.asarray(up, float); up = up / np.linalg.norm(up)
    if len(P) < min_points or not cams:
        return None
    h = P @ up
    hc = float(np.median([c @ up for c in cams.values()]))
    below = h[h < hc]
    if len(below) < min_points // 2:
        return None
    lo = float(np.percentile(below, 0.2)); width = (hc - lo) / 70.0
    cnt, edges = np.histogram(below, bins=np.arange(lo, hc, width))
    plateau = float(np.median(cnt[len(cnt) // 4: 3 * len(cnt) // 4 + 1]))
    if plateau < 5:
        return None
    for k in range(len(cnt) - 1):
        if cnt[k] >= 0.5 * plateau and cnt[k + 1] >= 0.5 * plateau:
            return float(edges[k])
    return None


LAST_CEILING = {}        # stop index -> metres from the floor edge to the top of the dense map's point cloud near that stop (set by vio_cam_heights; a rough roof height, for scale_priors.py)


def vio_cam_heights(frames, scale, cams_stops, ups, ids, log=print, window=40.0):
    """Each stop's height above the floor in metres, from the dense 1 Hz map of vio_map.py (55,000 points instead of the stops-only map's 2,500). It is done LOCALLY, per stop, in a window of
    +-`window` seconds of the walk: the dense map is a chain of pictures and drifts in scale over a long walk (walk 2: the stop map and the dense map agree to 3 % over the first minute and to
    74 % over the whole walk), so the floor is found from the points that pictures of that stretch saw (sparse_floor), up is the pictures' own up carried into the map, and metres per map unit come
    from the stops of that stretch that are in both maps. {stop index: metres}, for the stops where that worked, or None."""
    vio = os.path.join(frames, "vio")
    if not os.path.exists(os.path.join(vio, "poses.npz")) or not os.path.isdir(os.path.join(vio, "sparse")):
        return None
    try:
        V = np.load(os.path.join(vio, "poses.npz"))
        tname = {str(n): float(t_) for n, t_ in zip(V["names"], V["t"])}
        models = [os.path.join(vio, "sparse", d) for d in sorted(os.listdir(os.path.join(vio, "sparse")))]
        rv = max([pycolmap.Reconstruction(m) for m in models], key=lambda r: r.num_reg_images())
        Cv, Rv, tim = {}, {}, {}
        for iid, im in rv.images.items():
            T = im.cam_from_world() if callable(im.cam_from_world) else im.cam_from_world
            R = T.rotation.matrix(); Cv[im.name] = -R.T @ T.translation; Rv[im.name] = R; tim[iid] = tname.get(im.name)
        Fm = np.diag([1.0, -1.0, 1.0])
        stop_names = {k: "s_" + f"{ids[k]}.jpg" for k in cams_stops}
        have = [k for k in cams_stops if stop_names[k] in Cv and tname.get(stop_names[k]) is not None]
        if len(have) < 4:
            return None
        # the points once: position and the median time of the pictures that saw them
        P, Pt = [], []
        for p3 in rv.points3D.values():
            if p3.track.length() >= 3 and p3.error < 1.0:
                ts = [tim[e.image_id] for e in p3.track.elements if tim.get(e.image_id) is not None]
                if ts:
                    P.append(p3.xyz); Pt.append(float(np.median(ts)))
        P = np.array(P); Pt = np.array(Pt)
        names_t = {n: tname[n] for n in Cv if n in tname}
        out, good = {}, set()
        for k in have:
            ts_ = tname[stop_names[k]]
            near = [i for i in have if abs(tname[stop_names[i]] - ts_) <= window]
            if len(near) < 4:
                continue
            up = np.mean([Rv[stop_names[i]].T @ (Fm @ np.asarray(ups[i], float)) for i in near], axis=0); up /= np.linalg.norm(up)
            ratios = [np.linalg.norm(Cv[stop_names[i]] - Cv[stop_names[j]]) / np.linalg.norm(cams_stops[i] - cams_stops[j])
                      for n_, i in enumerate(near) for j in near[n_ + 1:] if np.linalg.norm(cams_stops[i] - cams_stops[j]) > 1e-6]
            k_unit = float(np.median(ratios)); spread = 1.4826 * float(np.median(np.abs(np.array(ratios) - k_unit))) / k_unit
            m = np.abs(Pt - ts_) <= window
            cams_w = {n: Cv[n] for n, t_ in names_t.items() if abs(t_ - ts_) <= window}
            floor = sparse_floor_pts(P[m], up, cams_w, min_points=600) if m.sum() > 600 else None
            if floor is None or spread > 0.15:
                continue
            out[k] = float((Cv[stop_names[k]] @ up - floor) * scale / k_unit)
            hh = P[m] @ up
            LAST_CEILING[k] = float((np.percentile(hh[hh > Cv[stop_names[k]] @ up], 98) - floor) * scale / k_unit) if (hh > Cv[stop_names[k]] @ up).sum() > 100 else None
            if spread <= 0.07:
                good.add(k)                                                # the two maps agree to 7 % here: the stretches whose height is believed
        if len(out) < max(3, len(have) // 2):
            log(f"vio map: a clear floor edge and agreement with the stop map in only {len(out)} of {len(have)} stretches: not used for the camera height")
            return None
        vals = [out[k] for k in good] if len(good) >= 3 else list(out.values())
        med = float(np.median(vals))
        log(f"camera height from the dense map's floor edge (local, +-{window:.0f} s): {len(out)} of {len(have)} stops had an edge, {len(good)} where the two maps agree to 7 %; the stick height used {med:.2f} m "
            f"(these stretches: {min(vals):.2f}-{max(vals):.2f}; all: {min(out.values()):.2f}-{max(out.values()):.2f})")
        for k in have:                                                   # the stick stays at about one height: stretches where the maps do not agree (or have no edge) get the median of those that do
            if len(good) >= 3 and k not in good:
                out[k] = med
            else:
                out.setdefault(k, med)
        return out
    except Exception as e:                                             # never let the experiment-grade estimate break the build
        log(f"vio map not usable for the camera height ({str(e)[:100]})")
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", default="frames")
    ap.add_argument("--sparse", default=None, help="COLMAP model folder (default: frames/colmap/best, the model map_stops.py chose)")
    ap.add_argument("--occ", type=float, default=0.85, help="hide a ring if depth < this fraction of its distance")
    ap.add_argument("--prior-range", type=float, nargs=2, default=None, help="experiment, off by default: when the camera height has to come from the depth model's floor layer, a height outside this range (m: a person holding a stick) is moved to its nearest end. On walk 2 (1.45 m found, 1.8 used) it cut the door and ceiling errors by a third but moved the median error from 27.1 to 28.6 %, so it is not applied")
    ap.add_argument("--depth-floor", action="store_true", help="take the camera height from the depth model's floor layer (the old way) instead of the sparse map's floor edge")
    ap.add_argument("--cam-h", type=float, default=None, help="experiment: use this camera height above the floor (map metres) for every stop instead of the one found from the depth")
    a = ap.parse_args()
    src = os.path.join(a.frames, "poses_map.json")              # always start from the untouched map, so reruns don't slide rings twice
    poses = json.load(open(src if os.path.exists(src) else os.path.join(a.frames, "poses.json")))
    scale = poses["scale"]
    ups = json.load(open(os.path.join(a.frames, "up.json")))
    V = json.load(open(os.path.join(a.frames, "depth", "views.json")))
    n, views = V["size"], V["views"]
    sparse = a.sparse or (os.path.join(a.frames, "colmap", "best") if os.path.isdir(os.path.join(a.frames, "colmap", "best"))
                          else os.path.join(a.frames, "colmap", "sparse", "0"))
    rec = pycolmap.Reconstruction(sparse)
    # a stop the motion track placed (map_stops.py vio_place, "by": "vio") has no COLMAP camera and no map points: it is not counted against the model
    placed = sum(1 for nd in poses["nodes"] if nd.get("x") is not None and nd.get("by") != "vio")
    if rec.num_reg_images() < placed:
        raise SystemExit(f"{sparse} has {rec.num_reg_images()} registered images but the map placed {placed} stops: wrong COLMAP model; rerun map_stops.py")
    ids = [nd["id"] for nd in poses["nodes"]]
    stats, per_stop, all_fits, cams = [], {}, [], {}
    for im in rec.images.values():
        k = int(os.path.splitext(im.name)[0]) - 1
        sid = ids[k]
        T = im.cam_from_world() if callable(im.cam_from_world) else im.cam_from_world
        R = T.rotation.matrix(); C = -R.T @ T.translation
        cams[k] = C
        cam = rec.cameras[im.camera_id]
        uv, X = [], []
        for p in im.points2D:
            if p.has_point3D():
                uv.append(p.xy); X.append(rec.points3D[p.point3D_id].xyz)
        uv, X = np.array(uv), np.array(X)
        rays = bearings(uv, cam.width, cam.height)                     # our coords
        rng_m = np.linalg.norm(X - C, axis=1) * scale                  # metres to each 3D point
        L = level_matrix(ups[k])
        depth_views, fits = [], []
        for vi, v in enumerate(views):
            disp = np.load(os.path.join(a.frames, "depth", f"{sid}_v{vi}.npy")).astype(np.float32)
            col, row, zc = to_view_px(rays, v, n)
            ok = (zc > 0.2) & (col >= 0) & (col < n - 1) & (row >= 0) & (row < n - 1)
            fit = None
            if ok.sum() >= 12:
                d_at = disp[row[ok].round().astype(int), col[ok].round().astype(int)]
                zdepth = rng_m[ok] * zc[ok]                             # metres along the view axis
                fit = robust_affine(d_at, 1.0 / zdepth, seed=k * 10 + vi)
                if fit and (fit[2] < 15 or fit[2] < 0.5 * ok.sum()):
                    fit = None                                          # too few points agree: don't trust this view
            fits.append(fit)
            depth_views.append(disp)
        good = [f for f in fits if f]
        if not good:
            # few 3D points in every view (bare walls, blur): fit all four views together with looser limits,
            # else borrow the typical fit from the other stops so this stop still gets depth, occlusion and the glide
            pd, pz = [], []
            for vi, v in enumerate(views):
                col, row, zc = to_view_px(rays, v, n)
                ok = (zc > 0.2) & (col >= 0) & (col < n - 1) & (row >= 0) & (row < n - 1)
                pd.append(depth_views[vi][row[ok].round().astype(int), col[ok].round().astype(int)]); pz.append(rng_m[ok] * zc[ok])
            pd, pz = np.concatenate(pd), np.concatenate(pz)
            f = robust_affine(pd, 1.0 / pz, thr=0.15, seed=k) if len(pd) >= 8 else None
            if f is None and all_fits:
                f = (float(np.median([x[0] for x in all_fits])), float(np.median([x[1] for x in all_fits])), 0)
            if f is None:
                print(sid, "no depth fit at all, skipped"); continue
            print(sid, "weak depth: used", "a pooled fit" if f[2] else "the walk's typical fit")
            good = [f]; fits = [f] * len(views)
        all_fits.extend(good)
        # views with too few points borrow the median fit of the others
        med = (float(np.median([f[0] for f in good])), float(np.median([f[1] for f in good])))
        fits = [f[:2] if f else med for f in fits]
        # metric range for each view pixel, plus its levelled 3D point
        heights, ranges = [], []
        for vi, v in enumerate(views):
            aa, bb = fits[vi]
            inv = aa * depth_views[vi] + bb
            z = np.where(inv > 1e-3, 1.0 / np.maximum(inv, 1e-3), 60.0)
            r = view_rays(v["yaw"], v["pitch"], v["fov"], n)            # pano coords, unit
            zc = (r @ view_basis(v).T)[..., 2]
            rg = z / np.maximum(zc, 1e-3)
            ranges.append(rg)
            P = (r * rg[..., None]).reshape(-1, 3) @ L.T              # levelled, metres
            el = np.degrees(np.arcsin(np.clip((r.reshape(-1, 3) @ L.T)[:, 1], -1, 1)))
            heights.append(P[(el < -12) & (rg.reshape(-1) < 12), 1])
        hy = -np.concatenate(heights)                                   # drop below the camera, metres
        hist, edges = np.histogram(hy, bins=50, range=(1.0, 3.5))
        hist = np.convolve(hist, [1, 2, 3, 2, 1], "same")
        centers = (edges[:-1] + edges[1:]) / 2
        peak = float(centers[np.argmax(hist)])                          # the floor is the biggest flat layer
        sharp = float(hist.max() / (np.median(hist) + 1))
        per_stop[k] = (peak, sharp, depth_views, ranges, fits, L)
    # the stick stays at about one height for the whole walk: trust a stop's own floor only near the walk's median
    med_h = float(np.median([v[0] for v in per_stop.values() if v[1] > 2.5] or [v[0] for v in per_stop.values()]))
    print(f"camera height (median over the walk, from the depth model's floor layer): {med_h:.2f} map-metres")
    vio_h = None if a.depth_floor else vio_cam_heights(a.frames, scale, cams, ups, ids)
    if vio_h is None:
        print(f"using the depth model's floor layer for the camera height ({med_h:.2f} m)")
    else:
        print(f"(the depth model's floor layer says {med_h:.2f} m)")
    for node in poses["nodes"]:                    # a placed stop the rings could not measure keeps the walk's median height (mvs and the dollhouse read it)
        if node.get("x") is not None and node.get("cam_h") is None:
            node["cam_h"] = round(med_h, 2)
    for k, node in enumerate(poses["nodes"]):      # a stop placed by the motion track has no depth of its own: its rings stand at the walk's height, shown (2026-10-30)
        if node.get("x") is None or k in per_stop:
            continue
        for Lk in node["links"]:
            Lk["el"] = round(float(np.degrees(np.arctan2(node["cam_h"], max(Lk["d"], 0.3)))), 1)
            Lk.setdefault("occ", False); Lk.setdefault("floor", 1.0)
    for k, (peak, sharp, depth_views, ranges, fits, L) in sorted(per_stop.items()):
        sid = ids[k]
        cam_h = peak if (sharp > 2.5 and abs(peak - med_h) < 0.3) else med_h
        if vio_h is not None and k in vio_h:
            cam_h = vio_h[k]                                                                    # this stop's own height above the floor, in metres
        elif a.prior_range:
            cam_h = float(min(max(cam_h, a.prior_range[0]), a.prior_range[1]))                  # a soft physical prior: the depth model's floor layer is not trusted outside what a person can hold
        if a.cam_h:
            cam_h = a.cam_h
        node = poses["nodes"][k]
        node["cam_h"] = round(cam_h, 2)
        if LAST_CEILING.get(k):
            node["roof_m"] = round(LAST_CEILING[k], 2)
        fpaths = [os.path.join(a.frames, "depth", f"{sid}_f{vi}.png") for vi in range(len(views))]
        fm = [cv2.imread(p, cv2.IMREAD_GRAYSCALE) for p in fpaths] if all(os.path.exists(p) for p in fpaths) else None
        fmask, eqd = write_depth_png(os.path.join(a.frames, "depth", f"eq_{sid}.png"), ranges, views, n, cam_h, L, fmasks=fm)
        hid = 0
        for Lk in node["links"]:
            d = max(Lk["d"], 0.3)
            Lk["el"] = round(float(np.degrees(np.arctan2(cam_h, d))), 1)
            # depth along the ring's direction (in this pano's own, unlevelled frame)
            azr, elr = np.radians(Lk["az"]), -np.arctan2(cam_h, d)
            ray_l = np.array([np.cos(elr) * np.sin(azr), np.sin(elr), np.cos(elr) * np.cos(azr)])
            ray = L.T @ ray_l
            Hq, Wq = eqd.shape
            uq = int((np.degrees(np.arctan2(ray[0], ray[2])) + 180) / 360 * Wq) % Wq
            vq = min(Hq - 1, max(0, int((90 - np.degrees(np.arcsin(np.clip(ray[1], -1, 1)))) / 180 * Hq)))
            seen = float(np.median(eqd[max(0, vq - 2):vq + 3, max(0, uq - 2):uq + 3]))
            expect = float(np.hypot(d, cam_h))
            Lk["occ"] = bool(seen is not None and seen < a.occ * expect and not Lk["near"])
            # floor check: is the spot under that stop visible floor from here? if not, slide to the nearest
            # open floor within 0.8 m; if there is none, hide the ring (next/prev are kept so you can always move)
            azr = np.radians(Lk["az"])
            c0 = np.array([np.sin(azr) * d, -cam_h, np.cos(azr) * d])
            ff = floor_fraction(fmask, L, c0)
            if ff < 0.5:
                best_c, best_f = None, ff
                for rr in (0.3, 0.55, 0.8):
                    for t in np.linspace(0, 2 * np.pi, 12, endpoint=False):
                        c = c0 + rr * np.array([np.cos(t), 0, np.sin(t)])
                        f = floor_fraction(fmask, L, c)
                        if f > best_f + 0.05:
                            best_c, best_f = c, f
                    if best_f >= 0.6:
                        break
                if best_c is not None and best_f >= 0.5:
                    dn = float(np.hypot(best_c[0], best_c[2]))
                    Lk["az"] = round(float(np.degrees(np.arctan2(best_c[0], best_c[2]))), 1)
                    Lk["d"] = round(dn, 2); Lk["el"] = round(float(np.degrees(np.arctan2(cam_h, dn))), 1)
                    Lk["slid"] = True
                elif not Lk["near"]:
                    Lk["occ"] = True
            Lk["floor"] = round(float(ff), 2)
            hid += Lk["occ"]
        print(f"{sid}  floor {cam_h:.2f} below camera ({'own' if cam_h == peak else 'walk median'})   rings shown {len(node['links']) - hid:2d}  hidden {hid:2d}", flush=True)
    json.dump(poses, open(os.path.join(a.frames, "poses.json"), "w"), indent=1)
    print("updated", os.path.join(a.frames, "poses.json"))


if __name__ == "__main__":
    main()
