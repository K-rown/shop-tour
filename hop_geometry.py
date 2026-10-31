"""Measure, for every consecutive pair of panos, the local geometry the viewer needs.

For hop i -> i+1 (camera 1 = pano i, camera 2 = pano i+1; x right, y up, z forward = pano centre):
  az_fwd   azimuth (deg, clockwise from pano centre) of stop i+1 as seen from pano i
  az_back  azimuth of stop i as seen from pano i+1
  yaw      how far camera 2 is turned clockwise relative to camera 1
It uses only the two photos (SIFT + spherical 8-point RANSAC), so nothing drifts along the chain.

Usage: python hop_geometry.py [--frames frames] [--out frames/hops.json]
"""
import argparse, json, os
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import cv2
import numpy as np
from imutil import imread

W_SIFT = 2880


def bearings(pts, w, h):
    u, v = pts[:, 0], pts[:, 1]
    th = u / w * 2 * np.pi - np.pi
    ph = np.pi / 2 - v / h * np.pi
    return np.stack([np.cos(ph) * np.sin(th), np.sin(ph), np.cos(ph) * np.cos(th)], 1)


def features(path, sift):
    img = imread(path, cv2.IMREAD_GRAYSCALE)
    h0, w0 = img.shape
    img = cv2.resize(img, (W_SIFT, W_SIFT // 2), interpolation=cv2.INTER_AREA)
    img = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(16, 8)).apply(img)  # concrete floors are low contrast
    h, w = img.shape
    mask = np.zeros_like(img)
    mask[int(h * 0.08):int(h * 0.78), :] = 255  # drop zenith smear and the stick/body at the bottom
    kp, des = sift.detectAndCompute(img, mask)
    return np.array([k.pt for k in kp], np.float32), des, w, h


def solve_E(b1, b2):
    A = np.einsum('ni,nj->nij', b2, b1).reshape(len(b1), 9)
    _, _, Vt = np.linalg.svd(A)
    E = Vt[-1].reshape(3, 3)
    U, _, Vt = np.linalg.svd(E)
    return U @ np.diag([1, 1, 0]) @ Vt


def ransac(b1, b2, iters=3000, thr_deg=1.5, seed=0):
    rng = np.random.default_rng(seed)
    n = len(b1)
    thr = np.sin(np.radians(thr_deg))
    best, best_in = None, np.zeros(n, bool)
    for _ in range(iters):
        idx = rng.choice(n, 8, replace=False)
        E = solve_E(b1[idx], b2[idx])
        Eb1 = b1 @ E.T
        err = np.abs(np.einsum('ni,ni->n', b2, Eb1)) / (np.linalg.norm(Eb1, axis=1) + 1e-12)
        inl = err < thr
        if inl.sum() > best_in.sum():
            best, best_in = E, inl
    if best_in.sum() >= 12:
        best = solve_E(b1[best_in], b2[best_in])
    return best, best_in


def triangulate(R, t, b1, b2):
    """Depths (l1, l2) along each ray, in units of |t|, from  l2*b2 = l1*R*b1 + t  (least squares, per point)."""
    Rb = b1 @ R.T
    a11 = np.einsum('ni,ni->n', Rb, Rb); a12 = -np.einsum('ni,ni->n', Rb, b2); a22 = np.einsum('ni,ni->n', b2, b2)
    r1 = -np.einsum('ni,i->n', Rb, t); r2 = np.einsum('ni,i->n', b2, t)
    det = a11 * a22 - a12 * a12 + 1e-12
    l1 = (a22 * r1 - a12 * r2) / det
    l2 = (a11 * r2 - a12 * r1) / det
    return l1, l2


def decompose(E, b1, b2):
    if len(b1) > 150:
        sel = np.random.default_rng(1).choice(len(b1), 150, replace=False); b1, b2 = b1[sel], b2[sel]
    U, _, Vt = np.linalg.svd(E)
    if np.linalg.det(U) < 0: U[:, -1] *= -1
    if np.linalg.det(Vt) < 0: Vt[-1] *= -1
    Wm = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], float)
    best, bestn = None, -1
    for R in (U @ Wm @ Vt, U @ Wm.T @ Vt):
        for sgn in (1, -1):
            t = sgn * U[:, 2]
            l1, l2 = triangulate(R, t, b1, b2)
            n_pos = int(((l1 > 0) & (l2 > 0)).sum())
            if n_pos > bestn:
                best, bestn = (R, t, l1, l2), n_pos
    return best[0], best[1], bestn, best[2], best[3], b1


CAM_H = 1.85  # metres, camera height on the stick; only used to guess how far apart two stops are


def level_matrix(u):
    """Rotation that takes the pano's true up vector u to +y."""
    u = np.asarray(u, float); u /= np.linalg.norm(u)
    y = np.array([0, 1.0, 0]); c = float(u @ y)
    if c > 0.99999:
        return np.eye(3)
    ax = np.cross(u, y); s = np.linalg.norm(ax); ax /= s
    K = np.array([[0, -ax[2], ax[1]], [ax[2], 0, -ax[0]], [-ax[1], ax[0], 0]])
    return np.eye(3) + s * K + (1 - c) * (K @ K)


def floor_height_units(l1, bl):
    """Camera height above the floor in |t| units, from points below the horizon (bl = leveled bearings)."""
    el = -bl[:, 1]
    m = (el > 0.3) & (l1 > 0)            # more than ~17 deg below the horizon
    if m.sum() < 8:
        return None
    drop = l1[m] * el[m]
    top = np.percentile(drop, 98)
    hist, edges = np.histogram(drop, bins=40, range=(0, top))
    centers = (edges[:-1] + edges[1:]) / 2
    ok = centers >= np.median(drop)       # the floor is the deepest well-supported layer, not shelves or boxes
    if not ok.any():
        return None
    return float(centers[ok][np.argmax(hist[ok])])


def floor_scale(l1, b1):
    """Metres per |t| unit, from floor points: the vertical drop of a floor point equals the camera height."""
    m = (b1[:, 1] < -0.25) & (l1 > 0)
    if m.sum() < 6:
        return None
    h_units = float(np.median(l1[m] * -b1[m, 1]))
    return CAM_H / h_units if h_units > 1e-6 else None


def az(v):
    return float(np.degrees(np.arctan2(v[0], v[2])))


def measure(i, j, feats, bf, Ls):
    k1, d1, w, h = feats[i]; k2, d2, _, _ = feats[j]
    m = [x for x, y in bf.knnMatch(d1, d2, k=2) if x.distance < 0.75 * y.distance]
    if len(m) < 20:
        return {"i": i, "j": j, "ok": False, "matches": len(m), "inliers": 0}
    b1 = bearings(k1[[x.queryIdx for x in m]], w, h)
    b2 = bearings(k2[[x.trainIdx for x in m]], w, h)
    E, inl = ransac(b1, b2, seed=i * 100 + j)
    R, t, npos, l1, l2, b1s = decompose(E, b1[inl], b2[inl])
    C = -R.T @ t; C = C / np.linalg.norm(C)          # camera j centre in camera i frame
    t_n = t / np.linalg.norm(t)                       # camera i centre in camera j frame
    Li, Lj = Ls[i], Ls[j]
    Ci, Tj = Li @ C, Lj @ t_n                         # same directions, in the levelled frames
    Rl = Lj @ R @ Li.T                                # rotation between levelled frames
    yaw = az(Rl.T @ np.array([0, 0, 1.0]))
    # floor: depth of every inlier, then the camera height in |t| units
    l1a, l2a = triangulate(R, t, b1[inl], b2[inl])
    h_u = floor_height_units(l1a, b1[inl] @ Li.T)
    hz_i, hz_j = float(np.hypot(Ci[0], Ci[2])), float(np.hypot(Tj[0], Tj[2]))
    el_ij = el_ji = D = None
    if h_u:
        el_ij = float(np.degrees(np.arctan2(h_u, max(hz_i, 1e-3))))
        el_ji = float(np.degrees(np.arctan2(h_u, max(hz_j, 1e-3))))
        D = CAM_H / h_u * 0.5 * (hz_i + hz_j)         # metres along the floor, if the camera is CAM_H up
    return {"i": i, "j": j, "ok": bool(inl.sum() >= 25), "matches": len(m), "inliers": int(inl.sum()),
            "az_ij": round(az(Ci), 1), "az_ji": round(az(Tj), 1), "yaw": round(yaw, 1),
            "el_ij": None if el_ij is None else round(el_ij, 1), "el_ji": None if el_ji is None else round(el_ji, 1),
            "dist_m": None if D is None else round(D, 2), "R": R.round(5).tolist()}


_G = {}


def _init(feats, Ls):
    _G["feats"], _G["Ls"], _G["bf"] = feats, Ls, cv2.BFMatcher()


def _measure(ij):
    i, j = ij
    return ij, measure(i, j, _G["feats"], _G["bf"], _G["Ls"])


def rot_angle(Ra, Rb):
    c = (np.trace(Ra @ Rb.T) - 1) / 2
    return float(np.degrees(np.arccos(np.clip(c, -1, 1))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", default="frames")
    ap.add_argument("--out", default=None)
    ap.add_argument("--window", type=int, default=5, help="also measure stops up to this many steps apart")
    a = ap.parse_args()
    out = a.out or os.path.join(a.frames, "hops.json")
    pts = json.load(open(os.path.join(a.frames, "points.json")))
    n = len(pts)
    sift = cv2.SIFT_create(nfeatures=6000)
    with ThreadPoolExecutor(max_workers=min(6, os.cpu_count() or 1)) as ex:     # SIFT releases the GIL
        feats = list(ex.map(lambda p: features(os.path.join(a.frames, "points", p["filename"]), cv2.SIFT_create(nfeatures=6000)), pts))
    upf = os.path.join(a.frames, "up.json")
    Ls = [level_matrix(u) for u in json.load(open(upf))] if os.path.exists(upf) else [np.eye(3)] * n
    print("levelling:", "on (up.json)" if os.path.exists(upf) else "OFF (run level_panos.py first)")
    jobs = [(i, j) for i in range(n) for j in range(i + 1, min(n, i + a.window + 1))]
    pairs = {}
    with ProcessPoolExecutor(max_workers=min(10, os.cpu_count() or 1), initializer=_init, initargs=(feats, Ls)) as ex:
        for (i, j), r in ex.map(_measure, jobs, chunksize=2):             # each pair is seeded, so results match a serial run
            pairs[(i, j)] = r
            print(f"{pts[i]['id']}->{pts[j]['id']}  inl {r['inliers']:4d}/{r['matches']:4d}  ok={r['ok']}  "
                  f"az {r.get('az_ij')}  D {r.get('dist_m')}", flush=True)
    # consistency: a far pair must agree with the chain of consecutive hops
    chain = {}
    for i in range(n - 1):
        r = pairs[(i, i + 1)]
        chain[(i, i + 1)] = np.array(r["R"]) if r["ok"] else None
    for (i, j), r in pairs.items():
        r["consistent"] = True
        if j == i + 1 or not r["ok"]:
            continue
        Rc = np.eye(3)
        for k in range(i, j):
            if chain[(k, k + 1)] is None: Rc = None; break
            Rc = chain[(k, k + 1)] @ Rc
        if Rc is not None:
            err = rot_angle(np.array(r["R"]), Rc)
            r["chain_err_deg"] = round(err, 1)
            r["consistent"] = err < 15 + 4 * (j - i)
    for r in pairs.values(): r.pop("R", None) if r["i"] != r["j"] and False else None
    hops = [{"from": i, "to": i + 1, "ok": pairs[(i, i + 1)]["ok"], "inliers": pairs[(i, i + 1)]["inliers"],
             "az_fwd": pairs[(i, i + 1)].get("az_ij"), "az_back": pairs[(i, i + 1)].get("az_ji"),
             "yaw": pairs[(i, i + 1)].get("yaw"), "dist_m": pairs[(i, i + 1)].get("dist_m")} for i in range(n - 1)]
    json.dump({"hops": hops, "pairs": [{k: v for k, v in r.items() if k != "R"} for r in pairs.values()]}, open(out, "w"), indent=1)
    good = sum(1 for r in pairs.values() if r["ok"] and r["consistent"])
    print(f"wrote {out}: {len(pairs)} pairs measured, {good} usable")


if __name__ == "__main__":
    main()
