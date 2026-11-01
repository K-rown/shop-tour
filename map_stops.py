"""Put every stop on ONE shared map with COLMAP (native 360 / equirectangular camera), so the viewer can
show a ring for every stop you can see from where you stand, Matterport-style, and the minimap is exact.

Needs: pip install "pycolmap>=4.1"     (COLMAP 4.1+ has the EQUIRECTANGULAR camera model)
Run after level_panos.py and hop_geometry.py (it uses their up vectors and hop distances for scale):
    python map_stops.py
Writes frames/poses.json, which build_tour.py uses when present.
"""
import argparse, json, os, shutil, time
from concurrent.futures import ThreadPoolExecutor
import cv2
import numpy as np
import pycolmap
from hop_geometry import level_matrix, CAM_H

if not os.environ.get("SHOP_TOUR_COLMAP_LOG"):
    pycolmap.logging.minloglevel = 2         # COLMAP's own INFO/WARNING flood (hundreds of lines) buries the progress lines; set SHOP_TOUR_COLMAP_LOG=1 to see it
from imutil import imread, imwrite
import colmap_det
import heading_check
colmap_det.start(pycolmap)

WEAK_POINTS = 50                # a stop the map tied by fewer of its points than this is not trusted where the motion track can place it
F = np.diag([1.0, -1.0, 1.0])   # COLMAP camera (x right, y down) <-> ours (x right, y up); both z = pano centre


def az(v):
    return float(np.degrees(np.arctan2(v[0], v[2])))


def run_sfm_apart(frames, work, width, features, window):
    """A detail try in a process of its own (2026-10-31: COLMAP's matcher aborted in C++ on a retry at 8192 features and took
    the whole step with it, throwing away a map that had already placed 8 of 15 stops). A try that dies is no map; the
    map already made stands."""
    import subprocess, sys
    here = os.path.dirname(os.path.abspath(__file__))
    code = "import sys; sys.path.insert(0, sys.argv[1]); import map_stops as m; m.run_sfm(sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5]), int(sys.argv[6]))"
    r = subprocess.run([sys.executable, "-c", code, here, frames, work, str(width), str(features), str(window)])
    pts = json.load(open(os.path.join(frames, "points.json")))
    best = os.path.join(work, "best")
    if r.returncode != 0:
        print(f"the {width} px / {features} features try stopped (exit {r.returncode}); the map already made is kept", flush=True)
        run_sfm_apart.died = True                      # a heavier try after it would only die the same way (memory): stop trying
        return pts, None
    if not os.path.isdir(best) or not os.listdir(best):
        return pts, None
    return pts, pycolmap.Reconstruction(best)


def run_sfm(frames, work, width=2048, features=4096, window=0):
    img_dir, mask_dir = os.path.join(work, "images"), os.path.join(work, "masks")
    shutil.rmtree(work, ignore_errors=True)
    os.makedirs(img_dir); os.makedirs(mask_dir)
    pts = json.load(open(os.path.join(frames, "points.json")))
    def prep(p):
        im = imread(os.path.join(frames, "points", p["filename"]))
        im = cv2.resize(im, (width, width // 2), interpolation=cv2.INTER_AREA)
        imwrite(os.path.join(img_dir, p["filename"]), im, [cv2.IMWRITE_JPEG_QUALITY, 92])
        mk = np.full((width // 2, width), 255, np.uint8)
        mk[int(width // 2 * 0.80):] = 0      # stick + whoever is holding it
        mk[:int(width // 2 * 0.05)] = 0      # zenith smear
        imwrite(os.path.join(mask_dir, p["filename"] + ".png"), mk)
    with ThreadPoolExecutor(max_workers=min(8, os.cpu_count() or 1)) as ex:
        list(ex.map(prep, pts))
    db = os.path.join(work, "db.db")
    ro = pycolmap.ImageReaderOptions(); ro.camera_model = "EQUIRECTANGULAR"; ro.mask_path = mask_dir
    t0 = time.time()
    print("features..."); colmap_det.extract(pycolmap, db, img_dir, ro, features)
    print(f"  features {time.time() - t0:.0f} s", flush=True); t0 = time.time()
    print("matching...")
    if window and len(pts) > window + 1:               # long tours: exhaustive matching grows with the square of the stop count
        lst = os.path.join(work, "pairs.txt")
        with open(lst, "w") as fh:
            for i in range(len(pts)):
                for j in range(i + 1, min(len(pts), i + window + 1)):
                    fh.write(f"{pts[i]['filename']} {pts[j]['filename']}\n")
        colmap_det.match_pairs(pycolmap, db, lst, work)
    else:
        colmap_det.match_exhaustive(pycolmap, db, work)
    print(f"  matching {time.time() - t0:.0f} s", flush=True); t0 = time.time()
    print("mapping...")
    best = None
    # COLMAP starts a new model whenever it cannot register the next stop with enough inliers (defaults: 30 and 25%), so a walk with a
    # weak link comes out as several models and only one would be used. Start strict, and loosen step by step until one model has every stop.
    for inl, ratio in ((30, 0.25), (20, 0.15), (15, 0.10), (12, 0.05)):
        out = os.path.join(work, "sparse"); shutil.rmtree(out, ignore_errors=True); os.makedirs(out)
        opts = colmap_det.mapper_options(pycolmap)
        opts.mapper.abs_pose_min_num_inliers = inl; opts.mapper.abs_pose_min_inlier_ratio = ratio
        recs = pycolmap.incremental_mapping(db, img_dir, out, opts)
        if not recs:                                   # COLMAP found no starting pair at these thresholds: try the next
            print(f"mapping made no model (registration needs {inl} inliers, {int(ratio * 100)}%)", flush=True)
            continue
        cand = max(recs.values(), key=lambda r: r.num_reg_images())
        if best is None or cand.num_reg_images() > best.num_reg_images():
            best = cand
        if cand.num_reg_images() >= len(pts):
            break
        print(f"mapping placed {cand.num_reg_images()}/{len(pts)} stops (registration needs {inl} inliers, {int(ratio * 100)}%); trying looser thresholds...", flush=True)
    if best is None:
        return pts, None
    best_dir = os.path.join(work, "best")                  # COLMAP may split a walk into several models; place_rings.py must read THIS one
    shutil.rmtree(best_dir, ignore_errors=True); os.makedirs(best_dir)
    best.write(best_dir)
    print(f"  mapping {time.time() - t0:.0f} s", flush=True)
    print(best.summary())
    return pts, best


def segment_scale(video, frames, C, pts, ext, clock_offset_s):
    """Metres per unit of this map from windows of the walk (imu_scale.solve_segments): a window counts when its fit is good (R2 > 0.4, bootstrap < 15 %), at least 4 stops of the window are in both
    maps and those agree to 12 %; the windows' scales (metres per stop-map unit = metres per dense-map unit x dense-map units per stop-map unit, both local) must agree to 15 % and be at least two."""
    import imu_scale
    sj = os.path.join(frames, "scale_imu_segments.json")
    V = np.load(os.path.join(frames, "vio", "poses.npz"))
    if os.path.exists(sj) and os.path.getmtime(sj) >= os.path.getmtime(os.path.join(frames, "vio", "poses.npz")):
        segs = json.load(open(sj))
    else:
        segs = imu_scale.solve_segments(video, frames, clock_offset_s)
        json.dump(segs, open(sj, "w"), indent=1)
    tstop = {str(n)[2:-4]: float(tt) for n, tt in zip(V["names"], V["t"]) if str(n).startswith("s_")}
    good = []
    for s in segs:
        ids = [k for k in C if pts[k]["id"] in ext and s["t0"] <= tstop.get(pts[k]["id"], -1) < s["t1"]]
        ks = [np.linalg.norm(ext[pts[a]["id"]] - ext[pts[b]["id"]]) / np.linalg.norm(C[a] - C[b]) for i, a in enumerate(ids) for b in ids[i + 1:] if np.linalg.norm(C[a] - C[b]) > 1e-6]
        s["stops"] = len(ids)
        if len(ks) < 6 or s["r2"] <= 0.4 or s["bootstrap_std_pct"] >= 15:
            s["accepted"] = False; continue
        k = float(np.median(ks)); rob = 1.4826 * float(np.median(np.abs(np.array(ks) - k))) / k
        s["map_ratio_spread"] = rob; s["scale"] = s["s_forward"] * k
        s["accepted"] = rob < 0.12
        if s["accepted"]:
            good.append(s["scale"])
    print(f"stretches: {len(good)} of {len(segs)} accepted (" + ", ".join(f"{s['t0']:.0f}-{s['t1']:.0f} s: {'%.3f' % s['scale'] if s.get('accepted') else 'no'}" for s in segs) + ")")
    json.dump(segs, open(sj, "w"), indent=1)
    if len(good) < 2:
        return None
    med = float(np.median(good)); spread = (max(good) - min(good)) / med
    if spread > 0.30:
        print(f"the accepted stretches disagree by {spread * 100:.0f} %")
        return None
    return dict(scale=med, info=dict(source_detail="stretches of the walk", stretches=[dict(t0=s["t0"], t1=s["t1"], scale=s.get("scale"), r2=s["r2"], accepted=bool(s.get("accepted"))) for s in segs], stretch_spread=spread))


def so3_mean(Qs):
    """The rotation nearest the mean of rotation matrices (chordal L2 mean)."""
    U, _, Vt = np.linalg.svd(np.asarray(Qs, float).mean(0))
    D = np.eye(3)
    if np.linalg.det(U @ Vt) < 0:
        D[2, 2] = -1
    return U @ D @ Vt


def rot_angle(Rm):
    """Angle of a rotation matrix, degrees."""
    return float(np.degrees(np.arccos(np.clip((np.trace(Rm) - 1) / 2, -1.0, 1.0))))


def collapsed_stops(C, rel=1e-4):
    """Stops the map put at one and the same centre (walk 16ddac14: five stops of a 25 m walk at one point to the millimetre — the mapper gave
    them a rotation and no baseline of their own). Their headings may be right, but every bearing to and from them is not a measurement, so they
    are placed by the motion track instead. Returns the indices of every stop that shares its centre with another."""
    ks = sorted(C)
    if len(ks) < 2:
        return []
    P = np.array([C[k] for k in ks])
    ext = float(np.linalg.norm(P - P.mean(0), axis=1).max()) or 1.0
    out = set()
    for a_ in range(len(ks)):
        for b_ in range(a_ + 1, len(ks)):
            if np.linalg.norm(P[a_] - P[b_]) < rel * ext:
                out.update((ks[a_], ks[b_]))
    return sorted(out)


def track_fit(pts, R, C, Cv, Rv, ks):
    """similarity X_map = s Q X_vio + t over the stops ks. THE TURN COMES FROM THE STOPS' ROTATIONS, NOT THEIR POSITIONS (2026-10-30, walk
    16ddac14: stop 017 came out facing ~145 degrees from where its pictures look): fitted to the centres, the turn follows wherever the map put
    its stops, and a few misplaced ones (the walk's far half was) swing it by tens of degrees. Each stop in both maps carries its whole
    rotation, so Q = R_map^T R_vio is measured once per stop; they are averaged (chordal mean, those more than 10 degrees from the first mean
    dropped) and the scale and shift then come from the positions with that turn held."""
    Qs = np.array([R[k].T @ Rv[pts[k]["id"]] for k in ks])
    Q = so3_mean(Qs)
    dev = np.array([rot_angle(Q.T @ q) for q in Qs])
    if (dev <= 10.0).sum() >= 1:
        Q = so3_mean(Qs[dev <= 10.0])
    A = np.array([Cv[pts[k]["id"]] for k in ks]); B = np.array([C[k] for k in ks])
    ma, mb = A.mean(0), B.mean(0); A0, B0 = A - ma, B - mb
    QA = (Q @ A0.T).T
    var = (A0 ** 2).sum()
    s = float((QA * B0).sum() / var) if var > 1e-12 else 1.0
    t = mb - s * Q @ ma
    res = np.sqrt(((B - (s * (Q @ A.T).T + t)) ** 2).sum(1).mean())
    return s, Q, t, res


def read_track(frames):
    """The motion track's own pose of every stop (vio_map.py, frames/vio/poses.npz "s_<file>"): ({id: centre}, {id: camera-from-world}); None, None
    when there is none."""
    vio = os.path.join(frames, "vio", "poses.npz")
    if not os.path.exists(vio):
        return None, None
    V = np.load(vio)
    names = [str(nm) for nm in V["names"]]
    return ({nm[2:-4]: np.asarray(c, float) for nm, c in zip(names, V["C"]) if nm.startswith("s_")},
            {nm[2:-4]: np.asarray(r, float) for nm, r in zip(names, V["R"]) if nm.startswith("s_")})


def vio_place(frames, pts, R, C, missing_k, window=40.0, log=print):
    """A STOP THE MAP COULD NOT PLACE TAKES ITS PLACE FROM THE WALK'S OWN MOTION TRACK (2026-10-30 — the owner's 91-second road walk: the stop map tied
    the first 8 of 15 stops and the far side of the street was left out of the tour, "I swear I went across the street but cannot see that in tour").
    vio_map.py's dense 1 Hz chain (frames/vio/poses.npz) has a camera pose for every picture of the walk, the stops' own among them ("s_<id>.jpg"), and
    it had no gap where the stop map did. For each missing stop the chain is laid onto the map by a similarity fit (Umeyama, rotation + scale + shift)
    over the PLACED stops within +-`window` seconds of it that are in both (the chain drifts in scale over a long walk, so the fit is local; at least
    three; the whole walk when a stretch has too few), and the stop's centre and rotation come through that fit. Fills R and C for the stops it could
    place and returns their indices; nothing it cannot. The stop has no map points (no rings measured from it: place_rings keeps the walk's median
    height), and poses.json says it was placed by the motion track ("by": "vio")."""
    vio = os.path.join(frames, "vio", "poses.npz")
    if not missing_k or not os.path.exists(vio):
        return []
    try:
        V = np.load(vio)
        names = [str(n) for n in V["names"]]
        Cv = {nm[2:-4]: np.asarray(c, float) for nm, c in zip(names, V["C"]) if nm.startswith("s_")}
        Rv = {nm[2:-4]: np.asarray(r, float) for nm, r in zip(names, V["R"]) if nm.startswith("s_")}
    except Exception as e:
        log(f"motion track unreadable ({e}): the unplaced stops stay out")
        return []
    placed = [k for k in sorted(C) if pts[k]["id"] in Cv]
    if len(placed) < 3:
        log("too few placed stops in both the map and the motion track: the unplaced stops stay out")
        return []

    fit = lambda ks: track_fit(pts, R, C, Cv, Rv, ks)

    out = []
    for k in missing_k:
        sid = pts[k]["id"]
        if sid not in Cv or sid not in Rv:
            continue
        ts = float(pts[k]["timestamp"])
        near = [i for i in placed if abs(float(pts[i]["timestamp"]) - ts) <= window]
        ks = near if len(near) >= 3 else placed
        s, Q, t, res = fit(ks)
        if not np.isfinite(s) or s <= 0:
            continue
        C[k] = s * Q @ Cv[sid] + t
        R[k] = Rv[sid] @ Q.T
        out.append(k)
        log(f"stop {sid} placed by the motion track: fit over {len(ks)} stops{' of its stretch' if ks is near else ' of the whole walk'}, residual {res:.2f} map units")
    return out


def imu_metric_scale(video, frames, C, pts):
    """Metres per unit of THIS map from the accelerometer (imu_scale.py on the 1 Hz path of vio_map.py), or None (and a warning) if that is not available or not trustworthy."""
    vio = os.path.join(frames, "vio", "poses.npz")
    if not os.path.exists(vio):
        print("no frames/vio/poses.npz (vio_map.py): using the camera-height prior for the scale")
        return None
    try:
        import imu_scale
        sj = os.path.join(frames, "scale_imu.json")
        if os.path.exists(sj) and os.path.getmtime(sj) >= os.path.getmtime(vio):
            r = json.load(open(sj))
        else:
            r = imu_scale.solve_scale(video, frames)
            json.dump(r, open(sj, "w"), indent=1)
        V = np.load(vio)
        ext = {str(n)[2:-4]: c for n, c in zip(V["names"], V["C"]) if str(n).startswith("s_")}
        ks = []
        ids = [k for k in C if pts[k]["id"] in ext]
        for x in range(len(ids)):
            for y in range(x + 1, len(ids)):
                dc = np.linalg.norm(C[ids[x]] - C[ids[y]]); de = np.linalg.norm(ext[pts[ids[x]]["id"]] - ext[pts[ids[y]]["id"]])
                if dc > 1e-6 and de > 1e-6:
                    ks.append(de / dc)
        if len(ks) < 3:
            raise RuntimeError("too few stops in both maps")
        k_unit = float(np.median(ks))
        rob = 1.4826 * float(np.median(np.abs(np.array(ks) - np.median(ks)))) / float(np.median(ks))      # robust spread of how the two maps' distances agree
        ok = r["r2"] > 0.3 and r["bootstrap_std_pct"] < 15 and rob < 0.12
        info = dict(imu_m_per_vio_unit=r["scale_m_per_unit"], vio_units_per_map_unit=k_unit, r2=r["r2"], bootstrap_std_pct=r["bootstrap_std_pct"],
                    s_forward=r["s_forward"], s_inverse=r["s_inverse"], clock_offset_s=r["clock_offset_s"], map_ratio_spread=rob)
        if not ok:
            print(f"the whole-walk accelerometer scale is not trustworthy here (r2 {r['r2']:.2f}, bootstrap {r['bootstrap_std_pct']:.0f} %, maps agree to {rob * 100:.0f} %): trying stretches of the walk")
            seg = segment_scale(video, frames, C, pts, ext, r["clock_offset_s"])
            if seg is None:
                print("no two stretches agree: using the camera-height prior")
                return None
            info.update(seg["info"])
            return dict(scale=seg["scale"], info=info)
        return dict(scale=r["scale_m_per_unit"] * k_unit, info=info)
    except Exception as e:
        print(f"the accelerometer scale failed ({str(e)[:120]}): using the camera-height prior")
        return None


_IMU_JOB = r"""
import json, os, sys
sys.path.insert(0, sys.argv[1])
import imu_scale
r = imu_scale.solve_scale(sys.argv[2], sys.argv[3], log=lambda *a, **k: None)
tmp = sys.argv[4] + ".part"
json.dump(r, open(tmp, "w"), indent=1)
os.replace(tmp, sys.argv[4])
"""


def start_imu_scale(video, frames):
    """The accelerometer fit (imu_scale.solve_scale) needs only the video and frames/vio/poses.npz, not this map, and takes about a minute of single-core
    numerics: start it now, beside the features and the mapping, and imu_metric_scale then finds frames/scale_imu.json ready (SHOP_TOUR_IMU_BESIDE=0 turns
    this off). The answer is the same code on the same inputs, so scale_imu.json is the same file either way. Returns the process, or None."""
    import subprocess, sys
    vio = os.path.join(frames, "vio", "poses.npz")
    sj = os.path.join(frames, "scale_imu.json")
    if not video or not os.path.exists(vio) or os.environ.get("SHOP_TOUR_IMU_BESIDE", "1").strip().lower() in ("0", "false", "no", "off"):
        return None
    if os.path.exists(sj) and os.path.getmtime(sj) >= os.path.getmtime(vio):
        return None
    try:
        return subprocess.Popen([sys.executable, "-c", _IMU_JOB, os.path.dirname(os.path.abspath(__file__)), video, frames, sj],
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default=None, help="the .insv: its accelerometer fixes the metric scale (needs frames/vio from vio_map.py)")
    ap.add_argument("--frames", default="frames")
    ap.add_argument("--work", default=os.path.join("frames", "colmap"))
    ap.add_argument("--width", type=int, default=2048, help="width the stops are shrunk to for COLMAP (4096 is slower, similar result)")
    ap.add_argument("--features", type=int, default=4096, help="SIFT features per stop (matching time grows with the square)")
    ap.add_argument("--window", type=int, default=-1, help="match each stop only with the next N stops (default: all pairs up to 40 stops, a window of 20 above)")
    ap.add_argument("--allow-partial", action="store_true", help="build the tour even if some stops could not be placed")
    ap.add_argument("--max-dist", type=float, default=14.0, help="no ring for stops farther than this (m)")
    ap.add_argument("--min-shared", type=int, default=12, help="3D points two stops must both see to count as visible")
    a = ap.parse_args()
    # the fast settings (2048 px, 4096 features) are enough for a normal walk; if some stops still cannot be placed (a long stretch between
    # two stops, little texture), redo the features and matching with more detail before giving up
    # the detail retry never upscales (2026-10-28: a 2560-wide streamed walk retried at 4096 px placed 0 of 25 where 2048 px placed 20):
    # the second try is more features at the pano's own width (4096 at most); a third, when stops are still unplaced, matches every pair
    first = json.load(open(os.path.join(a.frames, 'points.json')))[0]["filename"]
    native = imread(os.path.join(a.frames, "points", first)).shape[1]
    w2 = min(4096, native)
    tries = [(a.width, a.features, None)]
    if (a.width, a.features) != (w2, 8192):
        tries.append((w2, 8192, None))
    tries.append((w2, 8192, 0))
    import stepsync
    early = stepsync.enabled()                     # begun while vio is still working (tour.EARLY): the stops' own map needs only the stops' pictures; vio's path is waited for below
    imu_proc = None if early else start_imu_scale(getattr(a, "video", None), a.frames)
    # each try maps in its own folder and the BEST one is kept (2026-10-28: a road walk placed 20 of 25 stops at the fast
    # settings, the detailed retry made no model at all, and the walk failed with the good map thrown away)
    kept = None
    for t, (width, nfeat, force_win) in enumerate(tries):
        npts = len(json.load(open(os.path.join(a.frames, 'points.json'))))
        win = a.window if a.window >= 0 else (20 if npts > 40 else 0)
        if force_win is not None:
            if win == force_win:                       # the exhaustive try is only a different try when the window was on
                continue
            win = force_win
        wdir = a.work if t == 0 else f"{a.work}_try{t}"
        pts, rec = (run_sfm if t == 0 else run_sfm_apart)(a.frames, wdir, width, nfeat, win)
        if getattr(run_sfm_apart, "died", False):
            break
        if rec is not None and (kept is None or rec.num_reg_images() > kept[1].num_reg_images()):
            kept = (pts, rec, wdir)
        if rec is not None and rec.num_reg_images() >= len(pts):
            break
        print(f"only {rec.num_reg_images() if rec is not None else 0}/{len(pts)} stops placed at {width} px / {nfeat} features" +
              (f"; trying again ({tries[t + 1][0]} px, {tries[t + 1][1]} features{', every pair' if tries[t + 1][2] == 0 else ''})..." if t + 1 < len(tries) else ""), flush=True)
    if kept is None:
        raise SystemExit("ERROR: no stop could be placed on the map at any setting. Usually the camera moved too fast or the place has too "
                         "little texture; record again with a stop every 1.5-2 m.")
    pts, rec, wdir = kept
    if wdir != a.work:                                 # place_rings.py reads <work>/best: the kept map goes where it looks
        shutil.rmtree(a.work, ignore_errors=True); shutil.copytree(wdir, a.work)
    for t in range(1, len(tries)):
        shutil.rmtree(f"{a.work}_try{t}", ignore_errors=True)
    if early:
        if stepsync.wait_for("vio", log=lambda m: print(m, flush=True)) != "ok":
            raise SystemExit("vio did not finish: there is no camera path to take the metric scale from")
        imu_proc = start_imu_scale(getattr(a, "video", None), a.frames)          # (the accelerometer fit needs vio's path: it begins now, beside the rest of this step)
    n = len(pts)
    idx = {p["filename"]: k for k, p in enumerate(pts)}
    R, C, obs = {}, {}, {}
    for im in rec.images.values():
        if not im.has_pose:
            continue
        k = idx[im.name]
        T = im.cam_from_world() if callable(im.cam_from_world) else im.cam_from_world
        R[k] = T.rotation.matrix(); C[k] = -R[k].T @ T.translation
        obs[k] = {p.point3D_id for p in im.points2D if p.has_point3D()}
    # THE MAP'S STOPS ARE CHECKED AGAINST THEIR OWN PICTURES BEFORE THE TOUR STANDS ON THEM (2026-10-30, walk 16ddac14 — "no photo was shot out in this
    # direction but actually across the street": its map, tied by 1,100 points, put stops 011-015 on ONE point and 010-018 10 to 170 degrees off the
    # directions their pictures measure to each other, while the walk's own motion track agreed with every one of them within 2). A stop is HELD from
    # the map when it shares its centre with another (collapsed_stops), when the map tied it by fewer than WEAK_POINTS points, or when its pictures
    # disagree with where it stands (heading_check.worst_out: the worst out first, until every stop left agrees). The held and the unplaced are laid
    # on from the motion track, fitted onto the stops that passed (vio_place); a held stop keeps whichever of its two poses agrees better with its
    # pictures, and with nothing to judge by, the track's — a weakly tied or collapsed map pose is not a measurement. Too few passed: nothing is held.
    ups = json.load(open(os.path.join(a.frames, "up.json")))
    L = [level_matrix(u) for u in ups]
    HJ = json.load(open(os.path.join(a.frames, "hops.json")))
    PAIRS = heading_check.good_pairs(HJ)
    EVID = heading_check.good_pairs(HJ, min_inliers=heading_check.MIN_EVIDENCE)
    have_vio = os.path.exists(os.path.join(a.frames, "vio", "poses.npz"))
    stacked = collapsed_stops(C)
    weak = [k for k in R if k not in stacked and len(obs[k]) < WEAK_POINTS]
    rest = {k: R[k] for k in R if k not in stacked and k not in weak}
    off, _ = heading_check.worst_out(heading_check.errors_3d(rest, {k: C[k] for k in rest}, L, PAIRS))
    hold = sorted(set(stacked) | set(weak) | set(off))
    held = {}
    if hold and have_vio and len(R) - len(hold) >= 3:
        held = {k: (R.pop(k), C.pop(k), obs.pop(k)) for k in hold}
        print(f"HELD FROM THE MAP (the motion track places them if it agrees better): on one point {[pts[k]['id'] for k in stacked]}, "
              f"weakly tied {[pts[k]['id'] for k in sorted(weak)]}, off their own pictures {[pts[k]['id'] for k in sorted(off)]}")
    missing = [pts[k]["id"] for k in range(n) if k not in R]
    by_vio, filled = [], []
    if missing:
        # THE MOTION TRACK PLACES WHAT THE MAP COULD NOT (2026-10-30): before anything is left out, each unplaced stop takes its place from the walk's
        # own 1 Hz camera path laid onto the map — see vio_place
        filled = vio_place(a.frames, pts, R, C, [k for k in range(n) if k not in R])
        for k in filled:
            obs[k] = set()
        for k, (r_, c_, o_) in held.items():
            if k not in R:
                R[k], C[k], obs[k] = r_, c_, o_
                continue
            ev = heading_check.stop_errors(heading_check.errors_3d(R, C, L, EVID)).get(k, (None, 0))[0]
            tr, tc = R[k], C[k]; R[k], C[k] = r_, c_
            em = heading_check.stop_errors(heading_check.errors_3d(R, C, L, EVID)).get(k, (None, 0))[0]
            if em is not None and (ev is None or em <= ev) and k not in stacked:
                obs[k] = o_; filled = [f for f in filled if f != k]
            else:
                R[k], C[k] = tr, tc
            print(f"  stop {pts[k]['id']}: its pictures disagree with the map's pose by {em if em is None else round(em, 1)} deg, with the track's by "
                  f"{ev if ev is None else round(ev, 1)} deg: the {'map' if k not in filled else 'track'}'s is kept")
        by_vio = [pts[k]["id"] for k in filled]
        if by_vio:
            print(f"PLACED BY THE MOTION TRACK (no map points of their own): {by_vio}")
        missing = [pts[k]["id"] for k in range(n) if k not in R]
    # THE PLAN FOLLOWS THE MOTION TRACK WHEN IT AGREES BETTER WITH THE PICTURES (2026-10-30). The stop map ties each stop to the walk by the few
    # hundred points two stops share; the motion track ties every stop to the pictures taken a second before and after it. Measured on five walks,
    # the track alone put every stop where its pictures say within 1-3 degrees (median), the map within 2-25 (16ddac14: 16, fe0b8a49: 26). When the
    # track has every stop, it is laid onto the map in ONE fit (its own geometry kept whole: track_fit over the map's stops that agree with their
    # pictures) and, if the pictures agree with it better than with the map, every stop's centre and turn come from it. The map still gives the
    # scale's hops, the points two stops share (which stops see each other) and, in place_rings, each stop's depth.
    plan_from = "map"
    Cv, Rv = (read_track(a.frames) if have_vio else (None, None))
    if Cv and Rv and len(R) == n and all(p["id"] in Cv and p["id"] in Rv for p in pts):
        good = [k for k in R if k not in held and k not in filled]
        if len(good) >= 3:
            ts_, Q_, t_, res_ = track_fit(pts, R, C, Cv, Rv, good)
            if np.isfinite(ts_) and ts_ > 0:
                Rt = {k: Rv[pts[k]["id"]] @ Q_.T for k in range(n)}
                Ct = {k: ts_ * Q_ @ Cv[pts[k]["id"]] + t_ for k in range(n)}
                e_map = heading_check.summary(heading_check.errors_3d(R, C, L, PAIRS))[0]
                e_trk = heading_check.summary(heading_check.errors_3d(Rt, Ct, L, PAIRS))[0]
                print(f"the pictures agree with the map's stops within {e_map} degrees (median), with the motion track's within {e_trk}")
                if e_trk is not None and (e_map is None or e_trk < e_map):
                    R, C, plan_from = Rt, Ct, "track"
                    print(f"THE PLAN FOLLOWS THE MOTION TRACK (one fit over {len(good)} stops, residual {res_:.2f} map units)")
    if missing:
        msg = (f"{len(missing)} of {n} stops could not be placed on the map: {missing}. The tour would have stops without depth, floor or a position. "
               "Usually the camera moved too fast or a stretch had too little texture between two stops; record again with a stop every 1.5-2 m. "
               "To build the partial tour anyway: python map_stops.py --allow-partial")
        if not a.allow_partial:
            raise SystemExit("ERROR: " + msg)
        if len(missing) * 2 > n:                       # a tour of fewer than half its stops is not the walk
            raise SystemExit(f"ERROR: only {n - len(missing)} of {n} stops could be placed on the map; record again with a stop every 1.5-2 m.")
        print("WARNING:", msg)
    # world up = average of every pano's measured up, carried into the map
    U = np.array([R[k].T @ (F @ np.array(ups[k], float)) for k in R])
    up = U.mean(0); up /= np.linalg.norm(up)
    ax = np.cross(up, [1.0, 0, 0]); ax /= np.linalg.norm(ax); bx = np.cross(up, ax)
    # metric scale from the hop distances hop_geometry measured with the camera height
    H = json.load(open(os.path.join(a.frames, "hops.json")))
    ratios = []
    for p in H["pairs"]:
        i, j = p["i"], p["j"]
        if p.get("dist_m") and p["ok"] and p.get("consistent", True) and i in C and j in C:
            d = np.linalg.norm((C[j] - C[i]) - up * ((C[j] - C[i]) @ up))
            if d > 1e-6:
                ratios.append(p["dist_m"] / d)
    scale = float(np.median(ratios)) if ratios else 1.0
    print(f"scale {scale:.3f} m/unit from {len(ratios)} hops (camera-height prior)")
    scale_info = {"source": "hops (camera-height prior)", "hops_scale": scale}
    t0 = time.time()
    if imu_proc is not None:
        imu_proc.wait()                            # (if it failed, imu_metric_scale finds no fresh scale_imu.json and does the fit itself)
        print(f"  waited {time.time() - t0:.0f} s for the accelerometer fit", flush=True)
    if getattr(a, "video", None):
        imu_scale_m = imu_metric_scale(a.video, a.frames, C, pts)
        if imu_scale_m:
            scale_info.update(source="accelerometer + pictures (imu_scale.py)", **imu_scale_m["info"])
            print(f"scale {imu_scale_m['scale']:.3f} m/unit from the accelerometer (the camera-height prior said {scale:.3f}; ratio {imu_scale_m['scale'] / scale:.3f})")
            scale = imu_scale_m["scale"]
    xy = {k: np.array([C[k] @ ax, C[k] @ bx]) * scale for k in C}
    o = xy[min(xy)]
    xy = {k: v - o for k, v in xy.items()}
    heading = {}
    for k in R:
        f = R[k].T @ np.array([0, 0, 1.0])
        heading[k] = float(np.degrees(np.arctan2(f @ ax, f @ bx)))
    # the plan axes can come out mirrored; check against a measured bearing and flip if so
    errs = []
    for k in R:
        if k + 1 in R:
            d = F @ (R[k] @ (C[k + 1] - C[k])); a_ij = az(L[k] @ d)
            v = xy[k + 1] - xy[k]; bearing = np.degrees(np.arctan2(v[0], v[1]))
            errs.append(((bearing - heading[k] - a_ij + 540) % 360 - 180, (bearing - heading[k] + a_ij + 540) % 360 - 180))
    e = np.abs(np.array(errs)).mean(0)
    if e[1] < e[0]:
        xy = {k: np.array([-v[0], v[1]]) for k, v in xy.items()}
        heading = {k: -h for k, h in heading.items()}
    # COLMAP's world frame is arbitrary (it came out rotated and even mirrored between runs of the same walk), so put the plan in a
    # canonical frame: principal axis of the stops along +x, the stop farthest from the first one on the right. Rigid motion only
    # (links, az and turn are relative to each stop and do not change), so builds are comparable and the minimap does not flip.
    ks = sorted(xy)
    P = np.array([xy[k] for k in ks]); c = P.mean(0)
    w, V = np.linalg.eigh(np.cov((P - c).T))
    beta = np.arctan2(V[1, np.argmax(w)], V[0, np.argmax(w)])           # math angle of the principal axis
    far = np.argmax(np.linalg.norm(P - P[0], axis=1))
    if ((P[far] - c) @ np.array([np.cos(beta), np.sin(beta)])) < 0:
        beta += np.pi
    Rm = np.array([[np.cos(-beta), -np.sin(-beta)], [np.sin(-beta), np.cos(-beta)]])
    xy = {k: Rm @ (xy[k] - c) for k in xy}
    heading = {k: float((h + np.degrees(beta) + 180) % 360 - 180) for k, h in heading.items()}   # bearings (clockwise from +y) drop by the rotation angle: -(-beta)
    # every pair of stops that see each other
    links = {k: [] for k in range(n)}
    for i in R:
        for j in R:
            if i == j:
                continue
            shared = len(obs[i] & obs[j])
            dvec = C[j] - C[i]
            dist = float(np.linalg.norm(dvec - up * (dvec @ up)) * scale)
            if abs(i - j) > 1 and (shared < a.min_shared or dist > a.max_dist):
                continue
            d = L[i] @ (F @ (R[i] @ dvec))
            Rij = F @ (R[j] @ R[i].T) @ F
            Rl = L[j] @ Rij @ L[i].T
            yaw = az(Rl.T @ np.array([0, 0, 1.0]))
            links[i].append({"to": j, "az": round(az(d), 1), "turn": round(-yaw, 1), "d": round(dist, 2),
                             "el": round(float(np.degrees(np.arctan2(CAM_H, max(dist, 0.3)))), 1),
                             "near": abs(i - j) == 1, "shared": shared})
    vio_ids = set(by_vio)
    out = {"scale": scale, "scale_info": scale_info, "up": up.tolist(), "missing": missing, "by_vio": by_vio, "plan": plan_from,
           "nodes": [{"id": pts[k]["id"], "x": round(float(xy[k][0]), 2) if k in xy else None,
                      "y": round(float(xy[k][1]), 2) if k in xy else None,
                      "H": round(heading[k], 1) if k in heading else None, "links": links[k],
                      **({"by": "vio"} if pts[k]["id"] in vio_ids else {})} for k in range(n)]}
    se = heading_check.stop_errors(heading_check.errors_3d(R, C, L, PAIRS))           # each stop's agreement with its own pictures: its heading's confidence
    for k, nd in enumerate(out["nodes"]):
        if k in se:
            nd["check"] = {"deg": round(se[k][0], 1), "pairs": se[k][1]}
    json.dump(out, open(os.path.join(a.frames, "poses.json"), "w"), indent=1)
    json.dump(out, open(os.path.join(a.frames, "poses_map.json"), "w"), indent=1)   # untouched copy; place_rings starts from this
    vis = [len(links[k]) for k in range(n)]
    print(f"wrote {os.path.join(a.frames, 'poses.json')}: {len(R)}/{n} stops placed, rings per stop {min(vis)}-{max(vis)} (avg {np.mean(vis):.1f})")


if __name__ == "__main__":
    main()
