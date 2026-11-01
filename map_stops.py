"""Put every stop on ONE shared map with COLMAP (native 360 / equirectangular camera), so the viewer can
show a ring for every stop you can see from where you stand, Matterport-style, and the minimap is exact.

Needs: pip install "pycolmap>=4.1"     (COLMAP 4.1+ has the EQUIRECTANGULAR camera model)
Run after level_panos.py and hop_geometry.py (it uses their up vectors and hop distances for scale):
    python map_stops.py
Writes frames/poses.json, which build_tour.py uses when present.
"""
import argparse, json, os, shutil
from concurrent.futures import ThreadPoolExecutor
import cv2
import numpy as np
import pycolmap
from hop_geometry import level_matrix, CAM_H

if not os.environ.get("SHOP_TOUR_COLMAP_LOG"):
    pycolmap.logging.minloglevel = 2         # COLMAP's own INFO/WARNING flood (hundreds of lines) buries the progress lines; set SHOP_TOUR_COLMAP_LOG=1 to see it
from imutil import imread, imwrite

F = np.diag([1.0, -1.0, 1.0])   # COLMAP camera (x right, y down) <-> ours (x right, y up); both z = pano centre


def az(v):
    return float(np.degrees(np.arctan2(v[0], v[2])))


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
    eo = pycolmap.FeatureExtractionOptions(); eo.sift.max_num_features = features
    ro = pycolmap.ImageReaderOptions(); ro.camera_model = "EQUIRECTANGULAR"; ro.mask_path = mask_dir
    print("features..."); pycolmap.extract_features(db, img_dir, camera_mode=pycolmap.CameraMode.SINGLE,
                                                    reader_options=ro, extraction_options=eo)
    print("matching...")
    if window and len(pts) > window + 1:               # long tours: exhaustive matching grows with the square of the stop count
        lst = os.path.join(work, "pairs.txt")
        with open(lst, "w") as fh:
            for i in range(len(pts)):
                for j in range(i + 1, min(len(pts), i + window + 1)):
                    fh.write(f"{pts[i]['filename']} {pts[j]['filename']}\n")
        pycolmap.match_image_pairs(db, pairing_options=pycolmap.ImportedPairingOptions(match_list_path=lst))
    else:
        pycolmap.match_exhaustive(db)
    print("mapping...")
    best = None
    # COLMAP starts a new model whenever it cannot register the next stop with enough inliers (defaults: 30 and 25%), so a walk with a
    # weak link comes out as several models and only one would be used. Start strict, and loosen step by step until one model has every stop.
    for inl, ratio in ((30, 0.25), (20, 0.15), (15, 0.10), (12, 0.05)):
        out = os.path.join(work, "sparse"); shutil.rmtree(out, ignore_errors=True); os.makedirs(out)
        opts = pycolmap.IncrementalPipelineOptions()
        opts.mapper.abs_pose_min_num_inliers = inl; opts.mapper.abs_pose_min_inlier_ratio = ratio
        recs = pycolmap.incremental_mapping(db, img_dir, out, opts)
        cand = max(recs.values(), key=lambda r: r.num_reg_images())
        if best is None or cand.num_reg_images() > best.num_reg_images():
            best = cand
        if cand.num_reg_images() >= len(pts):
            break
        print(f"mapping placed {cand.num_reg_images()}/{len(pts)} stops (registration needs {inl} inliers, {int(ratio * 100)}%); trying looser thresholds...", flush=True)
    best_dir = os.path.join(work, "best")                  # COLMAP may split a walk into several models; place_rings.py must read THIS one
    shutil.rmtree(best_dir, ignore_errors=True); os.makedirs(best_dir)
    best.write(best_dir)
    print(best.summary())
    return pts, best


def main():
    ap = argparse.ArgumentParser()
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
    tries = [(a.width, a.features)] + ([(4096, 8192)] if (a.width, a.features) != (4096, 8192) else [])
    for width, nfeat in tries:
        npts = len(json.load(open(os.path.join(a.frames, 'points.json'))))
        win = a.window if a.window >= 0 else (20 if npts > 40 else 0)
        pts, rec = run_sfm(a.frames, a.work, width, nfeat, win)
        if rec.num_reg_images() >= len(pts):
            break
        print(f"only {rec.num_reg_images()}/{len(pts)} stops placed at {width} px / {nfeat} features" +
              ("; trying again with more detail (4096 px, 8192 features)..." if (width, nfeat) != tries[-1] else ""), flush=True)
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
    missing = [pts[k]["id"] for k in range(n) if k not in R]
    if missing:
        msg = (f"{len(missing)} of {n} stops could not be placed on the map: {missing}. The tour would have stops without depth, floor or a position. "
               "Usually the camera moved too fast or a stretch had too little texture between two stops; record again with a stop every 1.5-2 m. "
               "To build the partial tour anyway: python map_stops.py --allow-partial")
        if not a.allow_partial:
            raise SystemExit("ERROR: " + msg)
        print("WARNING:", msg)
    ups = json.load(open(os.path.join(a.frames, "up.json")))
    L = [level_matrix(u) for u in ups]
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
    print(f"scale {scale:.3f} m/unit from {len(ratios)} hops")
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
    out = {"scale": scale, "up": up.tolist(), "missing": missing,
           "nodes": [{"id": pts[k]["id"], "x": round(float(xy[k][0]), 2) if k in xy else None,
                      "y": round(float(xy[k][1]), 2) if k in xy else None,
                      "H": round(heading[k], 1) if k in heading else None, "links": links[k]} for k in range(n)]}
    json.dump(out, open(os.path.join(a.frames, "poses.json"), "w"), indent=1)
    json.dump(out, open(os.path.join(a.frames, "poses_map.json"), "w"), indent=1)   # untouched copy; place_rings starts from this
    vis = [len(links[k]) for k in range(n)]
    print(f"wrote {os.path.join(a.frames, 'poses.json')}: {len(R)}/{n} stops placed, rings per stop {min(vis)}-{max(vis)} (avg {np.mean(vis):.1f})")


if __name__ == "__main__":
    main()
