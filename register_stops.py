"""Put the stops where their depth says they stand: every stop's full-sphere depth (depth_sphere.py) is a point cloud in metres, neighbouring stops are
registered to each other (point-to-plane ICP, three scales), and one pose graph over the whole walk settles every stop's x, y and heading at once.

Usage:  python register_stops.py [--frames frames] [--reach 7] [--no-write]
Reads:  frames/poses.json (the sparse map's x, y, H and cam_h: the starting point), frames/up.json, frames/points.json, frames/depth_sphere/<id>.npz,
        frames/capturer/<id>_hole.png (the person holding the stick is never matched)
Writes: frames/poses.json (x, y, H moved; links' az/d/el redone; "registered" says how far each stop moved and how well the pairs agreed),
        frames/poses_sparse.json (the poses as they were, once), frames/register/stats.json

Why: the sparse map (COLMAP on a handful of points per pair of stops, scaled by the accelerometer) places a stop to within a few tens of centimetres and its
scale wanders along the walk; a dense metric depth of every stop gives the same wall from two stops, and ICP lines them up to centimetres. The dollhouse
and the floor plan are only as good as the stops' positions: two copies of one wall 20 cm apart is a torn wall.
Z and tilt are not changed: the floor was already put at z = 0 under every stop by the depth (stick height), and up came from the pictures.
"""
import argparse, json, math, os, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import dollhouse as D       # noqa: E402
import heading_check        # noqa: E402

SUB = 6                     # the 2048 x 1024 sphere sampled every 6th pixel: 58k rays a stop before the voxel grid
VOXEL = 0.07
MAX_R = 12.0
ACCEPT_DEG = 3.0            # a registration that leaves the stops' rings off their pictures by more than this (median) must beat the map to be kept


def stop_cloud(frames, p, node, up, voxel=VOXEL):
    """World-frame points (X east, Y north, Z up) of one stop, from its sphere depth: (N, 3) float64, with normals, as an Open3D cloud; None when there is no depth."""
    import open3d as o3d, cv2
    f = os.path.join(frames, "depth_sphere", p["id"] + ".npz")
    if not os.path.exists(f) or node.get("x") is None:
        return None
    z = np.load(f); rng = z["range"].astype(np.float32); conf = z["conf"]
    H, W = rng.shape
    rng = rng[::SUB, ::SUB]; conf = conf[::SUB, ::SUB]
    h, w = rng.shape
    az = (np.arange(w) + 0.5) / w * 2 * np.pi - np.pi; el = np.pi / 2 - (np.arange(h) + 0.5) / h * np.pi
    A, E = np.meshgrid(az, el)
    q = np.stack([np.cos(E) * np.sin(A), np.sin(E), np.cos(E) * np.cos(A)], -1)                        # levelled rays
    ok = (rng > 0.3) & (rng < MAX_R) & (conf > 60)
    hp = os.path.join(frames, "capturer", p["id"] + "_hole.png")
    if os.path.exists(hp):
        hole = D.hole_in_eq(cv2.imread(hp, 0) > 0, up, W, H)[::SUB, ::SUB]
        ok &= ~hole
    ok &= q[..., 1] > -0.985                                                                              # not the nadir itself
    pts_l = (q * rng[..., None])[ok]
    M = D.level_to_world(node["H"])
    pos = np.array([node["x"], node["y"], (node.get("cam_h") or 2.0) + float(node.get("z") or 0.0)])
    P = pts_l @ M.T + pos
    pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(P))
    pc = pc.voxel_down_sample(voxel)
    pc.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=voxel * 4, max_nn=30))
    pc.orient_normals_towards_camera_location(pos)
    return pc


def pair_icp(src, dst, init=np.eye(4)):
    """src onto dst (both already in world coordinates): the transform that moves src, its fitness and rmse, and the information matrix."""
    import open3d as o3d
    T = init.copy(); fit = rmse = 0.0
    for dist, iters in ((1.0, 15), (0.4, 15), (0.15, 20), (0.08, 20)):
        r = o3d.pipelines.registration.registration_icp(src, dst, dist, T, o3d.pipelines.registration.TransformationEstimationPointToPlane(),
                                                        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=iters))
        T, fit, rmse = r.transformation, r.fitness, r.inlier_rmse
    info = o3d.pipelines.registration.get_information_matrix_from_point_clouds(src, dst, 0.12, T)
    return np.asarray(T), fit, rmse, np.asarray(info)


def flatten(T):
    """Keep what the walk can change: a turn about Z and a move in X, Y and Z (up came from the pictures, so no tilt). Z moves because the ground is not
    one level: every stop's depth put ITS OWN ground at z = 0, and where a beach lies below a deck or a dock above a yard, the stops' clouds only
    meet when the stop is lifted or lowered to where its neighbours see that ground."""
    yaw = math.atan2(T[1, 0], T[0, 0])
    out = np.eye(4); c, s = math.cos(yaw), math.sin(yaw)
    out[:3, :3] = [[c, -s, 0], [s, c, 0], [0, 0, 1]]; out[0, 3], out[1, 3], out[2, 3] = T[0, 3], T[1, 3], T[2, 3]
    return out


def stamp_check(nodes, pairs):
    """Each placed stop's agreement with its own pictures, on the node: "check": {"deg": median degrees its rings are off, "pairs": how many} — the
    heading's confidence (heading_check.py); a stop no pair measured has none."""
    xy = {i: (nd["x"], nd["y"]) for i, nd in enumerate(nodes) if nd.get("x") is not None}
    se = heading_check.stop_errors(heading_check.errors_2d(xy, {i: nodes[i].get("H") for i in xy}, pairs))
    for i, nd in enumerate(nodes):
        nd.pop("check", None)
        if i in se:
            nd["check"] = {"deg": round(se[i][0], 1), "pairs": se[i][1]}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--frames", default="frames"); ap.add_argument("--reach", type=float, default=7.0, help="stops closer than this (m) are registered to each other")
    ap.add_argument("--k", type=int, default=5, help="each stop is registered to its k nearest stops (and the next one of the walk)")
    ap.add_argument("--min-fitness", type=float, default=0.35); ap.add_argument("--no-write", action="store_true")
    a = ap.parse_args()
    import open3d as o3d
    t0 = time.time()
    pf = os.path.join(a.frames, "poses.json"); poses = json.load(open(pf)); P = poses["nodes"]
    pts = json.load(open(os.path.join(a.frames, "points.json"))); UP = json.load(open(os.path.join(a.frames, "up.json")))
    sp = os.path.join(a.frames, "poses_sparse.json")
    if not os.path.exists(sp):
        json.dump(poses, open(sp, "w"), indent=1)
    clouds = {}
    for i, p in enumerate(pts):
        c = stop_cloud(a.frames, p, P[i], UP[i])
        if c is not None and len(c.points) > 2000:
            clouds[i] = c
    idx = sorted(clouds)
    if len(idx) < 2:
        raise SystemExit("fewer than two stops have a sphere depth: nothing to register (run depth_sphere.py first)")
    print(f"{len(idx)} stops as point clouds ({sum(len(clouds[i].points) for i in idx)} points) in {time.time() - t0:.0f} s", flush=True)
    xy = {i: np.array([P[i]["x"], P[i]["y"]]) for i in idx}
    pairs, seen = [], set()
    for n_, i in enumerate(idx):
        near = sorted(((float(np.linalg.norm(xy[i] - xy[j])), j) for j in idx if j != i))
        want = {j for d, j in near[:a.k] if d <= a.reach}
        if n_ + 1 < len(idx):
            want.add(idx[n_ + 1])
        for j in want:
            key = (min(i, j), max(i, j))
            if key not in seen:
                seen.add(key); pairs.append((key[0], key[1], float(np.linalg.norm(xy[key[0]] - xy[key[1]]))))
    pairs.sort()
    pg = o3d.pipelines.registration.PoseGraph()
    for i in idx:
        pg.nodes.append(o3d.pipelines.registration.PoseGraphNode(np.eye(4)))         # the clouds are already in world coordinates: every node starts at the identity
    pos_of = {i: k for k, i in enumerate(idx)}
    rep_pairs, kept = [], 0
    for i, j, d in pairs:
        T, fit, rmse, info = pair_icp(clouds[j], clouds[i])                         # T moves j's cloud onto i's: T_ij
        moved = float(np.hypot(T[0, 3], T[1, 3])); turned = math.degrees(math.atan2(T[1, 0], T[0, 0]))
        loose = bool(poses.get("positions_rescaled"))                                   # a map that had no metre of its own: the stops may be metres off
        ok = fit >= a.min_fitness and moved < (max(3.0, 0.5 * d) if loose else 1.5) and abs(turned) < (25 if loose else 15)
        consecutive = (pos_of[j] == pos_of[i] + 1)
        rep_pairs.append(dict(i=pts[i]["id"], j=pts[j]["id"], dist=round(d, 2), fitness=round(fit, 3), rmse=round(rmse, 4), moved=round(moved, 3), turned=round(turned, 2), used=bool(ok)))
        print(f"  {pts[i]['id']}-{pts[j]['id']}  {d:.1f} m apart  fitness {fit:.2f}  moved {moved:.2f} m  turned {turned:.1f} deg  {'ok' if ok else 'dropped'}  ({time.time() - t0:.0f} s)", flush=True)
        # Open3D's edge convention: the edge's transformation takes a point of the SOURCE node into the TARGET node's frame; nodes hold world <- node poses.
        # With every node at the identity and T moving j onto i, the edge (source j, target i) carries T.
        if ok:
            kept += 1
            pg.edges.append(o3d.pipelines.registration.PoseGraphEdge(pos_of[j], pos_of[i], T, info, uncertain=not consecutive))
        elif consecutive:
            # THE WALK STAYS ONE PIECE: a graph with a break in it is left where it was, every stop of it (Open3D moves nothing of a disconnected graph) —
            # so a pair of neighbours whose pictures would not line up keeps a weak edge: the ICP answer when it was half-way believable, else the map's
            # own relative pose (the identity, since the clouds already stand where the map put them), with little weight so the good pairs decide
            weakT = T if (fit >= 0.15 and moved < 3.0 and abs(turned) < 25) else np.eye(4)
            pg.edges.append(o3d.pipelines.registration.PoseGraphEdge(pos_of[j], pos_of[i], weakT, np.eye(6) * 5.0, uncertain=False))
    print(f"{kept} of {len(pairs)} pairs agreed (fitness >= {a.min_fitness}) in {time.time() - t0:.0f} s", flush=True)
    if kept < len(idx) - 1:
        print("warning: the walk is not fully connected by agreeing pairs; stops without a path keep their map position")
    opt = o3d.pipelines.registration.GlobalOptimizationOption(max_correspondence_distance=0.10, edge_prune_threshold=0.25, preference_loop_closure=0.4, reference_node=0)
    o3d.pipelines.registration.global_optimization(pg, o3d.pipelines.registration.GlobalOptimizationLevenbergMarquardt(),
                                                   o3d.pipelines.registration.GlobalOptimizationConvergenceCriteria(), opt)
    moves = []
    x0s = {i: (P[i]["x"], P[i]["y"], P[i]["H"]) for i in idx}
    for i in idx:
        T = flatten(np.asarray(pg.nodes[pos_of[i]].pose))                            # world <- node: where this stop's cloud should go
        x0, y0, H0, z0 = P[i]["x"], P[i]["y"], P[i]["H"], float(P[i].get("z") or 0.0)
        p_new = T @ np.array([x0, y0, 0.0, 1.0])
        yaw = math.degrees(math.atan2(T[1, 0], T[0, 0]))
        P[i]["x"], P[i]["y"], P[i]["H"] = round(float(p_new[0]), 3), round(float(p_new[1]), 3), round(float((H0 - yaw + 180) % 360 - 180), 2)
        P[i]["z"] = round(z0 + float(T[2, 3]), 3)                              # the stop's ground above the walk's zero (0 where the ground is one level)
        moves.append(dict(id=pts[i]["id"], moved=round(float(np.hypot(p_new[0] - x0, p_new[1] - y0)), 3), turned=round(yaw, 2), lifted=round(float(T[2, 3]), 3)))
    # THE PICTURES HAVE THE LAST WORD (2026-10-30, walk 16ddac14: 12 of 64 pairs agreed and the graph pulled the stops metres and up to 26 degrees,
    # so the rings pointed 8-20 degrees off where the pictures see the next stop, where the map had them within 3): the directions between stops that
    # hop_geometry measured in the pictures are checked before and after; a registration that leaves them off by more than ACCEPT_DEG (median) and
    # worse than the map was is not kept — x, y and heading go back to the map's, the ground's height (z) is kept
    pairs_pic = heading_check.good_pairs(json.load(open(os.path.join(a.frames, "hops.json")))) if os.path.exists(os.path.join(a.frames, "hops.json")) else []
    before = heading_check.summary(heading_check.errors_2d({i: (x0s[i][0], x0s[i][1]) for i in x0s}, {i: x0s[i][2] for i in x0s}, pairs_pic))[0]
    after = heading_check.summary(heading_check.errors_2d({i: (P[i]["x"], P[i]["y"]) for i in x0s}, {i: P[i]["H"] for i in x0s}, pairs_pic))[0]
    kept_reg = not (after is not None and before is not None and after > ACCEPT_DEG and after > before + 1.0)
    if not kept_reg:
        for i, (x0, y0, H0) in x0s.items():
            P[i]["x"], P[i]["y"], P[i]["H"] = x0, y0, H0
        print(f"the registration is not kept: the pictures disagree with it by {after} degrees (median), with the map by {before}; x, y and heading stay the map's")
    else:
        print(f"the pictures agree with the registered stops within {after} degrees (median; the map {before})")
    # the links: where the other stop is from here, redone from the new positions (occlusion and floor flags are kept)
    by_i = {i: P[i] for i in range(len(P))}
    for i, nd in enumerate(P):
        if nd.get("x") is None:
            continue
        for L in nd.get("links", []):
            o = by_i.get(L["to"])
            if not o or o.get("x") is None:
                continue
            dx, dy = o["x"] - nd["x"], o["y"] - nd["y"]; dist = math.hypot(dx, dy)
            L["az"] = round((math.degrees(math.atan2(dx, dy)) - nd["H"] + 180) % 360 - 180, 1)
            L["d"] = round(dist, 2)
            L["el"] = round(math.degrees(math.atan2(nd.get("cam_h") or 2.0, max(dist, 0.3))), 1)
            L["turn"] = round(nd["H"] - o["H"], 2)
    stamp_check(P, pairs_pic)
    # THE WALK'S ZERO IS THE LOWEST GROUND HE STOOD ON: the model's floor at z = 0 is where the plan and the viewer expect it
    zs = [float(P[i].get("z") or 0.0) for i in idx]
    if zs:
        base = float(np.percentile(zs, 5))
        for i in idx:
            P[i]["z"] = round(float(P[i].get("z") or 0.0) - base, 3)
    mv = np.array([m["moved"] for m in moves]); lz = np.array([m["lifted"] for m in moves])
    rep = dict(stops=len(idx), pairs=len(pairs), pairs_used=kept, moved_median=round(float(np.median(mv)), 3), moved_max=round(float(mv.max()), 3),
               lifted_spread=round(float(np.percentile(lz, 95) - np.percentile(lz, 5)), 3),
               fitness_median=round(float(np.median([r["fitness"] for r in rep_pairs if r["used"]] or [0])), 3), seconds=round(time.time() - t0, 1), moves=moves, pair_list=rep_pairs)
    os.makedirs(os.path.join(a.frames, "register"), exist_ok=True)
    json.dump(rep, open(os.path.join(a.frames, "register", "stats.json"), "w"), indent=1)
    poses["registered"] = {k: rep[k] for k in ("stops", "pairs", "pairs_used", "moved_median", "moved_max", "fitness_median")}
    poses["registered"].update(kept=kept_reg, pictures_deg_map=before, pictures_deg_registered=after)
    if not a.no_write:
        json.dump(poses, open(pf + ".tmp", "w"), indent=1)
        os.replace(pf + ".tmp", pf)                   # atomic: the early tour reads poses.json
    print(f"registered: stops moved {rep['moved_median']} m (median), {rep['moved_max']} m at most; their ground spans {rep['lifted_spread']} m of height; pairs' fitness {rep['fitness_median']} (median); {rep['seconds']} s")


if __name__ == "__main__":
    main()
