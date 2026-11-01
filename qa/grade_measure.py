"""Grade the tour's measurements against tape-measured ground truth.

  python qa/grade_measure.py <walk folder> <picks.json> [--truth D:/walks/place/ground_truth.json] [--calib door_w] [--json out.json]

picks.json: {"items": {"door_w": {"kind": "dist", "points": [{"stop": "014", "az": 12.0, "el": 3.0}, {...}]}, ...}}
  kind "dist": straight-line distance between the two picked points (metres -> inches);  kind "height": height of the 2nd point above the 1st (world Z).
A point is a direction (azimuth clockwise from the picture centre, elevation, degrees) in a stop's panorama; its 3D position is that stop's position plus the
measured range along that ray (frames/depth/eq_<id>.png, place_rings.py) in the shared map frame (frames/poses.json, metres). Two points may be in different stops.
Modes: (a) uncalibrated, (b) every length multiplied by one scale factor = tape value of the calibration item / measured value of that item. The calibration item
is excluded from the grade of (b); nothing else is used to tune anything. Prints the error per item and the median absolute % error.
"""
import argparse, json, math, os, sys
import cv2
import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "qa"))
import dollhouse as D            # noqa: E402
from imutil import imread        # noqa: E402

IN = 0.0254


class Walk:
    mode = "mesh"                                          # "mesh": the ray from the stop is cast into the dollhouse model; "depth": the stop's own depth map

    def __init__(self, walk, mode="mesh"):
        self.mode = mode; self._scene = None
        self.fr = os.path.join(walk, "frames")
        self.pts = json.load(open(os.path.join(self.fr, "points.json")))
        self.P = json.load(open(os.path.join(self.fr, "poses.json")))["nodes"]
        self.UP = json.load(open(os.path.join(self.fr, "up.json")))
        self.idx = {p["id"]: i for i, p in enumerate(self.pts)}
        self.rng = {}

    def range_map(self, sid):
        if sid not in self.rng:
            self.rng[sid] = D.decode_range(imread(os.path.join(self.fr, os.environ.get("DEPTH_DIR", "depth"), f"eq_{sid}.png")))
        return self.rng[sid]

    def scene(self):
        if self._scene is None:
            import open3d as o3d
            m = o3d.io.read_triangle_mesh(os.path.join(self.fr, "dollhouse", os.environ.get("MESH_NAME", "mesh") + ".ply"))
            self._scene = o3d.t.geometry.RaycastingScene(); self._scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(m))
        return self._scene

    def point(self, sid, az, el, win=2):
        if self.mode == "mesh":
            return self.point_mesh(sid, az, el)
        return self.point_depth(sid, az, el, win)

    def point_mesh(self, sid, az, el):
        import open3d as o3d
        i = self.idx[sid]; node = self.P[i]
        a, e = math.radians(az), math.radians(el)
        d_pano = np.array([math.cos(e) * math.sin(a), math.sin(e), math.cos(e) * math.cos(a)])
        d_world = D.level_to_world(node["H"]) @ (D.rot_to_y(self.UP[i]) @ d_pano)
        pos = np.array([node["x"], node["y"], (node["cam_h"] or 2.0) + float(node.get("z") or 0.0)])
        ray = o3d.core.Tensor([[*pos, *d_world]], dtype=o3d.core.Dtype.Float32)
        t = float(self.scene().cast_rays(ray)["t_hit"].numpy()[0])
        if not math.isfinite(t):
            raise SystemExit(f"the ray from stop {sid} (az {az:.1f}, el {el:.1f}) hits nothing in the model")
        return pos + d_world * t, t

    def point_depth(self, sid, az, el, win=2):
        """World position (X east, Y north, Z up, metres) and the range (m) of the direction (az, el) in stop `sid`; the median range of a small window (win px)."""
        i = self.idx[sid]; node = self.P[i]
        r = self.range_map(sid); H, W = r.shape
        u = int(((az + 180) % 360) / 360 * W) % W; v = int(np.clip((90 - el) / 180 * H, 0, H - 1))
        win_r = r[max(0, v - win):v + win + 1, max(0, u - win):u + win + 1]
        rg = float(np.median(win_r))
        a, e = math.radians(az), math.radians(el)
        d_pano = np.array([math.cos(e) * math.sin(a), math.sin(e), math.cos(e) * math.cos(a)])
        d_level = D.rot_to_y(self.UP[i]) @ d_pano
        pos = np.array([node["x"], node["y"], (node["cam_h"] or 2.0) + float(node.get("z") or 0.0)])
        return pos + D.level_to_world(node["H"]) @ (d_level * rg), rg


def resolve(p):
    """A pick is either {stop, az, el} or {stop, view: [yaw, pitch, fov], px: [x, y]} (a pixel of qa/persp.py's 1000 px view)."""
    if "px" in p:
        import persp
        az, el = persp.pixel_to_azel(p["px"][0], p["px"][1], *p["view"])
        return p["stop"], az, el
    return p["stop"], p["az"], p["el"]


def long_axis_theta(plan):
    """The plan's axis along the building's LONG side (the plan's u axis can come out either way round: theta and theta + 90 describe the same rectangle)."""
    th = float(plan["theta_deg"])
    sz = plan.get("outer_size_m")
    return th + 90.0 if sz and sz[0] < sz[1] else th


def measure(w, item, theta_deg=0.0):
    pts = [w.point(*resolve(p)) for p in item["points"]]
    a, b = pts[0][0], pts[-1][0]
    if item["kind"] == "height":
        m = float(abs(b[2] - a[2]))
    elif item["kind"] == "axis":                                    # along the building's main axis (the wall direction of the plan)
        t = math.radians(theta_deg); m = float(abs((b[0] - a[0]) * math.cos(t) + (b[1] - a[1]) * math.sin(t)))
    else:
        m = float(np.linalg.norm(b - a))
    return m / IN, [round(p[1], 2) for p in pts]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("walk"); ap.add_argument("picks")
    ap.add_argument("--truth", default="D:/walks/place/ground_truth.json"); ap.add_argument("--calib", default="door_w"); ap.add_argument("--mode", default="mesh", choices=["mesh", "depth"]); ap.add_argument("--json", default=None)
    a = ap.parse_args()
    truth = {t["id"]: t for t in json.load(open(a.truth, encoding="utf-8"))["items"]}
    picks = json.load(open(a.picks))["items"]
    w = Walk(a.walk, a.mode)
    print(f"measuring in: {'the dollhouse model (rays cast into the fused mesh)' if a.mode == 'mesh' else 'the single-stop depth maps'}")
    pj = os.path.join(w.fr, "dollhouse", "plan.json")
    theta = long_axis_theta(json.load(open(pj))) if os.path.exists(pj) else 0.0
    rows, meas = [], {}
    for k, item in picks.items():
        if k in truth:
            meas[k], ranges = measure(w, item, theta)
    scale = truth[a.calib]["value"] / meas[a.calib] if a.calib in meas else 1.0
    out = {"scale": scale, "items": {}}
    print(f"{'item':12s} {'tape in':>8s} {'model in':>9s} {'err in':>7s} {'err %':>6s} | {'calib in':>9s} {'err in':>7s} {'err %':>6s}")
    ea, eb = [], []
    for k, t in truth.items():
        if k not in meas:
            print(f"{k:12s} {t['value']:8.1f}   (no picks)"); continue
        m = meas[k]; ca = m * scale
        pa = 100 * abs(m - t["value"]) / t["value"]; pb = 100 * abs(ca - t["value"]) / t["value"]
        ea.append(pa)
        if k != a.calib:
            eb.append(pb)
        out["items"][k] = dict(tape=t["value"], model=round(m, 1), err_in=round(m - t["value"], 1), err_pct=round(pa, 1), calibrated=round(ca, 1),
                               cal_err_in=round(ca - t["value"], 1), cal_err_pct=round(pb, 1), held_out=(k != a.calib), within_tolerance=abs(m - t["value"]) <= t["tolerance_in"])
        print(f"{k:12s} {t['value']:8.1f} {m:9.1f} {m - t['value']:+7.1f} {pa:6.1f} | {ca:9.1f} {ca - t['value']:+7.1f} {pb:6.1f}{'  (calibration item)' if k == a.calib else ''}")
    out["median_pct_uncalibrated"] = round(float(np.median(ea)), 1) if ea else None
    out["median_pct_calibrated_held_out"] = round(float(np.median(eb)), 1) if eb else None
    print(f"\nmedian abs error: uncalibrated {out['median_pct_uncalibrated']} %   calibrated with {a.calib} only, held-out items {out['median_pct_calibrated_held_out']} %   (scale {scale:.4f})")
    if a.json:
        json.dump(out, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
