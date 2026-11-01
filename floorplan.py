"""A floor plan with dimensions from the dollhouse model: slice the mesh at about 1 m, find the wall lines, draw them with their lengths.

Usage:  python floorplan.py [--frames frames] [--out tour] [--z 1.0]
Reads:  frames/dollhouse/mesh.ply (dollhouse.py), frames/poses.json, frames/privacy.json
Writes: tour/plan.svg (the viewer's Plan button shows it), frames/dollhouse/plan.json (the wall lines and the outer size, in metres)

How: every mesh triangle that crosses the slice heights (z, z +- 0.15 m) gives a line segment; the segments' directions give the building's main axis
(walls are straight and meet at right angles); along that axis and across it the segment length is counted in 4 cm bins; a wall shows up as a narrow,
tall peak. The outermost long peaks are the outer walls, the other long ones interior walls and racks. Everything is in metres; lengths are written in
feet and inches as well (the shop was measured with a tape in inches).
"""
import argparse, json, math, os
import numpy as np


def slice_segments(V, T, z0):
    """Segments (N, 2, 2) of the mesh at height z0 (x, y)."""
    z = V[T][:, :, 2]
    cross = (z.min(1) <= z0) & (z.max(1) >= z0)
    tri = V[T[cross]]                                    # (n, 3, 3)
    segs = []
    for t in tri:
        pts = []
        for a, b in ((0, 1), (1, 2), (2, 0)):
            za, zb = t[a, 2] - z0, t[b, 2] - z0
            if za == 0 and zb == 0:
                continue
            if za * zb <= 0 and za != zb:
                k = za / (za - zb); pts.append(t[a, :2] + k * (t[b, :2] - t[a, :2]))
        if len(pts) == 2:
            segs.append(pts)
    return np.array(segs) if segs else np.zeros((0, 2, 2))


def main_axis(segs):
    d = segs[:, 1] - segs[:, 0]
    L = np.hypot(d[:, 0], d[:, 1]); ang = np.degrees(np.arctan2(d[:, 1], d[:, 0])) % 90.0
    hist = np.zeros(90)
    np.add.at(hist, np.minimum((ang).astype(int), 89), L)
    hist = np.convolve(np.r_[hist[-3:], hist, hist[:3]], np.ones(5) / 5, mode="same")[3:-3]
    k = int(np.argmax(hist)); a = ((ang - k + 45) % 90) - 45
    ok = np.abs(a) < 6
    return float(k + np.average(a[ok], weights=L[ok]) if ok.any() else k)


def sample_points(segs, step=0.02):
    pts, wts = [], []
    d = segs[:, 1] - segs[:, 0]; L = np.hypot(d[:, 0], d[:, 1])
    for s, l in zip(segs, L):
        n = max(1, int(l / step)); t = (np.arange(n) + 0.5) / n
        pts.append(s[0] + t[:, None] * (s[1] - s[0]))
    return np.concatenate(pts)


def peaks(h, bin_m, min_len_m, sep_m=0.25):
    """Bin indices of the local maxima of the (3-bin smoothed) histogram whose line length is at least min_len_m, strongest first, at least sep_m apart."""
    hs = np.convolve(h, np.ones(3), mode="same")
    out = []
    for k in np.argsort(-hs):
        if hs[k] < min_len_m:
            break
        if all(abs(k - j) * bin_m > sep_m for j in out):
            out.append(int(k))
    return sorted(out)


def lines_along(P, axis, pos, tol, min_len=1.0, gap=0.6):
    """Segments of the wall at coordinate `pos` of `axis` (0 = u, 1 = v): where along the other axis points lie within `tol` of it, merging gaps up to `gap`."""
    other = 1 - axis
    sel = np.abs(P[:, axis] - pos) < tol
    c = np.sort(P[sel][:, other])
    if len(c) == 0:
        return []
    out, s, prev = [], c[0], c[0]
    for x in c[1:]:
        if x - prev > gap:
            if prev - s >= min_len:
                out.append((s, prev))
            s = x
        prev = x
    if prev - s >= min_len:
        out.append((s, prev))
    return out


def ft_in(m):
    inch = m / 0.0254; ft = int(inch // 12); r = inch - 12 * ft
    return f"{ft}' {r:.0f}\""


def sample_surface(V, T, density=300.0, seed=0):
    """Points spread over the mesh surface (about `density` per square metre)."""
    rng = np.random.default_rng(seed)
    A, B, C = V[T[:, 0]], V[T[:, 1]], V[T[:, 2]]
    area = 0.5 * np.linalg.norm(np.cross(B - A, C - A), axis=1)
    n = np.maximum(1, np.round(area * density)).astype(int)
    n = np.minimum(n, 4000)
    idx = np.repeat(np.arange(len(T)), n)
    r1, r2 = rng.random(len(idx)), rng.random(len(idx)); flip = r1 + r2 > 1; r1[flip], r2[flip] = 1 - r1[flip], 1 - r2[flip]
    return A[idx] + r1[:, None] * (B[idx] - A[idx]) + r2[:, None] * (C[idx] - A[idx])


def wall_cells(V, T, cell=0.05, low=(0.3, 1.2), high=(2.8, 4.6)):
    """Centres of the 5 cm cells of the plan where a surface runs from near the floor up to the roof: walls reach the ceiling, racks and benches do not."""
    S = sample_surface(V, T)
    lo = S[:, :2].min(0) - 0.5
    ij = np.floor((S[:, :2] - lo) / cell).astype(int); shape = ij.max(0) + 2
    has_low = np.zeros(shape, bool); has_high = np.zeros(shape, bool)
    m = (S[:, 2] >= low[0]) & (S[:, 2] <= low[1]); has_low[ij[m, 0], ij[m, 1]] = True
    m = (S[:, 2] >= high[0]) & (S[:, 2] <= high[1]); has_high[ij[m, 0], ij[m, 1]] = True
    import cv2
    k = np.ones((7, 7), np.uint8)                                                      # the top may lean a little: look 15 cm around
    both = has_low & (cv2.dilate(has_high.astype(np.uint8), k) > 0)
    xs, ys = np.nonzero(both)
    return np.stack([(xs + 0.5) * cell + lo[0], (ys + 0.5) * cell + lo[1]], 1)


def build_plan_from_points(P_xy, theta=None, bin_m=0.05, min_wall_m=1.0):
    """Wall lines from cell centres: axes from the data (or `theta`), then peaks of the counts along and across it."""
    if theta is None:
        # the direction of the points' local structure: dominant orientation of the 2D point cloud via the covariance of neighbouring differences is fragile;
        # use the Hough-like histogram of angles between nearby points instead
        from scipy.spatial import cKDTree
        t = cKDTree(P_xy); d, nn = t.query(P_xy, k=6)
        vec = (P_xy[nn[:, 1:]] - P_xy[:, None, :]).reshape(-1, 2)
        ang = np.degrees(np.arctan2(vec[:, 1], vec[:, 0])) % 90.0; L = np.hypot(vec[:, 0], vec[:, 1])
        hist = np.zeros(90); np.add.at(hist, np.minimum(ang.astype(int), 89), 1.0 / np.maximum(L, 0.03))
        hist = np.convolve(np.r_[hist[-3:], hist, hist[:3]], np.ones(5) / 5, mode="same")[3:-3]
        theta = float(np.argmax(hist)) + 0.5
    c, s = math.cos(math.radians(-theta)), math.sin(math.radians(-theta)); R = np.array([[c, -s], [s, c]])
    P = P_xy @ R.T
    lo = P.min(0) - 0.5; hi = P.max(0) + 0.5
    walls = []
    for axis in (0, 1):
        edges = np.arange(lo[axis], hi[axis] + bin_m, bin_m)
        h, _ = np.histogram(P[:, axis], bins=edges); h = h * 0.05                      # metres of wall per bin (cells are 5 cm)
        for k in peaks(h, bin_m, min_wall_m * 0.5):
            pos = edges[k] + bin_m / 2
            for a, b in lines_along(P, axis, pos, bin_m * 2.0, min_len=min_wall_m, gap=1.0):
                walls.append(dict(axis="uv"[axis], pos=float(pos), a=float(a), b=float(b), length=float(b - a)))
    return dict(theta=theta, R=R.tolist(), walls=walls, lo=lo.tolist(), hi=hi.tolist(), points=P)


def build_plan(V, T, z0=1.0, bin_m=0.04, min_wall_m=0.7):
    segs = np.concatenate([slice_segments(V, T, z) for z in (z0 - 0.15, z0, z0 + 0.15)])
    theta = main_axis(segs)
    c, s = math.cos(math.radians(-theta)), math.sin(math.radians(-theta))
    R = np.array([[c, -s], [s, c]])
    P = sample_points(segs) @ R.T                                                      # u along the main axis, v across it
    lo = P.min(0) - 0.5; hi = P.max(0) + 0.5
    walls = []
    for axis in (0, 1):
        edges = np.arange(lo[axis], hi[axis] + bin_m, bin_m)
        h, _ = np.histogram(P[:, axis], bins=edges); h = h * 0.02                      # metres of line per bin (points are 2 cm apart)
        for k in peaks(h, bin_m, min_wall_m * 0.5):
            pos = edges[k] + bin_m / 2
            for a, b in lines_along(P, axis, pos, bin_m * 1.5, min_len=min_wall_m):
                walls.append(dict(axis="uv"[axis], pos=float(pos), a=float(a), b=float(b), length=float(b - a)))
    return dict(theta=theta, R=R.tolist(), walls=walls, lo=lo.tolist(), hi=hi.tolist(), points=P)


def outer(plan):
    """The outer rectangle: the outermost long lines of each direction."""
    out = {}
    for ax in ("u", "v"):
        w = [x for x in plan["walls"] if x["axis"] == ax and x["length"] >= 2.5]
        if w:
            out[ax] = (min(x["pos"] for x in w), max(x["pos"] for x in w))
    return out


def merge_walls(walls, min_len=3.0, near=0.45):
    """One line per wall: of parallel lines closer than `near` that overlap, only the longest is kept; short ones are dropped."""
    keep = []
    for w in sorted((w for w in walls if w["length"] >= min_len), key=lambda w: -w["length"]):
        dup = False
        for k in keep:
            if k["axis"] == w["axis"] and abs(k["pos"] - w["pos"]) < near:
                ov = min(k["b"], w["b"]) - max(k["a"], w["a"])
                if ov > 0.5 * min(k["length"], w["length"]):
                    dup = True; break
        if not dup:
            keep.append(w)
    return keep


def svg(plan, stops, title, scale=45.0):
    lo, hi = np.array(plan["lo"]), np.array(plan["hi"])
    W, H = (hi - lo) * scale + 120, (hi - lo)[1] * scale + 140
    W = (hi - lo)[0] * scale + 120
    X = lambda u: (u - lo[0]) * scale + 60
    Y = lambda v: (hi[1] - v) * scale + 60                                            # v up on the page
    o = []
    o.append(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W:.0f} {H:.0f}" width="{W:.0f}" height="{H:.0f}" role="img" aria-label="{title}">')
    o.append(f'<rect width="{W:.0f}" height="{H:.0f}" fill="#f4f1ea"/><g font-family="system-ui, sans-serif" fill="#222">')
    o.append(f'<text x="60" y="34" font-size="20" font-weight="700">{title}</text>')
    for w in merge_walls(plan["walls"]):
        if w["axis"] == "u":
            x1, y1, x2, y2 = X(w["pos"]), Y(w["a"]), X(w["pos"]), Y(w["b"])
        else:
            x1, y1, x2, y2 = X(w["a"]), Y(w["pos"]), X(w["b"]), Y(w["pos"])
        major = w["length"] >= 3.0
        o.append(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{"#222" if major else "#8a8577"}" stroke-width="{5 if major else 3}" stroke-linecap="round"/>')
    out = outer(plan)
    if "u" in out and "v" in out:                                                      # dimensions of the outer rectangle
        u0, u1 = out["u"]; v0, v1 = out["v"]
        o.append(f'<g stroke="#2b6cb0" stroke-width="1.5" fill="#2b6cb0" font-size="16">')
        yy = Y(v0) + 34
        o.append(f'<line x1="{X(u0):.1f}" y1="{yy:.1f}" x2="{X(u1):.1f}" y2="{yy:.1f}"/><line x1="{X(u0):.1f}" y1="{yy - 6:.1f}" x2="{X(u0):.1f}" y2="{yy + 6:.1f}"/><line x1="{X(u1):.1f}" y1="{yy - 6:.1f}" x2="{X(u1):.1f}" y2="{yy + 6:.1f}"/>')
        o.append(f'<text x="{(X(u0) + X(u1)) / 2:.1f}" y="{yy + 22:.1f}" text-anchor="middle" stroke="none">{ft_in(u1 - u0)}  ({u1 - u0:.2f} m)</text>')
        xx = X(u0) - 34
        o.append(f'<line x1="{xx:.1f}" y1="{Y(v0):.1f}" x2="{xx:.1f}" y2="{Y(v1):.1f}"/><line x1="{xx - 6:.1f}" y1="{Y(v0):.1f}" x2="{xx + 6:.1f}" y2="{Y(v0):.1f}"/><line x1="{xx - 6:.1f}" y1="{Y(v1):.1f}" x2="{xx + 6:.1f}" y2="{Y(v1):.1f}"/>')
        o.append(f'<text transform="translate({xx - 10:.1f} {(Y(v0) + Y(v1)) / 2:.1f}) rotate(-90)" text-anchor="middle" stroke="none">{ft_in(v1 - v0)}  ({v1 - v0:.2f} m)</text></g>')
    R = np.array(plan["R"])
    for sid, x, y in stops:                                                            # stop positions in the same rotated frame
        u, v = R @ np.array([x, y])
        o.append(f'<circle cx="{X(u):.1f}" cy="{Y(v):.1f}" r="5" fill="#e8590c"/><text x="{X(u) + 7:.1f}" y="{Y(v) - 6:.1f}" font-size="12" fill="#e8590c">{sid}</text>')
    sx, sy = 60, H - 30                                                                # scale bar: 1 m and 5 ft
    o.append(f'<line x1="{sx}" y1="{sy}" x2="{sx + scale:.0f}" y2="{sy}" stroke="#222" stroke-width="3"/><text x="{sx + scale + 8:.0f}" y="{sy + 5}" font-size="14">1 m ({ft_in(1.0)})</text>')
    o.append('<text x="' + f'{W - 60:.0f}" y="{H - 24:.0f}" text-anchor="end" font-size="12" fill="#777">made from the dollhouse model: lines at about 1 m, walls and racks</text>')
    o.append("</g></svg>")
    return "\n".join(o)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--frames", default="frames"); ap.add_argument("--out", default="tour"); ap.add_argument("--z", type=float, default=1.0)
    ap.add_argument("--title", default="Floor plan")
    a = ap.parse_args()
    import open3d as o3d
    m = o3d.io.read_triangle_mesh(os.path.join(a.frames, "dollhouse", "mesh.ply"))
    V, T = np.asarray(m.vertices), np.asarray(m.triangles)
    plan = build_plan_from_points(wall_cells(V, T))
    plan_slice = build_plan(V, T, a.z)
    plan["slice_walls"] = plan_slice["walls"]
    P = json.load(open(os.path.join(a.frames, "poses.json")))["nodes"]
    pts = json.load(open(os.path.join(a.frames, "points.json")))
    pv = json.load(open(os.path.join(a.frames, "privacy.json")))["stops"] if os.path.exists(os.path.join(a.frames, "privacy.json")) else {}
    stops = [(p["id"], n["x"], n["y"]) for p, n in zip(pts, P) if n.get("x") is not None and not (pv.get(p["id"], {}).get("held") and not pv.get(p["id"], {}).get("kept"))]
    open(os.path.join(a.out, "plan.svg"), "w", encoding="utf-8").write(svg(plan, stops, a.title))
    out = outer(plan)
    info = dict(theta_deg=plan["theta"], walls=plan["walls"], outer={k: [float(x) for x in v] for k, v in out.items()},
                outer_size_m=[float(out["u"][1] - out["u"][0]), float(out["v"][1] - out["v"][0])] if len(out) == 2 else None)
    json.dump(info, open(os.path.join(a.frames, "dollhouse", "plan.json"), "w"), indent=1)
    tj = os.path.join(a.out, "tour.json")
    if os.path.exists(tj):
        t = json.load(open(tj)); t["plan"] = True; json.dump(t, open(tj, "w"), separators=(",", ":"))
    sz = info["outer_size_m"]
    print(f"plan: {len(plan['walls'])} wall lines, main axis {plan['theta']:.1f} deg" + (f", outer {sz[0]:.2f} x {sz[1]:.2f} m ({ft_in(sz[0])} x {ft_in(sz[1])})" if sz else ""))


if __name__ == "__main__":
    main()
