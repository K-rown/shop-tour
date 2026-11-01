"""The floor plan: the dollhouse seen straight down with the roof taken off — the real floor, the racks, the benches, the walls as the model has them —
with the walls traced over it as lines and the building's size written on, in feet and inches and in metres.

Usage:  python floorplan.py [--frames frames] [--out tour] [--cut 2.3] [--px-per-m 100]
Reads:  frames/dollhouse/mesh_uv.npz + frames/dollhouse/mesh_atlas.jpg (dollhouse.py: the textured mesh), frames/dollhouse/mesh.ply (the untextured mesh, fallback),
        frames/poses.json, frames/points.json, frames/privacy.json
Writes: tour/plan.png (the top-down picture, 1 cm a pixel), tour/plan.svg (the picture with the wall lines, the dimensions and the stops drawn on it; the viewer's
        Plan button shows it), frames/dollhouse/plan.json (the wall lines, the outer size, the picture's frame, in metres)

How: (1) the picture: every triangle whose centre is under the cut height (the roof, the lights and the top of the ducting go) is sampled densely (about
2,500 points a square metre, colour from the atlas); the points are dropped into a 1 cm grid, the highest one in each cell wins (what you would see from above);
the few cells nothing landed in take the nearest colour. Vertical surfaces seen edge-on (the walls) get a thin dark ink so wall thickness reads.
(2) the walls: the mesh is sampled, and the 5 cm cells where a vertical surface runs from knee height to above head height are the walls (racks and benches do not
reach that high); the walls' main axis is found (walls are straight and meet at right angles), and along it and across it the cells are counted in 4 cm bins — a
wall is a narrow, tall peak. Only lines the model backs are drawn. The outer size is the outermost long lines.
"""
import argparse, json, math, os, time
import cv2
import numpy as np

MIN_WALL_SUPPORT = 0.5          # a wall line is drawn only if at least this share of its length has a vertical surface of the model beside it (wall_support)


def ft_in(m):
    inch = m / 0.0254; ft = int(inch // 12); r = inch - 12 * ft
    if round(r) == 12:
        ft, r = ft + 1, 0
    return f"{ft}' {r:.0f}\""


def sample_surface(V, T, density=300.0, seed=0, max_per_tri=4000):
    """Points spread over the mesh surface (about `density` per square metre): (points, triangle index of each)."""
    rng = np.random.default_rng(seed)
    A, B, C = V[T[:, 0]], V[T[:, 1]], V[T[:, 2]]
    area = 0.5 * np.linalg.norm(np.cross(B - A, C - A), axis=1)
    n = np.maximum(1, np.round(area * density)).astype(int)
    n = np.minimum(n, max_per_tri)
    idx = np.repeat(np.arange(len(T)), n)
    r1, r2 = rng.random(len(idx)), rng.random(len(idx)); flip = r1 + r2 > 1; r1[flip], r2[flip] = 1 - r1[flip], 1 - r2[flip]
    return A[idx] + r1[:, None] * (B[idx] - A[idx]) + r2[:, None] * (C[idx] - A[idx]), idx, (r1, r2)


def wall_grid(V, T, cell=0.05):
    """A 5 cm grid of the plan (True = wall): a cell where a VERTICAL surface of the model stands in every height band from knee height up to near the roof.
    A rack or a bench reaches 2 to 2.5 m and fails the top band; a wall, a column or a door reaches the roof. Returns (grid, lo) with grid[i, j] at (lo + (i + 0.5) cell)."""
    S, tri, _ = sample_surface(V, T, density=400.0)
    nrm = np.cross(V[T[:, 1]] - V[T[:, 0]], V[T[:, 2]] - V[T[:, 0]]); nrm /= np.maximum(np.linalg.norm(nrm, axis=1, keepdims=True), 1e-12)
    vert = np.abs(nrm[tri, 2]) < 0.6
    S = S[vert]
    roof = float(np.percentile(V[:, 2], 98))
    bands = [(0.3, 1.2), (1.2, 2.1)]
    if roof > 3.3:
        bands.append((2.1, min(3.0, roof - 0.5)))
    if roof > 3.8:
        bands.append((3.0, roof - 0.5))
    lo = V[:, :2].min(0) - 0.5
    shape = np.floor((V[:, :2].max(0) + 0.5 - lo) / cell).astype(int) + 2
    grid = np.ones(shape, bool)
    k = np.ones((3, 3), np.uint8)
    for z0, z1 in bands:
        m = (S[:, 2] >= z0) & (S[:, 2] <= z1)
        ij = np.floor((S[m, :2] - lo) / cell).astype(int)
        g = np.zeros(shape, bool); g[np.clip(ij[:, 0], 0, shape[0] - 1), np.clip(ij[:, 1], 0, shape[1] - 1)] = True
        grid &= cv2.dilate(g.astype(np.uint8), k) > 0                                      # a wall leans a little between bands: 5 cm of give
    return grid, lo, cell                                                                    # no opening: a true wall plane can be one cell thin


def wall_cells(V, T, cell=0.05):
    """Centres of the wall cells (plan metres, the model's own frame)."""
    grid, lo, cell = wall_grid(V, T, cell)
    xs, ys = np.nonzero(grid)
    return np.stack([(xs + 0.5) * cell + lo[0], (ys + 0.5) * cell + lo[1]], 1)


def peaks(h, bin_m, min_len_m, sep_m=0.25):
    hs = np.convolve(h, np.ones(3), mode="same")
    out = []
    for k in np.argsort(-hs):
        if hs[k] < min_len_m:
            break
        if all(abs(k - j) * bin_m > sep_m for j in out):
            out.append(int(k))
    return sorted(out)


def lines_along(P, axis, pos, tol, min_len=1.0, gap=0.6):
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


def build_plan_from_points(P_xy, theta=None, bin_m=0.05, min_wall_m=1.0):
    """Wall lines from cell centres: axes from the data (or `theta`), then peaks of the counts along and across it."""
    if theta is None:
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
    return dict(theta=theta, R=R.tolist(), walls=walls, lo=lo.tolist(), hi=hi.tolist())


def main_axis_normals(V, T):
    """The building's main axis (degrees, 0..90) from the directions the model's vertical surfaces FACE, weighted by area: walls are straight and meet at right
    angles, so their normals pile up at one angle and that angle plus 90. (Angles between 5 cm grid cells pile up on the grid's own axes instead.)"""
    n = np.cross(V[T[:, 1]] - V[T[:, 0]], V[T[:, 2]] - V[T[:, 0]]); area = np.linalg.norm(n, axis=1) / 2
    ok = (area > 1e-9) & (np.abs(n[:, 2]) / np.maximum(2 * area, 1e-12) < 0.5)
    ang = np.degrees(np.arctan2(n[ok, 1], n[ok, 0])) % 90.0
    hist = np.zeros(180); np.add.at(hist, np.minimum((ang * 2).astype(int), 179), area[ok])
    hist = np.convolve(np.r_[hist[-4:], hist, hist[:4]], np.ones(5) / 5, mode="same")[4:-4]
    k = int(np.argmax(hist)); a = ((ang - k / 2 + 45) % 90) - 45
    sel = np.abs(a) < 4
    return float(k / 2 + (np.average(a[sel], weights=area[ok][sel]) if sel.any() else 0.0)) % 90.0


def build_plan(V, T, z0=1.0, **kw):
    """The plan's wall lines from the mesh (z0 is kept for the old callers: the walls come from wall_grid now, every band, not one slice)."""
    kw.setdefault("theta", main_axis_normals(V, T))
    return build_plan_from_points(wall_cells(V, T), **kw)


def outer(plan):
    if "outer_override" in plan:
        return plan["outer_override"]
    out = {}
    for ax in ("u", "v"):
        w = [x for x in plan["walls"] if x["axis"] == ax and x["length"] >= 2.5]
        if w:
            out[ax] = (min(x["pos"] for x in w), max(x["pos"] for x in w))
    return out


def merge_walls(walls, min_len=3.0, near=0.45):
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


def wall_support(V, T, walls, theta_deg, cell=0.05, reach=2):
    nrm = np.cross(V[T[:, 1]] - V[T[:, 0]], V[T[:, 2]] - V[T[:, 0]]); area = np.linalg.norm(nrm, axis=1)
    vertical = np.abs(nrm[:, 2]) / np.maximum(area, 1e-12) < 0.5
    cen = V[T].mean(1)
    occ = {}
    for z in (0.5, 1.0, 1.5, 2.0):
        sel = vertical & (np.abs(cen[:, 2] - z) < 0.3)
        occ[z] = set(map(tuple, np.floor(cen[sel][:, :2] / cell).astype(int)))
    th = math.radians(-theta_deg); Rm = np.array([[math.cos(th), -math.sin(th)], [math.sin(th), math.cos(th)]])
    out = []
    for w in walls:
        uv_a, uv_b = ((w["pos"], w["a"]), (w["pos"], w["b"])) if w["axis"] == "u" else ((w["a"], w["pos"]), (w["b"], w["pos"]))
        a, b = Rm.T @ np.array(uv_a), Rm.T @ np.array(uv_b)
        n = max(2, int(np.linalg.norm(b - a) / 0.05)); ok = 0
        for k in range(n):
            c = np.floor((a + (b - a) * (k + 0.5) / n) / cell).astype(int)
            near = sum(any((c[0] + dx, c[1] + dy) in S for dx in range(-reach, reach + 1) for dy in range(-reach, reach + 1)) for S in occ.values())
            ok += near >= 3
        out.append(ok / n)
    return out


def raster_top(P2, Z, T, UV, atlas, lo, hi, px_per_m, chunk=6_000_000):
    """Every triangle rasterised from above into a 1/px_per_m grid: the highest surface in each cell wins (what you would see from above).
    P2: vertices in plan metres (n, 2), Z: their heights, T: triangles. Returns colour (H, W, 3) uint8, height (H, W) float32 (-9 = nothing), got (H, W) bool."""
    W = int(math.ceil((hi[0] - lo[0]) * px_per_m)); H = int(math.ceil((hi[1] - lo[1]) * px_per_m))
    X = (P2[:, 0] - lo[0]) * px_per_m; Y = (hi[1] - P2[:, 1]) * px_per_m                                  # pixel coordinates (row 0 at the top = v max)
    A, B, C = np.stack([X[T[:, 0]], Y[T[:, 0]]], 1), np.stack([X[T[:, 1]], Y[T[:, 1]]], 1), np.stack([X[T[:, 2]], Y[T[:, 2]]], 1)
    x0 = np.clip(np.floor(np.minimum(np.minimum(A[:, 0], B[:, 0]), C[:, 0])).astype(int), 0, W - 1); x1 = np.clip(np.ceil(np.maximum(np.maximum(A[:, 0], B[:, 0]), C[:, 0])).astype(int), 0, W - 1)
    y0 = np.clip(np.floor(np.minimum(np.minimum(A[:, 1], B[:, 1]), C[:, 1])).astype(int), 0, H - 1); y1 = np.clip(np.ceil(np.maximum(np.maximum(A[:, 1], B[:, 1]), C[:, 1])).astype(int), 0, H - 1)
    area = (B[:, 0] - A[:, 0]) * (C[:, 1] - A[:, 1]) - (C[:, 0] - A[:, 0]) * (B[:, 1] - A[:, 1])
    wv = x1 - x0 + 1; hv = y1 - y0 + 1; n = (wv * hv).astype(np.int64)
    n = np.where(np.abs(area) < 1e-9, 0, n)
    img = np.zeros((H * W, 3), np.uint8); zb = np.full(H * W, -9.0, np.float32)
    Hh, Ww = (atlas.shape[:2] if atlas is not None else (1, 1))
    start = 0; tris = np.arange(len(T))
    while start < len(T):                                                                           # chunks of about `chunk` candidate pixels
        cum = np.cumsum(n[start:]); end = start + int(np.searchsorted(cum, chunk, side="right")) if cum[-1] > chunk else len(T)
        end = max(end, start + 1)
        sl = slice(start, end); ns = n[sl]; tot = int(ns.sum())
        if tot:
            tid = np.repeat(tris[sl], ns)
            off = np.repeat(np.cumsum(ns) - ns, ns); local = np.arange(tot) - off
            dx = local % wv[tid]; dy = local // wv[tid]
            xs = x0[tid] + dx + 0.5; ys = y0[tid] + dy + 0.5
            a, b, c, ar = A[tid], B[tid], C[tid], area[tid]
            w0 = ((b[:, 0] - xs) * (c[:, 1] - ys) - (c[:, 0] - xs) * (b[:, 1] - ys)) / ar
            w1 = ((c[:, 0] - xs) * (a[:, 1] - ys) - (a[:, 0] - xs) * (c[:, 1] - ys)) / ar
            w2 = 1 - w0 - w1
            eps = -0.001
            inside = (w0 >= eps) & (w1 >= eps) & (w2 >= eps)
            # a triangle thinner than a cell still marks the cell under its centre (so walls seen edge-on leave a trace)
            thin = (ns[tid - start] <= 2)
            inside |= thin
            tid, w0, w1, w2 = tid[inside], np.clip(w0[inside], 0, 1), np.clip(w1[inside], 0, 1), np.clip(w2[inside], 0, 1)
            sw = np.maximum(w0 + w1 + w2, 1e-9); w0, w1, w2 = w0 / sw, w1 / sw, w2 / sw
            flat = ((y0[tid] + dy[inside]) * W + (x0[tid] + dx[inside]))
            z = w0 * Z[T[tid, 0]] + w1 * Z[T[tid, 1]] + w2 * Z[T[tid, 2]]
            if atlas is not None:
                uv = w0[:, None] * UV[T[tid, 0]] + w1[:, None] * UV[T[tid, 1]] + w2[:, None] * UV[T[tid, 2]]
                col = atlas[np.clip(((1 - uv[:, 1]) * Hh).astype(int), 0, Hh - 1), np.clip((uv[:, 0] * Ww).astype(int), 0, Ww - 1)]
            else:
                g = np.clip(60 + z / max(Z.max(), 0.1) * 150, 0, 255).astype(np.uint8); col = np.stack([g, g, g], 1)
            order = np.argsort(z, kind="stable")                                                         # low to high: the highest point of a cell is written last
            f_o = flat[order]
            better = z[order] >= zb[f_o]                                                                 # against what earlier chunks wrote
            f_o, order = f_o[better], order[better]
            img[f_o] = col[order]; zb[f_o] = z[order]
        start = end
    got = zb > -8
    return img.reshape(H, W, 3), zb.reshape(H, W), got.reshape(H, W)


def top_down(V, T, UV, atlas, R, cut, px_per_m=100.0):
    """The textured mesh seen from straight above (orthographic), in the plan's rotated frame (u right, v up), everything above `cut` metres removed.
    Returns (picture RGB uint8 with row 0 at the top (v max), the picture's (lo, hi) in plan metres, a mask of the cells that got a surface)."""
    cen_z = V[T][:, :, 2].mean(1)
    keep = cen_z < cut
    Tk = T[keep]
    if len(Tk) == 0:
        return None, None, None
    P2 = V[:, :2] @ R.T
    used = np.zeros(len(V), bool); used[Tk.ravel()] = True
    lo = P2[used].min(0) - 0.25; hi = P2[used].max(0) + 0.25
    img, zb, got = raster_top(P2, V[:, 2], Tk, UV, atlas, lo, hi, px_per_m)
    # the walls: the cells where a vertical surface runs from knee height to near the roof (wall_grid), painted as a dark slab (seen edge-on a wall is a hairline otherwise)
    grid, glo, gcell = wall_grid(V, T)
    gx, gy = np.nonzero(grid)
    if len(gx):
        cen = (np.stack([gx + 0.5, gy + 0.5], 1) * gcell + glo) @ R.T                            # cell centres in the plan frame
        W_, H_ = img.shape[1], img.shape[0]
        wall = np.zeros((H_, W_), np.uint8)
        half = int(round(gcell * px_per_m / 2)) + 1
        ix = ((cen[:, 0] - lo[0]) * px_per_m).astype(int); iy = ((hi[1] - cen[:, 1]) * px_per_m).astype(int)
        ok = (ix >= 0) & (ix < W_) & (iy >= 0) & (iy < H_)
        for x, y in zip(ix[ok], iy[ok]):
            wall[max(0, y - half):y + half + 1, max(0, x - half):x + half + 1] = 1
        wall = cv2.morphologyEx(wall, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8)) > 0
        img = np.where(wall[..., None], np.array([58, 56, 52], np.uint8), img); got |= wall
    # the empty cells take the nearest colour, up to 6 cm away; farther stays paper
    k = np.ones((3, 3), np.uint8); fill = img.copy(); fm = got.copy(); zf = zb.copy()
    for _ in range(int(0.06 * px_per_m)):
        dil = cv2.dilate(fill, k); zd = cv2.dilate(zf, k); grown = cv2.dilate(fm.astype(np.uint8), k) > 0
        new = ~fm & grown
        fill = np.where(new[..., None], dil, fill); zf = np.where(new, zd, zf); fm = grown
    img = fill
    # vertical surfaces as ink: a cell whose height jumps against its neighbours by more than 0.45 m is a wall or an edge seen from above
    zf = np.where(fm, zf, 0)
    edge = (cv2.dilate(zf, np.ones((3, 3), np.uint8)) - cv2.erode(zf, np.ones((3, 3), np.uint8))) > 0.45
    edge &= fm
    img = np.where(edge[..., None], (img * 0.45).astype(np.uint8), img)
    # RELIEF: the ground's slope shades the picture as a map does (light from the top left), and a thin contour every half metre where the ground
    # climbs more than a step across the picture — a beach's rock shelf, a yard's bank, a loading dock read as heights, not as colours on a flat sheet
    zs = cv2.GaussianBlur(np.where(fm, zf, 0).astype(np.float32), (0, 0), 1.5)
    gx = cv2.Sobel(zs, cv2.CV_32F, 1, 0, ksize=3) / (8.0 / px_per_m); gy = cv2.Sobel(zs, cv2.CV_32F, 0, 1, ksize=3) / (8.0 / px_per_m)   # metres per metre
    shade = np.clip(1.0 + 0.35 * np.tanh(1.5 * (-gx + gy)), 0.72, 1.18)
    img = np.clip(img.astype(np.float32) * shade[..., None], 0, 255).astype(np.uint8)
    rise = float(np.percentile(zs[fm], 97) - np.percentile(zs[fm], 3)) if fm.any() else 0.0
    if rise > 0.8:
        lv = np.floor(zs / 0.5).astype(np.int32)
        contour = ((cv2.dilate(lv.astype(np.float32), np.ones((3, 3), np.uint8)) - lv) > 0) & fm & ~edge
        img = np.where(contour[..., None], (img * 0.62).astype(np.uint8), img)
    paper = np.array([244, 241, 234], np.uint8)
    img = np.where(fm[..., None], img, paper)
    return img, (lo, hi), fm


def svg(plan, stops, title, pic, scale=45.0):
    """pic: (file name, (lo, hi) in plan metres, width px, height px) or None."""
    lo, hi = np.array(plan["lo"]), np.array(plan["hi"])
    if pic:
        lo = np.minimum(lo, pic[1][0]); hi = np.maximum(hi, pic[1][1])
    W = (hi - lo)[0] * scale + 120; H = (hi - lo)[1] * scale + 140
    X = lambda u: (u - lo[0]) * scale + 60
    Y = lambda v: (hi[1] - v) * scale + 60                                            # v up on the page
    o = []
    o.append(f'<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" viewBox="0 0 {W:.0f} {H:.0f}" width="{W:.0f}" height="{H:.0f}" role="img" aria-label="{title}">')
    o.append(f'<rect width="{W:.0f}" height="{H:.0f}" fill="#f4f1ea"/><g font-family="system-ui, sans-serif" fill="#222">')
    if pic:
        (plo, phi) = pic[1]
        o.append(f'<image href="{pic[0]}" xlink:href="{pic[0]}" x="{X(plo[0]):.1f}" y="{Y(phi[1]):.1f}" width="{(phi[0] - plo[0]) * scale:.1f}" height="{(phi[1] - plo[1]) * scale:.1f}" preserveAspectRatio="none" style="image-rendering:auto"/>')
    o.append(f'<text x="60" y="34" font-size="20" font-weight="700">{title}</text>')
    for w in merge_walls(plan["walls"]):
        if w["axis"] == "u":
            x1, y1, x2, y2 = X(w["pos"]), Y(w["a"]), X(w["pos"]), Y(w["b"])
        else:
            x1, y1, x2, y2 = X(w["a"]), Y(w["pos"]), X(w["b"]), Y(w["pos"])
        major = w["length"] >= 3.0
        o.append(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{"#1a1a1a" if major else "#4a4740"}" stroke-width="{6 if major else 4}" stroke-linecap="round" opacity="0.9"/>')
    out = outer(plan)
    if "u" in out and "v" in out:
        u0, u1 = out["u"]; v0, v1 = out["v"]
        o.append('<g stroke="#2b6cb0" stroke-width="1.5" fill="#2b6cb0" font-size="16">')
        yy = Y(v0) + 34
        o.append(f'<line x1="{X(u0):.1f}" y1="{yy:.1f}" x2="{X(u1):.1f}" y2="{yy:.1f}"/><line x1="{X(u0):.1f}" y1="{yy - 6:.1f}" x2="{X(u0):.1f}" y2="{yy + 6:.1f}"/><line x1="{X(u1):.1f}" y1="{yy - 6:.1f}" x2="{X(u1):.1f}" y2="{yy + 6:.1f}"/>')
        o.append(f'<text x="{(X(u0) + X(u1)) / 2:.1f}" y="{yy + 22:.1f}" text-anchor="middle" stroke="none">{ft_in(u1 - u0)}  ({u1 - u0:.2f} m)</text>')
        xx = X(u0) - 34
        o.append(f'<line x1="{xx:.1f}" y1="{Y(v0):.1f}" x2="{xx:.1f}" y2="{Y(v1):.1f}"/><line x1="{xx - 6:.1f}" y1="{Y(v0):.1f}" x2="{xx + 6:.1f}" y2="{Y(v0):.1f}"/><line x1="{xx - 6:.1f}" y1="{Y(v1):.1f}" x2="{xx + 6:.1f}" y2="{Y(v1):.1f}"/>')
        o.append(f'<text transform="translate({xx - 10:.1f} {(Y(v0) + Y(v1)) / 2:.1f}) rotate(-90)" text-anchor="middle" stroke="none">{ft_in(v1 - v0)}  ({v1 - v0:.2f} m)</text></g>')
    R = np.array(plan["R"])
    for k, (sid, x, y) in enumerate(stops):
        u, v = R @ np.array([x, y])
        o.append(f'<g class="stop" data-stop="{k}" style="cursor:pointer"><circle cx="{X(u):.1f}" cy="{Y(v):.1f}" r="9" fill="#fff" fill-opacity="0.001" stroke="none"/>'
                 f'<circle cx="{X(u):.1f}" cy="{Y(v):.1f}" r="5.5" fill="#fff" stroke="#1a73e8" stroke-width="2.5"/></g>')
    sx, sy = 60, H - 30
    o.append(f'<line x1="{sx}" y1="{sy}" x2="{sx + scale:.0f}" y2="{sy}" stroke="#222" stroke-width="3"/><text x="{sx + scale + 8:.0f}" y="{sy + 5}" font-size="14">1 m ({ft_in(1.0)})</text>')
    o.append('<text x="' + f'{W - 60:.0f}" y="{H - 24:.0f}" text-anchor="end" font-size="12" fill="#777">from the dollhouse model, seen from above with the roof off · sizes approximate</text>')
    o.append("</g></svg>")
    return "\n".join(o)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--frames", default="frames"); ap.add_argument("--out", default="tour"); ap.add_argument("--z", type=float, default=1.0)
    ap.add_argument("--cut", type=float, default=None, help="everything above this height (m) is taken off for the picture; default: 0.55 of the roof, 1.9 to 2.6 m")
    ap.add_argument("--px-per-m", type=float, default=0.0, help="pixels a metre of the picture; 0 = 100 (1 cm a pixel), 150 when the atlas is 8192 px")
    ap.add_argument("--title", default="Floor plan")
    a = ap.parse_args()
    t0 = time.time()
    import open3d as o3d
    uvf = os.path.join(a.frames, "dollhouse", "mesh_uv.npz"); atf = os.path.join(a.frames, "dollhouse", "mesh_atlas.jpg")
    UV = atlas = None
    if os.path.exists(uvf):
        z = np.load(uvf); V, T, UV = z["V"].astype(np.float64), z["T"].astype(np.int64), z["UV"].astype(np.float32)
        if os.path.exists(atf):
            atlas = cv2.cvtColor(cv2.imread(atf), cv2.COLOR_BGR2RGB)
    else:
        m = o3d.io.read_triangle_mesh(os.path.join(a.frames, "dollhouse", "mesh.ply"))
        V, T = np.asarray(m.vertices), np.asarray(m.triangles)
    cells = wall_cells(V, T); theta = main_axis_normals(V, T)
    if not a.px_per_m:
        a.px_per_m = 150.0 if (atlas is not None and atlas.shape[1] >= 8192) else 100.0
    plan = build_plan_from_points(cells, theta=theta)
    out0 = outer(plan)
    if "u" in out0 and "v" in out0 and (out0["u"][1] - out0["u"][0]) < (out0["v"][1] - out0["v"][0]):     # the long side lies along the page
        plan = build_plan_from_points(cells, theta=(theta + 90.0) % 180.0); out0 = outer(plan)
    if not merge_walls(plan["walls"]):                                                  # no walls (outdoors): a map, north up, as a person reads one
        plan = build_plan_from_points(cells, theta=0.0); out0 = outer(plan)
    merged = merge_walls(plan["walls"])
    sup = wall_support(V, T, merged, plan["theta"])
    kept = [w for w, s in zip(merged, sup) if s >= MIN_WALL_SUPPORT]
    plan["walls_dropped"] = [dict(w, support=round(s, 2)) for w, s in zip(merged, sup) if s < MIN_WALL_SUPPORT]
    plan["outer_override"] = out0
    plan["walls"] = kept
    P = json.load(open(os.path.join(a.frames, "poses.json")))["nodes"]
    pts = json.load(open(os.path.join(a.frames, "points.json")))
    pv = json.load(open(os.path.join(a.frames, "privacy.json")))["stops"] if os.path.exists(os.path.join(a.frames, "privacy.json")) else {}
    stops = [(p["id"], n["x"], n["y"]) for p, n in zip(pts, P) if n.get("x") is not None and not (pv.get(p["id"], {}).get("held") and not pv.get(p["id"], {}).get("kept"))]
    # the picture
    roof = float(np.percentile(V[:, 2], 98))
    cut = a.cut if a.cut else float(min(2.6, max(1.9, 0.55 * roof)))
    R = np.array(plan["R"])
    img, frame, got = top_down(V, T, UV, atlas, R, cut, px_per_m=a.px_per_m)
    if img is not None and "u" in out0 and "v" in out0:                               # outside the outer walls is the street, the next bay or noise: paper
        (plo, phi) = frame; H_, W_ = img.shape[:2]; m = 0.35
        u0, u1 = out0["u"]; v0, v1 = out0["v"]
        x0 = int((u0 - m - plo[0]) * a.px_per_m); x1 = int((u1 + m - plo[0]) * a.px_per_m)
        y0 = int((phi[1] - (v1 + m)) * a.px_per_m); y1 = int((phi[1] - (v0 - m)) * a.px_per_m)
        keep = np.zeros((H_, W_), bool); keep[max(0, y0):max(0, y1), max(0, x0):max(0, x1)] = True
        img = np.where(keep[..., None], img, np.array([244, 241, 234], np.uint8))
    pic = None
    os.makedirs(a.out, exist_ok=True)
    if img is not None:
        cv2.imwrite(os.path.join(a.out, "plan.png"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_PNG_COMPRESSION, 6])
        pic = ("plan.png", frame, img.shape[1], img.shape[0])
        plan["picture"] = dict(file="plan.png", lo=[float(x) for x in frame[0]], hi=[float(x) for x in frame[1]], px_per_m=a.px_per_m, cut_m=round(cut, 2), width=img.shape[1], height=img.shape[0])
    open(os.path.join(a.out, "plan.svg"), "w", encoding="utf-8").write(svg(plan, stops, a.title, pic))
    out = outer(plan)
    info = dict(theta_deg=plan["theta"], walls=plan["walls"], walls_dropped=plan.get("walls_dropped", []), outer={k: [float(x) for x in v] for k, v in out.items()},
                outer_size_m=[float(out["u"][1] - out["u"][0]), float(out["v"][1] - out["v"][0])] if len(out) == 2 else None, picture=plan.get("picture"),
                stops=[dict(id=s, x=x, y=y) for s, x, y in stops], R=plan["R"], lo=plan["lo"], hi=plan["hi"])
    json.dump(info, open(os.path.join(a.frames, "dollhouse", "plan.json"), "w"), indent=1)
    tj = os.path.join(a.out, "tour.json")
    if os.path.exists(tj):
        t = json.load(open(tj)); t["plan"] = True; t["axis_deg"] = round(float(plan["theta"]), 1); json.dump(t, open(tj, "w"), separators=(",", ":"))
    dj = os.path.join(a.out, "dollhouse.json")                                       # ONE ORIENTATION: the dollhouse opens and the minimap turns to the plan's axis
    if os.path.exists(dj):
        try:
            dm = json.load(open(dj)); dm["axis_deg"] = round(float(plan["theta"]), 1); json.dump(dm, open(dj, "w"))
        except (OSError, ValueError):
            pass
    sz = info["outer_size_m"]
    print(f"plan: {len(plan['walls'])} wall lines, main axis {plan['theta']:.1f} deg, cut at {cut:.2f} m, picture {img.shape[1] if img is not None else 0} x {img.shape[0] if img is not None else 0} px"
          + (f", outer {sz[0]:.2f} x {sz[1]:.2f} m ({ft_in(sz[0])} x {ft_in(sz[1])})" if sz else "") + f", {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
