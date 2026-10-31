"""Texture the dollhouse through a real UV atlas (xatlas, MIT): one 4096 x 4096 picture of the whole model, every texel coloured from the panorama that sees
that spot best, so the model reads like the photographs instead of a mosaic of 10-pixel blocks.

Used by dollhouse.py. `python dollhouse_uv.py --frames frames [--mesh mesh] [--size 4096]` bakes a mesh on its own (experiments).

How: xatlas cuts the mesh into charts and packs them (the mesh's vertices are split along chart borders, so the output has more vertices than the input).
For every texel inside a chart: its 3D point and the face normal; the (up to) 4 stops that face it best (score = cos(angle between the normal and the line to the
stop) / (distance + 0.5)); a stop counts only if nothing of the model stands between it and the point (a ray cast) and the point is not where the person holding
the stick was removed; the colours are blended with weights score^6, so the best view dominates and a change of best view is not a hard edge. Texels no stop
sees in the open take the best-facing stop's colour anyway. The colours are then pushed 6 texels out into the gutters, so mipmaps and bilinear filtering never
pull in black.
"""
import argparse, json, math, os, sys, time
import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from imutil import imread       # noqa: E402
import dollhouse as D           # noqa: E402

TOPK = 4
POWER = 6.0
NEAR_BEST = 0.9                 # a stop colours a texel only when its score is within this share of the best stop that sees it: one picture wins, two blend only when they tie
                                # (blending four views whose geometry disagrees by a few centimetres painted every spool and tool case twice on the floor)
VIS_TOL0, VIS_TOL1 = 0.06, 0.02
GUTTER = 6
GAIN_LO, GAIN_HI = 0.7, 1.4     # one stop's exposure is never moved more than this (a bad overlap must not turn a stop's picture into a different place)
GAIN_SAMPLES = 20000            # texels per pair of stops the exposure is measured on
FEATHER = 3                     # texels either side of a boundary between two stops' pictures that blend (a seam becomes a soft edge about 6 texels wide)


def box_unwrap(V, T, size=4096, pad=2, log=print):
    """A fast atlas for a scanned mesh (xatlas took over five minutes on 120k triangles of marching-cubes slivers): every triangle is projected along the axis
    its normal is nearest to (6 directions), triangles of one direction that share an edge form a chart, each chart is a flat rectangle, and the rectangles are
    shelf-packed into the square. Vertices are split per chart. Returns V2, T2, UV2 (0..1, v up), (size, size)."""
    t0 = time.time()
    nrm = np.cross(V[T[:, 1]] - V[T[:, 0]], V[T[:, 2]] - V[T[:, 0]])
    area = 0.5 * np.linalg.norm(nrm, axis=1)
    nrm = nrm / np.maximum(np.linalg.norm(nrm, axis=1, keepdims=True), 1e-12)
    # triangles that share an edge
    e = np.concatenate([T[:, [0, 1]], T[:, [1, 2]], T[:, [2, 0]]]); e.sort(axis=1)
    tri_of = np.concatenate([np.arange(len(T))] * 3)
    key = e[:, 0].astype(np.int64) * (V.shape[0] + 1) + e[:, 1]
    order = np.argsort(key, kind="stable"); key, tri_of = key[order], tri_of[order]
    same = key[1:] == key[:-1]
    A_, B_ = tri_of[:-1][same], tri_of[1:][same]
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    adj = coo_matrix((np.ones(len(A_) * 2), (np.r_[A_, B_], np.r_[B_, A_])), shape=(len(T), len(T))).tocsr()
    # the direction each triangle is projected along: its normal smoothed over its neighbours (a bumpy scanned wall must not split into thousands of charts)
    sm = nrm.copy()
    for _ in range(3):
        sm = sm + adj @ sm; sm /= np.maximum(np.linalg.norm(sm, axis=1, keepdims=True), 1e-12)
    axis = np.argmax(np.abs(sm), axis=1); sign = np.sign(sm[np.arange(len(T)), axis]); sign[sign == 0] = 1
    dirn = axis * 2 + (sign > 0)                                              # 0..5

    def components(dirn):
        ok = dirn[A_] == dirn[B_]
        g = coo_matrix((np.ones(ok.sum()), (A_[ok], B_[ok])), shape=(len(T), len(T)))
        return connected_components(g, directed=False)
    n_ch, chart = components(dirn)
    for _ in range(5):                                                        # a small chart takes the direction of its biggest neighbour and joins it
        size_ch = np.bincount(chart, minlength=n_ch)
        small = size_ch[chart] < 60
        if not small.any():
            break
        # for every edge between a small-chart triangle and another chart: candidate (triangle, neighbour's chart size, neighbour's direction)
        pa, pb = np.r_[A_, B_], np.r_[B_, A_]
        m = small[pa] & (chart[pa] != chart[pb])
        if not m.any():
            break
        pa, pb = pa[m], pb[m]
        best = np.full(len(T), -1); best_size = np.zeros(len(T), int)
        ordr = np.argsort(size_ch[chart[pb]])                                 # biggest last wins
        best[pa[ordr]] = dirn[pb[ordr]]; best_size[pa[ordr]] = size_ch[chart[pb[ordr]]]
        change = (best >= 0) & (best_size > size_ch[chart])
        if not change.any():
            break
        dirn = dirn.copy(); dirn[change] = best[change]
        axis = dirn // 2; sign = np.where(dirn % 2 == 1, 1, -1)
        n_ch, chart = components(dirn)
    # each chart's 2D frame: the two axes other than its projection axis (u, v chosen so the chart is not mirrored)
    U_AX = np.array([[1, 2], [0, 2], [0, 1]]); FLIP = np.array([[0, 1], [1, 0], [0, 1]])
    uv_raw = np.zeros((len(T), 3, 2), np.float32)
    for ax in range(3):
        m = axis == ax
        P = V[T[m]]                                                           # (n, 3, 3)
        u, v = P[..., U_AX[ax, 0]], P[..., U_AX[ax, 1]]
        sg = sign[m][:, None]
        uv_raw[m, :, 0] = np.where(sg > 0, u, -u) if FLIP[ax, 0] == 0 else np.where(sg > 0, -u, u)
        uv_raw[m, :, 1] = v
    # chart boxes
    cu = np.zeros((n_ch, 2)); cv = np.zeros((n_ch, 2))
    np.minimum.at(cu[:, 0], chart, uv_raw[:, :, 0].min(1)); np.maximum.at(cu[:, 1], chart, uv_raw[:, :, 0].max(1))
    cu[:, 0] = np.inf; cu[:, 1] = -np.inf; cv[:, 0] = np.inf; cv[:, 1] = -np.inf
    np.minimum.at(cu[:, 0], chart, uv_raw[:, :, 0].min(1)); np.maximum.at(cu[:, 1], chart, uv_raw[:, :, 0].max(1))
    np.minimum.at(cv[:, 0], chart, uv_raw[:, :, 1].min(1)); np.maximum.at(cv[:, 1], chart, uv_raw[:, :, 1].max(1))
    wm = cu[:, 1] - cu[:, 0]; hm = cv[:, 1] - cv[:, 0]
    ch_area = np.zeros(n_ch); np.add.at(ch_area, chart, area)
    # texels per metre: the boxes fill about 65 % of the square (boxes hold air round a chart, so this is iterated down if they do not fit)
    def pack(tpm):
        w_px = np.ceil(wm * tpm).astype(int) + 2 * pad + 1; h_px = np.ceil(hm * tpm).astype(int) + 2 * pad + 1
        ordr = np.argsort(-h_px)                                               # shelf packing, tallest first
        x = y = shelf_h = 0; place = np.zeros((n_ch, 2), int)
        for c in ordr:
            if x + w_px[c] > size:
                x = 0; y += shelf_h; shelf_h = 0
            if y + h_px[c] > size or w_px[c] > size:
                return None, y + shelf_h
            place[c] = (x, y); x += w_px[c]; shelf_h = max(shelf_h, h_px[c])
        return place, y + shelf_h
    tpm = math.sqrt(0.65 * size * size / max(1e-6, float(((wm + 0.0) * (hm + 0.0)).sum())))
    place, used = pack(tpm)
    while place is None:                                                      # too dense: back off
        tpm *= 0.93; place, used = pack(tpm)
    for _ in range(16):                                                       # then as dense as still fits
        p2, u2 = pack(tpm * 1.05)
        if p2 is None:
            break
        tpm *= 1.05; place, used = p2, u2
    y, shelf_h = used, 0
    # per-triangle-corner uvs (vertices split per chart: every triangle corner becomes its own vertex, then merged within a chart)
    px = place[chart][:, None, :] + pad + 0.5 + (uv_raw - np.stack([cu[chart, 0], cv[chart, 0]], 1)[:, None, :]) * tpm
    UVc = (px / size).astype(np.float32)                                      # (n, 3, 2), v up
    corner_key = (chart.astype(np.int64)[:, None] * (V.shape[0] + 1) + T).ravel()
    uniq, inv = np.unique(corner_key, return_inverse=True)
    V2 = V[(uniq % (V.shape[0] + 1)).astype(np.int64)].astype(np.float32)
    T2 = inv.reshape(-1, 3).astype(np.int64)
    UV2 = np.zeros((len(uniq), 2), np.float32); UV2[inv] = UVc.reshape(-1, 2)
    log(f"  atlas: {len(T)} triangles -> {n_ch} charts, {len(V2)} vertices, {tpm:.0f} texels a metre ({1000 / tpm:.1f} mm a texel), packed to row {y + shelf_h} of {size} ({time.time() - t0:.1f} s)")
    return V2, T2, UV2, (size, size)


def unwrap(V, T, log=print):
    """xatlas: (vertex map into the old vertices, new triangle indices, uvs 0..1). Returns V2, T2, UV2. Slow on a scan (minutes); box_unwrap is the default."""
    import xatlas
    t0 = time.time()
    atlas = xatlas.Atlas()
    atlas.add_mesh(V.astype(np.float32), T.astype(np.uint32))
    co = xatlas.ChartOptions(); co.max_iterations = 2
    po = xatlas.PackOptions(); po.padding = 3; po.bilinear = True; po.blockAlign = True; po.resolution = 0; po.texels_per_unit = 0
    atlas.generate(chart_options=co, pack_options=po)
    vmap, idx, uv = atlas[0]
    log(f"  xatlas: {len(T)} triangles -> {atlas.chart_count} charts, {len(vmap)} vertices, {atlas.width} x {atlas.height} px at xatlas' scale ({time.time() - t0:.0f} s)")
    return V[vmap], idx.astype(np.int64), uv.astype(np.float32), (atlas.width, atlas.height)


def texel_points(V2, T2, UV2, size):
    """Every texel covered by a triangle: (texel index flat, 3D point, triangle index). Rasterised per triangle in texture space."""
    S = size
    px = UV2 * S                                                           # texel coordinates (u right, v up in xatlas' space)
    tex_i, P, tri = [], [], []
    A, B, C = px[T2[:, 0]], px[T2[:, 1]], px[T2[:, 2]]
    lo = np.floor(np.minimum(np.minimum(A, B), C)).astype(int); hi = np.ceil(np.maximum(np.maximum(A, B), C)).astype(int)
    area = (B[:, 0] - A[:, 0]) * (C[:, 1] - A[:, 1]) - (C[:, 0] - A[:, 0]) * (B[:, 1] - A[:, 1])
    for t in range(len(T2)):
        if abs(area[t]) < 1e-9:
            continue
        x0, y0 = max(lo[t, 0], 0), max(lo[t, 1], 0); x1, y1 = min(hi[t, 0], S - 1), min(hi[t, 1], S - 1)
        if x1 < x0 or y1 < y0:
            continue
        xs, ys = np.meshgrid(np.arange(x0, x1 + 1) + 0.5, np.arange(y0, y1 + 1) + 0.5)
        a, b, c = A[t], B[t], C[t]
        w0 = ((b[0] - xs) * (c[1] - ys) - (c[0] - xs) * (b[1] - ys)) / area[t]
        w1 = ((c[0] - xs) * (a[1] - ys) - (a[0] - xs) * (c[1] - ys)) / area[t]
        w2 = 1 - w0 - w1
        eps = -0.002
        inside = (w0 >= eps) & (w1 >= eps) & (w2 >= eps)
        if not inside.any():
            # a sliver smaller than a texel: give it the texel under its centre
            cx, cy = int((a[0] + b[0] + c[0]) / 3), int((a[1] + b[1] + c[1]) / 3)
            if 0 <= cx < S and 0 <= cy < S:
                tex_i.append(np.array([cy * S + cx])); P.append(V2[T2[t]].mean(0)[None]); tri.append(np.array([t]))
            continue
        xi = (xs[inside] - 0.5).astype(int); yi = (ys[inside] - 0.5).astype(int)
        w0, w1, w2 = (np.clip(w, 0, 1) for w in (w0[inside], w1[inside], w2[inside]))
        sw = w0 + w1 + w2
        pts = (w0[:, None] * V2[T2[t, 0]] + w1[:, None] * V2[T2[t, 1]] + w2[:, None] * V2[T2[t, 2]]) / sw[:, None]
        tex_i.append(yi * S + xi); P.append(pts); tri.append(np.full(len(xi), t))
    return np.concatenate(tex_i), np.concatenate(P).astype(np.float32), np.concatenate(tri)


def assign_stops(T2, tri, ti, ts, vis, S, log=print, rounds=6):
    """One stop per triangle. Each triangle's vote for a stop is the sum of its texels' visible scores; then, round by round, a triangle moves to the
    stop its edge-neighbours use most when that stop sees at least a third of what the best one does. Returns an int array (triangle -> stop)."""
    nT = len(T2)
    rows = np.repeat(tri, ti.shape[1]); cols = ti.ravel(); w = np.where(vis, ts, 0.0).ravel()
    keep = w > 0
    votes = np.zeros((nT, S), np.float32)
    np.add.at(votes, (rows[keep], cols[keep]), w[keep])
    pick = votes.argmax(1)
    best = votes.max(1)
    # edge neighbours: triangles sharing an edge
    E = np.concatenate([T2[:, [0, 1]], T2[:, [1, 2]], T2[:, [2, 0]]]); E.sort(1)
    owner = np.tile(np.arange(nT), 3)
    key = E[:, 0].astype(np.int64) * (T2.max() + 1) + E[:, 1]
    order = np.argsort(key, kind="stable"); key, owner = key[order], owner[order]
    same = key[1:] == key[:-1]
    a, b = owner[:-1][same], owner[1:][same]
    nb_a = np.concatenate([a, b]); nb_b = np.concatenate([b, a])
    changed_total = 0
    for _ in range(rounds):
        nbv = np.zeros((nT, S), np.float32)
        np.add.at(nbv, (nb_a, pick[nb_b]), 1.0)
        cand = nbv.argmax(1)
        ok = (nbv.max(1) >= 2) & (votes[np.arange(nT), cand] >= 0.33 * best) & (cand != pick)
        if not ok.any():
            break
        pick = np.where(ok, cand, pick); changed_total += int(ok.sum())
    log(f"  one picture per triangle: {len(np.unique(pick))} stops used, {changed_total} triangles joined their neighbours' stop")
    return pick


def pair_rows(ti, vis, S, cap=GAIN_SAMPLES, seed=0):
    """The texels two stops both see in the open, per pair of stops: {(a, b): texel rows}, a < b, at most `cap` rows each (a random sample)."""
    rng = np.random.default_rng(seed)
    K = ti.shape[1]
    ra, rb, rr = [], [], []
    for c1 in range(K):
        for c2 in range(c1 + 1, K):
            m = vis[:, c1] & vis[:, c2] & (ti[:, c1] != ti[:, c2])
            r = np.nonzero(m)[0]
            if len(r):
                a, b = ti[r, c1], ti[r, c2]
                ra.append(np.minimum(a, b)); rb.append(np.maximum(a, b)); rr.append(r)
    if not rr:
        return {}
    ra, rb, rr = np.concatenate(ra), np.concatenate(rb), np.concatenate(rr)
    key = ra.astype(np.int64) * S + rb
    order = np.argsort(key, kind="stable"); key, rr = key[order], rr[order]
    cut = np.nonzero(np.diff(key))[0] + 1
    out = {}
    for k_, rows in zip(key[np.r_[0, cut]], np.split(rr, cut)):
        rows = np.unique(rows)
        if len(rows) > cap:
            rows = np.sort(rng.choice(rows, cap, replace=False))
        out[(int(k_ // S), int(k_ % S))] = rows
    return out


def exposure_gains(S, pairs, sampler, log=print, min_samples=200):
    """One gain per stop per channel (S, 3) that makes the stops agree on what they both see. `pairs` = {(a, b): texel rows} (pair_rows); `sampler(stop,
    rows)` -> (colours (n, 3) as that stop's picture gives them, ok (n,) bool). For each pair the mean colour each stop gives the SAME texels is compared;
    the log-gains are solved by least squares (weight = sqrt(samples)), the median stop is fixed at 1 and every gain is held to [GAIN_LO, GAIN_HI]."""
    need = {}
    for (a, b), rows in pairs.items():
        need.setdefault(a, []).append(((a, b), rows)); need.setdefault(b, []).append(((a, b), rows))
    got = {}
    for st, lst in need.items():
        col, ok = sampler(st, np.concatenate([r for _, r in lst]))
        o = 0
        for pr, r in lst:
            got[(st, pr)] = (col[o:o + len(r)], ok[o:o + len(r)]); o += len(r)
    eq, rhs, wt = [], [], []
    for (a, b), rows in pairs.items():
        ca, oa = got[(a, (a, b))]; cb, ob = got[(b, (a, b))]
        ok = oa & ob
        if ok.sum() < min_samples:
            continue
        ma, mb = ca[ok].mean(0), cb[ok].mean(0)
        row = np.zeros(S); row[a], row[b] = 1.0, -1.0
        eq.append(row); rhs.append(np.log(np.maximum(mb, 1.0) / np.maximum(ma, 1.0))); wt.append(math.sqrt(float(ok.sum())))
    gains = np.ones((S, 3), np.float32)
    if not eq:
        log("  exposure: no two stops share enough in the open to compare; every gain stays 1")
        return gains
    A = np.array(eq) * np.array(wt)[:, None]; B = np.array(rhs) * np.array(wt)[:, None]
    ridge = 1e-3 * max(wt)                                                     # a stop no pair reaches stays at 1 (and a disconnected group is centred)
    A = np.vstack([A, ridge * np.eye(S)]); B = np.vstack([B, np.zeros((S, 3))])
    lg = np.linalg.lstsq(A, B, rcond=None)[0]                                  # (S, 3) log gains
    lg -= np.median(lg, axis=0, keepdims=True)                                 # the median stop is 1
    gains = np.clip(np.exp(lg), GAIN_LO, GAIN_HI).astype(np.float32)
    log(f"  exposure: {len(eq)} stop pairs compared, gains {gains.min():.2f} to {gains.max():.2f} (R {gains[:, 0].min():.2f}-{gains[:, 0].max():.2f}, "
        f"G {gains[:, 1].min():.2f}-{gains[:, 1].max():.2f}, B {gains[:, 2].min():.2f}-{gains[:, 2].max():.2f})")
    return gains


def feather_plan(pick_img, ti_flat, eligible, width=FEATHER):
    """Where two stops' pictures meet in texture space. `pick_img` (S, S) int: the stop each texel takes (-1 empty). For each eligible texel (rows of `ti_flat`)
    within `width` texels of a texel that takes a DIFFERENT stop: (rows, other stop, weight of the other stop). The weight is 0.5 at the boundary and falls
    to 0 over `width` texels, so a seam is a soft edge 2 * width texels wide; the nearest different stop is the only other one, so no texel blends more than two."""
    H, W = pick_img.shape
    other = np.full((H, W), -1, np.int16); dist = np.zeros((H, W), np.int8)
    pad = np.pad(pick_img.astype(np.int16), width, constant_values=-1)
    for d in range(1, width + 1):
        for dy in range(-d, d + 1):
            for dx in range(-d, d + 1):
                if max(abs(dy), abs(dx)) != d:
                    continue
                nb = pad[width + dy:width + dy + H, width + dx:width + dx + W]
                hit = (other < 0) & (pick_img >= 0) & (nb >= 0) & (nb != pick_img)
                other[hit] = nb[hit]; dist[hit] = d
    ot, dt = other.ravel()[ti_flat], dist.ravel()[ti_flat]
    sel = np.nonzero((ot >= 0) & eligible)[0]
    w = 0.5 * (width + 1 - dt[sel]) / width
    return sel, ot[sel].astype(np.int64), w.astype(np.float32)


def bake(frames, V2, T2, UV2, stops, size=4096, log=print, flat_z=None):
    """The atlas (size x size RGB uint8, row 0 at the TOP of the image, i.e. v flipped for the file) and a report."""
    import torch, open3d as o3d
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    t0 = time.time()
    ti_flat, P, tri = texel_points(V2, T2, UV2, size)
    n = len(P)
    nrm_t = np.cross(V2[T2[:, 1]] - V2[T2[:, 0]], V2[T2[:, 2]] - V2[T2[:, 0]]); nrm_t /= np.maximum(np.linalg.norm(nrm_t, axis=1, keepdims=True), 1e-12)
    nrm = nrm_t[tri].astype(np.float32)
    log(f"  {n / 1e6:.1f} M texels to colour ({100 * n / size / size:.0f}% of the atlas) ({time.time() - t0:.0f} s)")
    Pf = torch.from_numpy(P).to(dev); nf = torch.from_numpy(nrm).to(dev)
    S = len(stops)
    pos = torch.tensor(np.array([s["pos"] for s in stops]), dtype=torch.float32, device=dev)
    K = min(TOPK, S)
    top_idx = torch.empty((n, K), dtype=torch.long, device=dev); top_sc = torch.empty((n, K), device=dev)
    for a in range(0, n, 1_000_000):
        sl = slice(a, min(n, a + 1_000_000))
        d = pos[None, :, :] - Pf[sl, None, :]
        dist = d.norm(dim=2).clamp(min=0.05)
        cos = (d * nf[sl, None, :]).sum(2).abs() / dist
        sc = torch.where((cos > 0.03) & (dist < 30.0), cos / (dist + 0.5), torch.zeros_like(cos))
        v_, i_ = torch.topk(sc, K, dim=1)
        top_idx[sl], top_sc[sl] = i_, v_
    scene = o3d.t.geometry.RaycastingScene()
    mt = o3d.t.geometry.TriangleMesh(); mt.vertex.positions = o3d.core.Tensor(V2.astype(np.float32)); mt.triangle.indices = o3d.core.Tensor(T2.astype(np.int32))
    scene.add_triangles(mt)
    ti, ts = top_idx.cpu().numpy(), top_sc.cpu().numpy()
    posn = pos.cpu().numpy()
    vis = np.zeros(ti.shape, bool)
    for c in range(K):
        o = posn[ti[:, c]]; d = P - o; dist = np.linalg.norm(d, axis=1)
        for a in range(0, n, 4_000_000):
            sl = slice(a, min(n, a + 4_000_000))
            rays = o3d.core.Tensor(np.concatenate([o[sl], d[sl] / np.maximum(dist[sl], 1e-6)[:, None]], 1).astype(np.float32))
            th = scene.cast_rays(rays)["t_hit"].numpy()
            vis[sl, c] = (ts[sl, c] > 0) & (th >= dist[sl] - (VIS_TOL0 + VIS_TOL1 * dist[sl]))
    log(f"  visibility: {vis.mean() * 100:.0f}% of the (texel, stop) candidates are in the open ({time.time() - t0:.0f} s)")
    vis_any = vis.copy()
    # ONE PICTURE PER TRIANGLE: every triangle takes the stop its texels see best, and that choice is smoothed over the mesh (a triangle joins the stop
    # most of its neighbours chose when that stop sees it too), so a surface is one photograph and never two laid a few centimetres apart - blending
    # four views whose depth disagrees by that much painted every spool and tool case on the floor twice; the seams between stops are sharp instead
    tri_pick = assign_stops(T2, tri, ti, ts, vis, S, log)
    chosen = (ti == tri_pick[tri][:, None])
    fallback = ~(chosen & vis).any(1)                                                           # the triangle's stop does not see this texel in the open: the best that does
    if flat_z is not None:
        # WATER BLENDS: the sea moves between one stop's picture and the next, so a seam between two pictures of it is a line across the water; it has
        # no detail a blend would blur - every stop that sees a water texel colours it
        cz = V2[T2].mean(1)[:, 2]
        water_tri = np.abs(cz - flat_z) < 0.15
        fallback |= water_tri[tri]
        log(f"  water: {int(water_tri.sum())} triangles on the sea level blend every stop")
    best_vis = np.where(vis, ts, 0.0).max(1, keepdims=True)
    vis &= np.where(fallback[:, None], ts >= NEAR_BEST * best_vis, chosen)
    if flat_z is not None:
        vis |= (water_tri[tri][:, None] & (ts > 0) & vis_any)                                     # water: back to every stop in the open
    def load_pano(s):
        im = imread(s["file"]); pw = 8192 if (size >= 8192 and im.shape[1] >= 8192) else 4096          # the texels are 5 mm on a big atlas: sample the full picture
        if im.shape[1] != pw:
            im = cv2.resize(im, (pw, pw // 2), interpolation=cv2.INTER_AREA)
        return torch.from_numpy(cv2.cvtColor(im, cv2.COLOR_BGR2RGB)).to(dev).permute(2, 0, 1)[None].half()

    def sample(pano, si, s, r):
        """The colours stop `si`'s picture gives the texels `r` (as the picture is, no gain) and whether each is outside the hole where the capturer stood."""
        Wm = torch.tensor(s["W"], dtype=torch.float32, device=dev)
        hole = s.get("hole")
        pts = Pf[r]; dvec = (pts - pos[si]) @ Wm
        dn = dvec.norm(dim=1, keepdim=True).clamp(min=1e-6)
        az = torch.atan2(dvec[:, 0], dvec[:, 2]); el = torch.asin((dvec[:, 1] / dn[:, 0]).clamp(-1, 1))
        grid = torch.stack([az / math.pi, -el / (math.pi / 2)], 1)[None, :, None, :]
        col = torch.nn.functional.grid_sample(pano, grid.half(), mode="bilinear", padding_mode="border", align_corners=False)[0, :, :, 0].T.float().cpu().numpy()
        ok = np.ones(len(r), bool)
        if hole is not None:
            ok = ~(torch.nn.functional.grid_sample(hole, grid, mode="nearest", padding_mode="border", align_corners=False)[0, 0, :, 0].cpu().numpy() > 0.5)
        return col, ok

    # EXPOSURE: every stop's picture is exposed a little differently (auto exposure follows what the camera faced), and one picture per triangle turns that
    # into a step at every seam. One gain per stop per channel is solved from the texels two stops both see in the open, so the steps are gone before the
    # seams are even blended
    def gain_sampler(si, rows):
        pano = load_pano(stops[si])
        out_c = np.zeros((len(rows), 3), np.float32); out_ok = np.zeros(len(rows), bool)
        for a in range(0, len(rows), 1_500_000):
            out_c[a:a + 1_500_000], out_ok[a:a + 1_500_000] = sample(pano, si, stops[si], rows[a:a + 1_500_000])
        del pano
        return out_c, out_ok
    gains = exposure_gains(S, pair_rows(ti, vis_any, S), gain_sampler, log)
    meta_gains = [float(gains.min()), float(gains.max())]
    # SEAMS: where two stops' pictures meet, the texels within FEATHER of the boundary take both (the nearest other stop only): a soft edge, not a line
    pick_img = np.full(size * size, -1, np.int16)
    pick_img[ti_flat] = tri_pick[tri]
    elig = ~fallback
    f_rows, f_other, f_w = feather_plan(pick_img.reshape(size, size), ti_flat, elig)
    Wt = (ts ** POWER * vis).astype(np.float32)
    if len(f_rows):
        have_o = (ti[f_rows] == f_other[:, None]) & vis_any[f_rows]
        cp = (chosen[f_rows] & vis[f_rows]).argmax(1)
        good = have_o.any(1) & (chosen[f_rows] & vis[f_rows]).any(1)
        f_rows, f_w, cp, co = f_rows[good], f_w[good], cp[good], have_o[good].argmax(1)
        Wt[f_rows] = 0.0
        Wt[f_rows, cp] = 1.0 - f_w; Wt[f_rows, co] = f_w
        log(f"  seams: {len(f_rows)} texels ({100 * len(f_rows) / n:.1f}%) within {FEATHER} of a boundary between two stops blend them")
    acc = np.zeros((n, 3), np.float32); wsum = np.zeros(n, np.float32)
    fall = np.zeros((n, 3), np.float32); fsc = np.zeros(n, np.float32)
    sel = np.ones(n, bool)
    for pass_ in range(2):
      if pass_ == 1:
        # SECOND PASS for what the first left unseen: a texel inside its chosen stop's capturer hole (where he stood) got no colour from the one stop it
        # was allowed - the floor under every stop was pruned as "seen by nobody"; those texels go back to every stop in the open
        sel = (wsum <= 1e-12) & vis_any.any(1)
        if not sel.any():
            break
        vis = np.where(sel[:, None], vis_any, vis)
        Wt = np.where(sel[:, None], (ts ** POWER * vis).astype(np.float32), 0.0).astype(np.float32)
        log(f"  second pass: {int(sel.sum())} texels unseen by their own stop take any stop in the open")
      for si, s in enumerate(stops):
        m = (ti == si) & sel[:, None]
        if not m.any():
            continue
        rows_, cols_ = np.nonzero(m)
        pano = load_pano(s)
        for a in range(0, len(rows_), 1_500_000):
            r = rows_[a:a + 1_500_000]; c = cols_[a:a + 1_500_000]
            col, ok = sample(pano, si, s, r)
            col = np.clip(col * gains[si], 0, 255)
            w = np.where(ok, Wt[r, c], 0.0)
            np.add.at(acc, r, col * w[:, None]); np.add.at(wsum, r, w)
            better = ts[r, c] > fsc[r]
            fall[r[better]] = col[better]; fsc[r[better]] = ts[r, c][better]
        del pano
    have = wsum > 1e-12
    out = np.where(have[:, None], acc / np.maximum(wsum, 1e-12)[:, None], fall)
    log(f"  colours: {have.mean() * 100:.1f}% of the texels seen in the open, the rest from the best-facing stop ({time.time() - t0:.0f} s)")
    seen_tri = np.zeros(len(T2), np.float32); cnt_tri = np.zeros(len(T2), np.float32)
    np.add.at(seen_tri, tri, have.astype(np.float32)); np.add.at(cnt_tri, tri, 1.0)
    seen_tri = seen_tri / np.maximum(cnt_tri, 1)
    atlas = np.zeros((size * size, 3), np.uint8); filled = np.zeros(size * size, bool)
    atlas[ti_flat] = np.clip(out, 0, 255).astype(np.uint8); filled[ti_flat] = True
    atlas = atlas.reshape(size, size, 3); filled = filled.reshape(size, size)
    # gutters: push the chart colours outward a few texels (a nearest-filled dilation), so filtering at a chart border never reads black
    img = atlas.copy(); fm = filled.copy()
    for _ in range(GUTTER):
        dil = cv2.dilate(img, np.ones((3, 3), np.uint8))
        grown = cv2.dilate(fm.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
        img = np.where((~fm & grown)[..., None], dil, img); fm = grown
    atlas = img[::-1]                                                      # xatlas' v is up; the image file's row 0 is the top
    return atlas, dict(size=size, charts_fill_pct=round(float(filled.mean() * 100), 1), seen_pct=round(float(have.mean() * 100), 1), stops=len(stops),
                       gain_range=[round(g, 3) for g in meta_gains], seam_texels=int(len(f_rows))), seen_tri


def texture_mesh(frames, mesh, stops, size=4096, log=print, flat_z=None):
    """Unwrap, bake, then drop the triangles no stop saw in the open (a hidden back side, a floater inside a wall: they could only carry made-up colours).
    Returns (V2, T2, UV2, atlas RGB, meta)."""
    V = np.asarray(mesh.vertices, np.float32); T = np.asarray(mesh.triangles, np.int64)
    V2, T2, UV2, _ = (unwrap(V, T, log=log) if os.environ.get("SHOP_TOUR_XATLAS") == "1" else box_unwrap(V, T, size=size, log=log))
    atlas, meta, seen_tri = bake(frames, V2, T2, UV2, stops, size=size, log=log, flat_z=flat_z)
    keep = seen_tri > 0
    log(f"  {int((~keep).sum())} of {len(T2)} triangles are seen by no stop (dropped)")
    T2 = T2[keep]
    used = np.zeros(len(V2), bool); used[T2.ravel()] = True                              # the vertices no triangle uses any more go too
    remap = np.cumsum(used) - 1
    V2, UV2, T2 = V2[used], UV2[used], remap[T2]
    meta["pruned_triangles"] = int((~keep).sum())
    return V2, T2, UV2, atlas, meta


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--frames", default="frames"); ap.add_argument("--mesh", default="mesh"); ap.add_argument("--size", type=int, default=4096)
    a = ap.parse_args()
    import open3d as o3d, dollhouse_texture as DT
    m = o3d.io.read_triangle_mesh(os.path.join(a.frames, "dollhouse", a.mesh + ".ply"))
    V2, T2, UV2, atlas, meta = texture_mesh(a.frames, m, DT.stops_for(a.frames), size=a.size)
    out = os.path.join(a.frames, "dollhouse", a.mesh + "_uv")
    np.savez_compressed(out + ".npz", V=V2, T=T2.astype(np.int32), UV=UV2)
    cv2.imwrite(out + ".jpg", cv2.cvtColor(atlas, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 90])
    print(meta, f"{os.path.getsize(out + '.jpg') / 1e6:.2f} MB")


if __name__ == "__main__":
    main()
