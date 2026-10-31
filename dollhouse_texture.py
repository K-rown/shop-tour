"""Colour the dollhouse from the real pictures: every little patch of the model takes its colour from the panorama that sees it best.

Used by dollhouse.py (after the mesh is cleaned); `python dollhouse_texture.py --frames frames --mesh mesh_mvs` bakes an experiment mesh.

Layout (no UV unwrapping, which is slow and falls apart on a noisy scanned mesh): every triangle owns a k x k square of the picture (a "block"), triangle t at
column t % per_row, row t // per_row. The triangle's three corners sit on the centres of three corner texels of its block, so a bilinear lookup never reads a
neighbour's block; the other half of the square is the mirror image over the long edge (that keeps the colour continuous across it). The viewer needs no UV
data: it derives them from the triangle number.

Colour of a texel: the 3D point of the texel is looked up in the (up to) 4 stops that face it best (score = cos(angle between the surface normal and the line to the
stop) / (distance + 0.5)); a stop counts only if nothing of the model is in the way (a ray from the stop to the point) and the point is not inside the area where the
person holding the camera was removed; the colours are blended with weights score^6, so the best view dominates and a change of best view is not a hard edge.
Held-back stops (privacy) are never read: only frames/safe/ is used.
"""
import argparse, json, math, os, sys, time
import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from imutil import imread       # noqa: E402
import dollhouse as D           # noqa: E402

K = 10
ATLAS_W = 2040                  # a multiple of K
TOPK = 4
POWER = 6.0
VIS_TOL0, VIS_TOL1 = 0.06, 0.02        # a texel counts as seen when nothing of the model is more than this much in front of it (a bumpy scanned surface needs some room)


def layout(n_tri, k=K, width=ATLAS_W):
    per_row = width // k
    rows = -(-n_tri // per_row)
    return per_row, rows * k


def block_points(V, T, k, per_row):
    """3D position and face normal of every texel: (n_tri, k, k, 3), (n_tri, 3)."""
    A, B, C = V[T[:, 0]], V[T[:, 1]], V[T[:, 2]]
    nrm = np.cross(B - A, C - A); nrm /= np.maximum(np.linalg.norm(nrm, axis=1, keepdims=True), 1e-12)
    i = np.arange(k); u, v = np.meshgrid(i / (k - 1.0), i / (k - 1.0), indexing="xy")        # u along x (columns), v along y (rows)
    over = u + v > 1.0 + 1e-9
    u2 = np.where(over, 1 - v, u); v2 = np.where(over, 1 - u, v)
    P = A[:, None, None, :] + u2[None, :, :, None] * (B - A)[:, None, None, :] + v2[None, :, :, None] * (C - A)[:, None, None, :]
    return P.astype(np.float32), nrm.astype(np.float32)


def bake(frames, V, T, stops, log=print, k=K, width=ATLAS_W, size=4096):
    """stops: list of dict(id, file, pos, W (pano ray -> world), up). Returns the atlas (H, W, 3) uint8 RGB and its layout."""
    import torch, open3d as o3d
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    t0 = time.time()
    n_tri = len(T); per_row, H = layout(n_tri, k, width)
    P, nrm = block_points(V, T, k, per_row)
    n = n_tri * k * k
    Pf = torch.from_numpy(P.reshape(n, 3)).to(dev)
    nf = torch.from_numpy(np.repeat(nrm, k * k, axis=0)).to(dev)
    S = len(stops)
    pos = torch.tensor(np.array([s["pos"] for s in stops]), dtype=torch.float32, device=dev)
    # best stops per texel
    top_idx = torch.empty((n, min(TOPK, S)), dtype=torch.long, device=dev); top_sc = torch.empty((n, min(TOPK, S)), device=dev)
    for a in range(0, n, 1_000_000):
        sl = slice(a, min(n, a + 1_000_000))
        d = pos[None, :, :] - Pf[sl, None, :]                                           # texel -> stop
        dist = d.norm(dim=2).clamp(min=0.05)
        cos = (d * nf[sl, None, :]).sum(2).abs() / dist
        sc = torch.where((cos > 0.03) & (dist < 30.0), cos / (dist + 0.5), torch.zeros_like(cos))
        v_, i_ = torch.topk(sc, min(TOPK, S), dim=1)
        top_idx[sl], top_sc[sl] = i_, v_
    # visibility by ray casting: nothing of the model between the stop and the point
    scene = o3d.t.geometry.RaycastingScene()
    mesh_t = o3d.t.geometry.TriangleMesh(); mesh_t.vertex.positions = o3d.core.Tensor(V.astype(np.float32)); mesh_t.triangle.indices = o3d.core.Tensor(T.astype(np.int32))
    scene.add_triangles(mesh_t)
    ti, ts = top_idx.cpu().numpy(), top_sc.cpu().numpy()
    Pn = Pf.cpu().numpy(); posn = pos.cpu().numpy()
    vis = np.zeros(ti.shape, bool)
    for c in range(ti.shape[1]):
        o = posn[ti[:, c]]; d = Pn - o; dist = np.linalg.norm(d, axis=1)
        rays = o3d.core.Tensor(np.concatenate([o, d / np.maximum(dist, 1e-6)[:, None]], 1).astype(np.float32))
        th = scene.cast_rays(rays)["t_hit"].numpy()
        vis[:, c] = (ts[:, c] > 0) & (th >= dist - (VIS_TOL0 + VIS_TOL1 * dist))
    log(f"  visibility: {vis.mean() * 100:.0f}% of the (texel, stop) candidates are in the open ({time.time() - t0:.0f} s)")
    # colours
    acc = np.zeros((n, 3), np.float32); wsum = np.zeros(n, np.float32)
    fall = np.zeros((n, 3), np.float32); fsc = np.zeros(n, np.float32)
    for si, s in enumerate(stops):
        m = (ti == si)
        if not m.any():
            continue
        rows_, cols_ = np.nonzero(m)
        pano = torch.from_numpy(cv2.cvtColor(cv2.resize(imread(s["file"]), (size, size // 2), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)).to(dev).permute(2, 0, 1)[None].float()
        Wm = torch.tensor(s["W"], dtype=torch.float32, device=dev)
        hole = s.get("hole")
        for a in range(0, len(rows_), 1_500_000):
            r = rows_[a:a + 1_500_000]; c = cols_[a:a + 1_500_000]
            pts = Pf[r]; dvec = (pts - pos[si]) @ Wm                                    # world offset -> the stop's panorama frame (W^T x)
            dn = dvec.norm(dim=1, keepdim=True).clamp(min=1e-6)
            az = torch.atan2(dvec[:, 0], dvec[:, 2]); el = torch.asin((dvec[:, 1] / dn[:, 0]).clamp(-1, 1))
            gx = az / math.pi; gy = -el / (math.pi / 2)
            grid = torch.stack([gx, gy], 1)[None, :, None, :]
            col = torch.nn.functional.grid_sample(pano, grid, mode="bilinear", padding_mode="border", align_corners=False)[0, :, :, 0].T.cpu().numpy()
            w = ts[r, c] ** POWER * vis[r, c]
            if hole is not None:
                hh = torch.nn.functional.grid_sample(hole, grid, mode="nearest", padding_mode="border", align_corners=False)[0, 0, :, 0].cpu().numpy() > 0.5
                w = np.where(hh, 0.0, w)
            np.add.at(acc, r, col * w[:, None]); np.add.at(wsum, r, w)
            better = ts[r, c] > fsc[r]
            fall[r[better]] = col[better]; fsc[r[better]] = ts[r, c][better]
        del pano
    have = wsum > 1e-12
    out = np.where(have[:, None], acc / np.maximum(wsum, 1e-12)[:, None], fall)
    log(f"  colours: {have.mean() * 100:.1f}% of the texels seen in the open, the rest from the best-facing stop ({time.time() - t0:.0f} s)")
    atlas = np.zeros((H, per_row * k, 3), np.uint8)
    blocks = np.clip(out, 0, 255).astype(np.uint8).reshape(n_tri, k, k, 3)
    r_ = np.arange(n_tri) // per_row; c_ = np.arange(n_tri) % per_row
    for y in range(k):
        for x in range(k):
            atlas[r_ * k + y, c_ * k + x] = blocks[:, y, x]
    seen_tri = have.reshape(n_tri, k * k).mean(1)
    return atlas, dict(k=k, per_row=per_row, width=per_row * k, height=H, stops=len(stops), seen_pct=round(float(have.mean() * 100), 1)), seen_tri


def visible_triangles(V, T, stops, W=2048, H=1024):
    """True for every triangle that is the first thing some stop sees along some ray of a W x H sphere of rays: a triangle that no stop can see (a hidden back side, a floater
    inside a wall) can only ever carry made-up colours, and the stops (held-back ones excluded) are the only places the model is meant to be looked at from."""
    import open3d as o3d
    scene = o3d.t.geometry.RaycastingScene()
    mt = o3d.t.geometry.TriangleMesh(); mt.vertex.positions = o3d.core.Tensor(V.astype(np.float32)); mt.triangle.indices = o3d.core.Tensor(T.astype(np.int32))
    scene.add_triangles(mt)
    az = (np.arange(W) + 0.5) / W * 2 * np.pi - np.pi; el = np.pi / 2 - (np.arange(H) + 0.5) / H * np.pi
    A, E = np.meshgrid(az, el)
    dirs = np.stack([np.cos(E) * np.sin(A), np.sin(E), np.cos(E) * np.cos(A)], -1).reshape(-1, 3).astype(np.float32)
    seen = np.zeros(len(T), bool)
    for s in stops:
        d = dirs @ np.asarray(s["W"], np.float32).T                                          # pano ray -> world ray
        o = np.broadcast_to(np.asarray(s["pos"], np.float32), d.shape)
        pid = scene.cast_rays(o3d.core.Tensor(np.concatenate([o, d], 1)))["primitive_ids"].numpy()
        pid = pid[(pid >= 0) & (pid < len(T))]
        seen[pid] = True
    return seen


def texture_mesh(frames, mesh, stops, log=print, k=K):
    """Drop the triangles that no stop sees, colour the rest from the pictures. Returns the pruned mesh, the atlas (RGB uint8) and its layout."""
    V = np.asarray(mesh.vertices, np.float32); T = np.asarray(mesh.triangles, np.int64)
    seen = visible_triangles(V, T, stops)
    log(f"  {int((~seen).sum())} of {len(T)} triangles are seen by no stop (dropped)")
    mesh.remove_triangles_by_mask(~seen); mesh.remove_unreferenced_vertices()
    tri_cl, n_tri, _ = mesh.cluster_connected_triangles()
    tri_cl, n_tri = np.asarray(tri_cl), np.asarray(n_tri)
    small = np.flatnonzero(n_tri < 60)                                                   # slivers left over by the pruning
    if len(small):
        mesh.remove_triangles_by_mask(np.isin(tri_cl, small)); mesh.remove_unreferenced_vertices()
    V = np.asarray(mesh.vertices, np.float32); T = np.asarray(mesh.triangles, np.int64)
    atlas, meta, _ = bake(frames, V, T, stops, log=log, k=k)
    meta["pruned_triangles"] = int((~seen).sum())
    return mesh, atlas, meta


def stops_for(frames, only=None):
    import torch
    import mvs
    st = mvs.load_stops(frames, only)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for s in st:
        hp = os.path.join(frames, "capturer", s["id"] + "_hole.png")
        if os.path.exists(hp):
            m = D.hole_in_eq(cv2.imread(hp, 0) > 0, s["up"], 1024, 512)
            m = cv2.dilate(m.astype(np.uint8), np.ones((15, 15), np.uint8)) > 0
            s["hole"] = torch.from_numpy(m.astype(np.float32))[None, None].to(dev)
    return st


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--frames", default="frames"); ap.add_argument("--mesh", default="mesh"); ap.add_argument("--k", type=int, default=K)
    ap.add_argument("--tol", type=float, nargs=2, default=None, help="visibility tolerance: constant and per metre (m)"); ap.add_argument("--prune", action="store_true", help="also drop the triangles no stop sees and write <mesh>_pruned.ply")
    a = ap.parse_args()
    global VIS_TOL0, VIS_TOL1
    if a.tol:
        VIS_TOL0, VIS_TOL1 = a.tol
    import open3d as o3d
    m = o3d.io.read_triangle_mesh(os.path.join(a.frames, "dollhouse", a.mesh + ".ply"))
    V = np.asarray(m.vertices, np.float32); T = np.asarray(m.triangles, np.int64)
    if a.prune:
        mesh, atlas, meta = texture_mesh(a.frames, m, stops_for(a.frames), k=a.k)
        o3d.io.write_triangle_mesh(os.path.join(a.frames, "dollhouse", a.mesh + "_pruned.ply"), mesh)
        a.mesh = a.mesh + "_pruned"
    else:
        atlas, meta, _ = bake(a.frames, V, T, stops_for(a.frames), k=a.k, width=(ATLAS_W // a.k) * a.k)
    out = os.path.join(a.frames, "dollhouse", a.mesh + "_atlas.jpg")
    cv2.imwrite(out, cv2.cvtColor(atlas, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 88])
    json.dump(meta, open(os.path.join(a.frames, "dollhouse", a.mesh + "_atlas.json"), "w"))
    print(meta, f"{os.path.getsize(out) / 1e6:.2f} MB")


if __name__ == "__main__":
    main()
