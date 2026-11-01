"""The dollhouse: fuse every stop's depth into ONE coloured mesh (TSDF, Open3D, MIT) that the viewer can orbit from above.

Usage:  python dollhouse.py [--frames frames] [--out tour] [--voxel 0.04] [--max-tris 90000]
Reads:  frames/poses.json (x, y, heading, camera height per stop), frames/up.json, frames/depth/eq_<id>.png (range along every ray, 8-bit log),
        frames/safe/<id>.jpg (the cleaned pictures: held-back stops are not in frames/safe/ and are not used), frames/capturer/<id>_hole.png
Writes: tour/dollhouse.bin + tour/dollhouse.json (a small quantised mesh the viewer loads on demand), frames/dollhouse/mesh.ply, frames/dollhouse/stats.json

World frame (metres): X east, Y north, Z up, floor at Z = 0. A stop's pano azimuth `a` (clockwise from the picture centre) points at bearing H + a.
Every stop is cut into six 90-degree faces (a cube map) because TSDF fusion wants pinhole images. The person who held the stick is replaced by floor in the
depth (otherwise 23 little statues would stand in the model).
"""
import argparse, json, math, os, sys, time
import cv2
import numpy as np
from imutil import imread

DMIN, DMAX = 0.2, 60.0          # place_rings.py: 8-bit log range encoding
MAX_RANGE = 12.0                # depth beyond this is not used
FACE = 512
FACES = {                        # (forward, right, up) in the stop's levelled frame (x right, y up, z forward)
    "front": ((0, 0, 1), (1, 0, 0), (0, 1, 0)), "right": ((1, 0, 0), (0, 0, -1), (0, 1, 0)), "back": ((0, 0, -1), (-1, 0, 0), (0, 1, 0)),
    "left": ((-1, 0, 0), (0, 0, 1), (0, 1, 0)), "up": ((0, 1, 0), (1, 0, 0), (0, 0, -1)), "down": ((0, -1, 0), (1, 0, 0), (0, 0, 1)),
}


def rot_to_y(u):
    u = np.asarray(u, float); u = u / np.linalg.norm(u); y = np.array([0, 1.0, 0]); v = np.cross(u, y); c = float(u @ y)
    if np.linalg.norm(v) < 1e-9:
        return np.eye(3)
    K = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + K + K @ K / (1 + c)


def level_to_world(heading_deg):
    """Matrix M: a vector in the levelled frame (x right, y up, z forward) -> world (X east, Y north, Z up). Forward is at bearing H."""
    h = math.radians(heading_deg); s, c = math.sin(h), math.cos(h)
    return np.array([[c, 0, s], [-s, 0, c], [0, 1, 0]], float)


def decode_range(png):
    return DMIN * (DMAX / DMIN) ** (png[..., 2].astype(np.float32) / 255.0)


def face_rays(fov_deg=90.0, n=FACE):
    f = n / 2 / math.tan(math.radians(fov_deg) / 2)
    c, r = np.meshgrid(np.arange(n) + 0.5, np.arange(n) + 0.5)
    d = np.stack([(c - n / 2) / f, -(r - n / 2) / f, np.ones_like(c)], -1)               # camera: x right, y up, z forward
    return d / np.linalg.norm(d, axis=-1, keepdims=True), f


def pano_uv(r, h, w):
    az = np.arctan2(r[..., 0], r[..., 2]); el = np.arcsin(np.clip(r[..., 1] / np.linalg.norm(r, axis=-1), -1, 1))
    return ((az + np.pi) / (2 * np.pi) * w - 0.5).astype(np.float32), ((np.pi / 2 - el) / np.pi * h - 0.5).astype(np.float32)


def hole_in_eq(hole1024, up, W, H, fov=130.0):
    """The capturer's straight-down mask (the 1024 px view of remove_capturer.py) as an equirect mask."""
    n = hole1024.shape[0]; f = n / 2 / math.tan(math.radians(fov) / 2)
    az = ((np.arange(W) + 0.5) / W * 2 * np.pi - np.pi); el = (np.pi / 2 - (np.arange(H) + 0.5) / H * np.pi)
    A, E = np.meshgrid(az, el)
    r = np.stack([np.cos(E) * np.sin(A), np.sin(E), np.cos(E) * np.cos(A)], -1) @ rot_to_y(up).T            # pano ray -> levelled ray
    dy = np.maximum(-r[..., 1], 1e-3)
    col = (f * r[..., 0] / dy + n / 2 - 0.5).astype(np.float32); row = (n / 2 - 0.5 - f * r[..., 2] / dy).astype(np.float32)
    m = cv2.remap(hole1024.astype(np.uint8), col, row, cv2.INTER_NEAREST, borderValue=0) > 0
    return m & (r[..., 1] < -0.3)


def stop_faces(frames, node, up, rng, pano, hole_eq):
    """The six faces of one stop: (colour uint8 HxWx3 RGB, z-depth float32, camera-to-world 4x4) each."""
    Hh, Ww = rng.shape
    M = level_to_world(node["H"]); R = rot_to_y(up)
    pos = np.array([node["x"], node["y"], node["cam_h"] or 2.0], float)
    d, f = face_rays()
    out = []
    for name, (fw, rt, upv) in FACES.items():
        fw, rt, upv = (np.array(v, float) for v in (fw, rt, upv))
        dl = d[..., 0:1] * rt + d[..., 1:2] * upv + d[..., 2:3] * fw                    # rays in the levelled frame
        pr = dl @ R                                                                       # levelled -> the pano's own frame
        u, v = pano_uv(pr, Hh, Ww)
        rg = cv2.remap(rng, u, v, cv2.INTER_NEAREST, borderMode=cv2.BORDER_WRAP)
        hm = cv2.remap(hole_eq.astype(np.uint8), u, v, cv2.INTER_NEAREST, borderMode=cv2.BORDER_WRAP) > 0 if hole_eq is not None else np.zeros(rg.shape, bool)
        down = -dl[..., 1]
        floor_r = (node["cam_h"] or 2.0) / np.maximum(down, 0.05)
        rg = np.where(hm & (down > 0.05), floor_r, rg)                                    # him: replaced by floor
        rg = np.where(hm & (down <= 0.05), 0.0, rg)
        rg = np.where(rg > MAX_RANGE, 0.0, rg)                                            # far depth is too noisy to build a wall from: a nearer stop will see it
        zd = (rg * d[..., 2]).astype(np.float32)                                          # z-depth along the face axis
        k = np.ones((5, 5), np.uint8)
        edge = (cv2.dilate(zd, k) - cv2.erode(zd, k)) > 0.12 * np.maximum(zd, 0.5)         # a jump in depth: 'flying pixels' between a thing and what is behind it
        zd = np.where(cv2.dilate(edge.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0, 0.0, zd).astype(np.float32)
        col = cv2.remap(pano, u, v, cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
        c2w = np.eye(4)
        right_w, up_w, fw_w = M @ rt, M @ upv, M @ fw
        c2w[:3, 0], c2w[:3, 1], c2w[:3, 2], c2w[:3, 3] = right_w, -up_w, fw_w, pos        # Open3D camera: x right, y down, z forward
        out.append((cv2.cvtColor(col, cv2.COLOR_BGR2RGB), zd, c2w, name))
    return out


class Tsdf:
    """A dense truncated signed distance volume on the GPU (torch): running weighted average of the signed distance and the colour of every voxel
    that a depth face sees. (Open3D's ScalableTSDFVolume gave an empty mesh on every input in the installed build, even on a flat test plane, so the
    fusion is done here; Open3D is used for cleaning and decimating the mesh.)"""
    def __init__(self, lo, hi, voxel, trunc, device):
        import torch
        self.t = torch; self.dev = device; self.voxel = voxel; self.trunc = trunc
        self.lo = np.asarray(lo, np.float64); self.dims = np.ceil((np.asarray(hi) - self.lo) / voxel).astype(int)
        shp = tuple(int(x) for x in self.dims)
        self.tsdf = torch.zeros(shp, dtype=torch.float32, device=device); self.w = torch.zeros(shp, dtype=torch.float32, device=device)
        self.col = torch.zeros(shp + (3,), dtype=torch.float16, device=device); self.cw = torch.zeros(shp, dtype=torch.float32, device=device)

    def integrate(self, depth, color, K, w2c, reach=14.0, cam_pos=None):
        """depth: z-depth image (0 = none), color: HxWx3 uint8, K: (f, cx, cy), w2c: 4x4 world->camera."""
        t = self.t; dev = self.dev
        H, W = depth.shape
        d_t = t.from_numpy(depth).to(dev); c_t = t.from_numpy(color).to(dev)
        R = t.from_numpy(w2c[:3, :3].astype(np.float32)).to(dev); tr = t.from_numpy(w2c[:3, 3].astype(np.float32)).to(dev)
        f, cx, cy = K
        # only the voxels near this camera
        lo = np.maximum(((cam_pos - reach) - self.lo) / self.voxel, 0).astype(int); hi = np.minimum(((cam_pos + reach) - self.lo) / self.voxel, self.dims).astype(int)
        xs, ys, zs = (t.arange(lo[k], hi[k], device=dev) for k in range(3))
        gx, gy, gz = t.meshgrid(xs, ys, zs, indexing="ij")
        P = t.stack([(gx.float() + 0.5) * self.voxel + float(self.lo[0]), (gy.float() + 0.5) * self.voxel + float(self.lo[1]), (gz.float() + 0.5) * self.voxel + float(self.lo[2])], -1)
        Xc = P @ R.T + tr
        z = Xc[..., 2]
        u = (Xc[..., 0] / z.clamp(min=1e-3) * f + cx).long(); v = (Xc[..., 1] / z.clamp(min=1e-3) * f + cy).long()
        ok = (z > 0.1) & (u >= 0) & (u < W) & (v >= 0) & (v < H)
        uu, vv = u.clamp(0, W - 1), v.clamp(0, H - 1)
        dd = d_t[vv, uu]
        sdf = dd - z
        ok &= (dd > 0) & (sdf > -self.trunc)
        val = (sdf / self.trunc).clamp(max=1.0)
        wt = (1.0 / (0.5 + dd * dd)) * ok.float()                       # near views count far more than far ones (depth error grows with the range)
        sl = (slice(lo[0], hi[0]), slice(lo[1], hi[1]), slice(lo[2], hi[2]))
        w0 = self.w[sl]; tot = w0 + wt
        self.tsdf[sl] = t.where(tot > 0, (self.tsdf[sl] * w0 + val * wt) / tot.clamp(min=1e-6), self.tsdf[sl])
        near = (sdf.abs() < self.trunc * 0.5) & ok                          # colour only from right at the surface
        cnew = wt * near.float()
        cw0 = self.cw[sl]; c0 = self.col[sl].float(); c_new = c_t[vv, uu].float()
        self.col[sl] = t.where(near[..., None], (c0 * cw0[..., None] + c_new * cnew[..., None]) / (cw0 + cnew).clamp(min=1e-6)[..., None], c0).half()
        self.cw[sl] = cw0 + cnew
        self.w[sl] = tot

    def mesh(self):
        from skimage import measure
        vol = self.tsdf.cpu().numpy(); obs = (self.w.cpu().numpy() > 0)
        verts, faces, normals, _ = measure.marching_cubes(vol, level=0.0, mask=obs, spacing=(self.voxel,) * 3)
        verts = verts + self.lo + self.voxel * 0.5
        # colours at the vertices (nearest voxel)
        idx = np.clip(((verts - self.lo) / self.voxel).astype(int), 0, self.dims - 1)
        cols = self.col.cpu().numpy()[idx[:, 0], idx[:, 1], idx[:, 2]].astype(np.float32)
        return verts, faces, cols


def fuse(frames, voxel=0.05, trunc=0.3, ids=None, log=print):
    import torch, open3d as o3d
    P = json.load(open(os.path.join(frames, "poses.json")))["nodes"]
    UP = json.load(open(os.path.join(frames, "up.json")))
    pts = json.load(open(os.path.join(frames, "points.json")))
    xy = np.array([[n["x"], n["y"]] for n, p in zip(P, pts) if n.get("x") is not None and (not ids or p["id"] in ids)])
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    margin = 9.0
    vol = Tsdf([xy[:, 0].min() - margin, xy[:, 1].min() - margin, -0.3], [xy[:, 0].max() + margin, xy[:, 1].max() + margin, 6.5], voxel, trunc, dev)
    log(f"volume {vol.dims.tolist()} voxels of {voxel} m on {dev}")
    f = FACE / 2 / math.tan(math.radians(45)); K = (f, FACE / 2, FACE / 2)
    used = []
    for i, p in enumerate(pts):
        node = P[i]
        sp = os.path.join(frames, "safe", p["filename"]); ep = os.path.join(frames, "depth", f"eq_{p['id']}.png")
        if (ids and p["id"] not in ids) or not os.path.exists(sp) or not os.path.exists(ep) or node.get("x") is None:
            continue
        pano = imread(sp); pano = cv2.resize(pano, (4096, 2048), interpolation=cv2.INTER_AREA)
        rng = decode_range(imread(ep)); rng = cv2.resize(rng, (1024, 512), interpolation=cv2.INTER_NEAREST)
        hp = os.path.join(frames, "capturer", p["id"] + "_hole.png")
        hole = hole_in_eq(cv2.imread(hp, 0) > 0, UP[i], rng.shape[1], rng.shape[0]) if os.path.exists(hp) else None
        for col, zd, c2w, name in stop_faces(frames, node, UP[i], rng, pano, hole):
            vol.integrate(zd, col, K, np.linalg.inv(c2w), cam_pos=c2w[:3, 3])
        used.append(p["id"])
    log(f"fused {len(used)} stops")
    V, T, C = vol.mesh()
    mesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(V), o3d.utility.Vector3iVector(T))
    mesh.vertex_colors = o3d.utility.Vector3dVector(np.clip(C / 255.0, 0, 1))
    return mesh, used


def clean(mesh, min_island_tris=400, max_tris=90000):
    """Drop floaters and small islands, then decimate."""
    import open3d as o3d
    mesh.remove_degenerate_triangles(); mesh.remove_duplicated_vertices(); mesh.remove_unreferenced_vertices()
    tri_cl, n_tri, area = mesh.cluster_connected_triangles()
    tri_cl, n_tri = np.asarray(tri_cl), np.asarray(n_tri)
    keep_cl = np.flatnonzero(n_tri >= max(min_island_tris, 0.02 * n_tri.max()))
    mesh.remove_triangles_by_mask(~np.isin(tri_cl, keep_cl))
    mesh.remove_unreferenced_vertices()
    mesh = mesh.filter_smooth_taubin(number_of_iterations=12)                        # smooths the stair-steps of the voxels without shrinking the model
    before = len(mesh.triangles)
    if before > max_tris:
        mesh = mesh.simplify_quadric_decimation(target_number_of_triangles=max_tris)
        mesh.remove_degenerate_triangles(); mesh.remove_unreferenced_vertices()
    return mesh, before


def export(mesh, out_dir, name="dollhouse"):
    """A small quantised binary: positions int16 (relative to the box), colours uint8, indices uint16 (or uint32 above 65k vertices)."""
    V = np.asarray(mesh.vertices, np.float64); T = np.asarray(mesh.triangles); C = np.clip(np.asarray(mesh.vertex_colors) * 255, 0, 255).astype(np.uint8)
    lo, hi = V.min(0), V.max(0); span = np.maximum(hi - lo, 1e-6)
    Q = np.round((V - lo) / span * 65535 - 32768).astype(np.int16)
    idx = T.astype(np.uint16 if len(V) < 65536 else np.uint32)
    blob = Q.tobytes() + C.tobytes() + idx.tobytes()
    meta = dict(vertices=int(len(V)), triangles=int(len(T)), lo=[float(x) for x in lo], span=[float(x) for x in span], index_bytes=int(idx.itemsize),
                bytes=len(blob), layout="int16[3*v] uint8[3*v] uint16|uint32[3*t]", frame="X east, Y north, Z up (metres), floor at Z=0")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, name + ".bin"), "wb") as fh:
        fh.write(blob)
    with open(os.path.join(out_dir, name + ".json"), "w") as fh:
        json.dump(meta, fh)
    return meta


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--frames", default="frames"); ap.add_argument("--out", default="tour")
    ap.add_argument("--voxel", type=float, default=0.05); ap.add_argument("--max-tris", type=int, default=90000)
    ap.add_argument("--only", nargs="*")
    a = ap.parse_args()
    t0 = time.time()
    mesh, used = fuse(a.frames, a.voxel, ids=a.only)
    mesh, raw_tris = clean(mesh, max_tris=a.max_tris)
    import open3d as o3d
    os.makedirs(os.path.join(a.frames, "dollhouse"), exist_ok=True)
    o3d.io.write_triangle_mesh(os.path.join(a.frames, "dollhouse", "mesh.ply"), mesh)
    meta = export(mesh, a.out)
    meta.update(stops=used, raw_triangles=int(raw_tris), seconds=round(time.time() - t0, 1), voxel=a.voxel)
    json.dump(meta, open(os.path.join(a.frames, "dollhouse", "stats.json"), "w"), indent=1)
    tj = os.path.join(a.out, "tour.json")
    if os.path.exists(tj):                                       # tell the viewer there is a dollhouse (build_tour.py does the same on a rebuild)
        with open(tj) as fh:
            t = json.load(fh)
        t["dollhouse"] = True
        with open(tj, "w") as fh:
            json.dump(t, fh, separators=(",", ":"))
    print(f"dollhouse: {meta['vertices']} vertices, {meta['triangles']} triangles, {meta['bytes'] / 1e6:.2f} MB, {len(used)} stops, {meta['seconds']} s")


if __name__ == "__main__":
    main()
