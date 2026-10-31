"""One consistent, metric, full-sphere depth for every stop, from MoGe-2 (MIT) on overlapping views.

Usage:  python depth_sphere.py [--frames frames] [--px 768] [--width 2048] [--only 003 004]
Reads:  frames/safe/<id>.jpg (else frames/points/<id>.jpg), frames/points.json, frames/poses.json (positions, the map's scale, cam_h), frames/up.json,
        frames/colmap/best (the sparse map: the metre), frames/depth/eq_<id>.png (the floor mask of place_rings.py, when there is one)
Writes: frames/depth_sphere/<id>.npz   range (float16, metres along every ray of a width x width/2 equirect, 0 = none), conf (uint8)
        frames/depth_sphere/eq_<id>.png the same encoding as frames/depth/eq_<id>.png (R = 8-bit log range, G = floor), at width x width/2
        frames/depth_sphere/stats.json  per stop: the per-view scales, how well the views agreed, the floor height the depth itself sees, the roof height

Why: the four looking-down views of depth_panos.py left the upper walls and the ceiling with no depth at all (the dollhouse had no roof and the walls stopped at
head height), and a small relative-depth model tied to a handful of sparse points put every stop's wall a few percent apart, so the fused walls tore.
MoGe-2 is far sharper and is metric on its own to within a few tens of percent, but EACH VIEW carries its own scale (a view of nothing but the ceiling has no
cue), so: 18 views that overlap (a ring of 8 at the horizon, 4 looking 55 degrees up, 4 down, straight up, straight down; 100 degrees each), one scale per
view solved from the overlaps between views and from the COLMAP points every view sees, then one range per equirect pixel from the views that see it, the
most central view counting most. The floor seen by the depth is snapped onto the floor plane and nothing goes below it. The sky is no surface
(UperNet's ADE20K labels, the floor step's own model: MoGe gives haze and clouds a distance of 15-25 m, which fused into a dome over an outdoor walk)
and water lies flat on its own level.
"""
import argparse, json, math, os, sys, time
import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import dollhouse as D                     # noqa: E402  (rot_to_y, DMIN/DMAX encoding)
from imutil import imread                 # noqa: E402

FOV = 100.0
VIEWS = ([dict(yaw=y, pitch=0.0) for y in range(0, 360, 60)] + [dict(yaw=y, pitch=55.0) for y in range(0, 360, 120)]      # 14 views of 100 degrees: every neighbouring pair overlaps
         + [dict(yaw=y, pitch=-55.0) for y in range(60, 360, 120)] + [dict(yaw=0.0, pitch=90.0), dict(yaw=0.0, pitch=-90.0)])
MODEL = "Ruicheng/moge-2-vitl-normal"
SEG_MODEL = "openmmlab/upernet-convnext-small"           # ADE20K (MIT weights): the floor step's model, here for the sky, the water and the ground
SEG_VIEWS = ([dict(yaw=y, pitch=0.0) for y in range(0, 360, 90)] + [dict(yaw=y, pitch=50.0) for y in range(45, 360, 90)] + [dict(yaw=0.0, pitch=90.0)]
             + [dict(yaw=y, pitch=-50.0) for y in range(45, 360, 90)] + [dict(yaw=0.0, pitch=-90.0)])      # the first nine never looked down: half the sphere, the sea with it, went unlabelled
SEG_PX = 512
INFER_BATCH = 1                          # views a MoGe call: measured on the 3070 Ti, 7 at once peaks at 9 GB (spills) and is no faster than one at a time (1.2 s a view)
SKY_IDS = {2}
WATER_IDS = {21, 26, 60, 128, 113}                        # water, sea, river, lake, waterfall
FLOOR_IDS = {3, 6, 9, 11, 13, 29, 46, 52}                 # floor, road, grass, sidewalk, earth, field, sand, path
EDGE = 0.92                              # a view's outer 8 % (of the half-width) is never trusted for scale, and weighs little in the blend
MAX_RANGE = 40.0


def basis(v):
    """Rows: the view's right, up, forward axes in the LEVELLED frame (x right, y up, z forward = pano centre at the horizon)."""
    p, yv = math.radians(v["pitch"]), math.radians(v["yaw"])
    Rx = np.array([[1, 0, 0], [0, math.cos(p), math.sin(p)], [0, -math.sin(p), math.cos(p)]])
    Ry = np.array([[math.cos(yv), 0, math.sin(yv)], [0, 1, 0], [-math.sin(yv), 0, math.cos(yv)]])
    return (Ry @ Rx).T                                                # levelled ray -> view coords


def view_rays(n, fov=FOV):
    f = n / 2 / math.tan(math.radians(fov) / 2)
    x, y = np.meshgrid(np.arange(n) - n / 2 + 0.5, n / 2 - np.arange(n) - 0.5)
    d = np.stack([x, y, np.full_like(x, f)], -1)
    return d / np.linalg.norm(d, axis=-1, keepdims=True), f


def eq_rays(W, H):
    az = (np.arange(W) + 0.5) / W * 2 * np.pi - np.pi; el = np.pi / 2 - (np.arange(H) + 0.5) / H * np.pi
    A, E = np.meshgrid(az, el)
    return np.stack([np.cos(E) * np.sin(A), np.sin(E), np.cos(E) * np.cos(A)], -1).astype(np.float32)


def bearings(uv, w, h):
    th = uv[:, 0] / w * 2 * np.pi - np.pi; ph = np.pi / 2 - uv[:, 1] / h * np.pi
    return np.stack([np.cos(ph) * np.sin(th), np.sin(ph), np.cos(ph) * np.cos(th)], 1)


def sparse_rays(frames, ids, scale):
    """{stop index: (rays in the pano frame (N, 3), metres (N,))} from the COLMAP map place_rings.py used."""
    import pycolmap
    sp = os.path.join(frames, "colmap", "best") if os.path.isdir(os.path.join(frames, "colmap", "best")) else os.path.join(frames, "colmap", "sparse", "0")
    out = {}
    try:
        rec = pycolmap.Reconstruction(sp)
    except Exception as e:
        print("no sparse map:", e); return out
    for im in rec.images.values():
        try:
            k = int(os.path.splitext(im.name)[0]) - 1
        except ValueError:
            continue
        if k < 0 or k >= len(ids):
            continue
        T = im.cam_from_world() if callable(im.cam_from_world) else im.cam_from_world
        R = T.rotation.matrix(); C = -R.T @ T.translation
        cam = rec.cameras[im.camera_id]
        uv, X = [], []
        for p in im.points2D:
            if p.has_point3D():
                p3 = rec.points3D[p.point3D_id]
                if p3.track.length() >= 2 and p3.error < 1.5:
                    uv.append(p.xy); X.append(p3.xyz)
        if len(uv) < 5:
            continue
        uv, X = np.array(uv), np.array(X)
        out[k] = (bearings(uv, cam.width, cam.height), np.linalg.norm(X - C, axis=1) * scale)
    return out


def cut(pano, R_lv, B, n, f):
    """A view's picture. R_lv: pano ray -> levelled (dollhouse.rot_to_y); B = basis(view). Returns BGR n x n."""
    d, _ = view_rays(n)
    dl = d @ B                                                          # view coords -> levelled (B's rows are the axes)
    pr = dl @ R_lv                                                      # levelled -> pano's own frame (R_lv is orthonormal: x @ R == R^T x)
    H, W = pano.shape[:2]
    u, v = D.pano_uv(pr, H, W)
    return cv2.remap(pano, u, v, cv2.INTER_AREA if W > 4 * n else cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)


def solve_scales(views_rng, views_mask, Bs, anchors, n, f, rng_seed=0):
    """One log-scale per view: overlaps between views agree, and the views match the sparse metres. Returns (log scales, report)."""
    V = len(Bs); half = n / 2
    rs = np.random.default_rng(rng_seed)
    q = rs.normal(size=(30000, 3)); q /= np.linalg.norm(q, axis=1, keepdims=True)             # directions, levelled
    samp = []
    for v in range(V):
        c = q @ Bs[v].T
        z = c[:, 2]; ok = z > 0.05
        x = np.where(ok, c[:, 0] / np.maximum(z, 1e-6) * f + half - 0.5, -1); y = np.where(ok, half - 0.5 - c[:, 1] / np.maximum(z, 1e-6) * f, -1)
        inside = ok & (np.abs(x - half + 0.5) < EDGE * half) & (np.abs(y - half + 0.5) < EDGE * half)
        xi = np.clip(x.round().astype(int), 0, n - 1); yi = np.clip(y.round().astype(int), 0, n - 1)
        r = views_rng[v][yi, xi]; m = views_mask[v][yi, xi] & inside & (r > 0.1)
        samp.append((m, np.log(np.maximum(r, 1e-3))))
    rows, rhs, wts = [], [], []
    pairs = 0
    for a in range(V):
        for b in range(a + 1, V):
            m = samp[a][0] & samp[b][0]
            if m.sum() < 200:
                continue
            dlt = samp[a][1][m] - samp[b][1][m]                       # log r_a - log r_b; want x_a + log r_a = x_b + log r_b  ->  x_a - x_b = -dlt
            med = np.median(dlt); mad = np.median(np.abs(dlt - med)) + 1e-3
            keep = np.abs(dlt - med) < 3 * mad + 0.02
            med = float(np.median(dlt[keep]))
            row = np.zeros(V); row[a] = 1; row[b] = -1
            w = min(1.0, keep.sum() / 1000) / (1 + 10 * mad)
            rows.append(row); rhs.append(-med); wts.append(w); pairs += 1
    anch = {}
    if anchors is not None:
        rays, rng_m = anchors
        for v in range(V):
            c = rays @ Bs[v].T
            z = c[:, 2]; ok = z > 0.05
            x = np.where(ok, c[:, 0] / np.maximum(z, 1e-6) * f + half - 0.5, -1); y = np.where(ok, half - 0.5 - c[:, 1] / np.maximum(z, 1e-6) * f, -1)
            inside = ok & (np.abs(x - half + 0.5) < EDGE * half) & (np.abs(y - half + 0.5) < EDGE * half)
            if inside.sum() < 8:
                continue
            xi = np.clip(x[inside].round().astype(int), 0, n - 1); yi = np.clip(y[inside].round().astype(int), 0, n - 1)
            r = views_rng[v][yi, xi]; mk = views_mask[v][yi, xi] & (r > 0.1)
            if mk.sum() < 8:
                continue
            lr = np.log(rng_m[inside][mk] / r[mk])                     # log(sparse / model): x_v should equal this
            med = np.median(lr); inl = np.abs(lr - med) < 0.25
            if inl.sum() < 6:
                continue
            med = float(np.median(lr[inl]))
            row = np.zeros(V); row[v] = 1
            w = 0.6 * min(1.0, inl.sum() / 40)
            rows.append(row); rhs.append(med); wts.append(w); anch[v] = (med, int(inl.sum()), int(inside.sum()))
    if not rows:
        return np.zeros(V), dict(pairs=0, anchors=0, map_over_moge=None)
    # the views' scales from the overlaps alone, held to MoGe's own metre (mean log scale 0): across a walk MoGe's metre is steadier than the sparse map's
    # local scale (walk 2: the roof at 4.4 m +- 7 % from every stop; the map's metre against it wandered 0.43 - 1.3 along the walk)
    rows_p, rhs_p, wts_p = rows[:pairs], rhs[:pairs], wts[:pairs]
    A = np.vstack([np.array(rows_p) * np.sqrt(np.array(wts_p))[:, None], np.full((1, V), 1.0)]) if pairs else np.full((1, V), 1.0)
    b = np.r_[np.array(rhs_p) * np.sqrt(np.array(wts_p)), 0.0] if pairs else np.zeros(1)
    x, *_ = np.linalg.lstsq(A, b, rcond=None)
    res = (np.array(rows_p) @ x - np.array(rhs_p)) if pairs else np.zeros(0)
    ratio = None
    if anch:                                                            # what the sparse map's metre says about this stop, for the record (and for --metre map)
        ratio = float(np.median([a[0] - x[v] for v, a in anch.items()]))
    rep = dict(pairs=pairs, anchors=len(anch), anchor_views={str(v): dict(log_ratio=round(a[0], 3), inliers=a[1], points=a[2]) for v, a in anch.items()},
               pair_residual_p90=round(float(np.percentile(np.abs(res), 90)) if pairs else 0.0, 4),
               map_over_moge=(round(float(np.exp(ratio)), 3) if ratio is not None else None),
               log_scales=[round(float(s), 3) for s in x])
    return x, rep


def composite(views_rng, views_mask, scales, Bs, n, f, W, H, dev):
    """One range per equirect pixel (levelled frame), blended in log space from every view that sees it; the most central view counts most."""
    import torch
    half = n / 2
    q = torch.from_numpy(eq_rays(W, H)).to(dev)                                   # (H, W, 3) levelled
    num = torch.zeros((H, W), device=dev); den = torch.zeros((H, W), device=dev); sq = torch.zeros((H, W), device=dev)
    for v in range(len(Bs)):
        B = torch.tensor(Bs[v], dtype=torch.float32, device=dev)
        c = q @ B.T
        z = c[..., 2]; ok = z > 0.05
        x = c[..., 0] / z.clamp(min=1e-6) * f + half - 0.5; y = half - 0.5 - c[..., 1] / z.clamp(min=1e-6) * f
        u = torch.maximum((x - half + 0.5).abs(), (y - half + 0.5).abs()) / half                                    # 0 centre .. 1 edge
        inside = ok & (u < 0.995)
        if not inside.any():
            continue
        rmap = torch.from_numpy(np.log(np.maximum(views_rng[v], 1e-3)).astype(np.float32)).to(dev)[None, None]
        mmap = torch.from_numpy(views_mask[v].astype(np.float32)).to(dev)[None, None]
        grid = torch.stack([(x + 0.5) / n * 2 - 1, (y + 0.5) / n * 2 - 1], -1)[None]
        lr = torch.nn.functional.grid_sample(rmap, grid, mode="bilinear", padding_mode="border", align_corners=False)[0, 0] + float(scales[v])
        mk = torch.nn.functional.grid_sample(mmap, grid, mode="nearest", padding_mode="zeros", align_corners=False)[0, 0] > 0.5
        w = torch.where(inside & mk, (1 - u.clamp(0, 1)) ** 3 + 1e-4, torch.zeros_like(u))
        num += w * lr; den += w; sq += w * lr * lr
    have = den > 1e-6
    lr = torch.where(have, num / den.clamp(min=1e-6), torch.zeros_like(num))
    var = torch.where(have, sq / den.clamp(min=1e-6) - lr * lr, torch.zeros_like(num)).clamp(min=0)
    rng = torch.where(have, lr.exp(), torch.zeros_like(lr))
    return rng.cpu().numpy(), var.sqrt().cpu().numpy(), have.cpu().numpy()


def segment(seg, pano, R_lv, Bs, n, f, W, H, dev):
    """ADE20K labels for every equirect pixel (levelled frame): each SEG_VIEW is labelled by UperNet, and a pixel takes the label of the view it is most
    central in. Returns an int16 (H, W) map, -1 where no view sees it."""
    import torch
    from PIL import Image
    proc, model = seg
    labs = []
    for B in Bs:
        img = cut(pano, R_lv, B, n, f)
        inputs = proc(images=Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)), return_tensors="pt").to(dev)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev.type == "cuda"):
            lg = model(**inputs).logits.float()
        lab = torch.nn.functional.interpolate(lg, size=(n, n), mode="bilinear", align_corners=False)[0].argmax(0).to(torch.int16)
        labs.append(lab)
    half = n / 2
    q = torch.from_numpy(eq_rays(W, H)).to(dev)
    best_u = torch.full((H, W), 9.0, device=dev); out = torch.full((H, W), -1, dtype=torch.int16, device=dev)
    for B, lab in zip(Bs, labs):
        Bt = torch.tensor(B, dtype=torch.float32, device=dev)
        c = q @ Bt.T
        z = c[..., 2]; ok = z > 0.05
        x = c[..., 0] / z.clamp(min=1e-6) * f + half - 0.5; y = half - 0.5 - c[..., 1] / z.clamp(min=1e-6) * f
        u = torch.maximum((x - half + 0.5).abs(), (y - half + 0.5).abs()) / half
        inside = ok & (u < 0.995)
        xi = x.round().long().clamp(0, n - 1); yi = y.round().long().clamp(0, n - 1)
        take = inside & (u < best_u)
        out = torch.where(take, lab[yi, xi], out); best_u = torch.where(take, u, best_u)
    return out.cpu().numpy()


def floor_height(rng, q, floor_mask):
    """Metres from the camera down to the floor, as the depth itself sees it: the median height of the floor pixels (semantic mask when there is one, else the
    densest layer under the camera). (value, how sure: share of the lower hemisphere that agrees within 8 cm)."""
    y = -(q[..., 1] * rng)                                              # drop below the camera
    # THE GROUND UNDER HIM, not the ground he sees: steep rays within a few metres (a beach below a staircase is ground too, 10 m down and 20 m off)
    down = (q[..., 1] < -0.35) & (rng > 0.3) & (rng < 6.0)
    if floor_mask is not None and (floor_mask & down).sum() > 2000:
        sel = floor_mask & down
        h = float(np.median(y[sel]))
    else:
        hist, edges = np.histogram(y[down], bins=np.arange(0.8, 4.0, 0.03))
        hist = np.convolve(hist, [1, 2, 3, 2, 1], "same")
        h = float((edges[np.argmax(hist)] + edges[np.argmax(hist) + 1]) / 2)
    sure = float(((np.abs(y - h) < 0.08) & down).sum() / max(1, down.sum()))
    return h, sure


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--frames", default="frames"); ap.add_argument("--px", type=int, default=768); ap.add_argument("--width", type=int, default=2048)
    ap.add_argument("--only", nargs="*"); ap.add_argument("--model", default=MODEL)
    ap.add_argument("--metre", choices=["moge", "map"], default="moge", help="moge: MoGe's own metric scale (the median over the walk); map: the sparse map's (accelerometer) metre, one ratio for the whole walk")
    ap.add_argument("--no-stick", action="store_true", help="do not tie the stops together by the stick height (each stop keeps MoGe's own scale)")
    ap.add_argument("--keep-cam-h", action="store_true", help="leave poses.json's camera heights as they are (default: the depth's own floor height is written, it is what the model stands on)")
    ap.add_argument("--no-seg", action="store_true", help="skip the sky / water / ground segmentation (the sky then stands as a far surface)")
    a = ap.parse_args()
    t0 = time.time()
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    import torch, warnings
    warnings.filterwarnings("ignore")
    from moge.model.v2 import MoGeModel
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        model = MoGeModel.from_pretrained(a.model).to(dev).eval()
    except OSError as e:
        raise SystemExit(f"cannot load {a.model}: it is downloaded from huggingface.co the first time (about 1.3 GB); connect once. ({str(e)[:120]})")
    seg = None
    if not a.no_seg:
        try:
            from transformers import AutoImageProcessor, UperNetForSemanticSegmentation
            seg = (AutoImageProcessor.from_pretrained(SEG_MODEL), UperNetForSemanticSegmentation.from_pretrained(SEG_MODEL).to(dev).eval())
        except Exception as e:                                            # never a gate: without it the sky stays a far surface, as before
            print(f"note: the segmenter {SEG_MODEL} did not load ({str(e)[:80]}); the sky and the water are not told apart")
    print(f"device {dev}, model loaded in {time.time() - t0:.0f} s", flush=True)
    pts = json.load(open(os.path.join(a.frames, "points.json")))
    pf = os.path.join(a.frames, "poses.json"); poses = json.load(open(pf)); P = poses["nodes"]
    UP = json.load(open(os.path.join(a.frames, "up.json")))
    ids = [p["id"] for p in pts]
    scale = float(poses.get("scale") or 1.0)
    trusted = str(poses.get("scale_info", {}).get("source", "")).startswith("accelerometer")
    anchors = sparse_rays(a.frames, ids, scale)
    out = os.path.join(a.frames, "depth_sphere"); os.makedirs(out, exist_ok=True)
    n = a.px; W, H = a.width, a.width // 2
    Bs = [basis(v) for v in VIEWS]
    d_view, f = view_rays(n)
    Bs_seg = [basis(v) for v in SEG_VIEWS]; _, f_seg = view_rays(SEG_PX)
    q_eq = eq_rays(W, H)
    stats = {"model": a.model, "views": VIEWS, "fov": FOV, "px": n, "width": W, "map_scale_trusted": trusted, "metre": a.metre, "stops": {}}
    # pass 1: every stop in MoGe's own metre
    kept = {}
    for i, p in enumerate(pts):
        if a.only and p["id"] not in a.only:
            continue
        if P[i].get("x") is None:
            continue
        ts = time.time()
        safe = os.path.join(a.frames, "safe", p["filename"])            # the cleaned picture when there is one: other people painted out, so they never become lumps in the model
        pano = imread(safe if os.path.exists(safe) else os.path.join(a.frames, "points", p["filename"]))
        R_lv = D.rot_to_y(UP[i])                                        # pano ray -> levelled
        rngs, masks = [], []
        for c0 in range(0, len(VIEWS), INFER_BATCH):                    # a few views a call: the card is kept busy, and 8 GB holds them
            imgs = [cut(pano, R_lv, B, n, f) for B in Bs[c0:c0 + INFER_BATCH]]
            tens = torch.from_numpy(np.stack([cv2.cvtColor(im, cv2.COLOR_BGR2RGB) for im in imgs])).to(dev).permute(0, 3, 1, 2).float() / 255
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev.type == "cuda"):
                o = model.infer(tens, fov_x=FOV)
            zs = o["depth"].float().cpu().numpy(); mks = o["mask"].cpu().numpy()
            for z, mk in zip(zs, mks):
                mk = mk & np.isfinite(z) & (z > 0)
                r = np.where(mk, z / np.maximum(d_view[..., 2], 1e-3), 0.0).astype(np.float32)
                rngs.append(r); masks.append(mk & (r > 0) & (r < MAX_RANGE))
        x, rep = solve_scales(rngs, masks, Bs, anchors.get(i), n, f, rng_seed=i)
        rng, spread, have = composite(rngs, masks, x, Bs, n, f, W, H, dev)
        old = os.path.join(a.frames, "depth", f"eq_{p['id']}.png")
        fmask = None
        if os.path.exists(old):
            g = cv2.imread(old)[..., 1]
            fmask = cv2.resize(g, (W, H), interpolation=cv2.INTER_NEAREST) >= 128
        sky = water = None
        if seg is not None:
            lab = segment(seg, pano, R_lv, Bs_seg, SEG_PX, f_seg, W, H, dev)
            sky = np.isin(lab, list(SKY_IDS)); water = np.isin(lab, list(WATER_IDS))
            fl = np.isin(lab, list(FLOOR_IDS)) & (q_eq[..., 1] < -0.03)
            fmask = fl                                                    # the ground, the whole sphere round; the floor step's old mask is not joined to it — it was
                                                                          # drawn in whatever frame the levelling then had (a shore walk's was 31 degrees over)
            # THE SKY IS NO SURFACE: MoGe gives clouds and haze a distance of 15-25 m (its reach), which fused into a dome over an outdoor walk
            sky_up = sky & (q_eq[..., 1] > -0.02)
            if sky_up.mean() > 0.03:
                # OUTDOORS, THE HORIZON IS NOT A WALL: the haze and the sea at the horizon that the labeller missed still carry MoGe's farthest distance,
                # which fused into a rim of shards round an outdoor model; above the horizon, anything as far as the stop's farthest 1 % is let go
                far = float(np.percentile(rng[have], 99)) if have.any() else 0.0
                rim = (q_eq[..., 1] > 0.02) & have & (rng > 0.85 * far)
                rim |= (water if water is not None else False) & (q_eq[..., 1] > 0.0)
                sky_up = sky_up | rim
            rng = np.where(sky_up, 0.0, rng); have = have & ~sky_up
            rep["sky_pct"] = round(float(sky_up.mean() * 100), 1); rep["water_pct"] = round(float((water & have).mean() * 100), 1)
        h_depth, sure = floor_height(rng, q_eq, fmask)
        kept[i] = dict(rng=rng.astype(np.float16), spread=spread.astype(np.float16), have=have, fmask=fmask, water=water, h=h_depth, sure=sure, rep=rep, seconds=time.time() - ts)
        print(f"{p['id']}  views agree p90 {rep['pair_residual_p90']:.3f}  map/MoGe {rep['map_over_moge']}  floor {h_depth:.2f} m below the camera (sure {sure:.2f})"
              + (f"  sky {rep['sky_pct']}%  water {rep['water_pct']}%" if 'sky_pct' in rep else "") + f"  {time.time() - ts:.1f} s", flush=True)
    if not kept:
        raise SystemExit("no stop to work on")
    # the metre of the walk
    ratios = [k["rep"]["map_over_moge"] for k in kept.values() if k["rep"].get("map_over_moge")]
    G = 1.0
    if a.metre == "map" and ratios:
        G = float(np.median(ratios))
    stats["map_over_moge_median"] = round(float(np.median(ratios)), 3) if ratios else None
    stats["map_over_moge_p10_p90"] = [round(float(x), 3) for x in np.percentile(ratios, [10, 90])] if ratios else None
    stats["global_scale"] = G
    # the stick: one height for the walk (the person holding it does not grow); a stop whose floor is seen clearly is pulled onto it
    hs = np.array([k["h"] * G for k in kept.values()]); su = np.array([k["sure"] for k in kept.values()])
    good = su > 0.12
    h_med = float(np.median(hs[good])) if good.sum() >= 3 else float(np.median(hs))
    stats["stick_m"] = round(h_med, 3); stats["stick_spread_p10_p90"] = [round(float(x), 3) for x in np.percentile(hs[good] if good.any() else hs, [10, 90])]
    metre_words = ("the sparse map, x%.3f of MoGe" % G) if (a.metre == "map" and ratios) else "MoGe"
    print(f"metre: {metre_words}; the map says x{stats['map_over_moge_median']} of MoGe (p10-p90 {stats['map_over_moge_p10_p90']}).  stick height {h_med:.2f} m (p10-p90 {stats['stick_spread_p10_p90']})", flush=True)
    for i, k in kept.items():
        p = pts[i]
        c = 1.0
        if not a.no_stick and k["sure"] > 0.12 and 0.6 < k["h"] * G < 4.5:
            c = float(np.clip(h_med / (k["h"] * G), 0.8, 1.25))
        s_i = G * c
        rng = k["rng"].astype(np.float32) * s_i; have = k["have"]; fmask = k["fmask"]
        use_h = k["h"] * s_i
        dy = -q_eq[..., 1]
        plane = np.where(dy > 0.02, use_h / np.maximum(dy, 0.02), MAX_RANGE)
        snap = (fmask if fmask is not None else np.zeros((H, W), bool)) & (dy > 0.03) & (np.abs(dy * rng - use_h) < 0.12 + 0.03 * rng)
        rng = np.where(snap, plane, np.minimum(rng, plane * 1.02))
        water = k.get("water"); h_w = None
        if water is not None and (water & have & (dy > 0.05)).sum() > 3000:
            # WATER LIES FLAT: the sea and a pool are one level, and MoGe's guesses at a specular surface wander by metres — every water pixel goes onto
            # the plane of the water's own median drop below the camera (never above the floor plane: water is below the ground he walks on, or level with it)
            wsel = water & have & (dy > 0.05)
            h_w = float(np.median(dy[wsel] * rng[wsel]))
            h_w = max(h_w, use_h)
            rng = np.where(water & (dy > 0.03), h_w / np.maximum(dy, 0.03), rng)
        rng = np.where(have, np.clip(rng, D.DMIN, D.DMAX), 0.0).astype(np.float32)
        up = (q_eq[..., 1] > 0.6) & (rng > 0)
        roof = float(np.percentile(q_eq[..., 1][up] * rng[up], 90)) + use_h if up.sum() > 500 else None
        spread = k["spread"].astype(np.float32)
        conf = np.clip(255 * (1 - spread / 0.3), 0, 255)
        if water is not None:
            conf = np.where(water, conf * 0.6, conf)
        conf = conf.astype(np.uint8)
        np.savez_compressed(os.path.join(out, p["id"] + ".npz"), range=rng.astype(np.float16), conf=conf, cam_h=np.float32(use_h),
                            water=(np.packbits(water) if water is not None else np.zeros(0, np.uint8)), water_drop=np.float32(h_w if h_w is not None else 0.0))
        v8 = np.clip(np.round(np.log(np.maximum(rng, D.DMIN) / D.DMIN) / np.log(D.DMAX / D.DMIN) * 255), 0, 255).astype(np.uint8)
        img = np.zeros((H, W, 3), np.uint8); img[..., 2] = v8
        if fmask is not None:
            img[..., 1] = fmask.astype(np.uint8) * 255
        cv2.imwrite(os.path.join(out, f"eq_{p['id']}.png"), img)
        cam_h_map = P[i].get("cam_h")
        if not a.keep_cam_h:
            P[i]["cam_h"] = round(use_h, 2)
        st = dict(k["rep"], scale=round(s_i, 4), stick_fix=round(c, 3), cam_h_map=cam_h_map, cam_h_moge=round(k["h"], 2), floor_sure=round(k["sure"], 3), cam_h=round(use_h, 2),
                  roof_m=(round(roof, 2) if roof else None), covered=round(float(have.mean()), 3), seconds=round(k["seconds"], 1))
        stats["stops"][p["id"]] = st
        print(f"{p['id']}  scale x{s_i:.3f} (stick {c:.3f})  camera {use_h:.2f} m up  roof {roof if roof is None else round(roof, 2)} m", flush=True)
    # a map whose metre nobody confirmed (no accelerometer: a stitched video, a phone clip) is put into the depth's metre, one factor for the walk, so the stops
    # stand roughly where their depth says before the registration pulls them exactly there (the map's own metre on the demo walk wandered 0.25 - 12 against MoGe)
    if a.metre == "moge" and not trusted and ratios:
        fac = 1.0 / float(np.median(ratios))
        for nd in P:
            if nd.get("x") is not None:
                nd["x"] = round(nd["x"] * fac, 3); nd["y"] = round(nd["y"] * fac, 3)
                for L in nd.get("links", []):
                    L["d"] = round(L["d"] * fac, 2)
        poses["positions_rescaled"] = round(fac, 4); stats["positions_rescaled"] = round(fac, 4)
        print(f"the map's metre was not confirmed: its positions are scaled x{fac:.3f} into MoGe's metre", flush=True)
    poses["metre"] = a.metre; poses["depth_scale_over_map"] = round(1.0 / stats["map_over_moge_median"] * G, 4) if stats["map_over_moge_median"] else None
    poses["stick_m"] = stats["stick_m"]
    json.dump(stats, open(os.path.join(out, "stats.json"), "w"), indent=1)
    json.dump(poses, open(pf + ".tmp", "w"), indent=1)
    os.replace(pf + ".tmp", pf)                       # atomic: the early tour reads poses.json while this step works
    print(f"done -> {out} in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
