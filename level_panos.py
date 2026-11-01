"""Estimate each pano's true vertical (the stick leans, so panos are tilted).
THE CAMERA'S OWN SENSOR FIRST (2026-10-30): an .insv carries the X5's accelerometer and gyroscope (imu.py). Gravity in the sensor frame, read over
1.5 s either side of the stop with every sample turned by the gyroscope into the stop's own instant (so a walker's strides average out), through the
fixed mounting of the sensor in the camera body (MOUNT_ROTVEC, measured once on three of the owner's walks: 25 stops, median 2.9 degrees from the
vertical-edge vote; a stop he carried at 59 degrees came out level), IS the pano's up. No picture can argue with it. The two picture-based ways below
serve a video that carries no sensor (a stitched .mp4 from somewhere else) or a stop the sensor did not cover.
Two ways, and the pano says which:
  - THE HORIZON, where there is sky: UperNet (ADE20K) labels the pano's sky and its ground, and the up vector is the one that puts the sky above a level
    horizon and the ground below it. A beach has no vertical edge at all - the edge vote below hit its own 30 degree clamp on every stop of a shore walk
    and the whole model was built 31 degrees over (the sea stood as a wall, the heights meant nothing).
  - VERTICAL EDGES, indoors: vertical 3D lines are great circles through the up vector, so we vote for the up vector that the most edges agree on.
A vote that lands past SATURATED degrees is not believed (the camera's own stitch is already near level; a 20 degree lean of a stick is not a thing
a person does) - with no sky to read, such a pano is taken as level.
THE VERTICALS COME FIRST (2026-10-30 - the owner's road walks: "it weirdly rotates and rolls when using my finger"): the skyline over a street is roofs and
trees, not a horizon, and the horizon read tilted his stops 20-41 degrees while the vertical edges (poles, door frames, building corners) put them within
0.5-10 degrees of level, where the raw frames plainly are. So where the edge vote is believable it is the answer, and the skyline is read only where
there is no vertical to vote with (the beach, where the edge vote saturates) - choose_up.
Writes frames/up.json: one [x, y, z] per point (camera frame: x right, y up, z forward), and frames/level.json with how each was found.
Usage: python level_panos.py [--frames frames]
"""
import argparse, json, os
from concurrent.futures import ThreadPoolExecutor
import numpy as np, cv2, json
from imutil import imread, parallel_jobs

SEG_MODEL = "openmmlab/upernet-convnext-small"      # ADE20K, MIT weights: the floor step's model
SKY_IDS = {2}
GROUND_IDS = {3, 6, 9, 11, 13, 29, 46, 52, 21, 26, 60, 128}     # floor, road, grass, sidewalk, earth, field, sand, path, water, sea, river, lake
WATER_IDS = {21, 26, 60, 128}
SKY_MIN = 0.03                                      # the sky must cover this much of the pano before the horizon is read off it
SATURATED = 20.0                                    # an edge vote past this many degrees is not believed
SEG_W = 1024
HORIZON_MIN = 0.08                                  # the skyline must sit on the horizon over at least this share of the columns (a gully between dunes shows none)

def estimate_up(path, W=1600, max_tilt=30):
    img=imread(path,cv2.IMREAD_GRAYSCALE)
    img=cv2.resize(img,(W,W//2),interpolation=cv2.INTER_AREA).astype(np.float32)
    img=cv2.GaussianBlur(img,(0,0),1.2)
    h,w=img.shape
    gx=cv2.Sobel(img,cv2.CV_32F,1,0,ksize=3); gy=cv2.Sobel(img,cv2.CV_32F,0,1,ksize=3)
    mag=np.hypot(gx,gy)
    v,u=np.mgrid[0:h,0:w].astype(np.float32)
    th=(u+0.5)/w*2*np.pi-np.pi; ph=np.pi/2-(v+0.5)/h*np.pi
    ok=(mag>np.percentile(mag,93))&(np.abs(ph)<np.radians(62))&(v<0.78*h)
    th,ph,gx,gy,mag=th[ok],ph[ok],gx[ok],gy[ok],mag[ok]
    p=np.stack([np.cos(ph)*np.sin(th),np.sin(ph),np.cos(ph)*np.cos(th)],1)
    # unit vectors along +u (east) and up-in-image(north) at each point
    e=np.stack([np.cos(th),np.zeros_like(th),-np.sin(th)],1)
    nn=np.stack([-np.sin(ph)*np.sin(th),np.cos(ph),-np.sin(ph)*np.cos(th)],1)
    # edge tangent is perpendicular to the gradient; gradient in (east, south) pixel axes -> (east, -north)
    gE=gx; gN=-gy
    tE,tN=-gN,gE
    tau=e*tE[:,None]+nn*tN[:,None]
    tau/= (np.linalg.norm(tau,axis=1,keepdims=True)+1e-9)
    n=np.cross(p,tau); n/=(np.linalg.norm(n,axis=1,keepdims=True)+1e-9)
    wgt=np.cos(ph)*np.minimum(mag,np.percentile(mag,99))
    # candidate up vectors within cone
    best=None
    def score(U):
        d=n@U.T                      # (N,K)
        return (wgt[:,None]*np.exp(-(d/0.035)**2)).sum(0)
    cands=[]
    for tx in np.arange(-max_tilt,max_tilt+0.1,2):
        for tz in np.arange(-max_tilt,max_tilt+0.1,2):
            if np.hypot(tx,tz)>max_tilt: continue
            cands.append((tx,tz))
    def mk(tx,tz):
        a,b=np.radians(tx),np.radians(tz)
        u=np.array([np.sin(a),np.cos(a)*np.cos(b),np.sin(b)*np.cos(a)]); return u/np.linalg.norm(u)
    U=np.array([mk(*c) for c in cands]); s=score(U); k=int(np.argmax(s)); c0=cands[k]
    # refine
    cands2=[(c0[0]+dx,c0[1]+dz) for dx in np.arange(-2,2.01,0.5) for dz in np.arange(-2,2.01,0.5)]
    U2=np.array([mk(*c) for c in cands2]); s2=score(U2); k2=int(np.argmax(s2)); u=U2[k2]
    conf=float(s2[k2]/np.median(s))
    return u,conf



AGREE_DEG = 6.0                                     # the skyline agreeing with the verticals within this is said so; further off it is named for what it is
MOUNT_ROTVEC = [-0.0331, 0.0573, -1.5663]           # sensor -> pano frame (x right, y up, z forward): the X5's sensor sits a quarter turn about the lens axis
IMU_CLOCK_OFFSET_S = 1.0                            # the sensor's clock runs this far ahead of the video's (imu_scale.py solved 1.00 s on every walk so far)
IMU_HALF_S = 1.5                                    # gravity is read over this much either side of the stop (a stride or two, so walking averages out)
IMU_RATE = 200                                      # the sensor's samples are averaged down to this rate before the gyroscope turns them


def up_from_imu(d, ts, half=IMU_HALF_S, mount=None):
    """The pano's up at sensor time ts from the sensor stream d (imu.read_imu's dict), or None when the stream does not cover it: the specific force of
    every sample within +-half s, turned by the gyroscope into the frame at ts, averaged (walking accelerates both ways; gravity does not), through the
    mount into the pano frame."""
    from scipy.spatial.transform import Rotation as Rot
    w = np.nonzero((d["t"] > ts - half) & (d["t"] < ts + half))[0]
    if len(w) < 20:
        return None
    k = max(1, int(round(d["rate"] / IMU_RATE)))
    w = w[: (len(w) // k) * k]
    if len(w) < 20:
        return None
    t = d["t"][w].reshape(-1, k).mean(1); f = d["acc"][w].reshape(-1, k, 3).mean(1); g = d["gyro"][w].reshape(-1, k, 3).mean(1)
    k0 = int(np.argmin(np.abs(t - ts)))
    dt = np.diff(t, prepend=t[0])
    inc = Rot.from_rotvec(g * dt[:, None]).as_matrix()
    Rm = np.empty_like(inc); Rm[0] = np.eye(3)
    for i in range(1, len(t)):
        Rm[i] = Rm[i - 1] @ inc[i]
    fs = np.einsum("ij,njk,nk->ni", Rm[k0].T, Rm, f)
    m = fs.mean(0)
    if not np.isfinite(m).all() or np.linalg.norm(m) < 1e-6:
        return None
    M = Rot.from_rotvec(MOUNT_ROTVEC if mount is None else mount).as_matrix()
    u = M @ (m / np.linalg.norm(m))
    return u / np.linalg.norm(u)


def imu_ups(video, pts, log=print):
    """[up or None per stop] from the video's own sensor, or None when the video carries none."""
    if not video or not os.path.exists(video):
        return None
    try:
        import imu as IMU
        d = IMU.read_imu(video)
    except Exception as e:                                             # noqa: BLE001
        log("no motion sensor in the video (%s): the pictures level themselves" % e); return None
    t0 = float(d["t"][0])
    out = [up_from_imu(d, float(p["timestamp"]) + IMU_CLOCK_OFFSET_S + t0) for p in pts]
    log("the camera's sensor levels %d of %d stops" % (sum(u is not None for u in out), len(pts)))
    return out


def tilt_of(u):
    return float(np.degrees(np.arccos(np.clip(float(u[1]) / (float(np.linalg.norm(u)) or 1.0), -1, 1))))


def choose_up(u_e, c_e, hz, saturated=SATURATED):
    """Which up a pano gets, as (u, conf, way): the vertical edges' vote (u_e, c_e) when it is believable (under `saturated` degrees), the skyline's
    (hz = (u, score, sky share) or None) only when the edges saturated - a beach has no vertical - and level when neither can be read."""
    tilt_e = tilt_of(u_e)
    if tilt_e < saturated:
        if hz is None:
            return u_e, c_e, "edges"
        d = float(np.degrees(np.arccos(np.clip(float(np.dot(u_e, hz[0])) / (np.linalg.norm(u_e) * np.linalg.norm(hz[0])), -1, 1))))
        if d <= AGREE_DEG:
            return u_e, c_e, "edges (the skyline agrees within %.0f deg)" % d
        return u_e, c_e, "edges (the skyline wanted %.0f deg - roofs and trees are not a horizon)" % tilt_of(hz[0])
    if hz is not None:
        return hz[0], hz[1], "horizon (sky %.0f%%; no vertical to vote with)" % (hz[2] * 100)
    return np.array([0.0, 1.0, 0.0]), 0.0, "level (the edge vote wanted %.0f deg; no sky to read, not believed)" % tilt_e


def mk_up(tx, tz):
    a, b = np.radians(tx), np.radians(tz)
    u = np.array([np.sin(a), np.cos(a) * np.cos(b), np.sin(b) * np.cos(a)]); return u / np.linalg.norm(u)


def load_seg():
    """The labeller, or None when torch/transformers are not here (the edge vote alone then)."""
    try:
        import torch
        from transformers import AutoImageProcessor, UperNetForSemanticSegmentation
    except Exception as e:                                             # noqa: BLE001
        print("no labeller (%s) - vertical edges only" % e, flush=True); return None
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    return (AutoImageProcessor.from_pretrained(SEG_MODEL), UperNetForSemanticSegmentation.from_pretrained(SEG_MODEL).to(dev).eval(), dev)


def labels_eq(seg, path, W=SEG_W):
    """ADE20K labels over the raw equirect (W x W/2): labelled as it stands and rolled half a turn, each pixel taking the pass it is nearer the middle of
    (the seam is at the picture's edge in one pass and in its middle in the other)."""
    import torch
    from PIL import Image
    proc, model, dev = seg
    img = imread(path); img = cv2.resize(img, (W, W // 2), interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    out = []
    for roll in (0, W // 2):
        pic = np.roll(rgb, roll, axis=1)
        inputs = proc(images=Image.fromarray(pic), return_tensors="pt").to(dev)
        with torch.no_grad():
            lg = model(**inputs).logits
        lab = torch.nn.functional.interpolate(lg, size=(W // 2, W), mode="bilinear", align_corners=False)[0].argmax(0).cpu().numpy().astype(np.int16)
        out.append(np.roll(lab, -roll, axis=1))
    u = np.arange(W)
    mid = np.abs(u - W / 2) < W / 4                                     # the middle half of the picture is pass 0's; the edges are pass 1's
    return np.where(mid[None, :], out[0], out[1])


def horizon_up(lab, max_tilt=45.0):
    """The up vector that puts the SKYLINE on or above a level horizon: in every column the lowest sky pixel (scanning down from the zenith) is where the
    sky meets something - the sea, a hill, a building. Nothing stands above the sky, so no skyline point lies below the horizon, and over open water the
    skyline IS the horizon. The score counts skyline points on the horizon and pays dearly for any below it; the ground labels add a weak term and a very
    weak preference for a small tilt breaks a tie (a stop with no open skyline). Returns (u, score, sky share) or None with too little sky."""
    H, W = lab.shape
    sky = np.isin(lab, list(SKY_IDS)); gnd = np.isin(lab, list(GROUND_IDS))
    share = float(sky.mean())
    if share < SKY_MIN:
        return None
    # the skyline: per column, the first row from the top where the sky stops for three rows running
    notsky = ~sky
    run = notsky[:-2] & notsky[1:-1] & notsky[2:]                       # (H-2, W): a non-sky run of 3 starting here
    has = run.any(0) & sky[0]                                            # the column's top is sky and the sky ends somewhere
    first = np.where(has, run.argmax(0), -1)
    cols = np.nonzero(has)[0]
    if len(cols) < W // 4:
        return None
    th = (cols + 0.5) / W * 2 * np.pi - np.pi; ph = np.pi / 2 - (first[cols] + 0.5) / H * np.pi
    pk = np.stack([np.cos(ph) * np.sin(th), np.sin(ph), np.cos(ph) * np.cos(th)], -1)
    v, u = np.mgrid[0:H, 0:W].astype(np.float32)
    thg = (u + 0.5) / W * 2 * np.pi - np.pi; phg = np.pi / 2 - (v + 0.5) / H * np.pi
    band = gnd & (np.abs(phg) < np.radians(max_tilt + 15))
    pg = np.stack([np.cos(phg[band]) * np.sin(thg[band]), np.sin(phg[band]), np.cos(phg[band]) * np.cos(thg[band])], -1)
    wg = np.cos(phg[band])
    rs = np.random.default_rng(0)
    if len(pg) > 30000:
        k = rs.choice(len(pg), 30000, replace=False); pg, wg = pg[k], wg[k]
    m = np.sin(np.radians(1.5))
    def score(U):
        e = pk @ U.T                                                     # (K, C) sine of each skyline point's elevation
        on = np.exp(-(e / m) ** 2).sum(0) / len(pk)                      # skyline on the horizon
        under = (e < -m).sum(0) / len(pk)                                # skyline below it - impossible
        gterm = ((wg[:, None] * np.clip(-(pg @ U.T) / 0.03, -1, 1)).sum(0) / max(wg.sum(), 1e-6)) if len(pg) else 0.0
        tilt = np.degrees(np.arccos(np.clip(U[:, 1], -1, 1)))
        return on - 4.0 * under + 0.15 * gterm - 1e-4 * tilt
    cands = [(tx, tz) for tx in np.arange(-max_tilt, max_tilt + 0.1, 1.5) for tz in np.arange(-max_tilt, max_tilt + 0.1, 1.5) if np.hypot(tx, tz) <= max_tilt]
    U = np.array([mk_up(*c) for c in cands]); sc = score(U); k = int(np.argmax(sc)); c0 = cands[k]
    cands2 = [(c0[0] + dx, c0[1] + dz) for dx in np.arange(-1.5, 1.51, 0.2) for dz in np.arange(-1.5, 1.51, 0.2)]
    U2 = np.array([mk_up(*c) for c in cands2]); s2 = score(U2); k2 = int(np.argmax(s2))
    return U2[k2], float(s2[k2]), share


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--frames", default="frames"); ap.add_argument("--video", default=None, help="the walk's own .insv: its sensor levels the stops")
    a = ap.parse_args()
    pts = json.load(open(os.path.join(a.frames, "points.json")))
    paths = [os.path.join(a.frames, "points", p["filename"]) for p in pts]
    with ThreadPoolExecutor(max_workers=min(parallel_jobs(1.0, 8), os.cpu_count() or 1)) as ex:     # panos are independent; numpy/cv2 release the GIL
        res = list(ex.map(estimate_up, paths))
    sensor = imu_ups(a.video, pts) or [None] * len(pts)
    seg = load_seg() if any(s is None for s in sensor) else None
    ups, how = [], []
    for p, path, (u, c), su in zip(pts, paths, res, sensor):
        if su is not None:
            te = tilt_of(u)
            d = float(np.degrees(np.arccos(np.clip(float(np.dot(su, u)), -1, 1))))
            note = "" if te >= SATURATED else (" (the vertical edges agree within %.0f deg)" % d if d <= AGREE_DEG else " (the vertical edges wanted %.0f deg, %.0f off)" % (te, d))
            ups.append([round(float(x), 5) for x in su]); how.append(dict(id=p["id"], tilt_deg=round(tilt_of(su), 2), way="gyro+accelerometer" + note, conf=1.0))
            print(p["id"], "tilt %.1f deg" % tilt_of(su), "by the camera's sensor" + note, flush=True)
            continue
        hz = None
        if seg is not None and tilt_of(u) >= SATURATED:               # the skyline is read only where the verticals could not vote (2026-10-30)
            try:
                hz = horizon_up(labels_eq(seg, path))
            except Exception as e:                                     # noqa: BLE001
                print(p["id"], "horizon read failed:", e, flush=True)
        if hz is not None and hz[1] < HORIZON_MIN:
            print(p["id"], "skyline read but no open horizon on it (score %.2f) - not used" % hz[1], flush=True); hz = None
        u, c, way = choose_up(u, c, hz)
        tilt = tilt_of(u)
        ups.append([round(float(x), 5) for x in u]); how.append(dict(id=p["id"], tilt_deg=round(tilt, 2), way=way, conf=round(float(c), 2)))
        print(p["id"], "tilt %.1f deg" % tilt, "conf %.1f" % c, "by", way, flush=True)
    json.dump(ups, open(os.path.join(a.frames, "up.json"), "w"))
    json.dump(how, open(os.path.join(a.frames, "level.json"), "w"), indent=0)
    print("wrote up.json")


if __name__ == "__main__":
    main()
