"""Build a phone-ready tour folder from frames/points/*.jpg + frames/points.json.

Needs frames/hops.json from hop_geometry.py (run that first).
Usage:  python build_tour.py [--frames frames] [--out tour] [--width 4096] [--nadir-mode fill|cap|none]

Output (tour/ is gitignored, it holds footage):
  tour/index.html     viewer (copied from viewer/index.html)
  tour/tour.json      points: id, x, y, heading, neighbours, file, thumb
  tour/panos/*.jpg    equirect panos resized for phones (4096 wide max texture on most phones)
  tour/thumbs/*.jpg   small previews for the reel
Serve it with:  python -m http.server 8000 --directory tour
"""
import argparse, json, math, os, shutil, sys, time
from concurrent.futures import ThreadPoolExecutor
import cv2
import numpy as np
from nadir import fill_nadir
from imutil import imread, imwrite, parallel_jobs

HERE = os.path.dirname(os.path.abspath(__file__))
# progressive + optimised Huffman tables: ~4% smaller at identical quality; q86 / q80 are visually lossless against q90 / q82 (45 / 44 dB PSNR)
JPEG_HI = [cv2.IMWRITE_JPEG_QUALITY, 86, cv2.IMWRITE_JPEG_OPTIMIZE, 1, cv2.IMWRITE_JPEG_PROGRESSIVE, 1]
JPEG_LO = [cv2.IMWRITE_JPEG_QUALITY, 80, cv2.IMWRITE_JPEG_OPTIMIZE, 1, cv2.IMWRITE_JPEG_PROGRESSIVE, 1]
JPEG_TILE = [cv2.IMWRITE_JPEG_QUALITY, 80, cv2.IMWRITE_JPEG_OPTIMIZE, 1, cv2.IMWRITE_JPEG_PROGRESSIVE, 1]
JPEG_XS = [cv2.IMWRITE_JPEG_QUALITY, 60, cv2.IMWRITE_JPEG_OPTIMIZE, 1, cv2.IMWRITE_JPEG_PROGRESSIVE, 1]
XS_WIDTH = 1024                                      # ~65 KB: the first thing a visitor on a slow connection sees, sharpened by the 4K copy a moment later
TILE_COLS, TILE_ROWS, TILE_PAD = 6, 3, 8            # 60 x 60 degrees each; a phone loads only the 3-4 tiles it is looking at


def tile_name(pid, r, c):
    return f"{pid}_{r}{c}.jpg"


def cut_tiles(img, cols=TILE_COLS, rows=TILE_ROWS, pad=TILE_PAD):
    """The full-detail panorama as rows x cols tiles, each with `pad` extra pixels on every side taken from the neighbouring tile
    (wrapping round the back, repeating the edge at the poles), so the viewer can sample inside the padding and no seam shows
    between tiles. Returns {(r, c): image}."""
    h, w = img.shape[:2]
    th, tw = h // rows, w // cols
    big = np.concatenate([img[:, -pad:], img, img[:, :pad]], axis=1)                         # wrap left/right
    big = np.concatenate([np.repeat(big[:1], pad, 0), big, np.repeat(big[-1:], pad, 0)], axis=0)   # repeat the edge rows at the poles
    return {(r, c): big[r * th: (r + 1) * th + 2 * pad, c * tw: (c + 1) * tw + 2 * pad] for r in range(rows) for c in range(cols)}


def nadir_cap(img, frac):
    """Hide the stick/body at the bottom of the equirect with a soft dark cap."""
    if frac <= 0:
        return img
    h, w = img.shape[:2]
    start = int(h * (1 - frac))
    feather = max(1, int(h * 0.05))
    out = img.astype(np.float32)
    cap = np.array([26, 26, 26], np.float32)  # BGR #1a1a1a
    for row in range(start - feather, h):
        a = 1.0 if row >= start else (row - (start - feather)) / feather
        out[row] = out[row] * (1 - a) + cap * a
    return out.astype(np.uint8)


def chain_map(points, hops):
    """Rough top-down layout for the minimap only: chain the measured turns and distances."""
    H, x, y = [0.0], [0.0], [0.0]
    for i, h in enumerate(hops):
        d = h.get("dist_m") or max(1.0, math.hypot(points[i + 1]["x"] - points[i]["x"], points[i + 1]["y"] - points[i]["y"]))
        d = min(max(d, 0.8), 12)
        if not h.get("ok"):
            H.append(H[-1]); x.append(x[-1]); y.append(y[-1] + d); continue
        b = math.radians(H[i] + h["az_fwd"])
        x.append(x[-1] + d * math.sin(b)); y.append(y[-1] + d * math.cos(b))
        H.append(H[i] + h["yaw"])
    return H, x, y


MAX_RINGS = 6


def _el(el, d):
    """Degrees below the horizon at which the floor under the other stop appears. Measured; fallback assumes 2 m up."""
    if el is None:
        el = math.degrees(math.atan2(2.0, d))
    return round(min(max(el, 6.0), 58.0), 1)


def build_links(pairs, n):
    """Rings for each stop. A pair is used if it matched well AND agrees with the chain of consecutive hops.
    az = where the other stop is in THIS pano (deg clockwise from centre); d = metres (rough);
    turn = change of view direction on arrival (lon_new = lon_old + turn)."""
    out = [[] for _ in range(n)]
    for r in pairs:
        i, j = r["i"], r["j"]
        near = j == i + 1
        good = r["ok"] and r.get("consistent", True) and (near or (r["inliers"] >= 50 and r["inliers"] >= 0.4 * r["matches"]))
        if not good:
            continue
        if not near and r.get("el_ij") is None:
            continue          # a far link without a measured floor line would float; skip it
        d = r.get("dist_m") or 3.0
        d = round(min(max(d, 1.6), 5.5), 2)
        out[i].append({"to": j, "az": r["az_ij"], "turn": -r["yaw"], "d": d, "el": _el(r.get("el_ij"), d), "near": near})
        out[j].append({"to": i, "az": r["az_ji"], "turn": r["yaw"], "d": d, "el": _el(r.get("el_ji"), d), "near": near})
    for i in range(n):
        out[i].sort(key=lambda L: (not L["near"], L["d"]))
        out[i] = out[i][:MAX_RINGS]
    return out


THREE_URL = "https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"      # MIT licensed, the licence header stays in the file


def ensure_three(out, frames):
    """Put three.js next to index.html so the tour also works offline / when the CDN is down. Downloaded once, cached in frames/."""
    dst, cache = os.path.join(out, "three.min.js"), os.path.join(frames, "three.min.js")
    if not os.path.exists(cache):
        try:
            import urllib.request
            data = urllib.request.urlopen(THREE_URL, timeout=30).read()
            if len(data) > 100000 and b"Three.js Authors" in data[:400]:
                open(cache, "wb").write(data)
        except Exception as e:
            print("note: could not download three.js (" + str(e) + "); the tour will load it from the CDN instead")
    if os.path.exists(cache):
        shutil.copy(cache, dst)


def prune(out, keep):
    """Remove pictures from earlier builds that are not part of this tour (fewer stops, or a different walk): they would still be
    uploaded and downloadable if the folder is published. keep: {sub-folder: set of file names that belong}. Returns how many were removed."""
    n = 0
    for sub, names in keep.items():
        d = os.path.join(out, sub)
        if os.path.isdir(d):
            for f in os.listdir(d):
                if f not in names and f.lower().endswith((".jpg", ".png")):
                    os.remove(os.path.join(d, f)); n += 1
    return n


def walked_path(frames, pts, MX, MY, ZS, CAMH, placed, window=40.0, step=0.2, log=print):
    """THE PATH HE WALKED (2026-10-30 — the owner, on the Vitronic tour: "you don't show the path that I walked… maybe that would be a useful
    toggle"). vio_map.py's 1 Hz chain (frames/vio/poses.npz) has a camera position for every second of the walk; the tour has map positions for
    the stops only. Each chain position is laid onto the map the way map_stops.vio_place lays a stop: a similarity fit (Umeyama, rotation +
    scale + shift) over the placed stops within +-`window` seconds of it that are in both (the chain drifts in scale, so the fit is local; at
    least three, the whole walk otherwise), in three dimensions (map x, y and the camera's height over the walk's zero). The map's x and y of
    every fitted position, in time order, thinned to `step` metres. None when there is no chain or too few stops to fit — the viewer then draws
    the stop chain as before. A drawing on the minimap, never a place one can walk to."""
    vio = os.path.join(frames, "vio", "poses.npz")
    if not os.path.exists(vio):
        return None
    try:
        V = np.load(vio)
        names = [str(n) for n in V["names"]]; Cv = np.asarray(V["C"], float); T = np.asarray(V["t"], float)
    except Exception as e:  # noqa: BLE001 — a bad chain is no path
        log(f"motion track unreadable ({e}): no walked path"); return None
    at = {nm[2:-4]: i for i, nm in enumerate(names) if nm.startswith("s_")}
    ks = [k for k in placed if pts[k]["id"] in at]
    if len(ks) < 3:
        return None
    A_all = np.array([Cv[at[pts[k]["id"]]] for k in ks])
    B_all = np.array([[MX[k], MY[k], float(ZS[k] or 0.0) + float(CAMH[k] or 2.0)] for k in ks])
    TS = np.array([float(pts[k]["timestamp"]) for k in ks])

    def fit(sel):
        A, B = A_all[sel], B_all[sel]
        ma, mb = A.mean(0), B.mean(0); A0, B0 = A - ma, B - mb
        H = B0.T @ A0 / len(A)
        U, S, Vt = np.linalg.svd(H)
        D = np.eye(3)
        if np.linalg.det(U @ Vt) < 0:
            D[2, 2] = -1
        Q = U @ D @ Vt
        var = (A0 ** 2).sum() / len(A)
        s = float((S * np.diag(D)).sum() / var) if var > 1e-12 else 0.0
        t = mb - s * Q @ ma
        res = float(np.sqrt(((B - (s * (Q @ A.T).T + t)) ** 2).sum(1).mean())) if s > 0 else 1e9
        flat = S[0] > 0 and S[1] / S[0] < 0.02                 # the stops lie on one line: the fit may swing about it
        return s, Q, t, res, flat

    whole = fit(np.arange(len(ks)))
    if not (np.isfinite(whole[0]) and whole[0] > 0):
        return None
    Qw = whole[1]

    def fit_scale(sel):
        """scale + shift only, the whole walk's rotation held: for a stretch whose stops lie on one line (a corridor, a road), where a free rotation would swing"""
        A = A_all[sel] @ Qw.T; B = B_all[sel]
        ma, mb = A.mean(0), B.mean(0); A0, B0 = A - ma, B - mb
        den = (A0 ** 2).sum()
        s = float((A0 * B0).sum() / den) if den > 1e-12 else 0.0
        t = mb - s * ma
        res = float(np.sqrt(((B - (s * A + t)) ** 2).sum(1).mean())) if s > 0 else 1e9
        return s, Qw, t, res, False

    good = lambda f: np.isfinite(f[0]) and f[0] > 0 and not f[4]
    out = []; last = None; res = []
    K = min(len(ks), 5)                                                    # the chain drifts in scale: each position is laid by its nearest few stops
    for i in np.argsort(T):
        near = np.argsort(np.abs(TS - T[i]))[:K]
        f = fit(near) if K >= 3 else whole
        if not good(f):
            f = fit_scale(near)                                            # on one line a free rotation may swing: the walk's rotation, this stretch's scale
        if not good(f) or f[3] > max(1.5, whole[3]):
            f = whole
        res.append(f[3])
        s, Q, t = f[0], f[1], f[2]
        p = s * Q @ Cv[i] + t
        if not np.all(np.isfinite(p)):
            continue
        if last is None or math.hypot(p[0] - last[0], p[1] - last[1]) >= step:
            out.append([round(float(p[0]), 2), round(float(p[1]), 2)]); last = p
    if len(out) < 2:
        return None
    log(f"walked path: {len(out)} points from {len(T)} chain positions, fit over {len(ks)} stops (residual {float(np.median(res)):.2f} m local, {whole[3]:.2f} m whole walk)")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", default="frames")       # relative to where it runs: tour.py runs it inside the walk's work folder
    ap.add_argument("--out", default="tour")
    ap.add_argument("--width", type=int, default=4096, help="quick-load copy (panos_lo)")
    ap.add_argument("--hi", type=int, default=8192, help="full-detail copy (panos), capped at the source width")
    ap.add_argument("--tiles", choices=["on", "off"], default="on", help="also write the full-detail picture as 18 tiles, so phones load only what they look at")
    ap.add_argument("--nadir", type=float, default=0.16, help="height of the dark cap (only with --nadir-mode cap)")
    ap.add_argument("--nadir-mode", choices=["fill", "cap", "none"], default="none", help="old fills for the bottom of the picture; the capturer step (remove_capturer.py) has already rebuilt it, so the default is none")
    ap.add_argument("--early", action="store_true", help="the first tour, made before the sphere depth, the registration and the dollhouse: tour.json says partial (build_early.py; the viewer shows no dollhouse or plan)")
    ap.add_argument("--title", default="Walkthrough", help="name shown in the viewer and the browser tab")
    ap.add_argument("--expose", choices=["on", "off"], default="on", help="match brightness/colour between neighbouring stops (exposure.py)")
    a = ap.parse_args()

    for need in ("points.json", "hops.json"):
        if not os.path.exists(os.path.join(a.frames, need)):
            sys.exit(f"{os.path.join(a.frames, need)} is missing: the earlier steps have not run. Use  python tour.py <your video>  (it runs every step in order).")
    pts = json.load(open(os.path.join(a.frames, "points.json")))
    # people: nothing goes into the tour unless privacy_gate.py cleared it (or you said keep). Not optional: no flag turns this off.
    pvf = os.path.join(a.frames, "privacy.json")
    if not os.path.exists(pvf):
        sys.exit(f"{pvf} is missing: the person-removal and privacy steps have not run. Use  python tour.py <your video>  (it runs every step in order), "
                 "or run  python remove_capturer.py  and  python privacy_gate.py  yourself.")
    PV = json.load(open(pvf))["stops"]
    safe_dir = os.path.join(a.frames, "safe")
    KEEP = [i for i, p in enumerate(pts) if os.path.exists(os.path.join(safe_dir, p["filename"]))
            and not (PV.get(p["id"], {"held": True})["held"] and not PV.get(p["id"], {}).get("kept"))]
    if not KEEP:
        try:                                                  # no tour, but the review is waiting: a manifest listing the held stops lets an app offer it
            import manifest
            manifest.write_all_held(a.frames, a.out, a.title)
        except Exception as e:
            print(f"note: manifest.json was not written ({e})")
        sys.exit(f"every stop was held back by the privacy check: nothing to build. Open {os.path.join(a.frames, 'review', 'review.html')}.")
    NEWIDX = {i: n for n, i in enumerate(KEEP)}
    held = [pts[i]["id"] for i in range(len(pts)) if i not in NEWIDX]
    if held:
        print(f"HELD BACK (not in the tour): {', '.join(held)}.  Review them: {os.path.join(a.frames, 'review', 'review.html')}")
    hj = json.load(open(os.path.join(a.frames, "hops.json")))
    hops, pairs = hj["hops"], hj["pairs"]
    LINKS = build_links(pairs, len(pts))
    MH, MX, MY = chain_map(pts, hops)
    pf = os.path.join(a.frames, "poses.json")
    if os.path.exists(pf):          # one shared map from map_stops.py: every visible stop, exact minimap
        P = json.load(open(pf))
        CAMH = [None] * len(pts); ZS = [0.0] * len(pts); CHK = [None] * len(pts)
        for k, nd in enumerate(P["nodes"]):
            if nd["x"] is None:
                continue
            MX[k], MY[k], MH[k] = nd["x"], nd["y"], nd["H"]
            CHK[k] = nd.get("check")                                                    # how well its heading agrees with its own pictures (heading_check.py)
            CAMH[k] = nd.get("cam_h"); ZS[k] = round(float(nd.get("z") or 0.0), 3)        # the stop's ground above the walk's zero (register_stops.py)
            LINKS[k] = sorted(nd["links"], key=lambda L: L["d"])
        print("using poses.json (shared map):", sum(len(L) for L in LINKS), "rings")
        # AN UNPLACED STOP IS NOT A PLACE IN THE WALK (2026-10-28, a road walk the map placed 20 of 25 stops of): a stop with no map
        # position used to take the per-hop chain's guess — a line of fake places the visitor glides to — so it is left out of the tour
        # and said in words; the manifest lists the stops that ARE in it
        unplaced = [i for i in KEEP if P["nodes"][i].get("x") is None]
        if unplaced:
            KEEP = [i for i in KEEP if i not in set(unplaced)]
            NEWIDX = {i: n for n, i in enumerate(KEEP)}
            print(f"NOT ON THE MAP (left out of the tour): {', '.join(pts[i]['id'] for i in unplaced)}")
            if not KEEP:
                sys.exit("no stop could be placed on the map: nothing to build. Record again with a stop every 1.5-2 m.")
        DEPTH = {}
        os.makedirs(os.path.join(a.out, "depth"), exist_ok=True)
        for nd in P["nodes"]:
            src = os.path.join(a.frames, "depth_sphere", f"eq_{nd['id']}.png")            # the full-sphere metric depth (depth_sphere.py) when there is one
            if not os.path.exists(src):
                src = os.path.join(a.frames, "depth_mvs", f"eq_{nd['id']}.png")           # else the multi-view anchored depth (mvs.py + depth_align.py)
            if not os.path.exists(src):
                src = os.path.join(a.frames, "depth", f"eq_{nd['id']}.png")
            if os.path.exists(src):
                shutil.copy(src, os.path.join(a.out, "depth", nd["id"] + ".png")); DEPTH[nd["id"]] = "depth/" + nd["id"] + ".png"
    else:
        DEPTH = {}; CAMH = [None] * len(pts); ZS = [0.0] * len(pts); CHK = [None] * len(pts)
        print("no poses.json: using per-hop links (run map_stops.py for the full map)")
    upf = os.path.join(a.frames, "up.json")
    UPS = json.load(open(upf)) if os.path.exists(upf) else [[0, 1, 0]] * len(pts)
    GAINS = None
    if a.expose == "on" and os.path.exists(os.path.join(a.frames, "poses_map.json")):
        import exposure
        GAINS, rep = exposure.solve_gains(a.frames)       # (n, 3) BGR gains from the floor both stops see
        json.dump(rep, open(os.path.join(a.frames, "exposure.json"), "w"), indent=1)
    VER = "?v=%d" % time.time()          # new picture URLs on every build: a re-uploaded tour never mixes with cached copies of the old one
    os.makedirs(os.path.join(a.out, "panos"), exist_ok=True)
    os.makedirs(os.path.join(a.out, "panos_lo"), exist_ok=True)
    os.makedirs(os.path.join(a.out, "thumbs"), exist_ok=True)
    os.makedirs(os.path.join(a.out, "panos_xs"), exist_ok=True)
    if a.tiles == "on":
        os.makedirs(os.path.join(a.out, "tiles"), exist_ok=True)

    def build_one(i):
        p = pts[i]
        src = os.path.join(safe_dir, p["filename"])             # the cleaned picture: capturer removed, faces blurred
        img = imread(src)
        if img is None:
            raise SystemExit(f"missing {src}")
        if a.nadir_mode == "fill":
            rd = lambda suffix: cv2.imread(os.path.join(a.frames, "depth", f"{p['id']}_{suffix}.png"), cv2.IMREAD_GRAYSCALE)
            img = fill_nadir(img, person=rd("n"), floor=rd("nf"))
        else:
            img = nadir_cap(img, a.nadir if a.nadir_mode == "cap" else 0)
        if GAINS is not None:
            ch = cv2.split(img); x = np.arange(256, dtype=np.float32)
            img = cv2.merge([cv2.LUT(c, np.clip(x * GAINS[i][k], 0, 255).astype(np.uint8)) for k, c in enumerate(ch)])
        wh = min(a.hi, img.shape[1])
        hi = img if wh == img.shape[1] else cv2.resize(img, (wh, wh // 2), interpolation=cv2.INTER_AREA)
        imwrite(os.path.join(a.out, "panos", p["filename"]), hi, JPEG_HI)
        if a.tiles == "on":
            for (r, c), t in cut_tiles(hi).items():
                imwrite(os.path.join(a.out, "tiles", tile_name(p["id"], r, c)), np.ascontiguousarray(t), JPEG_TILE)
        w = min(a.width, img.shape[1])
        img = cv2.resize(img, (w, w // 2), interpolation=cv2.INTER_AREA)
        imwrite(os.path.join(a.out, "panos_lo", p["filename"]), img, JPEG_LO)
        imwrite(os.path.join(a.out, "panos_xs", p["filename"]), cv2.resize(img, (XS_WIDTH, XS_WIDTH // 2), interpolation=cv2.INTER_AREA), JPEG_XS)
        # thumb: the forward-facing middle half of the pano
        hh, ww = img.shape[:2]
        crop = img[int(hh * 0.25):int(hh * 0.75), int(ww * 0.375):int(ww * 0.625)]
        imwrite(os.path.join(a.out, "thumbs", p["filename"]),
                    cv2.resize(crop, (240, 240), interpolation=cv2.INTER_AREA), [cv2.IMWRITE_JPEG_QUALITY, 75])
        node = {
            "id": p["id"], "t": p.get("timestamp"),
            "x": round(MX[i], 2), "y": round(MY[i], 2), "H": round(MH[i], 1),
            "f": "panos_lo/" + p["filename"] + VER, "fx": "panos_xs/" + p["filename"] + VER, "fh": "panos/" + p["filename"] + VER, "th": "thumbs/" + p["filename"] + VER,
            **({"tl": "tiles/" + p["id"] + "_{r}{c}.jpg" + VER} if a.tiles == "on" else {}),
            **({"hc": CHK[i]} if CHK[i] else {}),
            "up": UPS[i], "links": [dict(L, to=NEWIDX[L["to"]]) for L in LINKS[i] if L["to"] in NEWIDX], "dm": (DEPTH[p["id"]] + VER) if DEPTH.get(p["id"]) else None, "h": CAMH[i],
            "z": ZS[i],
        }
        if GAINS is not None:
            node["gain"] = [round(float(g), 3) for g in GAINS[i][::-1]]       # R, G, B
        print("built", p["filename"], flush=True)
        return node

    with ThreadPoolExecutor(max_workers=min(parallel_jobs(0.6, 10), os.cpu_count() or 1)) as ex:     # stops are independent; cv2 releases the GIL
        nodes = list(ex.map(build_one, KEEP))

    meta = {
        "title": a.title,
        "mapped": os.path.exists(os.path.join(a.frames, "poses.json")),
        # az/turn are in each pano's own frame (0 = pano centre, + = clockwise). x/y/H are the rough minimap only.
        "convention": "per_pano_local",
        "approximate_map": True,
        "nodes": nodes,
        **({"partial": True} if a.early else {}),
        **({"tiles": {"cols": TILE_COLS, "rows": TILE_ROWS, "pad": TILE_PAD, "w": min(a.hi, 8192)}} if a.tiles == "on" else {}),
    }
    KP = [pts[i] for i in KEEP]
    gone = prune(a.out, {"panos": {p["filename"] for p in KP}, "panos_lo": {p["filename"] for p in KP}, "panos_xs": {p["filename"] for p in KP}, "thumbs": {p["filename"] for p in KP},
                         "tiles": {tile_name(p["id"], r, c) for p in KP for r in range(TILE_ROWS) for c in range(TILE_COLS)} if a.tiles == "on" else set(),
                         "depth": {n["dm"].split("?")[0].split("/")[-1] for n in nodes if n.get("dm")}})
    if gone:
        print(f"removed {gone} pictures of an earlier build from {a.out}")
    if os.path.exists(pf):
        placed = [k for k, nd in enumerate(P["nodes"]) if nd.get("x") is not None]
        path = walked_path(a.frames, pts, MX, MY, ZS, CAMH, placed)
        if path:
            meta["path"] = path                                          # the walk he took, in map metres, for the minimap (never a place to walk to)
    for flag, fn in (("dollhouse", "dollhouse.json"), ("plan", "plan.svg")):          # extras made by later steps (dollhouse.py, floorplan.py) stay linked after a rebuild
        if os.path.exists(os.path.join(a.out, fn)):
            meta[flag] = True
    acc = None                                                          # how far the Measure button can be trusted: THIS walk's own grading against tape (qa/scoreboard.py), never another walk's; none = not checked
    for cand in (os.path.join(a.frames, "accuracy.json"),):
        if os.path.exists(cand):
            try:
                acc = float(json.load(open(cand))["measure_pct"]); break
            except (ValueError, KeyError, TypeError):
                pass
    if acc is not None:
        meta["measure_pct"] = round(acc, 1)                              # the viewer shows "+/- x %" and hides Measure above 10 %
    nf = os.path.join(os.path.dirname(os.path.abspath(a.frames)), "notes.json")           # the notes pinned in an earlier visit travel with the tour
    if os.path.exists(nf):
        shutil.copy(nf, os.path.join(a.out, "notes.json"))
    if os.path.exists(os.path.join(a.out, "notes.json")):
        meta["notes"] = True
    json.dump(meta, open(os.path.join(a.out, "tour.json"), "w"), separators=(",", ":"))      # compact: the phone downloads this first
    shutil.copy(os.path.join(HERE, "viewer", "index.html"), os.path.join(a.out, "index.html"))
    ensure_three(a.out, a.frames)
    gdir = os.path.join(HERE, "docs", "capture")                       # the "How to record a walk" pictures of the help panel
    if os.path.isdir(gdir):
        os.makedirs(os.path.join(a.out, "guide"), exist_ok=True)
        for fn in sorted(os.listdir(gdir)):
            if fn.endswith(".svg"):
                shutil.copy(os.path.join(gdir, fn), os.path.join(a.out, "guide", fn))
    print(f"done: {len(nodes)} points -> {a.out}")


if __name__ == "__main__":
    main()
