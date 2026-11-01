"""Find the stops in a walk and pick the sharpest frame at each one.

The walker pauses at each spot. We decode both lenses small (one ffmpeg pass), measure how much the picture
changes frame to frame (motion) and how crisp it is (sharpness), find the still stretches, and at each one
keep the sharpest frame. Writes frames/points.json (id, filename, timestamp, ...), which every later step uses.

  python find_stops.py raw/walk.insv              # detect stops from scratch
  python find_stops.py raw/walk.insv --refine     # keep the existing stops, move each to its sharpest frame (+-0.6 s)
  python find_stops.py raw/walk.insv --every 2    # no pauses in the walk? take the sharpest frame every 2 s

Writes frames/stops_debug.png (motion + sharpness over time, chosen stops marked) so thresholds can be checked.
"""
import argparse, json, os, subprocess
import cv2
import numpy as np
from imutil import video_kind, video_streams


import functools


@functools.lru_cache(maxsize=None)
def is_equirect(insv):
    return video_kind(video_streams(insv)) == "equirect"


def video_end(insv):
    """Time of a safely decodable last frame: a little before the end (the 30 fps scan can report times past the last real frame, and
    ffmpeg then writes no picture)."""
    try:
        out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=duration,r_frame_rate",
                              "-of", "csv=p=0", insv], capture_output=True, text=True, check=True).stdout.strip().split(",")
        num, den = out[0].split("/") if "/" in out[0] else (out[0], "1")
        fps = float(num) / float(den); dur = float(out[1])
        return dur - 3.0 / fps
    except Exception:
        return float("inf")


def frames(insv, fps, size, start=None, dur=None):
    """Yield (t, gray image size x 2*size) with both lenses side by side. Each lens is decoded by its own ffmpeg process (the hardware
    decoder handles two streams at once ~25% faster than one process doing both one after the other)."""
    base = ["ffmpeg", "-v", "error", "-hwaccel", "auto"]
    if start is not None:
        base += ["-ss", f"{max(0.0, start):.3f}"]
    if dur is not None:
        base += ["-t", f"{dur:.3f}"]
    if is_equirect(insv):
        # an already-stitched 2:1 video: its left and right halves play the part of the two lenses (each is a square covering 180 degrees)
        p = subprocess.Popen(base + ["-i", insv, "-vf", f"fps={fps},scale={2 * size}:{size},format=gray", "-f", "rawvideo", "-pix_fmt", "gray", "-"],
                             stdout=subprocess.PIPE)
        k = 0
        t0 = start or 0.0
        try:
            while True:
                buf = p.stdout.read(2 * size * size)
                if len(buf) < 2 * size * size:
                    break
                yield t0 + k / fps, np.frombuffer(buf, np.uint8).reshape(size, 2 * size)
                k += 1
        finally:
            if p.poll() is None:
                p.kill()
            p.stdout.close(); p.wait()
        return
    procs = [subprocess.Popen(base + ["-i", insv, "-map", f"0:v:{m}", "-vf", f"fps={fps},scale={size}:{size},format=gray",
                                      "-f", "rawvideo", "-pix_fmt", "gray", "-"], stdout=subprocess.PIPE) for m in (0, 1)]
    n = size * size
    k = 0
    t0 = start or 0.0
    try:
        while True:
            a, b = procs[0].stdout.read(n), procs[1].stdout.read(n)
            if len(a) < n or len(b) < n:
                break
            yield t0 + k / fps, np.hstack([np.frombuffer(a, np.uint8).reshape(size, size), np.frombuffer(b, np.uint8).reshape(size, size)])
            k += 1
    finally:
        for p in procs:
            if p.poll() is None:
                p.kill()
            p.stdout.close(); p.wait()


def sharpness(img):
    """Laplacian variance inside the middle of each fisheye (the rim is soft on every lens)."""
    h, w = img.shape
    s = h
    vals = []
    for x0 in (0, s):
        c = img[int(s * 0.2):int(s * 0.8), x0 + int(s * 0.2):x0 + int(s * 0.8)]
        vals.append(cv2.Laplacian(c, cv2.CV_32F).var())
    return float(min(vals))          # a frame is only as good as its blurrier lens


def scan(insv, fps, size, start=None, dur=None):
    T, M, S = [], [], []
    prev = None
    for t, img in frames(insv, fps, size, start, dur):
        small = cv2.GaussianBlur(cv2.resize(img, (160, 80), interpolation=cv2.INTER_AREA), (3, 3), 0).astype(np.float32)
        M.append(0.0 if prev is None else float(np.mean(np.abs(small - prev))))
        prev = small
        S.append(sharpness(img)); T.append(t)
    return np.array(T), np.array(M), np.array(S)


def safe(S, r):
    """Sharpness a frame can rely on even if the grab later lands a frame or two off: the worst of its neighbours."""
    S = np.asarray(S, float)
    pad = np.pad(S, r, mode="edge")
    return np.min(np.stack([pad[i:i + len(S)] for i in range(2 * r + 1)]), 0)


def runs(mask):
    out, start = [], None
    for i, m in enumerate(mask):
        if m and start is None:
            start = i
        if not m and start is not None:
            out.append((start, i)); start = None
    if start is not None:
        out.append((start, len(mask)))
    return out


def detect(T, M, S, fps, min_pause, min_gap):
    m = np.convolve(M, np.ones(5) / 5, "same")
    thr = max(np.percentile(m, 30) * 1.6, 0.6)                 # adaptive: "still" relative to this walk
    still = m < thr
    stops = []
    for a, b in runs(still):
        if (b - a) / fps < min_pause:
            continue
        k = a + int(np.argmax(safe(S, 1)[a:b]))
        stops.append(k)
    # keep stops at least min_gap apart (keep the sharper of two that are too close)
    kept = []
    for k in stops:
        if kept and T[k] - T[kept[-1]] < min_gap:
            if S[k] > S[kept[-1]]:
                kept[-1] = k
            continue
        kept.append(k)
    return kept, thr


def debug_plot(path, T, M, S, picks, thr):
    W, H = 1600, 400
    img = np.full((H, W, 3), 255, np.uint8)
    def line(v, col, top, hgt):
        v = np.asarray(v, float); v = (v - v.min()) / (np.ptp(v) + 1e-9)
        pts = np.stack([(T - T[0]) / (T[-1] - T[0] + 1e-9) * (W - 1), top + hgt - v * hgt], 1).astype(np.int32)
        cv2.polylines(img, [pts], False, col, 1)
    line(M, (40, 40, 200), 10, 180); line(S, (40, 150, 40), 210, 180)
    for k in picks:
        x = int((T[k] - T[0]) / (T[-1] - T[0] + 1e-9) * (W - 1))
        cv2.line(img, (x, 0), (x, H), (0, 0, 0), 1)
    cv2.putText(img, f"red: motion (still below {thr:.2f})   green: sharpness   black: chosen stops", (10, H - 8), 0, 0.5, (0, 0, 0), 1)
    cv2.imwrite(path, img)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("insv")
    ap.add_argument("--frames", default="frames")
    ap.add_argument("--refine", action="store_true", help="keep the existing stops, move each to its sharpest frame")
    ap.add_argument("--every", type=float, default=0, help="no pauses: sharpest frame in every window of this many seconds")
    ap.add_argument("--min-pause", type=float, default=0.7)
    ap.add_argument("--min-gap", type=float, default=1.5)
    a = ap.parse_args()
    os.makedirs(a.frames, exist_ok=True)
    pj = os.path.join(a.frames, "points.json")

    if a.refine:
        pts = json.load(open(pj))
        for p in pts:
            t = p.get("timestamp_orig", p["timestamp"])
            T, M, S = scan(a.insv, 30, 640, t - 0.6, 1.2)
            if len(T):
                k = int(np.argmax(safe(S, 2)))
                p["timestamp_orig"] = t
                p["timestamp"] = round(float(T[k]), 3)
                print(f"{p['id']}  {t:7.2f} -> {p['timestamp']:7.2f}   sharpness {S[k]:.0f} (was ~{S[int(np.argmin(np.abs(T - t)))]:.0f})", flush=True)
    else:
        fps = 10
        print("scanning the whole walk (both lenses, small)...", flush=True)
        T, M, S = scan(a.insv, fps, 480)
        def every_picks(sec):
            out = []
            for w0 in np.arange(T[0], T[-1], sec):
                idx = np.nonzero((T >= w0) & (T < w0 + sec))[0]
                if len(idx):
                    out.append(int(idx[np.argmax(S[idx])]))
            return out
        if a.every > 0:
            picks, thr = every_picks(a.every), 0.0
        else:
            picks, thr = detect(T, M, S, fps, a.min_pause, a.min_gap)
            if len(picks) < 2:                                   # a walk without pauses: take the sharpest frame every few seconds instead
                print(f"found {len(picks)} pause(s) of >= {a.min_pause} s: taking the sharpest frame in every 4 s instead (use --every N to choose, or --min-pause to be less strict)", flush=True)
                picks, thr = every_picks(4.0), 0.0
        debug_plot(os.path.join(a.frames, "stops_debug.png"), T, M, S, picks, thr)
        # refine each pick at full frame rate (the 10 fps scan can miss the crispest frame)
        def refine(k):
            t = float(T[k])
            T2, _, S2 = scan(a.insv, 30, 640, t - 0.25, 0.5)
            return float(T2[int(np.argmax(safe(S2, 2)))]) if len(T2) else t
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=3) as ex:          # each refinement is its own short ffmpeg decode; 3 at once is what the decoder takes
            refined = list(ex.map(refine, picks))
        pts = []
        for n, t in enumerate(refined):
            pid = f"{n + 1:03d}"
            pts.append({"id": pid, "filename": pid + ".jpg", "timestamp": round(t, 3), "x": 0.0, "y": 0.0, "heading": 0.0})
            print(f"stop {pid} at {t:7.2f} s", flush=True)
        print(f"{len(pts)} stops (still threshold {thr:.2f}); check {os.path.join(a.frames, 'stops_debug.png')}")
    t_end = video_end(a.insv)
    for p in pts:
        p["timestamp"] = round(min(p["timestamp"], t_end), 3)           # never past the last decodable frame
        s = p["timestamp"]; p["time_str"] = f"{int(s // 3600):02d}:{int(s % 3600 // 60):02d}:{s % 60:06.3f}"
    json.dump(pts, open(pj, "w"), indent=2)
    print("wrote", pj)


if __name__ == "__main__":
    main()
