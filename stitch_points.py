"""Re-stitch every stop in frames/points.json from the raw .insv with the calibrated lens model.

Usage:  python stitch_points.py raw/VID_xxx.insv [--calib frames/calib.json] [--frames frames]
The first run moves the old ffmpeg panos to frames/points_ffmpeg/ so nothing is lost.
Then rerun:  python level_panos.py ; python hop_geometry.py ; python build_tour.py
"""
import argparse, json, os, shutil, subprocess, tempfile, time
from concurrent.futures import ThreadPoolExecutor
import cv2
import stitch_lib as sl
from imutil import imread, imwrite, parallel_jobs, video_kind, video_streams


def grab_both(insv, t, f_out, b_out):
    """One decode per stop: stream 0 -> f_out, stream 1 -> b_out. GPU decode if available; if the hardware decoder is busy (too many
    parallel stops) or fails, retry once more on the GPU and then in software. If no picture comes out (a time at or past the last
    frame), step back a few frames."""
    last = ""
    for back in (0.0, 0.05, 0.12, 0.25):
        for hw in (["-hwaccel", "auto"], ["-hwaccel", "auto"], []):
            for o in (f_out, b_out):
                if os.path.exists(o):
                    os.remove(o)
            r = subprocess.run(["ffmpeg", "-v", "error", "-y", *hw, "-ss", f"{max(0.0, t - back):.3f}", "-i", insv,
                                "-map", "0:v:0", "-frames:v", "1", f_out,
                                "-map", "0:v:1", "-frames:v", "1", b_out], capture_output=True)
            last = r.stderr.decode(errors="replace")[-300:]
            if r.returncode == 0 and all(os.path.exists(o) and os.path.getsize(o) > 0 for o in (f_out, b_out)):
                return
    raise RuntimeError(f"ffmpeg could not decode a frame at t={t} s of {insv} (is that past the end of the video?): {last}")


def cut_equirect(a):
    """An already-stitched equirectangular video: no lens model, no stitching. Just cut the frame at each stop (kept at the video's own width,
    at most 7680) into frames/points/NNN.jpg."""
    pts = json.load(open(os.path.join(a.frames, "points.json")))
    pdir = os.path.join(a.frames, "points")
    os.makedirs(pdir, exist_ok=True)
    keep = {p["filename"] for p in pts}
    for f in os.listdir(pdir):
        if f.lower().endswith(".jpg") and f not in keep:
            os.remove(os.path.join(pdir, f))

    def one(k):
        p = pts[k]
        with tempfile.TemporaryDirectory() as tmp:
            f = os.path.join(tmp, "f.bmp")
            for back in (0.0, 0.05, 0.12, 0.25):
                subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", f"{max(0.0, p['timestamp'] - back):.3f}", "-i", a.insv, "-frames:v", "1", f],
                               capture_output=True)
                if os.path.exists(f) and os.path.getsize(f) > 0:
                    break
            img = imread(f)
            if img.shape[1] > 7680:
                img = cv2.resize(img, (7680, 3840), interpolation=cv2.INTER_AREA)
            imwrite(os.path.join(pdir, p["filename"]), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
            print(f"[{k + 1}/{len(pts)}] {p['filename']}  t={p['timestamp']}  done", flush=True)

    with ThreadPoolExecutor(max_workers=3) as ex:
        list(ex.map(one, range(len(pts))))
    print("done (stitched video: frames cut, nothing to stitch).")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("insv")
    ap.add_argument("--calib", default="frames/calib.json")
    ap.add_argument("--frames", default="frames")
    ap.add_argument("--width", type=int, default=7680, help="7680 keeps all the detail the X5 records")
    ap.add_argument("--jobs", type=int, default=min(parallel_jobs(2.2, 4), os.cpu_count() or 1), help="stops decoded and stitched at the same time (each needs ~2 GB; the hardware decoder is the limit, 4 is as fast as 6)")
    a = ap.parse_args()
    if video_kind(video_streams(a.insv)) == "equirect":
        return cut_equirect(a)
    P = sl.load_params(a.calib)
    pts = json.load(open(os.path.join(a.frames, "points.json")))
    pdir = os.path.join(a.frames, "points")
    bak = os.path.join(a.frames, "points_ffmpeg")
    if not os.path.exists(bak) and os.path.isdir(pdir):
        shutil.copytree(pdir, bak)
        print("old panos backed up to", bak)
    os.makedirs(pdir, exist_ok=True)
    keep = {p["filename"] for p in pts}
    stale = [f for f in os.listdir(pdir) if f.lower().endswith(".jpg") and f not in keep]      # stops of an earlier run with more stops
    for f in stale:
        os.remove(os.path.join(pdir, f))
    ddir = os.path.join(a.frames, "depth")
    if os.path.isdir(ddir):
        ids = {p["id"] for p in pts}
        for f in os.listdir(ddir):
            if f.split("_")[0] not in ids and (f[:3].isdigit() or f.startswith("eq_")):
                if f.startswith("eq_") and f[3:6] in ids:
                    continue
                os.remove(os.path.join(ddir, f))
    if stale:
        print(f"removed {len(stale)} old panoramas from an earlier run with more stops")
    print(f"building lens maps for {a.width}x{a.width // 2} (once)...", flush=True)
    M = sl.build_maps(P, a.width)
    def one(k):
        p = pts[k]
        with tempfile.TemporaryDirectory() as tmp:          # own temp files per stop, so stops can run side by side
            f, b = os.path.join(tmp, "f.bmp"), os.path.join(tmp, "b.bmp")      # BMP, not PNG: ffmpeg spent ~4 of its ~5 s per stop compressing the PNGs
            t0 = time.time()
            grab_both(a.insv, p["timestamp"], f, b)
            # stream 0 = lens F (looks backwards in the pano), stream 1 = lens A (pano centre)
            pano = sl.stitch_maps(imread(b), imread(f), M)    # cv2 and ffmpeg release the GIL, so threads share the maps
            imwrite(os.path.join(pdir, p["filename"]), pano, [cv2.IMWRITE_JPEG_QUALITY, 95])
            print(f"[{k + 1}/{len(pts)}] {p['filename']}  t={p['timestamp']}  done ({time.time() - t0:.0f}s)", flush=True)

    with ThreadPoolExecutor(max_workers=max(1, a.jobs)) as ex:
        list(ex.map(one, range(len(pts))))                  # list() re-raises any failure
    print("done. next: python level_panos.py ; python hop_geometry.py ; python build_tour.py")


if __name__ == "__main__":
    main()
