"""A continuous camera path for the whole walk (not just the stops), as the first half of the visual-inertial scale (imu_scale.py is the second half).

Usage:  python vio_map.py <video.insv> [--frames frames] [--step 1.0]
Reads:  the video, frames/calib.json (the lens fit), frames/points.json + frames/points/ (the stops' own pictures help tie the path to the stops)
Writes: frames/vio/poses.npz: image name, video time (s; the stops use their own time), camera centre (map units) and rotation (camera from world) of every picture
        that COLMAP placed, in COLMAP's own frame; frames/vio/images/ (small pictures, can be deleted)

How: one small (2048 px) stitched picture every `step` seconds, plus every stop, are put on one COLMAP map (equirectangular camera, matched with the pictures
within 8 places before and after in time, the stick and the person under the camera masked out). The path between stops is then known at 1 Hz, which is what
the accelerometer needs to be compared with.
"""
import argparse, json, os, queue, shutil, subprocess, sys, tempfile, threading, time
from concurrent.futures import ThreadPoolExecutor
import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import stitch_lib as sl                        # noqa: E402
from imutil import imread, imwrite, video_streams   # noqa: E402
from stitch_points import grab_both            # noqa: E402

W = 2048


def frame_times(dur, step):
    """The video times (s) of the pictures the map is made from: every `step` seconds from 0.8 s, stopping 0.5 s before the end."""
    return np.arange(0.8, dur - 0.5, step)


def _select_expr(t0, step):
    """ffmpeg select expression that keeps, for k = 0, 1, 2 ..., the first frame at or after t0 + k * step (what `-ss` + one frame gives for each time)."""
    return f"select='gte(t,{t0!r}+{step!r}*selected_n)'"


def decode_modes(video):
    """The ways to decode, best first: 'cuda' (frames stay on the video card, the select drops all but the wanted ones there, only those are downloaded and put
    back into the video's own yuvj420p before the BGR conversion - the pixels are bit for bit the software decoder's, as in grab_pairs.py), then 'auto' (the old
    -hwaccel auto: every frame is downloaded), then software. SHOP_TOUR_VIO_DECODE=cuda|auto|sw picks one."""
    want = os.environ.get("SHOP_TOUR_VIO_DECODE", "").strip().lower()
    if want in ("cuda", "auto", "sw"):
        return [want] + (["sw"] if want != "sw" else [])
    modes = []
    try:
        import grab_pairs
        if grab_pairs._has_cuda() and grab_pairs._pix_fmt(video) == "yuvj420p":
            modes.append("cuda")
    except Exception:
        pass
    return modes + ["auto", "sw"]


def decode_lens(video, stream, times, size, out_q, hw="auto"):
    """One sequential pass over one lens of the video: puts (k, BGR frame) on out_q for the first frame at or after each time in `times`
    (None at the end). A seek per picture (the old way) restarts the decoder 176 times; one pass decodes each frame once.
    Returns the number of frames delivered. hw: 'cuda', 'auto' (or True) or 'sw' (or False)."""
    w, h = size
    n = len(times)
    if hw is True:
        hw = "auto"
    sel = _select_expr(float(times[0]), float(times[1] - times[0]) if n > 1 else 1.0)
    if hw == "cuda":
        pre, vf = ["-hwaccel", "cuda", "-hwaccel_output_format", "cuda"], sel + ",hwdownload,format=nv12,format=yuvj420p"
    elif hw == "auto":
        pre, vf = ["-hwaccel", os.environ.get("SHOP_TOUR_HWACCEL", "auto")], sel
    else:
        pre, vf = [], sel
    cmd = ["ffmpeg", "-v", "error", "-nostdin", *pre, "-i", video, "-map", f"0:v:{stream}",
           "-vf", vf,
           "-fps_mode", "passthrough", "-frames:v", str(n), "-pix_fmt", "bgr24", "-f", "rawvideo", "pipe:1"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    err = []
    threading.Thread(target=lambda: err.append(proc.stderr.read()), daemon=True).start()     # a full stderr pipe would stall ffmpeg
    got, nbytes = 0, w * h * 3
    try:
        while got < n:
            buf = bytearray(nbytes)
            mv, have = memoryview(buf), 0
            while have < nbytes:
                r = proc.stdout.readinto(mv[have:])
                if not r:
                    break
                have += r
            if have < nbytes:
                break
            out_q.put((got, np.frombuffer(buf, np.uint8).reshape(h, w, 3)))
            got += 1
    finally:
        try:
            proc.stdout.close()
        except Exception:
            pass
        if proc.poll() is None:
            proc.kill()
        proc.wait()
    return got, b"".join(err)[-300:].decode(errors="replace")


def extract(video, frames, step, begin=None, on_saved=None):
    """The small stitched pictures of the walk (and the stops' own) in frames/vio/images. begin(vd, idir, mdir) is called once the folders are made; on_saved(name) as each
    picture (and its mask) is written - the SIFT features of a picture are made while the later ones are still being decoded (Overlap)."""
    vd = os.path.join(frames, "vio"); idir = os.path.join(vd, "images"); mdir = os.path.join(vd, "masks")
    shutil.rmtree(vd, ignore_errors=True); os.makedirs(idir); os.makedirs(mdir)
    if begin:
        begin(vd, idir, mdir)
    st = video_streams(video)
    dur = float(st[0].get("duration") or 0) if st else 0
    pts = json.load(open(os.path.join(frames, "points.json")))
    if dur <= 0:
        dur = max(p["timestamp"] for p in pts) + 3
    P = sl.load_params(os.path.join(frames, "calib.json"))
    M = sl.build_maps(P, W)
    times = frame_times(dur, step)
    mk = np.full((W // 2, W), 255, np.uint8); mk[int(W // 2 * 0.78):] = 0; mk[:int(W // 2 * 0.05)] = 0

    def save(k, f, b):
        name = f"f_{k:04d}.jpg"
        pano = sl.stitch_maps(b, f, M)
        imwrite(os.path.join(idir, name), cv2.resize(pano, (W, W // 2), interpolation=cv2.INTER_AREA), [cv2.IMWRITE_JPEG_QUALITY, 90])
        imwrite(os.path.join(mdir, name + ".png"), mk)
        if on_saved:
            on_saved(name)
        return name, float(times[k])

    def one(k):                                    # the old way, one seek and one decode for this time: the few times the single pass did not reach
        t = float(times[k])
        with tempfile.TemporaryDirectory() as tmp:
            f, b = os.path.join(tmp, "f.bmp"), os.path.join(tmp, "b.bmp")
            grab_both(video, t, f, b)
            return save(k, imread(f), imread(b))

    items = []
    if len(times) and len(st) >= 2 and st[0].get("width") and st[0].get("height") and not os.environ.get("SHOP_TOUR_VIO_SEEK"):
        size = (int(st[0]["width"]), int(st[0]["height"]))
        done = {}
        for hw in decode_modes(video):             # GPU decode if there is one; if that gives up part way, the rest is decoded the next way down
            todo = [k for k in range(len(times)) if k not in done]
            if not todo:
                break
            sub = times[todo]
            if len(sub) > 1 and not np.allclose(np.diff(sub), np.diff(sub)[0]):
                break
            qs = [queue.Queue(maxsize=2), queue.Queue(maxsize=2)]
            res = [None, None]
            ths = [threading.Thread(target=lambda i=i: res.__setitem__(i, decode_lens(video, i, sub, size, qs[i], hw)), daemon=True) for i in (0, 1)]
            for t_ in ths:
                t_.start()
            pending = threading.Semaphore(3)
            with ThreadPoolExecutor(max_workers=3) as ex:
                futs = []
                for j in range(len(sub)):
                    fr = [None, None]
                    for i in (0, 1):               # the lenses are read in step; a lens that ran out of frames leaves its queue empty
                        while fr[i] is None:
                            try:
                                fr[i] = qs[i].get(timeout=0.5)
                            except queue.Empty:
                                if not ths[i].is_alive() and qs[i].empty():
                                    break
                    if fr[0] is None or fr[1] is None:
                        break
                    pending.acquire()
                    k = todo[j]
                    fut = ex.submit(lambda k=k, f=fr[0][1], b=fr[1][1]: save(k, f, b))
                    fut.add_done_callback(lambda _f: pending.release())
                    futs.append((k, fut))
                for k, fut in futs:
                    done[k] = fut.result()
            t_end = time.time() + 60
            while any(t_.is_alive() for t_ in ths) and time.time() < t_end:      # a decoder still blocked on a full queue (the other lens ran out): let it finish
                for i in (0, 1):
                    while not qs[i].empty():
                        qs[i].get_nowait()
                time.sleep(0.05)
    else:
        done = {}
    with ThreadPoolExecutor(max_workers=4) as ex:  # whatever the single pass did not deliver (the last time at the very end of a video; an odd file)
        rest = [k for k in range(len(times)) if k not in done]
        for k, r in zip(rest, ex.map(one, rest)):
            done[k] = r
    items = [done[k] for k in range(len(times))]
    import stepsync
    if stepsync.enabled():                           # (begun early, while the stops were still being stitched: the stops' own pictures are the one thing read from that step)
        if stepsync.wait_for("stitch", log=lambda m: print(m, flush=True)) != "ok":
            raise SystemExit("the stitch step did not finish: there are no stop pictures to put in the map")
    for p in pts:
        sp = os.path.join(frames, "points", p["filename"])
        if os.path.exists(sp):
            name = "s_" + p["filename"]
            imwrite(os.path.join(idir, name), cv2.resize(imread(sp), (W, W // 2), interpolation=cv2.INTER_AREA), [cv2.IMWRITE_JPEG_QUALITY, 90])
            imwrite(os.path.join(mdir, name + ".png"), mk)
            if on_saved:
                on_saved(name)
            items.append((name, float(p["timestamp"])))
    items.sort(key=lambda x: x[1])
    return vd, items


def _env_num(name, cast, default):
    try:
        return cast(os.environ[name])
    except (KeyError, ValueError):
        return default


def sift_settings(features=None):
    """(SIFT features per picture, first octave) of the walk map."""
    import colmap_det
    fast = colmap_det.vio_fast()
    return features or _env_num("SHOP_TOUR_VIO_FEATURES", int, 4096), _env_num("SHOP_TOUR_VIO_FIRST_OCTAVE", int, 0 if fast else None)


def overlap_wanted():
    """SHOP_TOUR_VIO_OVERLAP=0: the features are made after every picture is written (the old order)."""
    return os.environ.get("SHOP_TOUR_VIO_OVERLAP", "").strip().lower() not in ("0", "false", "no", "off")


class Overlap:
    """The SIFT features of the pictures, made in batches while the later pictures are still being decoded and written. Each picture has its own features (the same
    keypoints and descriptors whenever and with whichever others they are made), the batches share the one camera, and the numbering is put in order of the names at the end
    (colmap_det.canonical_ids, as after a many-threaded extraction): the database comes out the same as one made by a single pass over the folder (checked table by table).
    finish() returns True when the database holds the features of every picture written; False (the database removed) when anything went wrong: the caller then makes them in
    one pass, as before."""
    def __init__(self, vd, idir, mdir, features, first_octave, batch=16, idle=1.5):
        self.db, self.idir, self.mdir = os.path.join(vd, "db.db"), idir, mdir
        self.features, self.first_octave, self.batch, self.idle = features, first_octave, batch, idle
        self.q = queue.Queue(); self.err = None; self.n = 0
        self.th = threading.Thread(target=self._run, daemon=True); self.th.start()

    def add(self, name):
        self.q.put(name)

    def _run(self):
        try:
            import pycolmap
            import colmap_det
            colmap_det.start(pycolmap)
            if not os.environ.get("SHOP_TOUR_COLMAP_LOG"):
                pycolmap.logging.minloglevel = 2
            eo = colmap_det.extraction_options(pycolmap, self.features, self.first_octave)
            names, closing = [], False
            while not closing:
                try:
                    n = self.q.get(timeout=self.idle)
                except queue.Empty:
                    n = ""                                                    # (nothing new for a moment: what is waiting is made now)
                if n is None:
                    closing = True
                elif n:
                    names.append(n)
                if names and (closing or n == "" or len(names) >= self.batch):
                    ro = pycolmap.ImageReaderOptions(); ro.camera_model = "EQUIRECTANGULAR"; ro.mask_path = self.mdir
                    colmap_det.extract_names(pycolmap, self.db, self.idir, names, ro, eo, same_camera=bool(self.n))      # (one camera for the whole walk, as one pass over the folder makes)
                    self.n += len(names); names = []
        except BaseException as e:                                            # the caller falls back to one pass
            self.err = e

    def finish(self, expect):
        self.q.put(None); self.th.join()
        try:
            if self.err is None and self.n == expect:
                import sqlite3
                import colmap_det
                c = sqlite3.connect(self.db)
                try:
                    have = c.execute("select count(*) from images").fetchone()[0]
                    kp = c.execute("select count(*) from keypoints").fetchone()[0]
                finally:
                    c.close()
                if have == expect and kp == expect:                            # (every picture is in the database, with its keypoints)
                    colmap_det.canonical_ids(self.db)                          # (the pictures are numbered in the order of their names, whatever order they arrived in)
                    return True
        except Exception as e:
            self.err = e
        for p in (self.db, self.db + "-wal", self.db + "-shm"):
            try:
                os.remove(p)
            except OSError:
                pass
        return False


def sfm(vd, items, window=8, features=None, log=print, features_done=False):
    import pycolmap
    import colmap_det
    colmap_det.start(pycolmap)
    fast = colmap_det.vio_fast()
    features, first_octave = sift_settings(features)
    map_threads = _env_num("SHOP_TOUR_VIO_THREADS", int, min(8, os.cpu_count() or 1) if fast else None)
    max_tracks = _env_num("SHOP_TOUR_VIO_MAX_TRACKS", int, 30000 if fast else None)
    if not os.environ.get("SHOP_TOUR_COLMAP_LOG"):
        pycolmap.logging.minloglevel = 2
    idir, mdir = os.path.join(vd, "images"), os.path.join(vd, "masks")
    db = os.path.join(vd, "db.db")
    ro = pycolmap.ImageReaderOptions(); ro.camera_model = "EQUIRECTANGULAR"; ro.mask_path = mdir
    t0 = time.time()
    log("features...", flush=True)
    if not features_done:
        colmap_det.extract(pycolmap, db, idir, ro, features, first_octave)
    lst = os.path.join(vd, "pairs.txt")
    with open(lst, "w") as fh:
        for i in range(len(items)):
            for j in range(i + 1, min(len(items), i + window + 1)):
                fh.write(f"{items[i][0]} {items[j][0]}\n")
    log(f"  features {time.time() - t0:.0f} s", flush=True); t0 = time.time()
    log(f"matching {len(items)} pictures...", flush=True)
    colmap_det.match_pairs(pycolmap, db, lst, vd)
    log(f"  matching {time.time() - t0:.0f} s", flush=True); t0 = time.time()
    log("mapping...", flush=True)
    best = None
    if colmap_det.use_global():                    # one global solve first; the incremental mapper below is the fallback when it places too few
        out = os.path.join(vd, "sparse"); shutil.rmtree(out, ignore_errors=True); os.makedirs(out)
        try:
            recs = pycolmap.global_mapping(db, idir, out, colmap_det.global_options(pycolmap, map_threads, max_tracks))
            best = max(recs.values(), key=lambda r: r.num_reg_images()) if recs else None
        except Exception as e:
            log(f"global mapper failed ({type(e).__name__}); using the incremental one", flush=True)
        if best is not None and best.num_reg_images() < 0.97 * len(items):
            log(f"global mapper placed {best.num_reg_images()}/{len(items)}; using the incremental one", flush=True)
            best = None
    for inl, ratio in (() if best is not None else ((30, 0.25), (20, 0.15), (15, 0.10))):
        out = os.path.join(vd, "sparse"); shutil.rmtree(out, ignore_errors=True); os.makedirs(out)
        opts = colmap_det.mapper_options(pycolmap)
        opts.mapper.abs_pose_min_num_inliers = inl; opts.mapper.abs_pose_min_inlier_ratio = ratio
        recs = pycolmap.incremental_mapping(db, idir, out, opts)
        cand = max(recs.values(), key=lambda r: r.num_reg_images())
        if best is None or cand.num_reg_images() > best.num_reg_images():
            best = cand
        if cand.num_reg_images() >= 0.97 * len(items):
            break
        log(f"placed {cand.num_reg_images()}/{len(items)}; trying looser thresholds...", flush=True)
    log(f"  mapping {time.time() - t0:.0f} s", flush=True)
    return best


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video"); ap.add_argument("--frames", default="frames"); ap.add_argument("--step", type=float, default=1.0)
    a = ap.parse_args()
    t0 = time.time()
    ov = []
    begin = (lambda vd_, idir_, mdir_: ov.append(Overlap(vd_, idir_, mdir_, *sift_settings()))) if overlap_wanted() else None
    vd, items = extract(a.video, a.frames, a.step, begin=begin, on_saved=(lambda n: ov[0].add(n)) if begin else None)
    print(f"{len(items)} pictures in {time.time() - t0:.0f} s", flush=True)
    rec = sfm(vd, items, features_done=bool(ov) and ov[0].finish(len(items)))
    tm = dict(items)
    names, times, Cs, Rs = [], [], [], []
    for im in rec.images.values():
        if not im.has_pose:
            continue
        T = im.cam_from_world() if callable(im.cam_from_world) else im.cam_from_world
        R = T.rotation.matrix(); C = -R.T @ T.translation
        names.append(im.name); times.append(tm[im.name]); Cs.append(C); Rs.append(R)
    o = np.argsort(times)
    np.savez(os.path.join(vd, "poses.npz"), names=np.array(names)[o], t=np.array(times)[o], C=np.array(Cs)[o], R=np.array(Rs)[o])
    print(f"placed {len(names)}/{len(items)} pictures; total {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
