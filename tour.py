"""One command: Insta360 .insv walk -> walkable Matterport-style tour.

  python tour.py D:/walks/place/2026-01-01/walk.insv                 # run everything that isn't already done
  python tour.py D:/walks/place/2026-01-01/walk.insv --from depth    # redo from a step onward (names below)
  python tour.py D:/walks/place/2026-01-01/walk.insv --only stitch   # run just one step
  python tour.py D:/walks/place/2026-01-01/walk.insv --serve         # ...and serve the result at http://localhost:8000

Steps (each is its own script, so you can also run them by hand):
  stops    find_stops.py      pauses in the walk + sharpest frame at each      -> frames/points.json
  calib    calibrate_stitch   fit the two lenses (only if frames/calib.json is missing)
  stitch   stitch_points.py   8K seam-cut panoramas                            -> frames/points/*.jpg
  level    level_panos.py     true "up" for each pano                          -> frames/up.json
  hops     hop_geometry.py    geometry between nearby stops                    -> frames/hops.json
  map      map_stops.py       one shared map (COLMAP, 360 camera)              -> frames/poses_map.json
  depth    depth_panos.py     per-photo depth (Depth Anything V2 Small)        -> frames/depth/*_v*.npy
  floor    floor_panos.py     which pixels are floor (UperNet ADE20K)          -> frames/depth/*_f*.png
  rings    place_rings.py     floor height, occlusion, floor/depth maps        -> frames/poses.json
  capturer remove_capturer.py the person holding the stick is painted out         -> frames/clean/*.jpg
  privacy  privacy_gate.py    bystanders, faces, leftovers: a stop that fails is held back -> frames/privacy.json
  build    build_tour.py      the tour folder (held-back stops are left out)   -> tour/
  dollhouse dollhouse.py      every stop's depth fused into one model          -> tour/dollhouse.bin
  plan     floorplan.py       floor plan with dimensions from the model        -> tour/plan.svg
A step runs when any of its outputs is missing or older than its inputs.
"""
import argparse, glob, json, os, shutil, subprocess, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
WORK = HERE                                  # where this walk's frames/ and tour/ live (--work); the code always comes from HERE
F = os.path.join(WORK, "frames")


def _abs(p):
    """Steps list data as frames/... or tour/... (this walk's folder); everything else (viewer/, *.py) is code in HERE."""
    return os.path.join(WORK if p.split("/")[0] in ("frames", "tour") else HERE, p)


def py(script, *args):
    return [sys.executable, os.path.join(HERE, script), *args]


def _gpat(p):
    """_abs(p) as a glob pattern: the folder part is taken literally (a work folder called "shop [day 1]" must not be read as a character class)."""
    head, _, rest = p.partition("/")
    base = WORK if head in ("frames", "tour") else HERE
    return os.path.join(glob.escape(base), head, *rest.split("/")) if rest else os.path.join(glob.escape(base), head)


def mtimes(patterns):
    files = [f for p in patterns for f in glob.glob(_gpat(p))]
    return [os.path.getmtime(f) for f in files], files


def stale(inputs, outputs):
    out_t, out_f = mtimes(outputs)
    if not out_f or any(not glob.glob(_gpat(p)) for p in outputs):
        return True
    in_t, _ = mtimes(inputs)
    return bool(in_t) and max(in_t) > min(out_t)


def n_stops():
    try:
        return len(json.load(open(os.path.join(F, "points.json"))))
    except Exception:
        return 0


def calib_cmds(insv):
    """Grab 3 raw lens pairs spread over the walk and fit the lens model."""
    pts = json.load(open(os.path.join(F, "points.json")))
    ts = [pts[int(len(pts) * f)]["timestamp"] for f in (0.15, 0.5, 0.85)]
    src = os.path.join(F, "src"); os.makedirs(src, exist_ok=True)
    cmds = []
    for t in ts:
        cmds.append(["ffmpeg", "-v", "error", "-y", "-ss", f"{t:.3f}", "-i", insv,
                     "-map", "0:v:0", "-frames:v", "1", os.path.join(src, f"f_{t}.png"),
                     "-map", "0:v:1", "-frames:v", "1", os.path.join(src, f"b_{t}.png")])
    cmds.append(py("calibrate_stitch.py", "--src", src, "--out", os.path.join(F, "calib.json")))
    return cmds


def steps(insv, allow_partial=False, title=None):
    return [
        ("stops", [insv], ["frames/points.json"], lambda: [py("find_stops.py", insv)]),
        ("calib", [], ["frames/calib.json"], lambda: calib_cmds(insv)),
        ("stitch", [insv, "frames/points.json", "frames/calib.json", "stitch_lib.py"], ["frames/points/*.jpg"],
         lambda: [py("stitch_points.py", insv)]),
        ("level", ["frames/points/*.jpg"], ["frames/up.json"], lambda: [py("level_panos.py")]),
        ("hops", ["frames/points/*.jpg", "frames/up.json"], ["frames/hops.json"], lambda: [py("hop_geometry.py")]),
        ("map", ["frames/points/*.jpg", "frames/up.json", "frames/hops.json"], ["frames/poses_map.json"],
         lambda: [py("map_stops.py", *(["--allow-partial"] if allow_partial else []))]),
        ("depth", ["frames/points/*.jpg"], ["frames/depth/views.json"], lambda: [py("depth_panos.py")]),
        ("floor", ["frames/points/*.jpg", "frames/depth/views.json"], ["frames/depth/*_f0.png"], lambda: [py("floor_panos.py")]),
        ("rings", ["frames/poses_map.json", "frames/depth/views.json", "frames/depth/*_f0.png", "place_rings.py"],
         ["frames/depth/eq_*.png"], lambda: [py("place_rings.py")]),
        ("capturer", ["frames/points/*.jpg", "frames/up.json", "frames/poses.json", "capturer.py", "remove_capturer.py"], ["frames/clean/*.jpg"],
         lambda: [py("remove_capturer.py")]),
        ("privacy", ["frames/clean/*.jpg", "privacy_gate.py", "capturer.py"], ["frames/privacy.json"], lambda: [py("privacy_gate.py")]),
        ("build", ["frames/poses.json", "frames/depth/eq_*.png", "frames/privacy.json", "viewer/index.html", "build_tour.py"],
         ["tour/tour.json"], lambda: [py("build_tour.py", *(["--title", title] if title else []))]),
        ("dollhouse", ["frames/safe/*.jpg", "frames/poses.json", "frames/depth/eq_*.png", "frames/privacy.json", "dollhouse.py"], ["tour/dollhouse.json"],
         lambda: [py("dollhouse.py")]),
        ("plan", ["frames/dollhouse/mesh.ply", "floorplan.py"], ["tour/plan.svg"], lambda: [py("floorplan.py")]),
    ]


def video_problem(info):
    """Why a video cannot be used, in plain words, or None. info: the parsed output of
    ffprobe -show_entries stream=codec_type,width,height,duration -of json (None/{} if ffprobe could not read the file)."""
    streams = [x for x in (info or {}).get("streams", []) if x.get("codec_type", "video") == "video"]
    if not streams:
        return "ffprobe cannot read any video from this file (damaged or cut-off recording, or not a video). A recording that stopped early (battery, card) may need repairing first."
    if len(streams) < 2:
        a = streams[0]
        if a.get("width") and a.get("width") == 2 * (a.get("height") or 0):
            return None                                       # one 2:1 stream: an already-stitched equirectangular video (Insta360 Studio export)
        return ("this file has one video stream that is not 2:1 (width = 2 x height), so it is neither an Insta360 .insv with its two fisheye "
                "lens streams nor a stitched equirectangular export. Use the original .insv, or export 'equirectangular' / 360 at 2:1.")
    a, b = streams[0], streams[1]
    if (a.get("width"), a.get("height")) != (b.get("width"), b.get("height")):
        return f"the two video streams differ in size ({a.get('width')}x{a.get('height')} and {b.get('width')}x{b.get('height')}); expected two equal fisheye frames"
    try:
        if float(a.get("duration") or 0) < 4.0:
            return "the video is shorter than 4 seconds"
    except ValueError:
        pass
    return None


def check_video(path):
    try:
        out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v", "-show_entries", "stream=codec_type,width,height,duration",
                              "-of", "json", path], capture_output=True, text=True)
        info = json.loads(out.stdout or "{}")
    except Exception:
        info = {}
    why = video_problem(info)
    if why:
        sys.exit(f"cannot use {path}: {why}")


def video_path(path):
    """The video as the steps should get it: relative to this folder when possible, absolute otherwise (on Windows a video on another
    drive, e.g. D:/Videos, has no relative path: os.path.relpath raises ValueError)."""
    p = os.path.abspath(path)
    try:
        return os.path.relpath(p, WORK)
    except ValueError:
        return p


REQUIRED = {"numpy": "numpy", "cv2": "opencv-python-headless", "scipy": "scipy", "PIL": "pillow", "pycolmap": "pycolmap>=4.1",
            "torch": "torch torchvision", "transformers": "transformers"}


def preflight():
    """Fail in the first second, not after a 90 s step: ffmpeg and the Python packages every step needs."""
    import importlib.util, shutil
    problems = []
    for exe in ("ffmpeg", "ffprobe"):
        if not shutil.which(exe):
            problems.append(f"{exe} is not on your PATH (install FFmpeg from https://ffmpeg.org and reopen the terminal)")
    missing = [pip for mod, pip in REQUIRED.items() if importlib.util.find_spec(mod) is None]
    if missing:
        problems.append("missing Python packages. Run:  pip install " + " ".join(f'"{m}"' if ">" in m else m for m in missing))     # quote pycolmap>=4.1: the shell would read > as a redirect
    where = WORK
    while not os.path.isdir(where):                        # the work folder may not exist yet: measure the drive it will be on
        where = os.path.dirname(where)
    free = shutil.disk_usage(where).free / 2**30
    if free < 0.5:                                         # a full disk makes image writes fail half-way through a step
        problems.append(f"only {free:.2f} GB free on the drive of this folder; a walk needs about 1 GB the first time (frames/, tour/, COLMAP files). Free some space.")
    elif free < 3.0:
        print(f"warning: only {free:.1f} GB free on this drive (a 25-stop walk needs about 1 GB, more if you keep several tours)")
    if problems:
        sys.exit("cannot start:\n  - " + "\n  - ".join(problems))
    try:                                                   # a tip, not an error: an NVIDIA GPU with a CPU-only PyTorch is ~4x slower
        import torch
        if not torch.cuda.is_available() and shutil.which("nvidia-smi"):
            print("tip: an NVIDIA GPU was found but this PyTorch is CPU-only; the CUDA build (pytorch.org) makes depth/floor ~15x faster")
    except Exception:
        pass


def calib_fits(insv):
    """True if frames/calib.json was fitted for this video's lens frame size (the file stores it; older files are the X5's 3840)."""
    try:
        with open(os.path.join(F, "calib.json")) as fh:
            size = int(json.load(fh).get("size", 3840))
        out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width", "-of", "csv=p=0", insv],
                             capture_output=True, text=True, check=True).stdout.strip()
        return int(out) == size
    except Exception:
        return True                                         # cannot tell: keep the old behaviour


def _ident(insv_abs):
    return {"name": os.path.basename(insv_abs), "size": os.path.getsize(insv_abs)}


def source_changed(insv_abs, frames=None):
    """frames/ remembers which video it was built from (name + size). Timestamps alone cannot tell: a second walk recorded EARLIER than
    the first looks 'up to date' and the old walk's tour would be rebuilt silently. True if frames/ holds another video's results.
    Only reads; remember_source() records the video once its stops have really been found (a failed attempt must not count)."""
    try:
        with open(os.path.join(frames or F, "source.json")) as fh:
            old = json.load(fh)
    except Exception:
        return False
    return old != _ident(insv_abs)


def remember_source(insv_abs, frames=None):
    frames = frames or F
    os.makedirs(frames, exist_ok=True)
    with open(os.path.join(frames, "source.json"), "w") as fh:
        json.dump(_ident(insv_abs), fh)


def incomplete(name, frames=None, out=None):
    """Why a step's outputs are NOT complete for the current stop list (an interrupted run leaves some of them), or None. Timestamps alone
    would call such a step 'up to date'. Only steps with per-stop outputs are checked."""
    frames = frames or F
    out = out or os.path.join(WORK, "tour")
    try:
        with open(os.path.join(frames, "points.json")) as fh:
            pts = json.load(fh)
    except Exception:
        return None
    n = len(pts)
    ex = lambda *parts: os.path.exists(os.path.join(*parts))

    def count_json(fn, key=None):
        try:
            with open(os.path.join(frames, fn)) as fh:
                d = json.load(fh)
            return len(d[key]) if key else len(d)
        except Exception:
            return -1
    if name == "stitch":
        miss = [p["filename"] for p in pts if not ex(frames, "points", p["filename"])]
        return f"{len(miss)} panoramas missing (e.g. {miss[0]})" if miss else None
    if name == "level" and count_json("up.json") != n:
        return "up.json does not cover every stop"
    if name == "hops" and n > 1 and count_json("hops.json", "hops") != n - 1:
        return "hops.json does not cover every stop"
    if name == "map" and count_json("poses_map.json", "nodes") != n:
        return "poses_map.json does not cover every stop"
    if name == "depth":
        miss = [p["id"] for p in pts for k in range(4) if not ex(frames, "depth", f"{p['id']}_v{k}.npy")]
        return f"depth missing for stop(s) {sorted(set(miss))[:3]}" if miss else None
    if name == "floor":
        miss = [p["id"] for p in pts if not all(ex(frames, "depth", f"{p['id']}_{s}.png") for s in ("f0", "f1", "f2", "f3", "n", "nf"))]
        return f"floor masks missing for stop(s) {miss[:3]}" if miss else None
    if name == "rings":
        if count_json("poses.json", "nodes") != n:
            return "poses.json is missing, unreadable or does not cover every stop"
        miss = [p["id"] for p in pts if not ex(frames, "depth", f"eq_{p['id']}.png")]
        return f"floor/depth maps missing for stop(s) {miss[:3]}" if miss else None
    if name == "capturer":
        miss = [p["id"] for p in pts if not (ex(frames, "clean", p["filename"]) and ex(frames, "capturer", p["id"] + "_mask.png"))]
        return f"the person is not removed at stop(s) {miss[:3]}" if miss else None
    if name == "privacy":
        try:
            with open(os.path.join(frames, "privacy.json")) as fh:
                pv = json.load(fh)["stops"]
        except Exception:
            return "privacy.json missing"
        return "privacy.json does not cover every stop" if any(p["id"] not in pv for p in pts) else None
    if name == "build":
        try:
            with open(os.path.join(frames, "privacy.json")) as fh:
                pv = json.load(fh)["stops"]
            n = sum(1 for p in pts if ex(frames, "safe", p["filename"]) and (not pv.get(p["id"], {"held": True})["held"] or pv.get(p["id"], {}).get("kept")))
        except Exception:
            return "privacy.json missing"
        try:
            with open(os.path.join(out, "tour.json")) as fh:
                nodes = json.load(fh)["nodes"]
        except Exception:
            return "tour.json missing"
        if len(nodes) != n:
            return "tour.json lists a different number of stops"
        miss = [x["fh"] for x in nodes if not ex(out, x["fh"].split("?")[0]) or not ex(out, x["f"].split("?")[0])]
        return f"{len(miss)} tour pictures missing" if miss else None
    return None


def _pid_alive(pid):
    if pid == os.getpid():
        return True
    try:
        if os.name == "nt":
            import ctypes
            k = ctypes.windll.kernel32
            h = k.OpenProcess(0x1000, False, pid)                    # PROCESS_QUERY_LIMITED_INFORMATION
            if not h:
                return False
            code = ctypes.c_ulong()
            ok = k.GetExitCodeProcess(h, ctypes.byref(code)); k.CloseHandle(h)
            return bool(ok) and code.value == 259                    # STILL_ACTIVE
        os.kill(pid, 0)
        return True
    except PermissionError:
        return True
    except OSError:
        return False


def acquire_lock(frames=None):
    """One tour.py per folder: two runs at once would write the same files. A lock left behind by a process that is gone is ignored."""
    frames = frames or F
    os.makedirs(frames, exist_ok=True)
    lf = os.path.join(frames, ".tour.lock")
    try:
        with open(lf) as fh:
            other = int(fh.read().strip())
        if other != os.getpid() and _pid_alive(other):
            sys.exit(f"another tour.py (process {other}) is already working in this folder; wait for it to finish (or stop it) and run this again.")
    except (OSError, ValueError):
        pass
    with open(lf, "w") as fh:
        fh.write(str(os.getpid()))
    return lf


def release_lock(lf):
    try:
        with open(lf) as fh:
            mine = int(fh.read().strip()) == os.getpid()
        if mine:                                                # removed after closing: Windows cannot delete an open file
            os.remove(lf)
    except (OSError, ValueError):
        pass


def tour_report(nodes, scale_ok=True):
    """Plain-language findings about the finished tour (list of strings): stops without a map position, big gaps between consecutive
    stops (visitors can only jump over a gap), stops with nothing around them to hop to. nodes: poses.json nodes (id, x, y, links)."""
    out = []
    missing = [n["id"] for n in nodes if n.get("x") is None]
    if missing:
        out.append(f"{len(missing)} stop(s) have no place on the map: {', '.join(missing)}")
    pts = [(n["id"], n["x"], n["y"]) for n in nodes if n.get("x") is not None]
    gaps = [(pts[k][0], pts[k + 1][0], ((pts[k + 1][1] - pts[k][1]) ** 2 + (pts[k + 1][2] - pts[k][2]) ** 2) ** 0.5) for k in range(len(pts) - 1)]
    if gaps:
        med = sorted(g[2] for g in gaps)[len(gaps) // 2]
        for a_, b_, d in gaps:
            if d > max(6.0, 3.0 * med):
                out.append(f"stops {a_} -> {b_} are {d:.1f} m apart (typical gap {med:.1f} m): visitors jump over this stretch; record a stop every 1.5-2 m")
    lonely = [n["id"] for n in nodes if n.get("x") is not None and len(n.get("links", [])) < 2]
    if lonely:
        out.append(f"stop(s) with fewer than 2 neighbours to hop to: {', '.join(lonely)}")
    return out


def walks_root():
    """Where videos and their builds live: the SHOP_TOUR_WALKS environment variable, else D:/walks on a Windows PC that has a D: drive, else ~/walks."""
    env = os.environ.get("SHOP_TOUR_WALKS")
    if env:
        return env
    return "D:/walks" if os.name == "nt" and os.path.isdir("D:/") else os.path.join(os.path.expanduser("~"), "walks")


def inside_project(path):
    """True if `path` is the project folder or anything under it (compared by real location, ignoring case on Windows)."""
    a = os.path.normcase(os.path.realpath(os.path.abspath(path)))
    h = os.path.normcase(os.path.realpath(HERE))
    return a == h or a.startswith(h + os.sep)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("insv")
    ap.add_argument("--from", dest="start", help="force this step and every step after it")
    ap.add_argument("--only", help="run just this one step")
    ap.add_argument("--title", default=None, help="name shown in the viewer (remembered for later runs; default Walkthrough)")
    ap.add_argument("--work", default=None, metavar="DIR",
                    help="folder that holds this walk's frames/ and tour/ (default: the folder the video is in, so the build sits next to the footage; "
                         "never inside the project folder: C: holds code only, the walks go under SHOP_TOUR_WALKS)")
    ap.add_argument("--serve", action="store_true")
    ap.add_argument("--lan", action="store_true", help="with --serve: let other devices on your network (a phone) open the tour; default is this computer only")
    ap.add_argument("--allow-partial", action="store_true", help="build the tour even if some stops could not be placed on the map")
    a = ap.parse_args()
    global WORK, F
    inside = [p for p in (a.insv, a.work or os.path.dirname(os.path.abspath(a.insv))) if inside_project(p)]
    if inside:
        sys.exit(f"{inside[0]} is inside the project folder. Footage and builds do not belong with the code: move the video to a folder under {walks_root()} "
                 f"(one folder per place and date, e.g. {os.path.join(walks_root(), 'shop', '2026-10-17')}; the root is the SHOP_TOUR_WALKS environment variable) and run again.")
    WORK = os.path.abspath(a.work or os.path.dirname(os.path.abspath(a.insv)))
    F = os.path.join(WORK, "frames")
    preflight()
    if not os.path.isfile(a.insv):
        sys.exit(f"video not found: {a.insv}")
    from imutil import video_kind, video_streams            # after preflight: imutil needs OpenCV, preflight says nicely if it is missing
    check_video(a.insv)
    kind = video_kind(video_streams(a.insv))
    if kind == "equirect":
        print("note: stitched equirectangular video: the lens calibration and stitching steps are not needed (frames are cut straight from it)")
    insv = video_path(a.insv)
    os.makedirs(WORK, exist_ok=True)
    os.chdir(WORK)                                          # the steps read and write frames/ and tour/ relative to here
    print(f"work folder: {WORK}")
    lock = acquire_lock()
    import atexit
    atexit.register(release_lock, lock)
    if source_changed(os.path.abspath(a.insv)) and not a.only:
        pj = os.path.join(F, "points.json")
        if os.path.exists(pj):                                # never throw away a stop list (it may be hand-picked) without a copy
            shutil.copy2(pj, os.path.join(F, "points_prev.json"))
        print("note: frames/ was built from a different video; rebuilding every step from 'stops' (the lens calibration is kept; "
              "the previous stop list is saved as frames/points_prev.json)")
        a.start = "stops"
    elif not os.path.exists(os.path.join(F, "source.json")):
        remember_source(os.path.abspath(a.insv))               # folders made by older versions: adopt this video
    tf_ = os.path.join(F, "title.txt")                       # the title is remembered, so a later rebuild does not reset it
    if a.title:
        os.makedirs(F, exist_ok=True)
        with open(tf_, "w", encoding="utf-8") as fh:
            fh.write(a.title)
    title = a.title
    if not title and os.path.exists(tf_):
        with open(tf_, encoding="utf-8") as fh:
            title = fh.read().strip() or None
    S = steps(insv, a.allow_partial, title)
    names = [s[0] for s in S]
    for flag in (a.start, a.only):
        if flag and flag not in names:
            sys.exit(f"unknown step {flag!r}; steps: {', '.join(names)}")
    forced = False
    t_all = time.time()
    timings = {}
    for name, ins, outs, cmds in S:
        if a.only and name != a.only:
            continue
        if a.start == name:
            forced = True
        refit = False
        if name == "calib" and kind == "equirect":
            print(f"[{name}] not needed for a stitched video, skipping"); continue
        if name == "calib" and os.path.exists(os.path.join(F, "calib.json")) and a.start != "calib" and a.only != "calib":
            if calib_fits(insv):
                print(f"[{name}] have frames/calib.json, skipping"); continue
            print("note: frames/calib.json was fitted for another lens frame size; fitting the lenses again")
            refit = True
        why = None if (forced or refit or a.only) else incomplete(name)
        if why and any(glob.glob(_gpat(o)) for o in outs):   # a first run has no output yet: nothing to call "incomplete"
            print(f"[{name}] output is incomplete ({why}); redoing it")
        if not (forced or refit or a.only or why or (name == "build" and a.title) or stale(ins, outs)):
            print(f"[{name}] up to date"); continue
        print(f"[{name}] running...", flush=True)
        if name == "stops" and os.path.exists(os.path.join(F, "points.json")):      # the stop list may be hand-picked: keep a copy of whatever is replaced
            shutil.copy2(os.path.join(F, "points.json"), os.path.join(F, "points_prev.json"))
        t0 = time.time()
        for c in cmds():
            r = subprocess.run(c)
            if r.returncode:
                sys.exit(f"[{name}] failed (exit {r.returncode}): {' '.join(c)}\nfix it, then rerun the same command; finished steps are skipped")
        timings[name] = round(time.time() - t0, 1)
        if name == "stops":
            remember_source(os.path.abspath(a.insv))           # only now: the stops of THIS video exist
            if n_stops() < 3:
                sys.exit(f"only {n_stops()} stop(s) found in this video; a tour needs at least 3. Record a longer walk, or take more stops with: "
                         "python find_stops.py <video> --every 3   (then run tour.py again)")
        print(f"[{name}] done in {timings[name]:.0f}s", flush=True)
    if timings:                                             # keep the times of steps this run skipped; a run with nothing to do changes nothing
        tf = os.path.join(F, "timings.json")
        try:
            with open(tf) as fh:
                old = json.load(fh)
        except Exception:
            old = {}
        old.update(timings)
        old.pop("total", None)
        old["total"] = round(sum(v for k, v in old.items() if isinstance(v, (int, float)) and k not in ("stops", "calib")), 1)   # a rebuild; stops and calib are one-off
        with open(tf, "w") as fh:
            json.dump(old, fh, indent=1)
    in_tour = n_stops()
    try:
        with open(os.path.join(WORK, "tour", "tour.json")) as fh:
            in_tour = len(json.load(fh)["nodes"])
    except Exception:
        pass
    print(f"\nall done in {time.time() - t_all:.0f}s: {in_tour} stops -> {os.path.join(WORK, 'tour')}  (step times in frames/timings.json)")
    if n_stops() > in_tour:
        print(f"{n_stops() - in_tour} stop(s) were HELD BACK because they may show a person: see {os.path.join(WORK, 'frames', 'review', 'review.html')}  "
              "(after you have looked: python privacy_gate.py --keep <id>, then rebuild)")
    try:                                                 # what a first-time user needs to know about their tour
        with open(os.path.join(F, "poses.json")) as fh:
            findings = tour_report(json.load(fh)["nodes"])
        print("\ntour check: " + ("looks fine" if not findings else f"{len(findings)} thing(s) to look at"))
        for f_ in findings:
            print("  - " + f_)
    except Exception:
        pass
    release_lock(lock)                                      # serving can go on for hours: do not block the next build
    if a.serve:
        bind = "0.0.0.0" if a.lan else "127.0.0.1"           # the tour shows the inside of a real place: this computer only, unless asked
        if a.lan:
            import socket
            try:
                ip = socket.gethostbyname(socket.gethostname())
            except Exception:
                ip = "<this computer's address>"
            print(f"serving on your network at http://{ip}:8000 (anyone on this Wi-Fi can open it; Ctrl+C to stop)")
        else:
            print("serving at http://localhost:8000 (this computer only; add --lan to open it on a phone; Ctrl+C to stop)")
        subprocess.run([sys.executable, "-m", "http.server", "8000", "--bind", bind, "--directory", os.path.join(WORK, "tour")])
    else:
        print("view it:  python -m http.server 8000 --bind 127.0.0.1 --directory tour   then open http://localhost:8000   (or rerun with --serve)")


if __name__ == "__main__":
    main()
