"""One command: Insta360 .insv walk -> walkable Matterport-style tour.

  python tour.py D:/walks/place/2026-01-01/walk.insv                 # run everything that isn't already done
  python tour.py D:/walks/place/2026-01-01/walk.insv --from depth    # redo from a step onward (names below)
  python tour.py D:/walks/place/2026-01-01/walk.insv --only stitch   # run just one step
  python tour.py D:/walks/place/2026-01-01/walk.insv --serve         # ...and serve the result at http://localhost:8000

Steps (each is its own script, so you can also run them by hand):
  stops    find_stops.py      pauses in the walk + sharpest frame at each      -> frames/points.json
  calib    calibrate_stitch   fit the two lenses (only if frames/calib.json is missing)
  stitch   stitch_points.py   8K seam-cut panoramas                            -> frames/points/*.jpg
  vio      vio_map.py         the camera path every second (for the scale)     -> frames/vio/poses.npz
  level    level_panos.py     true "up" for each pano                          -> frames/up.json
  hops     hop_geometry.py    geometry between nearby stops                    -> frames/hops.json
  map      map_stops.py       one shared map (COLMAP, 360 camera); true scale from the accelerometer (imu_scale.py) -> frames/poses_map.json
  depth    depth_panos.py     per-photo depth (Depth Anything V2 Small)        -> frames/depth/*_v*.npy
  floor    floor_panos.py     which pixels are floor (UperNet ADE20K)          -> frames/depth/*_f*.png
  rings    place_rings.py     floor height, occlusion, floor/depth maps        -> frames/poses.json
  capturer remove_capturer.py the person holding the stick is painted out         -> frames/clean/*.jpg
  privacy  privacy_gate.py    bystanders, faces, leftovers: a stop that fails is held back -> frames/privacy.json
                              (SHOP_TOUR_PEOPLE=blur, the default: people stay in the picture, smudged, and nothing is held; =remove: the paint-out and the holds described here; =keep: people untouched)
           privacy_margin.py  (same step, right after the gate) a stop that passed but WOULD be held with its camera height measured 7% lower or higher
                              is held too: the gate must not pass or hold a stop by luck of a centimetre. Only adds holds. SHOP_TOUR_PRIVACY_MARGIN=0 turns it off.
                              It also puts his answers (left out / kept) back after a re-scan and writes frames/privacy_margin.json as its LAST act: without
                              that marker the step is not done, and a carry-on runs just the margin again (see privacy_answers.py)
  pii      pii_blur.py        plates, faces and printed names blurred in every kept picture -> frames/pii.json (in keep / blur mode privacy_chain.py does it as each stop's
                              picture is final, and this step finds itself up to date; SHOP_TOUR_PII_STREAM=0 runs it after the chain)
  early    build_early.py     (only with SHOP_TOUR_EARLY_TOUR=1, set by an app that wants the tour sooner) the tour as soon as the pictures are cleaned and placed, without
                              dollhouse or plan, in tour_early/ (partial: true in its tour.json); the final tour/ is built later and replaces it. Never fails a run.
  sphere   depth_sphere.py    one metric full-sphere depth per stop (MoGe-2 on 18 views, one scale per view, the stick height ties the stops) -> frames/depth_sphere/
  register register_stops.py  the stops moved to where their depth agrees (ICP + a pose graph): x, y, heading in frames/poses.json
  mvs      mvs.py             multi-view stereo between the stops' pictures    -> frames/mvs/*.npz
  align    depth_align.py     the depth model's depth put on the stereo depth  -> frames/depth_mvs/
  build    build_tour.py      the tour folder (held-back stops are left out)   -> tour/
  dollhouse dollhouse.py      every stop's depth fused into one coloured model (dollhouse_texture.py) -> tour/dollhouse.bin
  plan     floorplan.py       floor plan with dimensions from the model        -> tour/plan.svg
A step runs when any of its outputs is missing or older than its inputs.
Steps run as soon as the steps they read from are done, several at once (tour.PREREQ is the list): vio (all CPU) runs beside everything it does not feed
and map waits for it; level -> hops runs beside depth -> floor; build runs beside dollhouse. A step that runs beside others writes frames/run_logs/<step>.log
(vio: frames/vio_run.log) and its lines are printed as "    <step>| ...". Every step still prints "[step] running..." and "[step] done in Ns".
frames/timings.json also keeps "front" (start to the capturer's start), "wall" and "parallel_hidden"; frames/timings_spans.json has every step's start and end.
Early starts (SHOP_TOUR_EARLY=0 turns them off, tour.EARLY is the list): vio begins when the lens fit is done (it reads the stops' pictures when it has them), the stops'
map begins when the hops are done (it takes vio's path when vio is done), and the privacy chain begins when the levelling is done (it makes its masks while the hops and the
depth are being worked out, and begins the floors when the hops are done). The step waits for the rest itself (stepsync.py).
SHOP_TOUR_SERIAL=1 runs every step one after the other.
By default (SHOP_TOUR_PRIVACY_INPUTS=own, see privacy_inputs.py; =map is the old way) the capturer and privacy make read nothing from vio or the map (a stick height and the neighbours from
hops.json), so they run beside vio too, and turns on fast matching and the global mapper by default (they cannot change which stops are held then).
"""
import argparse, glob, json, os, shutil, subprocess, sys, threading, time

import privacy_inputs as PI
import stepsync

HERE = os.path.dirname(os.path.abspath(__file__))
WORK = HERE                                  # where this walk's frames/ and tour/ live (--work); the code always comes from HERE
F = os.path.join(WORK, "frames")


def _abs(p):
    """Steps list data as frames/... or tour/... (this walk's folder); everything else (viewer/, *.py) is code in HERE."""
    return os.path.join(WORK if p.split("/")[0] in ("frames", "tour", "tour_early") else HERE, p)


def py(script, *args):
    return [sys.executable, os.path.join(HERE, script), *args]


def _gpat(p):
    """_abs(p) as a glob pattern: the folder part is taken literally (a work folder called "shop [day 1]" must not be read as a character class)."""
    head, _, rest = p.partition("/")
    base = WORK if head in ("frames", "tour", "tour_early") else HERE
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
    cmds = [py("grab_pairs.py", insv, src, *[str(t) for t in ts])]            # the three lens pairs, on the video card when it can (same pixels as the software decoder)
    cmds.append(py("calibrate_stitch.py", "--src", src, "--out", os.path.join(F, "calib.json")))
    return cmds


def steps(insv, allow_partial=False, title=None):
    return [
        ("stops", [insv], ["frames/points.json"], lambda: [py("find_stops.py", insv)]),
        ("calib", [], ["frames/calib.json"], lambda: calib_cmds(insv)),
        ("stitch", [insv, "frames/points.json", "frames/calib.json", "stitch_lib.py"], ["frames/points/*.jpg"],
         lambda: [py("stitch_points.py", insv)]),
        ("vio", [insv, "frames/points/*.jpg", "frames/calib.json", "vio_map.py"], ["frames/vio/poses.npz"], lambda: [py("vio_map.py", insv)]),
        ("level", [insv, "frames/points/*.jpg"], ["frames/up.json"], lambda: [py("level_panos.py", "--video", insv)]),
        ("hops", ["frames/points/*.jpg", "frames/up.json"], ["frames/hops.json"], lambda: [py("hop_geometry.py")]),
        ("map", ["frames/points/*.jpg", "frames/up.json", "frames/hops.json", "frames/vio/poses.npz"], ["frames/poses_map.json"],
         lambda: [py("map_stops.py", "--video", insv, *(["--allow-partial"] if allow_partial else []))]),
        ("depth", ["frames/points/*.jpg"], ["frames/depth/views.json"], lambda: [py("depth_panos.py")]),
        ("floor", ["frames/points/*.jpg", "frames/depth/views.json"], ["frames/depth/*_f0.png"], lambda: [py("floor_panos.py")]),
        ("rings", ["frames/poses_map.json", "frames/depth/views.json", "frames/depth/*_f0.png", "place_rings.py"],
         ["frames/depth/eq_*.png"], lambda: [py("place_rings.py")]),
        ("capturer", ["frames/points/*.jpg", "frames/up.json", "frames/hops.json", "capturer.py", "remove_capturer.py", "remove_others.py", "synth_fill.py", "people_mode.py", "people_blur.py", "privacy_inputs.py",
                      "privacy_chain.py", "privacy_gate.py", "privacy_margin.py", "privacy_answers.py", "neighbour_check.py", "qa/check_capturer.py"], ["frames/clean/*.jpg"],
         lambda: [py("privacy_chain.py")]),           # one process: the capturer, then the gate and the margin on each stop as its picture lands (2026-10-21);
                                                      # it leaves privacy.json and the margin's marker complete, so the privacy step below finds itself current
        ("privacy", ["frames/clean/*.jpg", "privacy_gate.py", "capturer.py", "people_mode.py", "people_blur.py", "privacy_inputs.py", "privacy_margin.py", "neighbour_check.py", "qa/check_capturer.py"], ["frames/privacy.json", "frames/privacy_margin.json"],
         lambda: privacy_cmds()),
        ("pii", ["frames/safe/*.jpg", "pii_blur.py", "privacy_gate.py", "remove_others.py"], ["frames/pii.json"], lambda: [py("pii_blur.py")]),    # plates, faces and printed names blurred in every kept picture; never a gate
        ("early", ["frames/privacy.json", "frames/pii.json", "frames/poses.json", "frames/hops.json", "viewer/index.html", "build_tour.py", "build_early.py"], ["tour_early/manifest.json"],
         lambda: [py("build_early.py", *(["--title", title] if title else []))]),     # the tour as soon as it is cleaned and placed, in tour_early/ (never in tour/); only when SHOP_TOUR_EARLY_TOUR=1; never fails the run
        ("sphere", ["frames/safe/*.jpg", "frames/poses_map.json", "frames/depth/eq_*.png", "depth_sphere.py"], ["frames/depth_sphere/stats.json"], lambda: [py("depth_sphere.py")]),
        ("register", ["frames/depth_sphere/stats.json", "frames/capturer/*_hole.png", "register_stops.py", "heading_check.py"], ["frames/register/stats.json"], lambda: [py("register_stops.py")]),
        ("mvs", ["frames/safe/*.jpg", "frames/poses.json", "mvs.py"], ["frames/mvs/stats.json"], lambda: [py("mvs.py")]),
        ("align", ["frames/mvs/stats.json", "frames/depth/eq_*.png", "depth_align.py"], ["frames/depth_mvs/stats.json"], lambda: [py("depth_align.py")]),
        ("build", ["frames/poses.json", "frames/depth/eq_*.png", "frames/depth_mvs/stats.json", "frames/depth_sphere/stats.json", "frames/register/stats.json", "frames/privacy.json", "viewer/index.html", "build_tour.py"],
         ["tour/tour.json"], lambda: [py("build_tour.py", *(["--title", title] if title else []))]),
        ("dollhouse", ["frames/safe/*.jpg", "frames/poses.json", "frames/depth_mvs/stats.json", "frames/depth_sphere/stats.json", "frames/register/stats.json", "frames/privacy.json",
                       "dollhouse.py", "dollhouse_texture.py", "dollhouse_uv.py"], ["tour/dollhouse.json"],
         lambda: [py("dollhouse.py")]),
        ("plan", ["frames/dollhouse/mesh.ply", "frames/dollhouse/mesh_uv.npz", "floorplan.py"], ["tour/plan.svg"], lambda: [py("floorplan.py")]),
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


def incomplete(name, frames=None, out=None, margin=True):
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
        if any(p["id"] not in pv for p in pts):
            return "privacy.json does not cover every stop"
        return privacy_margin_pending(frames) if margin else None
    if name == "pii":
        try:
            with open(os.path.join(frames, "pii.json")) as fh:
                pj = json.load(fh)
        except Exception:
            return "pii.json missing"
        if pj.get("enabled", True) is False:
            return None
        if pj.get("streamed") and pj.get("errors"):                    # the blur that rode along with the privacy chain could not look at some stops: the step looks again
            return f"the blur beside the privacy chain could not look at stop(s) {pj['errors'][:3]}"
        miss = [p["id"] for p in pts if ex(frames, "safe", p["filename"]) and p["id"] not in pj.get("stops", {})]
        return f"the blur step has not looked at stop(s) {miss[:3]}" if miss else None
    if name == "build":
        try:
            with open(os.path.join(frames, "privacy.json")) as fh:
                pv = json.load(fh)["stops"]
            n = sum(1 for p in pts if ex(frames, "safe", p["filename"]) and (not pv.get(p["id"], {"held": True})["held"] or pv.get(p["id"], {}).get("kept")))
        except Exception:
            return "privacy.json missing"
        try:
            with open(os.path.join(out, "tour.json")) as fh:
                tj = json.load(fh)
            nodes = tj["nodes"]
        except Exception:
            return "tour.json missing"
        if tj.get("held_all"):
            return "tour.json is the placeholder written when every stop was held back"
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


def _pid_image(pid):
    """The program name (lower case, no folder) of a running process, or None when it cannot be asked."""
    try:
        if os.name == "nt":
            import ctypes
            from ctypes import wintypes
            k = ctypes.windll.kernel32
            k.OpenProcess.restype = wintypes.HANDLE
            h = k.OpenProcess(0x1000, False, pid)                    # PROCESS_QUERY_LIMITED_INFORMATION
            if not h:
                return None
            try:
                buf = ctypes.create_unicode_buffer(1024)
                n = wintypes.DWORD(len(buf))
                if not k.QueryFullProcessImageNameW(h, 0, buf, ctypes.byref(n)):
                    return None
                return os.path.basename(buf.value).lower()
            finally:
                k.CloseHandle(h)
        with open(f"/proc/{pid}/comm") as fh:
            return fh.read().strip().lower()
    except Exception:
        return None


def _pid_is_python(pid):
    """Is this process a Python program? A lock names a process number only, and Windows hands a dead process's number to the next
    program: a lock whose number is now (say) a browser is not a tour.py at work. When the program cannot be told, it counts as Python
    (the careful answer: two runs in one folder write the same files)."""
    img = _pid_image(pid)
    return True if img is None else img.startswith("python") or img.startswith("pypy")


def acquire_lock(frames=None):
    """One tour.py per folder: two runs at once would write the same files. A lock left behind by a process that is gone, or whose number now belongs to a program that is not Python, is ignored."""
    frames = frames or F
    os.makedirs(frames, exist_ok=True)
    lf = os.path.join(frames, ".tour.lock")
    try:
        with open(lf) as fh:
            other = int(fh.read().strip())
        if other != os.getpid() and _pid_alive(other) and _pid_is_python(other):
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


_PRIVACY_MARGIN_ONLY = [False]


def privacy_cmds():
    """The privacy step's commands. Normally: keep a copy of the earlier privacy.json (his answers and the margin's holds), run the gate, then privacy_margin.py,
    which puts his answers back, adds its holds, and as its LAST act writes frames/privacy_margin.json. When only the margin is owed (the gate's result is
    current but the marker does not vouch for it: the margin was cut off, or its version changed) run just privacy_margin.py on top of the existing
    privacy.json - never the gate again, never anything that drops his answers."""
    m = lambda *a: py("privacy_margin.py", *a)
    return [m()] if _PRIVACY_MARGIN_ONLY[0] else [m("--snapshot"), py("privacy_gate.py"), m()]


def privacy_margin_pending(frames=None):
    """Why the privacy margin (and the restoring of his answers) is not finished for the current privacy.json, or None."""
    import privacy_answers, privacy_margin
    return privacy_answers.pending(frames or F, "on" if privacy_margin_on() else "off", privacy_margin.SCRIPT_VERSION)


def privacy_margin_only(forced, only, ins, frames=None):
    """True when the privacy step needs nothing but the margin: not forced or asked for by name, the gate's result is current (not older than its
    inputs) and covers every stop, and the margin's marker is what is missing."""
    if forced or only:
        return False
    frames = frames or F
    if stale(ins, ["frames/privacy.json"]) or incomplete("privacy", frames, None, margin=False):
        return False
    return privacy_margin_pending(frames) is not None


def privacy_margin_on():
    """The privacy step also runs privacy_margin.py (holds a stop that passed but would be held with its camera height measured a little differently).
    SHOP_TOUR_PRIVACY_MARGIN=0 turns it off."""
    return os.environ.get("SHOP_TOUR_PRIVACY_MARGIN", "").strip().lower() not in ("0", "false", "no", "off")


def serial_run():
    """SHOP_TOUR_SERIAL=1 runs every step one after the other, the way it always did (vio is the only step that normally runs beside others)."""
    return os.environ.get("SHOP_TOUR_SERIAL", "").strip().lower() not in ("", "0", "false", "no", "off")


# What each step has to wait for when steps run beside each other. These are the DATA dependencies read off steps() (each step's inputs and who writes them), not the
# pipeline order: level -> hops and depth -> floor are two independent chains over the panoramas; capturer needs hops.json (the floor step's masks were read and never used); vio needs the
# panoramas and the lens fit only; map is the first step that reads vio's camera path; rings needs the map and the floor; mvs needs the placed stops (poses.json,
# which rings updates) and the cleaned pictures (safe/, written by privacy); build and dollhouse both read align's depth and privacy.json and do not read each other
# (build_tour.py only keeps a link to a dollhouse that exists, dollhouse.py adds the link to tour.json when it finishes); plan reads dollhouse's mesh.
PREREQ = {
    "calib": ["stops"], "stitch": ["calib"],
    "vio": ["stitch"], "level": ["stitch"], "hops": ["level"],
    "depth": ["stitch"], "floor": ["depth"],
    "map": ["hops", "vio"],
    "rings": ["map", "floor"],
    "capturer": ["hops"],                                       # (the floor step's masks were read by the fill and never used: it no longer waits for them)
    "privacy": ["capturer"],
    "pii": ["privacy"],
    "early": ["pii", "rings"],                                  # the first tour: the cleaned pictures and the rings' positions; its own folder, so nothing later races it
    "sphere": ["rings", "pii", "early"],                        # (waits for early so poses.json is not rewritten under it)                                 # reads the cleaned pictures (frames/safe), and MoGe gets the card after the privacy chain's models
    "register": ["sphere", "capturer"],                         # the clouds leave the person out (the capturer's hole masks)
    "mvs": ["register", "pii"], "align": ["mvs"],
    "build": ["align", "pii", "register"], "dollhouse": ["align", "pii", "register"],
    "plan": ["dollhouse"],
}


def prereqs(name, names):
    """The steps `name` waits for (only those that are in this run's step list)."""
    want = list(PREREQ.get(name, []))
    if name == "capturer" and PI.mode() == "map":                # the old way reads the map's poses.json: the privacy steps stay behind map
        want.append("map")
    return [p for p in want if p in names]


# Early starts (2026-10-22). A step may BEGIN before every step it reads from is done when the part of it that comes first does not read them: vio decodes its pictures (it
# needs the lens fit only) while the stops are still being stitched and reads the stops' pictures when it has them; the stops' map makes its features, matches and map
# (pictures only) while vio still runs and takes vio's path when vio is done; the privacy chain loads its models and makes its masks (levelled pictures only) while the
# hops are still being worked out and begins the floors when they are done. A step started early waits for the rest through stepsync (frames/.done/<step>, written
# here the moment a step ends), so it never reads a file of a step that is still writing. Every step reads exactly what it read before: only the time it starts changes.
# SHOP_TOUR_EARLY=0 starts every step when all of PREREQ is done, as before.
EARLY = {"vio": ["calib"], "map": ["hops"], "capturer": ["level"]}


def early_default():
    return os.environ.get("SHOP_TOUR_EARLY", "").strip().lower() not in ("0", "false", "no", "off")


def start_after(name, names, early=False):
    """The steps `name` waits for before it BEGINS (see EARLY): the data dependencies, or with early starts only the ones that its first part needs."""
    if early and name in EARLY and not (name == "capturer" and PI.mode() == "map"):         # (the old privacy way reads the map's poses.json: it stays behind map)
        return [p for p in EARLY[name] if p in names]
    return prereqs(name, names)


def _kill_tree(proc):
    try:
        if os.name == "nt":
            subprocess.run(["taskkill", "/T", "/F", "/PID", str(proc.pid)], capture_output=True)
        else:
            proc.kill()
    except Exception:
        pass


class Background:
    """A step running beside the others. Its output goes to a log file next to frames/ (not frames/vio/: vio_map.py clears that folder on start); the main
    thread prints it as it grows. Its commands run one after another; the first failure stops it."""
    def __init__(self, name, cmds, log_path):
        self.name, self.cmds, self.log_path = name, cmds, log_path
        self.t0, self.t1 = time.time(), None
        self.failed, self.proc, self.cancelled = None, None, False
        self.shown = 0
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        self.th = threading.Thread(target=self._run, daemon=True)
        self.th.start()

    def _run(self):
        env = dict(os.environ, PYTHONUNBUFFERED="1")
        try:
            with open(self.log_path, "w", encoding="utf-8", errors="replace") as lf:
                for c in self.cmds:
                    if self.cancelled:
                        return
                    self.proc = subprocess.Popen(c, stdout=lf, stderr=subprocess.STDOUT, env=env)
                    rc = self.proc.wait()
                    if rc:
                        self.failed = (rc, c)
                        return
        except Exception as e:                                   # the command could not even start
            self.failed = (-1, [str(e)])
        finally:
            self.t1 = time.time()

    def running(self):
        return self.th.is_alive()

    def kill(self):
        self.cancelled = True
        if self.proc is not None and self.proc.poll() is None:
            _kill_tree(self.proc)
        self.th.join(10)

    def text(self):
        try:
            with open(self.log_path, encoding="utf-8", errors="replace") as fh:
                return fh.read()
        except OSError:
            return ""

    def show(self, multi, final=False):
        """Print what the step has said since the last call. Alone, as it comes (like a step run in line); beside others, whole lines each marked with the
        step's name (lines that do not start with "[step]" are ignored by the app's progress reader)."""
        out = self.text()
        new = out[self.shown:]
        if not new:
            return
        if not multi:
            self.shown = len(out)
            print(new.rstrip(chr(10)), flush=True)
            return
        if not final:
            cut = new.rfind(chr(10))
            if cut < 0:
                return
            new = new[:cut + 1]
        self.shown += len(new)
        for ln in new.rstrip(chr(10)).splitlines():
            print(f"    {self.name}| {ln}", flush=True)

    def tail(self, n=12):
        return "\n".join(self.text().rstrip().splitlines()[-n:])


def _union_len(spans):
    """Total length of the union of (start, end) intervals."""
    tot, cur = 0.0, None
    for a, b in sorted(spans):
        if cur is None or a > cur[1]:
            if cur:
                tot += cur[1] - cur[0]
            cur = [a, b]
        else:
            cur[1] = max(cur[1], b)
    return tot + (cur[1] - cur[0] if cur else 0.0)


def run_steps(S, kind, insv, orig_insv, start=None, only=None, title=None, serial=False, early=False):
    """Run the steps that are not up to date. Returns {step: seconds}; exits (SystemExit with a message) on the first failure.
    Unless serial (or --only), steps run as soon as the steps they read from are done, beside each other (PREREQ): vio beside everything it does not feed,
    level -> hops beside depth -> floor, build beside dollhouse. Each one's start and end are printed as "[step] running..." / "[step] done in Ns"."""
    names = [s[0] for s in S]
    byname = {s[0]: s for s in S}
    after = set(names[names.index(start):]) if start in names else set()      # --from X: X and every step after it in PIPELINE order
    parallel = (not serial) and not only
    early = bool(early) and parallel
    done_dir = os.path.join(F, ".done") if early else None
    if early:
        shutil.rmtree(done_dir, ignore_errors=True); os.makedirs(done_dir, exist_ok=True)
        os.environ["SHOP_TOUR_DONE_DIR"] = done_dir; os.environ["SHOP_TOUR_PARENT_PID"] = str(os.getpid())

    def prepare(name):
        """Decide what step `name` does now: None when it is skipped (the reason is printed), else its commands, after printing "[name] running..."."""
        _, ins, outs, cmds = byname[name]
        forced = name in after
        refit = False
        if name == "calib" and kind == "equirect":
            print(f"[{name}] not needed for a stitched video, skipping"); return None
        if name == "vio" and kind == "equirect":
            # the 1 Hz path is only the accelerometer's half of the scale, and a stitched video carries no lens fit and no
            # accelerometer: the stops' map takes the camera-height prior instead (map_stops.py, no frames/vio/poses.npz)
            print(f"[{name}] not needed for a stitched video (no lens fit, no motion sensors), skipping"); return None
        if name == "calib" and os.path.exists(os.path.join(F, "calib.json")) and start != "calib" and only != "calib":
            if calib_fits(insv):
                print(f"[{name}] have frames/calib.json, skipping"); return None
            print("note: frames/calib.json was fitted for another lens frame size; fitting the lenses again")
            refit = True
        why = None if (forced or refit or only) else incomplete(name)
        if why and any(glob.glob(_gpat(o)) for o in outs):   # a first run has no output yet: nothing to call "incomplete"
            print(f"[{name}] output is incomplete ({why}); redoing it")
        if not (forced or refit or only or why or (name == "build" and title) or stale(ins, outs)):
            print(f"[{name}] up to date"); return None
        if name == "privacy":
            _PRIVACY_MARGIN_ONLY[0] = privacy_margin_only(forced, only, ins)
            if _PRIVACY_MARGIN_ONLY[0]:
                print(f"[{name}] the gate's result is current; only the margin is owed ({privacy_margin_pending()})")
        print(f"[{name}] running...", flush=True)
        if name == "stops" and os.path.exists(os.path.join(F, "points.json")):      # the stop list may be hand-picked: keep a copy of whatever is replaced
            shutil.copy2(os.path.join(F, "points.json"), os.path.join(F, "points_prev.json"))
        try:
            return cmds()
        finally:
            _PRIVACY_MARGIN_ONLY[0] = False                     # only for the commands of this one run of the step

    def finish(name, secs):
        if name == "stops":
            remember_source(os.path.abspath(orig_insv))           # only now: the stops of THIS video exist
            if n_stops() < 3:
                sys.exit(f"only {n_stops()} stop(s) found in this video; a tour needs at least 3. Record a longer walk, or take more stops with: "
                         "python find_stops.py <video> --every 3   (then run tour.py again)")
        print(f"[{name}] done in {secs:.0f}s", flush=True)

    timings, spans = {}, {}
    t_run = time.time()
    if not parallel:
        for name in names:
            if only and name != only:
                continue
            cmds = prepare(name)
            if cmds is None:
                continue
            t0 = time.time()
            for c in cmds:
                r = subprocess.run(c)
                if r.returncode:
                    sys.exit(f"[{name}] failed (exit {r.returncode}): {' '.join(c)}\nfix it, then rerun the same command; finished steps are skipped")
            timings[name] = round(time.time() - t0, 1)
            finish(name, timings[name])
        if "vio" in timings:
            timings["vio_hidden"] = 0.0
        return timings

    state = {n: "wait" for n in names}                           # wait | run | done | failed | blocked
    running = {}

    def fresh(n):
        """True when step n is certain to run (forced, or nothing of its outputs exists yet): only then may it begin early - whether a step is up to date is decided
        from the files of the steps it reads, which an early start has not waited for."""
        return n in after or not any(glob.glob(_gpat(o)) for o in byname[n][2])

    def settle(n, st):
        """A step's state changes; with early starts the steps that wait for it are told the moment it ends."""
        state[n] = st
        if early and st in ("done", "failed", "blocked"):
            stepsync.mark(done_dir, n, "ok" if st == "done" else "failed")
    first_fail = None                                            # a vio failure lets what does not need it finish; any other failure stops everything
    last_show = 0.0

    def log_path(name):
        return os.path.join(F, "vio_run.log") if name == "vio" else os.path.join(F, "run_logs", name + ".log")

    def show_all(final_for=None):
        multi = len(running) > 1
        for n, b in list(running.items()):
            b.show(multi, final=(n == final_for))

    try:
        while True:
            progressed = True
            while progressed:
                progressed = False
                for n in names:                                  # in pipeline order: ready steps start in the order the app lists them
                    if state[n] != "wait":
                        continue
                    pre = start_after(n, names, early and fresh(n))
                    if any(state[p] in ("failed", "blocked") for p in pre):
                        settle(n, "blocked"); progressed = True; continue
                    if any(state[p] != "done" for p in pre):
                        continue
                    cmds = prepare(n)
                    if cmds is None:
                        settle(n, "done"); progressed = True; continue
                    if n == "capturer":
                        timings["front"] = round(time.time() - t_run, 1)      # how long everything before the capturer took
                    running[n] = Background(n, cmds, log_path(n))
                    state[n] = "run"; progressed = True
            if not running:
                break
            time.sleep(0.2)
            if time.time() - last_show >= 1.0:
                show_all(); last_show = time.time()
            for n in [k for k, b in running.items() if not b.running()]:
                b = running[n]
                show_all(final_for=n)
                del running[n]
                spans[n] = (b.t0 - t_run, b.t1 - t_run)
                if b.failed:
                    rc, c = b.failed
                    settle(n, "failed")
                    msg = f"[{n}] failed (exit {rc}): {' '.join(c)}\n{b.tail()}\nfix it, then rerun the same command; finished steps are skipped"
                    if n == "vio" or (n == "map" and state.get("vio") == "failed"):         # (a map that began early and found vio failed: vio's message is the news)
                        first_fail = first_fail or msg
                        continue
                    sys.exit(msg)
                timings[n] = round(b.t1 - b.t0, 1)
                settle(n, "done")
                finish(n, timings[n])
        if first_fail is not None:
            sys.exit(first_fail)
    finally:
        for b in running.values():                               # a step failed (or Ctrl+C): do not leave the others working
            if b.running():
                b.kill()
        if early:
            os.environ.pop("SHOP_TOUR_DONE_DIR", None); os.environ.pop("SHOP_TOUR_PARENT_PID", None)
    others = [v for k, v in spans.items() if k != "vio"]
    if "vio" in spans:                                           # how much of vio ran beside other steps: the total must not count it twice
        v0, v1 = spans["vio"]
        inter = _union_len([(max(a, v0), min(b, v1)) for a, b in others if min(b, v1) > max(a, v0)])
        timings["vio_hidden"] = round(inter, 1)
    timings["parallel_hidden"] = round(max(0.0, sum(b - a for a, b in others) - _union_len(others)), 1)     # steps (not vio) that ran beside each other
    timings["wall"] = round(time.time() - t_run, 1)
    try:
        with open(os.path.join(F, "timings_spans.json"), "w") as fh:
            json.dump({k: [round(a, 1), round(b, 1)] for k, (a, b) in sorted(spans.items(), key=lambda kv: kv[1])}, fh, indent=1)
    except OSError:
        pass
    return timings


def write_timings(timings, frames):
    """frames/timings.json: every step's own wall time; "total" is the wall time of a whole rebuild (stops and calib are one-off, and the part of
    vio that ran beside other steps is counted once)."""
    tf = os.path.join(frames, "timings.json")
    try:
        with open(tf) as fh:
            old = json.load(fh)
    except Exception:
        old = {}
    old.update(timings)
    old.pop("total", None)
    steps_t = sum(v for k, v in old.items() if isinstance(v, (int, float)) and k not in ("stops", "calib", "vio_hidden", "parallel_hidden", "wall", "front"))
    old["total"] = round(steps_t - float(old.get("vio_hidden", 0) or 0) - float(old.get("parallel_hidden", 0) or 0), 1)
    with open(tf, "w") as fh:
        json.dump(old, fh, indent=1)
    return old


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
                 f"(one folder per place and date, e.g. {os.path.join(walks_root(), 'place', '2026-01-01')}; the root is the SHOP_TOUR_WALKS environment variable) and run again.")
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
    t_all = time.time()
    timings = run_steps(S, kind, insv, a.insv, start=a.start, only=a.only, title=title, serial=serial_run(), early=early_default())
    if timings:                                             # keep the times of steps this run skipped; a run with nothing to do changes nothing
        write_timings(timings, F)
    in_tour = n_stops()
    try:
        with open(os.path.join(WORK, "tour", "tour.json")) as fh:
            in_tour = len(json.load(fh)["nodes"])
    except Exception:
        pass
    try:                                                 # the stable hand-off for anything that opens or imports this tour (README: manifest.json)
        import manifest
        manifest.write(F, os.path.join(WORK, "tour"))
    except Exception as e:                               # a tour without a manifest still opens; apps that import tours will say it is missing
        print(f"note: manifest.json was not written ({e})")
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
        subprocess.run([sys.executable, os.path.join(HERE, "serve_tour.py"), os.path.join(WORK, "tour"), "--bind", bind])
    else:
        print(f"view it:  python serve_tour.py {os.path.join(WORK, 'tour')}   then open http://localhost:8000   (this also saves pinned notes next to the tour; or rerun with --serve)")


if __name__ == "__main__":
    main()
