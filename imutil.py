"""Reading pictures the pipeline just wrote. On Windows a fresh file can be unreadable for a moment (antivirus scan, indexer), and
cv2.imread then returns None, which used to surface as an OpenCV '!ssize.empty()' assertion in the next step. Retry briefly, and
say which file is the problem."""
import os, time
import cv2


def imread(path, flags=cv2.IMREAD_COLOR, tries=8, wait=0.4):
    """cv2.imread that must succeed: waits up to ~tries*wait seconds for a file that is not there or not readable yet, then raises an
    error that names the file (FileNotFoundError if it is missing, IOError if it exists but cannot be decoded)."""
    for k in range(tries):
        img = cv2.imread(path, flags)
        if img is not None:
            return img
        if k < tries - 1:
            time.sleep(wait)
    if not os.path.exists(path):
        raise FileNotFoundError(f"missing picture: {path} (did the previous step finish? rerun the same command)")
    raise IOError(f"cannot read the picture {path} (corrupt or locked by another program?)")


def imwrite(path, img, params=None, tries=6, wait=0.5):
    """cv2.imwrite that must succeed (it returns False instead of raising, e.g. when the file is momentarily locked), and that verifies
    the file is really on disk afterwards."""
    base, ext = os.path.splitext(path)
    tmp = base + ".part" + ext                              # written under another name and renamed: a killed run never leaves a half-written picture
    for k in range(tries):
        try:
            ok = cv2.imwrite(tmp, img, params or [])
        except cv2.error:
            ok = False
        if ok and os.path.exists(tmp) and os.path.getsize(tmp) > 0:
            try:
                os.replace(tmp, path)
                return
            except OSError:
                pass                                        # the target is momentarily locked: try again
        if k < tries - 1:
            time.sleep(wait)
    raise IOError(f"could not write the picture {path} (disk full, no permission, or locked by another program?)")


def total_ram_gb():
    """Installed RAM in GB (Windows, Linux, macOS); 16 if it cannot be found. SHOP_TOUR_RAM_GB overrides it (to test, or to hold the pipeline back)."""
    if os.environ.get("SHOP_TOUR_RAM_GB"):
        return float(os.environ["SHOP_TOUR_RAM_GB"])
    try:
        if os.name == "nt":
            import ctypes

            class MEM(ctypes.Structure):
                _fields_ = [("l", ctypes.c_ulong), ("load", ctypes.c_ulong), ("total", ctypes.c_ulonglong), ("avail", ctypes.c_ulonglong),
                            ("tp", ctypes.c_ulonglong), ("ap", ctypes.c_ulonglong), ("tv", ctypes.c_ulonglong), ("av", ctypes.c_ulonglong),
                            ("ae", ctypes.c_ulonglong)]
            m = MEM(); m.l = ctypes.sizeof(MEM)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
            return m.total / 2 ** 30
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2 ** 30
    except Exception:
        return 16.0


def parallel_jobs(per_job_gb, cap, base_gb=5.0):
    """How many memory-hungry jobs to run at once: leave base_gb for the system and the rest of the pipeline, per_job_gb each, at most cap."""
    return int(max(1, min(cap, (total_ram_gb() - base_gb) // per_job_gb)))


def video_streams(path):
    """Video streams of a file as a list of dicts (codec_type, width, height, duration) via ffprobe; [] if it cannot be read."""
    import json, subprocess
    try:
        out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v", "-show_entries", "stream=codec_type,width,height,duration",
                              "-of", "json", path], capture_output=True, text=True).stdout
        return json.loads(out or "{}").get("streams", [])
    except Exception:
        return []


def video_kind(streams):
    """'dual' = the two fisheye lens streams of an Insta360 .insv; 'equirect' = one already-stitched 2:1 equirectangular video
    (e.g. an Insta360 Studio export); None = something else."""
    if len(streams) >= 2 and (streams[0].get("width"), streams[0].get("height")) == (streams[1].get("width"), streams[1].get("height")):
        return "dual"
    if len(streams) == 1 and streams[0].get("width") and streams[0].get("width") == 2 * (streams[0].get("height") or 0):
        return "equirect"
    return None
