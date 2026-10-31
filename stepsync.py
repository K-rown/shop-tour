"""Steps that start before everything they read is finished, and wait for the rest themselves.

tour.py starts a step as soon as the steps it *needs to begin* are done (tour.EARLY): vio's pictures can be decoded while the stops are still being stitched, the stops'
map can be made while vio still runs, the privacy chain can make its masks while the depth and the floor are still being worked out. The part of such a step that does read
the unfinished step's files waits for it here. tour.py writes frames/.done/<step> when a step ends ("ok", or "failed" when it did not) and tells the steps where
(SHOP_TOUR_DONE_DIR) and who it is (SHOP_TOUR_PARENT_PID). With no SHOP_TOUR_DONE_DIR (a step run by hand, or tour.py without early starts) nothing waits: every step
it needs has finished, as it always was.
"""
import os, time


def done_dir():
    d = os.environ.get("SHOP_TOUR_DONE_DIR", "").strip()
    return d or None


def enabled():
    return done_dir() is not None


def mark(directory, step, status="ok"):
    """Write frames/.done/<step> (whole, in one move)."""
    os.makedirs(directory, exist_ok=True)
    p = os.path.join(directory, step)
    with open(p + ".part", "w") as fh:
        fh.write(status)
    os.replace(p + ".part", p)


def status(step):
    """"ok", "failed" or None (not finished yet) for a step; "ok" when nothing is being waited for (see the top)."""
    d = done_dir()
    if d is None:
        return "ok"
    try:
        with open(os.path.join(d, step)) as fh:
            return fh.read().strip() or "ok"
    except OSError:
        return None


def _alive(pid):
    try:
        if os.name == "nt":
            import ctypes
            from ctypes import wintypes
            k = ctypes.windll.kernel32
            k.OpenProcess.restype = wintypes.HANDLE
            h = k.OpenProcess(0x1000, False, pid)
            if not h:
                return False
            code = wintypes.DWORD()
            ok = k.GetExitCodeProcess(h, ctypes.byref(code)); k.CloseHandle(h)
            return bool(ok) and code.value == 259                         # STILL_ACTIVE
        os.kill(pid, 0)
        return True
    except OSError:
        return False
    except Exception:
        return True


def wait_for(step, poll=0.25, log=None):
    """Wait until `step` has ended. Returns "ok" or "failed" (the step failed, or tour.py is gone: nothing is coming)."""
    s = status(step)
    if s is not None:
        return s
    t0 = time.time()
    try:
        parent = int(os.environ.get("SHOP_TOUR_PARENT_PID", "") or 0)
    except ValueError:
        parent = 0
    while True:
        s = status(step)
        if s is not None:
            if log:
                log(f"  waited {time.time() - t0:.0f} s for {step}")
            return s
        if parent and not _alive(parent):
            return "failed"
        time.sleep(poll)
