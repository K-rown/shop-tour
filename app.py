"""The upload button: a page on this computer with one "Choose your 360 video" button. Start it with start.bat / start.sh (or: python app.py).

It copies the video into the walks folder (one folder per place and date, the same place tour.py builds in), runs  tour.py <video> --title <title>
in the background, shows its progress, asks you about every stop the privacy check held back (never decided for you), and serves the finished tour.
Python standard library only. It answers this computer only (127.0.0.1), one build at a time.

  python app.py [--port 8765] [--no-browser]
"""
import argparse, hashlib, html, json, mimetypes, os, re, shutil, signal, subprocess, sys, threading, time, urllib.parse, webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import serve_tour                                      # noqa: E402  (the notes check and the atomic write, shared with python serve_tour.py)

TOUR_PY = os.environ.get("SHOP_TOUR_APP_TOUR") or os.path.join(HERE, "tour.py")            # the tests swap in a stand-in engine
GATE_PY = os.environ.get("SHOP_TOUR_APP_GATE") or os.path.join(HERE, "privacy_gate.py")
VIDEO_EXT = (".insv", ".mp4")
RESERVE = float(os.environ.get("SHOP_TOUR_APP_RESERVE_GB", "2")) * 2**30   # room the build needs beside the video
BOX_TYPES = (b"ftyp", b"free", b"wide", b"skip", b"mdat", b"moov")   # the first box of an MP4-family file (an .insv is one too)
STEPS = ["stops", "calib", "stitch", "vio", "level", "hops", "map", "depth", "floor", "rings", "capturer", "privacy", "pii", "early",
         "sphere", "register", "mvs", "align", "build", "dollhouse", "plan"]
WORDS = {"stops": "Finding where you stood still", "calib": "Fitting the camera's two lenses", "stitch": "Stitching the 360 pictures",
         "vio": "Following the camera's path", "level": "Levelling each picture", "hops": "Working out how the stops connect",
         "map": "Putting every stop on one map", "depth": "Measuring depth", "floor": "Finding the floor", "rings": "Placing the floor rings",
         "capturer": "Taking you out of the pictures", "privacy": "Checking every stop for people", "pii": "Blurring faces, plates and names",
         "early": "A first look at the tour", "sphere": "Measuring depth all round", "register": "Lining the stops up",
         "mvs": "Stereo depth", "align": "Matching the depth", "build": "Making the tour", "dollhouse": "Making the dollhouse", "plan": "Drawing the floor plan"}
LINE = re.compile(r"^\s*(?:\w+\|\s*)?\[(\w+)\]\s+(.*)$")


def walks_root():
    """The same rule as tour.py: SHOP_TOUR_WALKS, else D:/walks on a Windows PC with a D: drive, else ~/walks."""
    env = os.environ.get("SHOP_TOUR_WALKS")
    if env:
        return env
    return "D:/walks" if os.name == "nt" and os.path.isdir("D:/") else os.path.join(os.path.expanduser("~"), "walks")


def slug(text, fallback):
    s = re.sub(r"[^A-Za-z0-9]+", "-", text or "").strip("-").lower()[:40]
    return s or fallback


def safe_name(name):
    base = os.path.basename((name or "").replace("\\", "/"))
    stem, ext = os.path.splitext(base)
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", stem).strip("._") or "walk"
    return stem[:80] + ext.lower()


def walk_folder(title):
    """walks/<place>/<date>/, a new one when that folder already holds a video (one video per folder)."""
    base = os.path.join(walks_root(), slug(title, "my-tour"), time.strftime("%Y-%m-%d"))
    d, n = base, 2
    while os.path.isdir(d) and any(f.lower().endswith(VIDEO_EXT) for f in os.listdir(d)):
        d, n = f"{base}-{n}", n + 1
    return d


def tour_id(walk_dir):
    return hashlib.sha1(os.path.normcase(os.path.abspath(walk_dir)).encode("utf-8")).hexdigest()[:12]


def find_tours():
    """Every built tour under the walks folder (walks/<place>/<date>/tour/index.html), newest first."""
    root, out = walks_root(), []
    if not os.path.isdir(root):
        return out
    for place in os.listdir(root):
        p = os.path.join(root, place)
        if not os.path.isdir(p) or place.startswith((".", "_")):
            continue
        for day in os.listdir(p):
            w = os.path.join(p, day)
            if os.path.isfile(os.path.join(w, "tour", "index.html")):
                title, stops = place, None
                try:
                    with open(os.path.join(w, "tour", "manifest.json"), encoding="utf-8") as fh:
                        m = json.load(fh)
                    title, stops = m.get("title") or place, (m.get("stops") or {}).get("count")
                except (OSError, ValueError):
                    pass
                out.append({"id": tour_id(w), "dir": w, "title": title, "day": day, "stops": stops,
                            "at": os.path.getmtime(os.path.join(w, "tour", "index.html"))})
    return sorted(out, key=lambda t: -t["at"])


def has_gpu():
    return shutil.which("nvidia-smi") is not None


class Build:
    """The one build. state: idle | uploading | running | done | held | failed | stopped."""

    def __init__(self):
        self.lock = threading.Lock()
        self.reset()

    def reset(self):
        self.state, self.title, self.walk, self.video, self.proc = "idle", "", None, None, None
        self.done, self.running, self.tail, self.started, self.ended, self.stops = set(), [], [], None, None, None
        self.held, self.error, self.received, self.size, self.code = [], "", 0, 0, None

    def busy(self):
        return self.state in ("uploading", "running")

    def view(self):
        el = (self.ended or time.time()) - self.started if self.started else 0
        gpu = has_gpu()
        est = 9 + 0.18 * max(0, (self.stops or 25) - 25)       # minutes: 25 stops about 9 with an NVIDIA GPU, 42 about 12 (docs/GUIDE.md)
        est = est if gpu else est * 93 / 9                     # about 93 minutes without one
        return {"state": self.state, "title": self.title, "walk": self.walk, "received": self.received, "size": self.size,
                "steps_done": len(self.done), "steps_total": len(STEPS), "running": [WORDS.get(s, s) for s in self.running],
                "elapsed_s": int(el), "left_min": max(1, round(est - el / 60)) if self.state == "running" else None, "gpu": gpu,
                "stops": self.stops, "held": self.held, "tour_dir": os.path.join(self.walk, "tour") if self.walk else None, "error": self.error, "tail": self.tail[-15:],
                "tour": tour_id(self.walk) if self.walk and os.path.isfile(os.path.join(self.walk, "tour", "index.html")) else None}

    def line(self, text):
        self.tail.append(text)
        del self.tail[:-200]
        m = LINE.match(text)
        if not m or m.group(1) not in STEPS:
            return
        step, rest = m.group(1), m.group(2)
        if rest.startswith("running"):
            if step not in self.running:
                self.running.append(step)
        elif rest.startswith("done") or "skipping" in rest or "up to date" in rest or "not needed" in rest:
            self.done.add(step)
            if step in self.running:
                self.running.remove(step)
            if step == "stops":
                self.count_stops()

    def count_stops(self):
        try:
            with open(os.path.join(self.walk, "frames", "points.json"), encoding="utf-8") as fh:
                self.stops = len(json.load(fh))
        except (OSError, ValueError):
            pass


B = Build()


def start_engine():
    """Run tour.py on the video in the background (B.lock held by the caller)."""
    B.state, B.started, B.ended, B.done, B.running, B.tail, B.error, B.held = "running", time.time(), None, set(), [], [], "", []
    env = dict(os.environ, PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8")
    cmd = [sys.executable, "-u", TOUR_PY, B.video]
    if B.title:
        cmd += ["--title", B.title]
    kw = {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP} if os.name == "nt" else {"start_new_session": True}
    B.proc = subprocess.Popen(cmd, cwd=B.walk, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                              text=True, encoding="utf-8", errors="replace", bufsize=1, **kw)
    threading.Thread(target=watch, args=(B.proc,), daemon=True).start()


def watch(proc):
    with open(os.path.join(B.walk, "build.log"), "a", encoding="utf-8") as log:
        for text in proc.stdout:
            text = text.rstrip("\r\n")
            log.write(text + "\n"); log.flush()
            with B.lock:
                B.line(text)
    proc.stdout.close()
    code = proc.wait()
    with B.lock:
        if B.proc is not proc:
            return
        B.ended, B.running, B.code = time.time(), [], code
        if B.state == "stopped":
            return
        if code != 0 or not os.path.isfile(os.path.join(B.walk, "tour", "index.html")):
            B.state = "failed"
            B.error = "The build stopped before the tour was finished. The last lines it wrote are below (all of it is in build.log next to your video)."
            return
        B.held = held_stops(B.walk)
        B.state = "held" if B.held else "done"


def held_stops(walk):
    """The stops the privacy check held back and nobody has answered yet, from tour/manifest.json."""
    try:
        with open(os.path.join(walk, "tour", "manifest.json"), encoding="utf-8") as fh:
            p = json.load(fh).get("privacy") or {}
    except (OSError, ValueError):
        return []
    return [{"id": str(h.get("id")), "reasons": [str(r) for r in h.get("reasons", [])]} for h in p.get("held", []) if h.get("id")]


def stop_engine():
    p = B.proc
    if not p or p.poll() is not None:
        return
    if os.name == "nt":
        subprocess.run(["taskkill", "/T", "/F", "/PID", str(p.pid)], capture_output=True)
    else:
        try:
            os.killpg(p.pid, signal.SIGTERM)
        except OSError:
            pass


def answer_review(decisions):
    """Record each answer with the engine's own privacy_gate.py, then build again (only what changed is redone)."""
    keep = [s for s, v in decisions.items() if v == "keep"]
    out = [s for s, v in decisions.items() if v == "leave_out"]
    for flag, ids in (("--leave-out", out), ("--keep", keep)):         # privacy_gate.py takes one kind of answer a call
        if ids:
            r = subprocess.run([sys.executable, GATE_PY, "--frames", "frames", flag, *ids], cwd=B.walk, capture_output=True, text=True)
            with open(os.path.join(B.walk, "build.log"), "a", encoding="utf-8") as log:
                log.write(r.stdout + r.stderr)
            if r.returncode:
                return (r.stdout + r.stderr).strip()[-600:] or "privacy_gate.py failed"
    return None


def looks_like_video(head):
    return len(head) >= 8 and head[4:8] in BOX_TYPES


class H(BaseHTTPRequestHandler):
    server_version = "shop-tour"

    def log_message(self, *a):
        pass

    def send(self, code, body=b"", ctype="application/json", extra=None):
        if isinstance(body, (dict, list)):
            body = json.dumps(body).encode("utf-8")
        elif isinstance(body, str):
            body = body.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-cache")
        self.send_header("X-Content-Type-Options", "nosniff")
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def ours(self, write):
        """Only this computer's own page: the Host must be ours (no DNS rebinding) and a write must come from our own origin."""
        port = self.server.server_address[1]
        hosts = {f"127.0.0.1:{port}", f"localhost:{port}"}
        if self.headers.get("Host") not in hosts:
            return False
        if write:
            o = self.headers.get("Origin")
            if o is not None and o not in {f"http://{h}" for h in hosts}:
                return False
        return True

    def do_HEAD(self):
        self.do_GET()

    def do_GET(self):
        if not self.ours(False):
            return self.send(403, {"error": "this page answers this computer only"})
        path = urllib.parse.urlparse(self.path).path
        if path == "/":
            return self.send(200, PAGE, "text/html; charset=utf-8")
        if path == "/api/state":
            with B.lock:
                v = B.view()
            v["tours"] = [{k: t[k] for k in ("id", "title", "day", "stops")} for t in find_tours()[:20]]
            return self.send(200, v)
        if path == "/how-to-record":
            try:
                with open(os.path.join(HERE, "CAPTURE.md"), encoding="utf-8") as fh:
                    text = fh.read()
            except OSError:
                text = "CAPTURE.md is missing."
            return self.send(200, f"<!doctype html><meta charset=utf-8><meta name=viewport content='width=device-width'><title>How to record</title>"
                                  f"<pre style='white-space:pre-wrap;font:15px/1.5 system-ui;max-width:720px;margin:16px auto;padding:0 16px'>{html.escape(text)}</pre>",
                             "text/html; charset=utf-8")
        m = re.match(r"^/(t|review)/([0-9a-f]{12})(/.*)?$", path)
        if m:
            return self.static(*m.groups())
        self.send(404, {"error": "not found"})

    def folder(self, kind, tid):
        for t in ([{"id": tour_id(B.walk), "dir": B.walk}] if B.walk else []) + find_tours():
            if t["id"] == tid:
                return os.path.join(t["dir"], "tour") if kind == "t" else os.path.join(t["dir"], "frames", "review")
        return None

    def static(self, kind, tid, rest):
        base = self.folder(kind, tid)
        if not base or not os.path.isdir(base):
            return self.send(404, {"error": "no such tour"})
        if not rest:
            return self.send(301, b"", "text/plain", {"Location": f"/{kind}/{tid}/"})
        rel = urllib.parse.unquote(rest.lstrip("/")) or ("index.html" if kind == "t" else "review.html")
        full = os.path.realpath(os.path.join(base, rel))
        if not (full == os.path.realpath(base) or full.startswith(os.path.realpath(base) + os.sep)) or not os.path.isfile(full):
            return self.send(404, {"error": "not found"})
        ctype = TYPES.get(os.path.splitext(full)[1].lower()) or mimetypes.guess_type(full)[0] or "application/octet-stream"
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(os.path.getsize(full)))
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        if self.command != "HEAD":
            with open(full, "rb") as fh:
                try:
                    shutil.copyfileobj(fh, self.wfile, 1 << 20)
                except ConnectionError:
                    pass

    def body_json(self):
        n = int(self.headers.get("Content-Length") or 0)
        if n <= 0 or n > 1 << 20:
            return None
        try:
            return json.loads(self.rfile.read(n))
        except ValueError:
            return None

    def do_PUT(self):
        if not self.ours(True):
            return self.send(403, {"error": "not from this page"})
        m = re.match(r"^/t/([0-9a-f]{12})/notes\.json$", urllib.parse.urlparse(self.path).path)
        base = self.folder("t", m.group(1)) if m else None
        if not base:
            return self.send(405, {"error": "only a tour's notes.json can be written"})
        n = int(self.headers.get("Content-Length") or 0)
        if n <= 0 or n > serve_tour.MAX:
            return self.send(413, {"error": "too big"})
        try:
            notes = json.loads(self.rfile.read(n))
        except ValueError:
            return self.send(400, {"error": "not JSON"})
        if not serve_tour.valid(notes):
            return self.send(400, {"error": "not a list of notes"})
        data = json.dumps(notes, indent=1).encode("utf-8")
        serve_tour.write_atomic(os.path.join(base, "notes.json"), data)
        serve_tour.write_atomic(os.path.join(os.path.dirname(base), "notes.json"), data)
        self.send(204)

    def do_POST(self):
        if not self.ours(True):
            return self.send(403, {"error": "not from this page"})
        u = urllib.parse.urlparse(self.path)
        if u.path == "/api/upload":
            return self.upload(urllib.parse.parse_qs(u.query))
        if u.path == "/api/stop":
            with B.lock:
                if B.state == "running":
                    B.state = "stopped"
                    stop_engine()
            return self.send(200, {"ok": True})
        if u.path == "/api/again":                       # run the same video again: tour.py skips every step already done
            with B.lock:
                if B.busy() or not B.video or not os.path.isfile(B.video):
                    return self.send(409, {"error": "nothing to run again"})
                start_engine()
            return self.send(200, {"ok": True})
        if u.path == "/api/new":
            with B.lock:
                if B.busy():
                    return self.send(409, {"error": "a build is running"})
                B.reset()
            return self.send(200, {"ok": True})
        if u.path == "/api/review":
            j = self.body_json() or {}
            dec = j.get("decisions") if isinstance(j, dict) else None
            with B.lock:
                if B.state != "held":
                    return self.send(409, {"error": "nothing is waiting for your review"})
                want = {h["id"] for h in B.held}
                if not isinstance(dec, dict) or set(dec) != want or any(v not in ("keep", "leave_out") for v in dec.values()):
                    return self.send(400, {"error": "answer every held stop: keep it in, or leave it out"})
                why = answer_review(dec)
                if why:
                    return self.send(500, {"error": why})
                start_engine()
            return self.send(200, {"ok": True})
        self.send(404, {"error": "not found"})

    def upload(self, q):
        name = safe_name((q.get("name") or [""])[0])
        title = ((q.get("title") or [""])[0]).strip()[:80]
        size = int(self.headers.get("Content-Length") or 0)
        if not name.endswith(VIDEO_EXT):
            return self.send(400, {"error": "That is not a 360 video. Choose an .insv file from the camera, or a stitched 360 .mp4."})
        if size < 64:
            return self.send(400, {"error": "That file is empty."})
        with B.lock:
            if B.busy():
                return self.send(409, {"error": "A tour is already being made. One at a time: wait for it, or stop it."})
            B.reset()
            B.state, B.title, B.size = "uploading", title, size
        try:
            walk = walk_folder(title)
            where = walk
            while not os.path.isdir(where) and os.path.dirname(where) != where:
                where = os.path.dirname(where)
            if shutil.disk_usage(where).free < size + RESERVE:
                raise ValueError(f"Not enough free space on the drive of {walks_root()}: the video needs {size / 2**30:.1f} GB and the build about 2 GB more.")
            os.makedirs(walk, exist_ok=True)
            dst = os.path.join(walk, name)
            part = dst + ".part"
            left = size
            with open(part, "wb") as fh:
                head = self.rfile.read(min(left, 16))
                if not looks_like_video(head):
                    raise ValueError("That file is not a video. Choose an .insv file from the camera, or a stitched 360 .mp4.")
                fh.write(head); left -= len(head)
                while left > 0:
                    chunk = self.rfile.read(min(left, 1 << 20))
                    if not chunk:
                        raise ValueError("The copy was cut off. Choose the video again.")
                    fh.write(chunk); left -= len(chunk)
                    B.received = size - left
            os.replace(part, dst)
        except (ValueError, OSError) as e:
            with B.lock:
                B.reset()
            try:
                os.remove(part)
                os.rmdir(walk)
            except (OSError, NameError, UnboundLocalError):
                pass
            self.close_connection = True
            return self.send(400, {"error": str(e)})
        with B.lock:
            B.walk, B.video, B.received = walk, dst, size
            start_engine()
        self.send(200, {"ok": True, "walk": walk})


TYPES = {".html": "text/html; charset=utf-8", ".js": "text/javascript", ".mjs": "text/javascript", ".json": "application/json",
         ".css": "text/css", ".svg": "image/svg+xml", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png", ".webp": "image/webp",
         ".bin": "application/octet-stream", ".npy": "application/octet-stream", ".wasm": "application/wasm"}

PAGE = r"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>shop-tour</title><style>
:root{--ink:#111;--soft:#666;--line:#ddd;--bg:#fff;--key:#111;--on:#fff;--ok:#1a7f37;--bad:#c62828}
@media (prefers-color-scheme:dark){:root{--ink:#eee;--soft:#aaa;--line:#333;--bg:#121212;--key:#eee;--on:#111}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:17px/1.45 system-ui,sans-serif}
main{max-width:560px;margin:0 auto;padding:24px 16px 48px}h1{font-size:26px;margin:0 0 4px}p{margin:8px 0}.soft{color:var(--soft);font-size:15px}
label{display:block;font-weight:600;margin:20px 0 6px}input[type=text]{width:100%;font:inherit;padding:12px;border:1px solid var(--line);border-radius:10px;background:var(--bg);color:var(--ink)}
.big{display:block;width:100%;min-height:64px;margin:16px 0;font:600 19px system-ui;border:0;border-radius:14px;background:var(--key);color:var(--on);cursor:pointer;text-align:center;line-height:64px;text-decoration:none}
.btn{font:600 15px system-ui;min-height:44px;padding:0 16px;border:1px solid var(--line);border-radius:10px;background:var(--bg);color:var(--ink);cursor:pointer}
.btn[aria-pressed=true]{background:var(--key);color:var(--on);border-color:var(--key)}.btn:disabled,.big:disabled{opacity:.4;cursor:default}
.bar{height:10px;background:var(--line);border-radius:5px;overflow:hidden;margin:12px 0}.bar i{display:block;height:100%;background:var(--ok);width:0}
.card{border:1px solid var(--line);border-radius:12px;padding:12px;margin:10px 0}.err{color:var(--bad)}pre{white-space:pre-wrap;font-size:12px;color:var(--soft);max-height:240px;overflow:auto}
a{color:inherit}.row{display:flex;gap:8px;flex-wrap:wrap;margin-top:8px}ul{padding-left:20px}[hidden]{display:none!important}
</style></head><body><main>
<h1>Make a 3D tour</h1><p class="soft">From one walk with your 360 camera. Everything stays on this computer.</p>
<section id="pick">
 <label for="title">What is this place?</label><input type="text" id="title" maxlength="80" placeholder="My house" autocomplete="off">
 <input type="file" id="file" accept=".insv,.mp4" hidden>
 <button class="big" id="choose">Choose your 360 video</button>
 <p class="soft">An <b>.insv</b> file straight off the Insta360 camera, or a stitched 360 <b>.mp4</b>. Haven't recorded yet? <a href="/how-to-record" target="_blank">How to record a walk</a>.</p>
 <p class="err" id="pickErr" role="alert"></p>
</section>
<section id="work" hidden aria-live="polite">
 <h2 id="wTitle"></h2><p id="wWords"></p><div class="bar"><i id="wBar"></i></div><p class="soft" id="wTime"></p>
 <div class="row"><button class="btn" id="stop">Stop</button><button class="btn" id="again" hidden>Try again</button><button class="btn" id="new" hidden>Make another tour</button></div>
 <div id="held" hidden><h3>Some stops may show a person</h3>
  <p>They are not in your tour yet. Look at each one, then decide. Nothing is decided for you.</p>
  <a class="btn" id="reviewLink" target="_blank" style="display:inline-flex;align-items:center;text-decoration:none">Look at the held stops</a>
  <div id="heldList"></div><button class="big" id="answer" disabled>Save my answers and finish the tour</button></div>
 <a class="big" id="open" target="_blank" hidden>Open your tour</a>
 <details><summary class="soft">Show the details</summary><pre id="tail"></pre></details>
</section>
<section id="toursBox" hidden><h3>Your tours</h3><ul id="tours"></ul></section>
</main><script>
const $=id=>document.getElementById(id);let S={},answers={};
const post=(u,b)=>fetch(u,{method:'POST',headers:{'Content-Type':'application/json'},body:b?JSON.stringify(b):'{}'}).then(async r=>({ok:r.ok,j:await r.json().catch(()=>({}))}));
$('choose').onclick=()=>$('file').click();
$('file').onchange=()=>{const f=$('file').files[0];$('file').value='';if(!f)return;$('pickErr').textContent='';
 if(!/\.(insv|mp4)$/i.test(f.name)){$('pickErr').textContent='That is not a 360 video. Choose an .insv file from the camera, or a stitched 360 .mp4.';return}
 const x=new XMLHttpRequest();x.open('POST','/api/upload?name='+encodeURIComponent(f.name)+'&title='+encodeURIComponent($('title').value));
 x.upload.onprogress=e=>{if(e.lengthComputable)show({state:'uploading',received:e.loaded,size:e.total,title:$('title').value})};
 x.onload=()=>{if(x.status!==200){let m='The copy did not work.';try{m=JSON.parse(x.responseText).error||m}catch(e){}$('pickErr').textContent=m}poll()};
 x.onerror=()=>{$('pickErr').textContent='The copy was cut off. Choose the video again.';poll()};x.send(f);show({state:'uploading',received:0,size:f.size,title:$('title').value})};
$('stop').onclick=()=>post('/api/stop').then(poll);$('again').onclick=()=>post('/api/again').then(poll);$('new').onclick=()=>post('/api/new').then(poll);
$('answer').onclick=()=>{$('answer').disabled=true;post('/api/review',{decisions:answers}).then(r=>{if(!r.ok)alert(r.j.error||'That did not work');answers={};poll()})};
function mins(s){const m=Math.round(s/60);return m<1?'under a minute':m===1?'1 minute':m+' minutes'}
function show(s){S=s;const busy=s.state!=='idle';$('pick').hidden=busy;$('work').hidden=!busy;if(!busy)return;
 $('wTitle').textContent=s.title||'Your tour';let w='',pct=0,t='';
 if(s.state==='uploading'){pct=s.size?100*s.received/s.size:0;w='Copying your video… '+Math.floor(pct)+'%';t='Keep this page open until the copy is done.'}
 else if(s.state==='running'){pct=100*s.steps_done/s.steps_total;w=(s.running.length?s.running.join(' · '):'Starting')+' — step '+Math.min(s.steps_done+1,s.steps_total)+' of '+s.steps_total;
  t=mins(s.elapsed_s)+' so far · about '+s.left_min+' more '+(s.left_min===1?'minute':'minutes')+(s.gpu?'':' (no NVIDIA graphics card found, so this is slow)')+'. You can close this page; it keeps going.'}
 else if(s.state==='done'){pct=100;w='Your tour is ready.';t='It is in '+s.tour_dir+' — plain files you can put on any website. Look at the pictures before you share it.'}
 else if(s.state==='held'){pct=100;w='Almost done.';t=''}
 else if(s.state==='stopped'){w='Stopped.';t='Try again picks up where it stopped.'}
 else if(s.state==='failed'){w=s.error;t=''}
 $('wWords').textContent=w;$('wWords').className=s.state==='failed'?'err':'';$('wBar').style.width=pct+'%';$('wTime').textContent=t;
 $('stop').hidden=s.state!=='running';$('again').hidden=!['failed','stopped'].includes(s.state);$('new').hidden=['uploading','running'].includes(s.state);
 $('open').hidden=!s.tour||!['done','held'].includes(s.state);if(s.tour)$('open').href='/t/'+s.tour+'/';$('open').textContent=s.state==='held'?'Open your tour (without the held stops)':'Open your tour';
 $('tail').textContent=(s.tail||[]).join('\n');$('held').hidden=s.state!=='held';
 if(s.state==='held'){$('reviewLink').href='/review/'+s.tour+'/';const L=$('heldList');if(L.dataset.ids!==s.held.map(h=>h.id).join()){L.dataset.ids=s.held.map(h=>h.id).join();L.innerHTML='';answers={};
  for(const h of s.held){const c=document.createElement('div');c.className='card';const b=document.createElement('b');b.textContent='Stop '+h.id;const r=document.createElement('p');r.className='soft';r.textContent=h.reasons.join('; ');
   const row=document.createElement('div');row.className='row';for(const [v,lab] of [['keep','Keep it in (faces blurred)'],['leave_out','Leave it out']]){const k=document.createElement('button');k.className='btn';k.textContent=lab;k.setAttribute('aria-pressed','false');
    k.onclick=()=>{answers[h.id]=v;row.querySelectorAll('button').forEach(x=>x.setAttribute('aria-pressed',String(x===k)));$('answer').disabled=Object.keys(answers).length!==s.held.length};row.appendChild(k)}
   c.append(b,r,row);L.appendChild(c)}$('answer').disabled=true}}
}
function tours(list){$('toursBox').hidden=!list.length;$('tours').innerHTML='';for(const t of list){const li=document.createElement('li');const a=document.createElement('a');a.href='/t/'+t.id+'/';a.target='_blank';
 a.textContent=t.title+' · '+t.day+(t.stops?' · '+t.stops+' stops':'');li.appendChild(a);$('tours').appendChild(li)}}
function poll(){fetch('/api/state').then(r=>r.json()).then(s=>{if(!(S.state==='uploading'&&s.state==='idle'))show(s);tours(s.tours||[])}).catch(()=>{})}
poll();setInterval(poll,2000);
</script></body></html>"""


def already_open(port):
    import http.client
    try:
        c = http.client.HTTPConnection("127.0.0.1", port, timeout=1)
        c.request("GET", "/api/state", headers={"Host": f"127.0.0.1:{port}"})
        r = c.getresponse()
        ok = r.status == 200 and b"steps_total" in r.read()
        c.close()
        return ok
    except (OSError, http.client.HTTPException):
        return False


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--no-browser", action="store_true")
    a = ap.parse_args()
    srv = None
    for port in range(a.port, a.port + 10):
        if already_open(port):                                          # start was double-clicked twice: show the page that is open
            print(f"shop-tour is already open at http://127.0.0.1:{port}/")
            if not a.no_browser:
                webbrowser.open(f"http://127.0.0.1:{port}/")
            return
        try:
            srv = ThreadingHTTPServer(("127.0.0.1", port), H)          # this computer only
            break
        except OSError:
            continue
    if srv is None:
        sys.exit(f"ports {a.port} to {a.port + 9} are all in use; close something or run:  python app.py --port 9000")
    url = f"http://127.0.0.1:{srv.server_address[1]}/"
    print(f"shop-tour is open at {url}  (this computer only). Videos and tours go in {walks_root()}. Close this window to stop.")
    if not a.no_browser:
        threading.Timer(0.5, webbrowser.open, (url,)).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        stop_engine()


if __name__ == "__main__":
    main()
