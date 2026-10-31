"""Tests for app.py, the upload button, with a stand-in engine (the real tour.py is never run: no footage, GPU or packages needed).
    python qa/test_app.py
The stand-in obeys tour.py's command line and prints its "[step] running..." / "[step] done in Ns" lines; a stand-in privacy_gate.py records the answers.
"""
import http.client, json, os, sys, tempfile, textwrap, threading, time, unittest

TMP = tempfile.mkdtemp(prefix="shoptour-app-")
ENGINE = os.path.join(TMP, "engine")
os.makedirs(ENGINE)
with open(os.path.join(ENGINE, "tour.py"), "w") as fh:
    fh.write(textwrap.dedent('''
        import json, os, sys, time
        mode = os.environ.get("STANDIN_MODE", "ok")
        video = sys.argv[1]; title = sys.argv[sys.argv.index("--title") + 1] if "--title" in sys.argv else ""
        os.makedirs("frames/review", exist_ok=True)
        json.dump({"argv": sys.argv[1:], "cwd": os.getcwd()}, open("frames/argv.json", "w"))
        for step in ["stops", "calib", "stitch", "privacy", "build"]:
            print(f"[{step}] running...", flush=True)
            if step == "stops":
                json.dump([{"id": "001"}, {"id": "002"}, {"id": "003"}], open("frames/points.json", "w"))
            if mode == "slow":
                time.sleep(30)
            if mode == "fail" and step == "stitch":
                print("Traceback: something broke", flush=True); sys.exit(1)
            print(f"[{step}] done in 0s", flush=True)
        os.makedirs("tour", exist_ok=True)
        open("tour/index.html", "w").write("<!doctype html><title>t</title>tour of " + title)
        open("tour/tour.json", "w").write("{}")
        held = [] if (mode != "held" or os.path.exists("frames/answered.json")) else [{"id": "002", "reasons": ["a person may be in it"]}]
        open("frames/review/review.html", "w").write("review")
        json.dump({"title": title, "stops": {"count": 2}, "privacy": {"status": "held" if held else "all_clear", "held": held}}, open("tour/manifest.json", "w"))
        print("all done", flush=True)
    '''))
with open(os.path.join(ENGINE, "privacy_gate.py"), "w") as fh:
    fh.write(textwrap.dedent('''
        import json, os, sys
        calls = json.load(open("frames/gate_calls.json")) if os.path.exists("frames/gate_calls.json") else []
        calls.append(sys.argv[1:]); json.dump(calls, open("frames/gate_calls.json", "w")); open("frames/answered.json", "w").write("1")
    '''))
os.environ["SHOP_TOUR_APP_TOUR"] = os.path.join(ENGINE, "tour.py")
os.environ["SHOP_TOUR_APP_GATE"] = os.path.join(ENGINE, "privacy_gate.py")
os.environ["SHOP_TOUR_WALKS"] = os.path.join(TMP, "walks")
os.environ["SHOP_TOUR_APP_RESERVE_GB"] = "0"                # a test machine's drive may be nearly full; the check itself is the same
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
import app                                                  # noqa: E402
from http.server import ThreadingHTTPServer                 # noqa: E402

SRV = ThreadingHTTPServer(("127.0.0.1", 0), app.H)
PORT = SRV.server_address[1]
threading.Thread(target=SRV.serve_forever, daemon=True).start()
ORIGIN = f"http://127.0.0.1:{PORT}"
VIDEO = b"\0\0\0\x1cftypavc1" + b"\0" * 4000


def req(method, path, body=b"", headers=None, host=None):
    c = http.client.HTTPConnection("127.0.0.1", PORT, timeout=20)
    h = {"Host": host or f"127.0.0.1:{PORT}", "Origin": ORIGIN}
    h.update(headers or {})
    if body is not None and method in ("POST", "PUT"):
        h["Content-Length"] = str(len(body))
    c.request(method, path, body=body, headers=h)
    r = c.getresponse()
    data = r.read()
    c.close()
    return r.status, data


def load(path):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def state():
    return json.loads(req("GET", "/api/state")[1])


def wait(want, t=20):
    end = time.time() + t
    while time.time() < end:
        s = state()
        if s["state"] in want:
            return s
        time.sleep(0.1)
    raise AssertionError(f"state stayed {state()['state']}, wanted {want}")


def upload(name="VID_test_00_001.insv", title="My House", data=VIDEO):
    return req("POST", f"/api/upload?name={name}&title={title.replace(' ', '%20')}", data)


class App(unittest.TestCase):
    def setUp(self):
        s = state()
        if s["state"] == "running":
            req("POST", "/api/stop"); wait({"stopped", "failed", "done", "held"})
        req("POST", "/api/new")
        os.environ["STANDIN_MODE"] = "ok"

    def test_01_page_and_this_computer_only(self):
        code, page = req("GET", "/")
        self.assertEqual(code, 200)
        self.assertIn(b"Choose your 360 video", page)
        self.assertEqual(req("GET", "/", host="evil.example:80")[0], 403)                       # a page reached by another name (DNS rebinding)
        self.assertEqual(req("POST", "/api/new", b"{}", {"Origin": "http://evil.example"})[0], 403)  # a write from another site
        self.assertEqual(SRV.server_address[0], "127.0.0.1")
        self.assertIn('("127.0.0.1", port)', open(os.path.join(HERE, "app.py"), encoding="utf-8").read())

    def test_02_refuses_what_is_not_a_video(self):
        self.assertEqual(upload(name="notes.txt")[0], 400)
        code, body = upload(name="fake.mp4", data=b"hello this is a text file, not a video at all" * 10)
        self.assertEqual(code, 400)
        self.assertIn("not a video", json.loads(body)["error"])
        self.assertEqual(state()["state"], "idle")
        self.assertFalse(any(f.endswith((".mp4", ".part")) for _, _, fs in os.walk(os.environ["SHOP_TOUR_WALKS"]) for f in fs))

    def test_03_upload_build_and_open(self):
        code, body = upload()
        self.assertEqual(code, 200, body)
        walk = json.loads(body)["walk"]
        self.assertTrue(walk.startswith(os.path.join(os.environ["SHOP_TOUR_WALKS"], "my-house")))
        s = wait({"done"})
        self.assertEqual(s["steps_done"], 5)
        self.assertEqual(s["stops"], 3)
        self.assertTrue(s["tour"])
        argv = load(os.path.join(walk, "frames", "argv.json"))
        self.assertEqual(argv["argv"], [os.path.join(walk, "VID_test_00_001.insv"), "--title", "My House"])
        with open(os.path.join(walk, "VID_test_00_001.insv"), "rb") as fh:
            self.assertEqual(fh.read(), VIDEO)                                                   # the copy is the file, byte for byte
        code, page = req("GET", f"/t/{s['tour']}/")
        self.assertEqual(code, 200)
        self.assertIn(b"tour of My House", page)
        self.assertEqual(req("GET", f"/t/{s['tour']}/..%2F..%2Fframes%2Fargv.json")[0], 404)   # nothing outside tour/
        self.assertEqual(req("GET", f"/t/{s['tour']}/../frames/argv.json")[0], 404)
        note = json.dumps([{"id": "n1", "x": 1, "y": 2, "z": 3, "text": "panel"}]).encode()
        self.assertEqual(req("PUT", f"/t/{s['tour']}/notes.json", note, {"Content-Type": "application/json"})[0], 204)
        self.assertTrue(os.path.isfile(os.path.join(walk, "notes.json")))
        self.assertIn(s["tour"], [t["id"] for t in state()["tours"]])

    def test_04_one_build_at_a_time(self):
        os.environ["STANDIN_MODE"] = "slow"
        self.assertEqual(upload(title="first")[0], 200)
        wait({"running"})
        code, body = upload(title="second")
        self.assertEqual(code, 409)
        self.assertIn("One at a time", json.loads(body)["error"])
        self.assertEqual(req("POST", "/api/stop", b"{}")[0], 200)
        self.assertEqual(wait({"stopped"})["state"], "stopped")

    def test_05_held_stops_are_asked_never_decided(self):
        os.environ["STANDIN_MODE"] = "held"
        code, body = upload(title="held place")
        walk = json.loads(body)["walk"]
        s = wait({"held"})
        self.assertEqual([h["id"] for h in s["held"]], ["002"])
        self.assertEqual(req("GET", f"/review/{s['tour']}/")[0], 200)                           # the review page, this computer only
        self.assertFalse(os.path.exists(os.path.join(walk, "frames", "gate_calls.json")))      # nothing answered by itself
        self.assertEqual(req("POST", "/api/review", json.dumps({"decisions": {}}).encode())[0], 400)
        self.assertEqual(req("POST", "/api/review", json.dumps({"decisions": {"002": "maybe"}}).encode())[0], 400)
        self.assertEqual(req("POST", "/api/review", json.dumps({"decisions": {"002": "leave_out"}}).encode())[0], 200)
        s = wait({"done"})
        calls = load(os.path.join(walk, "frames", "gate_calls.json"))
        self.assertEqual(calls, [["--frames", "frames", "--leave-out", "002"]])

    def test_06_a_failed_build_says_so(self):
        os.environ["STANDIN_MODE"] = "fail"
        upload(title="broken")
        s = wait({"failed"})
        self.assertIn("stopped before the tour was finished", s["error"])
        self.assertTrue(any("something broke" in t for t in s["tail"]))
        os.environ["STANDIN_MODE"] = "ok"
        self.assertEqual(req("POST", "/api/again", b"{}")[0], 200)                              # tour.py picks up where it stopped
        wait({"done"})

    def test_07_one_video_per_folder(self):
        upload(title="twice"); wait({"done"}); req("POST", "/api/new")
        code, body = upload(title="twice")
        self.assertTrue(json.loads(body)["walk"].endswith("-2"))
        wait({"done"})

    def test_08_step_lines(self):
        b = app.Build(); b.walk = TMP
        for t in ["[stops] running...", "    vio| [vio] running...", "[calib] not needed for a stitched video, skipping", "[stops] done in 3s",
                  "[depth] up to date", "[nonsense] running...", "random line"]:
            b.line(t)
        self.assertEqual(b.done, {"calib", "stops", "depth"})
        self.assertEqual(b.running, ["vio"])

    def test_09_a_second_start_finds_the_open_page(self):
        self.assertTrue(app.already_open(PORT))
        self.assertFalse(app.already_open(1))


sys.path.insert(0, os.path.join(HERE, "tools"))
import setup_help                                           # noqa: E402


class Setup(unittest.TestCase):
    """The choices setup.bat and setup.sh make (tools/setup_help.py)."""

    def test_torch_build_by_driver(self):
        self.assertEqual(setup_help.torch_tag("win32", "610.12"), "cu130")
        self.assertEqual(setup_help.torch_tag("linux", "580.65.06"), "cu130")
        self.assertEqual(setup_help.torch_tag("win32", "552.44"), "cpu")         # too old for the CUDA 13.0 build
        self.assertEqual(setup_help.torch_tag("linux", ""), "cpu")                 # no NVIDIA card
        self.assertEqual(setup_help.torch_tag("darwin"), "pypi")

    def test_pins_come_from_requirements(self):
        p = setup_help.pins()
        req = open(os.path.join(HERE, "requirements.txt"), encoding="utf-8").read()
        self.assertIn(f"torch=={p['torch']}", req)
        self.assertIn(f"torchvision=={p['torchvision']}", req)

    def test_requirements_stamp(self):
        old = setup_help.STAMP
        setup_help.STAMP = os.path.join(TMP, "stamp", "x.sha256")
        try:
            self.assertEqual(setup_help.main(["reqs-current"]), 1)
            self.assertEqual(setup_help.main(["reqs-stamp"]), 0)
            self.assertEqual(setup_help.main(["reqs-current"]), 0)
        finally:
            setup_help.STAMP = old

    def test_scripts_are_idempotent_and_never_skip_the_gate(self):
        for f in ("setup.bat", "setup.sh"):
            t = open(os.path.join(HERE, f), encoding="utf-8").read()
            self.assertIn("--dry-run", t)
            self.assertIn("reqs-current", t)
            self.assertIn("torch-ok", t)
            self.assertIn("fetch_pii_models.py", t)
        a = open(os.path.join(HERE, "app.py"), encoding="utf-8").read()
        for word in ("SHOP_TOUR_PRIVACY", "SHOP_TOUR_PEOPLE", "SHOP_TOUR_PII_BLUR"):
            self.assertNotIn(word, a)                                         # the page never changes what the privacy check does


if __name__ == "__main__":
    unittest.main(verbosity=2)
