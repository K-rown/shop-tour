"""Automated QA for the built tour: opens it in a headless browser (desktop and phone sizes), walks it, and checks
the things a person would notice. Writes qa/report.json and qa/shots/*.png, and exits 1 if anything fails.

Setup (once):  pip install playwright  &&  python -m playwright install chromium
Run:           python qa/qa_tour.py            (serves tour/ itself on a free port)
               python qa/qa_tour.py --stops 1,5,10,15,20 --headed

Checks:
  load        page loads, no JS errors, badge shows the build, every stop count matches tour.json
  depth       every visited stop has its depth surface (needed for occlusion, cursor and glide)
  cursor      the cursor appears when hovering visible floor (desktop)
  tap-to-go   clicking/tapping floor moves to another stop and keeps facing (within 25 deg of the real-room heading)
  glide       glide takes 0.25-1.1 s (headless browsers render in software, so allow some slack)
  phone       the same tap-to-go works at 390x844 with touch
Also records pipeline timings (frames/timings.json) and stop sharpness if present, so speed and quality can be tracked.
"""
import math, argparse, functools, http.server, json, os, socket, sys, threading, time
import cv2, numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def serve(folder):
    s = socket.socket(); s.bind(("127.0.0.1", 0)); port = s.getsockname()[1]; s.close()
    class Silent(http.server.SimpleHTTPRequestHandler):
        def log_message(self, *a, **k):
            pass
    h = functools.partial(Silent, directory=folder)
    class Quiet(http.server.ThreadingHTTPServer):
        def handle_error(self, request, client_address):          # a browser closing a connection mid-download is normal, not an error to print
            if not isinstance(sys.exc_info()[1], ConnectionError):
                super().handle_error(request, client_address)
    srv = Quiet(("127.0.0.1", port), h)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return f"http://127.0.0.1:{port}/index.html"


def wrap(a):
    return (a + 540) % 360 - 180


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tour", default=os.path.join(HERE, "tour"))
    ap.add_argument("--stops", default="1,4,8,12,16,20")
    ap.add_argument("--headed", action="store_true")
    ap.add_argument("--browser", default="chromium", choices=["chromium", "webkit", "firefox"], help="webkit = the engine of Safari on iPhone")
    ap.add_argument("--three", help="serve three.min.js from this local file (offline machines)")
    a = ap.parse_args()
    from playwright.sync_api import sync_playwright
    url = serve(a.tour)
    if not os.path.exists(os.path.join(a.tour, "tour.json")):
        sys.exit(f"no built tour in {a.tour} (tour.json is missing): build one first with  python tour.py <your video>")
    tj = json.load(open(os.path.join(a.tour, "tour.json")))
    N = tj["nodes"]
    shots = os.path.join(HERE, "qa", "shots"); os.makedirs(shots, exist_ok=True)
    report = {"time": time.strftime("%Y-%m-%d %H:%M:%S"), "checks": [], "glides_ms": []}

    def check(name, ok, detail=""):
        report["checks"].append({"check": name, "ok": bool(ok), "detail": detail})
        print(("PASS " if ok else "FAIL ") + name + (f"  ({detail})" if detail else ""), flush=True)

    hops_m = [float(np.hypot(N[k + 1]["x"] - N[k]["x"], N[k + 1]["y"] - N[k]["y"])) for k in range(len(N) - 1) if N[k]["x"] is not None and N[k + 1]["x"] is not None]
    tap_limit = max(4.0, 2.5 * float(np.median(hops_m)) if hops_m else 4.0)     # a sparse tour (few stops far apart) has no stop near every spot
    extra = []                                              # nothing from an earlier build may stay in the folder (it would be published with the tour)
    for sub, key in (("panos", "fh"), ("panos_lo", "f"), ("panos_xs", "fx"), ("thumbs", "th")):
        have = set(os.listdir(os.path.join(a.tour, sub))) if os.path.isdir(os.path.join(a.tour, sub)) else set()
        extra += [f"{sub}/{f}" for f in have - {n[key].split("?")[0].split("/")[-1] for n in N}]
    extra += [f"depth/{f}" for f in (set(os.listdir(os.path.join(a.tour, "depth"))) if os.path.isdir(os.path.join(a.tour, "depth")) else set())
              - {n["dm"].split("?")[0].split("/")[-1] for n in N if n.get("dm")}]
    gains = [n["gain"] for n in N if n.get("gain")]
    if gains:                                              # M3: per-stop exposure gains must stay in a sane range
        gmin, gmax = min(map(min, gains)), max(map(max, gains))
        check("exposure gains in 0.7-1.4", 0.7 <= gmin and gmax <= 1.4, f"{gmin:.2f}..{gmax:.2f} over {len(gains)} stops")

    try:                                                   # nobody should be recognisable under the camera (needs torch; skipped otherwise)
        import subprocess, importlib.util
        if importlib.util.find_spec("torch") and importlib.util.find_spec("transformers"):
            r = subprocess.run([sys.executable, os.path.join(HERE, "qa", "check_nadir_person.py"), "--folder", os.path.join(a.tour, "panos_lo")],
                               capture_output=True, text=True)
            last = [l for l in r.stdout.splitlines() if l.startswith("stops over")]
            check("nadir: no person under the camera (segmentation)", r.returncode == 0, last[0] if last else r.stderr[-200:])
    except Exception as e:
        print("nadir person check skipped:", e)

    check("tour folder holds only this tour's pictures", not extra, f"{len(extra)} stray: {extra[:3]}" if extra else f"{len(N)} stops")
    if tj.get("tiles"):                                    # phones load the full-detail picture as tiles: every stop needs all of them, and nothing else may stay behind
        T = tj["tiles"]
        want = {n["tl"].split("?")[0].split("/")[-1].replace("{r}{c}", f"{r}{c}") for n in N for r in range(T["rows"]) for c in range(T["cols"])}
        have = set(os.listdir(os.path.join(a.tour, "tiles"))) if os.path.isdir(os.path.join(a.tour, "tiles")) else set()
        check("tiles: every stop has all its tiles and no stray ones", have == want, f"{len(have)} files, expected {len(want)}; missing {sorted(want - have)[:2]}, stray {sorted(have - want)[:2]}")
    with sync_playwright() as p:
        b = getattr(p, a.browser).launch(headless=not a.headed, **({"args": ["--ignore-gpu-blocklist", "--enable-webgl", "--enable-unsafe-swiftshader"]} if a.browser == "chromium" else {}))
        for label, vp, touch in (("desktop", {"width": 1280, "height": 800}, False), ("phone", {"width": 390, "height": 844}, True)):
            ctx = b.new_context(viewport=vp, has_touch=touch, **({"is_mobile": touch} if a.browser != "firefox" else {}))
            ctx.route(lambda url: not url.startswith("http://127.0.0.1"), lambda r: r.abort())     # the tour must work without internet (three.js ships with it)
            if a.three:
                ctx.route("**/three.min.js", lambda r: r.fulfill(path=a.three, content_type="application/javascript"))
            pg = ctx.new_page()
            errs = []
            pg.on("pageerror", lambda e: errs.append(str(e)))
            pg.on("console", lambda m: errs.append(m.text) if m.type == "error" else None)
            t_nav = time.time()
            pg.goto(url)
            try:
                pg.wait_for_function("window.__tour && !window.__tour.state().busy", timeout=60000)
                report.setdefault("first_view_ms", {})[label] = round((time.time() - t_nav) * 1000)       # navigation until the first picture is on screen (local server, no throttling)
            except Exception:
                check(f"{label}: load", False, "tour never became ready: " + "; ".join(errs[-3:])); ctx.close(); continue
            pg.evaluate("document.getElementById('help') && document.getElementById('help').classList.remove('show')")
            if not touch:                                      # an idle viewer must not redraw the scene every frame (phone battery)
                pg.wait_for_timeout(2500)
                dr = pg.evaluate("""() => new Promise(res => { const r0 = window.__renders(); setTimeout(() => res(window.__renders() - r0), 3000); })""")
                check(f"{label}: idle viewer does not redraw continuously", dr <= 10, f"{dr} scene draws in 3 s")
            if tj["nodes"][0].get("fx"):                       # the first picture is a tiny copy; the 4K one must replace it
                try:
                    pg.wait_for_function("window.__tour.state().res !== '1K'", timeout=30000)
                    sharp_ok = True
                except Exception:
                    sharp_ok = False
                check(f"{label}: the quick first picture is replaced by the 4K one", sharp_ok, f"first view {report['first_view_ms'].get(label)} ms, now {pg.evaluate('window.__tour.state().res')}")
            info = pg.evaluate("({b: window.__tour.build, n: window.__tour.n})")
            check(f"{label}: load", not errs and info["n"] == len(N), f"build {info['b']}, {info['n']} stops, errors: {errs[:2]}")
            # M2: no on-screen arrow buttons, about 100 degrees across the screen by default, zoom and the arrow keys still work
            st0 = pg.evaluate("window.__tour.state()")
            # 2026-10-29: the thumb cluster (previous / walk ahead / next) and the floor rings are the phone's way to move; each key is at least 44 px and clear of the bottom 52 px
            cl = pg.evaluate("['bPrev','bGo','bNext'].map(id => { const e = document.getElementById(id); if (!e) return null; const r = e.getBoundingClientRect(); return [r.width, r.height, innerHeight - r.bottom]; })")
            check(f"{label}: the thumb cluster is there, 44 px, clear of the bottom strip", all(c and c[0] >= 44 and c[1] >= 44 and c[2] >= 52 for c in cl), str(cl))
            check(f"{label}: the floor rings point at the stops you can walk to", pg.evaluate("window.__tour.rings().length") >= 1, f"{pg.evaluate('window.__tour.rings().length')} rings at stop 1")
            hf = 2 * math.degrees(math.atan(math.tan(math.radians(st0["vfov"] / 2)) * st0["aspect"]))
            check(f"{label}: default view is about 110 degrees across the screen", abs(st0["hfov"] - 110) < 1 and (abs(hf - 110) < 3 or st0["vfov"] >= 129), f"hfov setting {st0['hfov']}, vertical {st0['vfov']}, across the screen {hf:.0f}")
            lon0 = pg.evaluate("window.__tour.state().lon")
            pg.keyboard.down("ArrowRight"); pg.wait_for_timeout(500); pg.keyboard.up("ArrowRight")
            check(f"{label}: the arrow keys still turn the view", abs(pg.evaluate("window.__tour.state().lon") - lon0) > 3, f"turned {pg.evaluate('window.__tour.state().lon') - lon0:.0f} deg")
            wheel = lambda dy: pg.evaluate("dy => { const e = document.elementFromPoint(innerWidth / 2, innerHeight / 2); e.dispatchEvent(new WheelEvent('wheel', { deltaY: dy, bubbles: true, cancelable: true })); }", dy)   # mouse.wheel does not exist in mobile WebKit
            wheel(-600); pg.wait_for_timeout(200)
            z = pg.evaluate("window.__tour.state().hfov")
            check(f"{label}: scrolling zooms in (and stays within limits)", 44.9 <= z < st0["hfov"], f"{st0['hfov']} -> {z}")
            wheel(3000); pg.wait_for_timeout(200)
            check(f"{label}: zooming out stops at the limit", pg.evaluate("window.__tour.state().hfov") <= 110.01, f"{pg.evaluate('window.__tour.state().hfov')}")
            pg.evaluate("window.__tour.aim(window.__tour.state().lon, -12)")
            # M3: the help panel has the recording guide with its five pictures, readable at this width
            pg.evaluate("document.getElementById('help').classList.add('show'); document.getElementById('guide').open = true")
            pg.wait_for_timeout(800)
            hp = pg.evaluate("""() => { const imgs = [...document.querySelectorAll('#help figure img')]; const box = document.querySelector('#help > div');
                return { n: imgs.length, loaded: imgs.filter(i => i.complete && i.naturalWidth > 0).length, minW: Math.min(...imgs.map(i => i.getBoundingClientRect().width)),
                         overflow: box.scrollWidth - box.clientWidth, titles: !!document.querySelector('#guide summary'), why: document.querySelectorAll('#help figcaption i').length }; }""")
            check(f"{label}: help panel shows the five recording pictures with captions", hp["n"] == 5 and hp["loaded"] == 5 and hp["why"] == 5, str(hp))
            check(f"{label}: recording pictures are readable and nothing overflows sideways", hp["minW"] >= min(330, vp["width"] - 56) and hp["overflow"] <= 1, f"narrowest picture {hp['minW']:.0f}px, overflow {hp['overflow']}")
            pg.click("#hClose"); pg.wait_for_timeout(150)
            check(f"{label}: the help panel closes with its button", not pg.evaluate("document.getElementById('help').classList.contains('show')"), "")
            # dollhouse, measuring, plan: dollhouse, measuring, plan
            pg.evaluate("window.__tour.go(0)"); pg.wait_for_function("!window.__tour.state().busy", timeout=30000)
            mp = tj.get("measure_pct")
            hidden = pg.evaluate("document.getElementById('bMeas').hidden")
            if mp is not None and mp > 10:
                check(f"{label}: Measure is hidden because its measured error is over 10 %", hidden, f"measured error {mp} %")
            else:
                check(f"{label}: the Measure button is there" + (f" and says its error (+/- {mp:.0f} %)" if mp is not None else ""), (not hidden) and (mp is None or "\u00b1" in pg.evaluate("document.getElementById('bMeas').textContent")), "")
            fp = pg.evaluate("window.__tour.floorPoint()")
            if fp:
                fp2 = [fp[0] + min(80, vp["width"] * 0.15), fp[1] - 10]
                dm = pg.evaluate("([a, b]) => window.__tour.measure(a[0], a[1], b[0], b[1])", [fp, fp2])
                check(f"{label}: measuring two floor points gives a plausible length", dm is not None and 0.05 < dm < 30, f"{dm}")
                txt = pg.evaluate("document.getElementById('measure').textContent")
                check(f"{label}: the length is shown in feet and inches", " ft " in txt and " in " in txt, txt)
            if fp:                                                                         # pinned notes: place one on the floor spot, see its pin, keep it over a reload, delete it
                pg.evaluate("window.__tour.go(0)"); pg.wait_for_function("!window.__tour.state().busy", timeout=30000)
                lon_before = pg.evaluate("window.__tour.state().lon")
                n0 = pg.evaluate("window.__tour.notes.list().length")
                pos = pg.evaluate("([x, y]) => window.__tour.notes.place(x, y)", fp)
                check(f"{label}: a note can be pinned on a surface", bool(pos) and abs(pos["z"]) < 0.6, f"{pos}")
                if pos:
                    pg.evaluate("window.__tour.notes.add('QA note: check the panel here')")
                    pg.wait_for_timeout(400)
                    lst = pg.evaluate("window.__tour.notes.list()")
                    check(f"{label}: the note is kept and shown as a pin in the view", len(lst) == n0 + 1 and pg.evaluate("window.__tour.notes.pins()") >= 1, f"{len(lst)} notes, {pg.evaluate('window.__tour.notes.pins()')} pins")
                    pg.reload(); pg.wait_for_function("window.__tour && !window.__tour.state().busy", timeout=60000); pg.wait_for_timeout(500)
                    lst2 = pg.evaluate("window.__tour.notes.list()")
                    check(f"{label}: the note is still there after a reload", any("QA note" in (n.get("text") or "") for n in lst2), f"{len(lst2)} notes")
                    exported = pg.evaluate("window.__tour.notes.list().filter(n => n.text.includes('QA note')).length")
                    # the same spot seen from the neighbouring stop: the pin is kept in map metres, so where it shows there must be the same real place
                    if len(N) > 1:
                        nid_ = pg.evaluate("window.__tour.notes.list().filter(n => n.text.includes('QA note'))[0].id")
                        pg.evaluate("window.__tour.go(1)"); pg.wait_for_function("window.__tour.state().cur === 1 && !window.__tour.state().busy", timeout=30000); pg.wait_for_timeout(300)
                        pg.evaluate("id => window.__tour.notes.aimAt(id)", nid_); pg.wait_for_timeout(400)
                        sc = pg.evaluate("id => window.__tour.notes.screen(id)", nid_)
                        if sc and 0 < sc[0] < vp["width"] and 0 < sc[1] < vp["height"] - 120:
                            p2 = pg.evaluate("([x, y]) => window.__tour.notes.place(x, y - 6)", sc)
                            pg.evaluate("document.getElementById('noteClose').click()")
                            dd = math.hypot(p2["x"] - pos["x"], p2["y"] - pos["y"]) if p2 else 99
                            check(f"{label}: from the next stop the pin sits on the same real spot", p2 is not None and dd < 1.2, f"{dd:.2f} m apart")
                        else:
                            hid = nid_ not in pg.evaluate("window.__tour.notes.visible()")
                            check(f"{label}: from the next stop the note is hidden (behind something, or too far) rather than misplaced", hid, f"screen {sc}")
                        pg.evaluate("window.__tour.go(0)"); pg.wait_for_function("window.__tour.state().cur === 0 && !window.__tour.state().busy", timeout=30000)
                    pg.evaluate("window.__tour.notes.remove(window.__tour.notes.list().filter(n => n.text.includes('QA note')).map(n => n.id)[0])")
                    check(f"{label}: a note can be deleted", pg.evaluate("window.__tour.notes.list().filter(n => n.text.includes('QA note')).length") == 0, "")
                pg.evaluate(f"window.__tour.aim({lon_before}, -12)")                      # the rest of the run looks the same way as it did before this test
            if tj.get("dollhouse"):
                check(f"{label}: the Dollhouse button is shown", pg.evaluate("!document.getElementById('bDoll').hidden"), "")
                pg.click("#bDoll")
                try:
                    pg.wait_for_function("window.__tour.doll.state().loaded && window.__tour.doll.state().on", timeout=60000)
                    ds = pg.evaluate("window.__tour.doll.state()")
                except Exception:
                    ds = pg.evaluate("window.__tour.doll.state()")
                check(f"{label}: the dollhouse opens and its model loads", ds["on"] and ds["loaded"] and ds["info"]["triangles"] > 5000, str(ds))
                report.setdefault("dollhouse", {})[label] = ds["info"]
                budget = 6 * 1048576
                check(f"{label}: the dollhouse model is small enough for a phone (under 6 MB)", ds["info"]["bytes"] < budget, f"{ds['info']['bytes'] / 1048576:.2f} MB, {ds['info']['triangles']} triangles")
                a0 = ds["az"]
                if touch:
                    pg.evaluate("window.__tour.doll.drag(120, 20)")                      # a drag through the same handler (mobile WebKit has no mouse)
                else:
                    pg.mouse.move(vp["width"] / 2, vp["height"] / 2); pg.mouse.down(); pg.mouse.move(vp["width"] / 2 + 120, vp["height"] / 2 + 20, steps=6); pg.mouse.up()
                pg.wait_for_timeout(200)
                check(f"{label}: dragging orbits the model", abs(pg.evaluate("window.__tour.doll.state().az") - a0) > 0.2, "")
                target = 5 if len(N) > 5 else len(N) - 1
                sc = pg.evaluate("i => window.__tour.doll.markerScreen(i)", target)
                check(f"{label}: stop markers sit on screen in the dollhouse", sc is not None and 0 < sc[0] < vp["width"] and 0 < sc[1] < vp["height"], str(sc))
                if sc:
                    pg.touchscreen.tap(sc[0], sc[1]) if touch else pg.mouse.click(sc[0], sc[1])
                    try:
                        pg.wait_for_function("!window.__tour.doll.state().on && !window.__tour.state().busy", timeout=30000)
                    except Exception:
                        pass
                    st = pg.evaluate("({on: window.__tour.doll.state().on, cur: window.__tour.state().cur})")
                    check(f"{label}: tapping a stop in the dollhouse flies back into that stop", (not st["on"]) and st["cur"] == target, str(st))
            if tj.get("plan"):
                check(f"{label}: the Plan button is shown", pg.evaluate("!document.getElementById('bPlan').hidden"), "")
                pg.click("#bPlan"); pg.wait_for_timeout(500)
                check(f"{label}: the floor plan opens as an SVG with dimensions", pg.evaluate("!!document.querySelector('#planview.show svg') && document.querySelectorAll('#planview svg text').length > 2"), "")
                pg.click("#planClose")
            if touch and tj.get("tiles"):                      # a phone keeps the 4K picture and lays sharp tiles over what it looks at, so it never holds the whole 8K one
                pg2 = ctx.new_page(); perr = []
                pg2.on("pageerror", lambda e: perr.append(str(e)))
                pg2.goto(url + "?tiles")
                try:
                    pg2.wait_for_function("window.__tour && !window.__tour.state().busy && window.__tour.state().tiles >= 2", timeout=30000)
                except Exception:
                    pass
                st2 = pg2.evaluate("window.__tour ? window.__tour.state() : {}")
                pg3 = ctx.new_page(); pg3.goto(url + "?notiles")
                try:
                    pg3.wait_for_function("window.__tour && window.__tour.state().res === '8K'", timeout=30000)
                except Exception:
                    pass
                st3 = pg3.evaluate("window.__tour ? window.__tour.state() : {}")
                check(f"{label}: sharp tiles load in tile mode", st2.get("tileMode") and st2.get("tiles", 0) >= 2 and st2.get("res") == "8K tiles" and not perr,
                      f"{st2.get('tiles')} tiles, {st2.get('res')}, errors {perr[:1]}")
                check(f"{label}: tiles hold far less full-detail picture than the whole 8K one", 0 < st2.get("hiMB", 0) < 0.6 * st3.get("hiMB", 0),
                      f"{st2.get('hiMB', 0):.0f} MB of tiles vs {st3.get('hiMB', 0):.0f} MB for the whole 8K picture")
                pg2.close(); pg3.close()
            stops = [int(s) - 1 for s in a.stops.split(",") if 0 < int(s) <= len(N)]
            for i in stops:
                pg.evaluate(f"window.__tour.go({i})")
                pg.wait_for_function(f"window.__tour.state().cur === {i} && !window.__tour.state().busy", timeout=30000)
                pg.wait_for_timeout(800)
                st = pg.evaluate("window.__tour.state()")
                check(f"{label}: stop {i + 1:03d} depth", st["hasDepth"])
                pg.screenshot(path=os.path.join(shots, f"{label}_{i + 1:03d}.png"))
                fp = pg.evaluate("window.__tour.floorPoint()")
                if not touch:                                     # nadir: straight down must show floor, not a black blob
                    pg.evaluate("window.__tour.aim(window.__tour.state().lon, -88)"); pg.wait_for_timeout(500)
                    png = pg.screenshot(path=os.path.join(shots, f"nadir_{i + 1:03d}.png"))
                    g = cv2.cvtColor(cv2.imdecode(np.frombuffer(png, np.uint8), cv2.IMREAD_COLOR), cv2.COLOR_BGR2GRAY)
                    hh, ww = g.shape; c = g[int(hh * 0.3):int(hh * 0.7), int(ww * 0.3):int(ww * 0.7)]
                    dark = float((c < min(45.0, 0.25 * float(np.median(g)))).mean())      # relative to the whole view, so a dim recording is not 'black'
                    report.setdefault("nadir_dark", {})[f"{i + 1:03d}"] = round(dark, 3)
                    check(f"{label}: stop {i + 1:03d} nadir not black", dark < 0.35, f"{dark * 100:.0f}% of the centre is near-black")
                    pg.evaluate(f"window.__tour.aim(window.__tour.state().lon, -12)"); pg.wait_for_timeout(200)
                lon0 = pg.evaluate("window.__tour.state().lon")
                for dl, la in [(d, t) for d in (0, 90, 180, 270) for t in (-30, -50)]:   # a wall can fill the first view: look down / turn round like a person would
                    if fp:
                        break
                    pg.evaluate(f"window.__tour.aim({lon0 + dl}, {la})"); pg.wait_for_timeout(300)
                    fp = pg.evaluate("window.__tour.floorPoint()")
                    if fp:
                        pg.screenshot(path=os.path.join(shots, f"{label}_{i + 1:03d}_down.png"))
                if not fp:
                    check(f"{label}: stop {i + 1:03d} floor visible", False, "no floor found in the lower half of the view"); continue
                if not touch:
                    pg.mouse.move(fp[0], fp[1]); pg.wait_for_timeout(400)
                    check(f"{label}: stop {i + 1:03d} cursor", pg.evaluate("window.__tour.state().cursor"))
                    pg.screenshot(path=os.path.join(shots, f"{label}_{i + 1:03d}_cursor.png"))
                tt = pg.evaluate(f"window.__tour.tapTarget({fp[0]}, {fp[1]})")     # where this tap goes and how far that stop is from the tapped spot
                check(f"{label}: stop {i + 1:03d} tap lands near the tapped spot", bool(tt) and tt["dist"] <= tap_limit, f"{tt['dist']:.1f} m to stop {tt['to'] + 1:03d} (limit {tap_limit:.1f} m)" if tt else "no target")
                before = pg.evaluate("window.__tour.state()")
                world0 = before["lon"] + (N[before["cur"]].get("H") or 0)
                if touch:
                    pg.touchscreen.tap(fp[0], fp[1])
                else:
                    pg.mouse.click(fp[0], fp[1])
                try:
                    pg.wait_for_function(f"window.__tour.state().cur !== {before['cur']} && !window.__tour.state().busy", timeout=15000)
                    after = pg.evaluate("window.__tour.state()")
                    world1 = after["lon"] + (N[after["cur"]].get("H") or 0)
                    report["glides_ms"].append(after["lastGlideMs"])
                    check(f"{label}: stop {i + 1:03d} tap-to-go", True, f"-> {after['cur'] + 1:03d}")
                    check(f"{label}: stop {i + 1:03d} keeps facing", abs(wrap(world1 - world0)) < 25, f"drift {wrap(world1 - world0):.1f} deg")
                    if after["lastGlideMs"]:
                        check(f"{label}: stop {i + 1:03d} glide time", 250 <= after["lastGlideMs"] <= 1100, f"{after['lastGlideMs']:.0f} ms (headless software rendering runs slower than a real GPU)")
                except Exception:
                    check(f"{label}: stop {i + 1:03d} tap-to-go", False, "click on floor did not move")
            if touch and tj.get("tiles") and len(stops) >= 4:      # a phone must not keep every picture it has visited: memory stays flat however long the visit
                stt = pg.evaluate("window.__tour.state()")
                check(f"{label}: picture memory stays flat after {len(stops)} stops", stt.get("texMB", 0) < 340, f"{stt.get('texMB', 0):.0f} MB (whole 8K path: about 490 MB)")
            ctx.close()
        if len(N) >= 4:                                       # deep links: #NNN opens that stop, and the address follows you
            ctx = b.new_context(viewport={"width": 960, "height": 600}); ctx.route(lambda url: not url.startswith("http://127.0.0.1"), lambda r: r.abort())
            pg = ctx.new_page(); pg.goto(url + "#" + N[3]["id"])
            try:
                pg.wait_for_function("window.__tour && !window.__tour.state().busy", timeout=60000)
                first = pg.evaluate("window.__tour.state().cur")
                pg.evaluate("window.__tour.go(1)"); pg.wait_for_function("window.__tour.state().cur === 1 && !window.__tour.state().busy", timeout=20000)
                h = pg.evaluate("location.hash")
                check("deep link opens that stop and the address follows you", first == 3 and h == "#" + N[1]["id"], f"opened at stop {first + 1}, address {h}")
            except Exception as e:
                check("deep link opens that stop and the address follows you", False, str(e)[:150])
            ctx.close()
        if len(N) >= 4:                                       # impatient visitor: taps and keys while glides are running
            import random
            rnd = random.Random(3)
            ctx = b.new_context(viewport={"width": 960, "height": 600}); ctx.route(lambda url: not url.startswith("http://127.0.0.1"), lambda r: r.abort())
            pg = ctx.new_page(); perr = []
            pg.on("pageerror", lambda e: perr.append(str(e)[:80])); pg.goto(url)
            try:
                pg.wait_for_function("window.__tour && !window.__tour.state().busy", timeout=60000)
                pg.evaluate("document.getElementById('help') && document.getElementById('help').classList.remove('show')"); pg.wait_for_timeout(800)
                for k in range(8):
                    fp = pg.evaluate("window.__tour.floorPoint()")
                    if fp:
                        pg.mouse.click(fp[0] + rnd.randint(-40, 40), min(590, fp[1] + rnd.randint(-20, 20)))
                    if k % 3 == 0:
                        pg.keyboard.press(rnd.choice(["n", "p", "w", "s"]))
                    pg.wait_for_timeout(rnd.randint(30, 150))
                pg.wait_for_function("!window.__tour.state().busy", timeout=20000)
                check("rapid taps and keys during glides settle without errors", not perr, f"page errors {perr[:1]}" if perr else "settled")
            except Exception as e:
                check("rapid taps and keys during glides settle without errors", False, str(e)[:150])
            ctx.close()
        if len(N) >= 3:                                       # pictures decoded as bitmaps must come out the same way up and looking the same as the plain way
            shots_ = []; used_ = None
            for q_ in ("", "?nobitmap"):
                ctx = b.new_context(viewport={"width": 960, "height": 600}); ctx.route(lambda u: not u.startswith("http://127.0.0.1"), lambda r: r.abort())
                pg = ctx.new_page(); pg.goto(url + q_ + "#" + N[2]["id"])
                try:
                    pg.wait_for_function("window.__tour && !window.__tour.state().busy", timeout=60000)
                    pg.evaluate("document.getElementById('help') && document.getElementById('help').classList.remove('show')")
                    pg.wait_for_function("window.__tour.state().res !== '1K'", timeout=30000); pg.wait_for_timeout(3000)
                    if q_ == "":
                        used_ = pg.evaluate("window.__tour.state().bitmap")
                    shots_.append(cv2.cvtColor(cv2.imdecode(np.frombuffer(pg.screenshot(), np.uint8), 1), cv2.COLOR_BGR2GRAY).astype(np.float32))
                except Exception:
                    pass
                ctx.close()
            if len(shots_) == 2:
                dd_ = float(np.abs(shots_[0] - shots_[1]).mean())
                check("pictures decoded as bitmaps look the same way up as the plain way", dd_ < 8, f"mean difference {dd_:.2f} of 255, bitmap path {'used' if used_ else 'not used by this browser'}")
            else:
                check("pictures decoded as bitmaps look the same way up as the plain way", False, "could not load")
        if N[0].get("fx"):                                    # the tiny first picture is a bonus too: without it the 4K picture must open the tour
            ctx = b.new_context(viewport={"width": 960, "height": 600})
            ctx.route(lambda u: not u.startswith("http://127.0.0.1"), lambda r: r.abort()); ctx.route("**/panos_xs/*", lambda r: r.abort())
            pg = ctx.new_page(); perr = []
            pg.on("pageerror", lambda e: perr.append(str(e)[:80])); pg.goto(url)
            try:
                pg.wait_for_function("window.__tour && !window.__tour.state().busy", timeout=60000)
                pg.wait_for_function("window.__tour.state().res !== '1K'", timeout=20000)
                check("a missing tiny first picture falls back to the 4K one", not perr, f"res {pg.evaluate('window.__tour.state().res')}, errors {perr[:1]}")
            except Exception as e:
                check("a missing tiny first picture falls back to the 4K one", False, str(e)[:150])
            ctx.close()
        if tj.get("tiles") and len(N) >= 3:                   # sharp tiles are a bonus: if they cannot be fetched the 4K picture must carry on, quietly
            ctx = b.new_context(viewport={"width": 390, "height": 844}, has_touch=True, **({"is_mobile": True} if a.browser != "firefox" else {}))
            ctx.route(lambda u: not u.startswith("http://127.0.0.1"), lambda r: r.abort()); ctx.route("**/tiles/*", lambda r: r.abort())
            pg = ctx.new_page(); perr = []
            pg.on("pageerror", lambda e: perr.append(str(e)[:80])); pg.goto(url + "?tiles")
            try:
                pg.wait_for_function("window.__tour && !window.__tour.state().busy", timeout=60000)
                pg.wait_for_timeout(2500)
                pg.evaluate("window.__tour.go(2)"); pg.wait_for_function("window.__tour.state().cur === 2 && !window.__tour.state().busy", timeout=30000)
                pg.wait_for_timeout(1500)
                st = pg.evaluate("window.__tour.state()")
                check("tiles that cannot be fetched leave the 4K picture working", st["tileMode"] and st["tiles"] == 0 and st["cur"] == 2 and not perr,
                      f"{st['tiles']} tiles, stop {st['cur'] + 1}, errors {perr[:1]}")
            except Exception as e:
                check("tiles that cannot be fetched leave the 4K picture working", False, str(e)[:150])
            ctx.close()
        if len(N) >= 4:                                       # a picture that fails to load: message, stay put, keep working
            ctx = b.new_context(viewport={"width": 960, "height": 600}); ctx.route(lambda url: not url.startswith("http://127.0.0.1"), lambda r: r.abort())
            pg = ctx.new_page(); perr = []
            pg.on("pageerror", lambda e: perr.append(str(e)[:80]))
            bad = N[2]["f"].split("?")[0]
            pg.route(lambda url: bad in url, lambda r: r.abort())
            pg.goto(url)
            try:
                pg.wait_for_function("window.__tour && !window.__tour.state().busy", timeout=60000)
                pg.evaluate("window.__tour.go(2)"); pg.wait_for_timeout(5000)
                st = pg.evaluate("window.__tour.state()"); msg = pg.evaluate("document.getElementById('toast').textContent")
                pg.evaluate("window.__tour.go(1)"); pg.wait_for_function("window.__tour.state().cur === 1 && !window.__tour.state().busy", timeout=20000)
                check("a picture that fails to load is reported and the viewer keeps working", st["cur"] == 0 and not st["busy"] and "Could not load" in msg and not perr,
                      f"stayed at stop {st['cur'] + 1}, message '{msg}', page errors {perr[:1]}")
            except Exception as e:
                check("a picture that fails to load is reported and the viewer keeps working", False, str(e)[:150])
            ctx.close()
        b.close()
    WF = os.path.join(os.path.dirname(os.path.abspath(a.tour)), "frames")        # the walk's frames/ sits beside its tour/
    tf = os.path.join(WF, "timings.json")
    if os.path.exists(tf):
        report["pipeline_timings_s"] = json.load(open(tf))
    pj = os.path.join(WF, "points.json")
    if os.path.exists(pj):
        report["stops"] = len(json.load(open(pj)))
    fails = [c for c in report["checks"] if not c["ok"]]
    report["passed"] = not fails
    report["summary"] = f"{len(report['checks']) - len(fails)}/{len(report['checks'])} checks passed"
    json.dump(report, open(os.path.join(HERE, "qa", "report.json"), "w"), indent=1)
    print("\n" + report["summary"] + ("" if not fails else "  FAILED: " + ", ".join(c["check"] for c in fails)))
    print("report: qa/report.json   screenshots: qa/shots/")
    sys.exit(0 if not fails else 1)


if __name__ == "__main__":
    main()
