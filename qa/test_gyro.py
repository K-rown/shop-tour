"""MOTION: the viewer turns with the phone (the Motion key in the tools row). Opens a built tour in a headless phone-sized browser, sends it
synthetic deviceorientation events the way a phone does, and checks:
  - the key is there on a touch screen, off at first (aria-pressed false), its target at least 44 px tall
  - the reel and the thumb cluster rest clear of a phone's 52 px foot strip (Android's nav bar reports no safe-area inset)
  - with Motion off, turning the phone moves nothing
  - with Motion on, the view's yaw follows the phone's turn (alpha) and its pitch the phone's tilt (beta); pressing it never jumps the yaw
  - a drag turns the yaw AND the pitch while Motion is on; the phone's turns add to wherever the finger left it (2026-10-30)
  - pressed again, Motion is off and the phone moves nothing
  - no reading within 2.5 s of the press: the key says "No sensor" and goes back off
  - iOS: the press asks DeviceOrientationEvent.requestPermission; refused, the key says "Not allowed" and stays off; granted, it turns on;
    Motion is a press on every open (2026-10-30; it used to open on where a phone had chosen it, and a finger drag then fought the phone's tilt)

Run:  python qa/test_gyro.py --tour tour        (a built tour folder; the viewer under test is viewer/index.html, served in its place)
"""
import argparse, functools, http.server, json, math, os, socket, sys, threading, time

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VIEWER = os.path.join(HERE, "viewer", "index.html")


def serve(folder):
    s = socket.socket(); s.bind(("127.0.0.1", 0)); port = s.getsockname()[1]; s.close()

    class H(http.server.SimpleHTTPRequestHandler):
        def log_message(self, *a, **k):
            pass

        def translate_path(self, path):                      # index.html is the viewer under test; everything else is the built tour's
            p = path.split("?")[0].split("#")[0]
            if p in ("/", "/index.html"):
                return VIEWER
            return super().translate_path(path)

    class Quiet(http.server.ThreadingHTTPServer):
        def handle_error(self, request, client_address):
            if not isinstance(sys.exc_info()[1], ConnectionError):
                super().handle_error(request, client_address)
    srv = Quiet(("127.0.0.1", port), functools.partial(H, directory=folder))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return f"http://127.0.0.1:{port}/index.html"


def wrap(a):
    return (a + 540) % 360 - 180


FAILS = []


def check(name, ok, detail=""):
    print(("PASS " if ok else "FAIL ") + name + (f"  ({detail})" if detail else ""), flush=True)
    if not ok:
        FAILS.append(name)


TILT = """([a, b, g, type]) => window.dispatchEvent(new DeviceOrientationEvent(type || 'deviceorientation', { alpha: a, beta: b, gamma: g, absolute: false }))"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tour", default=os.path.join(HERE, "tour"))
    ap.add_argument("--headed", action="store_true")
    a = ap.parse_args()
    if not os.path.exists(os.path.join(a.tour, "tour.json")):
        sys.exit(f"no built tour in {a.tour} (tour.json is missing)")
    url = serve(a.tour)
    from playwright.sync_api import sync_playwright
    with sync_playwright() as p:
        b = p.chromium.launch(headless=not a.headed, args=["--ignore-gpu-blocklist", "--enable-webgl", "--enable-unsafe-swiftshader"])

        def phone(init=None):
            ctx = b.new_context(viewport={"width": 412, "height": 915}, has_touch=True, is_mobile=True, device_scale_factor=1)
            ctx.route(lambda u: not u.startswith("http://127.0.0.1"), lambda r: r.abort())       # nothing from the internet
            if init:
                ctx.add_init_script(init)
            return ctx

        def open_(ctx):
            pg = ctx.new_page(); errs = []
            pg.on("pageerror", lambda e: errs.append(str(e)))
            pg.goto(url)
            pg.wait_for_function("window.__tour && !window.__tour.state().busy && !document.getElementById('load')", timeout=90000)
            pg.evaluate("document.getElementById('help') && document.getElementById('help').classList.remove('show')")
            return pg, errs

        def st(pg):
            return pg.evaluate("window.__tour.state()")

        def tilt(pg, al, be, ga, n=4):
            for _ in range(n):                                  # a phone sends a stream; the render loop reads the newest
                pg.evaluate(TILT, [al, be, ga, None]); pg.wait_for_timeout(40)
            pg.wait_for_timeout(120)

        # ---------- Android-like: no permission to ask ----------
        ctx = phone()
        pg, errs = open_(ctx)
        key = pg.locator("#bGyro")
        check("the Motion key is shown on a touch screen", key.is_visible())
        check("it is off at first", key.get_attribute("aria-pressed") == "false" and st(pg)["gyro"] is False)
        bb = key.bounding_box()
        cx = bb["x"] + bb["width"] / 2
        hit = pg.evaluate("([x, y]) => { const e = document.elementFromPoint(x, y); return !!e && e.id === 'bGyro'; }", [cx, bb["y"] - 3]) and \
            pg.evaluate("([x, y]) => { const e = document.elementFromPoint(x, y); return !!e && e.id === 'bGyro'; }", [cx, bb["y"] + bb["height"] + 3])
        check("its target is at least 44 px tall and wide", hit and bb["width"] >= 44 and bb["height"] + 8 >= 44, f"{bb['width']:.0f}x{bb['height']:.0f} drawn, +5 px each side")

        geo = pg.evaluate("""() => { const b = id => document.getElementById(id).getBoundingClientRect(), r = document.querySelector('#reel button').getBoundingClientRect(), t = b('tools'), n = b('nav');
          return { reelTop: r.top, reelBottom: r.bottom, toolsTop: t.top, toolsBottom: t.bottom, navTop: n.top, navBottom: n.bottom, h: innerHeight }; }""")
        # the round-glass layout (2026-10-31): the stops, under them the tools, under them the walk keys — none over another, all clear of the strip
        check("the reel, the tools and the walk keys stack without overlapping, clear of the phone's 52 px foot strip",
              geo["reelBottom"] <= geo["toolsTop"] and geo["toolsBottom"] <= geo["navTop"] and geo["navBottom"] <= geo["h"] - 52, json.dumps(geo))
        lon0, lat0 = st(pg)["lon"], st(pg)["lat"]
        tilt(pg, 0, 90, 0); tilt(pg, 45, 120, 0)
        s = st(pg)
        check("Motion off: turning the phone moves nothing", abs(wrap(s["lon"] - lon0)) < 0.01 and abs(s["lat"] - lat0) < 0.01, f"lon {lon0:.1f}->{s['lon']:.1f} lat {lat0:.1f}->{s['lat']:.1f}")

        key.click(); pg.wait_for_timeout(100)
        check("pressed, it is on", key.get_attribute("aria-pressed") == "true" and st(pg)["gyro"] is True)
        lonA = st(pg)["lon"]
        tilt(pg, 0, 90, 0)                                      # held upright, facing the way the view already looks
        s = st(pg)
        check("the first reading never jumps the yaw", abs(wrap(s["lon"] - lonA)) < 0.5, f"{lonA:.1f}->{s['lon']:.1f}")
        check("held upright, the pitch is level", abs(s["lat"]) < 1.5, f"lat {s['lat']:.1f}")
        check("it reads deviceorientation", s["gyroSrc"] == "deviceorientation", str(s["gyroSrc"]))
        lonB = s["lon"]
        tilt(pg, 30, 90, 0)                                     # the phone turned 30 degrees to the left (alpha counter-clockwise from above)
        s = st(pg)
        check("turning the phone 30 deg left turns the view 30 deg left", abs(wrap(s["lon"] - lonB) + 30) < 1.5, f"yaw moved {wrap(s['lon'] - lonB):.1f}")
        lonC = s["lon"]
        tilt(pg, 30, 120, 0)                                    # its top tipped back: the camera looks 30 degrees up
        s = st(pg)
        check("tipping the phone back 30 deg looks 30 deg up", abs(s["lat"] - 30) < 1.5 and abs(wrap(s["lon"] - lonC)) < 1.5, f"lat {s['lat']:.1f}, yaw moved {wrap(s['lon'] - lonC):.1f}")
        tilt(pg, 30, 30, 0)                                     # pointed down at the floor: the yaw holds (steadied by the screen's top edge)
        s = st(pg)
        check("pointing the phone down looks down, the yaw held", abs(s["lat"] + 60) < 1.5 and abs(wrap(s["lon"] - lonC)) < 1.5, f"lat {s['lat']:.1f}, yaw moved {wrap(s['lon'] - lonC):.1f}")
        tilt(pg, 5, 90, 0); lonF = st(pg)["lon"]
        tilt(pg, 355, 90, 0)                                    # through north: 10 degrees to the right, never 350 to the left
        s = st(pg)
        check("a turn through 0/360 is the short way", abs(wrap(s["lon"] - lonF) - 10) < 1.5, f"yaw moved {wrap(s['lon'] - lonF):.1f}")
        tilt(pg, 30, 90, 0)

        # a drag still turns the yaw; the pitch stays the phone's
        s0 = st(pg)
        cv = pg.locator("#stage canvas")
        box = cv.bounding_box(); x0, y0 = box["x"] + box["width"] * 0.5, box["y"] + box["height"] * 0.4
        pg.mouse.move(x0, y0); pg.mouse.down(); pg.mouse.move(x0 - 120, y0 + 80, steps=8); pg.wait_for_timeout(150); pg.mouse.up(); pg.wait_for_timeout(200)
        tilt(pg, 30, 90, 0)
        s1 = st(pg)
        check("a drag still turns the yaw with Motion on", abs(wrap(s1["lon"] - s0["lon"])) > 5, f"yaw moved {wrap(s1['lon'] - s0['lon']):.1f}")
        check("a drag turns the pitch too (2026-10-30: the finger and the phone add up, they never fight)", s1["lat"] - s0["lat"] > 3, f"lat {s0['lat']:.1f} -> {s1['lat']:.1f}")
        lonD, latD = s1["lon"], s1["lat"]
        tilt(pg, 50, 90, 0)
        s2 = st(pg)
        check("after a drag the phone still turns the view", abs(wrap(s2["lon"] - lonD) + 20) < 1.5, f"yaw moved {wrap(s2['lon'] - lonD):.1f}")
        tilt(pg, 50, 110, 0)                                    # its top tipped back 20 degrees: 20 degrees up from where the finger left it
        s2 = st(pg)
        check("tipping the phone back 20 deg adds 20 deg to where the finger left it", abs((s2["lat"] - latD) - 20) < 1.5, f"lat {latD:.1f} -> {s2['lat']:.1f}")
        check("Motion is remembered as a preference", pg.evaluate("localStorage.getItem('tour:gyro')") == "1")

        key.click(); pg.wait_for_timeout(100)
        s3 = st(pg)
        tilt(pg, 120, 140, 0)
        s4 = st(pg)
        check("pressed again it is off, and the phone moves nothing", key.get_attribute("aria-pressed") == "false" and abs(wrap(s4["lon"] - s3["lon"])) < 0.01 and abs(s4["lat"] - s3["lat"]) < 0.01)
        check("off is remembered too", pg.evaluate("localStorage.getItem('tour:gyro')") == "0")

        # no sensor: nothing comes within 2.5 s
        key.click(); pg.wait_for_timeout(2900)
        check("no reading in 2.5 s: it says so and goes back off", key.get_attribute("aria-pressed") == "false" and key.inner_text().strip() == "No sensor" and st(pg)["gyro"] is False, key.inner_text())
        pg.wait_for_timeout(2700)
        check("the key reads Motion again after", key.inner_text().strip() == "Motion")
        check("no page errors", not errs, "; ".join(errs[:2]))

        # chosen before: still OFF on the next open — Motion is a press, every open (2026-10-30: opened on, a drag felt as if the picture rolled)
        pg.evaluate("localStorage.setItem('tour:gyro', '1')"); pg.close()
        pg, errs = open_(ctx)
        check("a phone that chose Motion still opens with it off", st(pg)["gyro"] is False and pg.locator("#bGyro").get_attribute("aria-pressed") == "false")
        ctx.close()

        # ---------- iOS-like: the press asks ----------
        for ans in ("denied", "granted"):
            ctx = phone("window.__asked = 0; DeviceOrientationEvent.requestPermission = () => { window.__asked++; return Promise.resolve('" + ans + "'); };"
                        "try { localStorage.setItem('tour:gyro', '1'); } catch (e) {}")
            pg, errs = open_(ctx)
            key = pg.locator("#bGyro")
            check(f"iOS ({ans}): it never opens on by itself, even chosen before", st(pg)["gyro"] is False and pg.evaluate("window.__asked") == 0)
            key.click(); pg.wait_for_timeout(150)
            asked = pg.evaluate("window.__asked")
            if ans == "denied":
                check("iOS refused: asked once, the key says Not allowed and stays off", asked == 1 and key.get_attribute("aria-pressed") == "false" and key.inner_text().strip() == "Not allowed", key.inner_text())
            else:
                lonE = st(pg)["lon"]; tilt(pg, 0, 90, 0); tilt(pg, -20, 90, 0)
                s = st(pg)
                check("iOS granted: asked once, on, and the view turns with the phone", asked == 1 and s["gyro"] is True and abs(wrap(s["lon"] - lonE) - 20) < 1.5, f"yaw moved {wrap(s['lon'] - lonE):.1f}")
            check(f"iOS ({ans}): no page errors", not errs, "; ".join(errs[:2]))
            ctx.close()

        # ---------- a desktop shows no Motion key ----------
        ctx = b.new_context(viewport={"width": 1280, "height": 800})
        ctx.route(lambda u: not u.startswith("http://127.0.0.1"), lambda r: r.abort())
        pg, errs = open_(ctx)
        check("a desktop (a mouse) has no Motion key", not pg.locator("#bGyro").is_visible())
        ctx.close()
        b.close()
    print(("ALL PASS" if not FAILS else f"{len(FAILS)} FAILED: " + ", ".join(FAILS)), flush=True)
    sys.exit(1 if FAILS else 0)


if __name__ == "__main__":
    main()
