"""THE RINGS ON THE FLOOR: the discs to the neighbouring stops, and the maps. Opens a built tour with viewer/index.html served in its place, and checks:
  - every ring lies on the floor under its stop (this stop's camera height below the eye, plus the ground's climb to the other stop), never
    floating: the old viewer clamped the link's angle to 60 degrees, so a stop under 1.3 m away hung up to a metre over the floor
  - a stop behind a wall (occ) has no ring; the walk's own next and previous stops always do, however far
  - where the depth surface is measured along a ring's ray, the ring stands near it (the median miss is reported; under 20 % passes)
  - desktop: the pointer over a ring lights it (state "hover") and turns into a hand; a click glides there
  - phone: a press on a ring shows it pressed; a touch 22 px off its centre still finds it (a 44 px target at least); lifting walks there
  - the big map: no two stop numbers overlap
  - no page errors
Run:  python qa/test_rings.py --tour tour
"""
import argparse, functools, http.server, json, math, os, socket, sys, threading, time

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VIEWER = os.path.join(HERE, "viewer", "index.html")
FAILS = []


def check(name, ok, detail=""):
    print(("PASS " if ok else "FAIL ") + name + (f"  ({detail})" if detail else ""), flush=True)
    if not ok:
        FAILS.append(name)


def serve(folder):
    s = socket.socket(); s.bind(("127.0.0.1", 0)); port = s.getsockname()[1]; s.close()

    class H(http.server.SimpleHTTPRequestHandler):
        def log_message(self, *a, **k):
            pass

        def translate_path(self, path):
            p = path.split("?")[0].split("#")[0]
            return VIEWER if p in ("/", "/index.html") else super().translate_path(path)
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", port), functools.partial(H, directory=folder))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return f"http://127.0.0.1:{port}/index.html"


def ready(pg, i=None):
    if i is None:
        pg.wait_for_function("window.__tour && !window.__tour.state().busy", timeout=60000)
    else:
        pg.evaluate(f"window.__tour.go({i})")
        pg.wait_for_function(f"window.__tour.state().cur === {i} && !window.__tour.state().busy", timeout=60000)
    time.sleep(0.6)


def aim_at(pg, pos, down=8):
    az = math.degrees(math.atan2(pos[0], -pos[2])); el = math.degrees(math.atan2(pos[1], math.hypot(pos[0], pos[2])))
    pg.evaluate(f"window.__tour.aim({az}, {el + down})"); time.sleep(0.5)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--tour", default=os.path.join(HERE, "tour")); ap.add_argument("--headed", action="store_true")
    a = ap.parse_args()
    from playwright.sync_api import sync_playwright
    tour = json.load(open(os.path.join(a.tour, "tour.json"), encoding="utf-8")); N = tour["nodes"]
    url = serve(a.tour)
    with sync_playwright() as p:
        b = p.chromium.launch(headless=not a.headed, args=["--ignore-gpu-blocklist", "--enable-webgl", "--enable-unsafe-swiftshader"])
        # ---- geometry, every stop
        ctx = b.new_context(viewport={"width": 1280, "height": 800}); ctx.add_init_script("try{localStorage.setItem('tour:hint','1')}catch(e){}")
        pg = ctx.new_page(); errs = []; pg.on("pageerror", lambda e: errs.append(str(e)))
        pg.goto(url); ready(pg)
        off, miss, wrong_set = [], [], []
        for i, n in enumerate(N):
            ready(pg, i)
            rings = pg.evaluate("window.__tour.rings()"); got = {r["to"] for r in rings}
            for L in n["links"]:
                if L.get("occ") and L["to"] in got: wrong_set.append(f"{i}->{L['to']} behind a wall")
                if L.get("near") and not L.get("occ") and L.get("d", 3) >= 0.3 and L["to"] not in got: wrong_set.append(f"{i}->{L['to']} next/prev missing")
            for r in rings:
                L = next(l for l in n["links"] if l["to"] == r["to"]); m = N[r["to"]]; d = L.get("d", 3)
                dz = max(-(0.6 * d + 0.3), min(0.6 * d + 0.3, (m.get("z") or 0) - (n.get("z") or 0)))
                off.append(abs(r["pos"][1] - (-(n.get("h") or 2.1) + dz + 0.02)))
                x, y, z = r["pos"]; hd = math.hypot(x, z)
                dep = pg.evaluate(f"window.__tm.rayDepth({math.degrees(math.atan2(x, -z))}, {math.degrees(math.atan2(y, hd))})")
                if dep: miss.append(abs(math.hypot(hd, y) / dep - 1))
        check("every ring on its stop's floor", max(off or [0]) < 1e-3, f"worst {max(off or [0]):.4f} m")
        check("no ring behind a wall, the walk's next/prev always", not wrong_set, "; ".join(wrong_set[:5]))
        med = sorted(miss)[len(miss) // 2] if miss else 0
        check("rings stand near the measured surface", med < 0.2, f"median miss {med:.0%} over {len(miss)} rings")
        # ---- desktop hover + click
        i = next(k for k, n in enumerate(N) if any(not l.get("occ") and l.get("d", 3) >= 0.3 for l in n["links"]))
        ready(pg, i); r0 = pg.evaluate("window.__tour.rings()")[0]; aim_at(pg, r0["pos"])
        x, y = pg.evaluate(f"window.__tour.ringScreen({r0['to']})")
        pg.mouse.move(x + 2, y + 2); time.sleep(0.6)
        st = {r["to"]: r["state"] for r in pg.evaluate("window.__tour.rings()")}
        check("desktop: the pointer over a ring lights it", st.get(r0["to"]) == "hover", str(st))
        check("desktop: the pointer is a hand over a ring", pg.evaluate("getComputedStyle(document.querySelector('#stage canvas')).cursor") == "pointer")
        pg.mouse.click(x + 2, y + 2)
        pg.wait_for_function(f"window.__tour.state().cur === {r0['to']} && !window.__tour.state().busy", timeout=20000)
        check("desktop: a click on the ring glides there", pg.evaluate("window.__tour.state().lastGlideMs") > 0)
        # ---- the big map's numbers never overlap
        pg.mouse.move(5, 400); pg.click("#map"); time.sleep(0.5)
        lab = pg.evaluate("window.__tour.mapLabels()")
        over = sum(1 for k, p_ in enumerate(lab) for q in lab[k + 1:] if p_[0] < q[2] and p_[2] > q[0] and p_[1] < q[3] and p_[3] > q[1])
        check("big map: no two stop numbers overlap", len(lab) > 0 and over == 0, f"{len(lab)} numbers, {over} overlapping")
        check("no page errors (desktop)", not errs, str(errs[:3])); ctx.close()
        # ---- phone press
        ctx = b.new_context(viewport={"width": 412, "height": 915}, has_touch=True, is_mobile=True, device_scale_factor=2)
        ctx.add_init_script("try{localStorage.setItem('tour:hint','1')}catch(e){}")
        pg = ctx.new_page(); errs = []; pg.on("pageerror", lambda e: errs.append(str(e)))
        pg.goto(url); ready(pg); ready(pg, i)
        r0 = pg.evaluate("window.__tour.rings()")[0]; aim_at(pg, r0["pos"], 4)
        x, y = pg.evaluate(f"window.__tour.ringScreen({r0['to']})")
        check("phone: a touch 22 px off a ring's centre finds it", pg.evaluate(f"window.__tour.ringAt({x + 22}, {y})") == r0["to"])
        cdp = ctx.new_cdp_session(pg)
        cdp.send("Input.dispatchTouchEvent", {"type": "touchStart", "touchPoints": [{"x": x, "y": y}]}); time.sleep(0.3)
        st = {r["to"]: r["state"] for r in pg.evaluate("window.__tour.rings()")}
        check("phone: a press on a ring shows it pressed", st.get(r0["to"]) == "press", str(st))
        cdp.send("Input.dispatchTouchEvent", {"type": "touchEnd", "touchPoints": []})
        pg.wait_for_function(f"window.__tour.state().cur === {r0['to']} && !window.__tour.state().busy", timeout=20000)
        check("phone: lifting walks there", True)
        check("no page errors (phone)", not errs, str(errs[:3])); ctx.close(); b.close()
    print("ALL PASS" if not FAILS else f"{len(FAILS)} FAILED: {FAILS}")
    sys.exit(1 if FAILS else 0)


if __name__ == "__main__":
    main()
