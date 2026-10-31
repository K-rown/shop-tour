"""THE BLACK SCREEN: the 3D context is lost (a phone's web view drops it in the background, or under memory pressure) and the tour must draw again.
Opens a built tour with viewer/index.html served in its place (phone size, touch) and, with WEBGL_lose_context:
  - lost and given back by the browser: the picture draws again, no message, no page error
  - lost and not given back: the viewer asks for it back by itself and draws again
  - lost and refused (the context cannot come back): within a few seconds the page says so with "Tap to reload"; the key asks again, and when
    the context comes back the message goes and the picture draws
  - a context that is fine never shows the message
Run:  python qa/test_context_loss.py --tour tour
"""
import argparse, functools, http.server, os, socket, sys, threading, time

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


# the page keeps a handle on the extension so a test can give the context back (or refuse to)
LOSE = """() => { const c = document.querySelector('#stage canvas'), g = c.getContext('webgl2') || c.getContext('webgl');
  window.__ext = g.getExtension('WEBGL_lose_context'); window.__ext.loseContext(); return true; }"""


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--tour", default=os.path.join(HERE, "tour")); a = ap.parse_args()
    from playwright.sync_api import sync_playwright
    url = serve(a.tour)
    with sync_playwright() as p:
        b = p.chromium.launch(args=["--ignore-gpu-blocklist", "--enable-webgl", "--enable-unsafe-swiftshader"])

        def fresh():
            ctx = b.new_context(viewport={"width": 412, "height": 915}, has_touch=True, is_mobile=True, device_scale_factor=2)
            ctx.add_init_script("try{localStorage.setItem('tour:hint','1')}catch(e){}")
            pg = ctx.new_page(); errs = []; pg.on("pageerror", lambda e: errs.append(str(e)))
            pg.goto(url); pg.wait_for_function("window.__tour && !window.__tour.state().busy", timeout=60000); time.sleep(5)
            return ctx, pg, errs

        ctx, pg, errs = fresh()
        st = pg.evaluate("window.__tour.gl()")
        check("a healthy context draws and shows no message", st["draws"] and not st["fail"], str(st))
        # 1. lost, the browser gives it back
        pg.evaluate(LOSE); time.sleep(0.4)
        check("lost: the viewer sees it", pg.evaluate("window.__tour.gl()")["lost"])
        pg.evaluate("window.__ext.restoreContext()"); time.sleep(1.5)
        st = pg.evaluate("window.__tour.gl()")
        check("given back: the picture draws again, no message", st["draws"] and not st["fail"] and st["restores"] == 1, str(st))
        pg.screenshot(path=os.path.join(HERE, "qa", "shots", "ctx-restored.png")) if os.path.isdir(os.path.join(HERE, "qa", "shots")) else None
        # the tour still works after: a step to the next stop
        pg.evaluate("window.__tour.go(1)"); pg.wait_for_function("window.__tour.state().cur === 1 && !window.__tour.state().busy", timeout=30000); time.sleep(0.8)
        check("after a restore the tour walks on and draws", pg.evaluate("window.__tour.gl()")["draws"])
        check("no page errors (given back)", not errs, str(errs[:3])); ctx.close()
        # 2. lost, nobody gives it back: the viewer asks for it itself
        ctx, pg, errs = fresh()
        pg.evaluate(LOSE)
        pg.wait_for_function("!window.__tour.gl().lost", timeout=10000); time.sleep(1)
        st = pg.evaluate("window.__tour.gl()")
        check("not given back: the viewer asks for it and draws again", st["draws"] and not st["fail"], str(st))
        check("no page errors (asked back)", not errs, str(errs[:3])); ctx.close()
        # 3. lost and refused: the message, then the key
        ctx, pg, errs = fresh()
        pg.evaluate("""() => { const c = document.querySelector('#stage canvas'), g = c.getContext('webgl2') || c.getContext('webgl');
          const x = g.getExtension('WEBGL_lose_context'); window.__ext = x; window.__real = x.restoreContext.bind(x);
          x.restoreContext = () => {}; x.loseContext(); }""")        # the browser refuses every ask
        pg.wait_for_function("window.__tour.gl().fail", timeout=12000)
        check("refused: the page says so with a key", pg.is_visible("#glfail") and pg.inner_text("#glfailGo") == "Tap to reload")
        os.makedirs(os.path.join(HERE, "_scratch", "shots"), exist_ok=True)
        pg.screenshot(path=os.path.join(HERE, "_scratch", "shots", "after-context-lost-phone.png"))
        pg.evaluate("() => { const x = window.__ext; x.restoreContext = window.__real; }")   # the next ask is answered
        pg.tap("#glfailGo")
        pg.wait_for_function("!window.__tour.gl().lost", timeout=10000); time.sleep(1)
        st = pg.evaluate("window.__tour.gl()")
        check("the key brings the picture back and the message goes", st["draws"] and not st["fail"], str(st))
        check("no page errors (refused)", not errs, str(errs[:3])); ctx.close()
        b.close()
    print("ALL PASS" if not FAILS else f"{len(FAILS)} FAILED: {FAILS}")
    sys.exit(1 if FAILS else 0)


if __name__ == "__main__":
    main()
