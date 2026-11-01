"""Do black holes appear during the glide? Slows the glide down (?slowglide), taps the floor from many stops, grabs frames while it runs
and reports the share of near-black pixels in the middle of the view (the depth meshes can leave gaps).   python qa/glide_holes.py [tour] [N stops]"""
import os, sys, time
import cv2, numpy as np
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "qa"))
from flash_test import serve


def main():
    folder = sys.argv[1] if len(sys.argv) > 1 else "tour"
    url = serve(os.path.join(HERE, folder) if not os.path.isabs(folder) else folder).replace("?noglide", "?slowglide")
    from playwright.sync_api import sync_playwright
    worst, rows = [], []
    with sync_playwright() as p:
        b = p.chromium.launch(args=["--ignore-gpu-blocklist", "--enable-webgl", "--enable-unsafe-swiftshader"])
        pg = b.new_page(viewport={"width": 960, "height": 600}); pg.goto(url)
        pg.wait_for_function("window.__tour && !window.__tour.state().busy", timeout=60000)
        pg.evaluate("document.getElementById('help') && document.getElementById('help').classList.remove('show')")
        n = pg.evaluate("window.__tour.n"); stops = range(0, n, max(1, n // int(sys.argv[2] if len(sys.argv) > 2 else 8)))
        for i in stops:
            pg.evaluate(f"window.__tour.go({i})"); pg.wait_for_function(f"window.__tour.state().cur === {i} && !window.__tour.state().busy", timeout=60000)
            pg.wait_for_timeout(1500); pg.evaluate("window.__tour.aim(window.__tour.state().lon, -15)")
            fp = pg.evaluate("window.__tour.floorPoint()")
            if not fp: continue
            pg.mouse.click(fp[0], fp[1]); fr = []; t0 = time.time()
            while time.time() - t0 < 20:
                g = cv2.cvtColor(cv2.imdecode(np.frombuffer(pg.screenshot(), np.uint8), 1), cv2.COLOR_BGR2GRAY)
                h, w = g.shape; mid = g[int(h * 0.1):int(h * 0.8), int(w * 0.05):int(w * 0.95)]
                fr.append(float((mid < 12).mean()) * 100)
                if not pg.evaluate("window.__tour.state().busy") and len(fr) > 2: break
            rows.append((i + 1, len(fr), max(fr), float(np.mean(fr))))
            print(f"stop {i + 1:02d}: {len(fr)} frames, near-black share: max {max(fr):.2f}%  mean {np.mean(fr):.2f}%", flush=True)
        b.close()
    a = np.array([r[2] for r in rows]); print(f"ALL glides: {len(rows)}; worst frame {a.max():.2f}% near-black, median of maxima {np.median(a):.2f}%")


if __name__ == "__main__":
    main()
