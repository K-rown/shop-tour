"""Make the README demo GIF from a built tour: opens it in headless Chromium, visits some stops, hovers the floor so the cursor shows,
and saves an animated GIF.   python qa/make_demo_gif.py [--tour tour] [--stops 8,10,12,20] [--out docs/demo.gif]
Only use a tour built from a walk you are allowed to publish (docs/demo.gif is git-ignored for that reason)."""
import argparse, io, os, sys
from PIL import Image

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "qa"))
from flash_test import serve


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tour", default="tour"); ap.add_argument("--stops", default="8,10,12,20")
    ap.add_argument("--out", default=os.path.join("docs", "demo.gif")); ap.add_argument("--width", type=int, default=960)
    a = ap.parse_args()
    from playwright.sync_api import sync_playwright
    url = serve(os.path.join(HERE, a.tour) if not os.path.isabs(a.tour) else a.tour)
    frames = []
    with sync_playwright() as p:
        b = p.chromium.launch(args=["--ignore-gpu-blocklist", "--enable-webgl", "--enable-unsafe-swiftshader"])
        pg = b.new_page(viewport={"width": 1280, "height": 800}); pg.goto(url)
        pg.wait_for_function("window.__tour && !window.__tour.state().busy", timeout=60000)
        pg.evaluate("document.getElementById('help') && document.getElementById('help').classList.remove('show')")
        n = pg.evaluate("window.__tour.n")
        for s in [int(x) - 1 for x in a.stops.split(",") if 0 < int(x) <= n]:
            pg.evaluate(f"window.__tour.go({s})"); pg.wait_for_function(f"window.__tour.state().cur === {s} && !window.__tour.state().busy", timeout=30000)
            pg.wait_for_timeout(900)
            pg.evaluate("window.__tour.aim(window.__tour.state().lon, -12)")
            fp = pg.evaluate("window.__tour.floorPoint()")
            if fp:
                pg.mouse.move(fp[0], fp[1]); pg.wait_for_timeout(400)
            im = Image.open(io.BytesIO(pg.screenshot())).convert("RGB")
            frames.append(im.resize((a.width, int(im.height * a.width / im.width)), Image.LANCZOS).quantize(colors=128, dither=Image.Dither.NONE))
        b.close()
    if not frames:
        sys.exit("no frames")
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    frames[0].save(a.out, save_all=True, append_images=frames[1:], duration=1400, loop=0, optimize=True)
    print(f"wrote {a.out}: {len(frames)} frames, {os.path.getsize(a.out) // 1024} KB")


if __name__ == "__main__":
    main()
