"""Record a short demo video of a built tour: opens it in headless Chromium, looks around, hovers the floor so the cursor shows, and glides
through a few stops. Draft quality (headless software rendering is slower than a real GPU): for the final video record your screen with a real browser.
    python qa/make_demo_video.py [--tour tour] [--stops 3,8,12,20] [--out docs/demo.mp4]
Only use a tour built from a walk you are allowed to publish (docs/ is git-ignored for demo.gif; keep demo.mp4 out of git too)."""
import argparse, glob, os, shutil, subprocess, sys, tempfile

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "qa"))
from flash_test import serve


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tour", default="tour"); ap.add_argument("--stops", default="3,8,12,20"); ap.add_argument("--out", default=os.path.join("docs", "demo.mp4"))
    a = ap.parse_args()
    from playwright.sync_api import sync_playwright
    url = serve(os.path.join(HERE, a.tour) if not os.path.isabs(a.tour) else a.tour).replace("?noglide", "")
    tmp = tempfile.mkdtemp()
    with sync_playwright() as p:
        b = p.chromium.launch(args=["--ignore-gpu-blocklist", "--enable-webgl", "--enable-unsafe-swiftshader"])
        ctx = b.new_context(viewport={"width": 1280, "height": 720}, record_video_dir=tmp, record_video_size={"width": 1280, "height": 720})
        pg = ctx.new_page(); pg.goto(url)
        pg.wait_for_function("window.__tour && !window.__tour.state().busy", timeout=60000)
        pg.evaluate("document.getElementById('help') && document.getElementById('help').classList.remove('show')"); pg.wait_for_timeout(1500)
        n = pg.evaluate("window.__tour.n")
        for s in [int(x) - 1 for x in a.stops.split(",") if 0 < int(x) <= n]:
            pg.evaluate(f"window.__tour.go({s})"); pg.wait_for_function(f"window.__tour.state().cur === {s} && !window.__tour.state().busy", timeout=30000)
            pg.wait_for_timeout(800)
            lon = pg.evaluate("window.__tour.state().lon")
            for d in range(0, 60, 6):                          # a slow pan
                pg.evaluate(f"window.__tour.aim({lon + d}, -12)"); pg.wait_for_timeout(60)
            fp = pg.evaluate("window.__tour.floorPoint()")
            if fp:
                pg.mouse.move(fp[0] - 60, fp[1]); pg.wait_for_timeout(300)
                pg.mouse.move(fp[0], fp[1], steps=12); pg.wait_for_timeout(900)
                pg.mouse.click(fp[0], fp[1]); pg.wait_for_function("!window.__tour.state().busy", timeout=30000); pg.wait_for_timeout(1200)
        ctx.close(); b.close()
    webm = glob.glob(os.path.join(tmp, "*.webm"))[0]
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", webm, "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "23", a.out], check=True)
    shutil.rmtree(tmp, ignore_errors=True)
    print(f"wrote {a.out}: {os.path.getsize(a.out) // 1024} KB")


if __name__ == "__main__":
    main()
