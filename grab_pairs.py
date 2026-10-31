"""Cut the raw lens pairs the lens calibration (calibrate_stitch.py) is fitted from: frames/src/f_<t>.png and b_<t>.png for each time t.

Usage:  python grab_pairs.py <video.insv> <out folder> <t1> <t2> <t3> ...

The three times are decoded beside each other and, when this ffmpeg can, on the video card (NVDEC), with the frame converted back to the video's own
yuvj420p before the PNG is made - the pictures are bit for bit the ones the software decoder gives (checked on a 3840x3840 HEVC walk: largest difference 0;
leaving that last conversion out differs by up to 32 levels). Anything that goes wrong on the card (no such decoder, another pixel format, a card that is
busy) is done again in software, the way it always was.
"""
import json, os, subprocess, sys
from concurrent.futures import ThreadPoolExecutor

PNG = ["-compression_level", "1"]                   # the pixels are the same at any level; level 1 writes the 3840x3840 pictures several times faster


def _pix_fmt(video):
    try:
        out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=pix_fmt", "-of", "json", video],
                             capture_output=True, text=True, timeout=60).stdout
        return json.loads(out)["streams"][0]["pix_fmt"]
    except Exception:
        return ""


def _has_cuda():
    try:
        return "cuda" in subprocess.run(["ffmpeg", "-hide_banner", "-hwaccels"], capture_output=True, text=True, timeout=30).stdout.split()
    except Exception:
        return False


def _cmd(video, t, f_out, b_out, hw):
    head = ["ffmpeg", "-v", "error", "-y"]
    if hw:
        head += ["-hwaccel", "cuda", "-hwaccel_output_format", "cuda"]
    vf = ["-vf", "hwdownload,format=nv12,format=yuvj420p"] if hw else []
    return head + ["-ss", f"{t:.3f}", "-i", video, "-map", "0:v:0", "-frames:v", "1", *vf, *PNG, f_out, "-map", "0:v:1", "-frames:v", "1", *vf, *PNG, b_out]


def main():
    video, out = sys.argv[1], sys.argv[2]
    times = [float(x) for x in sys.argv[3:]]
    os.makedirs(out, exist_ok=True)
    hw = _has_cuda() and _pix_fmt(video) == "yuvj420p"

    def one(t):
        f_out, b_out = os.path.join(out, f"f_{t}.png"), os.path.join(out, f"b_{t}.png")
        for use_hw in ((True, False) if hw else (False,)):
            for o in (f_out, b_out):
                if os.path.exists(o):
                    os.remove(o)
            r = subprocess.run(_cmd(video, t, f_out, b_out, use_hw), capture_output=True)
            if r.returncode == 0 and all(os.path.exists(o) and os.path.getsize(o) > 0 for o in (f_out, b_out)):
                return
        raise SystemExit(f"ffmpeg could not cut the lens pair at t={t} s of {video}: {r.stderr.decode(errors='replace')[-300:]}")

    with ThreadPoolExecutor(max_workers=3) as ex:       # (3 decodes at once is what the card takes; a raised error ends the run)
        list(ex.map(one, times))
    print(f"cut {len(times)} lens pairs into {out}", flush=True)


if __name__ == "__main__":
    main()
