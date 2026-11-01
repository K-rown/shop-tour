# shop-tour

Turn one walk recorded on an Insta360 camera into a Matterport-style walkable 3D tour. Free, open source, and it runs on your own PC.
Click or tap the floor (or a wall) and you glide to the nearest spot you recorded, keeping the direction you were looking. Works in any
desktop or phone browser, no app.

## Quick start

You need Python 3.10+, [ffmpeg](https://ffmpeg.org) on your PATH, and an Insta360 `.insv` file (tested with an Insta360 X5: two 3840x3840 lens streams in one file), or a stitched 2:1 equirectangular video exported from Insta360 Studio (the lens fit and stitching steps are then skipped).

```bash
pip install -r requirements.txt        # numpy scipy opencv-python-headless pillow transformers "pycolmap>=4.1" torch torchvision
python tour.py D:/walks/place/2026-01-01/walk.insv --serve        # builds everything, then serves http://localhost:8000 (this computer only)
# add --lan to open it on a phone on the same Wi-Fi, --title "My house" to name the tour
```

A full build of a 2.5 minute walk (23 stops) takes about 3 minutes (plus about 3 minutes the first time for finding the stops and fitting the lenses) on a PC with an NVIDIA GPU (install the CUDA build of torch from
pytorch.org) and about 15 minutes on CPU only (measured on a cold start: 25 stops, 925 s in all, the floor step alone 7 min; the model downloads add a minute or two). Re-running skips finished steps; `--from <step>` redoes a step and everything after it.
Steps: `stops calib stitch level hops map depth floor rings build`.

## Where videos and builds go

C: (or wherever the code is) holds code only. Put every video in its own folder under your walks folder, one folder per place and date, and the build goes next to it:

```
D:/walks/place/2026-01-01/VID_example.insv     <- the video
D:/walks/place/2026-01-01/frames/                             <- made by tour.py
D:/walks/place/2026-01-01/tour/                               <- the tour to publish (frames/ and the video never go public)
```

The walks folder is the `SHOP_TOUR_WALKS` environment variable (default `D:/walks` on a Windows PC with a D: drive, else `~/walks`). `python tour.py <video>` builds in the folder the video is in; `--work DIR` puts frames/ and tour/ somewhere else. A video inside the project folder is refused, with the folder to move it to. One video per folder.

Memory: the heavy steps size their parallelism from your RAM (measured peaks on a 32 GB machine: stitch 8.7 GB, level 8.2 GB, map 7 GB,
build 5 GB; with less RAM they use fewer parallel workers, e.g. an 8 GB setting peaks at ~3 GB per step and is a little slower). Set
`SHOP_TOUR_RAM_GB=8` to hold the pipeline back. Disk: about 1 GB for a 25-stop walk.

The result is the `tour/` folder: plain static files you can host anywhere. It contains your footage, so only publish it if you are
allowed to.

Sharing: `tour/` is plain static files (no server code, no CDN), so any static host works: GitHub Pages, Cloudflare Pages, Netlify, an
S3 bucket, or a USB stick plus `python -m http.server`. Size is about 7.5 MB per stop (172 MB for the 23-stop example; `--tiles off` on `build_tour.py` leaves out the 8K tiles for phones and brings it back to 4.6 MB per stop, 106 MB);
phones load only the quick-load copy (about 1 MB per stop) and desktops upgrade to the 8K copy. Hosting tips: turn on compression for html/js/json (three.js goes from 600 to about 150 KB; on 4G the first picture then appears in about 2.4 s, measured with `qa/load_time.py --gzip`), and let browsers cache `panos/`, `panos_lo/`, `panos_xs/`, `tiles/` and `depth/` for a long time: every file URL carries a `?v=` build number, so a rebuilt tour never mixes with old copies. Anyone with the link sees everything the
camera saw, so check the panoramas for people, screens, documents and addresses before you publish.

Viewer: drag or pinch to look, click/tap floor or wall to go (the address bar follows you, so `index.html#012` is a link to that spot), `N`/`P` next/previous stop, minimap in the corner. Debug flags in the URL:
`?dots ?noglide ?slowglide ?nomask ?nodepth ?nolevel ?lowres ?tiles ?notiles ?noxs`. On a phone (touch screen) the viewer keeps the 4K picture and loads only the sharp 8K tiles it is looking at (about a quarter less picture memory and under half the download of the whole 8K picture); `?tiles` forces that on a computer, `?notiles` turns it off.

## Recording tips

- The full guide, with pictures, is in [CAPTURE.md](CAPTURE.md) (and in the viewer: tap **?**, then *How to record a walk*).
- Stick vertical with the camera straight above your head (about 2.3 m), your free arm tucked in, both lenses wiped first.
- Walk, then stop for about 2 seconds every 1.5 to 2 m. The stops become the tour spots (the program finds the pauses itself).
  If it finds no pauses it automatically takes the sharpest frame in every 4 seconds; to choose the spacing yourself run `python find_stops.py D:/walks/place/2026-01-01/walk.insv --every 3` first and then `tour.py`.
- You and the stick are painted out of every panorama, and the floor under you is rebuilt from the neighbouring stops. The better you follow CAPTURE.md the cleaner that is; a stop that still shows a person is held back (see below).
- Even lighting helps: the program evens out exposure between neighbouring spots but cannot fix a very dark corner.

## How it works

1. **stops** (`find_stops.py`): measures motion in the video, finds the pauses and picks the sharpest frame near each one.
2. **calib / stitch** (`calibrate_stitch.py`, `stitch_points.py`): fits both fisheye lenses (focal length, centre, distortion, relative
   rotation) and stitches 8K equirectangular panoramas with a feathered blend over the lens overlap and a per-pano exposure match between
   the two lenses (no ffmpeg `v360` hard seam). Stops are decoded and stitched in parallel.
3. **level** (`level_panos.py`): finds each panorama's true vertical from vertical edges, because a hand-held stick leans.
4. **hops** (`hop_geometry.py`): measures the geometry between nearby stops from the photos alone (SIFT + spherical 8-point RANSAC).
5. **map** (`map_stops.py`): puts every stop on one shared map with COLMAP (native 360 camera model), then sets the metric scale.
6. **depth / floor** (`depth_panos.py`, `floor_panos.py`): Depth Anything V2 Small gives relative depth and UperNet (ADE20K) says which
   pixels are floor, on four down-tilted views per panorama. Runs on the GPU if there is one.
7. **rings** (`place_rings.py`): ties the depth to the COLMAP points, finds the floor height, and records which other stops are visible from
   each stop (so you can only go where you could see).
8. **capturer** (`remove_capturer.py`, `capturer.py`): finds the person holding the stick without a person detector: in a straight-down
   view, a compact, isolated, person-sized shape that stands more than 15 cm above the floor around it (Depth Anything V2 Small) is the seed,
   SAM 2.1 (Apache-2.0) with prompts from the nadir to that shape draws his outline, his shadow is added, and the floor underneath is copied
   from the neighbouring stops where two of them agree on it (floor plane + the known poses, colour-matched, Poisson-blended). Where no two
   stops agree it is a soft blur of the floor round the hole, never flat grey or noise. `qa/person_removal.jpg` shows before and after for every stop.
9. **privacy** (`privacy_gate.py`): fail closed. Every cleaned stop must pass the "nothing person-sized or taller than 15 cm where you stood"
   check, a person detector (RT-DETR, Apache-2.0) over the whole picture and a face detector (OpenCV YuNet, MIT). Faces are blurred; any stop
   that fails is **held back**: it is left out of the tour and listed with its picture and reason on `frames/review/review.html` (not inside
   `tour/`). `python privacy_gate.py --keep 010` records that you looked and want a held stop anyway.
10. **build** (`build_tour.py`, `exposure.py`, `viewer/index.html`): evens brightness between
   stops using the floor that neighbouring stops both see, and writes the static tour from the cleaned pictures (build refuses to run without
   the privacy result; there is no switch to skip it). The viewer wraps both photos on their depth surfaces and flies the camera between them (a short glide).

`qa/qa_tour.py` opens the built tour in a headless browser (desktop and phone sizes), walks it, and checks that it loads, has depth,
shows a cursor on the floor, moves on tap, keeps facing, glides in a sensible time, that exposure gains stay sane, and that the view
straight down is not a black blob. With torch installed it also runs `qa/check_nadir_person.py`, which segments the finished panoramas looking
straight down and fails if the person holding the stick is still visible.
Setup: `pip install -r requirements-dev.txt && python -m playwright install chromium`.
`python qa/check_capturer.py` re-checks every cleaned stop (the 4b check: nothing person-shaped or above 15 cm in the filled area, no flat or grainy patch). `python qa/test_units.py` runs 44 fast unit tests (lens model, coordinate frames, floor projection between stops, nadir mask, exposure
solver, step order, staleness, video checks, RAM-based parallelism, run lock) with no footage, GPU or browser. `qa/qa_tour.py --browser webkit|firefox` repeats the viewer checks in Safari's and Firefox's engines.

## Known limits

- If some stops cannot be placed on the map even after COLMAP's registration is loosened, `tour.py` stops with an error and the list of stops
  (usually a stretch with too little texture or too fast a move between two stops); `--allow-partial` builds the tour anyway.
- The map is as good as COLMAP: it can drift on long featureless stretches, and its metric scale varies by a few percent between runs (the plan is always drawn in the same orientation).
- Depth and floor come from single-photo models: thin objects and glossy floors can fool the floor test.
- The floor under the person is copied from neighbouring stops only where two of them agree on it; behind a bench or shelf, or in a narrow aisle, that is often nothing and the hole becomes a soft blur of the floor colour. It is smooth and never invents objects, but it is visible.
- The privacy checks fail closed on purpose: a poster of a person, or a mannequin, will hold a stop back until you look and keep it.
- Strong parallax on very close objects (ceiling ducts, people) can still show as a faint ghost at the stitch seam. `stitch_lib.stitch_seam` (cut along the best-matching line) exists but is not used: it removes the ghost but leaves brightness steps on plain walls.
- Tested with one camera (Insta360 X5) in one indoor space. The lens model is fitted per video and handles other frame sizes, but 2880x2880 and 30 fps were only emulated by re-encoding this walk, not tested on another camera.
- Long tours: COLMAP matching grows quickly with the number of stops (above 40 stops each stop is matched with the next 20; 76 stops take ~5 min for the map step).

## Credits and licences

This project is Apache-2.0 (see `LICENSE`). It uses, without bundling them:
SAM 2.1 small (Apache-2.0), RT-DETR (Apache-2.0), OpenCV YuNet face detector (MIT, shipped in `models/`), SciPy (BSD-3-Clause), COLMAP / pycolmap (BSD-3-Clause), Depth Anything V2 Small (Apache-2.0), UperNet ConvNeXt-small from openmmlab (MIT, trained on ADE20K),
Hugging Face transformers (Apache-2.0), PyTorch (BSD-style), OpenCV (Apache-2.0), NumPy (BSD), Pillow (MIT-CMU), three.js r128 (MIT, downloaded once at
build time and copied next to the viewer, so a finished tour works without internet), and FFmpeg as a separate program. Models that are not free for commercial use (for example Depth Anything V2 Base/Large,
DAP, MASt3R/DUSt3R) are deliberately not used.
