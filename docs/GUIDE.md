# shop-tour: the full guide

Everything about the engine: what it needs, installing by hand, the one command, what comes out, the privacy gate, the speed switches, the manifest,
how each step works, and the known limits. The short version is [README.md](../README.md).

## The one-click scripts

- `setup.bat` (Windows, double-click) / `bash setup.sh` (Linux, macOS): finds Python 3.11 to 3.13, makes `.venv/` in the folder, installs PyTorch at the
  pinned versions (the CUDA 13.0 build when `nvidia-smi` reports an NVIDIA driver of 580 or newer, else the CPU build), installs `requirements.txt`, checks
  `git`, `ffmpeg` and `ffprobe` (on Windows it installs a missing one with `winget install Gyan.FFmpeg` / `Git.Git`), and runs `tools/fetch_pii_models.py`.
  Running it again skips what is done (`tools/setup_help.py` makes those checks). `--dry-run` prints what it would do and changes nothing.
- `start.bat` / `bash start.sh`: runs `app.py` from `.venv` and opens `http://127.0.0.1:8765` (this computer only; `--port N` for another port,
  `--no-browser` not to open one). The page copies the chosen video into the walks folder (`SHOP_TOUR_WALKS`, see "Where videos and builds go") as
  `<walks>/<place>/<date>/<file>`, runs `tour.py <video> --title <title>` there (its output also goes to `build.log` beside the video), shows the step it
  is on, asks you about every stop the privacy gate held back (it runs `privacy_gate.py --keep` / `--leave-out` with your answers, then builds again), and
  serves the finished tour at `/t/<id>/`. One build at a time; closing the page does not stop a build, the Stop key does.

## What you need

| | |
|---|---|
| Camera | an Insta360 `.insv` file (tested with an X5: two 3840x3840 lens streams in one file). A stitched 2:1 equirectangular video (an Insta360 Studio export) also works, but then the camera's motion sensors are not read: no scale from the accelerometer, levelling by picture only |
| GPU | an NVIDIA card with CUDA, **driver R580 or newer** (the pinned PyTorch is the CUDA 13.0 build, which needs it). Reference machine: RTX 3070 Ti 8 GB. A 25-stop walk takes about 9 minutes with it. Without a GPU it still runs, measured once at about 93 minutes |
| OS | Windows 10/11 (the reference machine) or Linux x86-64 with glibc 2.35 or newer (Ubuntu 22.04+; Open3D's wheels need it). Linux runs the unit tests in CI; a full Linux build has not been timed. macOS is not supported (no CUDA) |
| Python | 3.11 to 3.13 (3.13 on the reference machine; 3.10 works for the unit tests but some packages have no 3.10 wheel any more) |
| Programs | `ffmpeg` and `ffprobe` on your PATH, and `git` (pip fetches MoGe from GitHub). COLMAP comes inside the `pycolmap` wheel |
| RAM / disk | 16 GB or more recommended (measured on 32 GB, peaks of 5 to 9 GB a step; `SHOP_TOUR_RAM_GB=8` holds a step to about 3 GB). About 1 GB of disk per 25-stop walk, plus about 2.5 GB of models downloaded on the first build and 165 MB from `tools/fetch_pii_models.py` |
| Network | only for the first model downloads (Hugging Face, GitHub) and three.js (cdnjs) once. After that a build and a finished tour need no internet |

## Install on Windows (once)

```powershell
# 1. Python 3.13 from python.org (tick "Add to PATH"), Git from git-scm.com, and FFmpeg:
winget install Gyan.FFmpeg            # or put ffmpeg.exe and ffprobe.exe from ffmpeg.org on your PATH; reopen the terminal after
# 2. the code
git clone https://github.com/<owner>/shop-tour.git
cd shop-tour
python -m venv .venv
.venv\Scripts\activate
# 3. PyTorch's CUDA 13.0 build first, at the pinned versions (skip this line for a CPU-only PC)
pip install torch==2.14.1 torchvision==0.29.1 --index-url https://download.pytorch.org/whl/cu130
# 4. everything else, and the two blur models (about 165 MB, checked by sha256)
pip install -r requirements.txt
python tools/fetch_pii_models.py
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"     # expect 2.14.1+cu130 True
```

## Install on Linux (once)

```bash
sudo apt install python3 python3-venv git ffmpeg        # Ubuntu 22.04+; any distro with Python 3.11-3.13 and glibc 2.35+
git clone https://github.com/<owner>/shop-tour.git
cd shop-tour
python3 -m venv .venv && . .venv/bin/activate
pip install torch==2.14.1 torchvision==0.29.1 --index-url https://download.pytorch.org/whl/cu130   # skip on a CPU-only machine
pip install -r requirements.txt
python tools/fetch_pii_models.py
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

`python tour.py <video>` checks ffmpeg, ffprobe and every package in its first second and prints the `pip install` line for anything missing.
Without `tools/fetch_pii_models.py` faces are still blurred but plates and printed text are not, and the log says so.

## The one command

Record first (10 minutes for a small place; the pictures are in [CAPTURE.md](../CAPTURE.md)): stick held straight up over your head, free arm tucked in,
lenses wiped; walk, and hold still for 2 seconds every 1.5 to 2 metres. Put the video in a folder of its own, outside the code folder
(see "Where videos and builds go"), then:

```bash
python tour.py D:/walks/myhouse/2026-11-02/VID_example.insv --serve --title "My house"
```

It builds everything and serves the tour at http://localhost:8000 (this computer only; add `--lan` to open it on a phone on the same Wi-Fi). Re-running
skips finished steps; `--from <step>` redoes a step and everything after it, `--only <step>` just one, `--work DIR` builds somewhere else, and
`--allow-partial` builds even if some stops could not be placed on the map. `python tour.py --help` lists every step.

## What comes out

Next to the video:

- `tour/` — **the tour**: plain static files (`index.html`, the panoramas at three sizes and 8K tiles for phones, depth, `dollhouse.bin`, `plan.png` and
  `plan.svg`, `manifest.json`). Host it anywhere static (GitHub Pages, Netlify, an S3 bucket, `python -m http.server`). About 7.5 MB per stop.
- `frames/` — the working files: the stops, the stitched panoramas before cleaning, the map, depth, and `frames/privacy.json` (what the privacy gate
  found and decided, per stop). **It contains the raw pictures: never publish `frames/` or the video.**
- `tour/manifest.json` — the stable summary an app reads (version, stops, privacy status, what was blurred, whether Measure is shown). See below.

## Privacy gate

A tour of a real place shows people and their things, so privacy is part of the build and cannot be skipped (the `build` step refuses to run without its
result). What happens to people depends on `SHOP_TOUR_PEOPLE`:

| `SHOP_TOUR_PEOPLE` | Other people in the picture | Stops held back for review |
|---|---|---|
| `keep` **(the default)** | **left as photographed**; only their faces are blurred | none |
| `blur` | smudged where they stand (a soft blur inside their mask) | none |
| `remove` | painted out (filled from neighbouring stops or patch synthesis); a stop where a person is still found is **held** | yes — you decide each one |

In every mode:

- **Faces, licence plates, printed names and other text are blurred** in every picture that ships (`SHOP_TOUR_PII_BLUR=0` turns it off). The blur is in
  the picture's own pixels and cannot be undone. It is never a gate: a stop whose blur fails ships as it was, and `frames/pii.json` and
  `tour/manifest.json` say so.
- **The person holding the stick** is smudged out (painted out and the floor rebuilt under them in `remove` mode).
- `tour/manifest.json` says `privacy.status`: `all_clear`, or `held` while a held stop waits for you (only in `remove` mode). A program that imports a
  tour should refuse `held`.

In `remove` mode a held stop is listed with its picture and the reason in `frames/review/review.html` (for you only). Answer from the walk's folder:
`python <code folder>/privacy_gate.py --frames frames --keep 010` puts stop 010 in (faces blurred); `--leave-out 002 005` keeps those out. Nothing is
decided for you.

**This is a safety net, not a guarantee.** The detectors can miss a far, small or dark face, plate or sign, and with the default `keep` everyone's body,
clothes and surroundings stay in the picture. Look at every panorama in `tour/` before you publish a tour of anywhere that is not yours, and do not
publish people who have not agreed. Everything runs on your computer; nothing is uploaded.

## Known limits, said plainly

- One camera was tested (Insta360 X5) in a handful of places. Other Insta360 models are untested; the motion-sensor format was only worked out for the X5.
- The scale comes from the camera's accelerometer and needs a walk with pauses; when that fit fails it falls back to an assumed stick height. Measure is
  not tape-measure accurate (median error about 9 % on the best test walk, 27 % on a bad one), so it is hidden unless this walk was graded under 10 %.
- The privacy verdicts were measured on one install; a different PyTorch or CUDA build moves the detectors' scores a little (one or two stops of 25
  in our runs). That is why torch is pinned.
- The full pipeline has been timed on Windows with an RTX 3070 Ti only. Linux, smaller cards and CPU-only builds are expected to work and are slower.
- The floor model (UperNet) and the face model (YuNet) have MIT weights but were trained on research-only datasets (ADE20K, WIDER FACE); see
  [THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md) if you sell what you build.
- More in "Known limits" at the end.

---

The rest of this file is the detail: where files go, the speed switches, the manifest, and how each step works.

How long: a full build of the 2.5 minute, 25-stop test walk takes about 9 minutes on a PC with an NVIDIA GPU (measured twice: 530 s and 534 s on an RTX 3070 Ti with 12 cores and a quiet card; the steps run as a dependency graph and some begin early; `SHOP_TOUR_SERIAL=1` runs them one after another), about 93 minutes on the CPU alone (measured once, `CUDA_VISIBLE_DEVICES=-1`: the floor model and the person removal are the slow parts), and a 42-stop walk 12 minutes with the GPU (710 s). Another process holding half the card's memory can slow the person removal many times over (it waits on the card, it does not fail).
Steps: `stops calib stitch vio level hops map depth floor rings capturer privacy pii early sphere register mvs align build dollhouse plan` (`tour.py --help` has the one-line meaning of each).

What you get: click or tap the floor or a wall to glide to the nearest spot; a Dollhouse of the whole place from above (coloured from your pictures); a floor plan; pinned notes (press and hold anywhere: text and a photo, kept next to the tour); and a Measure tool that is **hidden unless its measured error is under 10 %**.

Licences of every package and model: [THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md).

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

Speed switches (every one gives the same result as the plain way; set it to `0` to turn it off): `SHOP_TOUR_EARLY` (steps begin as soon as the part of them that comes
first can: vio with the lens fit, the stops' map with the hops, the privacy chain with the levelling; they wait for the rest themselves), `SHOP_TOUR_GPU_MAPS` and
`SHOP_TOUR_GPU_MORPH` (the capturer's pixel maps and its big erosions and dilations are made on the video card), `SHOP_TOUR_MASK_THREADS` (masks made at once, default 2),
`SHOP_TOUR_TSDF_TIGHT` (the dollhouse fuses only the voxels of each face's frustum), `SHOP_TOUR_VIO_OVERLAP` (the map's features are made while its pictures are still being
decoded) and `SHOP_TOUR_IMU_THREADS` (the accelerometer fit's independent fits side by side; 1 = one after the other).

The result is the `tour/` folder: plain static files you can host anywhere. It contains your footage, so only publish it if you are
allowed to.

Sharing: `tour/` is plain static files (no server code, no CDN), so any static host works: GitHub Pages, Cloudflare Pages, Netlify, an
S3 bucket, or a USB stick plus `python -m http.server`. Size is about 7.5 MB per stop (172 MB for the 23-stop example; `--tiles off` on `build_tour.py` leaves out the 8K tiles for phones and brings it back to 4.6 MB per stop, 106 MB);
phones load only the quick-load copy (about 1 MB per stop) and desktops upgrade to the 8K copy. Hosting tips: turn on compression for html/js/json (three.js goes from 600 to about 150 KB; on 4G the first picture then appears in about 2.4 s, measured with `qa/load_time.py --gzip`), and let browsers cache `panos/`, `panos_lo/`, `panos_xs/`, `tiles/` and `depth/` for a long time: every file URL carries a `?v=` build number, so a rebuilt tour never mixes with old copies. Anyone with the link sees everything the
camera saw, so check the panoramas for people, screens, documents and addresses before you publish.

Viewer: drag or pinch to look, click/tap floor or wall to go (the address bar follows you, so `index.html#012` is a link to that spot), `N`/`P` next/previous stop, minimap in the corner. Opened on its own (not inside another page's frame), the viewer shows a Back key when it was reached from the same site and a Share key where the browser has a share sheet; two optional tour.json fields are shown when present: `when` (a short line such as "Today 9:41", put before the stop number) and `walks` (`[{"label": "9:41", "href": "...", "on": true}, ...]`, two or more walks of the same place, drawn as pills under the title). The build does not write either yet. Debug flags in the URL:
`?roof` (the dollhouse with its roof on) `?dots ?noglide ?slowglide ?nomask ?nodepth ?nolevel ?lowres ?tiles ?notiles ?noxs`. On a phone (touch screen) the viewer keeps the 4K picture and loads only the sharp 8K tiles it is looking at (about a quarter less picture memory and under half the download of the whole 8K picture); `?tiles` forces that on a computer, `?notiles` turns it off.

## manifest.json (the stable hand-off)

Every build ends by writing `tour/manifest.json` (`python manifest.py` rewrites it by hand). Apps that open or import a tour read this file instead of guessing from the pictures. `manifest_version` is 1; a new version number is the only way fields are renamed or removed.

```json
{
 "manifest_version": 1,
 "engine": {"name": "shop-tour", "version": "1.0", "commit": "517efeb"},
 "created": "2026-10-18T14:03:11Z",
 "walked": "2026-10-16",
 "title": "My house",
 "stops": {"count": 17, "ids": ["001", "003", "..."]},
 "size_bytes": 122000000,
 "privacy": {"status": "held", "checked": 23, "held": [{"id": "002", "reasons": ["..."]}], "kept": [], "left_out": []},
 "measure": {"status": "shown", "error_pct": 7.9, "max_pct": 10.0},
 "pii": {"status": "ok", "blurred": {"plate": 3, "face": 1, "name": 0, "text": 2}, "stops_blurred": 3, "errors": []},
 "features": {"dollhouse": true, "plan": true, "notes": false}
}
```

- `walked` is the day the video was recorded (from the camera's file name), or `null`.
- `stops.ids` are the stops that are in the tour, in tour order. Held stops are **never** in `tour/`.
- `privacy.people_mode` (`blur` or `keep`) and `privacy.people_blurred` are present only when `SHOP_TOUR_PEOPLE` is not `remove`: then nothing is held and `privacy.status` is `all_clear`.
- `privacy.status` is `all_clear` when no stop is held, or every held stop has been looked at: `python privacy_gate.py --keep 010` (it goes into the tour, faces blurred; listed in `kept`) or `--leave-out 002 005` (it stays out; listed in `left_out`). It is `held` while any held stop is waiting (listed in `held` with the reasons) or when the privacy check has not run. Review pictures are in `frames/review/review.html`, never in `tour/`. A program that imports tours should refuse `held` and show these ids.
- `pii` is what the blur step (step 9b) did to the pictures in the tour: how many plates, faces and printed names were blurred, in how many stops, and the ids of stops it could not check (those ship as they were; `status` is then `partial`). `status` is `off` when `SHOP_TOUR_PII_BLUR=0` and `not_run` when the step has not run.
- `measure.status` is `shown` (with `error_pct`, the median error against tape) or `hidden` when that error is over `max_pct` or unknown.
- `size_bytes` is the size of the tour folder before the manifest itself.

## Recording tips

- The full guide, with pictures, is in [CAPTURE.md](../CAPTURE.md) (and in the viewer: tap **?**, then *How to record a walk*).
- Stick vertical with the camera straight above your head (about 2.3 m), your free arm tucked in, both lenses wiped first.
- Walk, then stop for about 2 seconds every 1.5 to 2 m. The stops become the tour spots (the program finds the pauses itself).
  If it finds no pauses it automatically takes the sharpest frame in every 4 seconds; to choose the spacing yourself run `python find_stops.py D:/walks/place/2026-01-01/walk.insv --every 3` first and then `tour.py`.
- You and the stick are smudged out of every panorama (painted out, with the floor under you rebuilt from the neighbouring stops, in `remove` mode). The better you follow CAPTURE.md the cleaner that is.
- Even lighting helps: the program evens out exposure between neighbouring spots but cannot fix a very dark corner.

## How it works

1. **stops** (`find_stops.py`): measures motion in the video, finds the pauses and picks the sharpest frame near each one.
2. **calib / stitch** (`calibrate_stitch.py`, `stitch_points.py`): fits both fisheye lenses (focal length, centre, distortion, relative
   rotation) and stitches 8K equirectangular panoramas with a feathered blend over the lens overlap and a per-pano exposure match between
   the two lenses (no ffmpeg `v360` hard seam). Stops are decoded and stitched in parallel.
3. **level** (`level_panos.py`): finds each panorama's true vertical, because a hand-held stick leans - three readings, the same everywhere: the ground's own vertical (the depth model's surface normals on the ground round the camera), the horizon where the sky meets flat ground or water, and vertical edges; they are combined by weight and a reading far from the ground's is dropped. `frames/level.json` says how each stop was read.
4. **hops** (`hop_geometry.py`): measures the geometry between nearby stops from the photos alone (SIFT + spherical 8-point RANSAC).
5. **vio / map** (`vio_map.py`, `map_stops.py`, `imu.py`, `imu_scale.py`): `vio` puts a small picture every second between the stops on a COLMAP map, so the
   camera's path is known once a second. `map` puts every stop on one shared map (native 360 camera model) and sets the **metric scale with no user input**: the camera
   records its own accelerometer and gyroscope inside the .insv; the scale is the one number that makes the accelerometer's acceleration (gravity taken out with the
   pictures' orientation) agree with the second difference of the path (`imu_scale.py` has the maths). If that fit is poor (R-squared below 0.3, a wide bootstrap, or a
   path that disagrees with the stop map) it says so and falls back to the old camera-height assumption. The result and its quality are in `frames/poses.json` (`scale_info`).
6. **depth / floor** (`depth_panos.py`, `floor_panos.py`): Depth Anything V2 Small gives relative depth and UperNet (ADE20K) says which
   pixels are floor, on four down-tilted views per panorama. Runs on the GPU if there is one.
7. **rings** (`place_rings.py`): ties the depth to the COLMAP points, finds the floor height, and records which other stops are visible from
   each stop (so you can only go where you could see).
7a. **sphere** (`depth_sphere.py`): one metric, full-sphere depth per stop from MoGe-2 (MIT) on 18 overlapping views (a ring of 8 at the horizon, 4 up, 4 down, straight up,
   straight down): each view's own scale is solved from the overlaps between views, so the roof and the upper walls have depth too (the four down-tilted views of step 6
   never saw them). The metre is MoGe's own: across a walk it is steadier than the sparse map's local scale (on the test shop the roof came out at 4.4 m from every
   stop, tape 4.36 m, while the map's metre against it wandered by 2x along the walk), and the stops are tied together by the one thing that does not change on a
   walk, the height of the stick (each stop's floor is pulled onto the walk's median). The floor seen by the depth is snapped onto the floor plane. `frames/depth_sphere/`.
7c. **register** (`register_stops.py`): every stop's depth as a point cloud in metres, each stop registered to its nearest stops (point-to-plane ICP at three scales)
   and one pose graph over the walk moves every stop's x, y and heading to where its depth agrees with its neighbours' (the sparse map placed stops tens of
   centimetres off, and a wall seen from two stops 20 cm apart is a torn wall in the model). `frames/poses.json` is updated; the map's own is kept in `frames/poses_sparse.json`.
7b. **mvs / align** (`mvs.py`, `depth_align.py`, optional, **off by default**): a spherical plane-sweep stereo on the GPU between each stop and its eight nearest neighbours
   (COLMAP's own patch-match needs a CUDA build that the pip wheel does not have) gives ranges where at least two other stops agree, and the single-photo depth is multiplied by a
   smooth field so that it matches them. On the two test walks it did not beat the depth model alone (and made a walk with poor poses clearly worse), so it only runs with
   `SHOP_TOUR_STEREO=1` (or `mvs.py --force`) and only when the accelerometer has confirmed the poses; otherwise the steps skip in a second. Held-back stops are never read.
   The camera height (and with it the floor plane and every height in the model) comes from the dense 1 Hz map: the floor is where its point cloud starts, because the depth
   model's floor layer was a layer of clutter half a metre above the real floor on the first test walk and made everything 25-35 % too short.
Steps 8 and 9 below describe the full check, which runs in every mode; holding a stop back and painting people out happen only with
`SHOP_TOUR_PEOPLE=remove` (see "Privacy gate" above).

8. **capturer** (`remove_capturer.py`, `capturer.py`; run by `privacy_chain.py` in ONE process with the privacy check, models loaded once, each stop checked as its picture lands): finds the person holding the stick in the straight-down
   view with a person detector (RT-DETR, Apache-2.0) on several fields of view and SAM 2.1 (Apache-2.0) prompted with his box (a compact,
   isolated, person-sized shape standing above the floor, from Depth Anything V2 Small, confirms and extends it), adds his shadow, and copies the floor underneath
   from the neighbouring stops where two of them agree on it (floor plane + the known poses, colour-matched, Poisson-blended); where only one neighbour lines up with
   the stop's own floor ring, that one supplies it; what nobody saw gets a soft fill with the floor's own grain, never flat grey. `qa/person_removal.jpg` shows before and after for every stop.
   Nothing here reads the map or the 1 Hz path (`privacy_inputs.py`, mode "own"), so a change to the mapping cannot change a privacy verdict; `SHOP_TOUR_PRIVACY_INPUTS=map` is the old way.
9. **privacy** (`privacy_gate.py`, `privacy_margin.py`): fail closed. Every cleaned stop must pass the "nothing person-sized or taller than 15 cm where you stood"
   check, a person detector (RT-DETR) over twelve views of the whole picture, a search of the straight-down view for a camera holder left under the camera, and a face detector (OpenCV YuNet, MIT). Faces are blurred; any stop
   that fails is **held back**: it is left out of the tour and listed with its picture and reason on `frames/review/review.html` (not inside
   `tour/`). `python privacy_gate.py --keep 010` records that you looked and want a held stop anyway. The margin holds a stop that would be held with the camera height measured a few percent differently.
   What is NOT a reason to hold (each measured against hand labels of every stop of two walks, each with a switch to turn it off): a person-sized shape that is only the soft fill where you stood;
   a shape the neighbouring stops also see at that place on the floor in their own pictures (`neighbour_check.py` — you are never in two stops' pictures at one spot; `SHOP_TOUR_NEIGHBOUR_CHECK=0`);
   a small face with no person near it (it is blurred, not held). What IS: a person seen by two views at the horizon with a score of 0.40 or more (`SHOP_TOUR_HORIZON_PERSONS=0`) —
   **on a real road walk a flagger standing 30–40 m off once scored 0.49 against the detector's 0.5 line and his stop passed**; that rule is what catches him now, at the cost of about one extra hold per 60 stops indoors.
   `python qa/judge_labels.py --frames <frames> --labels <labels.json> --walk <name> --must-hold <dir>:<ids>` prints a run's verdicts against labels you made by eye.
   **Other people (ON by default since 2026-10-22; `SHOP_TOUR_REMOVE_OTHERS=0` turns it off and the gate then holds every stop another person is in)**:
   the capturer step also paints out every OTHER person (and face, a poster's too) the gate's scan finds (`remove_others.py`: SAM 2 on the detector's box, the hole
   filled from the neighbouring stops' pictures, else from the picture round it) and the gate then looks at the painted picture again: a stop where a person or face
   is still found stays held. The gate then also looks at the horizon band in 30 degree views (a crew 40 m off, a driver behind a windscreen) at the bar `SHOP_TOUR_FINE_SCORE`
   (0.40 with the paint-out on). Changing the switch (or deleting `remove_others.py`'s output) does not re-run a finished capturer step: delete `frames/clean` (or touch `remove_others.py`) to redo it.
   What no neighbouring stop saw (a far person, the road behind a crew) is made by patch synthesis from the picture round it (`synth_fill.py`: nearest patches, matched on the smoothed
   picture and copied with their grain, from places near the hole and on its own horizontal strata, the colour at the edge carried in), not a smooth smudge or one copied piece; a neighbour
   is looked at only where it lines up (ncc 0.65) and never for a person smaller than 8 degrees across. And where the painted picture still holds a person-sized lump under the camera
   (the gate's own check C, at the margin's camera heights), the capturer's hole is made again the same way, up to three ways, the first that leaves no lump wins; none wins: the stop is held as before.
   `frames/capturer/others.json` says per stop which fill won (`refill`) and how long each person took.
9b. **pii** (`pii_blur.py`, ON by default; `SHOP_TOUR_PII_BLUR=0` turns it off): after the privacy step, every picture that will be in the tour (`frames/safe/`, which the tour, the dollhouse and the depth are all made from) has what identifies a
   person beyond their body blurred: **faces** (OpenCV YuNet, MIT: a face in the closer look at any person's head is blurred at ANY score the detector gives; a face seen only in the 26 wider views needs 0.6 AND a person or the upper half of a vehicle (a driver) under it, or 0.88 on its own (a poster's face) - measured by eye, not by feel: a head zoom only ever shows a person, while a shop's shelves and fans score 0.6-0.81 as faces),
   **licence plates**, **printed names** and **other text**. Two models beyond the privacy gate's own, both loaded BY PATH from `models/` with a sha256 check (a run never fetches; `python tools/fetch_pii_models.py` downloads them once
   and prints each file's sha256 and licence; if they are missing the step still blurs faces and says what it did not look for in the log and `pii.json`):
   - a scene text detector, PaddleOCR PP-OCRv3 DBNet (Apache-2.0, 2.4 MB ONNX, from OpenCV's model zoo), run by OpenCV's own DB reader (no PaddlePaddle, no onnxruntime), over ten tiles of the sphere at the picture's own resolution;
     a box must score 0.5, be at least 9 pixels tall and pass `texty()` (the detector fires on foliage and wires: letters are three or more glyph-like pieces of one height standing on one line, dark on light or light on dark),
     or score 0.9 and pass `texty_lite()` (wide, under a fifth leaf-coloured, strong contrast, two tall pieces: multi-line lettering, white on red, a sign of two big letters; the score alone is no guide - foliage scores 0.97);
   - a plate detector, `justjuu/rtdetr-v2-license-plate-detection` (RT-DETR v2 fine-tuned on plates; the model card declares Apache-2.0, base `PekingU/rtdetr_v2_r50vd` Apache-2.0; the card does not state its dataset's licence),
     over a view cut round every vehicle the person detector's network (RT-DETR, Apache-2.0: car, bus, truck, motorcycle) finds, and over the twelve views. A plate blur needs the model at 0.7 on its own, or at 0.45 with the text detector
     boxing the same place, or at 0.45 low on the vehicle (lower 70 % of its box), narrow beside it and plate-shaped; a plate is 1.3-6 times wider than tall and at most a fifth of its vehicle's height (a "plate" the size of the vehicle
     is the vehicle); with no vehicle found at all the model must be 0.8 sure and the box small. 164 MB, not in git.
   Every text region becomes a blur: on a vehicle, plate-shaped and low on it, a `plate`; other lettering on a vehicle (a door's company name) or on a person (a name tag, a badge) a `name`; anywhere else (a street address or a phone
   number on a van, an ad, a sign) `text`. A text box on a plate the plate model found is one blur, the plate's. Ultralytics YOLO and anything AGPL are not used.
   MEASURED (2026-10-22, second pass, `<output folder>/pii3`, every blur looked at beside the original on contact sheets with the region drawn): road walk, plate blurs 45 -> 26, of which on a real plate 25 -> 22 and not
   (a sign, a headlight, an amber light, a tailgate) 20 -> 4 (the three plates lost are far, tiny ones); pole walk 26 -> 9, real 10 -> 5 of 6 and the rest a lift's unit number, a street sign's phone number, a rental's lettering; text/name boxes
   on the road walk 23 -> 35, real lettering 15 -> 21 (a bus shelter's street name in every stop, a billboard's name, phone number and web address, a route sign) and not-lettering 8 -> 14 (foliage and a pole's gear); wall time unchanged,
   about 11-12 s a stop with the models loaded once. FACES: the raw stops of the road walk (before the paint-out) hold 45 real faces at 0.3-0.93 (the walker from below, the crew, a billboard): at the old 0.75 bar 21 were found and about 24 missed;
   38 of the 45 are in a head zoom, which shows only persons and held up at any score, so that is where the bar went. In the PAINTED pictures the tour ships (`frames/safe`) no real face is left at 0.5 or over on either walk (the
   crew and the walker are gone): the faces the step finds there are wheels, a hatch, foliage and one person in the dark. Earlier numbers: road walk 45 plates, 4 names and 19 other text regions blurred, 12 s a stop on the card
   (the heuristic it replaces blurred 7 plates, about 3 real, and missed a legible one); about two thirds of the first 30 plate blurs were on a plate (the first plate of each pair of a bumper or a van, a dealer's frame, a plate at
   45 degrees, a dark car's), the rest on a grille, a headlight, a wheel, a tailgate's "4x4" or a road sign; the pole walk 26 plates (about 10 on real plates, the rest on whole distant vehicles, whose plates are too small to read anyway),
   9 text regions (a union's lettering, a street number). The blur is generous and some plates too small or too dark are still missed; it is a stronger stop-gap, not a guarantee.
   A region is blurred in the picture's own pixels, its box grown by half again: pixelated in cells of at least 8 pixels and at most a quarter of the region across (so even a close plate is four cells wide), then Gaussian-blurred at the
   cell size, under a soft edge. It cannot be undone from the picture. It is never a gate: a stop whose detection or blur fails ships as it was, with the failure recorded, and the step always ends well.
   It writes `frames/pii.json` (per stop: counts by kind, and the regions as ANGLES, so they apply to any other copy of the stop at any size: `python pii_blur.py --apply-dir <folder> [--out <folder>]` blurs a folder of equirects named
   like `frames/safe/*.jpg`, such as a sharper stitch of the same moments, with them; the margin on each region covers a few pixels of stitch difference, not a different stitch), a `pii` entry per stop on `frames/privacy.json`,
   and `tour/manifest.json` counts them (below). A stop already blurred (its file is the one the step wrote) is skipped on a re-run, so nothing is blurred or recompressed twice.
10. **build** (`build_tour.py`, `exposure.py`, `viewer/index.html`): evens brightness between
   stops using the floor that neighbouring stops both see, and writes the static tour from the cleaned pictures (build refuses to run without
   the privacy result; there is no switch to skip it). The viewer wraps both photos on their depth surfaces and flies the camera between them (a short glide).

11. **dollhouse / plan** (`dollhouse.py`, `dollhouse_uv.py`, `floorplan.py`): every stop's full-sphere depth is fused (TSDF on the GPU, 3 cm voxels where the card's memory
   allows, the volume as big as what the stops saw) into one model of up to 300,000 triangles; the mesh is unwrapped into a real UV atlas (xatlas, MIT) and every texel
   takes its colour from the panorama that faces it best (blended, only where nothing is in the way, never from the person-removed hole); triangles no stop sees in the
   open are dropped. `--block-texture` keeps the old one-block-per-triangle look (`dollhouse_texture.py`). The plan is that model seen straight down with the roof off
   (`tour/plan.png`, 1 cm a pixel), with the walls traced over it, the building's size and the stops drawn on (`tour/plan.svg`; a stop on the plan opens it). Measure
   shows its own error ("+/- x %", the median error against tape measurements of the test walks) and is hidden when that error is over 10 %.

**Pinned notes.** Press and hold (or right-click, or the Note button and tap) any spot in a stop; type a note and add a photo. The pin is kept in map metres, so it shows
from every stop that can see the spot (hidden behind things) and as a green ball in the dollhouse. Notes are kept in the browser; opened from `python serve_tour.py tour`
(`tour.py --serve` does this) they are also written to `notes.json` next to the tour and in the walk's own folder, and every rebuild copies that file back into the tour.
On any other host use "Export notes" and put the downloaded `notes.json` next to `index.html`.

`qa/qa_tour.py` opens the built tour in a headless browser (desktop and phone sizes), walks it, and checks that it loads, has depth,
shows a cursor on the floor, moves on tap, keeps facing, glides in a sensible time, that exposure gains stay sane, and that the view
straight down is not a black blob. With torch installed it also runs `qa/check_nadir_person.py`, which segments the finished panoramas looking
straight down and fails if the person holding the stick is still visible.
Setup: `pip install -r requirements-dev.txt && python -m playwright install chromium`.
`python qa/check_capturer.py` re-checks every cleaned stop (the 4b check: nothing person-shaped or above 15 cm in the filled area, no flat or grainy patch). `python qa/test_units.py` runs 161 fast unit tests (lens model, coordinate frames, floor projection between stops, nadir mask, exposure
solver, step order and the dependency graph, staleness, video checks, RAM-based parallelism, run lock) with no footage, GPU or browser; `qa/test_privacy_chain.py`, `qa/test_false_holds.py`,
`qa/test_neighbour_check.py`, `qa/test_remove_others.py`, `qa/test_synth_fill.py`, `qa/test_pii_blur.py` and `qa/test_speed14.py` cover the person removal, the privacy rules, the blur and the speed-ups; `qa/test_engine_v2.py`, `qa/test_people_blur.py`, `qa/test_heading_check.py` and `qa/test_walked_path.py` the sphere depth, the registration, the people modes and the walked path (CONTRIBUTING.md lists them with their counts; no footage, GPU or model download needed). `qa/qa_tour.py --browser webkit|firefox` repeats the viewer checks in Safari's and Firefox's engines.

## Known limits

- If some stops cannot be placed on the map even after COLMAP's registration is loosened, `tour.py` stops with an error and the list of stops
  (usually a stretch with too little texture or too fast a move between two stops); `--allow-partial` builds the tour anyway.
- The map is as good as COLMAP: it can drift on long featureless stretches, and its metric scale varies by a few percent between runs (the plan is always drawn in the same orientation).
- Depth and floor come from single-photo models: thin objects and glossy floors can fool the floor test.
- The floor under the person is copied from neighbouring stops where they see it; a patch copied from one neighbour can be the wrong object (a rack, a chair) where that neighbour saw something else there, and what nobody saw is a soft fill. Your shadow on a wall, and the stick's shadow on grass, stay.
- The privacy checks fail closed on purpose: a poster of a person, or a mannequin, will hold a stop back until you look and keep it. Real clutter beside where you stood (a toolbox, a machine base) still holds a stop when no two neighbouring stops see it from two sides.
- The verdict is repeatable on one install (the same walk twice gives the same held list, field for field), but not across builds of torch: a different CUDA build, or the CPU, moves the models' numbers a little and held one or two more stops of 25 in our runs. Near the detector's line a real person can be missed: the horizon rule above is the measured answer, not a proof.
- In a live work zone with your crew in the picture the check holds nearly every stop (16 of 17 on a road walk). The tour then exists only if you keep each stop with faces blurred, or `SHOP_TOUR_REMOVE_OTHERS=0` (step 9) to go back to holding them; by default the paint-out (step 9) paints the other people out (by patch synthesis, so a ghost is faint rather than gone: look at `frames/review` and the clean pictures) and never ships a stop where it still finds one.
- Strong parallax on very close objects (ceiling ducts, people) can still show as a faint ghost at the stitch seam. `stitch_lib.stitch_seam` (cut along the best-matching line) exists but is not used: it removes the ghost but leaves brightness steps on plain walls.
- Tested with one camera (Insta360 X5) in one indoor space. The lens model is fitted per video and handles other frame sizes, but 2880x2880 and 30 fps were only emulated by re-encoding this walk, not tested on another camera.
- Long tours: COLMAP matching grows quickly with the number of stops (above 40 stops each stop is matched with the next 20; 76 stops take ~5 min for the map step).

- The accelerometer scale needs a walk with pauses (the path between them is what the accelerometer is compared with) and a clean 1 Hz path; on a second test walk the path
  did not agree with the stop map and the scale fell back to the camera-height assumption. The only camera whose sensor format was worked out is the X5.
- Measuring is not tape-measure accurate: against seven tape measurements the median error is about 9 % on the first test walk and 27 % on the second (where the accelerometer check did not pass), so the viewer shows the error next to Measure and hides the button above 10 %. The camera height (and with it every height) is taken from the dense map's floor; where that map is not trustworthy it falls back to the depth model's floor layer, which can sit above the real floor.

## Licences

Contributing: see [CONTRIBUTING.md](../CONTRIBUTING.md). Every dependency and model with its licence: [THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md).

This project is Apache-2.0 (see `LICENSE`). It uses, without bundling them:
SAM 2.1 small (Apache-2.0), RT-DETR (Apache-2.0), OpenCV YuNet face detector (MIT, shipped in `models/`; also used by the blur step, 9b, which adds PaddleOCR PP-OCRv3 DBNet text detection (Apache-2.0) and `justjuu/rtdetr-v2-license-plate-detection` (Apache-2.0 per its card), fetched once by `tools/fetch_pii_models.py`), SciPy (BSD-3-Clause), COLMAP / pycolmap (BSD-3-Clause), Depth Anything V2 Small (Apache-2.0), MoGe-2 (MIT; the full-sphere depth), xatlas (MIT; the dollhouse's UV atlas), UperNet ConvNeXt-small from openmmlab (MIT, trained on ADE20K),
Hugging Face transformers (Apache-2.0), PyTorch (BSD-style), OpenCV (Apache-2.0), NumPy (BSD), Pillow (MIT-CMU), three.js r128 (MIT, downloaded once at
build time and copied next to the viewer, so a finished tour works without internet), and FFmpeg as a separate program. Models that are not free for commercial use (for example Depth Anything V2 Base/Large,
DAP, MASt3R/DUSt3R) are deliberately not used.
