# Third-party software, models and data

shop-tour itself is Apache-2.0 (`LICENSE`). This repository **bundles** exactly one third-party model file (the YuNet face detector, with its licence
text) and one licence text for a model it fetches on request; everything else is installed or downloaded by you, from its own source, under its own
licence. Nothing used here is non-commercial or research-only, and nothing AGPL/GPL is used or bundled (FFmpeg is run as a separate program; see below).

How each licence was checked (last column): **card** = the `license` field of the model card, read through the huggingface.co API on 2026-10-30;
**repo** = the repository's LICENSE file / GitHub's licence API, read 2026-10-30; **package** = the licence metadata of the installed package on the
reference machine, read 2026-10-30; **file** = a licence text shipped next to the file; **registry** = the npm registry entry.

## What ships in this repository

| File | What it is | Licence | Notice |
|---|---|---|---|
| `models/face_detection_yunet_2023mar.onnx` | YuNet face detector (OpenCV model zoo, `models/face_detection_yunet`) | MIT | `models/LICENSE-yunet.txt` (Copyright (c) 2020 Shiqi Yu); keep it next to the file |
| `models/LICENSE-ppocr.txt` | licence text for the PP-OCRv3 text detector that `tools/fetch_pii_models.py` downloads (the model itself is not in git) | Apache-2.0 | Copyright (c) 2016 PaddlePaddle Authors |

A **built tour** (`tour/`) contains `three.min.js` (three.js r128, MIT, its `@license` header names the three.js authors), copied with its licence header intact so the
tour works offline. If you host a tour, that header is its notice; do not strip it.

## Model weights (downloaded by you, on the first build or by `tools/fetch_pii_models.py`)

| Model | Used for | Licence | Commercial use | Source | Checked |
|---|---|---|---|---|---|
| SAM 2.1 Hiera-small | masks of the camera holder and of other people (`capturer.py`, `remove_others.py`, `people_blur.py`) | Apache-2.0 | yes | `facebook/sam2.1-hiera-small` | card |
| Depth Anything V2 **Small** | single-photo depth (`depth_panos.py`, `capturer.py`) | Apache-2.0 | yes | `depth-anything/Depth-Anything-V2-Small-hf` | card |
| MoGe-2 ViT-L (normal) | one metric full-sphere depth per stop (`depth_sphere.py`) — the dollhouse, the plan, the glide and Measure stand on it; about 1.3 GB | MIT (code and weights) | yes | `Ruicheng/moge-2-vitl-normal`; code `microsoft/MoGe` | card + repo |
| UperNet ConvNeXt-small | which pixels are floor, sky, water (`floor_panos.py`, `level_panos.py`, `depth_sphere.py`) | MIT (weights) | weights yes — **see training data** | `openmmlab/upernet-convnext-small` | card |
| RT-DETR R50vd | finds people and vehicles for the privacy gate (`privacy_gate.py`) | Apache-2.0 | yes | `PekingU/rtdetr_r50vd` | card |
| RT-DETR v2 licence-plate detector | plate blur (`pii_blur.py`); about 164 MB | Apache-2.0 | yes (per its card) | `justjuu/rtdetr-v2-license-plate-detection`, base `PekingU/rtdetr_v2_r50vd` (Apache-2.0) | card |
| PaddleOCR PP-OCRv3 DBNet (English text detection) | text and name blur (`pii_blur.py`); 2.4 MB ONNX run by OpenCV | Apache-2.0 | yes | OpenCV model zoo, `models/text_detection_ppocr` | file |
| YuNet (2023 Mar) | faces (`privacy_gate.py`, `pii_blur.py`) | MIT | yes | shipped, see above | file |
| Depth Anything V2 Small (ONNX) | only the browser experiment in `experiments/webgpu-depth/index.html` (not part of a build) | Apache-2.0 | yes | `onnx-community/depth-anything-v2-small` | card |

MoGe-2's backbone is DINOv2 (Apache-2.0, `facebookresearch/dinov2`, repo). MoGe's repository carries both an MIT and an Apache-2.0 licence text (the latter
for the DINOv2 code it includes).

Deliberately **not** used: Depth Anything V2 Base/Large (CC-BY-NC-4.0), MASt3R / DUSt3R (non-commercial), DAP, Ultralytics YOLO (AGPL-3.0), and any
research-only weights. MapAnything is not used by this engine.

### Training data (read this if you sell what you build)

A weights licence does not change the terms of the data a model was trained on, and that question is unsettled law. What is known:

- **UperNet ConvNeXt-small was trained on ADE20K**, whose images are distributed for research and education use. The weights are MIT by their authors.
  If you need a floor/sky model with commercially clear training data, replace the model in `floor_panos.py` (and `SEG_MODEL` in `level_panos.py` and
  `depth_sphere.py`); nothing else depends on it.
- RT-DETR was trained on COCO (annotations CC-BY-4.0; the images keep their own Flickr licences).
- **YuNet was trained on WIDER FACE**, which its authors release for non-commercial research. The YuNet weights are MIT. The same question as
  ADE20K applies; faces are only ever *blurred* with it, never identified.
- The plate model's dataset `justjuu/license-plate-detection` is marked CC-BY-4.0 on its card.

## Python packages (`pip install -r requirements.txt`; none are bundled)

Versions are the ones on the reference machine.

| Package | Version | Licence | Checked |
|---|---|---|---|
| PyTorch (`torch`, CUDA 13.0 build) | 2.14.1 (pinned) | BSD-3-Clause / Apache-2.0 mix (per its metadata); the CUDA wheels pull NVIDIA runtime libraries under NVIDIA's own licence | package |
| torchvision | 0.29.1 (pinned) | BSD-3-Clause | package |
| NumPy | 2.2.6 | BSD-3-Clause | package |
| SciPy | 1.18.1 | BSD-3-Clause | package |
| OpenCV (`opencv-python-headless`) | 5.0.0.93 | Apache-2.0 | package |
| Pillow | 11.2.1 | MIT-CMU (HPND) | package |
| Hugging Face `transformers` | 5.20.1 | Apache-2.0 | package |
| `huggingface-hub`, `safetensors`, `tokenizers` (pulled by transformers) | 2.2.0, 0.8.0, 0.23.3 | Apache-2.0 | package |
| `pycolmap` (COLMAP) | 4.2.1 | BSD-3-Clause | package |
| Open3D | 0.20.0 | MIT | package |
| scikit-image | 0.26.0 | BSD-3-Clause | package |
| xatlas (xatlas-python) | 0.0.11 | MIT | package + repo |
| `moge` (microsoft/MoGe, pinned to commit `74fbce0`) | 3.0.0 | MIT (+ Apache-2.0 for its DINOv2 code) | package + repo |
| pulled by MoGe: `utils3d_moge`, `pipeline`, `flex-gemm` | 1.7, 1.0.0, 1.0.0 | MIT (all three, GitHub repos of EasternJournalist and JeffreyXiang) | package + repo |
| pulled by MoGe: `trimesh`, `gradio`, `matplotlib`, `click` | 5.2.0, 6.30.0, 3.11.0, 8.5.0 | MIT, Apache-2.0, matplotlib licence (PSF-style, permissive), BSD-3-Clause | package |
| Playwright (QA only, `requirements-dev.txt`) | 1.52.0 | Apache-2.0 | package |

## Programs and web assets

| Item | How it is used | Licence |
|---|---|---|
| FFmpeg / ffprobe | run as a separate program from your PATH; not bundled, not linked | LGPL-2.1+ or GPL-2.0+ depending on how your build was configured; shop-tour only runs the executable |
| three.js r128 | fetched once from cdnjs at build time and copied into each tour with its licence header | MIT (registry: `three@0.128.0`) |
| onnxruntime-web 1.20.1, `@huggingface/transformers` 3.5.1 | loaded from jsDelivr by the browser-only experiments in `experiments/webgpu-depth/` (not part of a build) | MIT, Apache-2.0 (registry) |
| Insta360 MediaSDK | **not used and not shipped.** The pipeline stitches from the raw `.insv` lens streams itself | Insta360's own terms, if you use it elsewhere |

## Sample data

No footage, panorama, thumbnail or tour of a real place is in this repository. The SVG drawings in `docs/capture/` are original to this project
(Apache-2.0). There is no bundled demo walk.
