# Third-party software and models

shop-tour itself is Apache-2.0 (`LICENSE`). It bundles one model file and one set of licence texts (below); everything else is installed or downloaded
by you. Nothing here is non-commercial or research-only, and nothing AGPL/GPL is used or bundled. How each licence was checked is in the last column:
**card** = the licence field of the model card on huggingface.co read through its API on 2026-10-26, **package** = the licence metadata of the installed
package, **file** = a licence text shipped next to the file, **registry** = the npm registry entry.

## Models (weights)

| Model | Used for | Licence | Source | Checked | Notes |
|---|---|---|---|---|---|
| SAM 2.1 Hiera-small | masks of the person holding the stick, and of other people the gate finds (`capturer.py`, `remove_others.py`) | Apache-2.0 | `facebook/sam2.1-hiera-small` | card | downloaded on the first build |
| Depth Anything V2 **Small** | single-photo depth (`depth_panos.py`, `capturer.py`) | Apache-2.0 | `depth-anything/Depth-Anything-V2-Small-hf` | card | Only the Small model is Apache-2.0; Depth Anything V2 Base/Large are not free for commercial use and are NOT used. |
| MoGe-2 ViT-L (normal) | one metric full-sphere depth per stop (`depth_sphere.py`) — the dollhouse, the plan, the glide and Measure are built on it | MIT (code and weights) | `Ruicheng/moge-2-vitl-normal` via `pip install git+https://github.com/microsoft/MoGe.git` | card + repo LICENSE | downloaded on the first build, about 1.3 GB; its backbone is DINOv2 (Apache-2.0) |
| xatlas (xatlas-python) | the dollhouse's UV atlas (`dollhouse_uv.py`) | MIT | PyPI `xatlas` | repo LICENSE | not a model: a mesh parametriser |
| UperNet ConvNeXt-small | which pixels are floor (`floor_panos.py`) | MIT | `openmmlab/upernet-convnext-small` | card | The weights licence is MIT; the network was trained on ADE20K, whose own data terms are research-oriented. See "Training data" below. |
| RT-DETR R50vd | finds people (and vehicles) for the privacy gate and the paint-out (`privacy_gate.py`) | Apache-2.0 | `PekingU/rtdetr_r50vd` | card | trained on COCO |
| RT-DETR v2 licence-plate detector | plate blur (`pii_blur.py`) | Apache-2.0 | `justjuu/rtdetr-v2-license-plate-detection` (base `PekingU/rtdetr_v2_r50vd`, Apache-2.0) | card | fetched once by `tools/fetch_pii_models.py`, about 164 MB, not in git. Its training dataset `justjuu/license-plate-detection` is CC-BY-4.0 (card). |
| PaddleOCR PP-OCRv3 DBNet (English text detection) | text/name blur (`pii_blur.py`) | Apache-2.0 | OpenCV model zoo, `models/text_detection_ppocr` | file (`models/LICENSE-ppocr.txt`) | fetched once by `tools/fetch_pii_models.py`, 2.4 MB ONNX, run by OpenCV's own reader |
| YuNet face detector (2023 Mar) | faces (`privacy_gate.py`, `pii_blur.py`) | MIT | OpenCV model zoo, `models/face_detection_yunet` | file (`models/LICENSE-yunet.txt`) | **shipped in `models/`** (232 KB) with its licence text |

Not used, on purpose: MASt3R / DUSt3R, DAP, Depth Anything V2 Base/Large, Ultralytics YOLO, or anything else that is non-commercial or AGPL.

### Training data
A weights licence does not change the licence of the data a model was trained on. The people and plate detectors were trained on COCO and a CC-BY-4.0 plate set;
UperNet-ConvNeXt was trained on ADE20K, which is distributed for research use. The weights themselves are licensed MIT by their authors. If you need a floor
model with commercially clear training data, replace `floor_panos.py`'s model; nothing else depends on it.

## Python packages (installed with `pip install -r requirements.txt`; none are bundled)

| Package | Version used | Licence | Checked |
|---|---|---|---|
| PyTorch (`torch`) | 2.14.1 (pinned) | BSD-3-Clause and Apache-2.0 (and others, per its own metadata); CUDA builds pull NVIDIA runtime libraries under NVIDIA's licence | package |
| torchvision | 0.29.1 (pinned) | BSD | package |
| NumPy | 2.2.6 | BSD-3-Clause | package |
| SciPy | 1.18.1 | BSD-3-Clause | package |
| OpenCV (`opencv-python-headless`) | 5.0.0.93 | Apache-2.0 | package |
| Pillow | 11.2.1 | MIT-CMU (HPND) | package |
| Hugging Face `transformers` | 5.20.1 | Apache-2.0 | package |
| `pycolmap` (COLMAP) | 4.2.1 | BSD-3-Clause | package |
| Open3D | 0.20.0 | MIT | package |
| scikit-image | 0.26.0 | BSD-3-Clause | package |
| Playwright (QA only, `requirements-dev.txt`) | 1.52.0 | Apache-2.0 | package |

## Programs and web assets

| Item | How it is used | Licence |
|---|---|---|
| FFmpeg / ffprobe | called as a separate program from your PATH; not bundled | LGPL-2.1+ or GPL-2+ depending on how your build was configured; shop-tour only runs the executable |
| three.js r128 | downloaded once at build time from cdnjs and copied next to the viewer, with its licence header, so a finished tour works offline | MIT (npm registry: `three@0.128.0`) |
| onnxruntime-web 1.20.1, `@huggingface/transformers` 3.5.1 | loaded from jsDelivr by the browser-only experiments in `experiments/webgpu-depth/` (not part of a tour build) | MIT, Apache-2.0 (registry) |
| Insta360 MediaSDK | **optional and not shipped.** Nothing in this repository calls it; the pipeline stitches from the raw `.insv` lens streams itself. | Insta360's own terms, if you use it elsewhere |
