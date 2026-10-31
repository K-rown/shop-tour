"""Export the floor model (openmmlab/upernet-convnext-small, MIT) to ONNX so a browser can run it, and save a test tensor + the Python answer.

Usage: python experiments/webgpu-depth/export_floor_onnx.py --out D:/somewhere/onnx [--libs D:/path/with/onnx]
Writes: floor.onnx (the model, ~200 MB, not committed), test_input.f32 (1x3x512x512 float32, little endian), test_logits.f32 (1x150x128x128),
        meta.json (input size, the class ids that count as floor, normalisation). Experiment only.
"""
import argparse, json, os, sys

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
ap.add_argument("--libs", default=None, help="folder with onnx / onnxscript (appended to sys.path)")
ap.add_argument("--pano", default=None, help="a 2:1 panorama to cut the test view from (default: noise-free gradient picture)")
a = ap.parse_args()
if a.libs:
    sys.path.append(a.libs)
import numpy as np
import torch
from transformers import AutoImageProcessor, UperNetForSemanticSegmentation

HERE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, HERE)
from floor_panos import MODEL, FLOORISH, NADIR_VIEW          # noqa: E402
from depth_panos import cut_view                              # noqa: E402

os.makedirs(a.out, exist_ok=True)
proc = AutoImageProcessor.from_pretrained(MODEL)
model = UperNetForSemanticSegmentation.from_pretrained(MODEL).eval()
floor_ids = [int(i) for i, n in model.config.id2label.items() if any(f in n.lower() for f in FLOORISH)]
print("preprocessing:", {k: getattr(proc, k, None) for k in ("do_resize", "size", "resample", "do_rescale", "do_normalize", "image_mean", "image_std")})

import cv2
from PIL import Image
if a.pano:
    pano = cv2.resize(cv2.imread(a.pano), (4096, 2048), interpolation=cv2.INTER_AREA)
    view = cv2.cvtColor(cut_view(pano, {"yaw": 0.0, "pitch": -25.0, "fov": 100.0}), cv2.COLOR_BGR2RGB)
else:
    view = (np.dstack([np.tile(np.linspace(0, 255, 518), (518, 1))] * 3)).astype(np.uint8)
x = proc(images=Image.fromarray(view), return_tensors="pt")["pixel_values"]
print("input tensor", tuple(x.shape))
x.numpy().astype("<f4").tofile(os.path.join(a.out, "test_input.f32"))
cv2.imwrite(os.path.join(a.out, "test_view.png"), cv2.cvtColor(view, cv2.COLOR_RGB2BGR))


class FixedAdaptivePool(torch.nn.Module):
    """AdaptiveAvgPool2d for a known input size, written with plain slices (the ONNX exporter cannot do 16 -> 3 or 16 -> 6): same windows as PyTorch,
    start = floor(i * n / out), end = ceil((i + 1) * n / out)."""
    def __init__(self, n, out):
        super().__init__()
        self.n, self.out = n, out

    def forward(self, t):
        import math
        w = [(i * self.n // self.out, math.ceil((i + 1) * self.n / self.out)) for i in range(self.out)]
        rows = []
        for (y0, y1) in w:
            rows.append(torch.stack([t[:, :, y0:y1, x0:x1].mean(dim=(2, 3)) for (x0, x1) in w], dim=-1))
        return torch.stack(rows, dim=-2)


class Head(torch.nn.Module):
    """Backbone + decode head; the final 4x upsample to the picture size is left to the caller (it is only an interpolation)."""
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, pixel_values):
        feats = self.m.backbone.forward_with_filtered_kwargs(pixel_values, output_hidden_states=False, output_attentions=False).feature_maps
        return self.m.decode_head(feats)


head0 = Head(model).eval()
with torch.no_grad():
    ref = head0(x)                                              # the original model's decode-head logits (128 x 128), before any rewrite
fm = 512 // 32                                                  # the last feature map is 16 x 16 for a 512 x 512 input
for block in model.decode_head.psp_modules.blocks:              # the block forwards through block.layers, so replace the entry there (and the registered module)
    pool = block.layers[0]
    new_pool = FixedAdaptivePool(fm, pool.output_size if isinstance(pool.output_size, int) else pool.output_size[0])
    block.layers[0] = new_pool; block._modules["0"] = new_pool
head = Head(model).eval()
with torch.no_grad():
    logits = head(x)                                            # recompute with the rewritten pooling: must still match the original model
print("logits (before the final upsample)", tuple(logits.shape))
print("rewritten pooling vs original: max |difference| of the logits = %.2e (logit range %.1f .. %.1f)" % (float((logits - ref).abs().max()), float(ref.min()), float(ref.max())))
logits.numpy().astype("<f4").tofile(os.path.join(a.out, "test_logits.f32"))
torch.onnx.export(head, (x,), os.path.join(a.out, "floor.onnx"), input_names=["pixel_values"], output_names=["logits"], opset_version=17, dynamo=False)
json.dump({"input": list(x.shape), "logits": list(logits.shape), "floor_ids": floor_ids, "model": MODEL,
           "mean": list(proc.image_mean), "std": list(proc.image_std), "size": str(proc.size)}, open(os.path.join(a.out, "meta.json"), "w"), indent=1)
print("wrote", a.out, "floor ids", floor_ids, "model MB", round(os.path.getsize(os.path.join(a.out, "floor.onnx")) / 1048576))
