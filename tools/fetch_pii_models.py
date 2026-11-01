"""Download the two models the picture-blur step (pii_blur.py) uses, ONCE, into models/ - a tour run never fetches anything.
Usage: python tools/fetch_pii_models.py
Prints each file's sha256 and licence; pii_blur.py loads them by path and refuses a file whose sha256 differs from PINNED in pii_blur.py.
  - text detector: PaddleOCR PP-OCRv3 DBNet text detection (English), the ONNX in OpenCV's model zoo (opencv_zoo, models/text_detection_ppocr) - Apache-2.0 (Copyright PaddlePaddle Authors; the zoo's copy is Apache-2.0 too). Run with OpenCV's own DB reader: no PaddlePaddle, no onnxruntime.
  - plate detector: justjuu/rtdetr-v2-license-plate-detection on Hugging Face (RT-DETR v2 fine-tuned on plates) - the model card declares apache-2.0 (base PekingU/rtdetr_v2_r50vd, Apache-2.0). The card does not state its training dataset's licence.
"""
import hashlib, os, sys, urllib.request

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
M = os.path.join(HERE, "models")
FILES = [  # (path under models/, url, licence)
    ("text_det_ppocrv3_en.onnx", "https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/text_detection_ppocr/text_detection_en_ppocrv3_2023may.onnx", "Apache-2.0 (PaddleOCR PP-OCRv3 det, via opencv_zoo)"),
    ("LICENSE-ppocr.txt", "https://raw.githubusercontent.com/opencv/opencv_zoo/main/models/text_detection_ppocr/LICENSE", "Apache-2.0 text"),
    ("plate_rtdetrv2/model.safetensors", "https://huggingface.co/justjuu/rtdetr-v2-license-plate-detection/resolve/main/model.safetensors", "Apache-2.0 (model card)"),
    ("plate_rtdetrv2/config.json", "https://huggingface.co/justjuu/rtdetr-v2-license-plate-detection/resolve/main/config.json", "Apache-2.0 (model card)"),
    ("plate_rtdetrv2/preprocessor_config.json", "https://huggingface.co/justjuu/rtdetr-v2-license-plate-detection/resolve/main/preprocessor_config.json", "Apache-2.0 (model card)"),
]


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def main():
    for rel, url, lic in FILES:
        p = os.path.join(M, rel)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        if not os.path.exists(p):
            print("fetching", url)
            urllib.request.urlretrieve(url, p)
        print(f"{rel}\n  sha256 {sha(p)}\n  licence {lic}")


if __name__ == "__main__":
    main()
