"""
Web demonstration — human boundary detection with classical computer vision.

Serves an interactive page where the user picks RGB or thermal mode, optionally
uploads an image, and sees — side by side — the classical (non-ML) segmentation:

  - the detected boundary contour drawn over the original image,
  - the binary mask + translucent overlay,
  - the frequency-domain gradient and high-pass edge map,
  - the log-magnitude spectrum (and the spectral-residual saliency for RGB),

together with the IoU/Dice against the SAM2 baseline when its reference mask is
available.

Run directly (python app.py) or via Docker (see README / docker-compose.yml).
"""

from __future__ import annotations

import base64
import os
from pathlib import Path

import cv2
import numpy as np
from flask import Flask, jsonify, render_template, request

import segmentation

app = Flask(__name__)

ROOT = Path(__file__).parent
MAX_SIDE = 900  # cap working resolution so the FFT stays fast / memory-safe

DEFAULT_IMAGES = {
    "rgb": ROOT / "images" / "rgb_person.jpg",
    "thermal": ROOT / "images" / "thermal_person.jpg",
}
SAM2_REF = {
    "rgb": ROOT / "images" / "sam2" / "rgb_sam2.png",
    "thermal": ROOT / "images" / "sam2" / "thermal_sam2.png",
}


# ======================================================================
# Helpers
# ======================================================================


def ndarray_to_data_url(arr_u8: np.ndarray) -> str:
    """Encode a uint8 image as a base64 PNG data URL."""
    ok, buf = cv2.imencode(".png", arr_u8)
    if not ok:
        raise ValueError("could not encode image")
    b64 = base64.b64encode(buf.tobytes()).decode("ascii")
    return "data:image/png;base64," + b64


def decode_upload(data_url: str) -> np.ndarray:
    """Decode a client-uploaded data URL to a BGR image."""
    header, _, b64 = data_url.partition(",")
    raw = base64.b64decode(b64)
    arr = np.frombuffer(raw, dtype=np.uint8)
    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("could not decode uploaded image")
    return bgr


def load_image(mode: str, data_url: str | None) -> np.ndarray:
    """Return the working image: an upload if given, else the default test image."""
    if data_url and data_url.startswith("data:image"):
        image = decode_upload(data_url)
    else:
        image = cv2.imread(str(DEFAULT_IMAGES[mode]), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"default image missing for mode {mode}")
    # Cap the working resolution.
    h, w = image.shape[:2]
    longest = max(h, w)
    if longest > MAX_SIDE:
        scale = MAX_SIDE / longest
        image = cv2.resize(image, (int(round(w * scale)), int(round(h * scale))),
                           interpolation=cv2.INTER_AREA)
    return image


def sam2_comparison(mode: str, mask: np.ndarray) -> dict | None:
    """Return IoU/Dice vs the SAM2 reference mask, if one exists."""
    ref_path = SAM2_REF[mode]
    if not ref_path.exists():
        return None
    ref = cv2.imread(str(ref_path), cv2.IMREAD_GRAYSCALE)
    if ref is None:
        return None
    if ref.shape != mask.shape:
        ref = cv2.resize(ref, (mask.shape[1], mask.shape[0]),
                         interpolation=cv2.INTER_NEAREST)
    return segmentation.mask_metrics(mask, ref)


def run_pipeline(mode: str, image: np.ndarray, with_sam2: bool = True) -> dict:
    """Run the classical segmentation for `mode` and package results for the client.

    `with_sam2` is False for uploaded images: the SAM2 reference masks exist only
    for the default test images, so comparing an upload against a default mask
    would be meaningless.
    """
    if mode == "thermal":
        result = segmentation.segment_thermal(image)
    else:
        result = segmentation.segment_rgb(image)

    mask = result["mask"]
    contour = result["contour"]
    h, w = mask.shape

    payload = {
        "mode": mode,
        "original": ndarray_to_data_url(image),
        "boundary": ndarray_to_data_url(
            segmentation.boundary_overlay(image, contour, color=(0, 255, 0), thickness=3)),
        "overlay": ndarray_to_data_url(
            segmentation.mask_overlay(image, mask, color=(0, 255, 0))),
        "mask": ndarray_to_data_url(mask),
        "gradient": ndarray_to_data_url(result["gradient"]),
        "edges": ndarray_to_data_url(result["edges"]),
        "spectrum": ndarray_to_data_url(result["spectrum"]),
        "coverage": float((mask > 0).mean()),
        "boundary_vertices": int(len(contour)),
        "boundary_length": float(cv2.arcLength(contour, True)),
        "width": int(w),
        "height": int(h),
    }
    if mode == "rgb":
        payload["saliency"] = ndarray_to_data_url(result["saliency"])

    payload["sam2"] = sam2_comparison(mode, mask) if with_sam2 else None
    return payload


# ======================================================================
# Routes
# ======================================================================


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/segment", methods=["POST"])
def api_segment():
    data = request.get_json(force=True) or {}
    mode = data.get("mode", "rgb")
    if mode not in ("rgb", "thermal"):
        mode = "rgb"

    try:
        image = load_image(mode, data.get("image"))
        # SAM2 comparison is only meaningful for the default test images.
        result = run_pipeline(mode, image, with_sam2=not data.get("image"))
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    return jsonify(result)


@app.route("/api/health")
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", "8000"))
    app.run(host=host, port=port, debug=False)
