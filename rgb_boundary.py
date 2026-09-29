"""
Script 1 — find the exact boundary of a human in an RGB image.

Uses only classical OpenCV / NumPy (fixed colour thresholds, morphology,
connected components, frequency-domain edge detection).  No deep learning or
machine learning.

Usage:
    python3 rgb_boundary.py                          # uses images/rgb_person.jpg
    python3 rgb_boundary.py path/to/photo.jpg        # any RGB image

Outputs (written to output/rgb_*):
    rgb_mask.png        binary mask of the person
    rgb_boundary.png    the boundary contour drawn over the original image
    rgb_overlay.png     translucent mask overlay
    rgb_skin.png        raw skin-colour mask (before cleanup)
    rgb_gradient.png    frequency-domain gradient magnitude (edges)
    rgb_edges.png       frequency-domain high-pass edge map
    rgb_spectrum.png    log |F(u,v)| image spectrum
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

import segmentation

DEFAULT_IMAGE = Path(__file__).parent / "images" / "rgb_person.jpg"
OUTPUT_DIR = Path(__file__).parent / "output"


def run(image_path: Path) -> dict:
    image_path = Path(image_path)
    bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"could not read image: {image_path}")

    result = segmentation.segment_rgb(bgr)
    mask = result["mask"]
    contour = result["contour"]
    h, w = mask.shape

    OUTPUT_DIR.mkdir(exist_ok=True)
    stem = "rgb"
    cv2.imwrite(str(OUTPUT_DIR / f"{stem}_mask.png"), mask)
    cv2.imwrite(str(OUTPUT_DIR / f"{stem}_boundary.png"),
                segmentation.boundary_overlay(bgr, contour, color=(0, 255, 0), thickness=3))
    cv2.imwrite(str(OUTPUT_DIR / f"{stem}_overlay.png"),
                segmentation.mask_overlay(bgr, mask, color=(0, 255, 0)))
    cv2.imwrite(str(OUTPUT_DIR / f"{stem}_skin.png"), result["raw_skin"])
    cv2.imwrite(str(OUTPUT_DIR / f"{stem}_saliency.png"), result["saliency"])
    cv2.imwrite(str(OUTPUT_DIR / f"{stem}_gradient.png"), result["gradient"])
    cv2.imwrite(str(OUTPUT_DIR / f"{stem}_edges.png"), result["edges"])
    cv2.imwrite(str(OUTPUT_DIR / f"{stem}_spectrum.png"), result["spectrum"])

    coverage = float((mask > 0).mean())
    boundary_length = float(cv2.arcLength(contour, True))
    print("=" * 60)
    print(f"RGB human boundary  —  {image_path.name}")
    print("=" * 60)
    print(f"  image size          : {w} x {h}")
    print(f"  person mask coverage: {100 * coverage:.1f}% of image")
    print(f"  boundary vertices   : {len(contour)}")
    print(f"  boundary length     : {boundary_length:.0f} px")
    print(f"  outputs written to  : {OUTPUT_DIR}/rgb_*")
    return result


if __name__ == "__main__":
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_IMAGE
    run(path)
