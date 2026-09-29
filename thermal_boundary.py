"""
Script 2 — find the exact boundary of a human in a thermal image.

Uses only classical OpenCV / NumPy (intensity thresholding, morphology,
connected components, frequency-domain edge detection).  No deep learning or
machine learning.

Thermal images render warmer objects brighter, so the person (near body
temperature) is bright against the cooler background; the image is thresholded
with Otsu's method to isolate the hot body, then its boundary is traced.

Usage:
    python3 thermal_boundary.py                       # uses images/thermal_person.jpg
    python3 thermal_boundary.py path/to/thermal.jpg   # any thermal image

Outputs (written to output/thermal_*):
    thermal_mask.png      binary mask of the person
    thermal_boundary.png  the boundary contour drawn over the original image
    thermal_overlay.png   translucent mask overlay
    thermal_hot.png       raw hot-body mask (before cleanup)
    thermal_gradient.png  frequency-domain gradient magnitude (edges)
    thermal_edges.png     frequency-domain high-pass edge map
    thermal_spectrum.png  log |F(u,v)| image spectrum
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

import segmentation

DEFAULT_IMAGE = Path(__file__).parent / "images" / "thermal_person.jpg"
OUTPUT_DIR = Path(__file__).parent / "output"


def run(image_path: Path) -> dict:
    image_path = Path(image_path)
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"could not read image: {image_path}")

    result = segmentation.segment_thermal(image)
    mask = result["mask"]
    contour = result["contour"]
    h, w = mask.shape

    OUTPUT_DIR.mkdir(exist_ok=True)
    stem = "thermal"
    cv2.imwrite(str(OUTPUT_DIR / f"{stem}_mask.png"), mask)
    cv2.imwrite(str(OUTPUT_DIR / f"{stem}_boundary.png"),
                segmentation.boundary_overlay(image, contour, color=(0, 255, 0), thickness=2))
    cv2.imwrite(str(OUTPUT_DIR / f"{stem}_overlay.png"),
                segmentation.mask_overlay(image, mask, color=(255, 0, 0)))
    cv2.imwrite(str(OUTPUT_DIR / f"{stem}_hot.png"), result["raw_hot"])
    cv2.imwrite(str(OUTPUT_DIR / f"{stem}_gradient.png"), result["gradient"])
    cv2.imwrite(str(OUTPUT_DIR / f"{stem}_edges.png"), result["edges"])
    cv2.imwrite(str(OUTPUT_DIR / f"{stem}_spectrum.png"), result["spectrum"])

    coverage = float((mask > 0).mean())
    boundary_length = float(cv2.arcLength(contour, True))
    print("=" * 60)
    print(f"Thermal human boundary  —  {image_path.name}")
    print("=" * 60)
    print(f"  image size          : {w} x {h}")
    print(f"  person mask coverage: {100 * coverage:.1f}% of image")
    print(f"  boundary vertices   : {len(contour)}")
    print(f"  boundary length     : {boundary_length:.0f} px")
    print(f"  outputs written to  : {OUTPUT_DIR}/thermal_*")
    return result


if __name__ == "__main__":
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_IMAGE
    run(path)
