"""
Compare the classical (non-ML) human boundaries against the SAM2 baseline.

For each test image it loads two binary masks and reports IoU, Dice, pixel
accuracy and boundary F1:

    classical mask  — written by rgb_boundary.py / thermal_boundary.py  (output/)
    SAM2 mask       — written by sam2_segment.py                        (images/sam2/)

Usage:
    python3 compare_sam2.py

Outputs:
    prints a comparison table, and writes output/sam2_comparison.csv
"""

from __future__ import annotations

import csv
from pathlib import Path

import cv2
import numpy as np

import segmentation

ROOT = Path(__file__).parent
OUTPUT_DIR = ROOT / "output"

# Each entry: (name, classical mask path, SAM2 reference mask path)
PAIRS = [
    ("rgb", OUTPUT_DIR / "rgb_mask.png", ROOT / "images" / "sam2" / "rgb_sam2.png"),
    ("thermal", OUTPUT_DIR / "thermal_mask.png", ROOT / "images" / "sam2" / "thermal_sam2.png"),
]


def load_mask(path: Path) -> np.ndarray | None:
    if not path.exists():
        return None
    m = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    return (m > 0).astype(np.uint8) if m is not None else None


def run() -> None:
    rows: list[dict] = []
    print(f"{'image':8s} {'IoU':>7s} {'Dice':>7s} {'pixel acc':>9s} {'boundary F1':>11s}")
    print("-" * 48)

    for name, pred_path, gt_path in PAIRS:
        pred = load_mask(pred_path)
        gt = load_mask(gt_path)

        if pred is None:
            print(f"{name:8s}  (classical mask missing: {pred_path.name})")
            continue
        if gt is None:
            print(f"{name:8s}  (SAM2 mask missing: {gt_path.name} — run sam2_segment.py first)")
            continue

        # Align the two masks to a common size if they differ.
        if pred.shape != gt.shape:
            gt = cv2.resize(gt, (pred.shape[1], pred.shape[0]),
                            interpolation=cv2.INTER_NEAREST)

        m = segmentation.mask_metrics(pred, gt)
        rows.append({"image": name, **m})
        print(
            f"{name:8s} {m['iou']:7.3f} {m['dice']:7.3f} "
            f"{m['pixel_accuracy']:9.3f} {m['boundary_f1']:11.3f}"
        )

    print("-" * 48)

    if rows:
        OUTPUT_DIR.mkdir(exist_ok=True)
        csv_path = OUTPUT_DIR / "sam2_comparison.csv"
        with open(csv_path, "w", newline="") as fh:
            writer = csv.DictWriter(
                fh,
                fieldnames=["image", "iou", "dice", "pixel_accuracy",
                            "boundary_f1", "pred_pixels", "gt_pixels"],
            )
            writer.writeheader()
            writer.writerows(rows)
        print(f"Wrote {csv_path}")


if __name__ == "__main__":
    run()
