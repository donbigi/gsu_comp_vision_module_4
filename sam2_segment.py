"""
Generate SAM2 reference masks for the test images (the deep-learning baseline).

SAM2 (Meta "Segment Anything Model 2") is used ONLY as a comparison baseline.
The classical solution in this module never uses it — the two boundary scripts
and `segmentation.py` rely solely on fixed thresholds, linear frequency-domain
filters, morphology and connected components.

SAM2 is an *interactive* segmenter: it is prompted, then predicts the object.
The standard evaluation protocol is to prompt it with a bounding box around the
object and take its mask.  We therefore give SAM2 the bounding box of the
classical mask (dilated slightly), which is exactly the standard "box prompt"
evaluation used in the SAM papers.  This isolates what we are comparing: given
the same rough location, how much better is SAM2's boundary than the classical
one.

One-time setup (see README):
    git clone https://github.com/facebookresearch/sam2
    pip install -e ./sam2        # installs the `sam2` package + deps (torch, hydra, ...)
    python3 sam2_segment.py      # downloads the tiny checkpoint on first run

Usage:
    python3 sam2_segment.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import cv2
import numpy as np

import segmentation

ROOT = Path(__file__).parent
CHECKPOINT_DIR = ROOT / "checkpoints"
CHECKPOINT = CHECKPOINT_DIR / "sam2_hiera_tiny.pt"
CHECKPOINT_URL = "https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_tiny.pt"
CONFIG_RELPATH = "configs/sam2/sam2_hiera_t.yaml"  # inside the sam2 package

IMAGES = [
    ("rgb", ROOT / "images" / "rgb_person.jpg", ROOT / "images" / "sam2" / "rgb_sam2.png"),
    ("thermal", ROOT / "images" / "thermal_person.jpg", ROOT / "images" / "sam2" / "thermal_sam2.png"),
]


def _import_sam2():
    """Import the `sam2` package, adding a local clone to sys.path if needed."""
    try:
        import sam2  # noqa: F401
        return sam2
    except ImportError:
        for candidate in (os.environ.get("SAM2_REPO"), "/tmp/sam2_repo", str(ROOT.parent / "sam2")):
            if candidate and Path(candidate, "sam2").is_dir():
                sys.path.insert(0, candidate)
                break
        try:
            import sam2  # noqa: F401
            return sam2
        except ImportError as exc:
            raise SystemExit(
                "SAM2 is not installed. Run:\n"
                "    git clone https://github.com/facebookresearch/sam2\n"
                "    pip install -e ./sam2\n"
                "or set SAM2_REPO to a cloned copy of the repository."
            ) from exc


def ensure_checkpoint() -> None:
    if CHECKPOINT.exists():
        return
    CHECKPOINT_DIR.mkdir(exist_ok=True)
    print(f"Downloading SAM2 tiny checkpoint to {CHECKPOINT} ...")
    import urllib.request

    urllib.request.urlretrieve(CHECKPOINT_URL, CHECKPOINT)
    print("Download complete.")


def classical_bbox(mode: str, image: np.ndarray) -> tuple[int, int, int, int]:
    """Bounding box of the classical mask (the prompt we give SAM2)."""
    result = segmentation.segment_rgb(image) if mode == "rgb" else segmentation.segment_thermal(image)
    ys, xs = np.where(result["mask"] > 0)
    if len(xs) == 0:
        h, w = image.shape[:2]
        return 0, 0, w, h
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def run() -> None:
    ensure_checkpoint()

    sam2_pkg = _import_sam2()
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    config_path = Path(sam2_pkg.__file__).parent / CONFIG_RELPATH
    if not config_path.exists():
        raise SystemExit(f"config not found: {config_path}")

    device = "cuda" if __import__("torch").cuda.is_available() else "cpu"
    print(f"Building SAM2 (tiny) on {device} ...")
    sam2_model = build_sam2(CONFIG_RELPATH, str(CHECKPOINT), device=device)
    predictor = SAM2ImagePredictor(sam2_model)

    for mode, image_path, out_path in IMAGES:
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        x1, y1, x2, y2 = classical_bbox(mode, image)
        h, w = image.shape[:2]
        # Dilate the box a little so SAM2 has margin around the person.
        mx, my = 0.05 * (x2 - x1), 0.05 * (y2 - y1)
        box = np.array([
            max(0, x1 - mx), max(0, y1 - my),
            min(w, x2 + mx), min(h, y2 + my),
        ], dtype=np.float32)
        print(f"Segmenting {image_path.name} (box={box.astype(int).tolist()}) ...")

        predictor.set_image(image_rgb)
        masks, scores, _ = predictor.predict(
            box=box,
            multimask_output=False,
        )
        mask = (masks[0] > 0).astype(np.uint8) * 255

        out_path.parent.mkdir(exist_ok=True)
        cv2.imwrite(str(out_path), mask)
        print(f"  wrote {out_path}  ({mask.shape[1]}x{mask.shape[0]}, "
              f"score={scores[0]:.3f}, {100 * float((mask > 0).mean()):.1f}% coverage)")


if __name__ == "__main__":
    run()
