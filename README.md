# Module 4 — Finding a Human's Boundary with Classical CV (vs SAM2)

This module finds the **exact boundary of a human** in an RGB image and in a
thermal image, using **only classical OpenCV / NumPy** — fixed colour and
intensity thresholds, frequency-domain (Fourier) filtering, morphology and
connected components. **No deep learning and no machine learning** is used in
the segmentation. The result is then compared against Meta's **SAM2**
(Segment Anything Model 2) as the deep-learning baseline.

It covers the three tasks in `task.md`:

1. **RGB human boundary** — `rgb_boundary.py`
2. **Thermal human boundary** — `thermal_boundary.py`
3. **Theory with derivations** — Fourier-domain derivations

A Flask web app demonstrates everything interactively.

## The idea (one paragraph)

The Fourier transform gives two facts that do all the work. The
**differentiation property** says a derivative is multiplication by a ramp in
the frequency domain — `DFT{∂f/∂x} = j·2πu·F(u,v)` — so a gradient (an edge
detector) is simply a *high-pass* filter. And an edge (a step) has energy that
decays only slowly with frequency, while a smooth region decays fast — so
removing low frequencies keeps edges and discards smooth interiors. The *region*
is found classically (figure-ground Otsu thresholding, confirmed by skin-colour
and spectral-residual saliency, for RGB; hot-body Otsu thresholding for thermal)
and its *exact boundary* is the contour traced around the cleaned mask, with the
frequency-domain gradient, high-pass edge maps and saliency computed alongside.

## Files

| File | Purpose |
| ---- | ------- |
| `segmentation.py` | Shared classical pipeline + frequency-domain edge detection + metrics (IoU/Dice/boundary-F1) |
| `rgb_boundary.py` | Task 1 — RGB image → mask, boundary, overlay, gradient, edges, spectrum |
| `thermal_boundary.py` | Task 2 — thermal image → same outputs |
| `sam2_segment.py` | Runs SAM2 locally (box-prompted) to produce reference masks |
| `compare_sam2.py` | IoU / Dice / pixel-accuracy / boundary-F1 vs SAM2 → table + CSV |
| `app.py` + `templates/` + `static/` | Flask web demo |
| `Dockerfile`, `docker-compose.yml`, `requirements.txt` | Packaging |

## Running the scripts

```bash
# classical segmentation (writes output/rgb_*, output/thermal_*)
python3 rgb_boundary.py
python3 thermal_boundary.py

# compare against the SAM2 reference masks
python3 compare_sam2.py
```

Each script prints the coverage, boundary vertices and length, and writes
intermediate images to `output/`.

## SAM2 baseline (one-time, offline)

The comparison needs SAM2 reference masks, which are generated once and shipped
as small PNGs under `images/sam2/` (so the web app and Docker image do **not**
need the model).

```bash
git clone https://github.com/facebookresearch/sam2
pip install -e ./sam2          # installs sam2 + deps (torch, hydra-core, ...)
python3 sam2_segment.py        # downloads sam2_hiera_tiny.pt on first run
```

`sam2_segment.py` prompts SAM2 with the classical mask's bounding box (the
standard SAM evaluation protocol) and writes `images/sam2/*.png`. If the
checkpoint or the `sam2` package is missing, `compare_sam2.py` and the web app
skip the comparison gracefully instead of crashing.

## Results

| Image | Classical coverage | IoU vs SAM2 | Dice | Pixel acc | Boundary F1 |
| ----- | ------------------ | ----------- | ---- | --------- | ----------- |
| RGB (full-length seated portrait) | 28.8% | 0.86 | 0.93 | 0.96 | 0.46 |
| Thermal (PDIWS)   | 14.2% | 0.43 | 0.60 | 0.89 | 0.38 |

The classical method recovers the person's silhouette with no training data,
but SAM2's boundaries are sharper — the gap is the module's finding.

## Web app

```bash
python3 app.py                 # http://localhost:8000
# or
docker compose up --build      # (module-level compose)
```

The UI lets you switch between RGB and thermal, upload your own image, and see
the boundary, mask overlay, frequency-domain gradient, high-pass edges, spectrum
and spectral-residual saliency side by side, with the IoU vs SAM2.

## Test images

- `images/rgb_person.jpg` — a CC BY-SA full-length studio portrait of a seated
  woman in dark clothing against a plain white background (Wikimedia Commons:
  "Melanie Williamson"). The clean figure-ground separation makes it ideal for
  the classical method while still giving SAM2 a natural colour photograph.
- `images/thermal_person.jpg` — a thermal frame from the Roboflow thermal
  dataset linked in `task.md`.
