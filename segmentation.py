"""
Human boundary detection with classical computer vision — no deep learning,
no machine learning, and no learned models.

The pipeline has three conceptual pieces, all of which use only fixed
thresholds, linear filtering, morphology and connected components:

  1.  Region detection              — Otsu figure-ground thresholding splits the
                                      person from the background in RGB,
                                      confirmed by skin-colour and spectral-
                                      residual saliency (the Fourier-domain
                                      analogue); hot-body (Otsu) thresholding
                                      does the same for thermal.
  2.  Frequency-domain edge detection — the Fourier transform of the image is
                                      multiplied by a high-pass (or derivative)
                                      transfer function to find *boundaries*.
  3.  Boundary extraction            — the mask is cleaned morphologically, the
                                      largest connected component (the person)
                                      is kept, and its contour is traced and
                                      smoothed into the final boundary.

Why the frequency domain?  The theory in `theory.md` derives two facts used
here:

  - The derivative of a function is a *high-pass* operation in frequency:
        DFT{ df/dx } = j * 2*pi*u * F(u, v)
    so a gradient operator literally multiplies the spectrum by a ramp.

  - An edge (a step) concentrates its energy at high frequencies, so a
    high-pass filter passes edges and removes smooth regions (low frequency).

Both are implemented below and used to detect the boundary, then compared
against the SAM2 (deep-learning) baseline in `compare_sam2.py`.
"""

from __future__ import annotations

import numpy as np
import cv2


# ======================================================================
# Frequency-domain edge detection
# ======================================================================


def log_magnitude_spectrum(gray: np.ndarray) -> np.ndarray:
    """Log-magnitude spectrum of a grayscale image, centred and scaled to uint8.

    The DC (zero-frequency) component is shifted to the middle so the familiar
    picture appears: a bright centre = low frequencies (smooth regions) and a
    dimmer outer ring = high frequencies (edges, texture, noise).
    """
    gray = gray.astype(np.float64)
    F = np.fft.fft2(gray)
    mag = np.abs(np.fft.fftshift(F))
    mag = np.log1p(mag)
    mag -= mag.min()
    mag /= (mag.max() + 1e-12)
    return (mag * 255.0).astype(np.uint8)


def _frequency_grid(shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    """Return 2-D frequency grids (u, v) in cycles-per-sample.

    `u` is a row vector (one value per column), `v` is a column vector (one
    value per row), following NumPy's `fftfreq` convention.  Broadcasting
    `u` and `v` yields the (row, column) frequency of every sample.
    """
    h, w = shape
    u = np.fft.fftfreq(w)
    v = np.fft.fftfreq(h)
    return u[None, :], v[:, None]


def fourier_gradient(gray: np.ndarray) -> np.ndarray:
    """Gradient magnitude computed with the differentiation property of the DFT.

    From the theory: differentiating the inverse-DFT reconstruction
        f(x, y) = (1/MN) * sum F(u, v) * exp(j*2*pi*(u*x + v*y))
    under the integral gives
        df/dx  <->  j * 2*pi*u * F(u, v)
        df/dy  <->  j * 2*pi*v * F(u, v)

    So the gradient (an edge operator) is just the inverse DFT of the spectrum
    multiplied by the ramps (j*2*pi*u, j*2*pi*v).  The result is the magnitude
    of the gradient, large at edges and ~0 in flat regions.
    """
    gray = gray.astype(np.float64)
    u, v = _frequency_grid(gray.shape)
    F = np.fft.fft2(gray)
    fx = np.fft.ifft2(F * (1j * 2.0 * np.pi * u)).real
    fy = np.fft.ifft2(F * (1j * 2.0 * np.pi * v)).real
    return np.hypot(fx, fy)


def gaussian_highpass_transfer(shape: tuple[int, int], cutoff: float) -> np.ndarray:
    """Gaussian high-pass transfer function, H(u, v) = 1 - exp(-D^2 / (2*cutoff^2)).

    D(u, v) = sqrt(u^2 + v^2) is the distance from the DC origin in cycles per
    sample.  A Gaussian *low-pass* is exp(-D^2 / (2*cutoff^2)); subtracting it
    from 1 turns it into a high-pass (keeps high frequencies, removes DC/low).

    The Gaussian is used because its inverse transform is also a Gaussian, so
    it does NOT ring at edges (unlike an ideal hard cut-off).
    """
    h, w = shape
    u, v = _frequency_grid(shape)
    D2 = u * u + v * v
    cutoff = max(float(cutoff), 1e-6)
    return 1.0 - np.exp(-D2 / (2.0 * cutoff * cutoff))


def ideal_highpass_transfer(shape: tuple[int, int], cutoff: float) -> np.ndarray:
    """Ideal high-pass transfer: 1 inside the cut-off circle, 0 outside.

    Sharpest possible cut, but it rings (Gibbs phenomenon) at edges — the
    theory discusses why the Gaussian is preferred in practice.
    """
    u, v = _frequency_grid(shape)
    D = np.sqrt(u * u + v * v)
    return (D >= float(cutoff)).astype(np.float64)


def butterworth_highpass_transfer(
    shape: tuple[int, int], cutoff: float, order: int = 2
) -> np.ndarray:
    """Butterworth high-pass transfer (smooth compromise between ideal and Gaussian)."""
    u, v = _frequency_grid(shape)
    D = np.sqrt(u * u + v * v)
    D = np.maximum(D, 1e-9)
    return 1.0 / (1.0 + (float(cutoff) / D) ** (2 * int(order)))


def highpass_filter(gray: np.ndarray, cutoff: float, kind: str = "gaussian") -> np.ndarray:
    """Apply a high-pass filter in the frequency domain.

    The spectrum F(u, v) is multiplied by a high-pass transfer H(u, v), then
    inverted.  The returned image keeps the edges (high frequencies) and
    suppresses smooth regions (low frequencies).
    """
    gray = gray.astype(np.float64)
    if kind == "ideal":
        H = ideal_highpass_transfer(gray.shape, cutoff)
    elif kind == "butterworth":
        H = butterworth_highpass_transfer(gray.shape, cutoff)
    else:
        H = gaussian_highpass_transfer(gray.shape, cutoff)
    F = np.fft.fft2(gray)
    return np.fft.ifft2(F * H).real


def highpass_edges(gray: np.ndarray, cutoff: float = 0.08, kind: str = "gaussian") -> np.ndarray:
    """Binary edge map from frequency-domain high-pass filtering.

    The high-pass response is thresholded with Otsu's method (a statistical
    threshold, not machine learning) so the strongest high-frequency pixels —
    the boundaries — become a binary mask.
    """
    filtered = highpass_filter(gray, cutoff, kind)
    mag = np.abs(filtered)
    mag = mag - mag.min()
    mag = (mag / (mag.max() + 1e-12) * 255.0).astype(np.uint8)
    _, edges = cv2.threshold(mag, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return edges


def spectral_residual_saliency(gray: np.ndarray, ksize: int = 5) -> np.ndarray:
    """Frequency-domain saliency map (spectral residual, Hou & Zhang 2007).

    A purely classical, Fourier-based way to find the "interesting" object in
    an image, with no learning.  The idea:

        L(f)    = log |F(u, v)|                       (log amplitude spectrum)
        R(f)    = L(f) - mean_filter(L(f))            (spectral residual)
        S(x, y) = | IDFT[ exp(R(f) + j * phase(F)) ] |^2

    Most of the log-amplitude spectrum is a smooth, predictable background
    (the 1/f fall-off common to natural images); subtracting its local average
    leaves the *residual* — the statistically surprising frequencies, which are
    exactly the ones that code the salient object (the person).  The inverse
    transform turns that residual back into a spatial saliency map.
    """
    gray = gray.astype(np.float64)
    F = np.fft.fft2(gray)
    mag = np.abs(F)
    phase = np.angle(F)
    L = np.log(mag + 1e-12)
    # Local average of the log-amplitude (a small mean filter).
    L_avg = cv2.blur(L, (max(3, int(ksize)), max(3, int(ksize))))
    R = L - L_avg
    sal = np.abs(np.fft.ifft2(np.exp(R + 1j * phase))) ** 2
    sal = cv2.GaussianBlur(sal, (0, 0), 4.0)
    return sal


def otsu_binary(arr: np.ndarray) -> np.ndarray:
    """Normalise a float array to uint8 and threshold it with Otsu's method."""
    arr = arr.astype(np.float64)
    arr -= arr.min()
    rng = arr.max() - arr.min()
    if rng > 0:
        arr = arr / rng
    arr = (arr * 255.0).astype(np.uint8)
    _, mask = cv2.threshold(arr, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return mask


def components_overlapping(mask: np.ndarray, seed: np.ndarray, min_fraction: float = 0.05) -> np.ndarray:
    """Keep the connected components of `mask` that overlap `seed` by at least
    `min_fraction` of their own area.

    Used to select, among the saliency mask's components, the one that is a
    human: a salient region is kept only if a meaningful share of its pixels
    are skin-coloured (so a salient stop-sign or tree is discarded).
    """
    mask = (mask > 0).astype(np.uint8)
    seed = (seed > 0).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    out = np.zeros_like(mask)
    for i in range(1, n):
        comp = (labels == i).astype(np.uint8)
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area == 0:
            continue
        frac = float((comp & seed).sum()) / area
        if frac >= min_fraction:
            out = cv2.bitwise_or(out, comp)
    return out * 255


# ======================================================================
# Region detection (colour / intensity)
# ======================================================================


def otsu_figure_ground(gray: np.ndarray) -> np.ndarray:
    """Split the person (foreground) from the background with Otsu's method.

    Otsu's method is a statistical threshold (not machine learning): it picks the
    intensity threshold T that best separates the image's two intensity classes.
    In a typical "person in a scene" photo the person occupies less of the frame
    than the background, so the person is the *minority* class.  Both polarities
    are tried and the smaller class is returned — this finds a dark-clothed
    person against a light background and a light-clothed person against a dark
    background equally well.
    """
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    dark = binary == 0      # pixels at or below T
    bright = binary == 255  # pixels above T
    person = dark if int(dark.sum()) <= int(bright.sum()) else bright
    return person.astype(np.uint8) * 255


def skin_mask(bgr: np.ndarray) -> np.ndarray:
    """Skin-colour mask from a BGR image, using fixed YCrCb thresholds.

    Human skin occupies a narrow, well-known band in the Cr/Cb chrominance
    plane regardless of ethnicity or lighting (chrominance is much more
    illumination-invariant than raw RGB).  The classic range is
        133 <= Cr <= 173  and  77 <= Cb <= 127.
    These are fixed constants — no model is trained and nothing is learned.
    """
    ycrcb = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(ycrcb)
    mask = ((cr >= 133) & (cr <= 173) & (cb >= 77) & (cb <= 127)).astype(np.uint8) * 255
    return mask


def thermal_hot_mask(image: np.ndarray) -> np.ndarray:
    """Hot-body mask from a thermal image.

    A thermal camera renders warmer objects brighter, so the person (near body
    temperature) is bright against the cooler background.  The intensity image
    is thresholded with Otsu's method to split "hot foreground" from "cool
    background".  Colourised thermal images (a jet/iron palette) are converted
    to intensity first; raw grayscale thermal passes through unchanged.
    """
    if image.ndim == 3:
        # A colourised thermal image: intensity = luminance of the palette.
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.astype(np.uint8)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return mask


# ======================================================================
# Morphology / components
# ======================================================================


def clean_mask(mask: np.ndarray, close_ksize: int = 15, open_ksize: int = 7) -> np.ndarray:
    """Morphologically clean a binary mask: close gaps, remove specks, fill holes.

    A closing (dilate then erode) fuses nearby skin/hot patches into one solid
    region; an opening (erode then dilate) removes isolated noise; hole-filling
    closes interior gaps so the boundary is a single clean contour.

    The mask is kept as 0/255 (not 0/1) because the flood-fill hole-filler
    relies on `bitwise_not(255) == 0`; with 0/1 the complement of 1 is 254,
    which the border fill would wrongly keep as foreground.
    """
    mask = (mask > 0).astype(np.uint8) * 255
    close_ksize = max(3, int(close_ksize))
    open_ksize = max(3, int(open_ksize))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_ksize, close_ksize))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_ksize, open_ksize))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # Fill holes: flood-fill the background from the border, then invert so the
    # interior holes (which the fill could not reach) become foreground.
    filled = mask.copy()
    h, w = mask.shape
    ff = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(filled, ff, (0, 0), 255)
    filled = cv2.bitwise_not(filled)  # background -> 0, holes -> 255
    return cv2.bitwise_or(mask, filled)  # person + holes, background -> 0


def largest_component(mask: np.ndarray) -> np.ndarray:
    """Keep only the largest connected component (assumed to be the person)."""
    mask = (mask > 0).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    if n <= 1:
        return mask
    # Label 0 is the background; pick the largest remaining component.
    sizes = stats[1:, cv2.CC_STAT_AREA]
    largest = 1 + int(np.argmax(sizes))
    return (labels == largest).astype(np.uint8) * 255


def boundary_from_mask(mask: np.ndarray, epsilon_ratio: float = 0.004) -> np.ndarray:
    """Return the smoothed boundary contour of a binary mask as an (N, 1, 2) array.

    The contour is the *exact* boundary between foreground and background at
    pixel resolution, then simplified with the Douglas-Peucker algorithm
    (`approxPolyDP`) so a modest number of vertices describe the silhouette
    without chasing single-pixel noise.
    """
    mask = (mask > 0).astype(np.uint8)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return np.empty((0, 1, 2), dtype=np.int32)
    contour = max(contours, key=cv2.contourArea)
    perimeter = cv2.arcLength(contour, True)
    epsilon = epsilon_ratio * max(perimeter, 1.0)
    return cv2.approxPolyDP(contour, epsilon, True)


# ======================================================================
# Overlays (for visualisation)
# ======================================================================


def boundary_overlay(image: np.ndarray, contour: np.ndarray, color=(0, 255, 0), thickness=3) -> np.ndarray:
    """Draw the boundary contour over a copy of `image`."""
    out = image.copy()
    cv2.drawContours(out, [contour], -1, color, thickness, cv2.LINE_AA)
    return out


def mask_overlay(image: np.ndarray, mask: np.ndarray, color=(0, 255, 0), alpha=0.45) -> np.ndarray:
    """Blend a translucent coloured mask over a copy of `image`."""
    out = image.copy()
    if image.ndim == 2:
        out = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    tint = np.zeros_like(out, dtype=np.uint8)
    tint[:] = color
    m = (mask > 0)
    out[m] = (out[m].astype(np.float32) * (1.0 - alpha) + tint[m].astype(np.float32) * alpha).astype(np.uint8)
    return out


# ======================================================================
# Full pipelines
# ======================================================================


def segment_rgb(bgr: np.ndarray) -> dict:
    """Segment the human in an RGB image and return every intermediate.

    Pipeline: Otsu figure-ground threshold -> skin confirmation -> morphology ->
    largest component -> boundary.  The frequency-domain gradient, high-pass edge
    map and spectral-residual saliency are computed in parallel and returned for
    visualisation (saliency is the Fourier-domain analogue of the region cue).

    Otsu figure-ground is the *primary* cue: it splits the person from the
    background by intensity, and because it works on clothing — not just skin —
    it finds a fully dressed subject as one solid region.  Skin colour and
    spectral-residual saliency are *supporting* cues: real skin (face/hands)
    confirms which region is the person and adds back pixels the threshold
    missed, while the saliency map demonstrates that the same foreground can be
    located in the frequency domain.
    """
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    # 1. Primary cue: figure-ground separation by Otsu's method.  The person is
    #    the minority intensity class, found whether it is darker or brighter
    #    than the background.
    mask = otsu_figure_ground(gray)

    # 2. Supporting cue: skin colour, only when genuinely present.  Human skin
    #    occupies a narrow Cr/Cb band; over too little of the frame there is no
    #    useful skin, and over too much of it the "skin" is really a warm print
    #    or skin-toned background — in either case the figure-ground mask alone
    #    is trusted.
    raw_skin = skin_mask(bgr)
    skin_frac = float((raw_skin > 0).mean())
    if 0.005 < skin_frac < 0.30:
        kept = components_overlapping(mask, raw_skin, min_fraction=0.02)
        mask = cv2.bitwise_or(kept, raw_skin)

    # 3. Clean morphologically (close/open/hole-fill) and keep the largest
    #    connected component — the person.
    mask = clean_mask(mask)
    mask = largest_component(mask)

    saliency = spectral_residual_saliency(gray)

    contour = boundary_from_mask(mask)
    gradient = fourier_gradient(gray)
    edges = highpass_edges(gray)

    return {
        "image": bgr,
        "mask": mask,
        "contour": contour,
        "raw_skin": raw_skin,
        "saliency": _to_uint8(saliency),
        "gradient": _to_uint8(gradient),
        "edges": edges,
        "spectrum": log_magnitude_spectrum(gray),
    }


def segment_thermal(image: np.ndarray) -> dict:
    """Segment the human in a thermal image and return every intermediate.

    Pipeline: hot-body threshold -> morphology -> largest component -> boundary,
    with the same frequency-domain edge detection alongside.
    """
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.astype(np.uint8)

    raw_hot = thermal_hot_mask(image)
    mask = clean_mask(raw_hot)
    mask = largest_component(mask)

    contour = boundary_from_mask(mask)
    gradient = fourier_gradient(gray)
    edges = highpass_edges(gray)

    return {
        "image": image,
        "mask": mask,
        "contour": contour,
        "raw_hot": raw_hot,
        "gradient": _to_uint8(gradient),
        "edges": edges,
        "spectrum": log_magnitude_spectrum(gray),
    }


def _to_uint8(arr: np.ndarray) -> np.ndarray:
    """Normalise a float array to [0, 255] uint8 (for display/saving)."""
    arr = arr.astype(np.float64)
    arr -= arr.min()
    rng = arr.max() - arr.min()
    if rng > 0:
        arr = arr / rng
    return (arr * 255.0).astype(np.uint8)


# ======================================================================
# Comparison metrics (classical mask vs SAM2 mask)
# ======================================================================


def mask_metrics(pred: np.ndarray, gt: np.ndarray) -> dict:
    """Compare two binary masks: IoU, Dice, pixel accuracy and boundary F1.

    `pred` is the classical mask, `gt` the SAM2 reference mask (both binary).
    """
    pred = (pred > 0).astype(bool)
    gt = (gt > 0).astype(bool)

    inter = int(np.logical_and(pred, gt).sum())
    union = int(np.logical_or(pred, gt).sum())
    p_sum = int(pred.sum())
    g_sum = int(gt.sum())

    iou = inter / union if union else 0.0
    dice = (2 * inter) / (p_sum + g_sum) if (p_sum + g_sum) else 0.0
    acc = float(np.mean(pred == gt))

    # Boundary F1: compare the boundary bands of the two masks, allowing a
    # few-pixel tolerance (standard for boundary agreement, e.g. BSDS).  Without
    # the tolerance, SAM2's pixel-precise boundary and the classical method's
    # slightly smoothed contour would be scored as a near-total miss even when
    # they track the same silhouette.
    def band(m: np.ndarray, tol: int = 2) -> np.ndarray:
        m = m.astype(np.uint8)
        dilated = cv2.dilate(m, np.ones((3, 3), np.uint8))
        boundary = (dilated.astype(bool) ^ m.astype(bool)).astype(np.uint8)
        if tol > 0:
            k = 2 * tol + 1
            boundary = cv2.dilate(boundary, np.ones((k, k), np.uint8))
        return boundary.astype(bool)

    bp, bg = band(pred), band(gt)
    tp = int(np.logical_and(bp, bg).sum())
    fp = int(np.logical_and(bp, np.logical_not(bg)).sum())
    fn = int(np.logical_and(bg, np.logical_not(bp)).sum())
    f1 = (2 * tp) / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0

    return {
        "iou": iou,
        "dice": dice,
        "pixel_accuracy": acc,
        "boundary_f1": f1,
        "pred_pixels": p_sum,
        "gt_pixels": g_sum,
    }
