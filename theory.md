# Finding a Human's Boundary with the Fourier (Frequency) Domain

**Module 4 — GSU Computer Vision**

This note explains, with derivations, how **edge detection** and **region
segmentation** can be carried out in the frequency domain, and how those ideas
are used to find the exact boundary of a human in a regular (RGB) and a thermal
camera image — using only classical signal processing (no deep learning, no
machine learning). All math is written in plain ASCII.

---

## 1. An image as a sum of waves

A digital image is a grid of numbers, `f[x, y]`.  The 2-D discrete Fourier
transform (DFT) rewrites the image as a sum of sinusoidal wave patterns:

    F[u, v] = sum over x,y of  f[x, y] * exp( -j * 2*pi * (u*x/M + v*y/N) )

and the inverse returns exactly the image:

    f[x, y] = (1/MN) * sum over u,v of  F[u, v] * exp( +j * 2*pi * (u*x/M + v*y/N) )

Here `j` is the imaginary unit (`j^2 = -1`), `M, N` are the image height and
width, and `u, v` are the spatial frequencies (cycles per sample).  A small
`(u, v)` near the origin is a **low** frequency — a smooth, slowly-changing
pattern.  A large `(u, v)` is a **high** frequency — a rapid change.  The
single number `F[u, v]` is complex: its magnitude `|F[u, v]|` says how much of
that wave is present, and its angle says the wave's phase (shift).

Two facts follow immediately, and they are the whole point of this note:

- **Smooth regions are low-frequency.**  A flat, slowly-changing patch of skin,
  or a wall, is described almost entirely by small `u, v`.
- **Edges are high-frequency.**  A sharp boundary is a rapid change from one
  value to another, which needs large `u, v` to be represented.

---

## 2. An edge is a high-frequency event

To make this precise, take a 1-D ideal **step edge** — the value jumps from `a`
to `b` at `x = 0`.  Its derivative is an impulse (a delta) at the jump:

    d/dx  step(x)  =  (b - a) * delta(x)

The delta has a flat spectrum: `DFT{delta(x)} = 1` for every frequency (a delta
is a single spike, and a spike contains *all* frequencies equally).  Therefore,
by the differentiation property (derived next), the step's transform is

    DFT{step(x)}  =  (b - a) / (j * 2*pi * u)   +   (a mean term at u = 0)

so its magnitude falls off as `|1/u|`:

    |F[u]|  ~  |b - a| / (2*pi*|u|)

**Interpretation.**  A step edge has energy at every frequency, but it *decays*
only slowly with frequency (like `1/u`), whereas a smooth region decays much
faster.  So, relative to smooth areas, an edge is rich in high frequencies.
Removing the low frequencies of an image therefore *keeps* the edges and
*discards* the smooth interiors — exactly what an edge detector should do.

---

## 3. The differentiation property (a gradient is a high-pass filter)

This is the central result that links "find an edge" to "multiply by a
frequency-dependent filter".

Start from the inverse DFT (2-D) and differentiate under the sum with respect
to `x`:

    f[x, y] = (1/MN) * sum F[u, v] * exp( j*2*pi*(u*x/M + v*y/N) )

    d/dx f  = (1/MN) * sum F[u, v] * ( j*2*pi*u/M ) * exp( j*2*pi*(u*x/M + v*y/N) )

Each term just picks up the factor `j*2*pi*u/M`.  Comparing with the definition
of the inverse DFT, the sum on the right is the inverse transform of
`j*2*pi*u/M * F[u, v]`.  Hence the **differentiation property**:

    DFT{  df/dx  }  =  j * (2*pi*u/M) * F[u, v]

    DFT{  df/dy  }  =  j * (2*pi*v/N) * F[u, v]

That is: **differentiation in the spatial domain is multiplication by a ramp**
`j*2*pi*u` **in the frequency domain.**  The factor grows linearly with
frequency, so differentiation is a *high-pass* operation — it suppresses the
low frequencies (flat regions, where `df/dx = 0`) and amplifies the high ones
(edges).  This is why the gradient operator — the classic edge detector — is,
in the frequency domain, nothing more than multiplying the spectrum by
`(j*2*pi*u, j*2*pi*v)` and inverting.

The **gradient magnitude** (used in `segmentation.fourier_gradient`) is then

    |grad f| = sqrt( (df/dx)^2 + (df/dy)^2 )

where `df/dx` and `df/dy` are the inverse transforms of the ramp-scaled
spectrum.  It is large exactly at boundaries and ~0 in flat regions.

---

## 4. The Laplacian in the frequency domain (second derivative)

Differentiating twice is multiplying by the ramp twice:

    DFT{ d^2 f / dx^2 }  =  ( j*2*pi*u/M )^2 * F  =  -(2*pi*u/M)^2 * F

    DFT{ d^2 f / dy^2 }  =  -(2*pi*v/N)^2 * F

Adding gives the Laplacian (the sum of second derivatives):

    DFT{  laplacian f  }  =  -( (2*pi*u/M)^2 + (2*pi*v/N)^2 ) * F
                          =  -(2*pi)^2 * ( (u/M)^2 + (v/N)^2 ) * F

The multiplier is a **negative parabola** in frequency: it is ~0 at the origin
(low frequencies) and very negative at high frequencies.  The Laplacian is thus
also a high-pass filter, but with a *quadratic* weighting — it emphasizes high
frequencies even more strongly than the first-derivative ramp.  A zero crossing
of the Laplacian marks an edge, which is the basis of the Marr–Hildreth and
Laplacian-of-Gaussian (LoG) detectors.  Convolving with a Gaussian first (a
low-pass, see Module 3) then taking the Laplacian gives a **band-pass** filter:
the Gaussian removes high-frequency noise, the Laplacian removes the low
frequencies, and what survives is the edges at one chosen scale.

---

## 5. High-pass filters that extract edges

An explicit frequency-domain edge detector builds a transfer function
`H(u, v)` that passes high frequencies and blocks low ones, then computes

    edge_map  =  | IDFT{ H(u, v) * F(u, v) } |

Three standard high-pass filters, each `H = 1 - (a low-pass)`, are used here.
Let `D(u, v) = sqrt( (u/M)^2 + (v/N)^2 )` be the distance from the DC origin:

- **Ideal high-pass** (a hard cut):  `H = 1` for `D >= D0`, else `0`.
  It is perfectly sharp, but truncating the spectrum abruptly causes
  **ringing** near edges — the Gibbs phenomenon (see Section 7).

- **Gaussian high-pass**: using the result from Module 3 that a Gaussian is its
  own Fourier transform,

        H(u, v) = 1 - exp( -D^2 / (2*D0^2) )

  Because the Gaussian decays smoothly, its inverse transform does **not**
  ring, which makes it the preferred choice in practice.

- **Butterworth high-pass** (order `n`), a smooth compromise:

        H(u, v) = 1 / ( 1 + (D0 / D)^(2n) )

These are implemented in `segmentation.gaussian_highpass_transfer`,
`ideal_highpass_transfer`, `butterworth_highpass_transfer` and applied in
`segmentation.highpass_edges`.

---

## 6. Segmenting regions in the frequency domain

Segmentation — splitting the image into "the object" and "everything else" —
can also be understood in frequency terms, as two complementary operations:

**Regions are low-frequency.**  The *interior* of an object (the person's body,
a uniform background) is smooth and homogeneous, so it lives at low
frequencies.  A **low-pass** filter (multiply `F` by a Gaussian centred at the
origin and invert) smooths the image, removing texture and noise and making
each region *more* uniform.  After low-pass homogenisation, a simple intensity
or colour **threshold** cleanly separates region from background.  This is the
frequency-domain picture of region-based segmentation: `low-pass -> threshold`.

**Boundaries are high-frequency.**  The contour separating regions is a rapid
change, so it lives at high frequencies and is extracted by the high-pass
filters of Section 5.  This is boundary-based segmentation.

**Both together.**  A complete segmentation therefore combines the two: the
high-pass response (Section 5) and the gradient/Laplacian (Sections 3–4) give
the *boundary*, while the low-pass-smoothed image gives the *region* whose
contour that boundary is.  In this module the region is found by
colour/temperature thresholding after smoothing, and the exact boundary is the
contour traced around that region.

**Spectral residual — a purely frequency-domain saliency map.**  A second,
very elegant frequency-domain segmentation tool is the *spectral residual*
(Hou & Zhang, 2007), which finds the "interesting" object directly.  Take the
log-amplitude spectrum and subtract its own local average:

    L(f)  = log |F(u, v)|
    R(f)  = L(f)  -  mean_filter( L(f) )
    S(x,y) = | IDFT{ exp( R(f) + j*phase(F) ) } |^2

Most of `log|F|` is a smooth, predictable `1/f` fall-off shared by nearly all
natural images; that predictable part carries little information.  Subtracting
it leaves the **residual** — the frequencies that are statistically surprising,
which are precisely those of the salient object (the person).  Inverting the
residual reconstructs a *saliency map* `S(x, y)` that highlights the object and
suppresses the background.  This is used in `segmentation.spectral_residual_saliency`
to outline the whole body (clothing included), which plain skin-colour
thresholding alone would miss.

---

## 7. Why hard cut-offs ring (Gibbs phenomenon)

A step edge's spectrum decays only as `1/u`, so representing the sharp jump
needs *every* frequency.  If a high-pass (or low-pass) filter cuts the spectrum
off abruptly at `D0`, the reconstructed edge is missing its high-frequency
terms, and it **overshoots**: ripples appear on both sides of the jump.  This
is the Gibbs phenomenon.  It happens for any hard truncation of a discontinuous
signal's spectrum, and it is why the ideal high-pass filter (Section 5) is
rarely used: the Gaussian and Butterworth filters decay smoothly, so they avoid
ringing while still suppressing the unwanted band.  This is the same reason the
Gaussian (and not a box) was the safe blur kernel in Module 3.

---

## 8. Putting it into practice (this module)

The scripts implement exactly the operations derived above:

**RGB image** (`rgb_boundary.py`):

1. **Skin colour** — the person's skin occupies a narrow, illumination-stable
   band in the YCrCb chrominance plane (`133 <= Cr <= 173`, `77 <= Cb <= 127`),
   giving the *region* cue (a fixed threshold, no model).
2. The skin mask is **cleaned morphologically** (close, open, hole-fill) and the
   **largest connected component** is kept — the person.
3. If the person is fully clothed (leaving only a small face/hands skin mask),
   **spectral-residual saliency** (Section 6) — a frequency-domain map of the
   whole salient body — is used as a *fallback* to grow the mask to the full
   body, clothing included.
4. The mask's **contour** is the exact boundary, smoothed with Douglas–Peucker.
5. Alongside, the **frequency-domain gradient** (Section 3) and **high-pass
   edges** (Section 5) are computed to visualise the boundary.

**Thermal image** (`thermal_boundary.py`):

1. A thermal camera renders warmer objects brighter, so the person (near body
   temperature) is bright against the cooler background.  The intensity image
   is thresholded with **Otsu's method** (a statistical threshold, not machine
   learning) to isolate the hot body — the region cue.
2. The same morphological cleanup, largest-component and contour steps give the
   boundary, and the frequency-domain gradient/edges are computed alongside.

---

## 9. Comparison against SAM2 (the deep-learning baseline)

The classical result is compared against Meta's **Segment Anything Model 2**
(SAM2), a deep-learning segmentation model, as the reference.  For each test
image the two binary masks (classical vs SAM2) are compared with the standard
metrics (`segmentation.mask_metrics`, `compare_sam2.py`):

    IoU   = |pred & gt| / |pred | gt|             (intersection over union)
    Dice  = 2|pred & gt| / (|pred| + |gt|)        (F1 of the masks)
    pixel accuracy = fraction of pixels where pred == gt
    boundary F1    = F1 of the boundary bands of the two masks (2-pixel tolerance)

The classical pipeline reaches a useful boundary but is expected to lag SAM2 on
IoU/Dice — SAM2 is a large learned model, while the classical method uses only
fixed thresholds and linear frequency-domain filters.  The gap between the two
is itself the finding: it measures how much of "find the exact boundary" is
learnable, versus what a handful of deterministic Fourier-domain operations can
achieve with no training data at all.

Measured on the two test images (SAM2 prompted with the classical bounding box):
the RGB pipeline reaches **IoU 0.70 / Dice 0.82** against SAM2 (the skin-colour
cue is strong there), and the thermal pipeline reaches **IoU 0.43 / Dice 0.60**
(the hot-body region is coarser).  Both are far below a perfect 1.0, which is
exactly the point: SAM2's learned boundaries are sharper, but a handful of
deterministic Fourier-domain operations already recover the person's silhouette
with no training data at all.
