# 1. Star Detection & FWHM Measurement

This stage finds stars in a single FITS image and measures how sharp they
are (FWHM — Full Width at Half Maximum, in pixels). Everything here lives in
`steps_photometry.py`.

## Why this matters

Later pipeline steps (aperture sizing, astrometric candidate selection) all
depend on knowing roughly how big a star's light blob is in your image. A
telescope with poor focus or bad seeing will have a large FWHM (blurry,
wide stars); a well-focused image will have a small FWHM (tight, sharp
stars).

## `find_stars(...)`

Detects stars using `photutils`' DAOStarFinder algorithm, with an optional
cleanup pass to remove hot pixels and cosmic rays.

| Parameter | Type | Default | Meaning |
|---|---|---|---|
| `image_path` | str | required | Path to the FITS file |
| `threshold_factor` | float | `5.0` | How bright (in units of background noise σ) a peak must be to count as a star. Higher = fewer, brighter-only detections. Typical range: `3.0`–`8.0`. |
| `apply_filter` | bool | `True` | Whether to run the extra footprint-based cleanup filter (recommended: keep `True`) |
| `filter_params` | dict or `None` | `None` | Custom settings for the cleanup filter (see below). `None` uses sensible defaults. |
| `fwhm_override` | float or `None` | `None` | If you already know the real FWHM (e.g. from a previous pass), pass it here for more accurate detection. Leave `None` on the first pass. |

**A note on the two-pass design:** DAOStarFinder needs an approximate FWHM
*before* it can detect anything — but you don't actually know the real FWHM
until you've already detected some stars and measured it. So the first call
uses a rough guess (3.0 px), and a second call can pass in the real
measured value via `fwhm_override` for more accurate results. The main
pipeline handles this automatically — you don't need to think about it when
running `photometry_pipeline.py`.

### Cleanup filter parameters (`filter_params`)

Passed as a dictionary, e.g. `{"threshold_sigma": 3, "min_area": 10, ...}`.
All are optional — omit any you don't want to override.

| Parameter | Default | Meaning |
|---|---|---|
| `threshold_sigma` | `3` | Background sigma threshold used to build the source mask for this filter step |
| `smoothing_sigma` | `0` | Gaussian smoothing applied before filtering (0 = none) |
| `min_area` | `10` | Minimum footprint size (pixels) before a detection is considered a potential hot pixel |
| `connectivity` | `8` | Pixel connectivity for grouping (4 or 8) |
| `peak_to_total_thresh` | `0.2` | If a small detection's peak brightness dominates its total flux above this fraction, it's flagged as a likely hot pixel |
| `hot_pixel_sigma_thresh` | `1e10` | Absolute peak brightness (in σ) above which a small detection is always flagged, regardless of shape |
| `merge_close` | `True` | Whether to merge multiple detections that are actually the same star |
| `merge_radius` | `10.0` | Distance (pixels) within which nearby detections get merged, keeping only the brightest |

## `estimate_fwhm_ensemble(stars, image_data, n_samples=10, cutout_size=40)`

Takes the brightest `n_samples` detected stars, fits a 2D Gaussian to each
one's pixel cutout, and returns the median FWHM across all successful fits.

| Parameter | Default | Meaning |
|---|---|---|
| `n_samples` | `10` | How many of the brightest stars to use for the FWHM measurement |
| `cutout_size` | `40` | Size (pixels) of the square cutout around each star used for the Gaussian fit |

**Returns:** a tuple `(fwhm, n_used)` — the median FWHM in pixels, and how
many stars were successfully used. Returns `(None, 0)` if no stars could be
measured (e.g. empty detection list, or an image with no real stars such as
a dark frame).

> **Known gotcha:** if `stars` is `None` or empty, this function currently
> returns a bare `None` instead of a `(None, 0)` tuple, which will crash
> code expecting to unpack two values. If you're calling this function
> directly (outside the main pipeline), check for `None` before unpacking.

## Typical usage

```python
from steps_photometry import find_stars, estimate_fwhm_ensemble

found_stars = find_stars(
    image_path="my_image.fits",
    threshold_factor=6.0,
    apply_filter=True
)

fwhm, n_used = estimate_fwhm_ensemble(found_stars, image_data)
print(f"FWHM: {fwhm:.2f} px, from {n_used} stars")
```

See `notebooks/01_detection_and_fwhm.ipynb` for a worked example on a real
image, including a plot of the detections.

