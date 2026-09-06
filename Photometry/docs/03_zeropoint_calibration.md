# 3. Zero-Point Calibration

This stage converts raw instrumental magnitudes (which have an arbitrary,
meaningless zero point) into real, standard magnitudes, by comparing
detected stars against the Pan-STARRS DR2 catalog. Everything here lives
in `steps_zeropoint.py`.

## Why this is needed

Aperture photometry gives you an *instrumental* magnitude
(`-2.5 * log10(flux)`) — this tells you the *relative* brightness of stars
in your own image correctly, but the absolute number has no external
meaning; it depends on your exposure time, telescope aperture, camera gain,
etc. To turn this into a real magnitude that can be compared with anyone
else's data, you need a **zero-point**: an offset that, when added to the
instrumental magnitude, gives you the true magnitude.

real_mag = inst_mag + zero_point


The zero-point is found by comparing your own instrumental magnitudes
against known, already-calibrated magnitudes from an external catalog, for
stars you can identify in both.

## Why Pan-STARRS, and why one query per field

Pan-STARRS DR2 is free, well-documented, and covers the whole sky visible
from the northern hemisphere at reasonable depth — a good match for this
kind of ground-based imaging.

The catalog is queried **once per image**, covering the whole field, rather
than once per individual star. This is both faster (one network request
instead of dozens) and more consistent (every star in the image is
compared against the same catalog snapshot). This scales correctly whether
an image has a handful of sources (a sparse standard-star field) or
hundreds (a dense cluster field) — the number of queries doesn't grow with
the number of stars.

## `query_field_catalog(ra_center, dec_center, radius_arcmin, ps1_filter='g')`

Queries Pan-STARRS DR2 for all catalog sources within `radius_arcmin` of
the given field center.

| Parameter | Default | Meaning |
|---|---|---|
| `ra_center` / `dec_center` | required | Field center, in degrees |
| `radius_arcmin` | required | Search radius (should comfortably cover your actual field of view) |
| `ps1_filter` | `'g'` | Which Pan-STARRS band to pull (`'g'`, `'r'`, `'i'`, `'z'`, `'y'`) — should match the filter your image was taken in |

**Returns:** an astropy `Table` of catalog sources, or `None` if the query
fails or returns nothing. Sources with no valid measurement in the
requested band (Pan-STARRS uses `-999` as a "no data" placeholder) are kept
in this raw result — they're filtered out later during matching.

## `match_sources_to_catalog(photometry_table, catalog_table, ps1_filter='g', max_sep_arcsec=1.0)`

Cross-matches your detected sources (by RA/Dec) against the catalog,
keeping only sources with a good positional match.

| Parameter | Default | Meaning |
|---|---|---|
| `photometry_table` | required | Your photometry table — must have `RA` and `Dec` columns |
| `catalog_table` | required | Result from `query_field_catalog` |
| `max_sep_arcsec` | `1.0` | Maximum separation allowed for a match to be accepted |

**Returns:** a Table with `x`, `y`, `RA`, `Dec`, `inst_mag`, `catalog_mag`
for matched sources only, or `None` if nothing matched.

## `compute_zeropoint(matched_table, sigma=3.0)`

Computes one robust zero-point from all matched sources.

For each matched star: `offset = catalog_mag - inst_mag`. These offsets are
then passed through `astropy`'s `sigma_clipped_stats`, which automatically
discards outliers (mismatches, blended stars, variable stars) before
averaging — the same statistical approach used in professional pipelines
for this exact step.

| Parameter | Default | Meaning |
|---|---|---|
| `sigma` | `3.0` | Sigma-clipping threshold |

**Returns:** `(zp, zp_sigma, n_used)` — the zero-point, its uncertainty,
and how many matches survived clipping. Returns `(None, None, 0)` if
nothing usable was found.

## `apply_zeropoint(photometry_table, zp)`

Adds a `real_mag` column to a photometry table: `real_mag = inst_mag + zp`.
Applied to **every** detected source in the image, not just the ones that
happened to match the catalog.

## Typical usage

```python
from steps_zeropoint import (
    query_field_catalog, match_sources_to_catalog,
    compute_zeropoint, apply_zeropoint
)
import numpy as np

ra_center = float(np.mean(table["RA"]))
dec_center = float(np.mean(table["Dec"]))

catalog = query_field_catalog(ra_center, dec_center, radius_arcmin=2.0, ps1_filter='g')
matched = match_sources_to_catalog(table, catalog, ps1_filter='g', max_sep_arcsec=1.0)
zp, zp_sigma, n_used = compute_zeropoint(matched)
table = apply_zeropoint(table, zp)
```

## Sanity-checking your results

Since dithered exposures of the same field should all produce nearly
identical zero-points, a useful sanity check after running a batch is to
compare the zero-point across all images — they should cluster tightly
together (typically within a few tenths of a magnitude). Wildly scattered
zero-points across a batch usually indicate a problem upstream (bad
astrometry, contaminated detections, wrong filter selected).

See `notebooks/03_zeropoint_calibration.ipynb` for a worked example,
including inspecting the matched stars and the final calibrated magnitudes.

