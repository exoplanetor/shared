# Pipeline Overview

## What this pipeline does

This pipeline takes a folder of calibrated FITS images and turns them into a
table of real, catalog-calibrated star magnitudes. In short:

1. Detect stars in each image
2. Measure how sharp the stars are (FWHM), used to tune later steps
3. Figure out where on the sky each image is actually pointing (astrometry) —
   automatically via astrometry.net, or manually if that fails
4. Convert star pixel positions into real RA/Dec coordinates
5. Compare detected stars against the Pan-STARRS DR2 catalog to calibrate
   instrumental magnitudes into real, standard magnitudes
6. Save the final results to disk as a FITS table, per image

The pipeline is designed to run on a batch of images at once (e.g. a full
night's dithered exposures of one target), producing one output file per
image.

## The files

| File | Role |
|---|---|
| `photometry_pipeline.py` | The main script you actually run. Orchestrates all the steps below in order. |
| `steps_photometry.py` | Star detection, FWHM measurement, aperture photometry, and saving final results. |
| `steps_astrometry.py` | Everything related to figuring out where each image points on the sky (WCS). |
| `steps_zeropoint.py` | Pan-STARRS catalog querying, cross-matching, and zero-point calibration. |
| `config.py` | Contains telescope-specific settings. Edit this before running. |

## How data flows through the pipeline

FITS images
│
▼
Detected sources (x, y, flux) <- steps_photometry.find_stars
│
▼
Measured FWHM <- steps_photometry.estimate_fwhm_ensemble
│
▼
WCS (sky pointing solution) <- steps_astrometry (3 possible methods, see 02_astrometry.md)
│
▼
RA / Dec per source <- steps_photometry.build_photometry_table
│
▼
Pan-STARRS catalog match <- steps_zeropoint.match_sources_to_catalog
│
▼
Zero-point (mag offset) <- steps_zeropoint.compute_zeropoint
│
▼
Real, calibrated magnitudes <- steps_zeropoint.apply_zeropoint
│
▼
Saved FITS table per image <- steps_photometry.save_photometry_table



## Requirements

Python packages needed:

- `numpy`, `scipy`
- `astropy`
- `photutils`
- `astroquery`
- `matplotlib`
- `requests`

Install with pip if any are missing:

pip install numpy scipy astropy photutils astroquery matplotlib requests


## One-time setup

Before running the pipeline for the first time, open `config.py` and make
sure it contains:

- Your own astrometry.net API key (free account at
  [nova.astrometry.net](https://nova.astrometry.net))
- The correct scale hint (`SCALE_LOWER_ARCMIN`, `SCALE_UPPER_ARCMIN`) for
  your current optical setup — this changes if a focal reducer or other
  optic is added/removed

## Where to go next

- [`01_detection_and_fwhm.md`](01_detection_and_fwhm.md) — how stars are found and measured
- [`02_astrometry.md`](02_astrometry.md) — how the pipeline figures out sky coordinates
- [`03_zeropoint_calibration.md`](03_zeropoint_calibration.md) — how magnitudes get calibrated
- [`04_saving_results.md`](04_saving_results.md) — what gets saved and how to read it back
- [`05_running_the_full_pipeline.md`](05_running_the_full_pipeline.md) — a full walkthrough of actually running it

Each doc page has a matching notebook in `notebooks/` with a hands-on worked
example.
