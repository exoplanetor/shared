# 5. Running the Full Pipeline

This page walks through actually launching `photometry_pipeline.py` from
start to finish — what you'll be asked, what each prompt means, and what
to do when things go wrong.

## Before you start

- Confirm `config.py` has your real astrometry.net API key and the correct
  scale hint for your current optical setup (see `00_overview.md`)
- Have your folder of **calibrated science FITS images** ready. Do not
  point the pipeline at a folder containing dark/bias/flat frames — these
  contain no real stars and cannot be plate-solved (see
  [Troubleshooting](#troubleshooting) below)

## Running it
python photometry_pipeline.py


## Step-by-step prompts

### 1. Image folder
Enter full path to folder containing calibrated FITS images:
Type the full path to your folder of science images (e.g.
`C:\Users\Data\test3\calibrated`).

### 2. Detection & cleaning settings
Adjust source detection and cleaning parameters? (y/n):

Type `n` to use sensible defaults (recommended for routine use). Type `y`
only if you need to fine-tune detection for unusual data — see
`01_detection_and_fwhm.md` for what each parameter means.

### 3. Astrometry method

astrometry.net (automatic, uses saved API key)
manual (type RA/Dec for a few stars)
Choose method [1/2] (default 2):

Type `1` for normal use — this solves every image automatically. Type `2`
only if you already know you want to skip automatic solving entirely (e.g.
no internet access at the telescope).

### 4. Automatic detection & solving (no input needed)

The pipeline now runs through every image: detecting stars, measuring
FWHM, and solving astrometry. This can take anywhere from seconds to a
couple of minutes per image, depending on astrometry.net's server load.
You'll see a `Solving.......` progress indicator followed by a
success/failure message for each image.

### 5. If automatic solving fails on every image (fallback)

If — and only if — every single image fails to solve automatically, you'll
see:
astrometry.net failed on all N images.
Identify one star to provide a position hint and retry? (y/n):


Type `y` to try a guided retry: you'll be shown a plot of candidate stars
in the reference image, asked to pick one you can identify, and asked for
its known RA/Dec. This narrows the search and retries all images.

If that **also** fails on every image, you'll be offered one last option:


Type `y` to try a guided retry: you'll be shown a plot of candidate stars
in the reference image, asked to pick one you can identify, and asked for
its known RA/Dec. This narrows the search and retries all images.

If that **also** fails on every image, you'll be offered one last option:


Type `y` to try a guided retry: you'll be shown a plot of candidate stars
in the reference image, asked to pick one you can identify, and asked for
its known RA/Dec. This narrows the search and retries all images.

If that **also** fails on every image, you'll be offered one last option:

Switch to manual astrometry for the reference image instead? (y/n):


Type `y` to fit a WCS by hand — see `02_astrometry.md` for full details on
this process (identifying multiple stars, typing their RA/Dec, and the
fit-quality check).

### 6. Zero-point calibration (no input needed)

Runs automatically for every successfully-solved image: queries Pan-STARRS,
cross-matches, computes the zero-point, and applies it. You'll see a
`Zero-point: XX.XX +/- X.XX (from N matches...)` line per image.

### 7. Results

Each image's final table is saved to
`<your_folder>/photometry_results/<image_name>_phot.fits`, and the full
tables print to the terminal at the end.

## Troubleshooting

**"astrometry.net failed on all images" and the data actually contains real
stars:**
- Check your scale hint in `config.py` matches your current optical setup
- Check your API key is valid
- Try again — astrometry.net's free service is occasionally slow/unstable

**Detected "stars" with only 1 candidate and a huge FWHM (100+ pixels):**
This is a strong sign the image doesn't actually contain real starlight —
most commonly, a **dark frame, bias frame, or flat field** accidentally
included in the input folder. Double-check your folder only contains
science exposures.

**A solve fails with a `RemoteDisconnected` / connection error message:**
This is a transient network issue, not a data problem. The pipeline
automatically retries a couple of times; if it still fails, simply run
again.

**Zero-points vary wildly across images in the same batch:**
Since dithered exposures of one field should produce nearly identical
zero-points, large scatter usually points to a problem upstream — check
individual images' astrometry solved correctly and that detections aren't
contaminated by artifacts.

See `notebooks/05_running_the_full_pipeline.ipynb` for a complete worked
run-through on real data.

