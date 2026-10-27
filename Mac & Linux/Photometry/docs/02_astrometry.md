# 2. Astrometry (Sky Coordinate Solving)

This stage figures out exactly where each image is pointing on the sky —
producing a WCS (World Coordinate System) that lets every pixel be
converted into a real RA/Dec. Everything here lives in `steps_astrometry.py`.

## The three-tier system

Each image is solved **independently** (not shared across a batch), since
dithered exposures point at slightly different sky positions. Solving
happens in up to three tiers, only escalating if the previous tier fails
on **every single image** in the batch:

Tier 1: Automatic, blind (astrometry.net, no position hint)
│ fails on ALL images?
▼
Tier 2: Automatic, guided (astrometry.net, with a position hint
│ from one manually-identified star)
│ still fails on ALL images?
▼
Tier 3: Fully manual (fit a WCS from several manually
identified stars, reference image only)


If Tier 1 succeeds on at least one image, Tiers 2 and 3 are never offered —
they only exist as a safety net for a batch that fails completely (e.g. bad
data, wrong scale hint, or a dataset that accidentally includes dark/blank
frames).

## Tier 1: `solve_with_astrometry_net(...)`

Sends your already-detected star positions (not the whole image) to
astrometry.net's plate-solving service.

| Parameter | Default | Meaning |
|---|---|---|
| `sources` | required | Table of detected stars (from `find_stars`) |
| `image_shape` | required | `(height, width)` of the image in pixels |
| `api_key` | required | Your astrometry.net API key (from `config.py`) |
| `scale_lower_arcmin` / `scale_upper_arcmin` | `3.0` / `3.6` | Expected field-of-view width range, in arcmin. **Must match your current optical setup** — see `config.py`. A wrong value here is the most common cause of solve failures. |
| `ra` / `dec` | `None` | Optional position hint (degrees) — used automatically by Tier 2, not normally set by hand |
| `search_radius_arcmin` | `None` | Search radius around the `ra`/`dec` hint, if given |
| `solve_timeout` | `300` | Max seconds to wait for a single solve attempt |
| `max_retries` | `2` | Automatic retries if the failure looks like a transient network/connection issue (not retried for genuine "could not solve" results, since resending identical data won't change that outcome) |

**Returns:** an `astropy.wcs.WCS` object on success, or `None` on failure.

If a solve fails, `print_solve_failure_help(...)` is called automatically,
printing tailored suggestions depending on whether it looks like a network
issue or a genuine solving problem (too few sources, wrong scale hint,
noisy detections, etc.).

## Tier 2: guided retry with a position hint

If every image failed under Tier 1, the pipeline offers to let you identify
one star in the reference image and provide its known RA/Dec. This narrows
astrometry.net's search dramatically, using:

- `review_astrometric_candidates(...)` / `get_position_hint_from_user(...)`
  — shows a numbered plot of candidate stars and asks which one you can
  identify

| Parameter | Meaning |
|---|---|
| `image_data` | The reference image's pixel data |
| `candidates` | The candidate star table to choose from |

**Returns:** `(ra_deg, dec_deg)` as plain floats, or `None` if no valid
selection was made.

The search radius for this retry is computed automatically as
`SCALE_UPPER_ARCMIN × 4` — it scales with your current optical setup
rather than a fixed number, so it stays correct even if your field of view
changes (e.g. a focal reducer is added later).

## Tier 3: fully manual astrometry

If Tier 2 also fails on every image, the pipeline offers full manual
astrometry for the reference image: you identify several stars and type in
their known RA/Dec, and a WCS is fitted from those points.

### `review_astrometric_candidates(found_stars, image_data, fwhm, ...)`

Interactive loop: shows a numbered plot of candidate stars, lets you reject
ones that look like hot pixels or artifacts rather than real stars, and
reselects fresh candidates to replace rejected ones. Repeats until you
confirm you're happy with the list.

| Parameter | Default | Meaning |
|---|---|---|
| `n_candidates` | `10` | How many candidates to show per round |
| `merge_factor` | `1.5` | Multiplier on FWHM for merging duplicate detections |
| `edge_factor` | `2.0` | Multiplier on FWHM for excluding stars too close to the image edge |

### `run_manual_astrometry(image_data, candidates, min_stars=3, warn_threshold_arcsec=1.0)`

Runs the full manual entry + fit + quality-check process.

| Parameter | Default | Meaning |
|---|---|---|
| `min_stars` | `3` | Minimum number of stars required before a fit is allowed. Fewer than 3 gives an unreliable, unverifiable fit. |
| `warn_threshold_arcsec` | `1.0` | If the fit's RMS residual (see below) exceeds this, you're warned and offered a chance to redo entry |

**What happens internally:**
1. Shows the candidate plot
2. For each star, asks for RA (`HH:MM:SS.ss`) and Dec (`+/-DD:MM:SS.s`) —
   retries automatically if a single entry can't be parsed
3. Shows a summary of everything entered and lets you redo specific stars
   by number before committing
4. Fits a WCS from the confirmed points
5. **Checks fit quality**: runs each star's pixel position back through the
   fitted WCS and compares to what you typed. If the RMS residual exceeds
   `warn_threshold_arcsec`, you're warned (this usually means one entry had
   a typo) and can choose to redo the whole entry or accept the fit anyway

**Returns:** an `astropy.wcs.WCS` object, or `None` if not enough stars were
available.

## Typical usage (automatic path)

```python
from steps_astrometry import solve_with_astrometry_net
from config import ASTROMETRY_API_KEY, SCALE_LOWER_ARCMIN, SCALE_UPPER_ARCMIN

wcs = solve_with_astrometry_net(
    sources=found_stars,
    image_shape=image_data.shape,
    api_key=ASTROMETRY_API_KEY,
    scale_lower_arcmin=SCALE_LOWER_ARCMIN,
    scale_upper_arcmin=SCALE_UPPER_ARCMIN
)
```

See `notebooks/02_astrometry.ipynb` for a worked example of all three
tiers, including what a failed solve and the fallback prompts look like.


