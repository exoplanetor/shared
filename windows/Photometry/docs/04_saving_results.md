# 4. Saving Results

This stage writes each image's final photometry table to disk, following
the same convention used by professional ground-based pipelines (ESO,
WFST, ZTF): a FITS binary table with the zero-point and data provenance
stored directly in the header. Lives in `steps_photometry.py`.

## Why FITS, and why store metadata in the header

Printing results to the terminal only gives you a temporary view — nothing
is left once the program exits. Saving as FITS with standardized header
keywords means:

- The zero-point travels **permanently with the data** — anyone opening
  the file later knows exactly what calibration was applied, without
  needing separate notes
- Provenance (which raw file this came from, when it was processed) is
  preserved automatically
- The format is directly readable by `astropy`, TOPCAT, and other standard
  astronomy tools

## `save_photometry_table(...)`

| Parameter | Meaning |
|---|---|
| `photometry_table` | The final per-image photometry table |
| `output_dir` | Base directory — a `photometry_results/` subfolder is created here |
| `original_filename` | Basename of the source FITS image (used for output naming + provenance) |
| `original_path` | Full path to the source FITS image (provenance) |
| `zp`, `zp_sigma`, `n_zp_stars` | Zero-point value, uncertainty, and number of stars used |
| `fwhm`, `aperture_radius` | Measurement parameters used for this image |
| `astrometry_method` | Which method solved the WCS (`"astrometry_net"` or `"manual"`) |
| `photsys` | Which catalog/band the zero-point is calibrated against (default `"PS1-g"`) |

**Output location:**
<output_dir>/photometry_results/<original_basename>_phot.fits


## What's inside the file

**Data (table columns):** `x`, `y`, `RA`, `Dec`, `inst_mag`, `e_inst_mag`,
`real_mag` — one row per detected star.

**Header keywords:**

| Keyword | Meaning |
|---|---|
| `MAGZP` | Photometric zero-point (mag) |
| `MAGZPERR` | Zero-point uncertainty (mag) |
| `MAGZPNS` | Number of stars used to derive the zero-point |
| `PHOTSYS` | Reference photometric system/catalog |
| `FWHMPIX` | Measured FWHM used for this image (pixels) |
| `APERAD` | Aperture radius used (pixels) |
| `ASTRSRC` | Astrometric solution method used |
| `ORIGFILE` | Original source FITS filename |
| `ARCFILE` | Full path to the original source file |
| `DATE-RED` | Date/time this reduction was performed |

## Reading results back

A saved file **cannot** be opened meaningfully in Notepad (the table data
is binary, not plain text) or properly browsed in DS9 (which is built for
images, not tables). Two good options:

**Option 1 — Python (quickest, no extra software):**

```python
from astropy.table import Table
from astropy import conf

conf.max_lines = -1
conf.max_width = -1

t = Table.read(
    "C:/Users/Observatory/Exo/Data/test/calibrated/photometry_results/"
    "m33_r_2026_07_20_1x1_exp00.01.40.000_High_1_masked_minus_dark_div_"
    "non-dark-reduced_flat_bg_final_phot.fits"
)
print(t)
```

`conf.max_lines = -1` and `conf.max_width = -1` tell astropy to print the
**entire** table with no truncation — useful when you want to see every
row and column rather than a shortened preview.

To read the header metadata (zero-point, provenance, etc.) separately:

```python
from astropy.io import fits

with fits.open("path/to/your_phot.fits") as hdul:
    print(hdul[1].header["MAGZP"], hdul[1].header["MAGZPERR"])
```

**Option 2 — TOPCAT (for casual browsing, no code):**

[TOPCAT](https://www.star.bris.ac.uk/~mbt/topcat/) is a free, standard
astronomy tool for viewing and exploring catalog tables. It opens FITS
binary tables directly as a spreadsheet-like view, with sorting, filtering,
and plotting built in — a better fit for casual inspection than DS9 or
Notepad.

See `notebooks/04_saving_results.ipynb` for a worked example of saving and
reading back a real result file.

