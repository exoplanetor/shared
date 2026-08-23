# INO340 Image Calibration Pipeline

**Bias/dark/flat calibration, hot-pixel masking, 2D background handling for INO340 telescope data.**

Developed for the [Iranian National Observatory (INO340)](https://ino.org.ir/en/for-astronomers/).

**Developed by:** [Leila Sadeghi Ardestani](https://github.com/exoplanetor), with valuable help from different INO Team members

**Repository:** https://github.com/exoplanetor/shared

**Contact:** lsadeghi@ipm.ir

---

## Key Features

- Automatic FITS header labeling (OBSTYPE, FILTER) from filename conventions
- Interactive frame preview by frame type (bias/dark/flat/object)
- Hot-pixel detection and masking from a matched short/long dark pair
- Master bias, dark, and flat construction (median/sigma-clipped combine)
- Three-mode QC comparison on a test frame before committing to a batch run
- Optional 2D background subtraction
- All calibration choices logged into output FITS headers for traceability

**Not included:** astrometric (WCS) solving, photometry, or multi-night stacking. See **Next Steps** below for photometry.

## Scope & Assumptions

Single detector, single filter wheel, data in ADU, one observing run per folder. Exposure time expected in `EXPTIME`/`EXPOSURE` (seconds). The frame-preview step opens Tkinter windows and requires a local display (not headless/SSH without X forwarding).

## Requirements

Python ≥ 3.9.

| Package | Purpose |
|---|---|
| `astropy` | FITS I/O, units, CCDData |
| `ccdproc` | Bias/dark/flat combination and correction |
| `photutils` | Source masking, background estimation |
| `numpy`, `pandas` | Array/table handling |
| `matplotlib` | QC plots, diagnostics |
| `tkinter` | Frame preview windows |

```bash
pip install astropy numpy matplotlib ccdproc photutils pandas
```

Get the code from the repository above; keep `calibration_steps.py` and `INO-Calibration.py` in the same directory.

## Preparing Your Data

Place all raw FITS files (`.fit`/`.fits`) in one folder.

**Filenames must contain:** `bias`, `dark`, or `flat` (otherwise treated as OBJECT/science); a filter code — `g`, `i`, `r`, `u`, or `clear` — set off by `_`, `-`, or filename boundaries (e.g. `flat_r_001.fits`). Files not following this convention won't get auto-labeled headers.

**For hot-pixel masking:** create a `mask/` subfolder containing exactly two DARK frames (short + long exposure).

**⚠ Before running masking:** applying a hot-pixel mask moves your original files into `raw/` and replaces them with `*_masked.fits` versions in place. Back up your data first.

## Glossary

**Bias** — zero-exposure readout baseline. 
**Dark** — closed-shutter exposure capturing thermal signal. 
**Flat** — uniform-illumination exposure capturing pixel sensitivity variation. 
**Master frame** — combined reference built from many bias/dark/flat frames. 
**ADU** — raw detector output unit. 
**Hot pixel** — pixel with abnormally high, non-representative dark current.

## Running the Pipeline

Run `INO-Calibration.py` and enter your data folder path. Each step below prompts **[y/n]** unless noted.

| Step | Action | Notes |
|---|---|---|
| 1. Fix headers | Auto-fills OBSTYPE/BUNIT/FILTER from filenames | Recommended on first run |
| 2. Inspect frames | Preview each frame type in a scrollable grid | Per-group y/n |
| 3. Build hot-pixel mask | Confirms `mask/` requirements, then: enter short/long exposure times and binning → adjust diagnostic plot axes as needed → enter threshold (e⁻/s) → confirm mask preview | Typical CCD dark current: ~0.001–0.1 e⁻/s at low temperature; set threshold near where the diagnostic plot deviates from the ideal x=y line, or ~median + 1σ as a start |
| 4. Apply hot-pixel mask | Applies saved mask to all frames; originals moved to `raw/` | See data-safety warning above |
| 5. Calibrate (always runs) | Builds master bias/dark/flat → optional flat QC → calibrates one test frame 3 ways → shows comparison plot → **[choose 1/2/3]** → optional global background subtraction → applies to all science frames | See mode comparison below |

**Calibration modes:**

| Mode | Method | When to use |
|---|---|---|
| 1. `minus_dark_div_flat` | Dark-subtract, then flat-correct with a dark-corrected flat | Default choice — prevents the flat's own thermal signal from being multiplied in |
| 2. `minus_dark_div_non-dark-reduced_flat` | Dark-subtract, then flat-correct with a raw flat | If flats have negligible dark current, or mode 1's flat looked noisy in QC |
| 3. `divide_flat_only` | Flat-correct only, no dark subtraction | If darks are unreliable or thermal signal is negligible |

## Output Structure

| Folder | Contents |
|---|---|
| `raw/` | Original unmasked files (if masking was run) |
| `mask/` | Hot-pixel mask + source dark frames |
| `combined/` | Master bias/dark/flat frames per filter |
| `calibrated/` | QC test products + final `<filename>_<mode>[_bg]_final.fits` |

Calibration choices are logged into master-frame headers (`combined`, `bias_subtracted`) for traceability.

## Troubleshooting

| Error | Cause | Fix |
|---|---|---|
| `Expected exactly 2 DARK frames, found N` | Wrong count in `mask/` | Check folder contents/headers |
| `No bias/dark frames found` | No matching OBSTYPE | Run header-fixing, check filenames |
| `No FILTER keyword in <frame>` | Filter not detected | Set manually or fix filename |
| `No dark-corrected/raw flat for FILTER=X` | Missing flats for that filter | Add matching flats |
| `No exposure keyword in <frame>` | Missing EXPTIME/EXPOSURE | Add to header manually |
| No mask found when applying | Mask not built | Run Step 3 first, or skip |

## Next Steps

A fully functional **Aperture Photometry** code is also available for use on your calibrated output. Contact lsadeghi@ipm.ir for more information.
