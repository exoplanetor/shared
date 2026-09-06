import os
import glob
import numpy as np
from astropy.io import fits
from astropy.stats import sigma_clipped_stats
from scipy import ndimage
from astropy.table import Table
from photutils.detection import DAOStarFinder
from astropy.modeling import models, fitting
from photutils.aperture import CircularAperture, CircularAnnulus, aperture_photometry
# ----------------------------
# INITIALIZATION
# ----------------------------
def initialize_containers():
    dict_images = {}
    dict_aper = {}
    dict_filter_short = {}
    dict_filter_long = {}
    ff_short = []
    det_short = []
    det_long = []
    ff_long = []
    detlist_short = []
    detlist_long = []
    filtlist_short = []
    filtlist_long = []
    filter_data = {
        'u': {'psf_fwhm': None, 'zp_AB': None},
        'g': {'psf_fwhm': None, 'zp_AB': None},
        'r': {'psf_fwhm': None, 'zp_AB': None},
        'i': {'psf_fwhm': None, 'zp_AB': None},
    }
    return {
        "dict_images": dict_images,
        "dict_aper": dict_aper,
        "dict_filter_short": dict_filter_short,
        "dict_filter_long": dict_filter_long,
        "ff_short": ff_short,
        "det_short": det_short,
        "det_long": det_long,
        "ff_long": ff_long,
        "detlist_short": detlist_short,
        "detlist_long": detlist_long,
        "filtlist_short": filtlist_short,
        "filtlist_long": filtlist_long,
        "filter_data": filter_data,
    }

# ----------------------------
# FILE HANDLING
# ----------------------------
def find_fits_files(images_dir):
    images = []
    for ext in ["*.fits"]:
        images.extend(glob.glob(os.path.join(images_dir, ext)))
    images = sorted(list(set(images)))
    print(f"Found {len(images)} FITS files.")
    return images
def load_fits_images(images):
    dict_images = {}
    for image in images:
        try:
            im = fits.open(image)
            dict_images[os.path.basename(image)] = {
                "path": image,
                "header": im[0].header,
                "data": im[0].data,
            }
            im.close()
        except Exception as e:
            print(f"Error loading {image}: {e}")
    print(f"Loaded {len(dict_images)} images successfully.")
    return dict_images
def create_dict_aper(images):
    dict_aper = {}
    for fname in images:
        key = os.path.basename(fname)
        dict_aper[key] = {
            "sources found": None,
            "aperture phot table": None,
            "final aperture phot table": None
        }
    return dict_aper

# ----------------------------
# FOOTPRINT FILTER
# ----------------------------
def apply_footprint_filter(found_stars, image_data,
                           threshold_sigma=3,
                           smoothing_sigma=0,
                           min_area=10,
                           connectivity=8,
                           peak_to_total_thresh=0.2,
                           hot_pixel_sigma_thresh=1e10,
                           merge_close=True,
                           merge_radius=10.0):
    if found_stars is None or len(found_stars) == 0:
        return None
    img = np.nan_to_num(image_data, nan=0.0, posinf=0.0, neginf=0.0)
    img_smooth = ndimage.gaussian_filter(img, smoothing_sigma) if smoothing_sigma > 0 else img
    _, median, std = sigma_clipped_stats(img_smooth, sigma=3.0)
    mask = img_smooth > median + threshold_sigma * std
    struct = ndimage.generate_binary_structure(2, 2 if connectivity == 8 else 1)
    labeled, _ = ndimage.label(mask, structure=struct)
    slices = ndimage.find_objects(labeled)
    small_labels = set()
    for lab, slc in enumerate(slices, 1):
        footprint = labeled[slc] == lab
        npix = footprint.sum()
        values = img_smooth[slc][footprint]
        if len(values) == 0:
            continue
        peak = values.max()
        total = values.sum()
        peak_frac = peak / (total + 1e-12)
        if npix < min_area and (
            peak_frac > peak_to_total_thresh or
            peak > hot_pixel_sigma_thresh * std
        ):
            small_labels.add(lab)
    small_mask = np.isin(labeled, list(small_labels))
    x = found_stars["xcentroid"]
    y = found_stars["ycentroid"]
    h, w = image_data.shape
    good_idx = [
        i for i in range(len(found_stars))
        if 0 <= int(round(x[i])) < w
        and 0 <= int(round(y[i])) < h
        and not small_mask[int(round(y[i])), int(round(x[i]))]
    ]
    filtered = found_stars[good_idx] if len(good_idx) > 0 else None
    if merge_close and filtered is not None and len(filtered) > 1:
        positions = np.vstack((filtered["xcentroid"], filtered["ycentroid"])).T
        keep = []
        used = set()
        for i, p1 in enumerate(positions):
            if i in used:
                continue
            group = [i]
            for j, p2 in enumerate(positions[i + 1:], start=i + 1):
                if j in used:
                    continue
                if np.linalg.norm(p1 - p2) < merge_radius:
                    group.append(j)
            fluxes = filtered["flux"][group]
            keep.append(group[np.argmax(fluxes)])
            used.update(group)
        filtered = filtered[keep]
    return filtered if filtered is not None and len(filtered) > 0 else None

# ----------------------------
# STAR DETECTION
# ----------------------------
def find_stars(image_path, image_index=0,
               threshold_factor=5.0,
               apply_filter=True,
               filter_params=None,
               verbose=False,
               fwhm_override=None):
    """
    Detect stars in a single FITS image using DAOStarFinder.
    FWHM handling (important!):
    - DAOStarFinder needs an FWHM value to know what size/shape to search for.
    - We don't know the *real* FWHM until we've already found some stars and measured it
      (that happens elsewhere, in estimate_fwhm_ensemble()).
    - So this function supports two modes:
        1) First pass: no real measurement exists yet -> use a rough guess (fwhm_guess).
        2) Second pass: caller already measured the real FWHM for this image and passes
           it in via fwhm_override -> we use that instead of the guess.
    """
    # Load the raw pixel data from the FITS file
    with fits.open(image_path) as hdul:
        data = hdul[0].data.astype(float)
    # Clean up bad pixel values before doing any math on this image.
    # Masked/calibrated FITS files often contain NaN (missing data) or
    # Inf (broken math from calibration) in certain pixels — e.g. bad pixels,
    # saturated pixels, or masked regions. DAOStarFinder can't handle these
    # and will throw "invalid value" warnings/errors if we don't clean first.
    data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
    # Rough starting guess for FWHM (pixels), used only if no real measurement is available yet
    fwhm_guess = 3.0
    # Decide which FWHM value to actually use:
    # - if a real measured value was passed in (fwhm_override), use that (2nd pass / accurate)
    # - otherwise, fall back to the rough guess above (1st pass / bootstrap)
    fwhm_used = fwhm_override if fwhm_override is not None else fwhm_guess
    # Estimate the image background level (median) and noise (std), ignoring outliers (sigma clipping)
    _, median, std = sigma_clipped_stats(data, sigma=3.0)
    # Subtract background so DAOStarFinder is only looking at star signal, not sky glow
    data_bkgsub = data - median
    # Set up the star-finding algorithm:
    # - threshold: how bright (in units of noise/std) a peak must be to count as a star
    # - fwhm: expected blob width in pixels (this is where fwhm_used goes in)
    # - ratio/sharphi/roundlo/roundhi: shape filters to reject non-star-like blobs
    #   (e.g. cosmic ray hits, hot pixels, elongated artifacts)
    daofind = DAOStarFinder(
        threshold=threshold_factor * std,
        fwhm=fwhm_used,
        ratio=0.8,
        sharphi=0.8,
        roundlo=-0.6,
        roundhi=0.6
    )
    # Run the actual detection on the background-subtracted image
    found = daofind(data_bkgsub)
    # Optional extra cleanup step: removes small/spurious detections
    # (e.g. hot pixels, cosmic rays) that slipped past DAOStarFinder's own shape filters
    if apply_filter and found is not None:
        filter_params = filter_params or {}
        found = apply_footprint_filter(found, data, **filter_params)
    return found
# ----------------------------
# ENSEMBLE FWHM ESTIMATION
# ----------------------------
def estimate_fwhm_ensemble(stars, image_data, n_samples=10, cutout_size=40):
    if stars is None or len(stars) == 0:
        return None, 0
    fitter = fitting.LevMarLSQFitter()
    fwhm_vals = []
    flux = stars["flux"] if "flux" in stars.colnames else np.ones(len(stars))
    idx = np.argsort(flux)[::-1][:n_samples]
    for i in idx:
        x = int(stars["xcentroid"][i])
        y = int(stars["ycentroid"][i])
        y0, y1 = y - cutout_size//2, y + cutout_size//2
        x0, x1 = x - cutout_size//2, x + cutout_size//2
        if y0 < 0 or x0 < 0 or y1 >= image_data.shape[0] or x1 >= image_data.shape[1]:
            continue
        stamp = image_data[y0:y1, x0:x1]
        if np.nanmax(stamp) <= 0:
            continue
        yy, xx = np.mgrid[:stamp.shape[0], :stamp.shape[1]]
        y_peak, x_peak = np.unravel_index(np.argmax(stamp), stamp.shape)
        model = models.Gaussian2D(
            amplitude=stamp.max(),
            x_mean=x_peak,
            y_mean=y_peak,
            x_stddev=2.0,
            y_stddev=2.0
        )
        try:
            fit = fitter(model, xx, yy, stamp)
            fwhm = 2.3548 * (fit.x_stddev.value + fit.y_stddev.value) / 2
            fwhm_vals.append(fwhm)
        except Exception:
            continue
    return (np.nanmedian(fwhm_vals), len(fwhm_vals)) if len(fwhm_vals) > 0 else (None, 0)

def perform_aperture_photometry(dict_images, dict_aper, radii,
                                sky_in=1.5, sky_out=2.5):
    image_list = list(dict_images.keys())
    for filename in image_list:
        data = dict_images[filename]["data"]
        sources = dict_aper[filename].get("sources found", None)
        if sources is None or len(sources) == 0:
            dict_aper[filename]["aperture phot table"] = None
            continue
        positions = np.transpose((sources["xcentroid"], sources["ycentroid"]))
        table_aper = Table()
        for rad in radii:
            rr = str(rad)
            aperture = CircularAperture(positions, r=rad)
            annulus_aperture = CircularAnnulus(positions, r_in=sky_in, r_out=sky_out)
            annulus_mask = annulus_aperture.to_mask(method='center')
            # -------------------------
            # BACKGROUND ESTIMATION
            # -------------------------
            bkg_median = []
            bkg_stdev = []
            for mask in annulus_mask:
                annulus_data = mask.multiply(data)
                annulus_data_1d = annulus_data[mask.data > 0]
                _, median_sigclip, stdev_sigclip = sigma_clipped_stats(annulus_data_1d)
                bkg_median.append(median_sigclip)
                bkg_stdev.append(stdev_sigclip)
            bkg_median = np.array(bkg_median)
            bkg_stdev = np.array(bkg_stdev)
            # -------------------------
            # APERTURE PHOTOMETRY
            # -------------------------
            phot = aperture_photometry(data, aperture, method='exact')
            phot['annulus_median'] = bkg_median
            phot['aper_bkg'] = bkg_median * aperture.area
            phot['aper_sum_bkgsub'] = phot['aperture_sum'] - phot['aper_bkg']
            # -------------------------
            # STORE COLUMNS
            # -------------------------
            table_aper['aper_sum_' + rr + 'px'] = phot['aperture_sum']
            table_aper['annulus_median_' + rr + 'px'] = phot['annulus_median']
            table_aper['aper_bkg_' + rr + 'px'] = phot['aper_bkg']
            table_aper['aper_sum_bkgsub_' + rr + 'px'] = phot['aper_sum_bkgsub']
            # -------------------------
            # ERROR PROPAGATION
            # -------------------------
            fluxerr = np.sqrt(
                phot['aperture_sum'] +                      # poisson
                aperture.area * (bkg_stdev ** 2) +         # sky scatter
                (bkg_stdev ** 2) * (aperture.area ** 2) / (annulus_aperture.area)
            )
            table_aper['flux_err_' + rr + 'px'] = fluxerr
        dict_aper[filename]["aperture phot table"] = table_aper
    return dict_aper

def build_photometry_table(dict_images, dict_aper, range_width=1.0,
                           aperture_mult=1.5, fallback_radius=10.0):
    """
    NOTE: WCS is looked up PER IMAGE from dict_aper[filename]["manual_wcs"],
    since each image gets its own independent astrometric solution (dithered
    frames are solved individually, not sharing one reference WCS). Images
    with no WCS (e.g. a failed solve) simply won't get RA/Dec columns, but
    still get full x/y photometry as normal.
    """
    import numpy as np
    from astropy.table import Table
    from photutils.aperture import CircularAperture
    from astropy.stats import sigma_clipped_stats
    from astropy.io import fits
    image_list = list(dict_images.keys())
    for filename in image_list:
        # -------------------------
        # LOAD IMAGE (ONCE ONLY)
        # -------------------------
        data = dict_images[filename]["data"].astype(float)
        if data is None:
            print(f"[WARNING] No data for {filename}")
            continue
        sources = dict_aper[filename]["sources found"]
        if sources is None or len(sources) == 0:
            print(f"[WARNING] No sources found for {filename}")
            continue
        # clean data
        data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
        # -------------------------
        # PER-IMAGE APERTURE RADIUS, BASED ON THIS IMAGE'S OWN FWHM
        # -------------------------
        fwhm = dict_aper[filename].get("fwhm", None)
        if fwhm is not None:
            radius = aperture_mult * fwhm
        else:
            # No FWHM was measured for this image (e.g. estimation failed) —
            # fall back to a fixed default so the pipeline doesn't crash.
            radius = fallback_radius
            print(f"[WARNING] No FWHM for {filename}, using fallback radius = {fallback_radius}px")
        # -------------------------
        # POSITIONS
        # -------------------------
        positions = np.transpose(
            (sources["xcentroid"], sources["ycentroid"])
        )

        # -------------------------
        # APERTURE PHOTOMETRY
        # -------------------------
        aperture = CircularAperture(positions, r=radius)
        phot = aperture_photometry(data, aperture, method="exact")
        # background estimate (simple + robust fallback)
        _, median, std = sigma_clipped_stats(data, sigma=3.0)
        bkg = median * aperture.area
        flux = phot["aperture_sum"] - bkg
        # flux error (poisson approximation)
        flux_err = np.sqrt(np.abs(phot["aperture_sum"]) + aperture.area * std**2)

        # -------------------------
        # BUILD TABLE (single build — RA/Dec no longer gets wiped out)
        # -------------------------
        table_phot = Table()
        table_phot["x"] = sources["xcentroid"]
        table_phot["y"] = sources["ycentroid"]

        # -------------------------
        # RA / DEC FROM THIS IMAGE'S OWN WCS
        # -------------------------
        wcs = dict_aper[filename].get("manual_wcs", None)
        if wcs is not None:
            sky = wcs.pixel_to_world(
                sources["xcentroid"],
                sources["ycentroid"]
            )
            table_phot["RA"] = sky.ra.deg
            table_phot["Dec"] = sky.dec.deg
        else:
            print(f"[INFO] No WCS available for {filename} — RA/Dec not computed.")

        # avoid log(0)
        flux_safe = np.where(flux <= 0, np.nan, flux)
        table_phot["inst_mag"] = -2.5 * np.log10(flux_safe)
        table_phot["e_inst_mag"] = 1.086 * (flux_err / np.abs(flux_safe))
        # store
        dict_aper[filename]["final_aperture_phot_table"] = table_phot
        dict_aper[filename]["aperture_radius_used"] = radius
        # -------------------------
        # STATS
        # -------------------------
        valid = np.isfinite(table_phot["inst_mag"])
        if np.sum(valid) > 0:
            median_mag = np.nanmedian(table_phot["inst_mag"])
            mask = (
                (table_phot["inst_mag"] >= median_mag - range_width) &
                (table_phot["inst_mag"] <= median_mag + range_width)
            )
            avg_inst_mag = np.nanmean(table_phot["inst_mag"][mask])
            avg_e_inst_mag = np.nanmean(table_phot["e_inst_mag"][mask])
        else:
            avg_inst_mag = np.nan
            avg_e_inst_mag = np.nan
        dict_aper[filename]["avg_inst_mag_dynamic_range"] = avg_inst_mag
        dict_aper[filename]["avg_e_inst_mag_dynamic_range"] = avg_e_inst_mag
    return dict_aper


# ----------------------------
# SAVE RESULTS TO DISK (FITS binary table, with zero-point + provenance in header)
# ----------------------------
def save_photometry_table(photometry_table, output_dir, original_filename,
                          original_path=None, zp=None, zp_sigma=None, n_zp_stars=None,
                          fwhm=None, aperture_radius=None, astrometry_method=None,
                          photsys="PS1-g"):
    """
    Save one image's final photometry table as a FITS binary table, following
    the ESO Phase 3-style convention: standardized photometric keywords
    (MAGZP, MAGZPERR, MAGZPNS, PHOTSYS) plus provenance keywords (ORIGFILE,
    ARCFILE, DATE-RED) written directly into the FITS header, not just
    applied to the data and discarded.

    photometry_table : astropy Table (the final per-image photometry table)
    output_dir : base directory (a 'photometry_results' subfolder is created here)
    original_filename : basename of the source FITS image (for naming + ORIGFILE)
    original_path : full path to the source FITS image (for ARCFILE), optional
    zp, zp_sigma, n_zp_stars : zero-point value, uncertainty, and star count used
    fwhm, aperture_radius : measurement parameters used for this image
    astrometry_method : "astrometry_net" or "manual"
    photsys : which catalog/band the zero-point is calibrated against

    Returns the full output file path, or None if saving failed.
    """
    import datetime

    if photometry_table is None or len(photometry_table) == 0:
        print(f"[WARNING] No photometry table to save for {original_filename}.")
        return None

    results_dir = os.path.join(output_dir, "photometry_results")
    os.makedirs(results_dir, exist_ok=True)

    base = os.path.splitext(original_filename)[0]
    out_path = os.path.join(results_dir, f"{base}_phot.fits")

    try:
        hdu = fits.BinTableHDU(data=photometry_table.as_array())

        # --- Photometric calibration keywords (ESO-style) ---
        if zp is not None:
            hdu.header["MAGZP"] = (round(float(zp), 4), "Photometric zero-point (mag)")
        if zp_sigma is not None:
            hdu.header["MAGZPERR"] = (round(float(zp_sigma), 4), "Zero-point uncertainty (mag)")
        if n_zp_stars is not None:
            hdu.header["MAGZPNS"] = (int(n_zp_stars), "Number of stars used for zero-point")
        hdu.header["PHOTSYS"] = (photsys, "Reference photometric system/catalog")

        # --- Measurement parameters ---
        if fwhm is not None:
            hdu.header["FWHMPIX"] = (round(float(fwhm), 3), "Measured FWHM (pixels)")
        if aperture_radius is not None:
            hdu.header["APERAD"] = (round(float(aperture_radius), 3), "Aperture radius used (pixels)")
        if astrometry_method is not None:
            hdu.header["ASTRSRC"] = (astrometry_method, "Astrometric solution method")

        # --- Provenance keywords ---
        hdu.header["ORIGFILE"] = (original_filename, "Original source FITS filename")
        if original_path is not None:
            hdu.header["ARCFILE"] = (original_path, "Full path to original source file")
        hdu.header["DATE-RED"] = (
            datetime.datetime.now().isoformat(timespec="seconds"),
            "Date/time this reduction was performed"
        )

        hdu.writeto(out_path, overwrite=True)
        print(f"Saved: {out_path}")
        return out_path

    except Exception as e:
        print(f"[WARNING] Failed to save photometry table for {original_filename}: {e}")
        return None


# ----------------------------
# FILTER IDENTIFICATION
# ----------------------------
PS1_SUPPORTED_FILTERS = {"g", "r", "i", "z", "y"}


def get_filter_from_header(header, filename="<unknown file>"):
    """
    Read the FILTER keyword from a FITS header, added by the calibration
    pipeline upstream of this photometry pipeline.

    Raises a RuntimeError (halting the pipeline) if the keyword is missing
    or empty - filter identity must be known before zero-point calibration
    can be trusted, so this is not something to silently guess or default.

    Returns the filter name as a clean, lowercase string (e.g. 'r'),
    with surrounding whitespace stripped (FITS string values are often
    padded, e.g. "'r       '").
    """
    if "FILTER" not in header:
        raise RuntimeError(
            f"'{filename}' has no FILTER keyword in its header. "
            "This pipeline requires the calibration step to record the "
            "filter used. Please add a FILTER keyword to this file's "
            "header before running photometry on it."
        )

    filt = str(header["FILTER"]).strip().lower()
    if not filt:
        raise RuntimeError(
            f"'{filename}' has an empty FILTER keyword. "
            "Please set a valid filter name in this file's header."
        )

    return filt
