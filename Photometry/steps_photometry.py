import os
import glob
import datetime

import numpy as np

from astropy.io import fits
from astropy.stats import sigma_clipped_stats
from scipy import ndimage
from scipy.spatial import cKDTree

from astropy.table import Table
from photutils.detection import DAOStarFinder
from astropy.modeling import models, fitting
from photutils.aperture import (
    CircularAperture,
    CircularAnnulus,
    aperture_photometry
)


# ============================================================
# INITIALIZATION
# ============================================================

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
        "u": {"psf_fwhm": None, "zp_AB": None},
        "g": {"psf_fwhm": None, "zp_AB": None},
        "r": {"psf_fwhm": None, "zp_AB": None},
        "i": {"psf_fwhm": None, "zp_AB": None},
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


# ============================================================
# FILE HANDLING
# ============================================================

def find_fits_files(images_dir):

    images = []

    for ext in ["*.fits"]:
        images.extend(
            glob.glob(
                os.path.join(images_dir, ext)
            )
        )

    images = sorted(list(set(images)))

    print(
        f"Found {len(images)} FITS files."
    )

    return images


def load_fits_images(images):

    dict_images = {}

    for image in images:

        try:

            with fits.open(
                image,
                memmap=False
            ) as im:

                raw_data = im[0].data

                if raw_data is None:
                    raise RuntimeError(
                        "Primary HDU contains no image data."
                    )

                header = im[0].header.copy()

                # Keep the original data for output products.
                data = np.asarray(
                    raw_data
                )

                # Create one cleaned floating-point copy.
                #
                # This is reused throughout the photometry pipeline,
                # avoiding repeated astype() / nan_to_num() operations.
                clean_data = np.nan_to_num(
                    data.astype(float),
                    nan=0.0,
                    posinf=0.0,
                    neginf=0.0
                )

                dict_images[
                    os.path.basename(image)
                ] = {
                    "path": image,
                    "header": header,
                    "data": data,
                    "clean_data": clean_data,
                }

        except Exception as e:

            print(
                f"Error loading {image}: {e}"
            )

    print(
        f"Loaded {len(dict_images)} images successfully."
    )

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


# ============================================================
# FOOTPRINT FILTER
# ============================================================

def apply_footprint_filter(
    found_stars,
    image_data,
    threshold_sigma=3,
    smoothing_sigma=0,
    min_area=10,
    connectivity=8,
    peak_to_total_thresh=0.2,
    hot_pixel_sigma_thresh=1e10,
    merge_close=True,
    merge_radius=10.0
):

    if (
        found_stars is None
        or len(found_stars) == 0
    ):
        return None

    # --------------------------------------------------------
    # IMAGE PREPARATION
    # --------------------------------------------------------

    img = np.asarray(
        image_data,
        dtype=float
    )

    # Avoid making another copy when data have already been
    # cleaned by load_fits_images().
    if not np.all(np.isfinite(img)):

        img = np.nan_to_num(
            img,
            nan=0.0,
            posinf=0.0,
            neginf=0.0
        )

    # --------------------------------------------------------
    # OPTIONAL SMOOTHING
    # --------------------------------------------------------

    if smoothing_sigma > 0:

        img_smooth = ndimage.gaussian_filter(
            img,
            smoothing_sigma
        )

    else:

        img_smooth = img

    # --------------------------------------------------------
    # GLOBAL IMAGE STATISTICS
    # --------------------------------------------------------

    _, median, std = sigma_clipped_stats(
        img_smooth,
        sigma=3.0
    )

    # Protect against pathological zero/noise images.
    if not np.isfinite(std) or std <= 0:

        return found_stars

    # --------------------------------------------------------
    # DETECTION MASK
    # --------------------------------------------------------

    mask = (
        img_smooth
        > median + threshold_sigma * std
    )

    struct = ndimage.generate_binary_structure(
        2,
        2 if connectivity == 8 else 1
    )

    labeled, _ = ndimage.label(
        mask,
        structure=struct
    )

    # --------------------------------------------------------
    # FOOTPRINT INSPECTION
    # --------------------------------------------------------

    slices = ndimage.find_objects(
        labeled
    )

    small_labels = set()

    for lab, slc in enumerate(
        slices,
        1
    ):

        if slc is None:
            continue

        footprint = (
            labeled[slc] == lab
        )

        npix = int(
            footprint.sum()
        )

        values = img_smooth[slc][
            footprint
        ]

        if len(values) == 0:
            continue

        peak = float(
            values.max()
        )

        total = float(
            values.sum()
        )

        peak_frac = (
            peak
            / (total + 1e-12)
        )

        if (
            npix < min_area
            and (
                peak_frac
                > peak_to_total_thresh
                or
                peak
                > hot_pixel_sigma_thresh * std
            )
        ):

            small_labels.add(lab)

    # --------------------------------------------------------
    # REMOVE REJECTED FOOTPRINTS
    # --------------------------------------------------------

    if small_labels:

        small_mask = np.isin(
            labeled,
            list(small_labels)
        )

    else:

        small_mask = np.zeros(
            labeled.shape,
            dtype=bool
        )

    # --------------------------------------------------------
    # KEEP VALID SOURCE POSITIONS
    # --------------------------------------------------------

    x = np.asarray(
        found_stars["xcentroid"],
        dtype=float
    )

    y = np.asarray(
        found_stars["ycentroid"],
        dtype=float
    )

    h, w = image_data.shape

    good_idx = []

    for i in range(
        len(found_stars)
    ):

        xi = int(
            round(x[i])
        )

        yi = int(
            round(y[i])
        )

        if (
            0 <= xi < w
            and
            0 <= yi < h
            and
            not small_mask[yi, xi]
        ):

            good_idx.append(i)

    if len(good_idx) == 0:
        return None

    filtered = found_stars[
        good_idx
    ]

    # ========================================================
    # CLOSE-SOURCE MERGING
    # ========================================================

    if (
        merge_close
        and filtered is not None
        and len(filtered) > 1
        and merge_radius > 0
    ):

        positions = np.column_stack(
            (
                np.asarray(
                    filtered["xcentroid"],
                    dtype=float
                ),
                np.asarray(
                    filtered["ycentroid"],
                    dtype=float
                )
            )
        )

        n_sources = len(
            filtered
        )

        # ----------------------------------------------------
        # FAST NEIGHBOUR SEARCH
        # ----------------------------------------------------
        #
        # The previous implementation compared every source
        # against every other source:
        #
        #     O(N^2)
        #
        # cKDTree finds only pairs within merge_radius.
        #
        # The distance criterion remains exactly:
        #
        #     distance <= merge_radius
        #
        # ----------------------------------------------------

        tree = cKDTree(
            positions
        )

        pairs = tree.query_pairs(
            r=float(merge_radius)
        )

        # ----------------------------------------------------
        # BUILD CONNECTIVITY GRAPH
        # ----------------------------------------------------

        adjacency = [
            []
            for _ in range(n_sources)
        ]

        for i, j in pairs:

            adjacency[i].append(j)
            adjacency[j].append(i)

        # ----------------------------------------------------
        # FIND CONNECTED GROUPS
        # ----------------------------------------------------

        groups = []

        visited = set()

        for i in range(n_sources):

            if i in visited:
                continue

            group = []

            stack = [i]

            visited.add(i)

            while stack:

                current = stack.pop()

                group.append(
                    current
                )

                for neighbor in adjacency[
                    current
                ]:

                    if neighbor not in visited:

                        visited.add(
                            neighbor
                        )

                        stack.append(
                            neighbor
                        )

            groups.append(
                group
            )

        # ----------------------------------------------------
        # KEEP HIGHEST-FLUX SOURCE
        # ----------------------------------------------------

        keep = []

        flux_values = np.asarray(
            filtered["flux"],
            dtype=float
        )

        for group in groups:

            group_fluxes = (
                flux_values[group]
            )

            finite = np.isfinite(
                group_fluxes
            )

            if np.any(finite):

                valid_group = np.asarray(
                    group
                )[finite]

                valid_fluxes = (
                    group_fluxes[finite]
                )

                best_index = (
                    valid_group[
                        np.argmax(
                            valid_fluxes
                        )
                    ]
                )

            else:

                best_index = group[0]

            keep.append(
                int(best_index)
            )

        filtered = filtered[
            keep
        ]

    return (
        filtered
        if filtered is not None
        and len(filtered) > 0
        else None
    )


# ============================================================
# STAR DETECTION
# ============================================================

def find_stars(
    image_path,
    image_index=0,
    threshold_factor=5.0,
    apply_filter=True,
    filter_params=None,
    verbose=False,
    fwhm_override=None,
    fwhm_guess=18.0,
    image_data=None
):
    """
    Detect stars using DAOStarFinder.

    Parameters
    ----------
    image_path : str
        Path to FITS image.

    image_data : ndarray, optional
        Already-loaded image data.

        If supplied, the FITS file is NOT reopened. This is the
        preferred mode for the main pipeline because the image
        has already been loaded by load_fits_images().

    fwhm_override : float, optional
        FWHM supplied externally.

    fwhm_guess : float
        Initial DAOStarFinder FWHM when no override is supplied.
    """

    # --------------------------------------------------------
    # USE ALREADY-LOADED IMAGE WHEN AVAILABLE
    # --------------------------------------------------------

    if image_data is None:

        with fits.open(
            image_path,
            memmap=False
        ) as hdul:

            raw_data = hdul[0].data

            if raw_data is None:
                return None

            data = np.nan_to_num(
                raw_data.astype(float),
                nan=0.0,
                posinf=0.0,
                neginf=0.0
            )

    else:

        data = np.asarray(
            image_data,
            dtype=float
        )

        if not np.all(
            np.isfinite(data)
        ):

            data = np.nan_to_num(
                data,
                nan=0.0,
                posinf=0.0,
                neginf=0.0
            )

    # --------------------------------------------------------
    # FWHM
    # --------------------------------------------------------

    fwhm_used = (
        fwhm_override
        if fwhm_override is not None
        else fwhm_guess
    )

    if (
        fwhm_used is None
        or not np.isfinite(fwhm_used)
        or fwhm_used <= 0
    ):

        fwhm_used = 18.0

    # --------------------------------------------------------
    # BACKGROUND
    # --------------------------------------------------------

    _, median, std = (
        sigma_clipped_stats(
            data,
            sigma=3.0
        )
    )

    if (
        not np.isfinite(std)
        or std <= 0
    ):

        return None

    data_bkgsub = (
        data - median
    )

    # --------------------------------------------------------
    # DAO STAR FINDER
    # --------------------------------------------------------

    daofind = DAOStarFinder(
        threshold=(
            threshold_factor * std
        ),
        fwhm=float(fwhm_used),
        ratio=0.8,
        sharphi=0.8,
        roundlo=-0.6,
        roundhi=0.6
    )

    found = daofind(
        data_bkgsub
    )

    # --------------------------------------------------------
    # FOOTPRINT FILTER
    # --------------------------------------------------------

    if (
        apply_filter
        and found is not None
    ):

        filter_params = (
            filter_params or {}
        )

        found = apply_footprint_filter(
            found,
            data,
            **filter_params
        )

    return found


# ============================================================
# ENSEMBLE FWHM ESTIMATION
# ============================================================

def estimate_fwhm_ensemble(
    stars,
    image_data,
    n_samples=10,
    cutout_size=60
):

    if (
        stars is None
        or len(stars) == 0
    ):
        return None, 0

    fitter = fitting.LevMarLSQFitter()

    fwhm_vals = []

    # --------------------------------------------------------
    # SELECT BRIGHTEST SOURCES
    # --------------------------------------------------------

    if "flux" in stars.colnames:

        flux = np.asarray(
            stars["flux"],
            dtype=float
        )

    else:

        flux = np.ones(
            len(stars)
        )

    finite_flux = np.isfinite(
        flux
    )

    if not np.any(
        finite_flux
    ):

        idx = np.arange(
            min(
                len(stars),
                n_samples
            )
        )

    else:

        safe_flux = np.where(
            finite_flux,
            flux,
            -np.inf
        )

        idx = np.argsort(
            safe_flux
        )[::-1][:n_samples]

    # --------------------------------------------------------
    # FIT EACH STAR
    # --------------------------------------------------------

    half = cutout_size // 2

    image_data = np.asarray(
        image_data,
        dtype=float
    )

    for i in idx:

        x = int(
            round(
                float(
                    stars["xcentroid"][i]
                )
            )
        )

        y = int(
            round(
                float(
                    stars["ycentroid"][i]
                )
            )
        )

        y0 = y - half
        y1 = y + half

        x0 = x - half
        x1 = x + half

        if (
            y0 < 0
            or x0 < 0
            or y1 > image_data.shape[0]
            or x1 > image_data.shape[1]
        ):

            continue

        stamp = image_data[
            y0:y1,
            x0:x1
        ]

        if stamp.size == 0:
            continue

        if not np.all(
            np.isfinite(stamp)
        ):

            stamp = np.nan_to_num(
                stamp,
                nan=0.0,
                posinf=0.0,
                neginf=0.0
            )

        # ----------------------------------------------------
        # LOCAL BACKGROUND
        # ----------------------------------------------------

        _, local_median, _ = (
            sigma_clipped_stats(
                stamp,
                sigma=3.0
            )
        )

        stamp_fit = (
            stamp - local_median
        )

        if (
            not np.any(
                stamp_fit > 0
            )
        ):

            continue

        # ----------------------------------------------------
        # INITIAL PEAK
        # ----------------------------------------------------

        y_peak, x_peak = (
            np.unravel_index(
                np.argmax(
                    stamp_fit
                ),
                stamp_fit.shape
            )
        )

        yy, xx = np.mgrid[
            :stamp_fit.shape[0],
            :stamp_fit.shape[1]
        ]

        # ----------------------------------------------------
        # GAUSSIAN MODEL
        # ----------------------------------------------------

        model = models.Gaussian2D(
            amplitude=float(
                stamp_fit.max()
            ),
            x_mean=float(
                x_peak
            ),
            y_mean=float(
                y_peak
            ),
            x_stddev=8.0,
            y_stddev=8.0
        )

        # ----------------------------------------------------
        # FIT
        # ----------------------------------------------------

        try:

            fit = fitter(
                model,
                xx,
                yy,
                stamp_fit
            )

            sx = abs(
                float(
                    fit.x_stddev.value
                )
            )

            sy = abs(
                float(
                    fit.y_stddev.value
                )
            )

            if not (
                np.isfinite(sx)
                and np.isfinite(sy)
            ):
                continue

            if (
                sx <= 0
                or sy <= 0
            ):
                continue

            fwhm = (
                2.3548
                * (sx + sy)
                / 2.0
            )

            # Reject pathological fits.
            if (
                fwhm < 5.0
                or fwhm > 50.0
            ):
                continue

            fwhm_vals.append(
                fwhm
            )

        except Exception:
            continue

    # --------------------------------------------------------
    # FINAL ENSEMBLE VALUE
    # --------------------------------------------------------

    if len(fwhm_vals) == 0:

        return None, 0

    return (
        float(
            np.nanmedian(
                fwhm_vals
            )
        ),
        len(fwhm_vals)
    )


# ============================================================
# MULTI-RADIUS APERTURE PHOTOMETRY
# ============================================================

def perform_aperture_photometry(
    dict_images,
    dict_aper,
    radii,
    sky_in=1.5,
    sky_out=2.5
):

    image_list = list(
        dict_images.keys()
    )

    for filename in image_list:

        # Prefer the cached cleaned image.
        data = dict_images[
            filename
        ].get(
            "clean_data",
            dict_images[
                filename
            ]["data"]
        )

        sources = dict_aper[
            filename
        ].get(
            "sources found",
            None
        )

        if (
            sources is None
            or len(sources) == 0
        ):

            dict_aper[
                filename
            ]["aperture phot table"] = None

            continue

        positions = np.transpose(
            (
                sources["xcentroid"],
                sources["ycentroid"]
            )
        )

        table_aper = Table()

        for rad in radii:

            rr = str(rad)

            aperture = CircularAperture(
                positions,
                r=rad
            )

            annulus_aperture = (
                CircularAnnulus(
                    positions,
                    r_in=sky_in,
                    r_out=sky_out
                )
            )

            annulus_mask = (
                annulus_aperture.to_mask(
                    method="center"
                )
            )

            # ------------------------------------------------
            # BACKGROUND
            # ------------------------------------------------

            bkg_median = []
            bkg_stdev = []

            for mask in annulus_mask:

                annulus_data = (
                    mask.multiply(data)
                )

                annulus_data_1d = (
                    annulus_data[
                        mask.data > 0
                    ]
                )

                _, median_sigclip, stdev_sigclip = (
                    sigma_clipped_stats(
                        annulus_data_1d
                    )
                )

                bkg_median.append(
                    median_sigclip
                )

                bkg_stdev.append(
                    stdev_sigclip
                )

            bkg_median = np.asarray(
                bkg_median
            )

            bkg_stdev = np.asarray(
                bkg_stdev
            )

            # ------------------------------------------------
            # APERTURE PHOTOMETRY
            # ------------------------------------------------

            phot = aperture_photometry(
                data,
                aperture,
                method="exact"
            )

            phot[
                "annulus_median"
            ] = bkg_median

            phot[
                "aper_bkg"
            ] = (
                bkg_median
                * aperture.area
            )

            phot[
                "aper_sum_bkgsub"
            ] = (
                phot["aperture_sum"]
                - phot["aper_bkg"]
            )

            # ------------------------------------------------
            # STORE COLUMNS
            # ------------------------------------------------

            table_aper[
                "aper_sum_" + rr + "px"
            ] = phot[
                "aperture_sum"
            ]

            table_aper[
                "annulus_median_" + rr + "px"
            ] = phot[
                "annulus_median"
            ]

            table_aper[
                "aper_bkg_" + rr + "px"
            ] = phot[
                "aper_bkg"
            ]

            table_aper[
                "aper_sum_bkgsub_" + rr + "px"
            ] = phot[
                "aper_sum_bkgsub"
            ]

            # ------------------------------------------------
            # ERROR PROPAGATION
            # ------------------------------------------------

            fluxerr = np.sqrt(
                np.abs(
                    phot[
                        "aperture_sum"
                    ]
                )
                +
                aperture.area
                * (
                    bkg_stdev ** 2
                )
                +
                (
                    bkg_stdev ** 2
                )
                * (
                    aperture.area ** 2
                )
                / (
                    annulus_aperture.area
                )
            )

            table_aper[
                "flux_err_" + rr + "px"
            ] = fluxerr

        dict_aper[
            filename
        ]["aperture phot table"] = (
            table_aper
        )

    return dict_aper


# ============================================================
# FINAL PHOTOMETRY TABLE
# ============================================================

def build_photometry_table(
    dict_images,
    dict_aper,
    range_width=1.0,
    aperture_mult=1.5,
    fallback_radius=10.0
):

    image_list = list(
        dict_images.keys()
    )

    for filename in image_list:

        # ----------------------------------------------------
        # USE CACHED CLEAN IMAGE
        # ----------------------------------------------------

        data = dict_images[
            filename
        ].get(
            "clean_data",
            dict_images[
                filename
            ]["data"]
        )

        if data is None:

            print(
                f"[WARNING] No data for {filename}"
            )

            continue

        sources = dict_aper[
            filename
        ].get(
            "sources found"
        )

        if (
            sources is None
            or len(sources) == 0
        ):

            print(
                f"[WARNING] No sources found for {filename}"
            )

            continue

        # ----------------------------------------------------
        # APERTURE RADIUS
        # ----------------------------------------------------

        fwhm = dict_aper[
            filename
        ].get(
            "fwhm",
            None
        )

        if (
            fwhm is not None
            and np.isfinite(fwhm)
            and fwhm > 0
        ):

            radius = (
                aperture_mult
                * fwhm
            )

        else:

            radius = fallback_radius

            print(
                f"[WARNING] No FWHM for {filename}, "
                f"using fallback radius = "
                f"{fallback_radius}px"
            )

        # ----------------------------------------------------
        # POSITIONS
        # ----------------------------------------------------

        positions = np.transpose(
            (
                sources["xcentroid"],
                sources["ycentroid"]
            )
        )

        # ----------------------------------------------------
        # APERTURE PHOTOMETRY
        # ----------------------------------------------------

        aperture = CircularAperture(
            positions,
            r=radius
        )

        phot = aperture_photometry(
            data,
            aperture,
            method="exact"
        )

        # ----------------------------------------------------
        # GLOBAL BACKGROUND
        # ----------------------------------------------------

        _, median, std = (
            sigma_clipped_stats(
                data,
                sigma=3.0
            )
        )

        bkg = (
            median
            * aperture.area
        )

        flux = (
            phot["aperture_sum"]
            - bkg
        )

        # ----------------------------------------------------
        # FLUX ERROR
        # ----------------------------------------------------

        flux_err = np.sqrt(
            np.abs(
                phot["aperture_sum"]
            )
            +
            aperture.area
            * std ** 2
        )

        # ----------------------------------------------------
        # BUILD FINAL TABLE
        # ----------------------------------------------------

        table_phot = Table()

        table_phot["x"] = (
            sources["xcentroid"]
        )

        table_phot["y"] = (
            sources["ycentroid"]
        )

        # ----------------------------------------------------
        # RA / DEC FROM IMAGE WCS
        # ----------------------------------------------------

        wcs = dict_aper[
            filename
        ].get(
            "manual_wcs",
            None
        )

        if wcs is not None:

            sky = wcs.pixel_to_world(
                sources["xcentroid"],
                sources["ycentroid"]
            )

            table_phot["RA"] = (
                sky.ra.deg
            )

            table_phot["Dec"] = (
                sky.dec.deg
            )

        else:

            print(
                f"[INFO] No WCS available for "
                f"{filename} — RA/Dec not computed."
            )

        # ----------------------------------------------------
        # INSTRUMENTAL MAGNITUDE
        # ----------------------------------------------------

        flux_safe = np.where(
            flux <= 0,
            np.nan,
            flux
        )

        table_phot[
            "inst_mag"
        ] = (
            -2.5
            * np.log10(
                flux_safe
            )
        )

        table_phot[
            "e_inst_mag"
        ] = (
            1.086
            * (
                flux_err
                / np.abs(
                    flux_safe
                )
            )
        )

        # ----------------------------------------------------
        # STORE
        # ----------------------------------------------------

        dict_aper[
            filename
        ]["final_aperture_phot_table"] = (
            table_phot
        )

        dict_aper[
            filename
        ]["aperture_radius_used"] = (
            radius
        )

        # ----------------------------------------------------
        # DYNAMIC MAGNITUDE RANGE STATISTICS
        # ----------------------------------------------------

        valid = np.isfinite(
            table_phot[
                "inst_mag"
            ]
        )

        if np.sum(valid) > 0:

            median_mag = (
                np.nanmedian(
                    table_phot[
                        "inst_mag"
                    ]
                )
            )

            mask = (
                (
                    table_phot[
                        "inst_mag"
                    ]
                    >=
                    median_mag
                    - range_width
                )
                &
                (
                    table_phot[
                        "inst_mag"
                    ]
                    <=
                    median_mag
                    + range_width
                )
            )

            avg_inst_mag = (
                np.nanmean(
                    table_phot[
                        "inst_mag"
                    ][mask]
                )
            )

            avg_e_inst_mag = (
                np.nanmean(
                    table_phot[
                        "e_inst_mag"
                    ][mask]
                )
            )

        else:

            avg_inst_mag = np.nan
            avg_e_inst_mag = np.nan

        dict_aper[
            filename
        ]["avg_inst_mag_dynamic_range"] = (
            avg_inst_mag
        )

        dict_aper[
            filename
        ]["avg_e_inst_mag_dynamic_range"] = (
            avg_e_inst_mag
        )

    return dict_aper


# ============================================================
# SAVE PHOTOMETRY TABLE
# ============================================================

def save_photometry_table(
    photometry_table,
    output_dir,
    original_filename,
    original_path=None,
    zp=None,
    zp_sigma=None,
    n_zp_stars=None,
    fwhm=None,
    aperture_radius=None,
    astrometry_method=None,
    photsys="PS1-g"
):

    if (
        photometry_table is None
        or len(photometry_table) == 0
    ):

        print(
            f"[WARNING] No photometry table to save "
            f"for {original_filename}."
        )

        return None

    results_dir = os.path.join(
        output_dir,
        "photometry_results"
    )

    os.makedirs(
        results_dir,
        exist_ok=True
    )

    base = os.path.splitext(
        original_filename
    )[0]

    out_path = os.path.join(
        results_dir,
        f"{base}_phot.fits"
    )

    try:

        hdu = fits.BinTableHDU(
            data=photometry_table.as_array()
        )

        # ----------------------------------------------------
        # PHOTOMETRIC CALIBRATION
        # ----------------------------------------------------

        if zp is not None:

            hdu.header["MAGZP"] = (
                round(
                    float(zp),
                    4
                ),
                "Photometric zero-point (mag)"
            )

        if zp_sigma is not None:

            hdu.header["MAGZPERR"] = (
                round(
                    float(zp_sigma),
                    4
                ),
                "Zero-point uncertainty (mag)"
            )

        if n_zp_stars is not None:

            hdu.header["MAGZPNS"] = (
                int(n_zp_stars),
                "Number of stars used for zero-point"
            )

        hdu.header["PHOTSYS"] = (
            photsys,
            "Reference photometric system/catalog"
        )

        # ----------------------------------------------------
        # MEASUREMENT PARAMETERS
        # ----------------------------------------------------

        if fwhm is not None:

            hdu.header["FWHMPIX"] = (
                round(
                    float(fwhm),
                    3
                ),
                "Measured FWHM (pixels)"
            )

        if aperture_radius is not None:

            hdu.header["APERAD"] = (
                round(
                    float(aperture_radius),
                    3
                ),
                "Aperture radius used (pixels)"
            )

        if astrometry_method is not None:

            hdu.header["ASTRSRC"] = (
                astrometry_method,
                "Astrometric solution method"
            )

        # ----------------------------------------------------
        # PROVENANCE
        # ----------------------------------------------------

        hdu.header["ORIGFILE"] = (
            original_filename,
            "Original source FITS filename"
        )

        if original_path is not None:

            hdu.header["ARCFILE"] = (
                original_path,
                "Full path to original source file"
            )

        hdu.header["DATE-RED"] = (
            datetime.datetime.now().isoformat(
                timespec="seconds"
            ),
            "Date/time this reduction was performed"
        )

        # ----------------------------------------------------
        # WRITE
        # ----------------------------------------------------

        hdu.writeto(
            out_path,
            overwrite=True
        )

        print(
            f"Saved: {out_path}"
        )

        return out_path

    except Exception as e:

        print(
            f"[WARNING] Failed to save photometry table "
            f"for {original_filename}: {e}"
        )

        return None


# ============================================================
# SAVE ASTROMETRIC IMAGE
# ============================================================

def save_astrometric_image(
    image_data,
    original_header,
    wcs,
    output_dir,
    original_filename,
    original_path=None
):

    if wcs is None:

        print(
            f"[INFO] No WCS available for "
            f"{original_filename} — skipping astrometric image save."
        )

        return None

    results_dir = os.path.join(
        output_dir,
        "astrometric_images"
    )

    os.makedirs(
        results_dir,
        exist_ok=True
    )

    base = os.path.splitext(
        original_filename
    )[0]

    out_path = os.path.join(
        results_dir,
        f"{base}_wcs.fits"
    )

    try:

        new_header = (
            original_header.copy()
        )

        wcs_header = (
            wcs.to_header(
                relax=True
            )
        )

        for key, value in (
            wcs_header.items()
        ):

            new_header[key] = value

        new_header["ORIGFILE"] = (
            original_filename,
            "Original source FITS filename"
        )

        if original_path is not None:

            new_header["ARCFILE"] = (
                original_path,
                "Full path to original source file"
            )

        hdu = fits.PrimaryHDU(
            data=image_data,
            header=new_header
        )

        hdu.writeto(
            out_path,
            overwrite=True
        )

        print(
            f"Saved astrometric image: {out_path}"
        )

        return out_path

    except Exception as e:

        print(
            f"[WARNING] Failed to save astrometric image "
            f"for {original_filename}: {e}"
        )

        return None


# ============================================================
# FILTER IDENTIFICATION
# ============================================================

PS1_SUPPORTED_FILTERS = {
    "g",
    "r",
    "i",
    "z",
    "y"
}


def get_filter_from_header(
    header,
    filename="<unknown file>"
):

    if "FILTER" not in header:

        raise RuntimeError(
            f"'{filename}' has no FILTER keyword in its header. "
            "This pipeline requires the calibration step to record "
            "the filter used. Please add a FILTER keyword to this "
            "file's header before running photometry on it."
        )

    filt = str(
        header["FILTER"]
    ).strip().lower()

    if not filt:

        raise RuntimeError(
            f"'{filename}' has an empty FILTER keyword. "
            "Please set a valid filter name in this file's header."
        )

    return filt
