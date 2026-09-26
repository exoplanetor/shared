import os
import numpy as np

from config import (
    ASTROMETRY_API_KEY,
    SCALE_LOWER_ARCMIN,
    SCALE_UPPER_ARCMIN
)

from steps_photometry import (
    initialize_containers,
    find_fits_files,
    load_fits_images,
    create_dict_aper,
    find_stars,
    estimate_fwhm_ensemble,
    build_photometry_table,
    save_photometry_table,
    save_astrometric_image,
    get_filter_from_header,
    PS1_SUPPORTED_FILTERS
)


# =============================================================
# USER SETTINGS
# =============================================================

def get_filter_params_from_user():
    """
    Ask the user whether to customize the footprint filter.

    Returns
    -------
    dict
        Parameters passed to apply_footprint_filter().
    """

    print()
    print("Source footprint filtering")
    print("--------------------------")

    customize = input(
        "Customize footprint filter? (y/n, default n): "
    ).strip().lower()

    if customize != "y":
        return {
            "threshold_sigma": 3.0,
            "smoothing_sigma": 0,
            "min_area": 10,
            "connectivity": 8,
            "peak_to_total_thresh": 0.2,
            "hot_pixel_sigma_thresh": 1e10,
            "merge_close": False,
            "merge_radius": 5.0
        }

    threshold_sigma = float(
        input(
            "Threshold sigma (default 3): "
        ) or 3
    )

    smoothing_sigma = float(
        input(
            "Smoothing sigma (default 0): "
        ) or 0
    )

    min_area = int(
        input(
            "Minimum footprint area in pixels (default 10): "
        ) or 10
    )

    connectivity = int(
        input(
            "Connectivity (4 or 8, default 8): "
        ) or 8
    )

    peak_to_total_thresh = float(
        input(
            "Peak/total threshold (default 0.2): "
        ) or 0.2
    )

    hot_pixel_sigma_thresh = float(
        input(
            "Hot-pixel sigma threshold (default 1e10): "
        ) or 1e10
    )

    merge_close = (
        input(
            "Merge nearby detections? (y/n, default n): "
        ).strip().lower() == "y"
    )

    merge_radius = float(
        input(
            "Merge radius in pixels (default 5): "
        ) or 5
    )

    return {
        "threshold_sigma": threshold_sigma,
        "smoothing_sigma": smoothing_sigma,
        "min_area": min_area,
        "connectivity": connectivity,
        "peak_to_total_thresh": peak_to_total_thresh,
        "hot_pixel_sigma_thresh": hot_pixel_sigma_thresh,
        "merge_close": merge_close,
        "merge_radius": merge_radius
    }


def get_astrometry_choice():
    """
    Ask whether astrometric calibration should be performed.
    """

    choice = input(
        "\nPerform astrometry/WCS calibration? "
        "(y/n, default y): "
    ).strip().lower()

    return choice != "n"


def get_astrometry_settings():
    """
    Ask the user which astrometry method to use.
    """

    print()
    print("Astrometry method")
    print("-----------------")
    print("1) astrometry.net automatic")
    print("2) manual")

    choice = input(
        "Choose method (1/2, default 2): "
    ).strip()

    if choice == "1":

        api_key = ASTROMETRY_API_KEY

        if not api_key:
            raise RuntimeError(
                "ASTROMETRY_API_KEY is not configured."
            )

        return {
            "method": "astrometry_net",
            "api_key": api_key
        }

    return {
        "method": "manual"
    }


# =============================================================
# MAIN
# =============================================================

def main():

    # =========================================================
    # INPUT DIRECTORY
    # =========================================================

    images_dir = input(
        "Enter directory containing FITS images: "
    ).strip()

    if not os.path.isdir(images_dir):
        raise RuntimeError(
            f"Directory does not exist: {images_dir}"
        )

    # =========================================================
    # INITIALIZE CONTAINERS
    # =========================================================
    
    containers = initialize_containers()
    
    dict_images = containers["dict_images"]
    dict_aper = containers["dict_aper"]
    
    dict_filter_short = containers["dict_filter_short"]
    dict_filter_long = containers["dict_filter_long"]
    
    ff_short = containers["ff_short"]
    det_short = containers["det_short"]
    det_long = containers["det_long"]
    ff_long = containers["ff_long"]
    
    detlist_short = containers["detlist_short"]
    detlist_long = containers["detlist_long"]
    
    filtlist_short = containers["filtlist_short"]
    filtlist_long = containers["filtlist_long"]
    
    filter_data = containers["filter_data"]
    # =========================================================
    # FIND FITS FILES
    # =========================================================

    images = find_fits_files(images_dir)

    if images is None or len(images) == 0:
        raise RuntimeError(
            "No FITS images found."
        )

    # =========================================================
    # LOAD FITS IMAGES
    # =========================================================

    dict_images = load_fits_images(images)

    if dict_images is None or len(dict_images) == 0:
        raise RuntimeError(
            "No FITS images could be loaded."
        )

    # =========================================================
    # CREATE APERTURE CONTAINERS
    # =========================================================

    dict_aper = create_dict_aper(images)
    filter_params = get_filter_params_from_user()
    initial_fwhm_guess = 18.0
    # =========================================================
    # SOURCE DETECTION + FWHM MEASUREMENT
    # =========================================================
    #
    # Single detection pass:
    #
    #   DAOStarFinder
    #        ↓
    #   footprint filtering
    #        ↓
    #   FWHM measurement
    #
    # The measured FWHM is then used later for the photometric
    # aperture radius (1.5 × FWHM).
    # =========================================================

    print()
    print("=================================================")
    print("SOURCE DETECTION")
    print("=================================================")

    for fname in dict_images.keys():

        image_data = dict_images[fname]["clean_data"]

        print(
            f"{fname}: detecting sources..."
        )

        found_stars = find_stars(
            image_path=dict_images[fname]["path"],
            image_index=0,
            threshold_factor=6.0,
            apply_filter=True,
            filter_params=filter_params,
            fwhm_guess=initial_fwhm_guess,
            image_data=image_data
        )

        if found_stars is None or len(found_stars) == 0:

            print(
                f"{fname}: no usable sources found."
            )

            dict_aper[fname]["sources found"] = None
            dict_aper[fname]["fwhm"] = None
            dict_aper[fname]["fwhm_n"] = 0

            continue

        print(
            f"{fname}: {len(found_stars)} sources found "
            "after footprint filtering."
        )

        # -----------------------------------------------------
        # FWHM MEASUREMENT — ONCE
        # -----------------------------------------------------

        print(
            f"{fname}: measuring FWHM..."
        )

        fwhm_final, n_fwhm_final = estimate_fwhm_ensemble(
            stars=found_stars,
            image_data=image_data
        )

        if fwhm_final is None or not np.isfinite(fwhm_final):

            print(
                f"{fname}: FWHM measurement failed; "
                f"using initial FWHM = "
                f"{initial_fwhm_guess:.2f}px."
            )

            fwhm_final = float(initial_fwhm_guess)
            n_fwhm_final = 0

        else:

            print(
                f"{fname}: FWHM = "
                f"{fwhm_final:.2f} px "
                f"({n_fwhm_final} successful fits)"
            )

        # -----------------------------------------------------
        # STORE FINAL SOURCE LIST / FWHM
        # -----------------------------------------------------
        #
        # There is deliberately NO second detection and
        # NO second footprint-filtering step here.
        #
        # The source list returned by find_stars() is already
        # footprint-filtered.
        # -----------------------------------------------------

        dict_aper[fname]["sources found"] = found_stars

        dict_aper[fname]["fwhm"] = fwhm_final
        dict_aper[fname]["fwhm_n"] = n_fwhm_final

        # Kept for compatibility with any later code that
        # expects this field. No additional merge is performed.
        dict_aper[fname]["merge_radius_used"] = None


    # =========================================================
    # ASTROMETRY CHOICE
    # =========================================================

    perform_astrometry = get_astrometry_choice()

    astrometry_settings = None

    if perform_astrometry:
        astrometry_settings = get_astrometry_settings()

    # =========================================================
    # ASTROMETRY
    # =========================================================

    if perform_astrometry:

        # -----------------------------------------------------
        # IMPORT ASTROMETRY FUNCTIONS
        # -----------------------------------------------------

        from steps_astrometry import (
            select_astrometric_candidates,
            solve_with_astrometry_net,
            solve_with_astrometry_net_raw_upload,
            review_astrometric_candidates,
            run_manual_wcs
        )
    # =========================================================
    # ASTROMETRY
    # =========================================================

    if perform_astrometry:

        # -----------------------------------------------------
        # IMPORT ASTROMETRY FUNCTIONS
        # -----------------------------------------------------

        from steps_astrometry import (
            select_astrometric_candidates,
            solve_with_astrometry_net,
            solve_with_astrometry_net_raw_upload,
            review_astrometric_candidates,
            run_manual_wcs
        )

        # -----------------------------------------------------
        # REFERENCE IMAGE
        # -----------------------------------------------------

        image_names = list(dict_images.keys())

        if len(image_names) == 0:
            raise RuntimeError(
                "No images available for astrometry."
            )

        reference_fname = image_names[0]

        print()
        print("=================================================")
        print("ASTROMETRY / WCS CALIBRATION")
        print("=================================================")

        print(
            f"Reference image: {reference_fname}"
        )

        # -----------------------------------------------------
        # SOLVE EACH IMAGE
        # -----------------------------------------------------

        for fname in image_names:

            found_stars = dict_aper[fname][
                "sources found"
            ]

            image_data = dict_images[fname][
                "data"
            ].astype(float)

            fwhm = dict_aper[fname].get(
                "fwhm",
                None
            )

            if found_stars is None or len(found_stars) == 0:

                print(
                    f"{fname}: no sources available "
                    "for astrometry."
                )

                dict_aper[fname][
                    "astrometric_candidates"
                ] = None

                dict_aper[fname][
                    "manual_wcs"
                ] = None

                continue

            if fwhm is None:

                print(
                    f"{fname}: no FWHM available; "
                    "skipping astrometry."
                )

                dict_aper[fname][
                    "astrometric_candidates"
                ] = None

                dict_aper[fname][
                    "manual_wcs"
                ] = None

                continue

            # -------------------------------------------------
            # ASTROMETRIC CANDIDATES
            # -------------------------------------------------

            candidates = select_astrometric_candidates(
                found_stars,
                image_data=image_data
            )

            dict_aper[fname][
                "astrometric_candidates"
            ] = candidates

            if candidates is None or len(candidates) == 0:

                print(
                    f"{fname}: no suitable astrometric "
                    "candidates."
                )

                dict_aper[fname][
                    "manual_wcs"
                ] = None

                continue

            # =================================================
            # ASTROMETRY.NET
            # =================================================

            if astrometry_settings["method"] == "astrometry_net":

                # ---------------------------------------------
                # FIRST TRY: SOURCE LIST
                # ---------------------------------------------

                wcs_solution = solve_with_astrometry_net(
                    sources=candidates,
                    image_shape=image_data.shape,
                    api_key=astrometry_settings["api_key"],
                    scale_lower_arcmin=SCALE_LOWER_ARCMIN,
                    scale_upper_arcmin=SCALE_UPPER_ARCMIN
                )

                used_method = "source_list"

                # ---------------------------------------------
                # SECOND TRY: RAW IMAGE UPLOAD
                # ---------------------------------------------

                if wcs_solution is None:

                    print(
                        f"{fname}: source-list solve failed, "
                        "trying image-upload solve instead..."
                    )

                    wcs_solution = (
                        solve_with_astrometry_net_raw_upload(
                            image_path=dict_images[fname]["path"],
                            api_key=astrometry_settings["api_key"],
                            scale_lower_arcmin=SCALE_LOWER_ARCMIN,
                            scale_upper_arcmin=SCALE_UPPER_ARCMIN
                        )
                    )

                    used_method = "image_upload"

                # ---------------------------------------------
                # STORE WCS RESULT
                # ---------------------------------------------

                dict_aper[fname][
                    "manual_wcs"
                ] = wcs_solution

                if wcs_solution is not None:

                    print(
                        f"{fname}: astrometry.net WCS solved "
                        f"successfully ({used_method} method)."
                    )

                    dict_aper[fname][
                        "astrometry_method_used"
                    ] = used_method

                else:

                    print(
                        f"{fname}: astrometry.net WCS solve failed "
                        "(both source-list and image-upload methods)."
                    )

            # =================================================
            # MANUAL ASTROMETRY
            # =================================================

            elif astrometry_settings["method"] == "manual":

                if fname != reference_fname:

                    print(
                        f"{fname}: manual astrometry will use "
                        "the reference-image workflow."
                    )

                reviewed_candidates = (
                    review_astrometric_candidates(
                        fname,
                        dict_images[fname],
                        candidates
                    )
                )

                if (
                    reviewed_candidates is None
                    or len(reviewed_candidates) == 0
                ):

                    print(
                        f"{fname}: no reviewed candidates "
                        "available for manual WCS."
                    )

                    dict_aper[fname][
                        "manual_wcs"
                    ] = None

                    continue

                wcs_solution = run_manual_wcs(
                    fname,
                    dict_images[fname],
                    reviewed_candidates
                )

                dict_aper[fname][
                    "manual_wcs"
                ] = wcs_solution

                if wcs_solution is not None:

                    print(
                        f"{fname}: manual WCS solution "
                        "created successfully."
                    )

                    dict_aper[fname][
                        "astrometry_method_used"
                    ] = "manual"

                else:

                    print(
                        f"{fname}: manual WCS solution failed."
                    )

    # =========================================================
    # AUTOMATIC ASTROMETRY FALLBACK
    # =========================================================

    if (
        perform_astrometry
        and astrometry_settings["method"] == "astrometry_net"
    ):

        image_names = list(dict_images.keys())

        all_failed = all(
            dict_aper[fname].get("manual_wcs") is None
            for fname in image_names
        )

        if all_failed:

            print()
            print("=================================================")
            print("ASTROMETRY FALLBACK")
            print("=================================================")

            print(
                "All initial astrometry.net solves failed."
            )

            # -------------------------------------------------
            # TRY POSITION-HINT RETRY
            # -------------------------------------------------

            from steps_astrometry import (
                solve_with_astrometry_net_position_hint
            )

            position_hint_solutions = {}

            for fname in image_names:

                print(
                    f"{fname}: trying position-hint retry..."
                )

                candidates = dict_aper[fname].get(
                    "astrometric_candidates"
                )

                if candidates is None:
                    continue

                wcs_solution = (
                    solve_with_astrometry_net_position_hint(
                        sources=candidates,
                        image_shape=dict_images[fname][
                            "data"
                        ].shape,
                        api_key=astrometry_settings["api_key"],
                        scale_lower_arcmin=SCALE_LOWER_ARCMIN,
                        scale_upper_arcmin=SCALE_UPPER_ARCMIN
                    )
                )

                if wcs_solution is not None:

                    position_hint_solutions[fname] = (
                        wcs_solution
                    )

                    dict_aper[fname][
                        "manual_wcs"
                    ] = wcs_solution

                    dict_aper[fname][
                        "astrometry_method_used"
                    ] = "position_hint"

                    print(
                        f"{fname}: position-hint solve succeeded."
                    )

            all_failed = all(
                dict_aper[fname].get("manual_wcs") is None
                for fname in image_names
            )

            # -------------------------------------------------
            # MANUAL REFERENCE FALLBACK
            # -------------------------------------------------

            if all_failed:

                print(
                    "Position-hint retry did not produce a "
                    "complete astrometric solution."
                )

                print(
                    "Falling back to manual solution for "
                    "the reference image."
                )

                reference_candidates = dict_aper[
                    reference_fname
                ].get("astrometric_candidates")

                if (
                    reference_candidates is not None
                    and len(reference_candidates) > 0
                ):

                    from steps_astrometry import (
                        review_astrometric_candidates,
                        run_manual_wcs
                    )

                    reviewed_candidates = (
                        review_astrometric_candidates(
                            reference_fname,
                            dict_images[reference_fname],
                            reference_candidates
                        )
                    )

                    if (
                        reviewed_candidates is not None
                        and len(reviewed_candidates) > 0
                    ):

                        reference_wcs = run_manual_wcs(
                            reference_fname,
                            dict_images[reference_fname],
                            reviewed_candidates
                        )

                        if reference_wcs is not None:

                            dict_aper[
                                reference_fname
                            ]["manual_wcs"] = reference_wcs

                            dict_aper[
                                reference_fname
                            ]["astrometry_method_used"] = (
                                "manual_reference"
                            )

                            print(
                                f"{reference_fname}: manual "
                                "reference WCS succeeded."
                            )

                            # ---------------------------------
                            # USE REFERENCE CENTER AS HINT
                            # ---------------------------------

                            reference_data = dict_images[
                                reference_fname
                            ]["data"]

                            ref_y, ref_x = (
                                np.asarray(
                                    reference_data.shape
                                )[:2] / 2.0
                            )

                            reference_sky = (
                                reference_wcs.pixel_to_world(
                                    ref_x,
                                    ref_y
                                )
                            )

                            hint_ra = reference_sky.ra.deg
                            hint_dec = reference_sky.dec.deg

                            print(
                                f"Reference center: "
                                f"RA={hint_ra:.6f}, "
                                f"Dec={hint_dec:.6f}"
                            )

                            # -----------------------------
                            # SOLVE REMAINING IMAGES
                            # -----------------------------

                            for fname in image_names:

                                if (
                                    fname
                                    == reference_fname
                                ):
                                    continue

                                if (
                                    dict_aper[fname].get(
                                        "manual_wcs"
                                    )
                                    is not None
                                ):
                                    continue

                                candidates = (
                                    dict_aper[fname].get(
                                        "astrometric_candidates"
                                    )
                                )

                                if (
                                    candidates is None
                                    or len(candidates) == 0
                                ):
                                    continue

                                from steps_astrometry import (
                                    solve_with_astrometry_net_position_hint
                                )

                                print(
                                    f"{fname}: solving using "
                                    "manual reference center..."
                                )

                                wcs_solution = (
                                    solve_with_astrometry_net_position_hint(
                                        sources=candidates,
                                        image_shape=dict_images[
                                            fname
                                        ]["data"].shape,
                                        api_key=(
                                            astrometry_settings[
                                                "api_key"
                                            ]
                                        ),
                                        scale_lower_arcmin=(
                                            SCALE_LOWER_ARCMIN
                                        ),
                                        scale_upper_arcmin=(
                                            SCALE_UPPER_ARCMIN
                                        ),
                                        ra_hint=hint_ra,
                                        dec_hint=hint_dec
                                    )
                                )

                                if wcs_solution is not None:

                                    dict_aper[fname][
                                        "manual_wcs"
                                    ] = wcs_solution

                                    dict_aper[fname][
                                        "astrometry_method_used"
                                    ] = (
                                        "reference_position_hint"
                                    )

                                    print(
                                        f"{fname}: reference-hint "
                                        "solve succeeded."
                                    )

    # =========================================================
    # SOURCE COUNT
    # =========================================================

    image_names = list(dict_images.keys())

    reference_sources = dict_aper[
        image_names[0]
    ].get("sources found")

    if reference_sources is not None:
        source_count = len(reference_sources)
    else:
        source_count = 0

    print()
    print(
        f"Reference-image source count: {source_count}"
    )

    # =========================================================
    # APERTURE PHOTOMETRY
    # =========================================================
    #
    # build_photometry_table() now:
    #
    #   - uses the per-image FWHM
    #   - calculates aperture flux
    #   - calculates instrumental magnitude
    #   - calculates instrumental magnitude error
    #   - adds RA/Dec when WCS exists

    # =========================================================

    print()
    print("=================================================")
    print("APERTURE PHOTOMETRY")
    print("=================================================")

    dict_aper = build_photometry_table(
        dict_images,
        dict_aper,
        aperture_mult=1.5
    )

    # =========================================================
    # NO ASTROMETRY
    # =========================================================

    if not perform_astrometry:

        print()
        print("=================================================")
        print("INSTRUMENTAL PHOTOMETRY")
        print("=================================================")

        for fname in image_names:

            table = dict_aper[fname].get(
                "final_aperture_phot_table"
            )

            if table is None:
                print(
                    f"{fname}: no photometry table to save."
                )
                continue

            fwhm = dict_aper[fname].get(
                "fwhm",
                None
            )

            aperture_radius = dict_aper[fname].get(
                "aperture_radius_used",
                None
            )

            save_photometry_table(
                photometry_table=table,
                original_filename=fname,
                output_dir=images_dir,
                zp=None,
                zp_sigma=None,
                n_zp_stars=0,
                fwhm=fwhm,
                aperture_radius=aperture_radius,
                astrometry_method="none",
                photsys="INSTRUMENTAL"
            )

        # -----------------------------------------------------
        # DISPLAY FINAL TABLES
        # -----------------------------------------------------

        print()
        print("=================================================")
        print("FINAL INSTRUMENTAL PHOTOMETRY TABLES")
        print("=================================================")

        for fname in image_names:

            table = dict_aper[fname].get(
                "final_aperture_phot_table"
            )

            if table is None:
                continue

            print()
            print(fname)
            print(table)

        print()
        print("Pipeline complete.")
        return

    # =========================================================
    # ASTROMETRIC IMAGE SAVING
    # =========================================================

    for fname in image_names:

        wcs = dict_aper[fname].get(
            "manual_wcs",
            None
        )

        if wcs is None:
            print(
                f"{fname}: no WCS available; "
                "astrometric image not saved."
            )
            continue

        save_astrometric_image(
            image_path=dict_images[fname]["path"],
            output_dir=images_dir,
            wcs=wcs
        )

    # =========================================================
    # PHOTOMETRIC CALIBRATION
    # =========================================================

    from steps_zeropoint import (
        query_ps1_catalog,
        match_sources_to_ps1,
        calculate_zeropoint,
        apply_zeropoint
    )

    print()
    print("=================================================")
    print("PHOTOMETRIC CALIBRATION")
    print("=================================================")

    for fname in image_names:

        table = dict_aper[fname].get(
            "final_aperture_phot_table"
        )

        if table is None:
            print(
                f"{fname}: no photometry table; "
                "skipping calibration."
            )
            continue

        if "RA" not in table.colnames or "Dec" not in table.colnames:

            print(
                f"{fname}: RA/Dec unavailable; "
                "skipping PS1 calibration."
            )
            continue

        wcs = dict_aper[fname].get(
            "manual_wcs",
            None
        )

        if wcs is None:

            print(
                f"{fname}: no WCS available; "
                "skipping PS1 calibration."
            )
            continue

        # -----------------------------------------------------
        # FILTER
        # -----------------------------------------------------

        try:

            filter_name = get_filter_from_header(
                dict_images[fname]["header"]
            )

        except Exception as exc:

            print(
                f"{fname}: could not determine filter: {exc}"
            )
            continue

        if filter_name not in PS1_SUPPORTED_FILTERS:

            print(
                f"{fname}: filter '{filter_name}' is not "
                "supported for PS1 calibration."
            )

            continue

        print(
            f"{fname}: calibrating filter "
            f"'{filter_name}'."
        )

        # -----------------------------------------------------
        # FIELD CENTER
        # -----------------------------------------------------

        ra_center = np.nanmean(
            np.asarray(
                table["RA"],
                dtype=float
            )
        )

        dec_center = np.nanmean(
            np.asarray(
                table["Dec"],
                dtype=float
            )
        )

        if (
            not np.isfinite(ra_center)
            or not np.isfinite(dec_center)
        ):

            print(
                f"{fname}: invalid field center; "
                "skipping calibration."
            )
            continue

        # -----------------------------------------------------
        # QUERY PS1
        # -----------------------------------------------------

        ps1_table = query_ps1_catalog(
            ra_center,
            dec_center
        )

        if ps1_table is None or len(ps1_table) == 0:

            print(
                f"{fname}: no PS1 catalog sources found."
            )
            continue

        # -----------------------------------------------------
        # MATCH SOURCES
        # -----------------------------------------------------

        matched = match_sources_to_ps1(
            table,
            ps1_table,
            filter_name
        )

        if matched is None or len(matched) == 0:

            print(
                f"{fname}: no usable PS1 matches."
            )
            continue

        # -----------------------------------------------------
        # ZERO POINT
        # -----------------------------------------------------

        zp, zp_err, n_zp_stars = (
            calculate_zeropoint(
                matched,
                filter_name
            )
        )

        if zp is None or not np.isfinite(zp):

            print(
                f"{fname}: zero-point calculation failed."
            )
            continue

        print(
            f"{fname}: ZP = {zp:.4f} "
            f"+/- {zp_err:.4f} "
            f"({n_zp_stars} stars)"
        )

        # -----------------------------------------------------
        # APPLY ZERO POINT
        # -----------------------------------------------------

        calibrated_table = apply_zeropoint(
            table,
            zp,
            zp_err
        )

        dict_aper[fname][
            "calibrated_phot_table"
        ] = calibrated_table

        # -----------------------------------------------------
        # SAVE CALIBRATED TABLE
        # -----------------------------------------------------

        fwhm = dict_aper[fname].get(
            "fwhm",
            None
        )

        aperture_radius = dict_aper[fname].get(
            "aperture_radius_used",
            None
        )

        astrometry_method = dict_aper[fname].get(
            "astrometry_method_used",
            "unknown"
        )

        save_photometry_table(
            table=calibrated_table,
            image_filename=fname,
            output_dir=images_dir,
            zp=zp,
            zp_err=zp_err,
            n_zp_stars=n_zp_stars,
            fwhm=fwhm,
            aperture_radius=aperture_radius,
            astrometry_method=astrometry_method,
            photsys="PS1"
        )

    # =========================================================
    # DISPLAY FINAL TABLES
    # =========================================================

    print()
    print("=================================================")
    print("FINAL PHOTOMETRY TABLES")
    print("=================================================")

    for fname in image_names:

        calibrated_table = dict_aper[fname].get(
            "calibrated_phot_table",
            None
        )

        if calibrated_table is not None:

            print()
            print(fname)
            print(calibrated_table)

        else:

            table = dict_aper[fname].get(
                "final_aperture_phot_table",
                None
            )

            if table is not None:

                print()
                print(
                    f"{fname} "
                    "(instrumental / uncalibrated):"
                )

                print(table)

    # =========================================================
    # FINAL SUMMARY
    # =========================================================

    print()
    print("=================================================")
    print("PIPELINE COMPLETE")
    print("=================================================")

    for fname in image_names:

        sources = dict_aper[fname].get(
            "sources found"
        )

        fwhm = dict_aper[fname].get(
            "fwhm"
        )

        wcs = dict_aper[fname].get(
            "manual_wcs"
        )

        if sources is None:
            n_sources = 0
        else:
            n_sources = len(sources)

        if fwhm is not None:
            fwhm_text = f"{fwhm:.2f}px"
        else:
            fwhm_text = "N/A"

        if wcs is not None:
            wcs_text = "yes"
        else:
            wcs_text = "no"

        print(
            f"{fname}: "
            f"sources={n_sources}, "
            f"FWHM={fwhm_text}, "
            f"WCS={wcs_text}"
        )


if __name__ == "__main__":
    main()
