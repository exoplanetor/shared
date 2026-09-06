import os
import numpy as np
from config import ASTROMETRY_API_KEY, SCALE_LOWER_ARCMIN, SCALE_UPPER_ARCMIN
from steps_photometry import (
    initialize_containers,
    find_fits_files,
    load_fits_images,
    create_dict_aper,
    find_stars,
    estimate_fwhm_ensemble,
    build_photometry_table,
    save_photometry_table,
    get_filter_from_header,
    PS1_SUPPORTED_FILTERS
)
from steps_astrometry import (
    select_astrometric_candidates,
    show_astrometric_candidates,
    enter_star_coordinates,
    create_manual_wcs,
    run_manual_astrometry,
    review_astrometric_candidates,
    get_position_hint_from_user,
    solve_with_astrometry_net
)
from steps_zeropoint import (
    query_field_catalog,
    match_sources_to_catalog,
    compute_zeropoint,
    apply_zeropoint
)
# -----------------------------
# USER FILTER SETTINGS
# -----------------------------
def get_filter_params_from_user():
    print("\n--- Source detection & cleaning settings ---")
    use_custom = input(
        "Adjust source detection and cleaning parameters? (y/n): "
    ).strip().lower()
    if use_custom != "y":
        return None
    threshold_sigma = input("threshold_sigma (default 3): ").strip() or 3
    smoothing_sigma = input("smoothing_sigma (default 0): ").strip() or 0
    min_area = input("min_area (default 10): ").strip() or 10
    connectivity = input("connectivity (default 8): ").strip() or 8
    peak_to_total_thresh = input("peak_to_total_thresh (default 0.2): ").strip() or 0.2
    hot_pixel_sigma_thresh = input("hot_pixel_sigma_thresh (default 1e10): ").strip() or 1e10
    merge_close = input("merge_close (default False): ").strip().lower() == "true"
    merge_radius = input("merge_radius (default 5): ").strip() or 5
    return {
        "threshold_sigma": float(threshold_sigma),
        "smoothing_sigma": float(smoothing_sigma),
        "min_area": float(min_area),
        "connectivity": int(connectivity),
        "peak_to_total_thresh": float(peak_to_total_thresh),
        "hot_pixel_sigma_thresh": float(hot_pixel_sigma_thresh),
        "merge_close": merge_close,
        "merge_radius": float(merge_radius),
    }

# -----------------------------
# USER ASTROMETRY METHOD CHOICE
# -----------------------------
def get_astrometry_settings():
    print("\n--- Astrometry method ---")
    print("1) astrometry.net (automatic, uses saved API key)")
    print("2) manual (type RA/Dec for a few stars)")
    choice = input("Choose method [1/2] (default 2): ").strip() or "2"
    if choice == "1":
        return {"method": "astrometry_net", "api_key": ASTROMETRY_API_KEY}
    return {"method": "manual", "api_key": None}

# -----------------------------
# MAIN PIPELINE
# -----------------------------
def main():
    images_dir = input(
        "Enter full path to folder containing calibrated FITS images: "
    ).strip()
    if not images_dir or not os.path.isdir(images_dir):
        print("Invalid directory.")
        return
    # init containers
    containers = initialize_containers()
    # load images once
    images = find_fits_files(images_dir)
    dict_images = load_fits_images(images)
    dict_aper = create_dict_aper(images)
    filter_params = get_filter_params_from_user()
    astrometry_settings = get_astrometry_settings()
    reference_filename = list(dict_images.keys())[0]
    # -----------------------------
    # STAR DETECTION
    # -----------------------------
    for fname in dict_images.keys():
        image_data = dict_images[fname]["data"]
        found_stars = find_stars(
            image_path=dict_images[fname]["path"],
            image_index=0,
            threshold_factor=6.0,
            apply_filter=True,
            filter_params=filter_params
        )
        fwhm_measured, n_samples_used = estimate_fwhm_ensemble(
            found_stars,
            image_data
        )
        if fwhm_measured is not None:
            print(
                f"{fname}: FWHM (median from {n_samples_used} stars) = {fwhm_measured:.2f} px"
            )
        else:
            print(
                f"{fname}: FWHM estimation failed"
            )
        # Store detections
        dict_aper[fname]["sources found"] = found_stars
        dict_aper[fname]["fwhm"] = fwhm_measured

        # ---------------------------------
        # ASTROMETRIC CANDIDATE SELECTION
        # ---------------------------------
        print(
            f"{fname}: stars={found_stars is not None}, fwhm={fwhm_measured}"
        )
        if found_stars is not None and fwhm_measured is not None:
            candidates = select_astrometric_candidates(
                sources=found_stars,
                image_shape=image_data.shape,
                fwhm=fwhm_measured
            )
            dict_aper[fname]["astrometric candidates"] = candidates
            if candidates is not None:
                print(
                    candidates["xcentroid", "ycentroid", "flux"]
                )
                print(
                    f"{fname}: {len(candidates)} astrometric candidates selected"
                )
            else:
                print(
                    f"{fname}: no astrometric candidates found"
                )

            # ---------------------------------
            # WCS CREATION
            # astrometry.net: every image, independently solved
            # manual: reference image only
            # ---------------------------------
            if astrometry_settings["method"] == "astrometry_net":
                wcs_solution = solve_with_astrometry_net(
                    sources=found_stars,
                    image_shape=image_data.shape,
                    api_key=astrometry_settings["api_key"],
                    scale_lower_arcmin=SCALE_LOWER_ARCMIN,
                    scale_upper_arcmin=SCALE_UPPER_ARCMIN
                )
                dict_aper[fname]["manual_wcs"] = wcs_solution
                if wcs_solution is not None:
                    print(f"{fname}: astrometry.net WCS solved successfully.")
                else:
                    print(f"{fname}: astrometry.net WCS solve failed.")
            elif astrometry_settings["method"] == "manual" and fname == reference_filename:
                if candidates is not None:
                    reviewed_candidates = review_astrometric_candidates(
                        found_stars,
                        image_data,
                        fwhm_measured
                    )
                    dict_aper[fname]["astrometric candidates"] = reviewed_candidates
                    if reviewed_candidates is not None:
                        manual_wcs = run_manual_astrometry(
                            image_data,
                            reviewed_candidates,
                            title=fname
                        )
                        dict_aper[fname]["manual_wcs"] = manual_wcs
                    else:
                        print(f"{fname}: no candidates left after review — skipping manual WCS.")
                        dict_aper[fname]["manual_wcs"] = None
        else:
            dict_aper[fname]["astrometric candidates"] = None

    # -----------------------------
    # AUTOMATIC SOLVE FALLBACK
    # If astrometry.net failed on EVERY image, offer a guided retry
    # (position hint from one identified star), then full manual as
    # a last resort.
    # -----------------------------
    if astrometry_settings["method"] == "astrometry_net":
        n_solved = sum(
            1 for fname in dict_images.keys()
            if dict_aper[fname].get("manual_wcs") is not None
        )
        if n_solved == 0:
            print("\n" + "=" * 80)
            print(f"astrometry.net failed on all {len(dict_images)} images.")
            print("=" * 80)
            try_hint = input(
                "Identify one star to provide a position hint and retry? (y/n): "
            ).strip().lower()

            if try_hint == "y":
                ref_candidates = dict_aper[reference_filename].get("astrometric candidates")
                ref_data = dict_images[reference_filename]["data"]
                hint = get_position_hint_from_user(ref_data, ref_candidates)

                if hint is not None:
                    ra_hint, dec_hint = hint
                    search_radius_arcmin = SCALE_UPPER_ARCMIN * 4  # covers FOV + dithering, auto-scales with optics

                    print(f"\nRetrying all images with position hint "
                          f"(RA={ra_hint:.5f}, Dec={dec_hint:.5f}, "
                          f"radius={search_radius_arcmin:.1f} arcmin)...\n")

                    for fname in dict_images.keys():
                        found_stars = dict_aper[fname].get("sources found")
                        image_data = dict_images[fname]["data"]
                        if found_stars is None:
                            continue
                        wcs_solution = solve_with_astrometry_net(
                            sources=found_stars,
                            image_shape=image_data.shape,
                            api_key=astrometry_settings["api_key"],
                            scale_lower_arcmin=SCALE_LOWER_ARCMIN,
                            scale_upper_arcmin=SCALE_UPPER_ARCMIN,
                            ra=ra_hint,
                            dec=dec_hint,
                            search_radius_arcmin=search_radius_arcmin
                        )
                        dict_aper[fname]["manual_wcs"] = wcs_solution
                        if wcs_solution is not None:
                            print(f"{fname}: astrometry.net WCS solved successfully (with position hint).")
                        else:
                            print(f"{fname}: astrometry.net WCS solve failed (with position hint).")

                    n_solved = sum(
                        1 for fname in dict_images.keys()
                        if dict_aper[fname].get("manual_wcs") is not None
                    )

            if n_solved == 0:
                print("\n" + "=" * 80)
                print("Automatic solving still failed on all images.")
                print("=" * 80)
                try_manual = input(
                    "Switch to manual astrometry for the reference image instead? (y/n): "
                ).strip().lower()
                if try_manual == "y":
                    ref_candidates = dict_aper[reference_filename].get("astrometric candidates")
                    ref_data = dict_images[reference_filename]["data"]
                    if ref_candidates is not None:
                        reviewed_candidates = review_astrometric_candidates(
                            dict_aper[reference_filename]["sources found"],
                            ref_data,
                            dict_aper[reference_filename]["fwhm"]
                        )
                        dict_aper[reference_filename]["astrometric candidates"] = reviewed_candidates
                        if reviewed_candidates is not None:
                            manual_wcs = run_manual_astrometry(
                                ref_data,
                                reviewed_candidates,
                                title=reference_filename
                            )
                            dict_aper[reference_filename]["manual_wcs"] = manual_wcs

                            # -----------------------------------------------
                            # TIER 3 FOLLOW-UP: use the just-fitted manual WCS
                            # as a position hint to auto-retry astrometry.net
                            # on every OTHER image in the batch, so the whole
                            # batch isn't reduced to a single calibrated frame.
                            # -----------------------------------------------
                            if manual_wcs is not None:
                                h, w = ref_data.shape
                                center_sky = manual_wcs.pixel_to_world(w / 2, h / 2)
                                ra_hint = center_sky.ra.deg
                                dec_hint = center_sky.dec.deg
                                search_radius_arcmin = SCALE_UPPER_ARCMIN * 4

                                print(f"\nUsing manual WCS as a position hint "
                                      f"(RA={ra_hint:.5f}, Dec={dec_hint:.5f}, "
                                      f"radius={search_radius_arcmin:.1f} arcmin) "
                                      "to auto-solve the remaining images...\n")

                                for fname in dict_images.keys():
                                    if fname == reference_filename:
                                        continue  # already solved manually
                                    found_stars = dict_aper[fname].get("sources found")
                                    image_data = dict_images[fname]["data"]
                                    if found_stars is None:
                                        continue
                                    wcs_solution = solve_with_astrometry_net(
                                        sources=found_stars,
                                        image_shape=image_data.shape,
                                        api_key=ASTROMETRY_API_KEY,
                                        scale_lower_arcmin=SCALE_LOWER_ARCMIN,
                                        scale_upper_arcmin=SCALE_UPPER_ARCMIN,
                                        ra=ra_hint,
                                        dec=dec_hint,
                                        search_radius_arcmin=search_radius_arcmin
                                    )
                                    dict_aper[fname]["manual_wcs"] = wcs_solution
                                    if wcs_solution is not None:
                                        print(f"{fname}: astrometry.net WCS solved successfully "
                                              "(using manual reference as hint).")
                                    else:
                                        print(f"{fname}: astrometry.net WCS solve failed "
                                              "(using manual reference as hint). "
                                              "This image will not have RA/Dec or zero-point calibration.")
                        else:
                            print(f"{reference_filename}: no candidates left after review — skipping manual WCS.")
                    else:
                        print(f"{reference_filename}: no candidates available for manual astrometry.")

    # -----------------------------
    # SOURCE COUNT SUMMARY
    # -----------------------------
    n_sources = (
        len(dict_aper[reference_filename]["sources found"])
        if dict_aper[reference_filename]["sources found"] is not None
        else 0
    )
    print(
        f"\nDetected sources (reference): {n_sources}"
    )
    # -----------------------------
    # PHOTOMETRY TABLE
    # -----------------------------
    dict_aper = build_photometry_table(
        dict_images=dict_images,
        dict_aper=dict_aper,
        range_width=1.0
    )

    # -----------------------------
    # ZERO-POINT CALIBRATION
    # (Pan-STARRS DR2, per image with a valid WCS + photometry table)
    # Filter is read from each image's own FITS header (FILTER keyword),
    # since different images in the same batch may use different filters.
    # -----------------------------
    print("\n" + "=" * 80)
    print("Zero-point calibration (Pan-STARRS DR2)")
    print("=" * 80)
    for fname in dict_images.keys():
        table = dict_aper[fname].get("final_aperture_phot_table", None)
        if table is None or "RA" not in table.colnames:
            print(f"{fname}: no RA/Dec available — skipping zero-point.")
            continue

        # Missing FILTER keyword is a hard stop - filter identity must be
        # known before zero-point calibration can be trusted.
        filt = get_filter_from_header(dict_images[fname]["header"], filename=fname)

        if filt not in PS1_SUPPORTED_FILTERS:
            print(f"{fname}: FILTER='{filt}' is not supported by Pan-STARRS DR2 "
                  f"(supported: {sorted(PS1_SUPPORTED_FILTERS)}). "
                  "Skipping zero-point for this image.")
            continue

        ra_center = float(np.mean(table["RA"]))
        dec_center = float(np.mean(table["Dec"]))

        catalog = query_field_catalog(ra_center, dec_center, radius_arcmin=2.0, ps1_filter=filt)
        if catalog is None:
            print(f"{fname}: no Pan-STARRS catalog data — skipping zero-point.")
            continue

        matched = match_sources_to_catalog(table, catalog, ps1_filter=filt, max_sep_arcsec=1.0)
        zp, zp_sigma, n_used = compute_zeropoint(matched)

        dict_aper[fname]["zeropoint"] = zp
        dict_aper[fname]["zeropoint_sigma"] = zp_sigma
        dict_aper[fname]["filter"] = filt

        table = apply_zeropoint(table, zp)
        dict_aper[fname]["final_aperture_phot_table"] = table
        print(f"{fname}: zero-point applied (filter={filt}).")

        # -----------------------------
        # SAVE RESULTS TO DISK
        # (FITS binary table, zero-point + provenance in header)
        # -----------------------------
        save_photometry_table(
            photometry_table=table,
            output_dir=images_dir,
            original_filename=fname,
            original_path=dict_images[fname]["path"],
            zp=zp,
            zp_sigma=zp_sigma,
            n_zp_stars=n_used,
            fwhm=dict_aper[fname].get("fwhm"),
            aperture_radius=dict_aper[fname].get("aperture_radius_used"),
            astrometry_method=astrometry_settings["method"],
            photsys=f"PS1-{filt}"
        )
        print()

    # -----------------------------
    # DISPLAY TABLES
    # -----------------------------
    any_table = False
    for fname in dict_images.keys():
        table = dict_aper[fname].get(
            "final_aperture_phot_table",
            None
        )
        print("\n" + "=" * 80)
        print(
            f"Photometry Table: {fname}"
        )
        print("=" * 80)
        if table is None:
            print(
                "NO TABLE GENERATED (no sources)"
            )
            continue
        if len(table) == 0:
            print(
                "EMPTY TABLE"
            )
            continue
        any_table = True
        table.pprint(
            max_width=-1,
            max_lines=-1
        )
    if not any_table:
        print(
            "\nWARNING: No photometry tables were produced."
        )

    # -----------------------------
    # FINAL SUMMARY
    # -----------------------------
    print(
        f"\nDone: {n_sources} sources → FWHM computed → photometry tables built"
    )
# -----------------------------
# ENTRY POINT
# -----------------------------
if __name__ == "__main__":
    main()
