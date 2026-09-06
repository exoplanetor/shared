import numpy as np

def select_astrometric_candidates(
        sources,
        image_shape,
        fwhm,
        n_candidates=10,
        merge_factor=1.5,
        edge_factor=2.0
):
    if sources is None or len(sources) == 0:
        return None
    h, w = image_shape
    # ---------------------------------
    # Extract positions
    # ---------------------------------
    x = np.array(sources["xcentroid"])
    y = np.array(sources["ycentroid"])
    flux = (
        np.array(sources["flux"])
        if "flux" in sources.colnames
        else np.ones(len(sources))
    )
    # ---------------------------------
    # Remove edge sources
    # ---------------------------------
    edge_margin = edge_factor * fwhm
    good = (
        (x > edge_margin) &
        (x < w-edge_margin) &
        (y > edge_margin) &
        (y < h-edge_margin)
    )

    sources = sources[good]
    if len(sources) == 0:
        return None
    x = np.array(sources["xcentroid"])
    y = np.array(sources["ycentroid"])
    flux = np.array(sources["flux"])
    # ---------------------------------
    # Merge duplicate detections
    # ---------------------------------
    
    merge_radius = merge_factor * fwhm
    
    positions = np.column_stack((x, y))
    visited = np.zeros(len(sources), dtype=bool)
    keep = []
    
    for i in range(len(sources)):
    
        if visited[i]:
            continue
    
        # Start a group with star i
        group = [i]
        visited[i] = True
    
        changed = True
    
        # Expand the group until no more nearby stars are found
        while changed:
    
            changed = False
    
            for j in range(len(sources)):
    
                if visited[j]:
                    continue
    
                # Is star j close to ANY star already in the group?
                for k in group:
    
                    distance = np.linalg.norm(
                        positions[j] - positions[k]
                    )
    
                    if distance < merge_radius:
                        group.append(j)
                        visited[j] = True
                        changed = True
                        break
    
        # Keep only the brightest member of the group
        brightest = group[np.argmax(flux[group])]
        keep.append(brightest)
    
    sources = sources[keep]
    # ---------------------------------
    # Keep only reasonably bright stars
    # ---------------------------------
    
    flux = np.asarray(sources["flux"])
    
    max_flux = np.max(flux)
    
    bright = flux >= 0.05 * max_flux    # keep stars brighter than 5% of the brightest
    
    sources = sources[bright]
    
    # ---------------------------------
    # Rank astrometric candidates
    # ---------------------------------
    
    flux = np.asarray(sources["flux"])
    x = np.asarray(sources["xcentroid"])
    y = np.asarray(sources["ycentroid"])
    
    height, width = image_shape
    
    # -------------------------
    # Flux score
    # -------------------------
    flux_score = flux / max(np.max(flux), 1e-12)
    
    # -------------------------
    # Isolation score
    # -------------------------
    positions = np.column_stack((x, y))
    
    nearest = np.full(len(sources), np.inf)
    
    for i in range(len(sources)):
    
        d = np.sqrt(np.sum((positions - positions[i])**2, axis=1))
    
        d[i] = np.inf
    
        nearest[i] = np.min(d)
    
    isolation_score = nearest / max(np.max(nearest), 1e-12)
    
    # -------------------------
    # Edge score
    # -------------------------
    edge_distance = np.minimum.reduce([
        x,
        y,
        width - x,
        height - y
    ])
    
    edge_score = edge_distance / max(np.max(edge_distance), 1e-12)
    
    # -------------------------
    # Final score
    # -------------------------
    score = (
        0.30 * flux_score +
        0.40 * isolation_score +
        0.30 * edge_score
    )
    
    sources["astrometry_score"] = score
    
    idx = np.argsort(score)[::-1]
    
    sources = sources[idx]
    
    return sources[:n_candidates]

import matplotlib.pyplot as plt
from astropy.visualization import ZScaleInterval
def show_astrometric_candidates(
        image_data,
        candidates,
        title="Astrometric candidates"
):
    if candidates is None:
        print("No candidates to display")
        return
    interval = ZScaleInterval()
    vmin, vmax = interval.get_limits(image_data)
    plt.figure(figsize=(10,10))
    plt.imshow(
        image_data,
        cmap="gray",
        origin="lower",
        vmin=vmin,
        vmax=vmax
    )
    x = candidates["xcentroid"]
    y = candidates["ycentroid"]
    plt.scatter(
        x,
        y,
        s=150,
        facecolors="none",
        edgecolors="red"
    )
    for i, (xx, yy) in enumerate(zip(x, y), start=1):
        plt.text(
            xx + 20,
            yy + 20,
            str(i),
            color="yellow",
            fontsize=14
        )
    plt.title(title)
    plt.xlabel("X pixel")
    plt.ylabel("Y pixel")
    plt.show()

import numpy as np
import matplotlib.pyplot as plt
from astropy.wcs.utils import fit_wcs_from_points
from astropy.coordinates import SkyCoord
import astropy.units as u


def _prompt_single_star_coords(candidates, i):
    """
    Prompt for one star's RA/Dec, retrying on parse failure instead of
    crashing the whole entry session. Returns (ra_str, dec_str).
    """
    print(
        f"Star {i+1}: "
        f"x={candidates['xcentroid'][i]:.2f}, "
        f"y={candidates['ycentroid'][i]:.2f}"
    )
    while True:
        ra = input("RA: ").strip()
        dec = input("Dec: ").strip()
        try:
            SkyCoord(ra, dec, unit=(u.hourangle, u.deg))
        except Exception as e:
            print(f"Could not parse that RA/Dec ({e}) — please re-enter this star.")
            continue
        return ra, dec


def enter_star_coordinates(candidates, min_stars=3):
    """
    Interactively collect RA/Dec for each candidate star, with:
      - per-star retry on invalid input (doesn't crash the session)
      - a summary review before committing
      - the ability to redo specific stars by number if something looks wrong

    Requires at least `min_stars` candidates (a 2-star fit is technically
    possible but not trustworthy; 3+ allows the fit quality check in
    create_manual_wcs to mean something).

    Returns a SkyCoord array, or None if there aren't enough candidates.
    """
    n = len(candidates)
    if n < min_stars:
        print(f"Need at least {min_stars} stars for manual astrometry, got {n}.")
        return None

    print("\nEnter RA/Dec for selected stars")
    print("RA format: HH:MM:SS.ss")
    print("Dec format: +/-DD:MM:SS.s\n")

    entries = [None] * n
    for i in range(n):
        entries[i] = _prompt_single_star_coords(candidates, i)

    while True:
        print("\n--- Entered coordinates ---")
        for i in range(n):
            print(
                f"{i + 1}: x={candidates['xcentroid'][i]:.2f}, "
                f"y={candidates['ycentroid'][i]:.2f}  ->  "
                f"RA={entries[i][0]}, Dec={entries[i][1]}"
            )
        confirm = input("\nConfirm these entries? (y/n): ").strip().lower()
        if confirm == "y":
            break

        redo = input(
            "Enter the star numbers to redo, comma-separated (e.g. 2, 4): "
        ).strip()
        if not redo:
            continue
        try:
            redo_positions = [int(v.strip()) - 1 for v in redo.split(",") if v.strip()]
        except ValueError:
            print("Could not parse that input — please enter numbers separated by commas.")
            continue
        for pos in redo_positions:
            if 0 <= pos < n:
                entries[pos] = _prompt_single_star_coords(candidates, pos)
            else:
                print(f"Ignoring invalid number: {pos + 1}")

    ra_list = [e[0] for e in entries]
    dec_list = [e[1] for e in entries]
    return SkyCoord(
        ra_list,
        dec_list,
        unit=(u.hourangle, u.deg)
    )
    
def create_manual_wcs(
        candidates,
        sky_coords
):
    xy = (
        np.array(candidates["xcentroid"]),
        np.array(candidates["ycentroid"])
    )
    wcs = fit_wcs_from_points(
        xy,
        sky_coords
    )
    return wcs


def check_manual_wcs_fit(wcs, candidates, sky_coords):
    """
    Fit quality check: run the candidates' own pixel positions back through
    the fitted WCS and compare to what was typed in. If the fit is good,
    these should match almost exactly (the WCS was fit FROM these points,
    so large residuals mean something is wrong with an entry, not the fit
    method itself).

    Returns (residuals_arcsec, rms_arcsec) — residuals_arcsec is a per-star
    array, rms_arcsec is the overall root-mean-square residual.
    """
    predicted = wcs.pixel_to_world(
        candidates["xcentroid"],
        candidates["ycentroid"]
    )
    residuals_arcsec = predicted.separation(sky_coords).to(u.arcsec).value
    rms_arcsec = float(np.sqrt(np.mean(residuals_arcsec ** 2)))
    return residuals_arcsec, rms_arcsec


def run_manual_astrometry(image_data, candidates, title="Reference image",
                          min_stars=3, warn_threshold_arcsec=1.0):
    """
    Manual RA/Dec entry for a few stars, fit a WCS, and check the fit
    quality before accepting it.

    warn_threshold_arcsec : if the RMS residual between typed positions
        and what the fitted WCS predicts exceeds this, warn the user and
        offer to redo the entry (likely indicates a typo in one star's
        coordinates) rather than silently accepting a bad WCS.
    """
    show_astrometric_candidates(image_data, candidates, title=title)
    print("\nReference image selected for manual astrometry.")

    while True:
        sky_coords = enter_star_coordinates(candidates, min_stars=min_stars)
        if sky_coords is None:
            return None

        manual_wcs = create_manual_wcs(candidates, sky_coords)
        residuals_arcsec, rms_arcsec = check_manual_wcs_fit(manual_wcs, candidates, sky_coords)

        print("\n--- Fit quality check ---")
        for i in range(len(candidates)):
            print(f"Star {i + 1}: residual = {residuals_arcsec[i]:.3f} arcsec")
        print(f"RMS residual: {rms_arcsec:.3f} arcsec")

        if rms_arcsec <= warn_threshold_arcsec:
            print("Manual WCS successfully created.")
            return manual_wcs

        print(
            f"\nWARNING: RMS residual ({rms_arcsec:.3f} arcsec) exceeds the "
            f"expected threshold ({warn_threshold_arcsec} arcsec). "
            "This usually means one or more RA/Dec entries has a typo."
        )
        redo = input("Redo the coordinate entry? (y/n, 'n' accepts this fit anyway): ").strip().lower()
        if redo != "y":
            print("Proceeding with this WCS despite the high residual.")
            return manual_wcs
        # else: loop back and redo entry from scratch


def review_astrometric_candidates(
        found_stars,
        image_data,
        fwhm,
        n_candidates=10,
        merge_factor=1.5,
        edge_factor=2.0
):
    """
    Interactive review loop: show candidate stars, let the user reject
    fake ones by their printed number, then reselect from the remaining
    pool and repeat until the user is happy.

    Rejections are tracked by DAOStarFinder's unique 'id' column, so a
    rejected star can never reappear in a later round even if the
    candidate list gets reshuffled.

    Returns the final accepted candidates Table, or None if nothing
    is left to select from.
    """
    excluded_ids = set()

    while True:
        # Remove any previously rejected stars from the full pool
        if excluded_ids:
            remaining = found_stars[~np.isin(found_stars["id"], list(excluded_ids))]
        else:
            remaining = found_stars

        candidates = select_astrometric_candidates(
            sources=remaining,
            image_shape=image_data.shape,
            fwhm=fwhm,
            n_candidates=n_candidates,
            merge_factor=merge_factor,
            edge_factor=edge_factor
        )

        if candidates is None or len(candidates) == 0:
            print("No candidates left to select from.")
            return None

        show_astrometric_candidates(image_data, candidates, title="Review astrometric candidates")

        print("\nCandidate stars:")
        for i in range(len(candidates)):
            print(
                f"{i + 1}: id={candidates['id'][i]}, "
                f"x={candidates['xcentroid'][i]:.2f}, "
                f"y={candidates['ycentroid'][i]:.2f}, "
                f"flux={candidates['flux'][i]:.1f}"
            )

        happy = input("\nHappy with these candidates? (y/n): ").strip().lower()
        if happy == "y":
            return candidates

        to_remove = input(
            "Enter the numbers to remove, comma-separated (e.g. 3, 5): "
        ).strip()
        if not to_remove:
            continue

        try:
            reject_positions = [
                int(v.strip()) - 1 for v in to_remove.split(",") if v.strip()
            ]
        except ValueError:
            print("Could not parse that input — please enter numbers separated by commas.")
            continue

        for pos in reject_positions:
            if 0 <= pos < len(candidates):
                excluded_ids.add(candidates["id"][pos])
            else:
                print(f"Ignoring invalid number: {pos + 1}")


def get_position_hint_from_user(image_data, candidates, title="Identify one star for position hint"):
    """
    Show candidate stars on a plot, let the user pick ONE star by number,
    and type its known RA/Dec. Returns (ra_deg, dec_deg) as plain floats,
    intended to be passed as a position hint (ra, dec) into
    solve_with_astrometry_net — used to narrow a blind search when
    automatic solving has failed on every frame.

    Returns None if the user provides no valid selection.
    """
    if candidates is None or len(candidates) == 0:
        print("No candidates available to pick a position-hint star from.")
        return None

    show_astrometric_candidates(image_data, candidates, title=title)

    print("\nCandidate stars:")
    for i in range(len(candidates)):
        print(
            f"{i + 1}: x={candidates['xcentroid'][i]:.2f}, "
            f"y={candidates['ycentroid'][i]:.2f}"
        )

    choice = input("\nWhich star number do you want to identify? ").strip()
    try:
        pos = int(choice) - 1
    except ValueError:
        print("Invalid selection.")
        return None

    if not (0 <= pos < len(candidates)):
        print("Invalid selection.")
        return None

    print(f"\nStar {pos + 1}: x={candidates['xcentroid'][pos]:.2f}, y={candidates['ycentroid'][pos]:.2f}")
    print("RA format: HH:MM:SS.ss")
    print("Dec format: +/-DD:MM:SS.s")
    ra = input("RA: ").strip()
    dec = input("Dec: ").strip()

    try:
        coord = SkyCoord(ra, dec, unit=(u.hourangle, u.deg))
    except Exception as e:
        print(f"Could not parse RA/Dec: {e}")
        return None

    return coord.ra.deg, coord.dec.deg


from astropy.wcs import WCS
from astroquery.astrometry_net import AstrometryNet


def print_solve_failure_help(error, n_sources, scale_lower, scale_upper):
    """
    Print diagnostic suggestions after a failed astrometry.net solve.
    Tailored based on whether the failure was a connection problem or
    an actual solve failure.
    """
    print("\n--- astrometry.net solve failed: suggestions ---")
    if error is not None and (
        "RemoteDisconnected" in str(error) or "Connection" in str(error)
    ):
        print("This looks like a network/connection issue, not a data problem.")
        print("Suggestion: simply try again — astrometry.net's free service can be intermittent.")
        return
    print(f"Sent {n_sources} sources, scale hint {scale_lower}-{scale_upper} arcmin.")
    if n_sources < 10:
        print("- Very few sources sent. Try lowering threshold_factor in find_stars to detect more stars.")
    print("- Double-check the scale hint matches your actual field of view.")
    print("- If detections look noisy (hot pixels, cosmic rays), tighten source detection/cleaning settings "
          "(the 'y' option at pipeline start).")
    print("- If this keeps failing, try widening the scale range slightly as a sanity check.")


def _is_connection_error(error):
    """True if the exception looks like a transient network/connection issue."""
    return "RemoteDisconnected" in str(error) or "Connection" in str(error)


def solve_with_astrometry_net(
        sources,
        image_shape,
        api_key,
        scale_lower_arcmin=3.0,
        scale_upper_arcmin=3.6,
        ra=None,
        dec=None,
        search_radius_arcmin=None,
        solve_timeout=300,
        max_retries=2,
):
    """
    Plate-solve using astrometry.net's source-list solver.
    Uses the x/y positions we already detected (faster than sending
    the whole image, since it skips astrometry.net's own detection step).

    sources : Table with 'xcentroid', 'ycentroid', ideally 'flux'
    image_shape : (height, width) of the image
    api_key : your nova.astrometry.net API key
    scale_lower_arcmin, scale_upper_arcmin : expected field-of-view width
        range (arcmin). For INO Lens Array + Kepler KL4040: ~3.3 arcmin FOV,
        so defaults are set at 3.0-3.6 arcmin as a tight, accurate hint.
        NOTE: update these if the optical configuration changes (e.g. when
        a focal reducer is in use, changing the FOV to ~8 arcmin).
    ra, dec : optional rough pointing guess (degrees) to narrow the search
    search_radius_arcmin : optional search radius around ra/dec (arcmin)
    max_retries : number of extra attempts if a transient connection error
        occurs (does NOT retry on genuine "could not solve" failures,
        since resending identical data won't change that outcome).

    Returns astropy.wcs.WCS on success, or None if it fails / doesn't solve.
    """
    if sources is None or len(sources) == 0:
        print("No sources to send to astrometry.net.")
        return None

    height, width = image_shape
    x = np.asarray(sources["xcentroid"])
    y = np.asarray(sources["ycentroid"])
    flux = (
        np.asarray(sources["flux"])
        if "flux" in sources.colnames
        else np.ones(len(sources))
    )
    order = np.argsort(flux)[::-1]  # astrometry.net expects brightest-first
    x, y = x[order], y[order]
    n_sources = len(x)

    solve_kwargs = dict(
        image_width=width,
        image_height=height,
        solve_timeout=solve_timeout,
        scale_units="arcminwidth",
        scale_type="ul",
        scale_lower=scale_lower_arcmin,
        scale_upper=scale_upper_arcmin,
    )

    if ra is not None and dec is not None:
        solve_kwargs["center_ra"] = ra
        solve_kwargs["center_dec"] = dec
        if search_radius_arcmin is not None:
            solve_kwargs["radius"] = search_radius_arcmin / 60.0  # degrees

    attempt = 0
    while True:
        attempt += 1
        ast = AstrometryNet()
        ast.api_key = api_key
        try:
            wcs_header = ast.solve_from_source_list(x, y, **solve_kwargs)
        except Exception as e:
            if _is_connection_error(e) and attempt <= max_retries:
                print(f"\nConnection issue (attempt {attempt}/{max_retries + 1}), retrying...")
                continue
            print(f"astrometry.net solve failed: {e}")
            print_solve_failure_help(e, n_sources, scale_lower_arcmin, scale_upper_arcmin)
            return None

        if not wcs_header:
            print("astrometry.net could not solve this field.")
            print_solve_failure_help(None, n_sources, scale_lower_arcmin, scale_upper_arcmin)
            return None

        return WCS(wcs_header)

