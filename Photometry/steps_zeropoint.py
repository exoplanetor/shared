"""
Zero-point calibration using Pan-STARRS DR2 as external truth.

Approach (matches JWST's design, adapted for ground-based data with no
pre-tabulated instrument zero-point):
  1. Query Pan-STARRS DR2 once per image, covering the whole field.
  2. Locally cross-match our detected sources (by RA/Dec) against the
     Pan-STARRS catalog using astropy's vectorized matcher.
  3. For each match: zp_i = catalog_mag - inst_mag
  4. Sigma-clip those offsets to get one robust zero-point (+ uncertainty)
     per image, discarding outlier matches automatically.
  5. Apply: real_mag = inst_mag + zp
"""

import numpy as np
import requests
from astropy.table import Table
from astropy.io import ascii
from astropy.coordinates import SkyCoord
from astropy.coordinates import match_coordinates_sky
from astropy.stats import sigma_clipped_stats
import astropy.units as u

BASEURL = "https://catalogs.mast.stsci.edu/api/v0.1/panstarrs"


def checklegal(table, release):
    if release not in ('dr1', 'dr2'):
        raise ValueError('release must be dr1 or dr2')
    if release == 'dr1' and table not in ('mean', 'stack'):
        raise ValueError('for dr1 table must be mean or stack')
    if release == 'dr2' and table not in ('mean', 'stack', 'detection'):
        raise ValueError('for dr2 table must be mean, stack, or detection')


def ps1search(table='mean', release='dr1', format='csv', columns=None, verbose=False, **kw):
    data = kw.copy()
    if not data:
        raise ValueError('You must specify some parameters for search (e.g., ra, dec, radius)')
    checklegal(table, release)
    if format not in ('csv', 'votable', 'json'):
        raise ValueError('format must be csv, votable, or json')
    url = f"{BASEURL}/{release}/{table}.{format}"
    if columns:
        data['columns'] = '[{}]'.format(','.join(columns))
    r = requests.get(url, params=data, timeout=30)
    r.raise_for_status()
    if format == 'json':
        return r.json()
    else:
        return r.text


def ps1cone(ra, dec, radius_deg, table='mean', release='dr2', format='csv', columns=None, verbose=False, **kw):
    """Cone search of PS1 via MAST catalogs.
    ra, dec in degrees, radius_deg in degrees.
    Returns an astropy Table.
    """
    radius = float(radius_deg)
    params = kw.copy()
    params['ra'] = float(ra)
    params['dec'] = float(dec)
    params['radius'] = radius
    txt = ps1search(table=table, release=release, format=format, columns=columns, verbose=verbose, **params)
    tbl = ascii.read(txt)
    return tbl


def query_field_catalog(ra_center, dec_center, radius_arcmin, ps1_filter='g'):
    """
    Query Pan-STARRS DR2 once for the whole field.

    ra_center, dec_center : degrees — center of the field (e.g. mean RA/Dec
        of your detected sources, or the field center from the WCS)
    radius_arcmin : field search radius, arcmin. Should comfortably cover
        your actual field of view (e.g. INO Lens Array ~3.3 arcmin FOV ->
        use ~2.0 arcmin radius to cover the ~3.3 arcmin diameter field)
    ps1_filter : which PS1 band to pull ('g', 'r', 'i', 'z', 'y')

    Returns an astropy Table with columns: objID, raMean, decMean, and
    the requested filter's mean PSF magnitude column, or None if the
    query fails or returns nothing.
    """
    radius_deg = radius_arcmin / 60.0
    mag_col = f"{ps1_filter}MeanPSFMag"
    columns = ["objID", "raMean", "decMean", "nDetections", mag_col]

    try:
        result = ps1cone(
            ra_center, dec_center, radius_deg,
            table='mean', release='dr2', columns=columns
        )
    except Exception as e:
        print(f"Pan-STARRS query failed: {e}")
        return None

    if result is None or len(result) == 0:
        print("Pan-STARRS query returned no sources for this field.")
        return None

    return result


def match_sources_to_catalog(photometry_table, catalog_table, ps1_filter='g', max_sep_arcsec=1.0):
    """
    Cross-match our detected sources (photometry_table, with RA/Dec columns)
    against a Pan-STARRS catalog table (from query_field_catalog), using
    astropy's vectorized nearest-neighbor matcher.

    Returns a new Table with: x, y, RA, Dec, inst_mag, catalog_mag,
    for only the sources that had a good match within max_sep_arcsec.
    Returns None if no matches were found.
    """
    if catalog_table is None or len(catalog_table) == 0:
        return None
    if "RA" not in photometry_table.colnames or "Dec" not in photometry_table.colnames:
        print("Photometry table has no RA/Dec columns — cannot match to catalog.")
        return None

    mag_col = f"{ps1_filter}MeanPSFMag"
    if mag_col not in catalog_table.colnames:
        print(f"Catalog table has no column {mag_col}.")
        return None

    our_coords = SkyCoord(
        ra=np.asarray(photometry_table["RA"]) * u.deg,
        dec=np.asarray(photometry_table["Dec"]) * u.deg
    )
    cat_coords = SkyCoord(
        ra=np.asarray(catalog_table["raMean"]) * u.deg,
        dec=np.asarray(catalog_table["decMean"]) * u.deg
    )

    idx, d2d, _ = match_coordinates_sky(our_coords, cat_coords)
    max_sep = max_sep_arcsec * u.arcsec
    good = d2d < max_sep

    if np.sum(good) == 0:
        print("No sources matched the Pan-STARRS catalog within the separation limit.")
        return None

    matched = Table()
    matched["x"] = photometry_table["x"][good]
    matched["y"] = photometry_table["y"][good]
    matched["RA"] = photometry_table["RA"][good]
    matched["Dec"] = photometry_table["Dec"][good]
    matched["inst_mag"] = photometry_table["inst_mag"][good]
    matched["catalog_mag"] = np.asarray(catalog_table[mag_col])[idx[good]]

    # Pan-STARRS uses -999 as a "no measurement" sentinel — drop those
    valid = matched["catalog_mag"] > 0
    matched = matched[valid]

    return matched if len(matched) > 0 else None


def compute_zeropoint(matched_table, sigma=3.0):
    """
    Compute a single robust zero-point from a matched source table
    (must have 'inst_mag' and 'catalog_mag' columns).

    Uses the same core idea as JWST's zero-point derivation: take the
    per-star offset (catalog_mag - inst_mag), then sigma-clip to get a
    robust median/mean and standard deviation, automatically discarding
    outlier matches (mismatches, variable stars, blends, etc.).

    Returns (zp, zp_sigma, n_used) — zp_sigma is the clipped std dev,
    n_used is how many matches survived clipping. Returns (None, None, 0)
    if there's nothing usable.
    """
    if matched_table is None or len(matched_table) == 0:
        print("No matched sources available to compute a zero-point.")
        return None, None, 0

    offsets = np.asarray(matched_table["catalog_mag"]) - np.asarray(matched_table["inst_mag"])
    valid = np.isfinite(offsets)
    offsets = offsets[valid]

    if len(offsets) == 0:
        print("No finite offsets available to compute a zero-point.")
        return None, None, 0

    _, zp, zp_sigma = sigma_clipped_stats(offsets, sigma=sigma)
    n_used = np.sum(np.abs(offsets - zp) < sigma * zp_sigma) if zp_sigma > 0 else len(offsets)

    print(f"Zero-point: {zp:.4f} +/- {zp_sigma:.4f} (from {len(offsets)} matches, ~{n_used} after clipping)")
    return zp, zp_sigma, n_used


def apply_zeropoint(photometry_table, zp):
    """
    Add a 'real_mag' column to a photometry table using the given
    zero-point: real_mag = inst_mag + zp.

    Does nothing (returns table unchanged) if zp is None.
    """
    if zp is None:
        print("No zero-point available — 'real_mag' not added.")
        return photometry_table

    photometry_table["real_mag"] = photometry_table["inst_mag"] + zp
    return photometry_table
