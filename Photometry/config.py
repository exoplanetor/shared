# ---------------------------------------------
# API keys and settings
# Edit these values directly when they change.
# ---------------------------------------------

ASTROMETRY_API_KEY = "ebooqrmawchqdjfa"

# ---------------------------------------------
# Astrometry.net scale hint (field-of-view width, arcmin)
# Update these whenever the optical configuration changes
# (e.g. focal reducer in/out changes the effective FOV).
#
# Current setup: Kepler KL4040, no reducer -> ~3.3 arcmin FOV
# With focal reducer (when in use) -> ~8 arcmin FOV (update below when active)
# ---------------------------------------------

SCALE_LOWER_ARCMIN = 3.0
SCALE_UPPER_ARCMIN = 3.6