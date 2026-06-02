DOMAIN_EXTENTS: dict[str, list[float]] = {
    "atlantic": [290, 349.75, 16, 55.75],
    "conus": [228, 299.75, 25, 56.75],
    "ecmwf": [0, 359.75, -89.75, 90],
    "full": [130, 369.75, 0.25, 80],
    "global": [0, 359.75, -89.75, 90],
    "pacific": [145, 234.75, 16, 55.75],
}

FRONT_COLORS: dict[str, str] = {
    "CF": "blue",
    "WF": "red",
    "SF": "limegreen",
    "OF": "darkviolet",
    "DL": "chocolate",
}

CONTOUR_CMAPS: dict[str, str] = {
    "CF": "Blues",
    "WF": "Reds",
    "SF": "Greens",
    "OF": "Purples",
    "DL": "copper_r",
}

FRONT_NAMES: dict[str, str] = {
    "CF": "Cold front",
    "WF": "Warm front",
    "SF": "Stationary front",
    "OF": "Occluded front",
    "DL": "Dryline",
}

# Maps front type label to its class index in model output (class 0 is background).
# Matches the ordering of FRONT_CLASS_MAP values in data/targets.py.
FRONT_TYPE_CLASS_INDEX: dict[str, int] = {"CF": 1, "WF": 2, "SF": 3, "OF": 4, "DL": 5}
