"""
Various data tools.

References
----------
* Snyder 1987: https://doi.org/10.3133/pp1395

Author: Andrew Justin (andrewjustinwx@gmail.com)
Script version: 2025.5.3

TODO
    * Finish adding masks for xarray datasets
"""

import pandas as pd
from shapely.geometry import LineString
import numpy as np
import xarray as xr
import tensorflow as tf
import regionmask
from fronts.utils import constants
from collections import namedtuple


BoundingBox = namedtuple("BoundingBox", ["lat_min", "lat_max", "lon_min", "lon_max"])


def maybe_convert_lon(lon: float, da: xr.DataArray) -> float:
    """Convert a longitude from -180/180 to the convention used by ``ds``.

    The ARCO ERA5 zarr store uses 0-360 longitudes. ``domain_extent`` is
    specified in the conventional -180/180 range.  When the store's minimum
    longitude is ≥ 0, shift any negative values by +360 so that slice
    selection returns the correct grid points.
    """
    if float(da.longitude.min()) >= 0 and lon < 0:
        return lon + 360.0
    return lon


def convert_domain_extent_to_bounding_box(domain_extent: list[float]) -> BoundingBox:
    """Converts a domain extent from constants.py to a BoundingBox namedtuple.

    Args:
        domain_extent: A list of four floats representing the domain extent in the
            format [lon_min, lon_max, lat_min, lat_max].

    Returns a BoundingBox named tuple with the corresponding values.
    """
    if len(domain_extent) != 4:
        raise ValueError("Domain extent must be a list of four floats.")
    return BoundingBox(
        lon_min=domain_extent[0],
        lon_max=domain_extent[1],
        lat_min=domain_extent[2],
        lat_max=domain_extent[3],
    )


def expand_fronts(
    fronts: np.ndarray | tf.Tensor | xr.Dataset | xr.DataArray, iterations: int = 1
):
    """
    Expands front labels in all directions.

    Parameters
    ----------
    fronts: array_like of ints of shape (T, M, N) or (M, N)
        2-D or 3-D array of integers that identify the front type at each point. The longitude and latitude dimensions with
            shapes (M,) and (N,) can be in any order, but the time dimension must be the first dimension if it is passed.
    iterations: int
        Integer representing the number of times to expand the fronts in all directions.

    Returns
    -------
    fronts: array_like of ints of shape (T, M, N) or (1, M, N)
        Array of integers for the expanded fronts. If the array_like object passed into the function was 2-D, a third dimension
            will be added to the beginning of the array with size 1.

    Examples
    --------
    * Expanding labels for one front type.
    >>> arr = np.zeros((5, 5))
    >>> arr[2, 2] = 1  # add cold front point
    >>> arr
    array([[0., 0., 0., 0., 0.],
           [0., 0., 0., 0., 0.],
           [0., 0., 1., 0., 0.],
           [0., 0., 0., 0., 0.],
           [0., 0., 0., 0., 0.]])
    >>> expand_fronts(arr, iterations=1)
    array([[[0., 0., 0., 0., 0.],
            [0., 1., 1., 1., 0.],
            [0., 1., 1., 1., 0.],
            [0., 1., 1., 1., 0.],
            [0., 0., 0., 0., 0.]]])

    * Expanding labels for two front types.
    >>> arr = np.zeros((5, 5))
    >>> arr[1, 1] = 1  # add cold front point
    >>> arr[3, 3] = 2  # add warm front point
    >>> arr
    array([[0., 0., 0., 0., 0.],
           [0., 1., 0., 0., 0.],
           [0., 0., 0., 0., 0.],
           [0., 0., 0., 2., 0.],
           [0., 0., 0., 0., 0.]])
    >>> expand_fronts(arr, iterations=1)
    array([[[1., 1., 1., 0., 0.],
            [1., 1., 1., 0., 0.],
            [1., 1., 2., 2., 2.],
            [0., 0., 2., 2., 2.],
            [0., 0., 2., 2., 2.]]])
    """
    if isinstance(fronts, (xr.Dataset, xr.DataArray)):
        identifier = (
            fronts["identifier"].values if isinstance(fronts, xr.Dataset) else fronts.values
        )

    elif tf.is_tensor(fronts):
        identifier = (
            tf.expand_dims(fronts, axis=0) if len(fronts.shape) == 2 else fronts
        )
    else:
        identifier = (
            np.expand_dims(fronts, axis=0) if len(fronts.shape) == 2 else fronts
        )

    if tf.is_tensor(identifier):
        for _ in range(iterations):
            # 8 tensors representing all directions for the front expansion
            identifier_up_left = tf.Variable(tf.zeros_like(identifier))
            identifier_up_right = tf.Variable(tf.zeros_like(identifier))
            identifier_down_left = tf.Variable(tf.zeros_like(identifier))
            identifier_down_right = tf.Variable(tf.zeros_like(identifier))
            identifier_up = tf.Variable(tf.zeros_like(identifier))
            identifier_down = tf.Variable(tf.zeros_like(identifier))
            identifier_left = tf.Variable(tf.zeros_like(identifier))
            identifier_right = tf.Variable(tf.zeros_like(identifier))

            identifier_down_left[..., 1:, :-1].assign(
                tf.where(
                    (identifier[..., :-1, 1:] > 0) & (identifier[..., 1:, :-1] == 0),
                    identifier[..., :-1, 1:],
                    identifier[..., 1:, :-1],
                )
            )
            identifier_down[..., 1:, :].assign(
                tf.where(
                    (identifier[..., :-1, :] > 0) & (identifier[..., 1:, :] == 0),
                    identifier[..., :-1, :],
                    identifier[..., 1:, :],
                )
            )
            identifier_down_right[..., 1:, 1:].assign(
                tf.where(
                    (identifier[..., :-1, :-1] > 0) & (identifier[..., 1:, 1:] == 0),
                    identifier[..., :-1, :-1],
                    identifier[..., 1:, 1:],
                )
            )
            identifier_up_left[..., :-1, :-1].assign(
                tf.where(
                    (identifier[..., 1:, 1:] > 0) & (identifier[..., :-1, :-1] == 0),
                    identifier[..., 1:, 1:],
                    identifier[..., :-1, :-1],
                )
            )
            identifier_up[..., :-1, :].assign(
                tf.where(
                    (identifier[..., 1:, :] > 0) & (identifier[..., :-1, :] == 0),
                    identifier[..., 1:, :],
                    identifier[..., :-1, :],
                )
            )
            identifier_up_right[..., :-1, 1:].assign(
                tf.where(
                    (identifier[..., 1:, :-1] > 0) & (identifier[..., :-1, 1:] == 0),
                    identifier[..., 1:, :-1],
                    identifier[..., :-1, 1:],
                )
            )
            identifier_left[..., :, :-1].assign(
                tf.where(
                    (identifier[..., :, 1:] > 0) & (identifier[..., :, :-1] == 0),
                    identifier[..., :, 1:],
                    identifier[..., :, :-1],
                )
            )
            identifier_right[..., :, 1:].assign(
                tf.where(
                    (identifier[..., :, :-1] > 0) & (identifier[..., :, 1:] == 0),
                    identifier[..., :, :-1],
                    identifier[..., :, 1:],
                )
            )

            identifier = tf.reduce_max(
                [
                    identifier_up_left,
                    identifier_up,
                    identifier_up_right,
                    identifier_down_left,
                    identifier_down,
                    identifier_down_right,
                    identifier_left,
                    identifier_right,
                ],
                axis=0,
            )

    else:
        for _ in range(iterations):
            # 8 arrays representing all directions for the front expansion
            identifier_up_left = np.zeros_like(identifier)
            identifier_up_right = np.zeros_like(identifier)
            identifier_down_left = np.zeros_like(identifier)
            identifier_down_right = np.zeros_like(identifier)
            identifier_up = np.zeros_like(identifier)
            identifier_down = np.zeros_like(identifier)
            identifier_left = np.zeros_like(identifier)
            identifier_right = np.zeros_like(identifier)

            identifier_down_left[..., 1:, :-1] = np.where(
                (identifier[..., :-1, 1:] > 0) & (identifier[..., 1:, :-1] == 0),
                identifier[..., :-1, 1:],
                identifier[..., 1:, :-1],
            )
            identifier_down[..., 1:, :] = np.where(
                (identifier[..., :-1, :] > 0) & (identifier[..., 1:, :] == 0),
                identifier[..., :-1, :],
                identifier[..., 1:, :],
            )
            identifier_down_right[..., 1:, 1:] = np.where(
                (identifier[..., :-1, :-1] > 0) & (identifier[..., 1:, 1:] == 0),
                identifier[..., :-1, :-1],
                identifier[..., 1:, 1:],
            )
            identifier_up_left[..., :-1, :-1] = np.where(
                (identifier[..., 1:, 1:] > 0) & (identifier[..., :-1, :-1] == 0),
                identifier[..., 1:, 1:],
                identifier[..., :-1, :-1],
            )
            identifier_up[..., :-1, :] = np.where(
                (identifier[..., 1:, :] > 0) & (identifier[..., :-1, :] == 0),
                identifier[..., 1:, :],
                identifier[..., :-1, :],
            )
            identifier_up_right[..., :-1, 1:] = np.where(
                (identifier[..., 1:, :-1] > 0) & (identifier[..., :-1, 1:] == 0),
                identifier[..., 1:, :-1],
                identifier[..., :-1, 1:],
            )
            identifier_left[..., :, :-1] = np.where(
                (identifier[..., :, 1:] > 0) & (identifier[..., :, :-1] == 0),
                identifier[..., :, 1:],
                identifier[..., :, :-1],
            )
            identifier_right[..., :, 1:] = np.where(
                (identifier[..., :, :-1] > 0) & (identifier[..., :, 1:] == 0),
                identifier[..., :, :-1],
                identifier[..., :, 1:],
            )

            identifier = np.max(
                [
                    identifier_up_left,
                    identifier_up,
                    identifier_up_right,
                    identifier_down_left,
                    identifier_down,
                    identifier_down_right,
                    identifier_left,
                    identifier_right,
                ],
                axis=0,
            )

    if isinstance(fronts, xr.Dataset):
        fronts["identifier"].values = identifier
    elif isinstance(fronts, xr.DataArray):
        fronts.values = identifier
    else:
        fronts = identifier

    return fronts


def haversine(lon: np.ndarray | int | float, lat: np.ndarray | int | float):
    """
    Haversine formula. Transforms lon/lat points to an x/y cartesian plane.

    Parameters
    ----------
    lon: array_like of shape (N,), int, or float
        Longitude component of the point(s) expressed in degrees.
    lat: array_like of shape (N,), int, or float
        Latitude component of the point(s) expressed in degrees.

    Returns
    -------
    x: array_like of shape (N,) or float
        X component of the transformed points expressed in kilometers.
    y: array_like of shape (N,) or float
        Y component of the transformed points expressed in kilometers.

    Examples
    --------
    >>> lon = -95
    >>> lat = 35
    >>> x, y = haversine(lon, lat)
    >>> x, y
    (-10077.330945462296, 3892.875)

    >>> lon = np.arange(10, 80.1, 10)
    >>> lat = np.arange(10, 80.1, 10)
    >>> x, y = haversine(lon, lat)
    >>> x, y
    (array([1108.01755295, 2190.70484658, 3223.05300087, 4180.69246988,
           5040.20418066, 5779.42053216, 6377.71302882, 6816.26345487]), array([1112.25, 2224.5 , 3336.75, 4449.  , 5561.25, 6673.5 , 7785.75,
           8898.  ]))
    """
    C = 40041  # average circumference of earth in kilometers
    x = lon * C * np.cos(lat * np.pi / 360) / 360
    y = lat * C / 360
    return x, y


def reverse_haversine(x, y):
    """
    Reverse haversine formula. Transforms x/y cartesian coordinates to a lon/lat grid.

    Parameters
    ----------
    x: array_like of shape (N,), int, or float
        X component of the point(s) expressed in kilometers.
    y: array_like of shape (N,), int, or float
        Y component of the point(s) expressed in kilometers.

    Returns
    -------
    lon: array_like of shape (N,) or float
        Longitude component of the transformed point(s) expressed in degrees.
    lat: array_like of shape (N,) or float
        Latitude component of the transformed point(s) expressed in degrees.

    Examples
    --------
    Values pulled from haversine examples.

    >>> x = -10077.330945462296
    >>> y = 3892.875
    >>> lon, lat = reverse_haversine(x, y)
    >>> lon, lat
    (-95.0, 35.0)

    >>> x = np.array(
    ...     [
    ...         1108.01755295,
    ...         2190.70484658,
    ...         3223.05300087,
    ...         4180.69246988,
    ...         5040.20418066,
    ...         5779.42053216,
    ...         6377.71302882,
    ...         6816.26345487,
    ...     ]
    ... )
    >>> y = np.array(
    ...     [1112.25, 2224.5, 3336.75, 4449.0, 5561.25, 6673.5, 7785.75, 8898.0]
    ... )
    >>> lon, lat = reverse_haversine(x, y)
    >>> lon, lat
    (array([10., 20., 30., 40., 50., 60., 70., 80.]), array([10., 20., 30., 40., 50., 60., 70., 80.]))
    """
    C = 40041  # average circumference of earth in kilometers
    lon = x * 360 / np.cos(y * np.pi / C) / C
    lat = y * 360 / C
    return lon, lat


def geometric(x_km_new, y_km_new):
    """
    Turn longitudinal/latitudinal distance (km) lists into LineString for interpolation.

    Parameters
    ----------
    x_km_new: List containing longitude coordinates of fronts in kilometers.
    y_km_new: List containing latitude coordinates of fronts in kilometers.

    Returns
    -------
    xy_linestring: LineString object containing coordinates of fronts in kilometers.
    """
    df_xy = pd.DataFrame(
        list(zip(x_km_new, y_km_new)), columns=["Longitude_km", "Latitude_km"]
    )
    geometry = [xy for xy in zip(df_xy.Longitude_km, df_xy.Latitude_km)]
    xy_linestring = LineString(geometry)
    return xy_linestring


def redistribute_vertices(xy_linestring, distance):
    """
    Interpolate x/y coordinates at a specified distance.

    Parameters
    ----------
    xy_linestring: LineString object containing coordinates of fronts in kilometers.
    distance: Distance at which to interpolate the x/y coordinates.

    Returns
    -------
    xy_vertices: Normalized MultiLineString that contains the interpolated coordinates of fronts in kilometers.

    Sources
    -------
    https://stackoverflow.com/questions/34906124/interpolating-every-x-distance-along-multiline-in-shapely/35025274#35025274
    """
    if xy_linestring.geom_type == "LineString":
        num_vert = int(round(xy_linestring.length / distance))
        if num_vert == 0:
            num_vert = 1
        return LineString(
            [
                xy_linestring.interpolate(float(n) / num_vert, normalized=True)
                for n in range(num_vert + 1)
            ]
        )
    elif xy_linestring.geom_type == "MultiLineString":
        parts = [redistribute_vertices(part, distance) for part in xy_linestring]
        return type(xy_linestring)([p for p in parts if not p.is_empty])
    else:
        raise ValueError("unhandled geometry %s", (xy_linestring.geom_type,))


def reformat_fronts(fronts, front_types):
    """
    Reformat a front dataset, tensor, or array with a given set of front types.

    Parameters
    ----------
    front_types: str or list of strs
        Code(s) that determine how the dataset will be reformatted.
    fronts: xarray Dataset or DataArray, tensor, or np.ndarray
        Dataset containing the front data.
        '''
        Available options for individual front types (cannot be passed with any special codes):

        Code (class #): Front Type
        --------------------------
        CF (1): Cold front
        WF (2): Warm front
        SF (3): Stationary front
        OF (4): Occluded front
        CF-F (5): Cold front (forming)
        WF-F (6): Warm front (forming)
        SF-F (7): Stationary front (forming)
        OF-F (8): Occluded front (forming)
        CF-D (9): Cold front (dissipating)
        WF-D (10): Warm front (dissipating)
        SF-D (11): Stationary front (dissipating)
        OF-D (12): Occluded front (dissipating)
        INST (13): Squall line ????
        TROF (14): Trough
        TT (15): Tropical Trough
        DL (16): Dryline


        Special codes (cannot be passed with any individual front codes):
        -----------------------------------------------------------------
        F_BIN (1 class): 1-4, but treat all front types as one type.
            (1): CF, WF, SF, OF

        MERGED-F (4 classes): 1-12, but treat forming and dissipating fronts as standard fronts.
            (1): CF, CF-F, CF-D
            (2): WF, WF-F, WF-D
            (3): SF, SF-F, SF-D
            (4): OF, OF-F, OF-D

        MERGED-F_BIN (1 class): 1-12, but treat all front types and stages as one type. This means that classes 1-12 will all be one class (1).
            (1): CF, CF-F, CF-D, WF, WF-F, WF-D, SF, SF-F, SF-D, OF, OF-F, OF-D

        MERGED-T (1 class): 14-15, but treat troughs and tropical troughs as the same. In other words, TT (15) becomes TROF (14).
            (1): TROF, TT

        MERGED-ALL (7 classes): 1-16, but make the changes in the MERGED-F and MERGED-T codes.
            (1): CF, CF-F, CF-D
            (2): WF, WF-F, WF-D
            (3): SF, SF-F, SF-D
            (4): OF, OF-F, OF-D
            (5): TROF, TT
            (6): INST
            (7): DL

        **** NOTE - Class 0 is always treated as 'no front'.
        '''

    Returns
    -------
    fronts_ds: xr.Dataset
        Reformatted dataset based on the provided code(s).
    """

    if isinstance(front_types, str):
        front_types = [
            front_types,
        ]

    if isinstance(fronts, (xr.DataArray, xr.Dataset)):
        where_function = xr.where
    elif isinstance(fronts, np.ndarray):
        where_function = np.where
    else:
        where_function = tf.where

    front_types_classes = {
        "CF": 1,
        "WF": 2,
        "SF": 3,
        "OF": 4,
        "CF-F": 5,
        "WF-F": 6,
        "SF-F": 7,
        "OF-F": 8,
        "CF-D": 9,
        "WF-D": 10,
        "SF-D": 11,
        "OF-D": 12,
        "INST": 13,
        "TROF": 14,
        "TT": 15,
        "DL": 16,
    }

    if front_types == [
        "F_BIN",
    ]:
        fronts = where_function(fronts > 4, 0, fronts)  # Classes 5-16 are removed
        fronts = where_function(fronts > 0, 1, fronts)  # Merge 1-4 into one class

        labels = [
            "CF-WF-SF-OF",
        ]
        num_types = 1

    elif front_types == ["MERGED-F"]:
        fronts = where_function(
            fronts == 5, 1, fronts
        )  # Forming cold front ---> cold front
        fronts = where_function(
            fronts == 6, 2, fronts
        )  # Forming warm front ---> warm front
        fronts = where_function(
            fronts == 7, 3, fronts
        )  # Forming stationary front ---> stationary front
        fronts = where_function(
            fronts == 8, 4, fronts
        )  # Forming occluded front ---> occluded front
        fronts = where_function(
            fronts == 9, 1, fronts
        )  # Dying cold front ---> cold front
        fronts = where_function(
            fronts == 10, 2, fronts
        )  # Dying warm front ---> warm front
        fronts = where_function(
            fronts == 11, 3, fronts
        )  # Dying stationary front ---> stationary front
        fronts = where_function(
            fronts == 12, 4, fronts
        )  # Dying occluded front ---> occluded front
        fronts = where_function(fronts > 4, 0, fronts)  # Remove all other fronts

        labels = ["CF_any", "WF_any", "SF_any", "OF_any"]
        num_types = 4

    elif front_types == ["MERGED-F_BIN"]:
        fronts = where_function(fronts > 12, 0, fronts)  # Classes 13-16 are removed
        fronts = where_function(
            fronts > 0, 1, fronts
        )  # Classes 1-12 are merged into one class

        labels = [
            "CF-WF-SF-OF_any",
        ]
        num_types = 1

    elif front_types == ["MERGED-T"]:
        fronts = where_function(fronts < 14, 0, fronts)  # Remove classes 1-13

        # Merge troughs into one class
        fronts = where_function(fronts == 14, 1, fronts)
        fronts = where_function(fronts == 15, 1, fronts)

        fronts = where_function(fronts == 16, 0, fronts)  # Remove drylines

        labels = [
            "TR_any",
        ]
        num_types = 1

    elif front_types == ["MERGED-ALL"]:
        fronts = where_function(
            fronts == 5, 1, fronts
        )  # Forming cold front ---> cold front
        fronts = where_function(
            fronts == 6, 2, fronts
        )  # Forming warm front ---> warm front
        fronts = where_function(
            fronts == 7, 3, fronts
        )  # Forming stationary front ---> stationary front
        fronts = where_function(
            fronts == 8, 4, fronts
        )  # Forming occluded front ---> occluded front
        fronts = where_function(
            fronts == 9, 1, fronts
        )  # Dying cold front ---> cold front
        fronts = where_function(
            fronts == 10, 2, fronts
        )  # Dying warm front ---> warm front
        fronts = where_function(
            fronts == 11, 3, fronts
        )  # Dying stationary front ---> stationary front
        fronts = where_function(
            fronts == 12, 4, fronts
        )  # Dying occluded front ---> occluded front

        # Merge troughs together into class 5
        fronts = where_function(fronts == 14, 5, fronts)
        fronts = where_function(fronts == 15, 5, fronts)

        fronts = where_function(
            fronts == 13, 6, fronts
        )  # Move outflow boundaries to class 6
        fronts = where_function(fronts == 16, 7, fronts)  # Move drylines to class 7

        labels = ["CF_any", "WF_any", "SF_any", "OF_any", "TR_any", "INST", "DL"]
        num_types = 7

    else:
        # Select the front types that are being used to pull their class identifiers
        filtered_front_types = dict(
            sorted(
                dict(
                    [
                        (i, front_types_classes[i])
                        for i in front_types_classes
                        if i in set(front_types)
                    ]
                ).items(),
                key=lambda item: item[1],
            )
        )
        front_types, num_types = (
            list(filtered_front_types.keys()),
            len(filtered_front_types.keys()),
        )

        for i in range(num_types):
            if i + 1 != front_types_classes[front_types[i]]:
                fronts = where_function(fronts == i + 1, 0, fronts)
                fronts = where_function(
                    fronts == front_types_classes[front_types[i]], i + 1, fronts
                )  # Reformat front classes

        fronts = where_function(
            fronts > num_types, 0, fronts
        )  # Remove unused front types

        labels = front_types

    if isinstance(fronts, (xr.Dataset, xr.DataArray)):
        fronts.attrs["front_types"] = front_types
        fronts.attrs["num_front_types"] = num_types
        fronts.attrs["labels"] = labels

    return fronts


def normalize_dataset(
    ds, method="standard", normalization_parameters=constants.NORMALIZATION_PARAMS
) -> xr.Dataset:
    """
    Normalizes variables in an xarray dataset. This function can also accept xarray datasets for GOES satellite data.

    Parameters
    ----------
    ds: xarray dataset
        Dataset containing variables to normalize.
    method: 'standard', 'standard_weighted', 'min-max'
        Normalization method to perform on the variables.
        - 'standard': Standard z-score normalization.
        - 'standard_weighted': Standard z-score normalization with latitude-weighted means and standard deviations.
        - 'min-max': Min-max normalization.
    normalization_parameters: dict
        Dictionary containing parameters for normalization.

    Returns
    -------
    normalized_ds: xarray dataset
        Normalized xarray dataset.
    """

    ds_copy = ds.copy()

    variables = list(ds_copy.keys())

    is_satellite_dataset = "band_" in variables[0]  # check for satellite variables

    try:
        if is_satellite_dataset:
            norm_params = xr.Dataset(
                data_vars={
                    var: ("param", normalization_parameters["%s" % var])
                    for var in variables
                },
                coords={
                    "param": [
                        "min",
                        "max",
                        "mean",
                        "std",
                        "mean_weighted",
                        "std_weighted",
                    ]
                },
            )
        else:
            pressure_levels = ds_copy["pressure_level"].values.astype(
                int
            )  # TODO: will not work with surface data

            norm_params = xr.Dataset(
                data_vars={
                    var: (
                        ("pressure_level", "param"),
                        [
                            normalization_parameters["%s_%s" % (var, lvl)]
                            for lvl in pressure_levels
                        ],
                    )
                    for var in variables
                },
                coords={
                    "param": [
                        "min",
                        "max",
                        "mean",
                        "std",
                        "mean_weighted",
                        "std_weighted",
                    ],
                    "pressure_level": pressure_levels,
                },
            )
    except (
        ValueError
    ):  # models before the 2025.1.10 update only have min and max values
        pressure_levels = ds_copy["pressure_level"].values.astype(
            int
        )  # TODO: will not work with surface data
        norm_params = xr.Dataset(
            data_vars={
                var: (
                    ("pressure_level", "param"),
                    [
                        normalization_parameters["%s_%s" % (var, lvl)]
                        for lvl in pressure_levels
                    ],
                )
                for var in variables
            },
            coords={"param": ["max", "min"], "pressure_level": pressure_levels},
        )

    if method == "min-max":
        p_min = norm_params.sel(param="min")
        p_max = norm_params.sel(param="max")
        normalized_ds = (ds_copy - p_min) / (p_max - p_min)
    elif method == "standard":
        p_mean = norm_params.sel(param="mean")
        p_std = norm_params.sel(param="std")
        normalized_ds = (ds_copy - p_mean) / p_std
    elif method == "standard_weighted":
        p_mean = norm_params.sel(param="mean_weighted")
        p_std = norm_params.sel(param="std_weighted")
        normalized_ds = (ds_copy - p_mean) / p_std
    else:
        raise ValueError(
            "Unrecognized normalization method: %s. Valid normalization methods are 'min-max', 'standard', 'standard_weighted'."
            % method
        )

    return normalized_ds


def combine_datasets(tf_files: list[str]):
    """
    Combine many tensorflow datasets into one entire dataset.

    Returns
    -------
    dataset: tf.data.Dataset object
        Concatenated tensorflow dataset.
    """
    dataset = tf.data.Dataset.load(tf_files[0])
    for file in tf_files[1:]:
        dataset = dataset.concatenate(tf.data.Dataset.load(file))

    return dataset


def lambert_conformal_to_cartesian(
    lon: np.ndarray | tuple | list | int | float,
    lat: np.ndarray | tuple | list | int | float,
    std_parallels: tuple | list = (20.0, 50.0),
    lon_ref: int | float = 0.0,
    lat_ref: int | float = 0.0,
):
    """
    Transform points on a Lambert Conformal lat/lon grid to cartesian coordinates.

    Parameters
    ----------
    lon: array_like of shape (N,), int, or float
        Longitude point(s) expressed as degrees.
    lat: array_like of shape (N,), int, or float
        Latitude point(s) expressed as degrees.
    std_parallels: tuple or list of 2 ints or floats
        Standard parallels to use in the coordinate transformation, expressed as degrees.
    lon_ref: int or float
        Reference longitude point expressed as degrees.
    lat_ref: int or float
        Reference latitude point expressed as degrees.

    Returns
    -------
    x: array_like of shape (N,) or float
        X-component of the transformed coordinates, expressed as meters.
    y: array_like of shape (N,) or float
        Y-component of the transformed coordinates, expressed as meters.

    Examples
    --------
    * Using parameters from example on Page 295 of Snyder 1987 (except the output here is expressed as meters):
    >>> x, y = lambert_conformal_to_cartesian(
    ...     lon=-75, lat=35, std_parallels=(33, 45), lon_ref=-96, lat_ref=23
    ... )
    >>> x, y
    (1890206.4076610378, 1568668.1244433122)

    * Same as above but with longitudes expressed from 0 to 360 degrees east:
    >>> x, y = lambert_conformal_to_cartesian(
    ...     lon=285, lat=35, std_parallels=(33, 45), lon_ref=264, lat_ref=23
    ... )
    >>> x, y
    (1890206.4076610343, 1568668.1244433112)

    References
    ----------
    * Snyder 1987: https://doi.org/10.3133/pp1395

    Notes
    -----
    lon and lon_ref must be both expressed in the same longitude range (e.g. -180 to 180 degrees or 0 to 360 degrees)
        to get correct values for x and y.
    """

    R = 6371229  # radius of earth (meters)

    # Points and standard parallels need to be expressed as radians for the transformation formulas
    lon = np.radians(lon)
    lon_ref = np.radians(lon_ref)
    lat = np.radians(lat)
    lat_ref = np.radians(lat_ref)
    std_parallels_rad = np.radians(std_parallels)

    if std_parallels_rad[0] == std_parallels_rad[1]:
        n = np.sin(std_parallels_rad[0])
    else:
        n = np.divide(
            np.log(np.cos(std_parallels_rad[0]) / np.cos(std_parallels_rad[1])),
            np.log(
                np.tan(np.pi / 4 + std_parallels_rad[1] / 2)
                / np.tan(np.pi / 4 + std_parallels_rad[0] / 2)
            ),
        )
    F = (
        np.cos(std_parallels_rad[0])
        * np.power(np.tan(np.pi / 4 + std_parallels_rad[0] / 2), n)
        / n
    )
    rho = R * F / np.power(np.tan(np.pi / 4 + lat / 2), n)
    rho0 = R * F / np.power(np.tan(np.pi / 4 + lat_ref / 2), n)

    x = rho * np.sin(n * (lon - lon_ref))
    y = rho0 - rho * np.cos(n * (lon - lon_ref))

    return x, y


def mask_xarray_dataset(ds, mask, lon="longitude", lat="latitude"):
    """
    Apply a geospatial mask from the regionmask package to an Xarray dataset.

    Parameters
    ----------
    ds: xarray Dataset
        Xarray dataset that must have longitude and latitude dimensions.
    mask: str
        Geospatial mask to apply to the dataset.
    lon: str
        Longitude dimension key in the xarray dataset.
    lat: str
        Latitude dimension key in the xarray dataset.

    Returns
    -------
    masked_ds: xarray Dataset
        Masked xarray dataset.
    """
    # {region_key: region_index}
    regions_crossing_prime_meridian = ["north_atlantic_ocean"]
    ocean_basins = {
        "arctic_ocean": [0, 13, 31, 32, 40, 47, 56, 57],
        "north_atlantic_ocean": [2, 37, 60, 83, 88, 99, 100],
        "south_atlantic_ocean": [
            6,
        ],
        "indian_ocean": [5, 10, 12, 36, 43, 44, 50, 52, 61, 90, 105],
        "north_pacific_ocean": [3, 8, 20, 59],
        "south_pacific_ocean": [4, 9, 27, 74, 80, 85, 86],
        "southern_ocean": [1, 23, 26, 38, 53, 54, 58],
    }

    region_is_ocean_basin = mask in ocean_basins
    region_crosses_prime_meridian = mask in regions_crossing_prime_meridian

    if region_is_ocean_basin:
        regions = regionmask.defined_regions.natural_earth_v5_1_2.ocean_basins_50
        indices = ocean_basins[mask]

    base_mask = regions.mask(ds[lon], ds[lat])
    region_mask = base_mask.isin(indices)
    masked_ds = ds.where(region_mask, other=0)

    if region_crosses_prime_meridian:
        lons = masked_ds[lon]
        lon_east_hemi, lon_west_hemi = lons[lons <= 180], lons[lons > 180]
        masked_ds = masked_ds.reindex(
            {lon: np.concatenate([lon_west_hemi, lon_east_hemi])}
        )
        new_lons = np.concatenate([lon_west_hemi, lon_east_hemi + 360])
        masked_ds[lon] = new_lons

    return masked_ds
