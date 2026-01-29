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

# [min, max, mean, std, mean (lat weighted), std (lat weighted)]
NORMALIZATION_PARAMS = dict()
NORMALIZATION_PARAMS["q_300"] = [0.0, 2.25, 0.1484, 0.1701, 0.1854, 0.1957]
NORMALIZATION_PARAMS["q_500"] = [0.0, 9.375, 0.8906, 1.1321, 1.1552, 1.2790]
NORMALIZATION_PARAMS["q_700"] = [0.0, 16.0, 2.4991, 2.6180, 3.2535, 2.8386]
NORMALIZATION_PARAMS["q_850"] = [0.0, 21.75, 4.6545, 4.1544, 6.1259, 4.2482]
NORMALIZATION_PARAMS["q_900"] = [0.0, 23.25, 5.6176, 4.7384, 7.4258, 4.7040]
NORMALIZATION_PARAMS["q_950"] = [0.0, 25.125, 6.5983, 5.5294, 8.7716, 5.4365]
NORMALIZATION_PARAMS["q_1000"] = [0.0, 29.25, 7.0913, 5.9191, 9.4306, 5.7912]
NORMALIZATION_PARAMS["RH_300"] = [0.0, 1.0, 0.3426, 0.2172, 0.3225, 0.2226]
NORMALIZATION_PARAMS["RH_500"] = [0.0, 1.0, 0.4073, 0.264, 0.3737, 0.2755]
NORMALIZATION_PARAMS["RH_700"] = [0.0, 1.0, 0.4791, 0.273, 0.4614, 0.2811]
NORMALIZATION_PARAMS["RH_850"] = [0.0, 1.0, 0.6047, 0.2687, 0.6242, 0.2623]
NORMALIZATION_PARAMS["RH_900"] = [0.0, 1.0, 0.6682, 0.2657, 0.7033, 0.2444]
NORMALIZATION_PARAMS["RH_950"] = [0.0, 1.0, 0.7037, 0.2673, 0.7464, 0.2371]
NORMALIZATION_PARAMS["RH_1000"] = [0.0, 1.0, 0.659, 0.2457, 0.6897, 0.2104]
NORMALIZATION_PARAMS["sp_z_300"] = [796.0, 989.0, 913.0033, 52.2398, 935.2045, 43.2219]
NORMALIZATION_PARAMS["sp_z_500"] = [442.0, 605.0, 552.4796, 34.4652, 566.8235, 28.0863]
NORMALIZATION_PARAMS["sp_z_700"] = [206.0, 334.0, 295.3216, 22.0330, 303.9771, 17.9648]
NORMALIZATION_PARAMS["sp_z_850"] = [60.0, 177.0, 140.3339, 15.2319, 145.5283, 12.4728]
NORMALIZATION_PARAMS["sp_z_900"] = [16.0, 130.0, 93.9296, 13.5094, 98.0819, 11.1214]
NORMALIZATION_PARAMS["sp_z_950"] = [-27.0, 88.0, 49.7221, 12.1269, 52.8635, 10.0784]
NORMALIZATION_PARAMS["sp_z_1000"] = [-69.0, 49.0, 7.4702, 11.1432, 9.6184, 9.3651]
NORMALIZATION_PARAMS["T_300"] = [199.0, 257.0, 229.2386, 10.8319, 233.7161, 9.4612]
NORMALIZATION_PARAMS["T_500"] = [215.0, 284.0, 253.2812, 13.0718, 258.8222, 10.9391]
NORMALIZATION_PARAMS["T_700"] = [208.0, 302.0, 267.7375, 14.8156, 274.0165, 11.5235]
NORMALIZATION_PARAMS["T_850"] = [217.0, 315.0, 274.9058, 15.5217, 281.4309, 12.2972]
NORMALIZATION_PARAMS["T_900"] = [219.0, 319.0, 276.7593, 15.7479, 283.3672, 12.5458]
NORMALIZATION_PARAMS["T_950"] = [219.0, 322.0, 278.7952, 16.2086, 285.6539, 12.8296]
NORMALIZATION_PARAMS["T_1000"] = [216.0, 326.0, 281.4339, 16.9643, 288.7020, 13.3128]
NORMALIZATION_PARAMS["Td_300"] = [159.0, 254.0, 217.3682, 10.5844, 220.6169, 10.3878]
NORMALIZATION_PARAMS["Td_500"] = [165.0, 279.0, 239.8065, 13.0666, 243.1986, 13.2075]
NORMALIZATION_PARAMS["Td_700"] = [176.0, 291.0, 255.3803, 15.7529, 260.1953, 14.6048]
NORMALIZATION_PARAMS["Td_850"] = [186.0, 298.0, 266.3356, 17.3281, 272.8387, 13.9459]
NORMALIZATION_PARAMS["Td_900"] = [190.0, 300.0, 269.7803, 17.9404, 276.8977, 13.6627]
NORMALIZATION_PARAMS["Td_950"] = [194.0, 302.0, 272.4960, 18.8094, 280.1294, 14.0384]
NORMALIZATION_PARAMS["Td_1000"] = [195.0, 306.0, 274.0599, 19.37, 281.9399, 14.3432]
NORMALIZATION_PARAMS["Tv_300"] = [199.0, 257.0, 229.2578, 10.8478, 233.7418, 9.4773]
NORMALIZATION_PARAMS["Tv_500"] = [215.0, 285.0, 253.4233, 13.1756, 259.0079, 11.0404]
NORMALIZATION_PARAMS["Tv_700"] = [208.0, 303.0, 268.1590, 15.1075, 274.5694, 11.8080]
NORMALIZATION_PARAMS["Tv_850"] = [217.0, 316.0, 275.7130, 16.0917, 282.5005, 12.8374]
NORMALIZATION_PARAMS["Tv_900"] = [219.0, 320.0, 277.7404, 16.4410, 284.6725, 13.1932]
NORMALIZATION_PARAMS["Tv_950"] = [218.0, 323.0, 279.9575, 17.0354, 287.2087, 13.5979]
NORMALIZATION_PARAMS["Tv_1000"] = [216.0, 327.0, 282.6972, 17.8702, 290.3930, 14.1572]
NORMALIZATION_PARAMS["theta_300"] = [281.0, 362.0, 323.2393, 15.2710, 329.5529, 13.3377]
NORMALIZATION_PARAMS["theta_500"] = [262.0, 346.0, 308.6888, 15.9301, 315.4419, 13.3306]
NORMALIZATION_PARAMS["theta_700"] = [231.0, 334.0, 296.4287, 16.4027, 303.3806, 12.7576]
NORMALIZATION_PARAMS["theta_850"] = [227.0, 330.0, 287.9577, 16.2584, 294.7926, 12.8807]
NORMALIZATION_PARAMS["theta_900"] = [226.0, 329.0, 285.2081, 16.2285, 292.0178, 12.9286]
NORMALIZATION_PARAMS["theta_950"] = [222.0, 327.0, 282.9067, 16.4475, 289.8666, 13.0187]
NORMALIZATION_PARAMS["theta_1000"] = [
    216.0,
    326.0,
    281.4339,
    16.9643,
    288.7020,
    13.3128,
]
NORMALIZATION_PARAMS["theta_e_300"] = [
    281.0,
    368.0,
    323.7062,
    15.6360,
    330.1604,
    13.7126,
]
NORMALIZATION_PARAMS["theta_e_500"] = [
    262.0,
    360.0,
    311.2741,
    17.9377,
    318.7880,
    15.4055,
]
NORMALIZATION_PARAMS["theta_e_700"] = [
    229.0,
    367.0,
    303.0449,
    21.4989,
    312.0104,
    18.0769,
]
NORMALIZATION_PARAMS["theta_e_850"] = [
    226.0,
    372.0,
    299.6698,
    25.2475,
    310.2498,
    21.8210,
]
NORMALIZATION_PARAMS["theta_e_900"] = [
    224.0,
    373.0,
    299.1296,
    26.7758,
    310.4758,
    23.1717,
]
NORMALIZATION_PARAMS["theta_e_950"] = [
    220.0,
    375.0,
    299.0758,
    28.8183,
    311.4362,
    24.9726,
]
NORMALIZATION_PARAMS["theta_e_1000"] = [
    215.0,
    390.0,
    298.7088,
    30.2554,
    311.7681,
    26.1548,
]
NORMALIZATION_PARAMS["theta_v_300"] = [
    281.0,
    362.0,
    323.2664,
    15.2933,
    329.5891,
    13.3604,
]
NORMALIZATION_PARAMS["theta_v_500"] = [
    262.0,
    347.0,
    308.8620,
    16.0566,
    315.6682,
    13.4540,
]
NORMALIZATION_PARAMS["theta_v_700"] = [
    231.0,
    335.0,
    296.8954,
    16.7259,
    303.9928,
    13.0726,
]
NORMALIZATION_PARAMS["theta_v_850"] = [
    227.0,
    330.0,
    288.8032,
    16.8555,
    295.9130,
    13.4466,
]
NORMALIZATION_PARAMS["theta_v_900"] = [
    225.0,
    329.0,
    286.2193,
    16.9428,
    293.3630,
    13.5958,
]
NORMALIZATION_PARAMS["theta_v_950"] = [
    220.0,
    328.0,
    284.0862,
    17.2865,
    291.4443,
    13.7983,
]
NORMALIZATION_PARAMS["theta_v_1000"] = [
    216.0,
    327.0,
    282.6972,
    17.8702,
    290.3930,
    14.1572,
]
NORMALIZATION_PARAMS["u_300"] = [-65.2, 115.6, 11.6922, 17.1698, 12.5158, 17.4089]
NORMALIZATION_PARAMS["u_500"] = [-51.6, 82.4, 6.5193, 12.0255, 6.6043, 12.0727]
NORMALIZATION_PARAMS["u_700"] = [-50.4, 58.4, 3.3234, 9.24, 3.1705, 9.2717]
NORMALIZATION_PARAMS["u_850"] = [-55.2, 53.2, 1.4020, 8.2607, 1.0348, 8.2607]
NORMALIZATION_PARAMS["u_900"] = [-55.6, 50.8, 0.8581, 8.1317, 0.4330, 8.1176]
NORMALIZATION_PARAMS["u_950"] = [-47.6, 43.2, 7.6961, -0.0558, 7.6638]
NORMALIZATION_PARAMS["u_1000"] = [-33.6, 30.4, -0.0452, 6.1942, -0.4278, 6.1954]
NORMALIZATION_PARAMS["v_300"] = [-80.0, 94.0, -0.0227, 13.3571, -0.0253, 12.6938]
NORMALIZATION_PARAMS["v_500"] = [-65.6, 70.0, -0.0251, 9.2148, -0.0366, 8.5501]
NORMALIZATION_PARAMS["v_700"] = [-48.4, 53.2, 0.0255, 6.9146, -0.0117, 6.4236]
NORMALIZATION_PARAMS["v_850"] = [-49.6, 52.8, 0.1468, 6.2973, 0.0924, 5.8921]
NORMALIZATION_PARAMS["v_900"] = [-48.4, 51.6, 0.2032, 6.4313, 0.1744, 6.0884]
NORMALIZATION_PARAMS["v_950"] = [-44.8, 46.8, 0.2083, 6.4496, 0.1977, 6.187]
NORMALIZATION_PARAMS["v_1000"] = [-31.6, 30.8, 0.1949, 5.3437, 0.1984, 5.1694]


# default values for extents of domains [start lon, end lon, start lat, end lat]
DOMAIN_EXTENTS = {
    "atlantic": [290, 349.75, 16, 55.75],
    "conus": [228, 299.75, 25, 56.75],
    "ecmwf": [0, 359.75, -89.75, 90],
    "full": [130, 369.75, 0.25, 80],
    "global": [0, 359.75, -89.75, 90],
    "goes-merged": [144, 359.75, 2, 69.75],
    "hrrr": [
        225.90452026573686,
        299.0828072281622,
        21.138123000000018,
        52.61565330680793,
    ],
    "MERGIR": [130, 359.75, 20, 59.75],
    "namnest-conus": [
        225.90387325951775,
        299.08216099364034,
        21.138,
        52.61565399063001,
    ],
    "nam-12km": [
        207.12137749594984,
        310.58401341435564,
        12.190000000000005,
        61.30935757335816,
    ],
    "pacific": [145, 234.75, 16, 55.75],
}

# colors for plotted ground truth fronts
FRONT_COLORS = {
    "CF": "blue",
    "WF": "red",
    "SF": "limegreen",
    "OF": "darkviolet",
    "CF-F": "darkblue",
    "WF-F": "darkred",
    "SF-F": "darkgreen",
    "OF-F": "darkmagenta",
    "CF-D": "lightskyblue",
    "WF-D": "lightcoral",
    "SF-D": "lightgreen",
    "OF-D": "violet",
    "INST": "gold",
    "TROF": "goldenrod",
    "TT": "orange",
    "DL": "chocolate",
    "MERGED-CF": "blue",
    "MERGED-WF": "red",
    "MERGED-SF": "limegreen",
    "MERGED-OF": "darkviolet",
    "MERGED-F": "gray",
    "MERGED-T": "brown",
    "F_BIN": "tab:red",
    "MERGED-F_BIN": "tab:red",
}

# colormaps of probability contours for front predictions
CONTOUR_CMAPS = {
    "CF": "Blues",
    "WF": "Reds",
    "SF": "Greens",
    "OF": "Purples",
    "CF-F": "Blues",
    "WF-F": "Reds",
    "SF-F": "Greens",
    "OF-F": "Purples",
    "CF-D": "Blues",
    "WF-D": "Reds",
    "SF-D": "Greens",
    "OF-D": "Purples",
    "INST": "YlOrBr",
    "TROF": "YlOrRd",
    "TT": "Oranges",
    "DL": "copper_r",
    "MERGED-CF": "Blues",
    "MERGED-WF": "Reds",
    "MERGED-SF": "Greens",
    "MERGED-OF": "Purples",
    "MERGED-F": "Greys",
    "MERGED-T": "YlOrBr",
    "F_BIN": "Greys",
    "MERGED-F_BIN": "Greys",
}

# names of front types
FRONT_NAMES = {
    "CF": "Cold front",
    "WF": "Warm front",
    "SF": "Stationary front",
    "OF": "Occluded front",
    "CF-F": "Cold front (forming)",
    "WF-F": "Warm front (forming)",
    "SF-F": "Stationary front (forming)",
    "OF-F": "Occluded front (forming)",
    "CF-D": "Cold front (dying)",
    "WF-D": "Warm front (dying)",
    "SF-D": "Stationary front (dying)",
    "OF-D": "Occluded front (dying)",
    "INST": "Outflow boundary",
    "TROF": "Trough",
    "TT": "Tropical trough",
    "DL": "Dryline",
    "MERGED-CF": "Cold front (any)",
    "MERGED-WF": "Warm front (any)",
    "MERGED-SF": "Stationary front (any)",
    "MERGED-OF": "Occluded front (any)",
    "MERGED-F": "CF, WF, SF, OF (any)",
    "MERGED-T": "Trough (any)",
    "F_BIN": "Binary front",
    "MERGED-F_BIN": "Binary front (any)",
}

VARIABLE_NAMES = {
    "T": "Air temperature",
    "T_sfc": "2-meter Air temperature",
    "T_1000": "1000mb Air temperature",
    "T_950": "950mb Air temperature",
    "T_900": "900mb Air temperature",
    "T_850": "850mb Air temperature",
    "Td": "Dewpoint",
    "Td_sfc": "2-meter Dewpoint",
    "Td_1000": "1000mb Dewpoint",
    "Td_950": "950mb Dewpoint",
    "Td_900": "900mb Dewpoint",
    "Td_850": "850mb Dewpoint",
    "Tv": "Virtual temperature",
    "Tv_sfc": "2-meter Virtual temperature",
    "Tv_1000": "1000mb Virtual temperature",
    "Tv_950": "950mb Virtual temperature",
    "Tv_900": "900mb Virtual temperature",
    "Tv_850": "850mb Virtual temperature",
    "Tw": "Wet-bulb temperature",
    "Tw_sfc": "2-meter Wet-bulb temperature",
    "Tw_1000": "1000mb Wet-bulb temperature",
    "Tw_950": "950mb Wet-bulb temperature",
    "Tw_900": "900mb Wet-bulb temperature",
    "Tw_850": "850mb Wet-bulb temperature",
    "theta": "Potential temperature",
    "theta_sfc": "2-meter Potential temperature",
    "theta_1000": "1000mb Potential temperature",
    "theta_950": "950mb Potential temperature",
    "theta_900": "900mb Potential temperature",
    "theta_850": "850mb Potential temperature",
    "theta_e": "Theta-E",
    "theta_e_sfc": "2-meter Theta-E",
    "theta_e_1000": "1000mb Theta-E",
    "theta_e_950": "950mb Theta-E",
    "theta_e_900": "900mb Theta-E",
    "theta_e_850": "850mb Theta-E",
    "theta_v": "Virtual potential temperature",
    "theta_v_sfc": "2-meter Virtual potential temperature",
    "theta_v_1000": "1000mb Virtual potential temperature",
    "theta_v_950": "950mb Virtual potential temperature",
    "theta_v_900": "900mb Virtual potential temperature",
    "theta_v_850": "850mb Virtual potential temperature",
    "theta_w": "Wet-bulb potential temperature",
    "theta_w_sfc": "2-meter Wet-bulb potential temperature",
    "theta_w_1000": "1000mb Wet-bulb potential temperature",
    "theta_w_950": "950mb Wet-bulb potential temperature",
    "theta_w_900": "900mb Wet-bulb potential temperature",
    "theta_w_850": "850mb Wet-bulb potential temperature",
    "u": "U-wind",
    "u_sfc": "10-meter U-wind",
    "u_1000": "1000mb U-wind",
    "u_950": "950mb U-wind",
    "u_900": "900mb U-wind",
    "u_850": "850mb U-wind",
    "v": "V-wind",
    "v_sfc": "10-meter V-wind",
    "v_1000": "1000mb V-wind",
    "v_950": "950mb V-wind",
    "v_900": "900mb V-wind",
    "v_850": "850mb V-wind",
    "q": "Specific humidity",
    "q_sfc": "2-meter Specific humidity",
    "q_1000": "1000mb Specific humidity",
    "q_950": "950mb Specific humidity",
    "q_900": "900mb Specific humidity",
    "q_850": "850mb Specific humidity",
    "r": "Mixing ratio",
    "r_sfc": "2-meter Mixing ratio",
    "r_1000": "1000mb Mixing ratio",
    "r_950": "950mb Mixing ratio",
    "r_900": "900mb Mixing ratio",
    "r_850": "850mb Mixing ratio",
    "RH": "Relative humidity",
    "RH_sfc": "2-meter Relative humidity",
    "RH_1000": "1000mb Relative humidity",
    "RH_950": "950mb Relative humidity",
    "RH_900": "900mb Relative humidity",
    "RH_850": "850mb Relative humidity",
    "sp_z": "Pressure/heights",
    "sp_z_sfc": "Surface pressure",
    "sp_z_1000": "1000mb Geopotential height",
    "sp_z_950": "950mb Geopotential height",
    "sp_z_900": "900mb Geopotential height",
    "sp_z_850": "850mb Geopotential height",
    "mslp_z": "Pressure/heights",
    "mslp_z_sfc": "Mean sea level pressure",
    "mslp_z_1000": "1000mb Geopotential height",
    "mslp_z_950": "950mb Geopotential height",
    "mslp_z_900": "900mb Geopotential height",
    "mslp_z_850": "850mb Geopotential height",
}

VERTICAL_LEVELS = {
    "surface": "Surface",
    "1000": "1000 hPa",
    "950": "950 hPa",
    "900": "900 hPa",
    "850": "850 hPa",
    "700": "700 hPa",
}

# some months do not have complete front labels, so we need to specify what dates (indices) do NOT have data for the final prediction datasets
missing_fronts_ind = {
    "2007-05": np.array([122, 128, 130, 132]),
    "2007-06": np.array([32, 34, 36, 200, 202]),
    "2007-11": np.array([126, 128, 130, 132]),
    "2007-12": np.array([206, 207]),
    "2018-03": 203,
    "2022-09": np.append(np.array([44, 46]), np.arange(48, 95.1, 1)).astype(int),
    "2022-10": np.append(np.arange(80, 87.1, 1), np.arange(160, 167.1, 1)).astype(int),
    "2022-11": 196,
}

# 3-hourly indices with missing satellite data
missing_satellite_ind = {
    "2018-09": np.array([78, 79, 80, 81, 82, 83, 142, 146]),
    "2018-10": np.append(np.array([86, 134]), np.arange(189, 237.1)).astype(int),
    "2018-11": np.append(
        np.arange(0, 99.1, 1), np.array([120, 121, 122, 123, 124, 125, 126, 159])
    ).astype(int),
    "2018-12": np.array([153, 157, 205, 206, 207]),
    "2019-01": 22,
    "2019-02": np.array([197, 198]),
    "2019-03": 215,
    "2019-04": 189,
    "2019-05": 237,
    "2019-06": np.array([213, 221, 222]),
    "2019-08": np.array([114, 115, 116, 117]),
    "2020-06": np.array([22, 23, 24, 25, 26, 27]),
    "2020-07": np.array([207, 208]),
    "2020-08": 86,
    "2021-01": 167,
    "2021-03": np.array([125, 181, 182, 183]),
    "2021-04": 231,
    "2021-06": np.array([116, 228, 229, 230]),
    "2021-07": np.append(np.array([67]), np.arange(170, 179.1, 1)),
    "2022-01": 112,
    "2022-04": 141,
    "2022-05": np.array([189, 190]),
    "2022-08": np.array([42, 43, 50, 51, 58]),
    "2022-09": np.array([100, 101, 102, 103]),
    "2022-11": np.array([55, 56, 134]),
}


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
    if type(fronts) in [xr.Dataset, xr.DataArray]:
        identifier = (
            fronts["identifier"].values if type(fronts) == xr.Dataset else fronts.values
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

    if type(fronts) == xr.Dataset:
        fronts["identifier"].values = identifier
    elif type(fronts) == xr.DataArray:
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

    if type(front_types) == str:
        front_types = [
            front_types,
        ]

    fronts_argument_type = type(fronts)

    if fronts_argument_type == xr.DataArray or fronts_argument_type == xr.Dataset:
        where_function = xr.where
    elif fronts_argument_type == np.ndarray:
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

    if fronts_argument_type == xr.Dataset or fronts_argument_type == xr.DataArray:
        fronts.attrs["front_types"] = front_types
        fronts.attrs["num_front_types"] = num_types
        fronts.attrs["labels"] = labels

    return fronts


def normalize_dataset(
    ds, method="standard", normalization_parameters=NORMALIZATION_PARAMS
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
    ds.close()

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
        normalized_ds = (ds_copy - norm_params.sel(param="min")) / (
            norm_params.sel(param="max") - norm_params.sel(param="min")
        )
    elif method == "standard":
        normalized_ds = (ds_copy - norm_params.sel(param="mean")) / norm_params.sel(
            param="std"
        )
    elif method == "standard_weighted":
        normalized_ds = (
            ds_copy - norm_params.sel(param="mean_weighted")
        ) / norm_params.sel(param="std_weighted")
    else:
        raise ValueError(
            "Unrecognized normalization method: %s. Valid normalization methods are 'min-max', 'standard', 'standard-weighted'."
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
    std_parallels = np.radians(std_parallels)

    if std_parallels[0] == std_parallels[1]:
        n = np.sin(std_parallels[0])
    else:
        n = np.divide(
            np.log(np.cos(std_parallels[0]) / np.cos(std_parallels[1])),
            np.log(
                np.tan(np.pi / 4 + std_parallels[1] / 2)
                / np.tan(np.pi / 4 + std_parallels[0] / 2)
            ),
        )
    F = (
        np.cos(std_parallels[0])
        * np.power(np.tan(np.pi / 4 + std_parallels[0] / 2), n)
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

    region_mask = xr.merge(
        [
            (regions.mask(ds[lon], ds[lat]) == i).expand_dims(
                {"index": np.atleast_1d(i)}
            )
            for i in indices
        ]
    ).max("index")["mask"]
    masked_ds = ds.where(region_mask, 1, 0)

    if region_crosses_prime_meridian:
        lons = masked_ds[lon]
        lon_east_hemi, lon_west_hemi = lons[lons <= 180], lons[lons > 180]
        masked_ds = masked_ds.reindex(
            {lon: np.concatenate([lon_west_hemi, lon_east_hemi])}
        )
        new_lons = np.concatenate([lon_west_hemi, lon_east_hemi + 360])
        masked_ds[lon] = new_lons

    return masked_ds
