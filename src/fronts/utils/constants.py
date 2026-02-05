import numpy as np

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
