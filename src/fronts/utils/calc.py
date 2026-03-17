"""
Functions for deriving various thermodynamic variables.

References
----------
* Davies-Jones 2008: https://doi.org/10.1175/2007MWR2224.1
* Stull 2011: https://doi.org/10.1175/JAMC-D-11-0143.1

Author: Andrew Justin (andrewjustinwx@gmail.com)
Script version: 2025.5.3
"""

import logging

import numpy as np
import tensorflow as tf
import xarray as xr
from scipy.ndimage import maximum_filter

from fronts.utils import data_utils

logger = logging.getLogger("fronts.utils.calc")

Rd = 287.04  # Gas constant for dry air (J/kg/K)
Rv = 461.5  # Gas constant for water vapor (J/kg/K)
Cpd = 1005.7  # Specific heat of dry air at constant pressure (J/kg/K)
kd = Rd / Cpd  # Exponential constant for potential temperature
epsilon = Rd / Rv
e_knot = 611.2  # Pa
Lv = 2.257e6  # Latent heat of vaporization for water vapor (J/kg)


def saturation_vapor_pressure(T):
    """
    Calculates saturation vapor pressure from temperature.

    Parameters
    ----------
    T: float or iterable object
        Air temperature expressed as kelvin (K).

    Returns
    -------
    e: float or iterable object
        Vapor pressure expressed as Pascals (Pa).
    """

    exp_func = tf.math.exp if tf.is_tensor(T) else np.exp
    T = T - 273.15  # K -> C
    e = e_knot * exp_func(17.67 * T / (T + 243.5))
    return e


def dewpoint_from_mixing_ratio(P, r):
    """
    Calculates dewpoint temperature from mixing ratio, and pressure.

    Parameters
    ----------
    P: float or iterable object
        Air pressure expressed as pascals (Pa).
    r: float or iterable object
        Mixing ratio expressed as grams of water vapor per gram of dry air (unitless; g/g or kg/kg).

    Returns
    -------
    Td: float or iterable object
        Dewpoint temperature expressed as kelvin (K).

    Examples
    --------
    >>> P = 1e5  # Pa
    >>> r = 20 / 1000  # g/kg -> kg/kg
    >>> Td = dewpoint_from_mixing_ratio(P, r)
    >>> Td
    297.87277930360676

    >>> P = np.arange(800.0, 1001.0, 25) * 100  # Pa
    >>> P
    array([ 80000.,  82500.,  85000.,  87500.,  90000.,  92500.,  95000.,
            97500., 100000.])
    >>> r = np.arange(5, 25.01, 2.5) / 1000  # g/kg -> kg/kg
    >>> r
    array([0.005 , 0.0075, 0.01  , 0.0125, 0.015 , 0.0175, 0.02  , 0.0225,
           0.025 ])
    >>> Td = dewpoint_from_mixing_ratio(P, r)
    >>> Td
    array([273.74253532, 279.8787204 , 284.52674244, 288.32976779,
           291.58260633, 294.44604583, 297.01785056, 299.36210297,
           301.52319595])
    """

    log_func = tf.math.log if tf.is_tensor(P) else np.log

    P = P / 100  # Pa -> hPa

    e = r * P / (epsilon + r)  # vapor pressure

    Td = 243.5 * log_func(e / 6.112) / (17.67 - log_func(e / 6.112)) + 273.15  # C -> K
    return Td


def dewpoint_from_relative_humidity(T, RH):
    """
    Calculates dewpoint temperature from relative humidity and temperature.

    Parameters
    ----------
    T: float or iterable object
        Air temperature expressed as kelvin (K).
    RH: float or iterable object
        Relative humidity expressed as a decimal (e.g. 57% = 0.57).

    Returns
    -------
    Td: float or iterable object
        Dewpoint temperature expressed as kelvin (K).

    Examples
    --------
    >>> P = 1e5  # Pa
    >>> T = 300  # K
    >>> RH = 0.8
    >>> Td = dewpoint_from_relative_humidity(T, RH)
    >>> Td
    296.2618676251417

    >>> P = np.arange(800.0, 1001.0, 25) * 100  # Pa
    >>> P
    array([ 80000,  82500,  85000,  87500,  90000,  92500,  95000,  97500,
           100000])
    >>> T = np.arange(270.0, 311.0, 5)  # K
    >>> T
    array([270, 275, 280, 285, 290, 295, 300, 305, 310])
    >>> RH = np.arange(0.5, 0.91, 0.05)
    >>> RH
    array([0.5 , 0.55, 0.6 , 0.65, 0.7 , 0.75, 0.8 , 0.85, 0.9 ])
    >>> Td = dewpoint_from_relative_humidity(T, RH)
    >>> Td
    array([261.04058296, 266.91163292, 272.7737636 , 278.63451989,
           284.499794  , 290.37429558, 296.26186763, 302.16570498,
           308.08850911])
    """

    log_func = tf.math.log if tf.is_tensor(T) else np.log

    # T is converted to C in saturation_vapor_pressure function
    e = saturation_vapor_pressure(T) * RH

    Td = (
        243.5 * log_func(e / e_knot) / (17.67 - log_func(e / e_knot)) + 273.15
    )  # C -> K
    return Td


def dewpoint_from_specific_humidity(P, q):
    """
    Calculates dewpoint temperature from specific humidity, and pressure.

    Parameters
    ----------
    P: float or iterable object
        Air pressure expressed as pascals (Pa).
    q: float or iterable object
        Specific humidity expressed as grams of water vapor per gram of dry air (unitless; g/g or kg/kg).

    Returns
    -------
    Td: float or iterable object
        Dewpoint temperature expressed as kelvin (K).

    Examples
    --------
    >>> P = 1e5  # Pa
    >>> q = 20 / 1000  # g/kg -> kg/kg
    >>> Td = dewpoint_from_specific_humidity(P, q)
    >>> Td
    298.20035572272803

    >>> P = np.arange(800.0, 1001.0, 25) * 100  # Pa
    >>> P
    array([ 80000,  82500,  85000,  87500,  90000,  92500,  95000,  97500,
           100000])
    >>> q = np.arange(5, 25.01, 2.5) / 1000  # g/kg -> kg/kg
    >>> q
    array([0.005 , 0.0075, 0.01  , 0.0125, 0.015 , 0.0175, 0.02  , 0.0225,
           0.025 ])
    >>> Td = dewpoint_from_specific_humidity(P, q)
    >>> Td
    array([273.81141134, 279.98701245, 284.67615887, 288.52165869,
           291.8180981 , 294.72610931, 297.34334081, 299.73378468,
           301.94176072])
    """

    log_func = tf.math.log if tf.is_tensor(P) else np.log

    r = q / (1 - q)  # mixing ratio
    e = r * P / (epsilon + r)  # vapor pressure

    Td = (
        243.5 * log_func(e / e_knot) / (17.67 - log_func(e / e_knot)) + 273.15
    )  # C -> K
    return Td


def mixing_ratio_from_dewpoint(P, Td):
    """
    Calculates mixing ratio from dewpoint temperature and air pressure.

    Parameters
    ----------
    P: float or iterable object
        Air pressure expressed as pascals (Pa).
    Td: float or iterable object
        Dewpoint temperature expressed as kelvin (K).

    Returns
    -------
    r: float or iterable object
        Mixing ratio expressed as grams of water per gram of dry air (unitless; g/g or kg/kg).

    Examples
    --------
    >>> P = 1e5  # Pa
    >>> Td = 290  # K
    >>> r = mixing_ratio_from_dewpoint(P, Td)
    >>> r
    0.010947893449979635

    >>> P = np.arange(800, 1001, 25) * 100  # Pa
    >>> P
    array([ 80000,  82500,  85000,  87500,  90000,  92500,  95000,  97500,
           100000])
    >>> Td = np.arange(260, 301, 5)  # K
    >>> Td
    array([260, 265, 270, 275, 280, 285, 290, 295, 300])
    >>> r = mixing_ratio_from_dewpoint(P, Td)
    >>> r
    array([0.00192723, 0.0026682 , 0.00365056, 0.00493959, 0.00661507,
           0.00877413, 0.01153478, 0.01504042, 0.01946563])
    """
    e = saturation_vapor_pressure(Td)
    r = epsilon * e / (P - e)  # mixing ratio
    return r


def potential_temperature(P, T):
    """
    Returns potential temperature expressed as kelvin (K).

    Parameters
    ----------
    P: float or iterable object
        Air pressure expressed as pascals (Pa).
    T: float or iterable object
        Air temperature expressed as kelvin (K).

    Returns
    -------
    theta: float or iterable object
        Potential temperature expressed as kelvin (K).

    Examples
    --------
    >>> P = 9e4  # Pa
    >>> T = 275  # K
    >>> theta = potential_temperature(P, T)
    >>> theta
    283.3951954331142

    >>> P = np.arange(800, 1001, 25) * 100  # Pa
    >>> P
    array([ 80000,  82500,  85000,  87500,  90000,  92500,  95000,  97500,
           100000])
    >>> T = np.arange(270, 311, 5)  # K
    >>> T
    array([270, 275, 280, 285, 290, 295, 300, 305, 310])
    >>> theta = potential_temperature(P, T)
    >>> theta
    array([287.75518363, 290.52120387, 293.2937428 , 296.07144546,
           298.85311518, 301.63769299, 304.42424008, 307.21192283,
           310.        ])
    """
    theta = (
        T * tf.pow(1e5 / P, kd)
        if tf.is_tensor(T) and tf.is_tensor(P)
        else T * np.power(1e5 / P, kd)
    )
    return theta


def relative_humidity_from_dewpoint(T, Td):
    """
    Returns relative humidity from temperature and dewpoint temperature.

    Parameters
    ----------
    T: float or iterable object
        Air temperature expressed as kelvin (K).
    Td: float or iterable object
        Dewpoint temperature expressed as kelvin (K).

    Returns
    -------
    RH: float or iterable object
        Relative humidity.

    Examples
    --------
    >>> T = 300.0  # K
    >>> Td = 290.0  # K
    >>> RH = relative_humidity_from_dewpoint(T, Td)
    >>> RH
    0.542647138543181

    >>> T = np.arange(270.0, 311.0, 5)  # K
    >>> T
    array([270, 275, 280, 285, 290, 295, 300, 305, 310])
    >>> Td = np.arange(260.0, 301.0, 5)  # K
    >>> Td
    array([260, 265, 270, 275, 280, 285, 290, 295, 300])
    >>> RH = relative_humidity_from_dewpoint(T, Td)
    >>> RH
    array([0.45971571, 0.47466998, 0.48916171, 0.50319697, 0.51678347,
           0.52993019, 0.54264714, 0.55494506, 0.56683529])
    """

    e = saturation_vapor_pressure(Td)
    es = saturation_vapor_pressure(T)
    RH = e / es
    return RH


def specific_humidity_from_relative_humidity(P, T, RH):
    """
    Calculates specific humidity from relative humidity, air temperature, and pressure.

    Parameters
    ----------
    P: float or iterable object
        Air pressure expressed as pascals (Pa).
    T: float or iterable object
        Air temperature expressed as kelvin (K).
    RH: float or iterable object
        Relative humidity.

    Returns
    -------
    q: float or iterable object
        Specific humidity expressed as grams of water vapor per gram of dry air (unitless; g/g or kg/kg).

    Examples
    --------
    >>> P = 1e5  # Pa
    >>> T = 300  # K
    >>> RH = 0.8
    >>> q = specific_humidity_from_relative_humidity(P, T, RH)
    >>> q * 1000  # kg/kg -> g/kg
    15.239787946316241

    >>> P = np.arange(800, 1001, 25) * 100  # Pa
    >>> P
    array([ 80000,  82500,  85000,  87500,  90000,  92500,  95000,  97500,
           100000])
    >>> T = np.arange(270, 311, 5)  # K
    >>> T
    array([270, 275, 280, 285, 290, 295, 300, 305, 310])
    >>> RH = np.arange(0.5, 0.91, 0.05)
    >>> RH
    array([0.5 , 0.55, 0.6 , 0.65, 0.7 , 0.75, 0.8 , 0.85, 0.9 ])
    >>> q = specific_humidity_from_relative_humidity(P, T, RH)
    >>> q * 1000
    array([ 1.93030564,  2.86370132,  4.16882688,  5.96677284,  8.41051444,
           11.6918278 , 16.04970636, 21.78077898, 29.25247145])
    """
    es = saturation_vapor_pressure(T)
    e = RH * es
    r = epsilon * e / (P - e)
    q = r / (r + 1)
    return q


def equivalent_potential_temperature(P, T, Td):
    """
    Calculates equivalent potential temperature (theta-e) from temperature, dewpoint, and pressure.

    Parameters
    ----------
    P: float or iterable object
        Air pressure expressed as pascals (Pa).
    T: float or iterable object
        Air temperature expressed as kelvin (K).
    Td: float or iterable object
        Dewpoint temperature expressed as kelvin (K).

    Returns
    -------
    theta_e: float or iterable object
        Equivalent potential temperature expressed as kelvin (K).

    Examples
    --------
    >>> P = 1e5  # Pa
    >>> T = 300  # K
    >>> Td = 290  # K
    >>> theta_e = equivalent_potential_temperature(P, T, Td)
    >>> theta_e
    326.52430009577137

    >>> P = np.arange(800, 1001, 25) * 100  # Pa
    >>> P
    array([ 80000,  82500,  85000,  87500,  90000,  92500,  95000,  97500,
           100000])
    >>> T = np.arange(270, 311, 5)  # K
    >>> T
    array([270, 275, 280, 285, 290, 295, 300, 305, 310])
    >>> Td = np.arange(260, 301, 5)  # K
    >>> Td
    array([260, 265, 270, 275, 280, 285, 290, 295, 300])
    >>> theta_e = equivalent_potential_temperature(P, T, Td)
    >>> theta_e
    array([292.582033  , 297.1606042 , 302.3295548 , 308.2501438 ,
           315.12588597, 323.21500183, 332.84798857, 344.45298499,
           358.59322677])
    """
    RH = relative_humidity_from_dewpoint(T, Td)
    theta = potential_temperature(P, T)
    rv = mixing_ratio_from_dewpoint(P, Td)
    theta_e = (
        theta * tf.pow(RH, -rv * Rv / Cpd) * tf.exp(Lv * rv / (Cpd * T))
        if all(tf.is_tensor(var) for var in [T, Td, P])
        else theta * np.power(RH, -rv * Rv / Cpd) * np.exp(Lv * rv / (Cpd * T))
    )
    return theta_e


def wet_bulb_temperature(T, Td):
    """
    Calculates wet-bulb temperature from temperature and dewpoint.

    Parameters
    ----------
    T: float or iterable object
        Air temperature expressed as kelvin (K).
    Td: float or iterable object
        Dewpoint temperature expressed as kelvin (K).

    Returns
    -------
    Tw: float or iterable object
        Wet-bulb temperature expressed as kelvin (K).

    Examples
    --------
    >>> T = 300  # K
    >>> Td = 290  # K
    >>> Tw = wet_bulb_temperature(T, Td)
    >>> Tw
    293.8520102695189

    >>> T = np.arange(270, 311, 5)  # K
    >>> T
    array([270, 275, 280, 285, 290, 295, 300, 305, 310])
    >>> Td = np.arange(260, 301, 5)  # K
    >>> Td
    array([260, 265, 270, 275, 280, 285, 290, 295, 300])
    >>> Tw = wet_bulb_temperature(T, Td)
    >>> Tw
    array([266.93692268, 271.30728293, 275.72885706, 280.1976556 ,
           284.70999386, 289.26248252, 293.85201027, 298.47572385,
           303.13100751])
    """
    RH = relative_humidity_from_dewpoint(T, Td) * 100
    c1 = 0.151977
    c2 = 8.313659
    c3 = 1.676331
    c4 = 0.00391838
    c5 = 0.023101
    c6 = 4.686035

    # Wet-bulb temperature approximation (Stull 2011, Eq. 1)
    if tf.is_tensor(T):
        Tw = (T - 273.15) * tf.atan(c1 * tf.sqrt(RH + c2))
        Tw += tf.atan(T - 273.15 + RH) - tf.atan(RH - c3)
        Tw += c4 * tf.pow(RH, 1.5) * tf.atan(c5 * RH)
    else:
        Tw = (T - 273.15) * np.arctan(c1 * np.sqrt(RH + c2))
        Tw += np.arctan(T - 273.15 + RH) - np.arctan(RH - c3)
        Tw += c4 * np.power(RH, 1.5) * np.arctan(c5 * RH)
    Tw += 273.15 - c6

    return Tw


def wet_bulb_potential_temperature(P, T, Td):
    """
    Returns wet-bulb potential temperature (theta-w) in kelvin (K).

    Parameters
    ----------
    P: float or iterable object
        Air pressure expressed as pascals (Pa).
    T: float or iterable object
        Air temperature expressed as kelvin (K).
    Td: float or iterable object
        Dewpoint temperature expressed as kelvin (K).

    Returns
    -------
    theta_w: float or iterable object
        Wet-bulb potential temperature expressed as kelvin (K).

    Examples
    --------
    >>> T = 300  # K
    >>> Td = 290  # K
    >>> P = 1e5  # Pa
    >>> theta_w = wet_bulb_potential_temperature(T, Td, P)
    >>> theta_w
    array(290.65670769)

    >>> T = np.arange(270, 311, 5)  # K
    >>> T
    array([270, 275, 280, 285, 290, 295, 300, 305, 310])
    >>> Td = np.arange(260, 301, 5)  # K
    >>> Td
    array([260, 265, 270, 275, 280, 285, 290, 295, 300])
    >>> P = np.arange(800, 1001, 25) * 100  # Pa
    >>> P
    array([ 80000,  82500,  85000,  87500,  90000,  92500,  95000,  97500,
           100000])
    >>> theta_w = wet_bulb_potential_temperature(T, Td, P)
    >>> theta_w
    array([277.86226914, 280.00624048, 282.24273162, 284.58957316,
           287.06192678, 289.67108554, 292.42350276, 295.32019538,
           298.35666949])
    """

    # Wet-bulb potential temperature constants and X variable (Davies-Jones 2008, Section 3).
    a0 = 7.101574
    a1 = -20.68208
    a2 = 16.11182
    a3 = 2.574631
    a4 = -5.205688
    b1 = -3.552497
    b2 = 3.781782
    b3 = -0.6899655
    b4 = -0.592934
    C = 273.15

    theta_e = equivalent_potential_temperature(P, T, Td)
    X = theta_e / C

    # Wet-bulb potential temperature approximation (Davies-Jones 2008, Eq. 3.8).
    if all(tf.is_tensor(var) for var in [P, T, Td]):
        theta_wc = theta_e - tf.exp(
            (
                a0
                + (a1 * X)
                + (a2 * tf.pow(X, 2))
                + (a3 * tf.pow(X, 3))
                + (a4 * tf.pow(X, 4))
            )
            / (
                1
                + (b1 * X)
                + (b2 * tf.pow(X, 2))
                + (b3 * tf.pow(X, 3))
                + (b4 * tf.pow(X, 4))
            )
        )
        theta_w = tf.where(theta_e > 173.15, theta_wc, theta_e)
    else:
        theta_wc = theta_e - np.exp(
            (
                a0
                + (a1 * X)
                + (a2 * np.power(X, 2))
                + (a3 * np.power(X, 3))
                + (a4 * np.power(X, 4))
            )
            / (
                1
                + (b1 * X)
                + (b2 * np.power(X, 2))
                + (b3 * np.power(X, 3))
                + (b4 * np.power(X, 4))
            )
        )
        theta_w = np.where(theta_e > 173.15, theta_wc, theta_e)

    return theta_w


def virtual_potential_temperature(P, T, Td):
    """
    Calculates virtual potential temperature (theta-v) from temperature, dewpoint, and pressure.

    Parameters
    ----------
    P: float or iterable object
        Air pressure expressed as pascals (Pa).
    T: float or iterable object
        Air temperature expressed as kelvin (K).
    Td: float or iterable object
        Dewpoint temperature expressed as kelvin (K).


    Returns
    -------
    theta_v: float or iterable object
        Virtual potential temperature expressed as kelvin (K).

    Examples
    --------
    >>> P = 1e5  # Pa
    >>> T = 300  # K
    >>> Td = 290  # K
    >>> theta_v = virtual_potential_temperature(P, T, Td)
    >>> theta_v
    301.9745879930382

    >>> P = np.arange(800, 1001, 25) * 100  # Pa
    >>> P
    array([ 80000,  82500,  85000,  87500,  90000,  92500,  95000,  97500,
           100000])
    >>> T = np.arange(270, 311, 5)  # K
    >>> T
    array([270, 275, 280, 285, 290, 295, 300, 305, 310])
    >>> Td = np.arange(260, 301, 5)  # K
    >>> Td
    array([260, 265, 270, 275, 280, 285, 290, 295, 300])
    >>> theta_v = virtual_potential_temperature(P, T, Td)
    >>> theta_v
    array([288.09159758, 290.9910892 , 293.94212815, 296.95595223,
           300.04678011, 303.23228364, 306.53413747, 309.97866221,
           313.59758378])
    """
    Tv = virtual_temperature_from_dewpoint(P, T, Td)
    theta_v = potential_temperature(P, Tv)
    return theta_v


def virtual_temperature_from_mixing_ratio(T, r):
    """
    Calculates virtual temperature from temperature and mixing ratio.

    Parameters
    ----------
    T: float or iterable object
        Air temperature expressed as kelvin (K).
    r: float or iterable object
        Mixing ratio expressed as grams of water per gram of dry air (unitless; g/g or kg/kg).

    Returns
    -------
    Tv: float or iterable object
        Virtual temperature expressed as kelvin (K).

    Examples
    --------
    >>> T = 300  # K
    >>> r = 20 / 1000  # g/kg -> kg/kg
    >>> Tv = virtual_temperature_from_mixing_ratio(T, r)
    >>> Tv
    303.5752344416027

    >>> T = np.arange(270, 311, 5)  # K
    >>> T
    array([270, 275, 280, 285, 290, 295, 300, 305, 310])
    >>> r = np.arange(5, 25.01, 2.5) / 1000  # g/kg -> kg/kg
    >>> r
    array([0.005 , 0.0075, 0.01  , 0.0125, 0.015 , 0.0175, 0.02  , 0.0225,
           0.025 ])
    >>> Tv = virtual_temperature_from_mixing_ratio(T, r)
    >>> Tv
    array([270.81643413, 276.24423481, 281.68496197, 287.13851986,
           292.60481366, 298.08374951, 303.57523444, 309.07917641,
           314.59548427])
    """
    return T * (1 + (r / epsilon)) / (1 + r)


def virtual_temperature_from_dewpoint(P, T, Td):
    """
    Calculates virtual temperature from temperature, dewpoint, and pressure.

    Parameters
    ----------
    P: float or iterable object
        Air pressure expressed as pascals (Pa).
    T: float or iterable object
        Air temperature expressed as kelvin (K).
    Td: float or iterable object
        Dewpoint temperature expressed as kelvin (K).

    Returns
    -------
    Tv: float or iterable object
        Virtual temperature expressed as kelvin (K).

    Examples
    --------
    >>> P = 1e5  # Pa
    >>> T = 300  # K
    >>> Td = 290  # K
    >>> Tv = virtual_temperature_from_dewpoint(P, T, Td)
    >>> Tv
    301.9745879930382

    >>> P = np.arange(800, 1001, 25) * 100  # Pa
    >>> P
    array([ 80000,  82500,  85000,  87500,  90000,  92500,  95000,  97500,
           100000])
    >>> T = np.arange(270, 311, 5)  # K
    >>> T
    array([270, 275, 280, 285, 290, 295, 300, 305, 310])
    >>> Td = np.arange(260, 301, 5)  # K
    >>> Td
    array([260, 265, 270, 275, 280, 285, 290, 295, 300])
    >>> Tv = virtual_temperature_from_dewpoint(P, T, Td)
    >>> Tv
    array([270.3156564 , 275.44478153, 280.61899683, 285.85143108,
           291.15830423, 296.55950086, 302.07923396, 307.74681888,
           313.59758378])
    """
    r = mixing_ratio_from_dewpoint(P, Td)
    Tv = virtual_temperature_from_mixing_ratio(T, r)
    return Tv


def advection(field, u, v, lons, lats):
    """
    Calculates advection of a scalar field.

    Parameters
    ----------
    field: Array-like with shape (M, N)
        Scalar field that is being advected.
    u: Array-like with shape (M, N)
        Zonal wind velocities expressed as meters per second (m/s).
    v: Array-like with shape (M, N)
        Meridional wind velocities expressed as meters per second (m/s).
    lons: Array-like with shape (M, )
        Longitude values expressed as degrees east.
    lats: Array-like with shape (N, )
        Latitude values expressed as degrees north.
    """

    advect = np.ones(u.shape) * np.nan

    Lons, Lats = np.meshgrid(lons, lats)
    x, y = data_utils.haversine(Lons, Lats)  # x and y are expressed as kilometers
    x = x.T
    y = y.T  # transpose x and y so arrays have shape M x N
    x *= 1e3
    y *= 1e3  # convert x and y coordinates to meters

    dfield_dx = np.diff(field, axis=0) / np.diff(x, axis=0)
    dfield_dy = np.diff(field, axis=1) / np.diff(y, axis=1)

    advect[:-1, :-1] = -(u[:-1, :-1] * dfield_dx[:, :-1]) - (
        v[:-1, :-1] * dfield_dy[:-1, :]
    )

    return advect


GEOPOTENTIAL_TO_DAM = 98.0665  # geopotential (m²/s²) → geopotential height (dam)


def convert_to_pascals(levels: xr.DataArray) -> xr.DataArray:
    """Convert pressure levels from hPa (int) or "surface" (str) to Pa.
    Convert surface if exists to 1013.25 hPa and then convert to Pascals for all levels.

    Args:
        levels: Pressure levels in hPa (int)

    Returns a DataArray of pressure levels in Pascals.
    """
    levels_pa = levels * 100
    return levels_pa


def dewpoint(specific_humidity: xr.DataArray, level: xr.DataArray) -> xr.DataArray:
    """Derive dewpoint temperature from specific_humidity and pressure.

    Args:
        specific_humidity: Specific humidity in kg/kg.
        level: Pressure level in hPa (int) and/or "surface" (str).

    Returns a DataArray of dewpoint temperatures.
    """
    # Build a pressure array matching the data shape from the level coordinate.
    # Level values are hPa (int) or "surface" (str).  For "surface" we
    # approximate with 1013.25 hPa.
    levels_pa = convert_to_pascals(level)
    dewpoint = dewpoint_from_specific_humidity(levels_pa, specific_humidity)
    return dewpoint


def virtual_temperature(
    temperature: xr.DataArray, dewpoint: xr.DataArray, level: xr.DataArray
) -> xr.DataArray:
    """Derive virtual temperature from temperature, dewpoint, and pressure.

    Args:
        temperature: Air temperature in K.
        dewpoint: Dewpoint temperature in K.
        level: Pressure level in hPa (int) and/or "surface" (str).
    """
    levels_pa = convert_to_pascals(level)
    virtual_temperature = virtual_temperature_from_dewpoint(
        levels_pa, temperature, dewpoint
    )
    return virtual_temperature


def relative_humidity(
    temperature: xr.DataArray, dewpoint: xr.DataArray
) -> xr.DataArray:
    """Derive relative humidity from temperature and dewpoint.

    Args:
        temperature: Air temperature in K.
        dewpoint: Dewpoint temperature in K.

    Returns a DataArray of relative humidity values.
    """
    relative_humidity = relative_humidity_from_dewpoint(temperature, dewpoint)
    return relative_humidity


def theta_e(
    temperature: xr.DataArray, dewpoint: xr.DataArray, level: xr.DataArray
) -> xr.DataArray:
    """Derive equivalent potential temperature from temperature, dewpoint, and pressure.

    Args:
        temperature: Air temperature in K.
        dewpoint: Dewpoint temperature in K.
        level: Pressure level in hPa (int) and/or "surface" (str).

    Returns a DataArray of equivalent potential temperature values.
    """
    levels_pa = convert_to_pascals(level)
    return equivalent_potential_temperature(levels_pa, temperature, dewpoint)


def geopotential_height(geopotential: xr.DataArray) -> xr.DataArray:
    """Convert geopotential (m²/s²) to geopotential height in decameters.

    Args:
        geopotential: Geopotential in m²/s².

    Returns a DataArray of geopotential height in decameters.
    """
    geopotential_height = geopotential / GEOPOTENTIAL_TO_DAM
    return geopotential_height

def _expand_2d(arr_2d: np.ndarray, iterations: int) -> np.ndarray:
    size = 2 * iterations + 1
    zero_mask = arr_2d == 0
    result = np.zeros_like(arr_2d)
    for label in np.unique(arr_2d[arr_2d > 0]):
        dilated = maximum_filter(arr_2d == label, size=size)
        result = np.where(dilated & zero_mask & (result < label), label, result)
    return np.where(arr_2d > 0, arr_2d, result)


def _expand_block(block: np.ndarray, iterations: int) -> np.ndarray:
    return np.stack([_expand_2d(block[t], iterations) for t in range(block.shape[0])])


def _expand_dataarray(data: xr.DataArray, iterations: int) -> xr.DataArray:
    is_2d = data.ndim == 2
    if is_2d:
        data = data.expand_dims("time", axis=0)

    result = xr.apply_ufunc(
        _expand_block,
        data,
        kwargs={"iterations": iterations},
        dask="parallelized",
        output_dtypes=[data.dtype],
        dask_gufunc_kwargs={"allow_rechunk": False},
    )

    if is_2d:
        result = result.squeeze("time")

    return result


def maybe_expand_fronts_parallelized(
    fronts: np.ndarray | xr.Dataset | xr.DataArray, iterations: int = 1
):
    if isinstance(fronts, xr.Dataset):
        fronts["identifier"] = _expand_dataarray(fronts["identifier"], iterations)
        return fronts
    elif isinstance(fronts, xr.DataArray):
        return _expand_dataarray(fronts, iterations)
    else:
        identifier = fronts
        is_2d = identifier.ndim == 2
        if is_2d:
            identifier = np.expand_dims(identifier, axis=0)
        return np.stack([_expand_2d(identifier[t], iterations) for t in range(identifier.shape[0])])