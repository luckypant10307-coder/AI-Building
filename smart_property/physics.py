"""Small psychrometric helpers shared by the simulator and the rule engine."""

import numpy as np


def saturation_vapour_pressure(temp_c):
    """Tetens equation, kPa."""
    t = np.asarray(temp_c, dtype=float)
    return 0.61078 * np.exp(17.27 * t / (t + 237.3))


def vapour_pressure(temp_c, rh_pct):
    return saturation_vapour_pressure(temp_c) * np.asarray(rh_pct, dtype=float) / 100.0


def surface_rh(air_temp_c, air_rh_pct, surface_temp_c):
    """RH in the air film touching a surface that is colder than the room.

    The vapour pressure is set by the room air; the saturation pressure by the
    surface. A cold wall therefore sits at a higher RH than the room it is in,
    which is why condensation shows up on a thermal bridge first.
    """
    e = vapour_pressure(air_temp_c, air_rh_pct)
    es_surface = saturation_vapour_pressure(surface_temp_c)
    return np.clip(100.0 * e / es_surface, 0.0, 100.0)


def dew_point(temp_c, rh_pct):
    """Magnus-Tetens dew point, degC."""
    t = np.asarray(temp_c, dtype=float)
    rh = np.clip(np.asarray(rh_pct, dtype=float), 1e-3, 100.0)
    gamma = np.log(rh / 100.0) + 17.625 * t / (243.04 + t)
    return 243.04 * gamma / (17.625 - gamma)
