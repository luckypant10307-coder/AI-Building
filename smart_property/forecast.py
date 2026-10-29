"""Damage forecaster (component 3).

Forecasts wall moisture and wall-surface humidity a week ahead, then reads off
the days until visible damp or mould risk.

Two back ends. The default is a robust linear trend with a band taken from its
own residuals -- no dependencies, always available. If `chronos-forecasting`
is installed, Chronos-2 is used instead: it is zero-shot, takes a context of up
to 8,192 points and a prediction length of up to 1,024, so a 168-hour horizon
fits in a single pass.

Series are resampled to hourly first. At 15-minute spacing a 7-day horizon is
672 steps, which exceeds Chronos-Bolt's 64-step maximum and would have to be
rolled; hourly keeps the horizon inside one pass and costs nothing that matters
for a figure reported in days.
"""

import numpy as np
import pandas as pd

from .config import MOULD_RISK_SURFACE_RH, VISIBLE_DAMP_MOISTURE

HORIZON_HOURS = 168          # 7 days
CONTEXT_HOURS = 168


def chronos_available():
    try:
        import chronos  # noqa: F401
        return True
    except Exception:
        return False


def _hourly(series):
    return series.resample("1h").mean().interpolate().dropna()


def _trend_forecast(history, horizon=HORIZON_HOURS):
    """Least-squares trend through the context, with a band from the residuals."""
    y = history.to_numpy()[-CONTEXT_HOURS:]
    x = np.arange(len(y), dtype=float)
    slope, intercept = np.polyfit(x, y, 1)
    resid = y - (slope * x + intercept)
    sigma = float(np.std(resid)) or 1e-6

    future_x = np.arange(len(y), len(y) + horizon, dtype=float)
    mid = slope * future_x + intercept
    # the band widens with the square root of how far ahead we are looking
    spread = 1.28 * sigma * np.sqrt(1.0 + np.arange(1, horizon + 1) / 24.0)
    return mid, mid - spread, mid + spread


def _chronos_forecast(history, horizon=HORIZON_HOURS):
    from chronos import Chronos2Pipeline

    pipeline = Chronos2Pipeline.from_pretrained("amazon/chronos-2")
    context = history.to_numpy()[-2048:]
    frame = pd.DataFrame({
        "id": "room",
        "timestamp": history.index[-len(context):],
        "target": context,
    })
    pred = pipeline.predict_df(
        frame, prediction_length=horizon, quantile_levels=[0.1, 0.5, 0.9],
        id_column="id", timestamp_column="timestamp", target="target")
    return (pred["0.5"].to_numpy(), pred["0.1"].to_numpy(), pred["0.9"].to_numpy())


def _first_crossing_days(mid, low, high, threshold, already_over):
    """Hours until the forecast crosses `threshold`, converted to days."""
    if already_over:
        return {"days": 0.0, "days_low": 0.0, "days_high": 0.0, "crosses": True}

    def cross(curve):
        hit = np.flatnonzero(curve >= threshold)
        return float(hit[0] + 1) / 24.0 if hit.size else None

    centre = cross(mid)
    if centre is None:
        return {"days": None, "days_low": cross(high), "days_high": None,
                "crosses": False}
    # the upper band crosses soonest, so it gives the earliest plausible date
    return {"days": centre, "days_low": cross(high), "days_high": cross(low),
            "crosses": True}


def forecast_room(room_df, backend="auto"):
    """Days until visible damp and until mould risk, for one room."""
    frame = room_df.set_index("ts").sort_index()
    use_chronos = backend == "chronos" or (backend == "auto" and chronos_available())
    engine = _chronos_forecast if use_chronos else _trend_forecast

    out = {"backend": "chronos-2" if use_chronos else "trend"}
    for column, threshold, name in [
        ("wall_moisture", VISIBLE_DAMP_MOISTURE, "visible_damp"),
        ("wall_surface_rh", MOULD_RISK_SURFACE_RH, "mould_risk"),
    ]:
        history = _hourly(frame[column])
        mid, low, high = engine(history)
        current = float(history.iloc[-1])
        out[name] = _first_crossing_days(mid, low, high, threshold, current >= threshold)
        out[name]["now"] = current
        out[name]["threshold"] = threshold
        out[name]["forecast_end"] = float(mid[-1])
    return out
