"""Per-room anomaly detection (component 1).

One Isolation Forest per room, fitted on that room's own first week. No
pretrained model knows what normal looks like inside one particular flat, so
the baseline has to be learned per room, from the room itself.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from .config import BASELINE_DAYS


FEATURES = ["wall_moisture", "wall_surface_rh", "air_rh", "flow_lpm", "pipe_db", "evap_gap"]


def _hourly_features(room_df):
    """Collapse 15-minute rows to hourly means and add the evaporation gap."""
    frame = room_df.set_index("ts").sort_index()
    frame = frame.assign(evap_gap=frame["air_temp"] - frame["wall_temp"])
    return frame[FEATURES].resample("1h").mean().dropna()


class RoomAnomalyDetector:
    """Learns one room's normal week, then scores every later hour against it."""

    def __init__(self, baseline_days=BASELINE_DAYS, contamination=0.02, seed=0):
        self.baseline_days = baseline_days
        self.model = IsolationForest(
            n_estimators=200, contamination=contamination, random_state=seed)
        self.columns = None
        self.baseline_end = None

    def fit(self, room_df):
        hourly = _hourly_features(room_df)
        self.baseline_end = hourly.index.min() + pd.Timedelta(days=self.baseline_days)
        baseline = hourly[hourly.index < self.baseline_end]
        if len(baseline) < 24:
            raise ValueError("need at least a day of baseline readings")
        self.columns = list(baseline.columns)
        self.model.fit(baseline.values)
        return self

    def score(self, room_df):
        """Anomaly score per hour: higher means further outside the normal week."""
        hourly = _hourly_features(room_df)[self.columns]
        raw = self.model.score_samples(hourly.values)   # less negative == more normal
        return pd.Series(-raw, index=hourly.index, name="anomaly")

    def recent_anomaly_fraction(self, room_df, window_days=1.0, threshold=None):
        """Share of the last `window_days` of hours that fall outside the baseline.

        The cut is taken from the baseline's own worst hour, so a quiet room
        keeps a fraction near zero rather than drifting with the flat.
        """
        scores = self.score(room_df)
        baseline = scores[scores.index < self.baseline_end]
        if threshold is None:
            threshold = baseline.quantile(0.995) + 0.02
        cutoff = scores.index.max() - pd.Timedelta(days=window_days)
        recent = scores[scores.index > cutoff]
        if recent.empty:
            return 0.0, threshold
        return float((recent > threshold).mean()), float(threshold)
