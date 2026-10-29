"""End-to-end pipeline: readings in, alerts and advice out."""

from .advisor import advise
from .config import BASELINE_DAYS, ROOMS
from .detector import RoomAnomalyDetector
from .forecast import forecast_room
from .fusion import fuse
from .rules import classify_room


def analyse_room(room_df, baseline_days=BASELINE_DAYS, language="en",
                 backend="auto", llm=None, seed=0):
    """Run every detector over one room and fuse them into a single alert."""
    detector = RoomAnomalyDetector(baseline_days=baseline_days, seed=seed).fit(room_df)
    anomaly_fraction, threshold = detector.recent_anomaly_fraction(room_df)

    rule_result = classify_room(room_df, baseline_days=baseline_days)
    forecast = forecast_room(room_df, backend=backend)
    alert = fuse(rule_result, anomaly_fraction, forecast)

    return {
        "room": str(room_df["room"].iloc[0]),
        "alert": alert,
        "forecast": forecast,
        "measures": rule_result["measures"],
        "ranked": rule_result["ranked"],
        "anomaly_threshold": threshold,
        "advice": advise(alert, str(room_df["room"].iloc[0]), language, llm),
    }


def analyse_flat(df, rooms=ROOMS, **kwargs):
    """Analyse every room in a flat. Returns one result per room."""
    results = []
    for room in rooms:
        room_df = df[df["room"] == room]
        if room_df.empty:
            continue
        results.append(analyse_room(room_df, **kwargs))
    order = {"urgent": 0, "action": 1, "watch": 2, "none": 3}
    results.sort(key=lambda r: (order[r["alert"]["level"]], -r["alert"]["confidence"]))
    return results


def raised_alerts(results):
    return [r for r in results if r["alert"]["level"] != "none"]
