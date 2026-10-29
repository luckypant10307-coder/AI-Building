"""Alert fusion (no model).

Combines the rule confidence, the Isolation Forest score and -- when one is
trained -- a learned classifier's probability into a single alert level per
room. Urgency comes from the forecast, not from the detector: a problem that
becomes visible in two days outranks one that takes a month.

High-urgency alerts are marked as needing technician confirmation. Each verdict
a technician returns is a real label, which is how this gets real training data.
"""

LEVELS = ["none", "watch", "action", "urgent"]

# A room has to clear both the rule bar and show some anomaly before it alarms,
# unless the physical evidence on its own is overwhelming.
MIN_CONFIDENCE = 0.60
MIN_ANOMALY_FRACTION = 0.05
STANDALONE_CONFIDENCE = 0.80


def fuse(rule_result, anomaly_fraction, forecast, learned_prob=None):
    """Turn the detectors' separate opinions into one alert."""
    best = rule_result["best"]
    confidence = best["confidence"]

    if learned_prob is not None:
        # a second learned opinion nudges confidence but never creates an alert
        confidence = 0.75 * confidence + 0.25 * float(learned_prob)

    raised = confidence >= MIN_CONFIDENCE and (
        anomaly_fraction >= MIN_ANOMALY_FRACTION or confidence >= STANDALONE_CONFIDENCE)

    if not raised:
        return {
            "level": "none",
            "kind": None,
            "confidence": confidence,
            "anomaly_fraction": anomaly_fraction,
            "needs_technician_confirmation": False,
            "days_to_damage": None,
        }

    days = _soonest(forecast)
    if days is not None and days <= 3:
        level = "urgent"
    elif days is not None and days <= 10:
        level = "action"
    else:
        level = "watch"

    return {
        "level": level,
        "kind": best["kind"],
        "confidence": confidence,
        "anomaly_fraction": anomaly_fraction,
        "needs_technician_confirmation": level == "urgent"
        or (level == "action" and confidence < STANDALONE_CONFIDENCE),
        "days_to_damage": days,
        "evidence_for": best["evidence_for"],
        "evidence_against": best["evidence_against"],
    }


def _soonest(forecast):
    """Whichever kind of damage arrives first, in days."""
    candidates = [
        forecast[name]["days"]
        for name in ("visible_damp", "mould_risk")
        if forecast.get(name, {}).get("days") is not None
    ]
    return min(candidates) if candidates else None
