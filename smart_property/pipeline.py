"""End-to-end pipeline: readings in, alerts and advice out."""

from pathlib import Path

from .advisor import advise
from .config import BASELINE_DAYS, ROOMS
from .detector import RoomAnomalyDetector
from .forecast import forecast_room
from .fusion import fuse
from .rules import classify_room


def inspect_photo(photo, defect_model=None, previous_photo=None,
                  segmenter=None):
    """The camera side of one room: what the wall looks like right now.

    `photo` is a BGR array or an image path. Returns None when neither the
    defect classifier nor an ArUco marker gives anything to say, so that a
    photograph nothing can be read from behaves exactly like no photograph.
    """
    import cv2
    from .crack import growth, measure

    image = cv2.imread(str(photo)) if isinstance(photo, (str, Path)) else photo
    if image is None:
        return None

    evidence = {}
    if defect_model is not None:
        if hasattr(defect_model, "predict_array"):
            prediction = defect_model.predict_array(image)
        elif isinstance(photo, (str, Path)):
            prediction = defect_model.predict(photo)
        else:
            prediction = None
        if prediction:
            evidence["damp_probability"] = prediction["damp_probability"]
            evidence["top_class"] = prediction["top_class"]

    # Detection and measurement are separate questions. Component 5 answers
    # "is there a crack" from the photograph alone; converting it to
    # millimetres additionally needs an ArUco marker in frame. A wall photo
    # with a crack and no marker is still a useful finding.
    crack_mask = None
    if segmenter is not None:
        found = segmenter.predict(image)
        evidence["crack_detected"] = found["instances"] > 0
        evidence["crack_confidence"] = found["max_confidence"]
        if found["instances"] == 0:
            return evidence or None          # nothing to measure, but we looked
        crack_mask = found["mask"]

    crack = measure(image, crack_mask)
    if "error" not in crack:
        evidence["crack_mm"] = crack["max_mm"]
        evidence["crack_width_trustworthy"] = crack["width_trustworthy"]
        if previous_photo is not None:
            earlier = previous_photo
            if isinstance(previous_photo, (str, Path)):
                earlier = cv2.imread(str(previous_photo))
            before = measure(earlier,
                             segmenter.predict(earlier)["mask"] if segmenter else None)
            if "error" not in before:
                change = growth(before, crack)
                if change:
                    evidence["crack_growth_mm"] = change["growth_mm"]
    return evidence or None


def analyse_room(room_df, baseline_days=BASELINE_DAYS, language="en",
                 backend="auto", llm=None, seed=0, sequence_model=None,
                 photo=None, previous_photo=None, defect_model=None,
                 segmenter=None):
    """Run every detector over one room and fuse them into a single alert.

    `sequence_model` is the optional layer-3 opinion (smart_property.sequence).
    It only ever adjusts confidence in an alert the physics has already raised;
    it cannot create one on its own, because it is trained on the simulator.
    """
    detector = RoomAnomalyDetector(baseline_days=baseline_days, seed=seed).fit(room_df)
    anomaly_fraction, threshold = detector.recent_anomaly_fraction(room_df)

    rule_result = classify_room(room_df, baseline_days=baseline_days)
    forecast = forecast_room(room_df, backend=backend)

    sequence = sequence_model.predict(room_df) if sequence_model else None
    learned_prob = None
    if sequence is not None:
        # agreement with the rule engine's pick, not just "something is wrong"
        learned_prob = sequence["by_label"].get(rule_result["best"]["kind"], 0.0)

    photo_evidence = None
    if photo is not None:
        photo_evidence = inspect_photo(photo, defect_model, previous_photo,
                                       segmenter)

    alert = fuse(rule_result, anomaly_fraction, forecast, learned_prob,
                 photo_evidence)

    return {
        "room": str(room_df["room"].iloc[0]),
        "alert": alert,
        "forecast": forecast,
        "measures": rule_result["measures"],
        "ranked": rule_result["ranked"],
        "anomaly_threshold": threshold,
        "sequence": sequence,
        "photo": photo_evidence,
        "advice": advise(alert, str(room_df["room"].iloc[0]), language, llm),
    }


def analyse_flat(df, rooms=ROOMS, photos=None, previous_photos=None, **kwargs):
    """Analyse every room in a flat. Returns one result per room.

    `photos` maps room name to an image or path, because a photograph belongs
    to the one wall it was taken of -- passing a single photo through to every
    room would corroborate alerts in rooms the camera never saw.
    """
    photos = photos or {}
    previous_photos = previous_photos or {}
    results = []
    for room in rooms:
        room_df = df[df["room"] == room]
        if room_df.empty:
            continue
        results.append(analyse_room(room_df, photo=photos.get(room),
                                    previous_photo=previous_photos.get(room),
                                    **kwargs))
    order = {"urgent": 0, "action": 1, "watch": 2, "none": 3}
    results.sort(key=lambda r: (order[r["alert"]["level"]], -r["alert"]["confidence"]))
    return results


def raised_alerts(results):
    return [r for r in results if r["alert"]["level"] != "none"]
