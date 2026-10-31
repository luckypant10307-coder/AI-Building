"""Flask backend for Smart Property AI, serving the React UI in web/dist.

    python app.py                 then open http://127.0.0.1:5000

For UI work, run Vite's dev server alongside it:

    cd web && npm run dev         http://127.0.0.1:5173, /api proxied here

Endpoints
    GET    /                 the React app (built)
    GET    /api/components   which of the ten components are live
    POST   /api/demo         load a simulated flat
    POST   /api/ingest       append real readings
    GET    /api/alerts       analyse what is loaded
    POST   /api/photo        attach a wall photo to one room (multipart)
    DELETE /api/photo        remove a room's photo
"""

import warnings

import numpy as np
import pandas as pd
from flask import Flask, jsonify, request, send_from_directory

warnings.filterwarnings("ignore")

from smart_property import registry
from smart_property.config import FAULT_KINDS, ROOMS, SENSORS
from smart_property.pipeline import analyse_flat
from smart_property.simulator import simulate

DIST = "web/dist"
app = Flask(__name__, static_folder=DIST, static_url_path="")

# A prototype keeps its state in memory. A deployment would use a time-series
# store for readings and object storage for photographs.
STORE = {"readings": None, "photos": {}, "previous": {}}

REQUIRED = {"ts", "room", *SENSORS}


# ----------------------------------------------------------------- readings

@app.post("/api/ingest")
def ingest():
    payload = request.get_json(silent=True) or {}
    rows = payload.get("readings")
    if not rows:
        return jsonify({"error": "send {'readings': [...]}"}), 400
    frame = pd.DataFrame(rows)
    missing = REQUIRED - set(frame.columns)
    if missing:
        return jsonify({"error": f"missing columns: {sorted(missing)}"}), 400
    frame["ts"] = pd.to_datetime(frame["ts"])
    if "rain_mm_h" not in frame:
        frame["rain_mm_h"] = 0.0
    current = STORE["readings"]
    STORE["readings"] = frame if current is None else pd.concat(
        [current, frame], ignore_index=True)
    return jsonify({"ok": True, "rows": len(STORE["readings"])})


@app.post("/api/demo")
def demo():
    payload = request.get_json(silent=True) or {}
    fault = payload.get("fault") or None
    if fault and fault not in FAULT_KINDS:
        return jsonify({"error": f"fault must be one of {FAULT_KINDS}"}), 400
    STORE["readings"] = simulate(
        seed=int(payload.get("seed", 3)),
        days=int(payload.get("days", 18)),
        fault_kind=fault,
        room=payload.get("room", "bathroom") if fault else None,
        start_day=float(payload.get("start_day", 9.0)),
    )
    return jsonify({"ok": True, "fault": fault, "room": payload.get("room")})


# -------------------------------------------------------------------- photos

@app.post("/api/photo")
def add_photo():
    import cv2

    room = request.form.get("room")
    if room not in ROOMS:
        return jsonify({"error": f"room must be one of {ROOMS}"}), 400
    upload = request.files.get("photo")
    if upload is None:
        return jsonify({"error": "attach a file under the 'photo' field"}), 400

    data = np.frombuffer(upload.read(), np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        return jsonify({"error": "could not decode that image"}), 400

    # replacing a photo keeps the old one, so crack growth between two visits
    # can be measured without the user having to manage versions
    if room in STORE["photos"]:
        STORE["previous"][room] = STORE["photos"][room]
    STORE["photos"][room] = image
    return jsonify({"ok": True, "room": room,
                    "shape": list(image.shape),
                    "has_previous": room in STORE["previous"]})


@app.delete("/api/photo")
def drop_photo():
    payload = request.get_json(silent=True) or {}
    room = payload.get("room")
    STORE["photos"].pop(room, None)
    STORE["previous"].pop(room, None)
    return jsonify({"ok": True, "room": room})


# ------------------------------------------------------------------ analysis

@app.get("/api/components")
def components():
    return jsonify({"components": registry.describe()})


@app.get("/api/alerts")
def alerts():
    df = STORE["readings"]
    if df is None:
        return jsonify({"error": "no readings loaded"}), 404
    language = request.args.get("language", "en")
    results = analyse_flat(
        df, language=language,
        photos=STORE["photos"] or None,
        previous_photos=STORE["previous"] or None,
        **registry.components(language))
    return jsonify({
        "rooms": results,
        "raised": sum(r["alert"]["level"] != "none" for r in results),
        "simulated_sensors": True,
    })


# ----------------------------------------------------------------------- UI

@app.get("/")
def index():
    try:
        return send_from_directory(DIST, "index.html")
    except Exception:
        return ("<h1>UI not built</h1><p>Run <code>cd web &amp;&amp; npm install "
                "&amp;&amp; npm run build</code>, then reload.</p>"), 200


if __name__ == "__main__":
    app.run(debug=False, port=5000)
