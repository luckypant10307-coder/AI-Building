"""Backend for the Smart Property AI dashboard.

Serves the contract the published UI expects, so the page runs against the real
engine -- Isolation Forest, the physics rules, the sequence CNN, the forecaster,
the BD3 damp classifier and YOLO crack segmentation -- instead of the z-score
port it falls back to in the browser.

    python server.py            then open http://127.0.0.1:5000

The UI fetches `api/config` with a RELATIVE path, so it has to be served from
this origin. Put the dashboard's index.html at web/artifact/index.html.

Endpoints, exactly as the page calls them:
    GET  api/config
    GET  api/state                       (also polled as the clock tick)
    GET  api/history?room=&hours=
    POST api/inspections
    POST api/simulate/{fault,clear,speed,pause}

Why there are two loops: the page ticks once a second, and a full analysis --
fitting an Isolation Forest per room, forecasting two series per room -- takes
seconds. So the clock thread only advances the flat and stores readings, while
a slower analysis thread recomputes findings and caches them. `api/state`
merges cached findings with the newest readings, so the numbers on screen are
always live even when the verdict behind them is a few seconds old.
"""

import threading
import time
import warnings
from datetime import timedelta

import numpy as np
import pandas as pd
from flask import Flask, jsonify, request, send_from_directory

warnings.filterwarnings("ignore")

from smart_property.env import load_dotenv

load_dotenv()          # before anything reads GEMINI_API_KEY or TELEGRAM_*

from smart_property import flat_model as fm
from smart_property import registry
from smart_property.advisor import advise
from smart_property.notify import Notifier
from smart_property.config import (MOULD_RISK_SURFACE_RH, SAMPLE_MINUTES,
                                   VISIBLE_DAMP_MOISTURE)
from smart_property.physics import dew_point
from smart_property.pipeline import analyse_flat
from smart_property.simulator import FlatSimulator, Fault

UI_DIR = "web/artifact"
WARMUP_DAYS = 9          # a normal week plus a little, so a baseline exists
HORIZON_DAYS = 40        # how much future the simulator pre-computes
ANALYSIS_EVERY = 6.0     # seconds between full re-analyses
CONTINUOUS_HOURS = 4     # flow that never stops this long means a leak
SIM_START = "2026-06-01"

app = Flask(__name__, static_folder=None)


class Engine:
    """The running flat: a clock, a reading store and a cached analysis."""

    def __init__(self):
        self.lock = threading.Lock()
        self.speed = 4               # simulator steps per tick
        self.paused = False
        self.injected = []           # [{type, room, since}]
        self.inspections = []
        self.analysis = {}
        self.analysed_at = 0.0
        self.analysis_seconds = None
        # one simulation pass at boot: the seeded leak is decided before the
        # flat is generated, rather than generating it twice
        self.cursor = WARMUP_DAYS * 24 * 60 // SAMPLE_MINUTES
        self._seed_scenario()

    def _seed_scenario(self, days_ago=2.5):
        """Open on a leak that has already been running.

        A fault injected at the current instant is correctly invisible: the
        rules wait for the plaster to be measurably wetter than its baseline,
        which takes days. Booting with a fresh fault would show an empty
        dashboard and look broken. So the demo starts with a bathroom leak
        that began two and a half days ago, which is long enough for the
        sensors to have something to say and short enough that nothing is
        visible on the wall yet -- the exact window this product is about.
        """
        start_day = max(0.5, self.cursor * SAMPLE_MINUTES / (60 * 24) - days_ago)
        began = pd.Timestamp(SIM_START) + pd.Timedelta(days=start_day)
        self.injected = [{"type": "pipe_leak", "room": "bathroom",
                          "since": pd.Timestamp(began).isoformat(),
                          "start_day": start_day}]
        self._rebuild()

    # ------------------------------------------------------------- data

    def _rebuild(self):
        """Regenerate the whole run. Called at boot and whenever a fault changes.

        The simulator produces a complete history in one pass, so changing the
        injected fault means recomputing from the start rather than splicing.
        The cursor keeps the clock where it was.
        """
        fault = Fault()
        if self.injected:
            first = self.injected[0]
            fault = Fault(kind=fm.TO_ENGINE[first["type"]], room=first["room"],
                          start_day=first["start_day"])
        self.sim = FlatSimulator(seed=20261021, days=HORIZON_DAYS, fault=fault,
                                 start=SIM_START)
        self.frame = self.sim.run()
        self.frame["ts"] = pd.to_datetime(self.frame["ts"])
        self.stamps = np.sort(self.frame["ts"].unique())
        if not hasattr(self, "cursor"):
            self.cursor = WARMUP_DAYS * 24 * 60 // SAMPLE_MINUTES
        self.cursor = min(self.cursor, len(self.stamps) - 1)

    def advance(self, steps):
        self.cursor = min(self.cursor + steps, len(self.stamps) - 1)

    @property
    def now(self):
        return pd.Timestamp(self.stamps[self.cursor])

    def upto(self):
        return self.frame[self.frame["ts"] <= self.now]

    # --------------------------------------------------------- analysis

    def reanalyse(self):
        frame = self.upto()
        try:
            results = analyse_flat(frame, rooms=fm.ROOMS, backend="trend")
        except Exception:
            return
        with self.lock:
            self.analysis = {r["room"]: r for r in results}
            self.analysed_at = time.time()


ENGINE = Engine()
NOTIFIER = Notifier()


# ------------------------------------------------------------- state shaping

def _latest(room_frame):
    row = room_frame.iloc[-1]
    return {
        "moisture": round(float(row.wall_moisture), 2),
        "surface_rh": round(float(row.wall_surface_rh), 1),
        "wall_temp": round(float(row.wall_temp), 2),
        "temp": round(float(row.air_temp), 2),
        "rh": round(float(row.air_rh), 1),
        "dew_point": round(float(dew_point(row.air_temp, row.air_rh)), 2),
        "acoustic": (round(float(row.pipe_db), 1)
                     if fm.ACOUSTIC[room_frame.room.iloc[0]] else None),
    }


def _finding(room, result, now):
    alert = result["alert"]
    if alert["level"] == "none":
        return None
    ui_kind = fm.TO_UI.get(alert["kind"], alert["kind"])
    level = "alert" if alert["level"] in ("urgent", "action") else "watch"
    damp = result["forecast"].get("visible_damp", {})
    mould = result["forecast"].get("mould_risk", {})
    forecast = None
    if damp.get("days") is not None:
        forecast = (f"Visible damp in about {damp['days']:.1f} days "
                    f"at {VISIBLE_DAMP_MOISTURE:.0f}% moisture.")
    elif mould.get("days") is not None:
        forecast = (f"Mould risk in about {mould['days']:.1f} days "
                    f"at {MOULD_RISK_SURFACE_RH:.0f}% surface humidity.")
    advice = result.get("advice") or {}
    since = now - timedelta(hours=6)
    for inj in ENGINE.injected:
        if inj["room"] == room and fm.TO_ENGINE[inj["type"]] == alert["kind"]:
            since = pd.Timestamp(inj["since"])
    return {
        "type": ui_kind,
        "title": fm.FAULTS.get(ui_kind, {}).get("label", ui_kind.replace("_", " ")),
        "level": level,
        "room": room,
        "room_name": fm.ROOM_NAME[room],
        "confidence": round(float(alert["confidence"]), 3),
        "first_seen": since.isoformat(),
        "evidence": alert.get("evidence_for", []),
        "forecast": forecast,
        "advice": {
            "urgency": {"urgent": "high", "action": "medium"}.get(
                alert["level"], "low"),
            "cause": advice.get("cause", ""),
            "action": advice.get("action", ""),
            "who": advice.get("who_to_call", ""),
            "ignore": forecast or "Damage develops slowly from here.",
        },
    }


def build_state():
    engine = ENGINE
    frame = engine.upto()
    now = engine.now
    with engine.lock:
        analysis = dict(engine.analysis)

    rooms, alerts = [], []
    for rid in fm.ROOMS:
        room_frame = frame[frame["room"] == rid]
        if room_frame.empty:
            continue
        result = analysis.get(rid)
        findings = []
        anomaly = 0.0
        m_z = 0.0
        if result:
            got = _finding(rid, result, now)
            if got:
                findings.append(got)
                alerts.append(got)
            anomaly = float(result["alert"].get("anomaly_fraction") or 0.0)
            measures = result.get("measures") or {}
            m_z = float(measures.get("moisture_excess") or 0.0)
        rooms.append({
            "id": rid, "name": fm.ROOM_NAME[rid],
            "latest": _latest(room_frame),
            "anomaly_score": round(anomaly, 3),
            "features": {"m_z": round(m_z, 2)},
            "findings": findings,
        })

    day = frame[frame["ts"] > now - pd.Timedelta(hours=24)]
    flow_series = day.groupby("ts")["flow_lpm"].sum()
    night = flow_series[flow_series.index.hour.isin([1, 2, 3, 4])]
    continuous = bool(len(night) and float(night.min()) > 0.04)
    litres = float(flow_series.sum()) * SAMPLE_MINUTES
    worst = max((a["confidence"] for a in alerts), default=0.0)
    penalty = sum(28 if a["level"] == "alert" else 10 for a in alerts)
    health = int(max(5, min(100, round(100 - penalty - 10 * worst))))

    last = frame[frame["ts"] == now]
    rain_now = float(last["rain_mm_h"].iloc[0]) if len(last) else 0.0
    ahead = engine.frame[(engine.frame["ts"] > now)
                         & (engine.frame["ts"] <= now + pd.Timedelta(hours=72))]
    rain_72 = float(ahead.groupby("ts")["rain_mm_h"].first().sum()) \
        * (SAMPLE_MINUTES / 60.0)
    outdoor = last["air_temp"].mean() + 2.2 if len(last) else 29.0

    return {
        # Ready as soon as there are readings, not once analysis finishes. The
        # page draws nothing while ready is false, and on a 0.1-CPU instance
        # the first analysis takes two minutes: the clock would be running
        # underneath a frozen screen. Findings simply arrive when they arrive.
        "ready": True,
        "analysing": not analysis,
        "mode": "simulation",
        "sim_time": now.isoformat(),
        "speed": engine.speed,
        "paused": engine.paused,
        "flat": {
            "health": health,
            "flow_now": round(float(flow_series.iloc[-1]) if len(flow_series) else 0.0, 3),
            "used_last_24h_l": int(litres),
            "continuous_flow": continuous,
            "water_loss_l_day": int(float(night.min()) * 1440) if continuous else 0,
        },
        "weather": {
            "temp": round(float(outdoor), 1),
            "rh": round(float(last["air_rh"].mean()) if len(last) else 55.0, 1),
            "rain_now": round(rain_now, 1),
            "rain_forecast_72h_mm": round(rain_72, 1),
        },
        "alerts": alerts,
        "rooms": rooms,
        "walls": build_walls(rooms, now),
        "injected": [
            {"type": i["type"], "label": fm.FAULTS[i["type"]]["label"],
             "room": i["room"], "room_name": fm.ROOM_NAME[i["room"]],
             "since": i["since"]}
            for i in engine.injected
        ],
    }


def build_walls(rooms, now):
    """Camera findings pinned to walls, cross-checked against the sensors."""
    inspections = []
    for item in ENGINE.inspections:
        wall = fm.WALL_BY_ID.get(item["wall"])
        if not wall:
            continue
        age = max(0.0, (now - pd.Timestamp(item["history"][-1]["date"])).days)
        room = next((r for r in rooms if r["id"] == wall["room"]), None)
        wet = [f for f in (room["findings"] if room else [])
               if f["type"] in fm.WET_FAULTS]
        widths = [h["width_mm"] for h in item["history"] if h.get("width_mm")]
        if item["type"] == "crack" and len(widths) > 1:
            grew = widths[-1] - widths[0]
            verdict = ({"code": "growing", "level": "alert",
                        "text": f"Widened {grew:.1f} mm since the first photo."}
                       if grew > 0.2 else
                       {"code": "stable", "level": "ok",
                        "text": f"No change ({widths[-1]:.1f} mm). Recheck in 3 months."})
        elif not wall["probe"]:
            verdict = {"code": "no_sensor", "level": "watch",
                       "text": "No sensor on this wall. Recheck by photo."}
        elif wet:
            verdict = {"code": "confirmed", "level": "alert",
                       "text": f"Sensors and camera agree: "
                               f"{wet[0]['title'].lower()} on this wall."}
        elif room and room["features"]["m_z"] < 2:
            verdict = {"code": "old_stain", "level": "watch",
                       "text": "Sensors read dry here, so this is likely an old "
                               "stain. Fix the cause, then repaint."}
        else:
            verdict = {"code": "wet", "level": "watch",
                       "text": "Wall moisture is above normal. Keep watching."}
        inspections.append({**item, "room": wall["room"],
                            "marker_id": wall["marker_id"],
                            "age_days": age, "verdict": verdict})

    links = {}
    for room in rooms:
        probe = fm.PROBES.get(room["id"])
        if not probe:
            continue
        wall = fm.WALL_BY_ID[probe["wall"]]
        for finding in room["findings"]:
            if finding["type"] not in fm.WET_FAULTS:
                continue
            seen = [i for i in inspections
                    if i["wall"] == wall["id"] and i["type"] != "crack"]
            links[f"{room['id']}:{finding['type']}"] = (
                {"code": "confirmed", "wall": wall["id"],
                 "marker_id": wall["marker_id"],
                 "text": f"Confirmed by camera on "
                         f"{seen[-1]['history'][-1]['date']}."}
                if seen else
                {"code": "photograph", "wall": wall["id"],
                 "marker_id": wall["marker_id"],
                 "text": f"Next: photograph the {wall['side']} wall "
                         f"(marker #{wall['marker_id']}) to confirm."})
    return {"inspections": inspections, "sensor_links": links}


# ------------------------------------------------------------------ routes

@app.get("/api/config")
def api_config():
    return jsonify(fm.config())


@app.get("/api/state")
def api_state():
    return jsonify(build_state())


@app.get("/api/history")
def api_history():
    room = request.args.get("room", fm.ROOMS[0])
    hours = int(request.args.get("hours", 72))
    frame = ENGINE.upto()
    frame = frame[(frame["room"] == room)
                  & (frame["ts"] > ENGINE.now - pd.Timedelta(hours=hours))]
    if frame.empty:
        return jsonify({"ts": [], "moisture": [], "surface_rh": [],
                        "flow": [], "rain": []})
    return jsonify({
        "ts": [t.isoformat() for t in frame["ts"]],
        "moisture": [round(v, 2) for v in frame["wall_moisture"]],
        "surface_rh": [round(v, 1) for v in frame["wall_surface_rh"]],
        "flow": [round(v, 3) for v in frame["flow_lpm"]],
        "rain": [round(v, 2) for v in frame["rain_mm_h"]],
        "acoustic": ([round(v, 1) for v in frame["pipe_db"]]
                     if fm.ACOUSTIC[room] else None),
    })


@app.post("/api/inspections")
def api_inspections():
    item = request.get_json(silent=True) or {}
    if item.get("wall") not in fm.WALL_BY_ID:
        return jsonify({"ok": False, "error": "unknown wall"}), 400
    date = item.get("date") or ENGINE.now.date().isoformat()
    entry = {"date": date}
    if item.get("width_mm") not in (None, ""):
        entry["width_mm"] = float(item["width_mm"])
    else:
        entry["area_cm2"] = float(item.get("area_cm2") or 200)

    same = next((i for i in ENGINE.inspections
                 if i["wall"] == item["wall"] and i["type"] == item.get("type")
                 and abs(i["x_m"] - float(item.get("x_m", 0))) < 0.25
                 and abs(i["y_m"] - float(item.get("y_m", 0))) < 0.25), None)
    if same:
        # the same defect photographed again: a second point in its history,
        # which is what makes growth measurable
        same["history"].append(entry)
    else:
        ENGINE.inspections.append({
            "id": f"ins{len(ENGINE.inspections) + 1}",
            "wall": item["wall"], "type": item.get("type", "stain"),
            "x_m": float(item.get("x_m", 0.5)), "y_m": float(item.get("y_m", 0.5)),
            "length_m": float(item.get("length_m") or 0.4),
            "history": [entry],
        })
    return jsonify({"ok": True})


@app.post("/api/simulate/fault")
def api_fault():
    body = request.get_json(silent=True) or {}
    kind, room = body.get("type"), body.get("room")
    if kind not in fm.FAULTS or room not in fm.ROOMS:
        return jsonify({"ok": False, "error": "unknown fault or room"}), 400
    start_day = ENGINE.cursor * SAMPLE_MINUTES / (60 * 24)
    ENGINE.injected = [{"type": kind, "room": room,
                        "since": ENGINE.now.isoformat(),
                        "start_day": start_day}]
    ENGINE._rebuild()
    ENGINE.reanalyse()
    return jsonify({"ok": True})


@app.post("/api/simulate/clear")
def api_clear():
    ENGINE.injected = []
    ENGINE._rebuild()
    ENGINE.reanalyse()
    return jsonify({"ok": True})


@app.post("/api/simulate/speed")
def api_speed():
    body = request.get_json(silent=True) or {}
    ENGINE.speed = max(1, min(32, int(body.get("steps_per_tick", 4))))
    return jsonify({"ok": True, "speed": ENGINE.speed})


@app.post("/api/simulate/pause")
def api_pause():
    body = request.get_json(silent=True) or {}
    ENGINE.paused = bool(body.get("paused"))
    return jsonify({"ok": True, "paused": ENGINE.paused})


@app.get("/api/notifications")
def api_notifications():
    return jsonify(NOTIFIER.status())


@app.post("/api/notifications/test")
def api_notifications_test():
    NOTIFIER.test()
    return jsonify({"ok": True, "channel": NOTIFIER.channel.name})


@app.get("/")
def index():
    try:
        return send_from_directory(UI_DIR, "index.html")
    except Exception:
        return (f"<h1>Dashboard not installed</h1><p>Save the published "
                f"dashboard's HTML to <code>{UI_DIR}/index.html</code> and "
                f"reload. The API is already running: "
                f"<a href='/api/config'>/api/config</a></p>"), 200


@app.get("/<path:asset>")
def static_asset(asset):
    return send_from_directory(UI_DIR, asset)


# ------------------------------------------------------------------- loops

def clock():
    while True:
        time.sleep(1.0)
        if not ENGINE.paused:
            ENGINE.advance(ENGINE.speed)


def notify_changes():
    """Hand each room's current finding to the notifier.

    The notifier decides whether anything changed; this just reports what the
    detectors currently say. Advice in another language is produced only when
    a message is actually going out, so Gemini is not called every cycle.
    """
    with ENGINE.lock:
        analysis = dict(ENGINE.analysis)
    now = ENGINE.now
    for rid in fm.ROOMS:
        result = analysis.get(rid)
        if result is None:
            continue
        finding = _finding(rid, result, now)

        def advice_for(language, alert=result["alert"], room=rid):
            text = advise(alert, room, language, registry.advisor(language))
            return text and {"cause": text["cause"], "action": text["action"],
                             "who": text["who_to_call"]}

        NOTIFIER.observe(rid, fm.ROOM_NAME[rid], finding, advice_for)
    NOTIFIER.release_held()


def analyst():
    """Re-run the detectors, pacing to the machine.

    A fixed 6-second interval is fine on a laptop. On a 0.1-CPU instance one
    analysis takes far longer than 6 seconds, so the thread never rests and
    every page request queues behind it. Resting at least as long as the last
    analysis took keeps half the CPU free for serving the dashboard.
    """
    while True:
        started = time.time()
        ENGINE.reanalyse()
        notify_changes()
        took = time.time() - started
        ENGINE.analysis_seconds = round(took, 1)
        time.sleep(max(ANALYSIS_EVERY, took))


_started = threading.Lock()
_running = False


def start_background():
    """Start the clock and the analyst, once per process.

    Called at import, not from __main__, because gunicorn imports this module
    rather than executing it: under `gunicorn server:app` the __main__ block
    never runs, so threads started there would never start at all and the
    dashboard would load with a clock frozen at boot.

    Safe to call twice; the flag makes the second call a no-op.
    """
    global _running
    with _started:
        if _running:
            return
        _running = True
        threading.Thread(target=clock, daemon=True, name="clock").start()
        threading.Thread(target=analyst, daemon=True, name="analyst").start()


start_background()


if __name__ == "__main__":
    import os
    app.run(debug=False, host="0.0.0.0",
            port=int(os.environ.get("PORT", 5000)), threaded=True)
