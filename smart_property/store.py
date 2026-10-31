"""Durable store for a flat that is actually running.

Until now everything lived in a process dict and vanished on restart, which is
fine for a demo and useless for a building. This keeps readings, the frozen
baseline, open alerts and technician verdicts in SQLite.

Two decisions matter more than the schema.

**The baseline is frozen.** `RoomAnomalyDetector` and the rule engine both take
"the first `baseline_days` of the frame you hand me" as that room's normal
week. On a fixed 18-day window that is right. On a rolling window it is a bug:
the baseline slides forward with the window, so a leak that has been running
for a month becomes the new normal and the alert quietly switches off. So the
first week of readings a room ever produces is kept permanently, and every
analysis frame is that baseline plus the recent window -- never just the recent
window.

**Verdicts are kept.** Alert fusion marks high-urgency alerts as needing a
technician. Those verdicts are the only real labels this system will ever
generate, and component 10 cannot be trained without them, so they are written
down from the first install rather than collected later.
"""

import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path

import pandas as pd

from .config import BASELINE_DAYS, SENSORS

DEFAULT_PATH = Path("data/flat.db")
COLUMNS = ["ts", "room", *SENSORS, "rain_mm_h"]

SCHEMA = """
CREATE TABLE IF NOT EXISTS readings (
    ts TEXT NOT NULL, room TEXT NOT NULL,
    wall_moisture REAL, wall_surface_rh REAL, air_temp REAL, air_rh REAL,
    wall_temp REAL, flow_lpm REAL, pipe_db REAL, rain_mm_h REAL,
    PRIMARY KEY (room, ts)
);
CREATE INDEX IF NOT EXISTS readings_room_ts ON readings (room, ts);

-- One row per room, written once, never updated: the frozen normal week.
CREATE TABLE IF NOT EXISTS baseline (
    room TEXT PRIMARY KEY, frozen_at TEXT NOT NULL,
    first_ts TEXT NOT NULL, last_ts TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS alerts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    room TEXT NOT NULL, kind TEXT NOT NULL, state TEXT NOT NULL,
    level TEXT NOT NULL, confidence REAL, days_to_damage REAL,
    opened_at TEXT NOT NULL, last_seen_at TEXT NOT NULL, closed_at TEXT,
    clear_ticks INTEGER NOT NULL DEFAULT 0, evidence TEXT
);
CREATE INDEX IF NOT EXISTS alerts_open ON alerts (room, state);

CREATE TABLE IF NOT EXISTS verdicts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    alert_id INTEGER NOT NULL, at TEXT NOT NULL,
    verdict TEXT NOT NULL, note TEXT,
    FOREIGN KEY (alert_id) REFERENCES alerts (id)
);
"""


class FlatStore:
    def __init__(self, path=DEFAULT_PATH):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as db:
            db.executescript(SCHEMA)

    @contextmanager
    def connect(self):
        db = sqlite3.connect(self.path, timeout=30)
        db.row_factory = sqlite3.Row
        try:
            yield db
            db.commit()
        finally:
            db.close()

    # ---------------------------------------------------------- readings

    def append(self, frame):
        """Insert readings, ignoring ones already stored (replays are safe)."""
        frame = frame.copy()
        if "rain_mm_h" not in frame:
            frame["rain_mm_h"] = 0.0
        frame["ts"] = pd.to_datetime(frame["ts"]).astype(str)
        rows = frame[COLUMNS].itertuples(index=False, name=None)
        with self.connect() as db:
            before = db.execute("SELECT COUNT(*) FROM readings").fetchone()[0]
            db.executemany(
                f"INSERT OR IGNORE INTO readings ({','.join(COLUMNS)}) "
                f"VALUES ({','.join('?' * len(COLUMNS))})", rows)
            after = db.execute("SELECT COUNT(*) FROM readings").fetchone()[0]
        self.freeze_baselines()
        return after - before

    def rooms(self):
        with self.connect() as db:
            return [r[0] for r in db.execute(
                "SELECT DISTINCT room FROM readings ORDER BY room")]

    def span(self, room):
        with self.connect() as db:
            row = db.execute(
                "SELECT MIN(ts) a, MAX(ts) b FROM readings WHERE room = ?",
                (room,)).fetchone()
        if not row or row["a"] is None:
            return None, None
        return pd.Timestamp(row["a"]), pd.Timestamp(row["b"])

    # ---------------------------------------------------------- baseline

    def freeze_baselines(self, baseline_days=BASELINE_DAYS):
        """Record each room's normal week once it has one. Never rewritten."""
        with self.connect() as db:
            known = {r[0] for r in db.execute("SELECT room FROM baseline")}
            for room in [r[0] for r in db.execute(
                    "SELECT DISTINCT room FROM readings")]:
                if room in known:
                    continue
                row = db.execute(
                    "SELECT MIN(ts) a, MAX(ts) b FROM readings WHERE room = ?",
                    (room,)).fetchone()
                first, last = pd.Timestamp(row["a"]), pd.Timestamp(row["b"])
                if last - first < pd.Timedelta(days=baseline_days):
                    continue                 # not a full week yet
                end = first + pd.Timedelta(days=baseline_days)
                db.execute(
                    "INSERT INTO baseline (room, frozen_at, first_ts, last_ts) "
                    "VALUES (?,?,?,?)",
                    (room, str(pd.Timestamp.utcnow()), str(first), str(end)))

    def has_baseline(self, room):
        with self.connect() as db:
            return db.execute("SELECT 1 FROM baseline WHERE room = ?",
                              (room,)).fetchone() is not None

    # ------------------------------------------------------------ window

    def analysis_frame(self, room, recent_days=21, baseline_days=BASELINE_DAYS):
        """The frozen baseline week plus the recent window, as one frame.

        Returned together because every detector downstream derives "normal"
        from the earliest rows it is given. Handing it only recent data would
        make a month-old leak look like this room's normal condition.
        """
        with self.connect() as db:
            base = db.execute(
                "SELECT first_ts, last_ts FROM baseline WHERE room = ?",
                (room,)).fetchone()
            last = db.execute(
                "SELECT MAX(ts) t FROM readings WHERE room = ?", (room,)
            ).fetchone()["t"]
            if last is None:
                return None
            cutoff = pd.Timestamp(last) - pd.Timedelta(days=recent_days)

            if base is None:
                # no frozen week yet: everything there is, which is what the
                # fixed-window code already assumed
                query, args = ("SELECT * FROM readings WHERE room = ? "
                               "ORDER BY ts", (room,))
            else:
                query = ("SELECT * FROM readings WHERE room = ? AND "
                         "((ts >= ? AND ts < ?) OR ts >= ?) ORDER BY ts")
                args = (room, base["first_ts"], base["last_ts"], str(cutoff))
            frame = pd.DataFrame(
                [dict(r) for r in db.execute(query, args)])
        if frame.empty:
            return None
        frame["ts"] = pd.to_datetime(frame["ts"])
        return frame

    # ------------------------------------------------------------ alerts

    def open_alert(self, room):
        with self.connect() as db:
            row = db.execute(
                "SELECT * FROM alerts WHERE room = ? AND state = 'open' "
                "ORDER BY id DESC LIMIT 1", (room,)).fetchone()
        return dict(row) if row else None

    def record(self, room, alert, now):
        """Reconcile one room's alert against what is already open."""
        existing = self.open_alert(room)
        raised = alert["level"] != "none"
        with self.connect() as db:
            if raised and existing and existing["kind"] == alert["kind"]:
                db.execute(
                    "UPDATE alerts SET level=?, confidence=?, days_to_damage=?,"
                    " last_seen_at=?, clear_ticks=0, evidence=? WHERE id=?",
                    (alert["level"], alert["confidence"],
                     alert.get("days_to_damage"), now,
                     json.dumps(alert.get("evidence_for", [])), existing["id"]))
                return "ongoing", existing["id"]
            if raised and existing:
                db.execute("UPDATE alerts SET state='superseded', closed_at=? "
                           "WHERE id=?", (now, existing["id"]))
            if raised:
                cur = db.execute(
                    "INSERT INTO alerts (room, kind, state, level, confidence,"
                    " days_to_damage, opened_at, last_seen_at, evidence)"
                    " VALUES (?,?,'open',?,?,?,?,?,?)",
                    (room, alert["kind"], alert["level"], alert["confidence"],
                     alert.get("days_to_damage"), now, now,
                     json.dumps(alert.get("evidence_for", []))))
                return "new", cur.lastrowid
            if existing:
                ticks = existing["clear_ticks"] + 1
                if ticks >= CLEAR_TICKS_TO_CLOSE:
                    db.execute("UPDATE alerts SET state='resolved', closed_at=?"
                               " WHERE id=?", (now, existing["id"]))
                    return "resolved", existing["id"]
                db.execute("UPDATE alerts SET clear_ticks=? WHERE id=?",
                           (ticks, existing["id"]))
                return "clearing", existing["id"]
        return "quiet", None

    def alerts(self, state=None, limit=100):
        sql = "SELECT * FROM alerts"
        args = ()
        if state:
            sql += " WHERE state = ?"
            args = (state,)
        sql += " ORDER BY id DESC LIMIT ?"
        with self.connect() as db:
            return [dict(r) for r in db.execute(sql, (*args, limit))]

    # ----------------------------------------------------------- verdicts

    def add_verdict(self, alert_id, verdict, note=None):
        """A technician's answer. These are the only real labels this produces."""
        with self.connect() as db:
            db.execute(
                "INSERT INTO verdicts (alert_id, at, verdict, note) "
                "VALUES (?,?,?,?)",
                (alert_id, str(pd.Timestamp.utcnow()), verdict, note))

    def verdicts(self, limit=200):
        with self.connect() as db:
            return [dict(r) for r in db.execute(
                "SELECT v.*, a.room, a.kind FROM verdicts v "
                "JOIN alerts a ON a.id = v.alert_id "
                "ORDER BY v.id DESC LIMIT ?", (limit,))]


# A fault that straddles a threshold would otherwise open and close an alert
# every cycle. Three consecutive quiet ticks before closing.
CLEAR_TICKS_TO_CLOSE = 3
