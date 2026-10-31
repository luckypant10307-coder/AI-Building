"""The flat as the dashboard understands it: rooms, walls, markers, probes.

This is the model the published UI expects from `GET api/config`, so the names
and ids here are the contract, not a preference. Two things differ from the
rest of this package and are mapped rather than renamed:

  * the UI uses `pipe_leak` and `fixture_leak`; the engine calls the same
    faults `hidden_pipe_leak` and `running_fixture`
  * the UI has five rooms with floor-plan geometry and per-wall ArUco markers,
    which the engine never needed

Walls matter because a camera finding is pinned to a wall by its marker, and a
moisture probe sits on one specific wall. That is what lets a photographed
stain be matched against the sensor that is reading the same patch of plaster.
"""

# id, display name, exterior, acoustic (pipe mic), under terrace, base moisture,
# floor-plan rect [x, y, w, h] in a 100 x 100 grid
ROOM_META = [
    ("living",   "Living Room",    True,  False, True,  11.5, [0, 0, 55, 50]),
    ("master",   "Master Bedroom", True,  False, True,  12.0, [55, 0, 45, 50]),
    ("bedroom2", "Bedroom 2",      True,  False, True,  12.5, [55, 50, 45, 50]),
    ("kitchen",  "Kitchen",        True,  True,  False, 12.0, [0, 50, 30, 50]),
    ("bathroom", "Bathroom",       False, True,  False, 13.0, [30, 50, 25, 50]),
]

ROOMS = [r[0] for r in ROOM_META]
ROOM_NAME = {r[0]: r[1] for r in ROOM_META}
EXTERIOR = {r[0]: r[2] for r in ROOM_META}
ACOUSTIC = {r[0]: r[3] for r in ROOM_META}
UNDER_TERRACE = {r[0]: r[4] for r in ROOM_META}
BASE_MOISTURE = {r[0]: r[5] for r in ROOM_META}

# The UI's fault ids, and which rooms each can occur in.
FAULTS = {
    "pipe_leak":    {"label": "Hidden pipe leak in wall",
                     "rooms": ["bathroom", "kitchen"]},
    "roof_seepage": {"label": "Roof / terrace seepage",
                     "rooms": ["living", "master", "bedroom2"]},
    "condensation": {"label": "Condensation & mould risk",
                     "rooms": ["master", "bedroom2", "bathroom"]},
    "fixture_leak": {"label": "Running toilet / dripping tap",
                     "rooms": ["bathroom"]},
}

# UI id <-> engine id. The engine's vocabulary is used everywhere inside
# smart_property; translation happens only at the API edge.
TO_ENGINE = {
    "pipe_leak": "hidden_pipe_leak",
    "roof_seepage": "roof_seepage",
    "condensation": "condensation",
    "fixture_leak": "running_fixture",
}
TO_UI = {v: k for k, v in TO_ENGINE.items()}

# Faults that wet plaster, so a photographed stain can corroborate them.
WET_FAULTS = {"pipe_leak", "roof_seepage", "condensation"}

# id, room, side, width_m, height_m, marker id, marker (x,y), probe (x,y) or None
_WALLS = [
    ("living-n",   "living",   "north", 4.6, 2.9, 11, (0.45, 1.80), None),
    ("living-w",   "living",   "west",  3.8, 2.9, 12, (0.40, 1.80), (1.20, 0.45)),
    ("master-e",   "master",   "east",  3.6, 2.9, 21, (0.40, 1.80), (1.10, 0.50)),
    ("master-n",   "master",   "north", 3.9, 2.9, 22, (0.45, 1.80), None),
    ("bedroom2-e", "bedroom2", "east",  3.6, 2.9, 31, (0.40, 1.80), (1.15, 0.50)),
    ("bedroom2-s", "bedroom2", "south", 3.9, 2.9, 32, (0.45, 1.80), None),
    ("kitchen-w",  "kitchen",  "west",  3.2, 2.9, 41, (0.40, 1.80), (1.05, 0.55)),
    ("kitchen-s",  "kitchen",  "south", 2.8, 2.9, 42, (0.45, 1.80), None),
    ("bathroom-w", "bathroom", "west",  2.4, 2.9, 51, (0.35, 1.80), (0.95, 0.60)),
    ("bathroom-n", "bathroom", "north", 2.2, 2.9, 52, (0.40, 1.80), None),
]

WALLS = [
    {"id": wid, "room": room, "side": side,
     "width_m": w, "height_m": h, "marker_id": marker,
     "marker_pos": {"x_m": mx, "y_m": my},
     "probe": ({"x_m": probe[0], "y_m": probe[1]} if probe else None)}
    for wid, room, side, w, h, marker, (mx, my), probe in _WALLS
]

WALL_BY_ID = {w["id"]: w for w in WALLS}

# The one wall per room that carries the moisture probe: the wall a sensor
# alert is actually about, and therefore the one to photograph.
PROBES = {w["room"]: {"wall": w["id"]} for w in WALLS if w["probe"]}


def config():
    """Exactly the payload `GET api/config` must return."""
    return {
        "rooms": [
            {"id": rid, "name": name, "exterior": ext, "acoustic": ac,
             "under_terrace": terr, "base_moisture": base, "plan": plan}
            for rid, name, ext, ac, terr, base, plan in ROOM_META
        ],
        "faults": FAULTS,
        "walls": WALLS,
    }
