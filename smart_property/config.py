"""Flat layout, sensor spec and the physical thresholds damage is judged against."""

# Mirrors smart_property.flat_model, which is the dashboard's contract.
ROOMS = ["living", "master", "bedroom2", "kitchen", "bathroom"]

# Rooms on an external//top-floor wall are the ones rain can reach.
EXTERNAL_WALL = {"living": True, "master": True, "bedroom2": True,
                 "kitchen": True, "bathroom": False}

SENSORS = [
    "wall_moisture",     # % volumetric water content in the plaster
    "wall_surface_rh",   # % relative humidity in the film of air at the wall
    "air_temp",          # degC
    "air_rh",            # %
    "wall_temp",         # degC
    "flow_lpm",          # litres/min on the room's branch
    "pipe_db",           # dB, structure-borne noise on the pipe
]

SAMPLE_MINUTES = 15
SAMPLES_PER_DAY = 24 * 60 // SAMPLE_MINUTES

# Damage thresholds. These are what "days until visible damage" counts down to.
VISIBLE_DAMP_MOISTURE = 20.0   # % -- plaster darkens, a stain becomes visible
MOULD_RISK_SURFACE_RH = 80.0   # % -- sustained surface RH that lets mould grow

FAULT_KINDS = [
    "hidden_pipe_leak",
    "roof_seepage",
    "condensation",
    "running_fixture",
]

# Baseline learning window: the detector learns a "normal week" before it may alarm.
BASELINE_DAYS = 7
