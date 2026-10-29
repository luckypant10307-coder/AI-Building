"""A building-physics simulator for one Indian flat.

Generates 15-minute sensor readings for every room, driven by an outdoor
temperature cycle, monsoon-style rain events and a daily occupancy pattern.
A fault can be injected into one room; each fault kind leaves a different
signature across the sensors, which is what the rule engine reads back.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import EXTERNAL_WALL, ROOMS, SAMPLE_MINUTES, SAMPLES_PER_DAY
from .physics import surface_rh


@dataclass
class Fault:
    kind: str | None = None      # None means a fault-free flat
    room: str | None = None
    start_day: float = 10.0      # when it begins, in days from the start of the run

    def active(self, day_index):
        """Elapsed days since onset, 0 before it starts."""
        return np.clip(day_index - self.start_day, 0.0, None)


class FlatSimulator:
    """Simulate one flat for `days`, optionally with one injected fault."""

    def __init__(self, seed=0, days=30, fault=None, start="2026-06-01"):
        self.rng = np.random.default_rng(seed)
        self.days = days
        self.fault = fault or Fault()
        self.start = start
        self.n = days * SAMPLES_PER_DAY

    # ---------------------------------------------------------------- drivers

    def _time_index(self):
        return pd.date_range(self.start, periods=self.n, freq=f"{SAMPLE_MINUTES}min")

    def _rain(self):
        """Monsoon-ish rain: a few events a week, each a few hours long."""
        rain = np.zeros(self.n)
        n_events = max(self.rng.poisson(self.days / 3.5), 1)
        for _ in range(n_events):
            start = int(self.rng.integers(0, self.n))
            length = int(self.rng.integers(2, 8) * (60 // SAMPLE_MINUTES))
            peak = self.rng.uniform(2.0, 11.0)          # mm/h at the peak
            k = np.arange(length)
            shape = np.sin(np.pi * k / max(length - 1, 1)) ** 2
            end = min(start + length, self.n)
            rain[start:end] += peak * shape[: end - start]
        return rain

    def _outdoor_temp(self, hours):
        seasonal = 0.6 * np.sin(2 * np.pi * hours / (24 * 30))
        diurnal = 5.0 * np.sin(2 * np.pi * (hours - 9.0) / 24.0)
        wander = np.cumsum(self.rng.normal(0, 0.05, self.n))
        wander -= np.linspace(0, wander[-1], self.n)    # keep it from drifting away
        return 29.0 + seasonal + diurnal + wander

    def _occupancy_flow(self, hours):
        """Litres/min on the flat supply: showers, cooking, washing."""
        tod = hours % 24.0
        flow = np.zeros(self.n)
        for centre, width, rate in [(7.0, 0.5, 7.0), (8.0, 0.4, 2.0),
                                    (19.5, 0.5, 2.5), (21.0, 0.6, 4.0)]:
            flow += rate * np.exp(-0.5 * ((tod - centre) / width) ** 2)
        return np.clip(flow + self.rng.normal(0, 0.05, self.n), 0.0, None)

    # ----------------------------------------------------------------- faults

    def _leak_moisture(self, elapsed):
        """A concealed leak wets the plaster steadily and does not wait for rain."""
        return 0.75 * elapsed + 0.012 * elapsed ** 2

    def _seepage_moisture(self, rain, elapsed_mask):
        """Roof seepage: each rain event wets the wall, which then dries back slowly.

        Modelled as rainfall convolved with a decaying memory kernel, so moisture
        climbs through a wet spell and falls away between events. That shape --
        moisture that answers the rain -- is what separates seepage from a leak.
        """
        per_hour = 60 // SAMPLE_MINUTES
        tau = 5.0 * 24 * per_hour                    # the wall holds water ~5 days
        kernel = 0.14 * np.exp(-np.arange(int(6 * tau)) / tau)
        return np.convolve(rain * elapsed_mask, kernel)[: self.n]

    # -------------------------------------------------------------------- run

    def run(self):
        idx = self._time_index()
        hours = np.arange(self.n) * SAMPLE_MINUTES / 60.0
        day_index = hours / 24.0
        tod = hours % 24.0

        rain = self._rain()
        outdoor = self._outdoor_temp(hours)
        flat_flow = self._occupancy_flow(hours)

        fault = self.fault
        elapsed = fault.active(day_index)
        elapsed_mask = (elapsed > 0).astype(float)

        air_offset = {"living": 0.0, "kitchen": 0.9, "bathroom": 0.4, "bedroom": -0.3}
        flow_share = {"bathroom": 0.55, "kitchen": 0.3, "living": 0.05, "bedroom": 0.10}

        frames = []
        for room in ROOMS:
            air_temp = outdoor - 2.2 + air_offset[room] + self.rng.normal(0, 0.12, self.n)

            # Humidity: a damp baseline, wetter while it rains, with use spikes.
            air_rh = 56.0 + 0.9 * np.minimum(rain, 10.0) + self.rng.normal(0, 1.2, self.n)
            if room == "bathroom":
                air_rh += 22.0 * np.exp(-0.5 * ((tod - 7.0) / 0.45) ** 2)
            if room == "kitchen":
                air_rh += 12.0 * np.exp(-0.5 * ((tod - 19.5) / 0.6) ** 2)

            # A wall runs slightly cooler than the room air around it.
            wall_temp = air_temp - 0.6 + self.rng.normal(0, 0.08, self.n)

            wall_moisture = 8.0 + self.rng.normal(0, 0.10, self.n)
            if EXTERNAL_WALL[room]:
                # even a sound external wall takes up a little water when it rains
                wall_moisture += 0.004 * np.convolve(rain, np.ones(16) / 16)[: self.n]

            room_flow = flat_flow * flow_share[room]
            pipe_db = 28.0 + 2.2 * room_flow + self.rng.normal(0, 0.35, self.n)

            if fault.kind and fault.room == room:
                if fault.kind == "hidden_pipe_leak":
                    wall_moisture = wall_moisture + self._leak_moisture(elapsed)
                    room_flow = room_flow + 0.42 * elapsed_mask   # never stops, even at 3am
                    pipe_db = pipe_db + 6.0 * elapsed_mask
                elif fault.kind == "roof_seepage":
                    wall_moisture = wall_moisture + self._seepage_moisture(rain, elapsed_mask)
                elif fault.kind == "condensation":
                    wall_temp = wall_temp - 4.5 * elapsed_mask    # a thermal bridge appears
                    air_rh = air_rh + 8.0 * elapsed_mask
                elif fault.kind == "running_fixture":
                    room_flow = room_flow + 1.25 * elapsed_mask
                    pipe_db = pipe_db + 4.2 * elapsed_mask

            # Evaporation from a wet wall cools it: the tell for a concealed leak.
            wetness = np.clip(wall_moisture - 9.0, 0.0, None)
            wall_temp = wall_temp - 0.14 * wetness

            srh = surface_rh(air_temp, air_rh, wall_temp)
            # Sustained condensation slowly wets the plaster too.
            wall_moisture = wall_moisture + np.cumsum(
                np.clip(srh - 88.0, 0.0, None) * 0.00035)

            frames.append(pd.DataFrame({
                "ts": idx,
                "room": room,
                "wall_moisture": np.clip(wall_moisture, 0, 60),
                "wall_surface_rh": np.clip(srh, 0, 100),
                "air_temp": air_temp,
                "air_rh": np.clip(air_rh, 5, 100),
                "wall_temp": wall_temp,
                "flow_lpm": np.clip(room_flow, 0, None),
                "pipe_db": pipe_db,
                "rain_mm_h": rain,
            }))

        frame = pd.concat(frames, ignore_index=True)
        frame.attrs["fault"] = {
            "kind": fault.kind, "room": fault.room, "start_day": fault.start_day,
        }
        return frame


def simulate(seed=0, days=30, fault_kind=None, room=None, start_day=10.0):
    """Convenience wrapper used by the scripts."""
    fault = Fault(kind=fault_kind, room=room, start_day=start_day)
    return FlatSimulator(seed=seed, days=days, fault=fault).run()
