"""Benchmark the sensor pipeline on simulated flats.

Three measurements, all on simulated data -- which is exactly how they must be
reported:

  1. False alarms   -- fault-free flats, how often any room raises an alert.
  2. Localisation   -- injected faults, is the kind AND the room both right.
  3. Forecast error -- predicted days-to-visible-damp against when the
                       simulated wall actually crosses 20% moisture.

Usage:  python scripts/benchmark.py --seeds 25
"""

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
warnings.filterwarnings("ignore")

from smart_property.config import (          # noqa: E402
    EXTERNAL_WALL, FAULT_KINDS, ROOMS, SAMPLES_PER_DAY, VISIBLE_DAMP_MOISTURE)
from smart_property.pipeline import analyse_flat, raised_alerts   # noqa: E402
from smart_property.simulator import simulate                     # noqa: E402

RAIN_FED = [r for r in ROOMS if EXTERNAL_WALL[r]]


def room_for(kind, seed):
    """Roof seepage only happens on a wall the rain can reach."""
    pool = RAIN_FED if kind == "roof_seepage" else ROOMS
    return pool[seed % len(pool)]


def false_alarms(seeds, days):
    flats_with_alert, rooms_flagged, total_rooms = 0, 0, 0
    for seed in range(seeds):
        df = simulate(seed=1000 + seed, days=days)
        raised = raised_alerts(analyse_flat(df))
        total_rooms += len(ROOMS)
        rooms_flagged += len(raised)
        flats_with_alert += bool(raised)
    return {
        "flats": seeds, "days": days,
        "flats_with_alert": flats_with_alert,
        "rooms_flagged": rooms_flagged,
        "room_days": total_rooms * days,
    }


def localisation(seeds, days):
    rows, correct = [], 0
    for kind in FAULT_KINDS:
        hits = 0
        for seed in range(seeds):
            room = room_for(kind, seed)
            df = simulate(seed=seed, days=days, fault_kind=kind,
                          room=room, start_day=10.0)
            results = analyse_flat(df)
            raised = raised_alerts(results)
            ok = (len(raised) == 1 and raised[0]["room"] == room
                  and raised[0]["alert"]["kind"] == kind)
            hits += ok
        rows.append((kind, hits, seeds))
        correct += hits
    return rows, correct, seeds * len(FAULT_KINDS)


def actual_crossing_day(df, room, observed_days):
    """Day index at which the simulated wall truly crosses the damp threshold."""
    series = df[df["room"] == room]["wall_moisture"].to_numpy()
    over = np.flatnonzero(series >= VISIBLE_DAMP_MOISTURE)
    if over.size == 0:
        return None
    return over[0] / SAMPLES_PER_DAY - observed_days


def forecast_error(seeds, observed_days=20, total_days=45):
    errors, evaluated, missed = [], 0, 0
    for kind in ("hidden_pipe_leak", "roof_seepage"):
        for seed in range(seeds):
            room = room_for(kind, seed)
            # onset late enough that the wall is still below the threshold when
            # the observation window closes -- otherwise there is nothing to predict
            full = simulate(seed=seed, days=total_days, fault_kind=kind,
                            room=room, start_day=12.0)
            truth = actual_crossing_day(full, room, observed_days)
            if truth is None or truth <= 0:
                continue      # already damp, or never gets there: nothing to predict
            cut = full["ts"] < full["ts"].min() + np.timedelta64(observed_days, "D")
            window = full[cut]
            result = next((r for r in analyse_flat(window) if r["room"] == room), None)
            if result is None:
                continue
            predicted = result["forecast"]["visible_damp"]["days"]
            evaluated += 1
            if predicted is None:
                missed += 1
            else:
                errors.append(abs(predicted - truth))
    return errors, evaluated, missed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=25)
    parser.add_argument("--clean-days", type=int, default=60)
    parser.add_argument("--fault-days", type=int, default=25)
    args = parser.parse_args()

    print(f"Benchmarking on simulated data, {args.seeds} seeds per case.\n")

    fa = false_alarms(args.seeds, args.clean_days)
    print("1. False alarms on fault-free flats")
    print(f"   {fa['flats']} flats x {fa['days']} days "
          f"({fa['room_days']} room-days)")
    print(f"   flats raising any alert : {fa['flats_with_alert']}/{fa['flats']}")
    print(f"   rooms flagged           : {fa['rooms_flagged']}\n")

    rows, correct, total = localisation(args.seeds, args.fault_days)
    print("2. Fault type and room, both correct")
    for kind, hits, n in rows:
        print(f"   {kind:<18} {hits:>3}/{n}")
    print(f"   {'overall':<18} {correct:>3}/{total} "
          f"({100.0 * correct / total:.1f}%)\n")

    errors, evaluated, missed = forecast_error(args.seeds)
    print("3. Days-to-visible-damp, predicted vs simulated truth")
    if errors:
        print(f"   cases evaluated         : {evaluated}")
        print(f"   no crossing predicted   : {missed}")
        print(f"   mean absolute error     : {np.mean(errors):.2f} days")
        print(f"   median absolute error   : {np.median(errors):.2f} days")
        print(f"   within 2 days           : "
              f"{100.0 * np.mean(np.array(errors) <= 2.0):.0f}%")
    else:
        print("   no evaluable cases")
    print("\nAll figures above come from the simulator, not from a real flat.")


if __name__ == "__main__":
    main()
