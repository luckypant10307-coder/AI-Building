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


def forecast_error(seeds, observed_days=20, total_days=45, backend="auto"):
    """One record per case where the wall really does cross the threshold.

    A case where the forecaster declines to predict is still a case: it is
    recorded with declined=True and counts as a failure. Dropping those
    flatters the result, because the hard cases are exactly the ones a linear
    forecaster gives up on.

    `error` is signed: predicted minus truth, so positive means predicted late.
    A bias and a spread say different things and are reported separately.
    """
    records = []
    for kind in ("hidden_pipe_leak", "roof_seepage"):
        for seed in range(seeds):
            room = room_for(kind, seed)
            # onset late enough that the wall is still below the threshold when
            # the observation window closes -- otherwise there is nothing to predict
            full = simulate(seed=seed, days=total_days, fault_kind=kind,
                            room=room, start_day=12.0)
            truth = actual_crossing_day(full, room, observed_days)
            if truth is None or truth <= 0:
                continue      # already damp, or never gets there: not a case
            cut = full["ts"] < full["ts"].min() + np.timedelta64(observed_days, "D")
            result = next((r for r in analyse_flat(full[cut], backend=backend)
                           if r["room"] == room), None)
            if result is None:
                continue
            predicted = result["forecast"]["visible_damp"]["days"]
            records.append({
                "kind": kind,
                "declined": predicted is None,
                "error": None if predicted is None else predicted - truth,
            })
    return records


def summarise(records):
    """Within-2-days counts every case, a declined prediction included."""
    errors = [r["error"] for r in records if r["error"] is not None]
    hits = sum(r["error"] is not None and abs(r["error"]) <= 2.0 for r in records)
    return {
        "cases": len(records),
        "declined": sum(r["declined"] for r in records),
        "within2": hits,
        "bias": float(np.mean(errors)) if errors else None,
        "mae": float(np.mean(np.abs(errors))) if errors else None,
        "spread": float(np.std(errors)) if errors else None,
    }


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

    records = forecast_error(args.seeds)
    print("3. Days-to-visible-damp, predicted vs simulated truth")
    print("   A declined prediction counts as a failure, not as an excluded case.")
    print(f"   {'fault':<18} {'cases':>5} {'declined':>9} {'within 2d':>10} "
          f"{'bias':>8} {'spread':>8}")
    for kind in ("hidden_pipe_leak", "roof_seepage"):
        s = summarise([r for r in records if r["kind"] == kind])
        if not s["cases"]:
            continue
        bias = "-" if s["bias"] is None else f"{s['bias']:+.2f}d"
        spread = "-" if s["spread"] is None else f"{s['spread']:.2f}d"
        print(f"   {kind:<18} {s['cases']:>5} {s['declined']:>9} "
              f"{str(s['within2']) + '/' + str(s['cases']):>10} {bias:>8} {spread:>8}")

    total = summarise(records)
    if total["cases"]:
        share = 100.0 * total["within2"] / total["cases"]
        bias = "-" if total["bias"] is None else f"{total['bias']:+.2f}"
        print(f"   {'overall':<18} {total['cases']:>5} {total['declined']:>9} "
              f"{str(total['within2']) + '/' + str(total['cases']):>10} "
              f"{bias + 'd':>8} {total['spread']:.2f}d")
        print(f"\n   headline: {share:.0f}% of cases predicted within 2 days")
        if total["bias"] is not None and total["spread"] is not None:
            print(f"   bias {total['bias']:+.2f} d against a spread of "
                  f"{total['spread']:.2f} d: "
                  + ("mostly systematic, not noise"
                     if abs(total["bias"]) > total["spread"] else "mostly noise"))
    print("\nAll figures above come from the simulator, not from a real flat.")


if __name__ == "__main__":
    main()
